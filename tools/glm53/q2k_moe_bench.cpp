// NextN-block MoE prefill micro-benchmark (not a test): the GLM-5.3-Flash
// draft block's experts are Q2_K gate/up and Q3_K down (288 experts, 8
// routes), unlike the trunk's IQ2_XXS. Times what Ds4GateUp/Ds4Down run for a
// prompt of `tokens` rows (one mmq call per matrix, 16-column tiles) against
// the decode vec kernels run over 8-token chunks, and reports how far apart
// the outputs are.
// usage: glm53_flash_q2k_moe_bench [tokens=30] [reps=5]
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#include "src/models/deepseek_v4_flash/kernels/rocm/mmq/ds4_mmq.h"
#include "src/models/deepseek_v4_flash/kernels/rocm/mmq/ds4_pool.h"

namespace {
void Check(hipError_t e, const char* what) {
  if (e != hipSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e));
    std::exit(1);
  }
}
void Rc(int rc, const char* what) {
  if (rc != 0) {
    std::fprintf(stderr, "%s rc=%d\n", what, rc);
    std::exit(1);
  }
}
// Random k-quant blocks with sane fp16 d/dmin at the block tail.
std::vector<std::uint8_t> g_last;  // host copy of the last RandomWeights
void* RandomWeights(std::mt19937& rng, std::size_t blocks, std::size_t bytes,
                    std::size_t d_at, bool dmin) {
  std::vector<std::uint8_t> host(blocks * bytes);
  for (auto& v : host) v = static_cast<std::uint8_t>(rng());
  for (std::size_t o = 0; o < host.size(); o += bytes) {
    const std::uint16_t d = 0x1400 + rng() % 1024;
    std::memcpy(host.data() + o + d_at, &d, 2);
    if (dmin) {
      const std::uint16_t m = 0x1000 + rng() % 1024;
      std::memcpy(host.data() + o + d_at + 2, &m, 2);
    }
  }
  void* p = nullptr;
  // Coarse-grain managed, as GLM's executor allocates weights.
  Check(hipMallocManaged(&p, host.size(), hipMemAttachGlobal), "weights");
  if (hipMemAdvise(p, host.size(), hipMemAdviseSetCoarseGrain, 0) != hipSuccess)
    (void)hipGetLastError();
  Check(hipMemcpy(p, host.data(), host.size(), hipMemcpyHostToDevice), "copy");
  g_last = host;
  return p;
}
// ggml's dequantize_row_q2_K for one 256-value block.
void DequantQ2K(const std::uint8_t* b, float* y) {
  const auto half = [](const std::uint8_t* p) {
    _Float16 h;
    std::memcpy(&h, p, 2);
    return static_cast<float>(h);
  };
  const float d = half(b + 80), dmin = half(b + 82);
  const std::uint8_t* q = b + 16;
  int is = 0;
  for (int n = 0; n < 256; n += 128, q += 32)
    for (int shift = 0; shift < 8; shift += 2)
      for (int h = 0; h < 2; ++h) {
        const std::uint8_t sc = b[is++];
        for (int l = 0; l < 16; ++l)
          *y++ = d * (sc & 0xF) * ((q[l + 16 * h] >> shift) & 3) - dmin * (sc >> 4);
      }
}
double RelL2(const std::vector<float>& a, const std::vector<float>& b) {
  double d = 0, n = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    d += (double(a[i]) - b[i]) * (double(a[i]) - b[i]);
    n += double(b[i]) * b[i];
  }
  return std::sqrt(d / n);
}
}  // namespace

int main(int argc, char** argv) {
  const int T = argc > 1 ? std::atoi(argv[1]) : 30;
  const int reps = argc > 2 ? std::atoi(argv[2]) : 5;
  constexpr int E = 288, used = 8, F = 2048, H = 4096, kVec = 8;
  std::mt19937 rng(745);
  // Q2_K: 84 bytes per 256 (d, dmin at 80); Q3_K: 110 bytes (d at 108).
  void* wg = RandomWeights(rng, std::size_t{E} * F * (H / 256), 84, 80, true);
  const std::vector<std::uint8_t> host_g = g_last;
  void* wu = RandomWeights(rng, std::size_t{E} * F * (H / 256), 84, 80, true);
  void* wd = RandomWeights(rng, std::size_t{E} * H * (F / 256), 110, 108, false);
  std::uniform_real_distribution<float> dist(-1, 1);
  std::vector<float> hx(std::size_t{T} * H), hm(std::size_t{T} * used * F);
  for (auto& v : hx) v = dist(rng);
  for (auto& v : hm) v = dist(rng);
  std::vector<std::int32_t> hid(std::size_t{T} * used);
  std::vector<int> perm(E);
  std::iota(perm.begin(), perm.end(), 0);
  for (int t = 0; t < T; ++t) {
    std::shuffle(perm.begin(), perm.end(), rng);
    std::copy_n(perm.begin(), used, hid.begin() + std::size_t{t} * used);
  }
  // Down runs over tokens * routes flattened rows, one expert each.
  std::vector<std::int32_t> hflat(hid);
  float *x, *mid, *g, *u, *y;
  std::int32_t *ids, *flat;
  Check(hipMalloc(&x, hx.size() * 4), "x");
  Check(hipMalloc(&mid, hm.size() * 4), "mid");
  Check(hipMalloc(&g, hm.size() * 4), "g");
  Check(hipMalloc(&u, hm.size() * 4), "u");
  Check(hipMalloc(&y, std::size_t{T} * used * H * 4), "y");
  Check(hipMalloc(&ids, hid.size() * 4), "ids");
  Check(hipMalloc(&flat, hflat.size() * 4), "flat");
  Check(hipMemcpy(x, hx.data(), hx.size() * 4, hipMemcpyHostToDevice), "x");
  Check(hipMemcpy(mid, hm.data(), hm.size() * 4, hipMemcpyHostToDevice), "mid");
  Check(hipMemcpy(ids, hid.data(), hid.size() * 4, hipMemcpyHostToDevice), "ids");
  Check(hipMemcpy(flat, hflat.data(), hflat.size() * 4, hipMemcpyHostToDevice), "flat");
  ds4_pool_use_stack_arena(1);
  ds4_mmq_set_routed_tile_cols(16);
  hipStream_t s;
  Check(hipStreamCreate(&s), "stream");
  hipEvent_t e0, e1;
  Check(hipEventCreate(&e0), "ev");
  Check(hipEventCreate(&e1), "ev");
  const auto time = [&](auto&& fn) {
    fn();
    Check(hipStreamSynchronize(s), "warm");
    Check(hipEventRecord(e0, s), "rec");
    for (int i = 0; i < reps; ++i) fn();
    Check(hipEventRecord(e1, s), "rec");
    Check(hipEventSynchronize(e1), "sync");
    float ms = 0;
    Check(hipEventElapsedTime(&ms, e0, e1), "elapsed");
    return ms / reps;
  };
  const auto get = [&](const float* p, std::size_t n) {
    std::vector<float> h(n);
    Check(hipMemcpy(h.data(), p, n * 4, hipMemcpyDeviceToHost), "get");
    return h;
  };
  const std::size_t gu = std::size_t{T} * used * F;
  const std::size_t dn = std::size_t{T} * used * H;
  std::printf("tokens=%d experts=%d used=%d ff=%d hidden=%d\n", T, E, used, F, H);

  const float gate_mmq = time([&] {
    Rc(ds4_mmq_q2_K_moe(wg, x, ids, g, F, H, T, E, used, s), "q2_K_moe");
    Rc(ds4_mmq_q2_K_moe(wu, x, ids, u, F, H, T, E, used, s), "q2_K_moe");
  });
  const auto ref_g = get(g, gu);
  const float gate_vec = time([&] {
    for (int t0 = 0; t0 < T; t0 += kVec) {
      const int n = std::min(kVec, T - t0);
      Rc(ds4_mmq_q2_K_moe_vec(wg, x + std::size_t{t0} * H, ids + t0 * used,
                              g + std::size_t{t0} * used * F, F, H, n, E, used, s),
         "q2_K_moe_vec");
      Rc(ds4_mmq_q2_K_moe_vec(wu, x + std::size_t{t0} * H, ids + t0 * used,
                              u + std::size_t{t0} * used * F, F, H, n, E, used, s),
         "q2_K_moe_vec");
    }
  });
  const auto vec_g = get(g, gu);
  // fp64 reference for the gate rows m < 64 of every (token, route).
  std::vector<float> want, got_mmq, got_vec, row(H);
  for (int t = 0; t < T; ++t)
    for (int r = 0; r < used; ++r)
      for (int m = 0; m < 64; ++m) {
        const std::size_t e = hid[std::size_t{t} * used + r];
        const std::uint8_t* w = host_g.data() + (e * F + m) * (H / 256) * 84;
        for (int k = 0; k < H / 256; ++k) DequantQ2K(w + k * 84, row.data() + k * 256);
        double acc = 0;
        for (int k = 0; k < H; ++k) acc += double(row[k]) * hx[std::size_t{t} * H + k];
        const std::size_t at = (std::size_t{t} * used + r) * F + m;
        want.push_back(float(acc));
        got_mmq.push_back(ref_g[at]);
        got_vec.push_back(vec_g[at]);
      }
  std::printf("gate+up Q2_K  mmq %8.3f ms   vec/8 %8.3f ms   rel_l2 %.3e"
              "   vs fp64: mmq %.3e vec %.3e\n",
              gate_mmq, gate_vec, RelL2(vec_g, ref_g), RelL2(got_mmq, want),
              RelL2(got_vec, want));

  const int rows = T * used;
  const float down_mmq = time([&] {
    ds4_mmq_set_moe_down_bounds(16, T);
    Rc(ds4_mmq_q3_K_moe(wd, mid, flat, y, H, F, rows, E, 1, s), "q3_K_moe");
    ds4_mmq_set_moe_down_bounds(0, 0);
  });
  const auto ref_y = get(y, dn);
  const float down_vec = time([&] {
    for (int r0 = 0; r0 < rows; r0 += kVec) {
      const int n = std::min(kVec, rows - r0);
      Rc(ds4_mmq_q3_K_moe_vec(wd, mid + std::size_t{r0} * F, flat + r0,
                              y + std::size_t{r0} * H, H, F, n, E, 1, s),
         "q3_K_moe_vec");
    }
  });
  const auto vec_y = get(y, dn);
  std::printf("down    Q3_K  mmq %8.3f ms   vec/8 %8.3f ms   rel_l2 %.3e\n",
              down_mmq, down_vec, RelL2(vec_y, ref_y));
  return 0;
}
