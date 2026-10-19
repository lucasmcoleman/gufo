// MoE prefill gate/up micro-benchmark (not a test): GLM-5.3-Flash expert shape
// (288 x [2048 x 4096] IQ2_XXS for gate and up, 8 routes per token) through
// ds4_mmq_iq2_xxs_moe_pair at several column-tile widths, with and without the
// token-bound variant. Every variant's output must equal the default bitwise.
// usage: glm53_flash_moe_bench [tokens=256] [reps=5]
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <utility>
#include <random>
#include <vector>

#include "src/models/deepseek_v4_flash/kernels/rocm/mmq/ds4_mmq.h"
#include "src/models/deepseek_v4_flash/kernels/rocm/mmq/ds4_pool.h"

namespace {
// MOE_BENCH_MANAGED=1: weights in coarse-grain managed memory, as GLM's
// executor allocates them (gufo::hip::AllocateManaged), instead of hipMalloc.
hipError_t WeightAlloc(void** p, std::size_t bytes) {
  const char* v = std::getenv("MOE_BENCH_MANAGED");
  if (!(v && *v == '1')) return hipMalloc(p, bytes);
  const hipError_t e = hipMallocManaged(p, bytes, hipMemAttachGlobal);
  if (e == hipSuccess && hipMemAdvise(*p, bytes, hipMemAdviseSetCoarseGrain, 0) != hipSuccess)
    (void)hipGetLastError();
  return e;
}
void Check(hipError_t e, const char* what) {
  if (e != hipSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e));
    std::exit(1);
  }
}
}  // namespace

int main(int argc, char** argv) {
  const int T = argc > 1 ? std::atoi(argv[1]) : 256;
  const int reps = argc > 2 ? std::atoi(argv[2]) : 5;
  constexpr int E = 288, used = 8, M = 2048, K = 4096;
  constexpr std::size_t block_bytes = 66;  // IQ2_XXS: fp16 d + 64 bytes per 256
  const std::size_t matrix = std::size_t{E} * M * (K / 256) * block_bytes;
  std::mt19937 rng(682);
  std::vector<std::uint8_t> host(matrix);
  for (auto& v : host) v = static_cast<std::uint8_t>(rng());
  for (std::size_t o = 0; o < host.size(); o += block_bytes) {
    const std::uint16_t scale = 0x1400 + rng() % 1024;
    std::memcpy(host.data() + o, &scale, 2);
  }
  void *wg, *wu, *x, *ids, *ga, *ua, *gb, *ub;
  Check(WeightAlloc(&wg, matrix), "wg");
  Check(WeightAlloc(&wu, matrix), "wu");
  Check(hipMemcpy(wg, host.data(), matrix, hipMemcpyHostToDevice), "wg copy");
  for (auto& v : host) v ^= 0x5a;
  Check(hipMemcpy(wu, host.data(), matrix, hipMemcpyHostToDevice), "wu copy");
  std::vector<float> hx(std::size_t{T} * K);
  std::uniform_real_distribution<float> d(-1, 1);
  for (auto& v : hx) v = d(rng);
  Check(hipMalloc(&x, hx.size() * 4), "x");
  Check(hipMemcpy(x, hx.data(), hx.size() * 4, hipMemcpyHostToDevice), "x copy");
  std::vector<std::int32_t> hid(std::size_t{T} * used);
  std::vector<int> perm(E);
  std::iota(perm.begin(), perm.end(), 0);
  for (int t = 0; t < T; ++t) {
    std::shuffle(perm.begin(), perm.end(), rng);  // 8 distinct experts per token
    std::copy_n(perm.begin(), used, hid.begin() + std::size_t{t} * used);
  }
  Check(hipMalloc(&ids, hid.size() * 4), "ids");
  Check(hipMemcpy(ids, hid.data(), hid.size() * 4, hipMemcpyHostToDevice), "ids copy");
  const std::size_t out = std::size_t{T} * used * M * 4;
  Check(hipMalloc(&ga, out), "ga"); Check(hipMalloc(&ua, out), "ua");
  Check(hipMalloc(&gb, out), "gb"); Check(hipMalloc(&ub, out), "ub");
  ds4_pool_use_stack_arena(std::getenv("MOE_BENCH_ARENA") ? std::atoi(std::getenv("MOE_BENCH_ARENA")) : 1);
  hipStream_t s;
  Check(hipStreamCreate(&s), "stream");
  hipEvent_t e0, e1;
  Check(hipEventCreate(&e0), "ev"); Check(hipEventCreate(&e1), "ev");
  auto call = [&](bool bound, void* g, void* u) {
    const int rc = bound
        ? ds4_mmq_iq2_xxs_moe_pair_token_bound(wg, wu, (const float*)x, (const int32_t*)ids,
                                               (float*)g, (float*)u, M, K, T, E, used, s)
        : ds4_mmq_iq2_xxs_moe_pair(wg, wu, (const float*)x, (const int32_t*)ids, (float*)g,
                                   (float*)u, M, K, T, E, used, s);
    if (rc != 0) { std::fprintf(stderr, "pair rc=%d\n", rc); std::exit(1); }
  };
  // Reference output: default width, generic pair.
  ds4_mmq_set_routed_tile_cols(0);
  call(false, ga, ua);
  Check(hipStreamSynchronize(s), "ref");
  std::vector<float> ref_g(out / 4), ref_u(out / 4), got_g(out / 4), got_u(out / 4);
  Check(hipMemcpy(ref_g.data(), ga, out, hipMemcpyDeviceToHost), "ref g");
  Check(hipMemcpy(ref_u.data(), ua, out, hipMemcpyDeviceToHost), "ref u");
  std::printf("tokens=%d experts=%d used=%d M=%d K=%d\n%-6s %-6s %9s %s\n", T, E, used, M, K,
              "width", "bound", "ms", "bitwise==default");
  for (int bound = 0; bound < 2; ++bound)
    for (int width : {0, 8, 16, 24, 32, 48, 64}) {
      ds4_mmq_set_routed_tile_cols(width);
      call(bound, gb, ub);  // warm-up
      Check(hipStreamSynchronize(s), "warm");
      Check(hipEventRecord(e0, s), "rec");
      for (int i = 0; i < reps; ++i) call(bound, gb, ub);
      Check(hipEventRecord(e1, s), "rec");
      Check(hipEventSynchronize(e1), "sync");
      float ms = 0;
      Check(hipEventElapsedTime(&ms, e0, e1), "elapsed");
      Check(hipMemcpy(got_g.data(), gb, out, hipMemcpyDeviceToHost), "g");
      Check(hipMemcpy(got_u.data(), ub, out, hipMemcpyDeviceToHost), "u");
      const bool same = std::memcmp(got_g.data(), ref_g.data(), out) == 0 &&
                        std::memcmp(got_u.data(), ref_u.data(), out) == 0;
      double diff = 0, norm = 0, max_rel = 0;
      std::size_t nonfinite = 0;
      for (std::size_t i = 0; i < ref_g.size(); ++i)
        for (const auto& [a, b] : {std::pair{got_g[i], ref_g[i]}, std::pair{got_u[i], ref_u[i]}}) {
          if (!std::isfinite(a)) { ++nonfinite; continue; }
          const double e = std::fabs(double(a) - b);
          diff += e * e; norm += double(b) * b;
          max_rel = std::fmax(max_rel, e / (std::fabs(double(b)) + 1e-3));
        }
      if (!same && std::getenv("MOE_BENCH_DIFFS")) {
        // Outputs are [tokens * used][M]: row = token*used + slot.
        std::size_t ndiff = 0, shown = 0; double max_abs = 0, max_ref = 0;
        for (std::size_t i = 0; i < ref_g.size(); ++i) {
          max_ref = std::fmax(max_ref, std::fabs(ref_g[i]));
          if (got_g[i] != ref_g[i]) {
            ++ndiff; max_abs = std::fmax(max_abs, std::fabs(double(got_g[i]) - ref_g[i]));
            if (shown < 6 && std::fabs(double(got_g[i]) - ref_g[i]) > 1e-4) {
              const std::size_t row = i / M, m = i % M;
              std::printf("    diff token=%zu slot=%zu expert=%d m=%zu got=%.6g ref=%.6g\n",
                          row / used, row % used, hid[row], m, got_g[i], ref_g[i]);
              ++shown;
            }
          }
        }
        std::printf("    gate: %zu of %zu differ, max_abs=%.3g, max|ref|=%.3g\n", ndiff,
                    ref_g.size(), max_abs, max_ref);
      }
      std::printf("%-6d %-6d %9.3f %-4s rel_l2=%.3e max_rel=%.3e nonfinite=%zu\n", width, bound,
                  ms / reps, same ? "yes" : "NO", std::sqrt(diff / norm), max_rel, nonfinite);
    }
  ds4_mmq_set_routed_tile_cols(0);

  // Decode gate/up (1 token): two ds4_mmq_iq2_xxs_moe_vec calls (what
  // Ds4GateUp does today) vs one ds4_mmq_iq2_xxs_moe_pair_raw_vec.
  {
    auto time = [&](auto&& fn) {
      fn(); Check(hipStreamSynchronize(s), "warm");
      Check(hipEventRecord(e0, s), "rec");
      for (int i = 0; i < reps * 20; ++i) fn();
      Check(hipEventRecord(e1, s), "rec"); Check(hipEventSynchronize(e1), "sync");
      float ms = 0; Check(hipEventElapsedTime(&ms, e0, e1), "elapsed");
      return ms / (reps * 20);
    };
    const std::size_t o1 = std::size_t{used} * M * 4;
    const float ms_two = time([&] {
      if (ds4_mmq_iq2_xxs_moe_vec(wg, (const float*)x, (const int32_t*)ids, (float*)ga, M, K, 1,
                                  E, used, s) != 0 ||
          ds4_mmq_iq2_xxs_moe_vec(wu, (const float*)x, (const int32_t*)ids, (float*)ua, M, K, 1,
                                  E, used, s) != 0) { std::fprintf(stderr, "vec failed\n"); std::exit(1); }
    });
    const float ms_pair = time([&] {
      if (ds4_mmq_iq2_xxs_moe_pair_raw_vec(wg, wu, (const float*)x, (const int32_t*)ids, (float*)gb,
                                           (float*)ub, M, K, 1, E, used, s) != 0) {
        std::fprintf(stderr, "pair_raw_vec failed\n"); std::exit(1); }
    });
    std::vector<float> a(o1 / 4), b(o1 / 4), c(o1 / 4), dd(o1 / 4);
    Check(hipMemcpy(a.data(), ga, o1, hipMemcpyDeviceToHost), "a");
    Check(hipMemcpy(b.data(), gb, o1, hipMemcpyDeviceToHost), "b");
    Check(hipMemcpy(c.data(), ua, o1, hipMemcpyDeviceToHost), "c");
    Check(hipMemcpy(dd.data(), ub, o1, hipMemcpyDeviceToHost), "d");
    const bool same = std::memcmp(a.data(), b.data(), o1) == 0 && std::memcmp(c.data(), dd.data(), o1) == 0;
    std::printf("decode gate/up (1 token): two moe_vec %.4f ms, pair_raw_vec %.4f ms, bitwise %s\n",
                ms_two, ms_pair, same ? "yes" : "NO");
  }

  // Down: [288][4096][2048] IQ3_XXS, one row per token-route (n_expert_used 1),
  // as Ds4Down calls it.
  for (int down_type = 0; down_type < 2; ++down_type) {
    constexpr int DM = 4096, DK = 2048;
    // IQ3_XXS: fp16 d + 96 bytes per 256; IQ2_S: d + 64 qs + 8 qh + 8 scales.
    const std::size_t dblock = down_type == 0 ? 98 : 82;
    const std::size_t dmat = std::size_t{E} * DM * (DK / 256) * dblock;
    std::vector<std::uint8_t> dh(dmat);
    for (auto& v : dh) v = static_cast<std::uint8_t>(rng());
    for (std::size_t o = 0; o < dh.size(); o += dblock) {
      const std::uint16_t scale = 0x1400 + rng() % 1024;
      std::memcpy(dh.data() + o, &scale, 2);
    }
    void *wd, *xd, *yd, *yr;
    const int R = T * used;
    Check(WeightAlloc(&wd, dmat), "wd");
    Check(hipMemcpy(wd, dh.data(), dmat, hipMemcpyHostToDevice), "wd copy");
    std::vector<float> hxd(std::size_t{R} * DK);
    for (auto& v : hxd) v = d(rng);
    Check(hipMalloc(&xd, hxd.size() * 4), "xd");
    Check(hipMemcpy(xd, hxd.data(), hxd.size() * 4, hipMemcpyHostToDevice), "xd copy");
    const std::size_t dout = std::size_t{R} * DM * 4;
    Check(hipMalloc(&yd, dout), "yd"); Check(hipMalloc(&yr, dout), "yr");
    auto down = [&](void* y) {
      const int rc = down_type == 0
          ? ds4_mmq_iq3_xxs_moe(wd, (const float*)xd, (const int32_t*)ids, (float*)y, DM, DK, R, E, 1, s)
          : ds4_mmq_iq2_s_moe(wd, (const float*)xd, (const int32_t*)ids, (float*)y, DM, DK, R, E, 1, s);
      if (rc != 0) { std::fprintf(stderr, "down failed\n"); std::exit(1); }
    };
    ds4_mmq_set_moe_down_bounds(0, 0);
    down(yr);
    Check(hipStreamSynchronize(s), "down ref");
    std::vector<float> rd(dout / 4), gd(dout / 4);
    Check(hipMemcpy(rd.data(), yr, dout, hipMemcpyDeviceToHost), "rd");
    std::printf("down %s rows=%d\n%-6s %-6s %9s\n", down_type == 0 ? "IQ3_XXS" : "IQ2_S", R, "width", "bound", "ms");
    for (int bound = 0; bound < 2; ++bound)
      for (int width : {0, 16, 32, 48, 64}) {
        ds4_mmq_set_moe_down_bounds(width, bound ? T : 0);
        down(yd);
        Check(hipStreamSynchronize(s), "warm");
        Check(hipEventRecord(e0, s), "rec");
        for (int i = 0; i < reps; ++i) down(yd);
        Check(hipEventRecord(e1, s), "rec");
        Check(hipEventSynchronize(e1), "sync");
        float ms = 0;
        Check(hipEventElapsedTime(&ms, e0, e1), "elapsed");
        Check(hipMemcpy(gd.data(), yd, dout, hipMemcpyDeviceToHost), "gd");
        double md = 0, mr = 0;
        for (std::size_t i = 0; i < rd.size(); ++i) {
          md = std::fmax(md, std::fabs(double(gd[i]) - rd[i]));
          mr = std::fmax(mr, std::fabs(rd[i]));
        }
        std::printf("%-6d %-6d %9.3f max|delta|=%.3g max|ref|=%.3g\n", width, bound, ms / reps, md, mr);
      }
    ds4_mmq_set_moe_down_bounds(0, 0);
    (void)hipFree(wd); (void)hipFree(xd); (void)hipFree(yd); (void)hipFree(yr);
  }
  return 0;
}
