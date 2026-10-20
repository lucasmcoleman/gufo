// NextN routed-expert kernels probe (not a test): ds4_mmq_q2_K_moe(_vec) on
// the real blk.45 Q2_K gate experts and ds4_mmq_q3_K_moe(_vec) on its Q3_K
// down experts, against a host dequantize-and-dot oracle (double
// accumulation) over sampled output rows. The GPU rounds activations to
// Q8_1, so the gate is a relative L2 bound, plus a mutant (wrong expert)
// that must fail it.
//
// usage: glm53_flash_kquant_moe_probe MODEL.gguf
// Exit 0 when every case is within kTol and every mutant is outside it.

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "src/models/deepseek_v4_flash/kernels/rocm/mmq/ds4_mmq.h"
#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/glm53_flash/weights.hpp"

namespace glm = gufo::models::glm53_flash;
namespace quant = gufo::quant;
using gufo::core::GgmlType;

namespace {

constexpr double kTol = 2e-2;
constexpr int kUsed = 8;

void Check(hipError_t e, const char* what) {
  if (e != hipSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e));
    std::exit(1);
  }
}

template<class T>
T* Upload(const void* host, std::size_t bytes) {
  void* p = nullptr;
  Check(hipMalloc(&p, bytes), "hipMalloc");
  Check(hipMemcpy(p, host, bytes, hipMemcpyHostToDevice), "upload");
  return static_cast<T*>(p);
}

void DequantRow(const glm::TensorRef& t, std::uint64_t e, std::uint64_t r,
                float* out) {
  const std::uint8_t* src = t.Expert(e) + (r * t.RowBytes());
  if (t.type == GgmlType::kQ2_K) {
    quant::DequantizeQ2_K(src, out, t.cols);
  } else {
    quant::DequantizeQ3_K(src, out, t.cols);
  }
}

// One probe: `rows` activation rows; row i uses experts ids[i * used ..].
// broadcast: every route of a token reads that token's x row (gate/up);
// otherwise one x row per (row) with used = 1 (down).
int Probe(const char* label, const glm::TensorRef& w, int n_tokens, int used,
          bool vec, std::mt19937& rng) {
  const int M = static_cast<int>(w.rows), K = static_cast<int>(w.cols),
            E = static_cast<int>(w.experts);
  std::normal_distribution<float> nd(0.0F, 1.0F);
  std::vector<float> x(static_cast<std::size_t>(n_tokens) * K);
  for (auto& v : x) v = nd(rng);
  std::vector<std::int32_t> ids(static_cast<std::size_t>(n_tokens) * used);
  for (int t = 0; t < n_tokens; ++t) {
    for (int s = 0; s < used; ++s) {
      // Distinct experts per token, as the router produces.
      ids[(t * used) + s] = static_cast<std::int32_t>(
          (std::uniform_int_distribution<int>(0, E / used - 1)(rng) * used) + s);
    }
  }
  static void* dw_cache = nullptr;
  static const void* dw_src = nullptr;
  if (dw_src != w.data) {
    if (dw_cache) (void)hipFree(dw_cache);
    dw_cache = Upload<void>(w.data, w.SizeBytes());
    dw_src = w.data;
  }
  auto* dx = Upload<float>(x.data(), x.size() * 4);
  auto* dids = Upload<std::int32_t>(ids.data(), ids.size() * 4);
  float* dy = nullptr;
  const std::size_t out_rows = static_cast<std::size_t>(n_tokens) * used;
  Check(hipMalloc(&dy, out_rows * M * 4), "hipMalloc y");
  int rc = -1;
  if (w.type == GgmlType::kQ2_K) {
    rc = (vec ? ds4_mmq_q2_K_moe_vec : ds4_mmq_q2_K_moe)(
        dw_cache, dx, dids, dy, M, K, n_tokens, E, used, nullptr);
  } else {
    rc = (vec ? ds4_mmq_q3_K_moe_vec : ds4_mmq_q3_K_moe)(
        dw_cache, dx, dids, dy, M, K, n_tokens, E, used, nullptr);
  }
  Check(hipDeviceSynchronize(), "sync");
  std::vector<float> y(out_rows * M);
  Check(hipMemcpy(y.data(), dy, y.size() * 4, hipMemcpyDeviceToHost), "y");
  (void)hipFree(dx);
  (void)hipFree(dids);
  (void)hipFree(dy);
  if (rc != 0) {
    std::cout << "FAIL " << label << ": kernel returned " << rc << "\n";
    return 1;
  }
  // Oracle over sampled rows (every 37th output row) of every route.
  std::vector<float> wr(K);
  double num = 0, den = 0, mnum = 0;
  for (std::size_t o = 0; o < out_rows; ++o) {
    const int t = static_cast<int>(o / used);
    const int e = ids[o];
    const int wrong = (e + 1) % E;
    const float* xt = x.data() + (static_cast<std::size_t>(t) * K);
    for (int r = static_cast<int>(o % 37); r < M; r += 37) {
      DequantRow(w, e, r, wr.data());
      double want = 0;
      for (int c = 0; c < K; ++c) want += static_cast<double>(wr[c]) * xt[c];
      DequantRow(w, wrong, r, wr.data());
      double mut = 0;
      for (int c = 0; c < K; ++c) mut += static_cast<double>(wr[c]) * xt[c];
      const double got = y[(o * M) + r];
      num += (got - want) * (got - want);
      den += want * want;
      mnum += (mut - want) * (mut - want);
    }
  }
  const double rel = std::sqrt(num / den), mrel = std::sqrt(mnum / den);
  const bool ok = std::isfinite(rel) && rel <= kTol && mrel > kTol;
  std::printf("%s %s: rel %.3e (<= %.0e), wrong-expert mutant %.3e\n",
              ok ? "ok  " : "FAIL", label, rel, kTol, mrel);
  return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: glm53_flash_kquant_moe_probe MODEL.gguf\n";
    return 2;
  }
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(argv[1], &error);
  if (!reader) {
    std::cerr << "open failed: " << error << "\n";
    return 1;
  }
  auto weights = glm::ModelWeights::Bind(*reader, /*bind_nextn=*/true, &error);
  if (!weights) {
    std::cerr << "bind failed: " << error << "\n";
    return 1;
  }
  const glm::LayerWeights& l = weights->layers.back();
  if (!l.nextn || l.ffn_gate_exps.type != GgmlType::kQ2_K ||
      l.ffn_down_exps.type != GgmlType::kQ3_K) {
    std::cerr << "expected NextN Q2_K gate / Q3_K down experts\n";
    return 1;
  }
  std::mt19937 rng(1234);
  int bad = 0;
  bad += Probe("Q2_K gate vec 1 token x 8 routes", l.ffn_gate_exps, 1, kUsed,
               true, rng);
  bad += Probe("Q2_K gate vec 5 tokens x 8 routes", l.ffn_gate_exps, 5, kUsed,
               true, rng);
  bad += Probe("Q2_K gate mmq 37 tokens x 8 routes", l.ffn_gate_exps, 37,
               kUsed, false, rng);
  bad += Probe("Q3_K down vec 8 rows", l.ffn_down_exps, 8, 1, true, rng);
  bad += Probe("Q3_K down vec 40 rows", l.ffn_down_exps, 40, 1, true, rng);
  bad += Probe("Q3_K down mmq 296 rows", l.ffn_down_exps, 296, 1, false, rng);
  std::cout << (bad == 0 ? "PASS" : "FAIL") << "\n";
  return bad == 0 ? 0 : 1;
}
