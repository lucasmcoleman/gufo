// Prefill projection micro-benchmark (not a test): times the exact FP32 path
// (LaunchBatchedQuantGEMMFp32) against the Q8_1 WMMA path (PrefillQuantGemm)
// for GLM-5.3-Flash's dense projection shapes, with synthetic weights.
// usage: glm53_flash_proj_bench [reps]
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/glm53_flash/kernels/prefill_gemm.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"

using gufo::core::GgmlType;

namespace {
struct Shape {
  const char* name;
  GgmlType type;
  std::size_t rows, cols;
};
void Check(hipError_t e, const char* what) {
  if (e != hipSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e));
    std::exit(1);
  }
}
}  // namespace

int main(int argc, char** argv) {
  const int reps = argc > 1 ? std::atoi(argv[1]) : 10;
  const Shape shapes[] = {
      {"kda/qkv", GgmlType::kQ6_K, 24576, 4096},
      {"kda/out", GgmlType::kQ6_K, 4096, 8192},
      {"kda/f_a,g_a", GgmlType::kQ8_0, 128, 4096},
      {"kda/beta", GgmlType::kQ8_0, 64, 4096},
      {"mla/q_a", GgmlType::kQ6_K, 1536, 4096},
      {"mla/q_b", GgmlType::kQ6_K, 16384, 1536},
      {"mla/kv_a", GgmlType::kQ6_K, 512, 4096},
      {"mla/out", GgmlType::kQ6_K, 4096, 16384},
      {"idx/q_b", GgmlType::kQ8_0, 4096, 1536},
      {"idx/k,gate", GgmlType::kQ8_0, 128, 4096},
      {"dense/gate,up", GgmlType::kQ6_K, 12288, 4096},
      {"dense/down", GgmlType::kQ6_K, 4096, 12288},
      {"shexp/gate,up", GgmlType::kQ6_K, 2048, 4096},
      {"shexp/down", GgmlType::kQ6_K, 4096, 2048},
  };
  hipStream_t stream;
  Check(hipStreamCreate(&stream), "stream");
  hipEvent_t a, b;
  Check(hipEventCreate(&a), "event");
  Check(hipEventCreate(&b), "event");
  std::mt19937 rng(7);
  std::printf("%-15s %6s %10s %10s %7s\n", "shape", "tokens", "fp32 ms", "wmma ms",
              "speedup");
  for (const auto& s : shapes) {
    const auto block = gufo::quant::QuantizedBlockElements(s.type);
    const auto block_bytes = s.type == GgmlType::kQ8_0 ? 34 : 210;
    const std::size_t row_bytes = s.cols / block * block_bytes;
    std::vector<std::uint8_t> host(row_bytes * s.rows);
    for (auto& v : host) v = static_cast<std::uint8_t>(rng());
    // Keep each block's fp16 scale small and finite (Q6_K: last 2 bytes).
    for (std::size_t o = 0; o < host.size(); o += block_bytes) {
      const std::uint16_t scale = 0x1400 + rng() % 1024;
      std::memcpy(host.data() + o + (s.type == GgmlType::kQ6_K ? block_bytes - 2 : 0),
                  &scale, 2);
    }
    void *w, *x, *y;
    const std::size_t max_t = 256;
    Check(hipMalloc(&w, host.size()), "w");
    Check(hipMemcpy(w, host.data(), host.size(), hipMemcpyHostToDevice), "w copy");
    std::vector<float> hx(max_t * s.cols);
    std::uniform_real_distribution<float> d(-1, 1);
    for (auto& v : hx) v = d(rng);
    Check(hipMalloc(&x, hx.size() * 4), "x");
    Check(hipMemcpy(x, hx.data(), hx.size() * 4, hipMemcpyHostToDevice), "x copy");
    Check(hipMalloc(&y, max_t * s.rows * 4), "y");
    for (std::size_t t : {std::size_t(10), std::size_t(16), std::size_t(24), std::size_t(32), std::size_t(64), std::size_t(128), std::size_t(256)}) {
      float ms[2];
      for (int path = 0; path < 2; ++path) {
        auto run = [&] {
          if (path == 0 ||
              !gufo::models::glm53_flash::PrefillQuantGemm(
                  s.type, w, static_cast<float*>(x), static_cast<float*>(y), t,
                  s.rows, s.cols, stream))
            ::gufo::hip::LaunchBatchedQuantGEMMFp32(
                s.type, w, static_cast<float*>(x), static_cast<float*>(y), t,
                s.rows, s.cols, stream);
        };
        run();  // warm-up (and scratch growth)
        Check(hipStreamSynchronize(stream), "warm");
        Check(hipEventRecord(a, stream), "rec");
        for (int i = 0; i < reps; ++i) run();
        Check(hipEventRecord(b, stream), "rec");
        Check(hipEventSynchronize(b), "sync");
        Check(hipEventElapsedTime(&ms[path], a, b), "elapsed");
        ms[path] /= reps;
      }
      std::printf("%-15s %6zu %10.3f %10.3f %6.2fx\n", s.name, t, ms[0], ms[1],
                  ms[0] / ms[1]);
    }
    (void)hipFree(w); (void)hipFree(x); (void)hipFree(y);
  }
  return 0;
}
