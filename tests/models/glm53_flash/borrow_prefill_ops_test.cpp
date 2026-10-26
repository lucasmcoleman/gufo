// GPU test of the prefill operator variants ported in the 2026-10-26 borrow
// round (all default off):
//   KdaLowRankQ8 (GUFO_GLM53_KDA_LOWRANK) vs an FP64 product and vs the
//     LaunchProjection GEMM it replaces;
//   ProjectHeadsQ6KRows 2 / 4 rows per wave (GUFO_GLM53_MLA_HEADS_ROWS) vs
//     the 1-row default kernel, bit for bit;
//   RouterSgemm (GUFO_GLM53_ROUTER_GEMM=sgemm) vs an FP64 product.
// Written in the source-only round; first run is the GPU round.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/glm53_flash/executor.hpp"
#include "src/models/glm53_flash/kernels/kda.hpp"
#include "src/models/glm53_flash/kernels/mla.hpp"
#include "src/models/glm53_flash/kernels/moe.hpp"

namespace glm = gufo::models::glm53_flash;
using gufo::core::GgmlType;

namespace {

int g_failures = 0;

void Check(const std::string& what, bool ok) {
  if (!ok) {
    ++g_failures;
    std::cerr << "FAIL: " << what << "\n";
  }
}

void Hip(hipError_t e, const char* what) {
  if (e != hipSuccess) {
    throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
  }
}

struct Dev {
  void* p{};
  explicit Dev(std::size_t n) { Hip(hipMalloc(&p, n), "hipMalloc"); }
  ~Dev() { (void)hipFree(p); }
  Dev(const Dev&) = delete;
  Dev& operator=(const Dev&) = delete;
  template<typename T>
  T* as() {
    return static_cast<T*>(p);
  }
};

template<typename T>
void Up(Dev& d, const std::vector<T>& h) {
  Hip(hipMemcpy(d.p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice),
      "upload");
}
template<typename T>
std::vector<T> Down(Dev& d, std::size_t n) {
  std::vector<T> h(n);
  Hip(hipMemcpy(h.data(), d.p, n * sizeof(T), hipMemcpyDeviceToHost),
      "download");
  return h;
}

float HalfToFloat(std::uint16_t bits) {
  __half_raw raw;
  raw.x = bits;
  return __half2float(__half(raw));
}

void CheckLowRank(std::uint32_t rows, std::uint32_t tokens,
                  hipStream_t stream) {
  constexpr std::uint32_t kCols = 128;
  const std::size_t row_bytes = (kCols / 32) * 34;
  std::mt19937 rng(17 + rows + tokens);
  std::vector<std::uint8_t> w(rows * row_bytes);
  for (auto& v : w) {
    v = static_cast<std::uint8_t>(rng());
  }
  std::vector<double> wf(std::size_t{rows} * kCols);
  for (std::uint32_t r = 0; r < rows; ++r) {
    for (std::uint32_t b = 0; b < kCols / 32; ++b) {
      std::uint8_t* blk = w.data() + (r * row_bytes) + (b * 34);
      const auto d = static_cast<std::uint16_t>(0x2000 + (rng() % 2048));
      std::memcpy(blk, &d, sizeof(d));
      for (int i = 0; i < 32; ++i) {
        wf[(std::size_t{r} * kCols) + (b * 32) + i] =
            static_cast<double>(HalfToFloat(d)) *
            static_cast<std::int8_t>(blk[2 + i]);
      }
    }
  }
  std::normal_distribution<float> normal(0.0F, 1.0F);
  std::vector<float> x(std::size_t{tokens} * kCols);
  for (auto& v : x) {
    v = normal(rng);
  }
  Dev dw(w.size()), dx(x.size() * 4), dy(std::size_t{tokens} * rows * 4),
      dg(std::size_t{tokens} * rows * 4);
  Up(dw, w);
  Up(dx, x);
  const std::string name =
      "lowrank r" + std::to_string(rows) + " t" + std::to_string(tokens);
  Check(name + " launch",
        glm::rocm::KdaLowRankQ8(dw.p, rows, kCols, dx.as<float>(),
                                dy.as<float>(), tokens, stream));
  const glm::DeviceMatrix m{.data = dw.p,
                            .type = GgmlType::kQ8_0,
                            .cols = kCols,
                            .rows = rows,
                            .experts = 1,
                            .row_bytes = row_bytes};
  Check(
      name + " gemm launch",
      glm::LaunchProjection(m, dx.as<float>(), dg.as<float>(), tokens, stream));
  Hip(hipStreamSynchronize(stream), "sync");
  const auto y = Down<float>(dy, std::size_t{tokens} * rows);
  const auto g = Down<float>(dg, std::size_t{tokens} * rows);
  double worst = 0.0, worst_gemm = 0.0;
  for (std::uint32_t t = 0; t < tokens; ++t) {
    for (std::uint32_t r = 0; r < rows; ++r) {
      double want = 0.0, scale = 0.0;
      for (std::uint32_t k = 0; k < kCols; ++k) {
        const double p =
            wf[(std::size_t{r} * kCols) + k] * x[(std::size_t{t} * kCols) + k];
        want += p;
        scale += std::fabs(p);
      }
      const double got = y[(std::size_t{t} * rows) + r];
      worst = std::max(worst, std::fabs(got - want) / (scale + 1e-30));
      worst_gemm =
          std::max(worst_gemm, std::fabs(got - g[(std::size_t{t} * rows) + r]) /
                                   (scale + 1e-30));
    }
  }
  std::cout << name << ": max |err|/sum|p| vs fp64 " << worst
            << ", vs the GEMM " << worst_gemm << "\n";
  Check(name + " vs fp64", worst <= 1e-6);
  Check(name + " vs the GEMM", worst_gemm <= 1e-5);
}

void CheckHeadsRows(std::uint32_t cols, std::uint32_t rows,
                    std::uint32_t tokens, hipStream_t stream) {
  constexpr std::uint32_t kHeads = 4;
  const std::size_t row_bytes = (cols / 256) * 210;
  std::mt19937 rng(23 + cols + rows + tokens);
  std::vector<std::uint8_t> w(std::size_t{kHeads} * rows * row_bytes);
  for (auto& v : w) {
    v = static_cast<std::uint8_t>(rng());
  }
  for (std::size_t b = 0; b < w.size() / 210; ++b) {
    const auto d = static_cast<std::uint16_t>(0x1800 + (rng() % 1024));
    std::memcpy(w.data() + (b * 210) + 208, &d, sizeof(d));
  }
  std::normal_distribution<float> normal(0.0F, 1.0F);
  std::vector<float> x(std::size_t{kHeads} * tokens * cols);
  for (auto& v : x) {
    v = normal(rng);
  }
  const std::size_t out = std::size_t{kHeads} * tokens * rows;
  Dev dw(w.size()), dx(x.size() * 4), d1(out * 4), d2(out * 4), d4(out * 4);
  Up(dw, w);
  Up(dx, x);
  const glm::DeviceMatrix m{.data = dw.p,
                            .type = GgmlType::kQ6_K,
                            .cols = cols,
                            .rows = rows,
                            .experts = kHeads,
                            .row_bytes = row_bytes};
  const std::string name = "heads k" + std::to_string(cols) + " r" +
                           std::to_string(rows) + " t" + std::to_string(tokens);
  Check(name + " launch",
        glm::mla::ProjectHeadsQ6KRows(m, kHeads, dx.as<float>(), d1.as<float>(),
                                      tokens, 1, stream) &&
            glm::mla::ProjectHeadsQ6KRows(m, kHeads, dx.as<float>(),
                                          d2.as<float>(), tokens, 2, stream) &&
            glm::mla::ProjectHeadsQ6KRows(m, kHeads, dx.as<float>(),
                                          d4.as<float>(), tokens, 4, stream));
  Check(name + " rejects 3 rows per wave",
        !glm::mla::ProjectHeadsQ6KRows(m, kHeads, dx.as<float>(),
                                       d4.as<float>(), tokens, 3, stream));
  Hip(hipStreamSynchronize(stream), "sync");
  const auto a = Down<float>(d1, out);
  const auto b = Down<float>(d2, out);
  const auto c = Down<float>(d4, out);
  Check(name + " 2 rows bitwise",
        std::memcmp(a.data(), b.data(), out * sizeof(float)) == 0);
  Check(name + " 4 rows bitwise",
        std::memcmp(a.data(), c.data(), out * sizeof(float)) == 0);
}

void CheckRouter(std::uint32_t tokens, hipStream_t stream) {
  constexpr std::uint32_t kExperts = 288;
  constexpr std::uint32_t kHidden = 4096;
  std::mt19937 rng(31 + tokens);
  std::normal_distribution<float> normal(0.0F, 1.0F);
  std::vector<float> w(std::size_t{kExperts} * kHidden);
  std::vector<float> x(std::size_t{tokens} * kHidden);
  for (auto& v : w) {
    v = normal(rng) * 0.02F;
  }
  for (auto& v : x) {
    v = normal(rng);
  }
  Dev dw(w.size() * 4), dx(x.size() * 4),
      dy(std::size_t{tokens} * kExperts * 4);
  Up(dw, w);
  Up(dx, x);
  const std::string name = "router t" + std::to_string(tokens);
  Check(name + " launch",
        glm::RouterSgemm(dw.as<float>(), dx.as<float>(), dy.as<float>(), tokens,
                         kExperts, kHidden, stream));
  Hip(hipStreamSynchronize(stream), "sync");
  const auto y = Down<float>(dy, std::size_t{tokens} * kExperts);
  double worst = 0.0;
  for (std::uint32_t t = 0; t < tokens; ++t) {
    for (std::uint32_t e = 0; e < kExperts; ++e) {
      double want = 0.0, scale = 0.0;
      for (std::uint32_t k = 0; k < kHidden; ++k) {
        const double p =
            static_cast<double>(w[(std::size_t{e} * kHidden) + k]) *
            x[(std::size_t{t} * kHidden) + k];
        want += p;
        scale += std::fabs(p);
      }
      worst =
          std::max(worst, std::fabs(y[(std::size_t{t} * kExperts) + e] - want) /
                              (scale + 1e-30));
    }
  }
  std::cout << name << ": max |err|/sum|p| vs fp64 " << worst << "\n";
  // Plain F32 accumulation over 4096 products: a few 1e-7 relative to the
  // absolute sum. The Kahan wave kernel it replaces is ~1e-7 at worst.
  Check(name + " vs fp64", worst <= 5e-6);
}

}  // namespace

int main() {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
    std::cerr << "no HIP device; skipping\n";
    return 77;
  }
  hipStream_t stream;
  Hip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "stream");
  for (const std::uint32_t tokens : {16U, 33U, 256U}) {
    CheckLowRank(200, tokens, stream);
    CheckLowRank(8192, tokens, stream);
  }
  for (const std::uint32_t tokens : {1U, 33U, 70U}) {
    CheckHeadsRows(256, 512, tokens, stream);  // k_b
    CheckHeadsRows(512, 256, tokens, stream);  // v_b
    CheckHeadsRows(256, 36, tokens, stream);   // partial row tile
  }
  for (const std::uint32_t tokens : {9U, 256U}) {
    CheckRouter(tokens, stream);
  }
  (void)hipStreamDestroy(stream);
  if (g_failures != 0) {
    std::cerr << g_failures << " failure(s)\n";
    return 1;
  }
  std::cout << "borrow_prefill_ops: all checks passed\n";
  return 0;
}
