// GPU test of the few-token routed expert matvec (kernels/expert_vec.hpp):
// gate/up pairs and per-route down launches for 1..8 tokens, a partial row
// tile and an invalid expert ID, against (a) the host build of the kernel's
// own block math on the activations the GPU quantized (tight: indexing,
// staging, slot/route mapping) and (b) core's IQ dequantization in FP64 on
// host-quantized activations (within the mmvq integer-scale bound).
// Written in the source-only round (2026-10-26); first run is the GPU round.

#include "src/models/glm53_flash/kernels/expert_vec.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"

namespace glm = gufo::models::glm53_flash;
namespace ev = gufo::models::glm53_flash::expert_vec;
using gufo::core::GgmlType;

namespace host_tables {
#define __device__
#define __constant__
#include "src/models/glm53_flash/kernels/mmq/iq_tables.hip.hpp"
#undef __device__
#undef __constant__
}  // namespace host_tables

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

struct Buffer {
  void* p{};
  explicit Buffer(std::size_t n) {
    Hip(hipMalloc(&p, std::max<std::size_t>(n, 1)), "hipMalloc");
  }
  ~Buffer() { (void)hipFree(p); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
};

ev::Kind KindOf(GgmlType t) {
  return t == GgmlType::kIQ2_XXS ? ev::Kind::kIq2xxs
         : t == GgmlType::kIQ2_S ? ev::Kind::kIq2s
                                 : ev::Kind::kIq3xxs;
}

void Dequantize(GgmlType t, const std::uint8_t* p, float* out, std::size_t k) {
  if (t == GgmlType::kIQ2_XXS) {
    gufo::quant::DequantizeIQ2_XXS(p, out, k);
  } else if (t == GgmlType::kIQ2_S) {
    gufo::quant::DequantizeIQ2_S(p, out, k);
  } else {
    gufo::quant::DequantizeIQ3_XXS(p, out, k);
  }
}

template<ev::Kind K, int KB>
float HostRow(const std::uint8_t* row, const std::uint8_t* act) {
  const ev::Grids grids{host_tables::kIq2xxsGrid, host_tables::kIq2sGrid,
                        host_tables::kIq3xxsGrid};
  int yq[KB * 64];
  float yd[KB * 8];
  for (int idx = 0; idx < KB * 64; ++idx) {
    int word;
    std::memcpy(&word, act + (idx * sizeof(int)), sizeof(word));
    yq[ev::LdsWord<KB>(idx >> 6, (idx >> 3) & 7, idx & 7)] = word;
  }
  for (int idx = 0; idx < KB * 8; ++idx) {
    float d;
    std::memcpy(&d, act + (KB * 256) + (idx * sizeof(float)), sizeof(d));
    yd[ev::LdsScale<KB>(idx >> 3, idx & 7)] = d;
  }
  float lane[KB];
  for (int b = 0; b < KB; ++b) {
    ev::BlockWords w;
    ev::LoadBlock<K>(row + (b * ev::BlockBytes(K)), &w);
    lane[b] = ev::BlockDot<K, KB>(w, grids, yq, yd, b);
  }
  for (int off = KB / 2; off > 0; off >>= 1) {
    float next[KB];
    for (int l = 0; l < KB; ++l) {
      next[l] = lane[l] + lane[l ^ off];
    }
    std::memcpy(lane, next, sizeof(lane));
  }
  return lane[0];
}

float HostRowAny(GgmlType t, std::uint32_t cols, const std::uint8_t* row,
                 const std::uint8_t* act) {
  const ev::Kind k = KindOf(t);
  if (cols == 4096) {
    return k == ev::Kind::kIq2xxs ? HostRow<ev::Kind::kIq2xxs, 16>(row, act)
           : k == ev::Kind::kIq2s ? HostRow<ev::Kind::kIq2s, 16>(row, act)
                                  : HostRow<ev::Kind::kIq3xxs, 16>(row, act);
  }
  return k == ev::Kind::kIq2xxs ? HostRow<ev::Kind::kIq2xxs, 8>(row, act)
         : k == ev::Kind::kIq2s ? HostRow<ev::Kind::kIq2s, 8>(row, act)
                                : HostRow<ev::Kind::kIq3xxs, 8>(row, act);
}

void Run(GgmlType type, std::uint32_t cols, std::uint32_t tokens, bool pair,
         hipStream_t stream) {
  constexpr std::uint32_t kRows = 40;  // not a multiple of any row tile
  constexpr std::uint32_t kExperts = 12;
  constexpr std::uint32_t kRoutes = 8;
  const std::string name = std::string(pair ? "pair " : "down ") +
                           std::to_string(static_cast<int>(type)) + " k" +
                           std::to_string(cols) + " t" + std::to_string(tokens);
  const std::size_t block = ev::BlockBytes(KindOf(type));
  const std::size_t row_bytes = (cols / 256) * block;
  std::mt19937 rng(97 + cols + tokens + (pair ? 1000 : 0));
  const auto make_weights = [&] {
    std::vector<std::uint8_t> w(kExperts * kRows * row_bytes);
    for (auto& v : w) {
      v = static_cast<std::uint8_t>(rng());
    }
    for (std::size_t b = 0; b < w.size() / block; ++b) {
      const auto d = static_cast<std::uint16_t>(0x1400 + (rng() % 2048));
      std::memcpy(w.data() + (b * block), &d, sizeof(d));
    }
    return w;
  };
  const auto w0 = make_weights();
  const auto w1 = pair ? make_weights() : std::vector<std::uint8_t>{};
  const std::uint32_t slots = tokens * kRoutes;
  const std::uint32_t act_rows = pair ? tokens : slots;
  std::normal_distribution<float> normal(0.0F, 1.0F);
  std::vector<float> x(std::size_t{act_rows} * cols);
  for (auto& v : x) {
    v = normal(rng);
  }
  std::vector<std::int32_t> ids(slots);
  for (std::uint32_t t = 0; t < tokens; ++t) {
    std::vector<std::int32_t> e(kExperts);
    for (std::uint32_t i = 0; i < kExperts; ++i) {
      e[i] = static_cast<std::int32_t>(i);
    }
    std::shuffle(e.begin(), e.end(), rng);
    std::copy_n(e.begin(), kRoutes, ids.begin() + (std::size_t{t} * kRoutes));
  }
  const std::uint32_t bad_slot = tokens > 1 ? kRoutes + 3 : 3;
  ids[bad_slot] = -1;

  Buffer dw0(w0.size()), dw1(std::max<std::size_t>(w1.size(), 1));
  Buffer dx(x.size() * sizeof(float)), dids(ids.size() * sizeof(std::int32_t));
  Buffer do0(std::size_t{slots} * kRows * sizeof(float));
  Buffer do1(std::size_t{slots} * kRows * sizeof(float));
  const std::size_t scratch_bytes = ev::ScratchBytes(act_rows, cols);
  Buffer dscratch(scratch_bytes);
  Hip(hipMemcpy(dw0.p, w0.data(), w0.size(), hipMemcpyHostToDevice), "w0");
  if (pair) {
    Hip(hipMemcpy(dw1.p, w1.data(), w1.size(), hipMemcpyHostToDevice), "w1");
  }
  Hip(hipMemcpy(dx.p, x.data(), x.size() * sizeof(float),
                hipMemcpyHostToDevice),
      "x");
  Hip(hipMemcpy(dids.p, ids.data(), ids.size() * sizeof(std::int32_t),
                hipMemcpyHostToDevice),
      "ids");
  const glm::DeviceMatrix m0{.data = dw0.p,
                             .type = type,
                             .cols = cols,
                             .rows = kRows,
                             .experts = kExperts,
                             .row_bytes = row_bytes};
  glm::DeviceMatrix m1 = m0;
  m1.data = dw1.p;
  Check(name + " supported", ev::Supported(type, cols));
  Check(name + " launch",
        ev::Launch(m0, pair ? &m1 : nullptr, static_cast<const float*>(dx.p),
                   static_cast<const std::int32_t*>(dids.p),
                   static_cast<float*>(do0.p), static_cast<float*>(do1.p),
                   tokens, kRoutes, /*per_route=*/!pair, dscratch.p,
                   scratch_bytes, stream));
  Hip(hipStreamSynchronize(stream), "sync");
  std::vector<float> got0(std::size_t{slots} * kRows);
  std::vector<float> got1(got0.size());
  std::vector<std::uint8_t> act(scratch_bytes);
  Hip(hipMemcpy(got0.data(), do0.p, got0.size() * sizeof(float),
                hipMemcpyDeviceToHost),
      "out0");
  if (pair) {
    Hip(hipMemcpy(got1.data(), do1.p, got1.size() * sizeof(float),
                  hipMemcpyDeviceToHost),
        "out1");
  }
  Hip(hipMemcpy(act.data(), dscratch.p, act.size(), hipMemcpyDeviceToHost),
      "act");

  // The GPU's quantization vs the host rule (a .5 tie may round either way
  // under -ffast-math division; allow one step, on few values).
  std::size_t off_by_one = 0;
  const std::size_t arow = ev::ActivationRowBytes(cols);
  for (std::uint32_t r = 0; r < act_rows; ++r) {
    for (std::uint32_t blk = 0; blk < cols / 32; ++blk) {
      float amax = 0.0F;
      for (int i = 0; i < 32; ++i) {
        amax = std::fmax(
            amax, std::fabs(x[(std::size_t{r} * cols) + (blk * 32) + i]));
      }
      const float d = amax / 127.0F;
      float gd;
      std::memcpy(&gd, act.data() + (r * arow) + cols + (blk * sizeof(float)),
                  sizeof(gd));
      Check(name + " activation scale", gd == d);
      for (int i = 0; i < 32; ++i) {
        const int want =
            ev::QuantizeValue(x[(std::size_t{r} * cols) + (blk * 32) + i], d);
        const int q =
            static_cast<std::int8_t>(act[(r * arow) + (blk * 32) + i]);
        off_by_one += q != want ? 1 : 0;
        Check(name + " activation value", std::abs(q - want) <= 1);
      }
    }
  }
  Check(name + " few rounding ties",
        off_by_one * 1000 <= std::size_t{act_rows} * cols);

  std::vector<float> wrow(cols);
  for (int which = 0; which < (pair ? 2 : 1); ++which) {
    const auto& w = which == 0 ? w0 : w1;
    const auto& got = which == 0 ? got0 : got1;
    for (std::uint32_t s = 0; s < slots; ++s) {
      const std::uint32_t arow_index = pair ? s / kRoutes : s;
      const std::uint8_t* a = act.data() + (arow_index * arow);
      for (std::uint32_t row = 0; row < kRows; ++row) {
        const float g = got[(std::size_t{s} * kRows) + row];
        if (ids[s] < 0) {
          Check(name + " invalid expert is NaN", std::isnan(g));
          continue;
        }
        const std::uint8_t* wr =
            w.data() +
            ((std::size_t{static_cast<std::uint32_t>(ids[s])} * kRows + row) *
             row_bytes);
        const float host = HostRowAny(type, cols, wr, a);
        Check(name + " vs host kernel math slot " + std::to_string(s) +
                  " row " + std::to_string(row),
              std::fabs(g - host) <= 1e-5F * (1.0F + std::fabs(host)));
        // FP64 oracle on the GPU's activations, within the scale bound.
        Dequantize(type, wr, wrow.data(), cols);
        double want = 0.0;
        double bound = 0.0;
        for (std::uint32_t blk = 0; blk < cols / 32; ++blk) {
          float d;
          std::memcpy(&d, a + cols + (blk * sizeof(float)), sizeof(d));
          for (int i = 0; i < 32; ++i) {
            const int q = static_cast<std::int8_t>(a[(blk * 32) + i]);
            want += static_cast<double>(wrow[(blk * 32) + i]) * q * d;
          }
          std::uint16_t dw;
          std::memcpy(&dw, wr + ((blk / 8) * block), sizeof(dw));
          bound += 1.5 * std::fabs(static_cast<double>(ev::HalfBits(dw)) * d);
        }
        Check(name + " vs fp64 dequant slot " + std::to_string(s),
              std::fabs(static_cast<double>(g) - want) <=
                  bound + (1e-4 * std::fabs(want)) + 1e-6);
      }
    }
  }
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
  for (const std::uint32_t tokens : {1U, 2U, 5U, 8U}) {
    Run(GgmlType::kIQ2_XXS, 4096, tokens, /*pair=*/true, stream);
    Run(GgmlType::kIQ2_XXS, 2048, tokens, /*pair=*/true, stream);
    Run(GgmlType::kIQ2_S, 2048, tokens, /*pair=*/false, stream);
    Run(GgmlType::kIQ3_XXS, 2048, tokens, /*pair=*/false, stream);
    Run(GgmlType::kIQ2_S, 4096, tokens, /*pair=*/false, stream);
  }
  // Rejected shapes launch nothing.
  glm::DeviceMatrix bad{.data = reinterpret_cast<void*>(0x1000),
                        .type = GgmlType::kQ2_K,
                        .cols = 4096,
                        .rows = 8,
                        .experts = 4,
                        .row_bytes = 16 * 84};
  Check("Q2_K unsupported", !ev::Supported(bad.type, bad.cols));
  Check("IQ2_XXS k=1024 unsupported", !ev::Supported(GgmlType::kIQ2_XXS, 1024));
  (void)hipStreamDestroy(stream);
  if (g_failures != 0) {
    std::cerr << g_failures << " failure(s)\n";
    return 1;
  }
  std::cout << "expert_vec: all checks passed\n";
  return 0;
}
