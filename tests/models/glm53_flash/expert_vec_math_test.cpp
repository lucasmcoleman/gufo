// CPU test of the few-token expert matvec arithmetic (expert_vec_math.hpp):
// the host build of the exact BlockDot / LDS staging / shuffle-tree code the
// HIP kernel runs, against core's llama.cpp-exact IQ dequantization times the
// dequantized Q8 activations in FP64. The only allowed gap is mmvq's integer
// scale truncation (< 1.5 units of d_w * d_y per 32-value sub-block) plus
// float rounding. No HIP: this runs on any host.

#include "src/models/glm53_flash/kernels/expert_vec_math.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"

namespace ev = gufo::models::glm53_flash::expert_vec;

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
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
  }
}

void CheckSignNibble() {
  std::mt19937 rng(7);
  for (int trial = 0; trial < 2000; ++trial) {
    std::uint32_t g = 0;
    std::int8_t bytes[4];
    for (int i = 0; i < 4; ++i) {
      bytes[i] = static_cast<std::int8_t>(1 + (rng() % 127));
      g |= static_cast<std::uint32_t>(static_cast<std::uint8_t>(bytes[i]))
           << (8 * i);
    }
    for (std::uint32_t nib = 0; nib < 16; ++nib) {
      const auto got = static_cast<std::uint32_t>(ev::ApplySignNibble(g, nib));
      bool ok = true;
      for (int i = 0; i < 4; ++i) {
        const int want = (nib >> i) & 1U ? -bytes[i] : bytes[i];
        ok = ok && static_cast<std::int8_t>(got >> (8 * i)) == want;
      }
      if (!ok) {
        Check("ApplySignNibble g=" + std::to_string(g) +
                  " nib=" + std::to_string(nib),
              false);
        return;
      }
    }
  }
}

void CheckDp4a() {
  std::mt19937 rng(11);
  for (int trial = 0; trial < 1000; ++trial) {
    const auto a = static_cast<int>(rng());
    const auto b = static_cast<int>(rng());
    const int c = static_cast<int>(rng() % 1000) - 500;
    int want = c;
    for (int i = 0; i < 4; ++i) {
      want +=
          static_cast<std::int8_t>(static_cast<std::uint32_t>(a) >> (8 * i)) *
          static_cast<std::int8_t>(static_cast<std::uint32_t>(b) >> (8 * i));
    }
    if (ev::Dp4a(a, b, c) != want) {
      Check("Dp4a", false);
      return;
    }
  }
}

void CheckHalf() {
  // A few exact values, a subnormal and the infinities.
  const struct {
    std::uint16_t bits;
    float value;
  } cases[] = {{0x3C00, 1.0F},        {0xC000, -2.0F},
               {0x3555, 0.33325195F}, {0x0001, 5.9604645e-08F},
               {0x7BFF, 65504.0F},    {0x0000, 0.0F}};
  for (const auto& c : cases) {
    Check("HalfBits " + std::to_string(c.bits),
          ev::HalfBits(c.bits) == c.value);
  }
  Check("HalfBits inf", std::isinf(ev::HalfBits(0x7C00)));
  Check("HalfBits nan", std::isnan(ev::HalfBits(0x7E00)));
}

void CheckParseMode() {
  const auto is = [](ev::Mode m, bool g, bool d) {
    return m.gate_up == g && m.down == d;
  };
  Check("mode unset", is(ev::ParseMode(nullptr), true, true));
  Check("mode empty", is(ev::ParseMode(""), true, true));
  Check("mode 0", is(ev::ParseMode("0"), false, false));
  Check("mode 1", is(ev::ParseMode("1"), true, true));
  Check("mode all", is(ev::ParseMode("all"), true, true));
  Check("mode gateup", is(ev::ParseMode("gateup"), true, false));
  Check("mode down", is(ev::ParseMode("down"), false, true));
  Check("mode junk", is(ev::ParseMode("yes"), false, false));
}

// Random but valid block bytes: every index/sign/scale pattern decodes; the
// scale d is a moderate positive or negative half.
std::vector<std::uint8_t> RandomBlocks(std::mt19937& rng, int block_bytes,
                                       int blocks) {
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(block_bytes) *
                                  blocks);
  for (auto& b : bytes) {
    b = static_cast<std::uint8_t>(rng());
  }
  for (int i = 0; i < blocks; ++i) {
    // |d| in [2^-9, 2^-6), random sign.
    const std::uint16_t d = static_cast<std::uint16_t>(
        ((rng() & 1U) << 15U) | ((6U + (rng() % 3U)) << 10U) |
        (rng() & 0x3FFU));
    std::memcpy(bytes.data() + (static_cast<std::size_t>(i) * block_bytes), &d,
                sizeof(d));
  }
  return bytes;
}

template<ev::Kind K>
void Dequantize(const std::uint8_t* row, float* out, std::size_t k) {
  if constexpr (K == ev::Kind::kIq2xxs) {
    gufo::quant::DequantizeIQ2_XXS(row, out, k);
  } else if constexpr (K == ev::Kind::kIq2s) {
    gufo::quant::DequantizeIQ2_S(row, out, k);
  } else {
    gufo::quant::DequantizeIQ3_XXS(row, out, k);
  }
}

template<ev::Kind K, int KB>
void CheckRows(const char* name, unsigned seed) {
  constexpr int kCols = KB * 256;
  constexpr int kRows = 24;
  std::mt19937 rng(seed);
  const ev::Grids grids{host_tables::kIq2xxsGrid, host_tables::kIq2sGrid,
                        host_tables::kIq3xxsGrid};
  const auto weights = RandomBlocks(rng, ev::BlockBytes(K), kRows * KB);
  std::normal_distribution<float> normal(0.0F, 1.0F);
  for (int variant = 0; variant < 3; ++variant) {
    // Activation row as the quantize kernel writes it (int8 values, then
    // F32 scales), including an all-zero block (d = 0) in variant 2.
    std::vector<float> x(kCols);
    for (int i = 0; i < kCols; ++i) {
      x[i] = normal(rng) * (variant == 1 ? 40.0F : 1.0F);
    }
    if (variant == 2) {
      std::fill(x.begin() + 64, x.begin() + 96, 0.0F);
    }
    std::vector<std::uint8_t> act(ev::ActivationRowBytes(kCols));
    std::vector<double> xq(kCols);
    for (int blk = 0; blk < kCols / 32; ++blk) {
      float amax = 0.0F;
      for (int i = 0; i < 32; ++i) {
        amax = std::fmax(amax, std::fabs(x[(blk * 32) + i]));
      }
      const float d = amax / 127.0F;
      for (int i = 0; i < 32; ++i) {
        const int q = ev::QuantizeValue(x[(blk * 32) + i], d);
        Check(std::string(name) + " q range", q >= -127 && q <= 127);
        act[(blk * 32) + i] =
            static_cast<std::uint8_t>(static_cast<std::int8_t>(q));
        xq[(blk * 32) + i] = static_cast<double>(q) * d;
      }
      std::memcpy(act.data() + kCols + (blk * sizeof(float)), &d, sizeof(d));
    }
    // LDS staging exactly as ExpertVecKernel does it.
    int yq[KB * 64];
    float yd[KB * 8];
    for (int idx = 0; idx < KB * 64; ++idx) {
      int word;
      std::memcpy(&word, act.data() + (idx * sizeof(int)), sizeof(word));
      yq[ev::LdsWord<KB>(idx >> 6, (idx >> 3) & 7, idx & 7)] = word;
    }
    for (int idx = 0; idx < KB * 8; ++idx) {
      float d;
      std::memcpy(&d, act.data() + kCols + (idx * sizeof(float)), sizeof(d));
      yd[ev::LdsScale<KB>(idx >> 3, idx & 7)] = d;
    }
    std::vector<float> w(kCols);
    for (int row = 0; row < kRows; ++row) {
      const std::uint8_t* p = weights.data() + (static_cast<std::size_t>(row) *
                                                KB * ev::BlockBytes(K));
      // Lanes b = 0..KB-1, then the kernel's __shfl_xor tree.
      float lane[KB];
      double bound = 0.0;
      for (int b = 0; b < KB; ++b) {
        ev::BlockWords words;
        ev::LoadBlock<K>(p + (b * ev::BlockBytes(K)), &words);
        lane[b] = ev::BlockDot<K, KB>(words, grids, yq, yd, b);
        for (int ib = 0; ib < 8; ++ib) {
          bound += 1.5 * std::fabs(static_cast<double>(words.d) *
                                   yd[ev::LdsScale<KB>(b, ib)]);
        }
      }
      for (int off = KB / 2; off > 0; off >>= 1) {
        float next[KB];
        for (int l = 0; l < KB; ++l) {
          next[l] = lane[l] + lane[l ^ off];
        }
        std::memcpy(lane, next, sizeof(lane));
      }
      Dequantize<K>(p, w.data(), kCols);
      double want = 0.0;
      double scale = 0.0;
      for (int i = 0; i < kCols; ++i) {
        want += static_cast<double>(w[i]) * xq[i];
        scale += std::fabs(static_cast<double>(w[i]) * xq[i]);
      }
      const double err = std::fabs(static_cast<double>(lane[0]) - want);
      const double tol = bound + (1e-5 * scale);
      if (!(err <= tol)) {
        Check(std::string(name) + " row " + std::to_string(row) + " variant " +
                  std::to_string(variant) + ": got " + std::to_string(lane[0]) +
                  " want " + std::to_string(want) + " err " +
                  std::to_string(err) + " tol " + std::to_string(tol),
              false);
      }
      // All lanes agree after the butterfly (every lane holds the sum).
      for (int l = 1; l < KB; ++l) {
        Check(std::string(name) + " butterfly lanes agree", lane[l] == lane[0]);
      }
    }
  }
}

}  // namespace

int main() {
  CheckSignNibble();
  CheckDp4a();
  CheckHalf();
  CheckParseMode();
  CheckRows<ev::Kind::kIq2xxs, 16>("iq2_xxs k4096", 1);
  CheckRows<ev::Kind::kIq2xxs, 8>("iq2_xxs k2048", 2);
  CheckRows<ev::Kind::kIq2s, 8>("iq2_s k2048", 3);
  CheckRows<ev::Kind::kIq2s, 16>("iq2_s k4096", 4);
  CheckRows<ev::Kind::kIq3xxs, 8>("iq3_xxs k2048", 5);
  CheckRows<ev::Kind::kIq3xxs, 16>("iq3_xxs k4096", 6);
  if (g_failures != 0) {
    std::fprintf(stderr, "%d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("expert_vec_math: all checks passed\n");
  return 0;
}
