#ifndef GUFO_MODELS_GLM53_FLASH_KERNELS_EXPERT_VEC_MATH_HPP_
#define GUFO_MODELS_GLM53_FLASH_KERNELS_EXPERT_VEC_MATH_HPP_

// Per-block arithmetic of the few-token routed expert matvec
// (kernels/expert_vec.hip.cpp), shared by the HIP kernel and a host
// emulation so the bit tricks have a CPU test (expert_vec_math_test.cpp).
//
// Idea ported from mikealanni/gufo c338778f / 8808f1a0 / 5b505adf (MIT, Mike
// Alani): one lane owns one 256-weight block of a row and reads it with
// 16-byte loads; the row's KB = K / 256 lanes reduce with shuffles; the
// slot's Q8 activation row sits in LDS, interleaved so the lanes of a row
// read consecutive words. The dot products are llama.cpp's mmvq dp4a sums
// (vec_dot_iq2_xxs_q8_1 / iq2_s / iq3_xxs) with their integer scale
// rounding; only the float summation order differs from mmvq.
//
// Activation format (GLM-local, not ggml's block_q8_1): per row, K int8
// values, then K / 32 F32 scales d, with q = round(x / d), d = amax / 127.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

#if defined(__HIPCC__)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#define GLM53_EV_HD __host__ __device__
#else
#define GLM53_EV_HD
#endif

namespace gufo::models::glm53_flash::expert_vec {

/// Bytes of one 256-weight block per encoding.
inline constexpr int kIq2xxsBlockBytes = 66;
inline constexpr int kIq2sBlockBytes = 82;
inline constexpr int kIq3xxsBlockBytes = 98;

/// GUFO_GLM53_EXPERT_VEC: which routed projections take this kernel.
struct Mode {
  bool gate_up{false};
  bool down{false};
};

/// Parses a GUFO_GLM53_EXPERT_VEC value: unset, empty, "0" or anything
/// unknown is off; "1" / "all" both, "gateup" or "down" one of them.
inline Mode ParseMode(const char* value) noexcept {
  if (value == nullptr) {
    return {};
  }
  if (std::strcmp(value, "1") == 0 || std::strcmp(value, "all") == 0) {
    return {.gate_up = true, .down = true};
  }
  if (std::strcmp(value, "gateup") == 0) {
    return {.gate_up = true, .down = false};
  }
  if (std::strcmp(value, "down") == 0) {
    return {.gate_up = false, .down = true};
  }
  return {};
}

enum class Kind : std::uint8_t { kIq2xxs, kIq2s, kIq3xxs };

GLM53_EV_HD constexpr int BlockBytes(Kind k) {
  return k == Kind::kIq2xxs ? kIq2xxsBlockBytes
         : k == Kind::kIq2s ? kIq2sBlockBytes
                            : kIq3xxsBlockBytes;
}

/// Bytes of one quantized activation row of `k` values (k % 32 == 0).
GLM53_EV_HD constexpr std::size_t ActivationRowBytes(std::size_t k) {
  return k + ((k / 32) * sizeof(float));
}

/// Index of activation word `wd` (0..7) of 32-value block `ib` (0..7) of
/// superblock `b` (0..KB-1) in the interleaved LDS copy: lanes b of one row
/// read consecutive words.
template<int KB>
GLM53_EV_HD constexpr int LdsWord(int b, int ib, int wd) {
  return (((ib * 8) + wd) * KB) + b;
}
template<int KB>
GLM53_EV_HD constexpr int LdsScale(int b, int ib) {
  return (ib * KB) + b;
}

GLM53_EV_HD inline std::uint32_t Popc(std::uint32_t v) {
#if defined(__HIP_DEVICE_COMPILE__)
  return static_cast<std::uint32_t>(__popc(v));
#else
  return static_cast<std::uint32_t>(__builtin_popcount(v));
#endif
}

/// Negates byte i of `g` where bit i of `nib` (0..15) is set. Grid
/// magnitudes are below 128 and never zero, so no byte borrows from the next.
GLM53_EV_HD inline int ApplySignNibble(std::uint32_t g, std::uint32_t nib) {
  const std::uint32_t spread = (nib * 0x00204081U) & 0x01010101U;
#if defined(__HIP_DEVICE_COMPILE__)
  const std::uint32_t mask =
      __builtin_amdgcn_perm(0U, 0U, 0x0C0C0C0CU + spread);
#else
  const std::uint32_t mask = spread * 0xFFU;
#endif
  return static_cast<int>((g ^ mask) + spread);
}

/// sum_i a.byte[i] * b.byte[i] (signed) + c.
GLM53_EV_HD inline int Dp4a(int a, int b, int c) {
#if defined(__HIP_DEVICE_COMPILE__) &&                                       \
    (defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
     defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__) || \
     defined(__gfx1152__) || defined(__gfx1200__) || defined(__gfx1201__))
  return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#else
  for (int i = 0; i < 4; ++i) {
    c += static_cast<int>(static_cast<std::int8_t>(
             static_cast<std::uint32_t>(a) >> (8 * i))) *
         static_cast<int>(static_cast<std::int8_t>(
             static_cast<std::uint32_t>(b) >> (8 * i)));
  }
  return c;
#endif
}

GLM53_EV_HD inline float HalfBits(std::uint16_t h) {
#if defined(__HIP_DEVICE_COMPILE__)
  return __half2float(__ushort_as_half(h));
#else
  const int exp = (h >> 10U) & 0x1F;
  const int man = h & 0x3FF;
  float mag;
  if (exp == 0x1F) {
    mag = man != 0 ? std::numeric_limits<float>::quiet_NaN()
                   : std::numeric_limits<float>::infinity();
  } else if (exp == 0) {
    mag = std::ldexp(static_cast<float>(man), -24);
  } else {
    mag = std::ldexp(static_cast<float>(man | 0x400), exp - 25);
  }
  return (h & 0x8000U) != 0 ? -mag : mag;
#endif
}

/// One block's raw bytes as a lane loads them: 64 bytes after the scale d
/// (`q`, 16 words) and, past them, up to 32 more bytes (`tail`, 8 words):
/// IQ2_S qh | scales (4 words), IQ3_XXS scales-and-signs (8 words).
struct BlockWords {
  std::uint32_t q[16];
  std::uint32_t tail[8];
  float d;
};

/// Grid tables, by reference so the kernel passes its __constant__ copies
/// and the host test its own.
struct Grids {
  const std::uint64_t* iq2xxs;  ///< [256]
  const std::uint64_t* iq2s;    ///< [1024]
  const std::uint32_t* iq3xxs;  ///< [256]
};

/// Dot product of one block (superblock b of the row) with the staged
/// activation: sum over the 8 sub-blocks of d * d_y * sumi, sumi scaled as
/// llama.cpp's mmvq vec_dot_<type>_q8_1 does.
template<Kind K, int KB>
GLM53_EV_HD inline float BlockDot(const BlockWords& w, const Grids& grids,
                                  const int* yq, const float* yd, int b) {
  float acc = 0.0F;
#if defined(__HIP_DEVICE_COMPILE__)
#pragma unroll
#endif
  for (int ib = 0; ib < 8; ++ib) {
    int sumi = 0;
    if constexpr (K == Kind::kIq2xxs) {
      const std::uint32_t idx = w.q[2 * ib];
      const std::uint32_t aux = w.q[(2 * ib) + 1];
#if defined(__HIP_DEVICE_COMPILE__)
#pragma unroll
#endif
      for (int k0 = 0; k0 < 8; k0 += 2) {
        const std::uint64_t g = grids.iq2xxs[(idx >> (4 * k0)) & 0xFFU];
        const std::uint32_t s7 = (aux >> (7 * k0 / 2)) & 127U;
        const std::uint32_t s8 = s7 | ((Popc(s7) & 1U) << 7U);
        sumi = Dp4a(ApplySignNibble(static_cast<std::uint32_t>(g), s8 & 15U),
                    yq[LdsWord<KB>(b, ib, k0)], sumi);
        sumi = Dp4a(
            ApplySignNibble(static_cast<std::uint32_t>(g >> 32U), s8 >> 4U),
            yq[LdsWord<KB>(b, ib, k0 + 1)], sumi);
      }
      const int ls = static_cast<int>((aux >> 27U) | 1U);
      sumi = sumi * ls / 8;
    } else if constexpr (K == Kind::kIq2s) {
      const std::uint32_t idx = w.q[ib];
      const std::uint32_t sgn = w.q[8 + ib];
      const int qh =
          static_cast<int>((w.tail[ib >> 2] >> (8 * (ib & 3))) & 0xFFU);
      const int scales =
          static_cast<int>((w.tail[2 + (ib >> 2)] >> (8 * (ib & 3))) & 0xFFU);
      int sumi0 = 0;
      int sumi1 = 0;
#if defined(__HIP_DEVICE_COMPILE__)
#pragma unroll
#endif
      for (int l0 = 0; l0 < 8; l0 += 2) {
        const std::uint64_t g =
            grids.iq2s[((idx >> (4 * l0)) & 0xFFU) |
                       (static_cast<std::uint32_t>(qh << (8 - l0)) & 0x300U)];
        const std::uint32_t s = (sgn >> (4 * l0)) & 0xFFU;
        const int lo = ApplySignNibble(static_cast<std::uint32_t>(g), s & 15U);
        const int hi =
            ApplySignNibble(static_cast<std::uint32_t>(g >> 32U), s >> 4U);
        int& part = l0 < 4 ? sumi0 : sumi1;
        part = Dp4a(lo, yq[LdsWord<KB>(b, ib, l0)], part);
        part = Dp4a(hi, yq[LdsWord<KB>(b, ib, l0 + 1)], part);
      }
      sumi = ((sumi0 * (scales & 0x0F)) + (sumi1 * (scales >> 4)) +
              ((sumi0 + sumi1) / 2)) /
             4;
    } else {
      // IQ3_XXS: q holds 64 grid-index bytes (8 per sub-block), tail the
      // per-sub-block {4 x 7-bit sign index, 4-bit scale} words.
      const std::uint32_t idx0 = w.q[2 * ib];
      const std::uint32_t idx1 = w.q[(2 * ib) + 1];
      const std::uint32_t aux = w.tail[ib];
#if defined(__HIP_DEVICE_COMPILE__)
#pragma unroll
#endif
      for (int l0 = 0; l0 < 8; l0 += 2) {
        const std::uint32_t word = l0 < 4 ? idx0 : idx1;
        const int shift = 8 * (l0 & 3);
        const std::uint32_t g0 = grids.iq3xxs[(word >> shift) & 0xFFU];
        const std::uint32_t g1 = grids.iq3xxs[(word >> (shift + 8)) & 0xFFU];
        const std::uint32_t s7 = (aux >> (7 * l0 / 2)) & 127U;
        const std::uint32_t s8 = s7 | ((Popc(s7) & 1U) << 7U);
        sumi = Dp4a(ApplySignNibble(g0, s8 & 15U), yq[LdsWord<KB>(b, ib, l0)],
                    sumi);
        sumi = Dp4a(ApplySignNibble(g1, s8 >> 4U),
                    yq[LdsWord<KB>(b, ib, l0 + 1)], sumi);
      }
      const int ls = static_cast<int>(aux >> 28U);
      sumi = ((ls * sumi) + (sumi / 2)) / 2;
    }
    acc += w.d * yd[LdsScale<KB>(b, ib)] * static_cast<float>(sumi);
  }
  return acc;
}

/// Host / reference decode of a block's bytes into BlockWords (the kernel
/// does the same with 16-byte loads).
template<Kind K>
inline void LoadBlock(const std::uint8_t* p, BlockWords* w) {
  std::uint16_t h;
  std::memcpy(&h, p, sizeof(h));
  w->d = HalfBits(h);
  std::memcpy(w->q, p + 2, sizeof(w->q));
  std::memset(w->tail, 0, sizeof(w->tail));
  if constexpr (K == Kind::kIq2s) {
    std::memcpy(w->tail, p + 66, 16);
  } else if constexpr (K == Kind::kIq3xxs) {
    std::memcpy(w->tail, p + 66, 32);
  }
}

/// One activation value as the quantize kernel stores it: q = round(x /
/// d) with d = amax / 127 of its 32-value block (0 when amax is 0).
GLM53_EV_HD inline int QuantizeValue(float x, float d) {
  return d == 0.0F ? 0 : static_cast<int>(roundf(x / d));
}

}  // namespace gufo::models::glm53_flash::expert_vec

#endif  // GUFO_MODELS_GLM53_FLASH_KERNELS_EXPERT_VEC_MATH_HPP_
