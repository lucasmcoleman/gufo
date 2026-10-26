#include "src/models/laguna/reference.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <thread>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/core/quant/ggml_gemm.hpp"

namespace gufo::models::laguna::reference {
namespace {

using core::GgmlType;

constexpr float kPi = 3.14159265358979323846F;
/// llama.cpp clamps the selected-sigmoid sum to the smallest normal F16.
constexpr float kRouterSumClamp = 0x1p-14F;

float Bf16ToFloat(std::uint16_t h) noexcept {
  const std::uint32_t bits = static_cast<std::uint32_t>(h) << 16U;
  float f = 0.0F;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

float RoundToBf16(float value) noexcept {
  if (!std::isfinite(value)) {
    return value;
  }
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7FFFU + ((bits >> 16U) & 1U);  // round to nearest even
  bits &= 0xFFFF0000U;
  float f = 0.0F;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

std::uint16_t FloatToHalfBits(float value) noexcept {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t sign = (bits >> 16U) & 0x8000U;
  const std::int32_t exponent =
      static_cast<std::int32_t>((bits >> 23U) & 0xFFU) - 127 + 15;
  std::uint32_t mantissa = bits & 0x7FFFFFU;
  if (((bits >> 23U) & 0xFFU) == 0xFFU) {  // inf / nan
    return static_cast<std::uint16_t>(sign | 0x7C00U |
                                      (mantissa != 0 ? 0x200U : 0U));
  }
  if (exponent >= 31) {  // overflow to inf
    return static_cast<std::uint16_t>(sign | 0x7C00U);
  }
  if (exponent <= 0) {  // subnormal half or zero
    if (exponent < -10) {
      return static_cast<std::uint16_t>(sign);
    }
    mantissa |= 0x800000U;
    const std::uint32_t shift = static_cast<std::uint32_t>(14 - exponent);
    std::uint32_t half = mantissa >> shift;
    const std::uint32_t remainder = mantissa & ((1U << shift) - 1U);
    const std::uint32_t halfway = 1U << (shift - 1U);
    if (remainder > halfway || (remainder == halfway && (half & 1U) != 0)) {
      ++half;
    }
    return static_cast<std::uint16_t>(sign | half);
  }
  std::uint32_t half =
      (static_cast<std::uint32_t>(exponent) << 10U) | (mantissa >> 13U);
  const std::uint32_t remainder = mantissa & 0x1FFFU;
  if (remainder > 0x1000U || (remainder == 0x1000U && (half & 1U) != 0)) {
    ++half;  // a carry into the exponent correctly yields the next binade/inf
  }
  return static_cast<std::uint16_t>(sign | half);
}

// ---- parallel row loop ----

unsigned ResolveThreads(unsigned requested) {
  if (requested != 0) {
    return requested;
  }
  if (const char* env = std::getenv("GUFO_LAGUNA_ORACLE_THREADS");
      env != nullptr && *env != '\0') {
    const long value = std::strtol(env, nullptr, 10);
    if (value > 0) {
      return static_cast<unsigned>(value);
    }
  }
  return std::max(1U, std::thread::hardware_concurrency());
}

template<class Fn>
void ParallelChunks(std::size_t count, std::size_t chunk, unsigned threads,
                    const Fn& fn) {
  const std::size_t chunks = (count + chunk - 1) / chunk;
  const unsigned workers =
      static_cast<unsigned>(std::min<std::size_t>(threads, chunks));
  if (workers <= 1) {
    for (std::size_t c = 0; c < chunks; ++c) {
      fn(c * chunk, std::min(count, (c + 1) * chunk));
    }
    return;
  }
  std::atomic<std::size_t> next{0};
  const auto worker = [&] {
    for (std::size_t c = next.fetch_add(1); c < chunks; c = next.fetch_add(1)) {
      fn(c * chunk, std::min(count, (c + 1) * chunk));
    }
  };
  std::vector<std::thread> pool;
  pool.reserve(workers - 1);
  for (unsigned i = 1; i < workers; ++i) {
    pool.emplace_back(worker);
  }
  worker();
  for (auto& thread : pool) {
    thread.join();
  }
}

/// Float partial sums per 32 elements (vectorizable), accumulated in double.
double DotRow(const float* w, const float* x, std::size_t n) {
  double total = 0.0;
  std::size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    float lane[8] = {};
    for (std::size_t j = 0; j < 32; j += 8) {
      for (std::size_t l = 0; l < 8; ++l) {
        lane[l] += w[i + j + l] * x[i + j + l];
      }
    }
    float part = 0.0F;
    for (const float v : lane) {
      part += v;
    }
    total += part;
  }
  float tail = 0.0F;
  for (; i < n; ++i) {
    tail += w[i] * x[i];
  }
  return total + tail;
}

/// `value` moved by `steps` fp16 steps (same sign: toward larger magnitude).
float NudgeHalf(float value, int steps) {
  std::uint16_t bits = FloatToHalfBits(value);
  bits = static_cast<std::uint16_t>(bits + steps);
  return quant::Fp16ToFloat(bits);
}

/// The ggml mul_mat_q tile arithmetic for Q4_K/Q5_K weights, which is not
/// "round the inputs, then multiply dequantized weights": the tile loader
/// stores the per-sub-block scale and min as fp16 products (d * sc and
/// -dmin * m, `x_dm = dm * make_half2(sc, m)`), the activation block carries
/// half2(d, d * sum(q)) (DS4 layout), and a sub-block contributes
///   x_dm.x * d8 * sum(qw * qx)  +  x_dm.y * half(d * sum(qx)).
/// Sub-blocks are 32 elements; the integer dot is exact.
void MmqKQuantMatMul(const TensorRef& w, std::uint64_t expert, const float* x,
                     std::size_t n, float* y, const Options& options) {
  const std::size_t cols = w.cols;
  const std::size_t rows = w.rows;
  if (cols % 256 != 0) {
    throw std::invalid_argument("laguna oracle: activation row not aligned");
  }
  const bool q5 = w.type == GgmlType::kQ5_K;
  const std::size_t groups = cols / 32;
  const ScaleNudge& nudge = options.nudge;
  std::vector<std::int8_t> q(n * cols);
  std::vector<float> d8(n * groups);
  std::vector<float> sb(n * groups);
  for (std::size_t t = 0; t < n; ++t) {
    for (std::size_t g = 0; g < groups; ++g) {
      const float* row = x + t * cols + g * 32;
      float amax = 0.0F;
      for (std::size_t j = 0; j < 32; ++j) {
        amax = std::max(amax, std::fabs(row[j]));
      }
      const float d_inv = 127.0F / amax;
      const float d = 1.0F / d_inv;
      int sum = 0;
      for (std::size_t j = 0; j < 32; ++j) {
        const int v =
            amax == 0.0F ? 0 : static_cast<int>(std::round(row[j] * d_inv));
        q[t * cols + g * 32 + j] = static_cast<std::int8_t>(v);
        sum += v;
      }
      d8[t * groups + g] = RoundToF16(amax == 0.0F ? 0.0F : d);
      sb[t * groups + g] =
          RoundToF16(amax == 0.0F ? 0.0F : d * static_cast<float>(sum));
    }
  }
  if (nudge.group >= 0 && static_cast<std::size_t>(nudge.group) < groups) {
    float& v = nudge.kind == 0 ? d8[nudge.group] : sb[nudge.group];
    v = NudgeHalf(v, nudge.steps);
  }
  const std::size_t block_bytes = q5 ? 176 : 144;
  const std::size_t row_bytes = w.RowBytes();
  const std::uint8_t* base = w.Expert(expert);
  ParallelChunks(
      rows, 8, ResolveThreads(options.threads),
      [&](std::size_t begin, std::size_t end) {
        for (std::size_t r = begin; r < end; ++r) {
          for (std::size_t t = 0; t < n; ++t) {
            double acc = 0.0;
            for (std::size_t b = 0; b < cols / 256; ++b) {
              const std::uint8_t* blk = base + r * row_bytes + b * block_bytes;
              std::uint16_t dh = 0;
              std::uint16_t dminh = 0;
              std::memcpy(&dh, blk, 2);
              std::memcpy(&dminh, blk + 2, 2);
              const float dw = quant::Fp16ToFloat(dh);
              const float dmin = quant::Fp16ToFloat(dminh);
              const std::uint8_t* scales = blk + 4;
              const std::uint8_t* qh = q5 ? blk + 16 : nullptr;
              const std::uint8_t* qs = blk + (q5 ? 48 : 16);
              for (int j = 0; j < 8; ++j) {
                int sc = 0;
                int m = 0;
                if (j < 4) {
                  sc = scales[j] & 63;
                  m = scales[j + 4] & 63;
                } else {
                  sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
                  m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
                }
                const float dm_x = RoundToF16(dw * static_cast<float>(sc));
                const float dm_y = -RoundToF16(dmin * static_cast<float>(m));
                long dot = 0;
                const std::int8_t* qx = q.data() + t * cols + b * 256 + j * 32;
                for (int l = 0; l < 32; ++l) {
                  const std::uint8_t byte = qs[32 * (j / 2) + l];
                  int qw = (j & 1) != 0 ? (byte >> 4) : (byte & 0xF);
                  if (q5 && ((qh[l] >> j) & 1) != 0) {
                    qw += 16;
                  }
                  dot += static_cast<long>(qw) * qx[l];
                }
                const std::size_t g = b * 8 + j;
                acc += static_cast<double>(dm_x) * d8[t * groups + g] *
                           static_cast<double>(dot) +
                       static_cast<double>(dm_y) * sb[t * groups + g];
              }
            }
            y[t * rows + r] = static_cast<float>(acc);
          }
        }
      });
}

/// y[t][r] = sum_c W[expert][r][c] * x[t][c], with the inputs rounded per the
/// activation mode. W rows are dequantized one at a time.
void MatMul(const TensorRef& w, std::uint64_t expert, const float* x,
            std::size_t n, float* y, const Options& options) {
  const std::size_t cols = w.cols;
  const std::size_t rows = w.rows;
  if (w.empty() || expert >= w.experts) {
    throw std::invalid_argument("laguna oracle: bad matmul operand");
  }
  if (options.activation_quant == ActivationQuant::kGpuMmq &&
      (w.type == GgmlType::kQ4_K || w.type == GgmlType::kQ5_K)) {
    MmqKQuantMatMul(w, expert, x, n, y, options);
    return;
  }
  std::vector<float> xr(x, x + n * cols);
  RoundActivations(w.type, options.activation_quant, xr.data(), n, cols,
                   &options.nudge);
  const std::uint8_t* base = w.Expert(expert);
  const std::size_t row_bytes = w.RowBytes();
  ParallelChunks(rows, 8, ResolveThreads(options.threads),
                 [&](std::size_t begin, std::size_t end) {
                   std::vector<float> row(cols);
                   for (std::size_t r = begin; r < end; ++r) {
                     DequantizeRow(w.type, base + r * row_bytes, row.data(),
                                   cols);
                     for (std::size_t t = 0; t < n; ++t) {
                       y[t * rows + r] = static_cast<float>(
                           DotRow(row.data(), xr.data() + t * cols, cols));
                     }
                   }
                 });
}

std::vector<float> LoadF32(const TensorRef& t) {
  std::vector<float> out(t.ElementCount());
  DequantizeRow(t.type, t.data, out.data(), out.size());
  return out;
}

/// ggml_compute_forward_rms_norm_f32 then the elementwise weight multiply.
void RmsNormRow(const float* x, const float* weight, std::size_t n, float eps,
                float* y) {
  double sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    sum += static_cast<double>(x[i]) * x[i];
  }
  const float mean = static_cast<float>(sum / static_cast<double>(n));
  const float scale = 1.0F / std::sqrt(mean + eps);
  for (std::size_t i = 0; i < n; ++i) {
    y[i] = (x[i] * scale) * weight[i];
  }
}

float Silu(float x) {
  return x / (1.0F + std::exp(-x));
}
float Softplus(float x) {
  return x > 20.0F ? x : std::log(1.0F + std::exp(x));
}
float Sigmoid(float x) {
  return 1.0F / (1.0F + std::exp(-x));
}

}  // namespace

void MatMulRows(const TensorRef& w, std::uint64_t expert, const float* x,
                std::size_t n, float* y, const Options& options) {
  MatMul(w, expert, x, n, y, options);
}

void RmsNormVector(const float* x, const float* weight, std::size_t n,
                   float eps, float* y) {
  RmsNormRow(x, weight, n, eps, y);
}

std::vector<float> DequantizeTensor(const TensorRef& t) {
  return LoadF32(t);
}

float SoftplusF32(float x) {
  return Softplus(x);
}

float RoundToF16(float value) noexcept {
  return quant::Fp16ToFloat(FloatToHalfBits(value));
}

void DequantizeRow(GgmlType type, const void* src, float* dst,
                   std::size_t cols) {
  switch (type) {
    case GgmlType::kIQ4_XS:
      quant::DequantizeIQ4_XS(src, dst, cols);
      return;
    case GgmlType::kF32:
    case GgmlType::kF16:
    case GgmlType::kBF16:
    case GgmlType::kQ4_K:
    case GgmlType::kQ5_K:
    case GgmlType::kQ6_K:
    case GgmlType::kQ8_0:
      quant::Dequantize(type, src, dst, cols);
      return;
    default:
      throw std::invalid_argument("laguna oracle: unsupported ggml type");
  }
}

void RoundActivations(GgmlType weight_type, ActivationQuant mode, float* x,
                      std::size_t rows, std::size_t cols,
                      const ScaleNudge* nudge) {
  enum class Kind {
    kNone,
    kQ8_0,
    kQ8_K,
    kF16,
    kBF16,
    kMmqHalfScale,
    kMmqFloatScale
  } kind = Kind::kNone;
  if (mode == ActivationQuant::kCpu) {
    switch (weight_type) {
      case GgmlType::kQ8_0:
      case GgmlType::kIQ4_NL:
        kind = Kind::kQ8_0;
        break;
      case GgmlType::kQ3_K:
      case GgmlType::kQ4_K:
      case GgmlType::kQ5_K:
      case GgmlType::kQ6_K:
      case GgmlType::kIQ4_XS:
        kind = Kind::kQ8_K;
        break;
      case GgmlType::kF16:
        kind = Kind::kF16;
        break;
      case GgmlType::kBF16:
        kind = Kind::kBF16;
        break;
      default:
        break;
    }
  } else if (mode == ActivationQuant::kGpuQ8_1) {
    if (quant::QuantizedBlockElements(weight_type) != 0) {
      kind = Kind::kQ8_0;  // Q8_1 rounds exactly like Q8_0
    }
  } else if (mode == ActivationQuant::kGpuMmq) {
    switch (weight_type) {
      case GgmlType::kQ4_K:
      case GgmlType::kQ5_K:
        kind = Kind::kMmqHalfScale;
        break;
      default:
        if (quant::QuantizedBlockElements(weight_type) != 0) {
          kind = Kind::kMmqFloatScale;
        }
        break;
    }
  }
  if (kind == Kind::kNone) {
    return;
  }
  const std::size_t block = kind == Kind::kQ8_K ? 256 : 32;
  if ((kind == Kind::kQ8_0 || kind == Kind::kQ8_K ||
       kind == Kind::kMmqHalfScale || kind == Kind::kMmqFloatScale) &&
      cols % block != 0) {
    throw std::invalid_argument(
        "laguna oracle: activation row not block aligned");
  }
  for (std::size_t r = 0; r < rows; ++r) {
    float* row = x + r * cols;
    switch (kind) {
      case Kind::kF16:
        for (std::size_t i = 0; i < cols; ++i) {
          row[i] = RoundToF16(row[i]);
        }
        break;
      case Kind::kBF16:
        for (std::size_t i = 0; i < cols; ++i) {
          row[i] = RoundToBf16(row[i]);
        }
        break;
      case Kind::kQ8_0:
        // quantize_row_q8_0_ref: d = amax/127, q = roundf(x / d), the scale
        // is stored as fp16 and that stored value is what the dot uses.
        for (std::size_t b = 0; b < cols; b += 32) {
          float amax = 0.0F;
          for (std::size_t j = 0; j < 32; ++j) {
            amax = std::max(amax, std::fabs(row[b + j]));
          }
          const float d = amax / 127.0F;
          const float id = d != 0.0F ? 1.0F / d : 0.0F;
          float stored = RoundToF16(d);
          if (nudge != nullptr && r == 0 && nudge->kind == 0 &&
              nudge->group == static_cast<int>(b / 32)) {
            stored = NudgeHalf(stored, nudge->steps);
          }
          for (std::size_t j = 0; j < 32; ++j) {
            row[b + j] = std::round(row[b + j] * id) * stored;
          }
        }
        break;
      case Kind::kMmqHalfScale:
      case Kind::kMmqFloatScale:
        // quantize_mmq_q8_1: d_inv = 127 / amax, q = roundf(x * d_inv), the
        // scale d = 1 / d_inv is stored as fp16 (DS4 layout) or as float (D4).
        for (std::size_t b = 0; b < cols; b += 32) {
          float amax = 0.0F;
          for (std::size_t j = 0; j < 32; ++j) {
            amax = std::max(amax, std::fabs(row[b + j]));
          }
          const float d_inv = 127.0F / amax;
          const float d = 1.0F / d_inv;
          const float stored = kind == Kind::kMmqHalfScale ? RoundToF16(d) : d;
          for (std::size_t j = 0; j < 32; ++j) {
            row[b + j] =
                amax == 0.0F ? 0.0F : std::round(row[b + j] * d_inv) * stored;
          }
        }
        break;
      case Kind::kQ8_K:
        // quantize_row_q8_K_ref: scale from the signed value of largest
        // magnitude, q = nearest_int(-127/max * x) (ties to even), float d.
        for (std::size_t b = 0; b < cols; b += 256) {
          float max = 0.0F;
          float amax = 0.0F;
          for (std::size_t j = 0; j < 256; ++j) {
            const float ax = std::fabs(row[b + j]);
            if (ax > amax) {
              amax = ax;
              max = row[b + j];
            }
          }
          if (amax == 0.0F) {
            std::fill(row + b, row + b + 256, 0.0F);
            continue;
          }
          const float iscale = -127.0F / max;
          const float d = 1.0F / iscale;
          for (std::size_t j = 0; j < 256; ++j) {
            const float q =
                std::min(127.0F, std::nearbyint(iscale * row[b + j]));
            row[b + j] = q * d;
          }
        }
        break;
      case Kind::kNone:
        break;
    }
  }
}

RopeParams RopeFor(const Config& config, std::uint32_t il) {
  RopeParams rope;
  rope.n_rot = config.rotary_dim(il);
  rope.freq_base = config.rope_theta(il);
  if (config.is_full_attention(il)) {
    rope.attn_factor = config.rope_attn_factor;
    rope.n_ctx_orig = config.yarn.original_context;
    if (config.uses_yarn(il)) {
      rope.freq_scale = 1.0F / config.yarn.factor;
      rope.ext_factor = 1.0F;
      rope.beta_fast = config.yarn.beta_fast;
      rope.beta_slow = config.yarn.beta_slow;
    }
  } else {
    rope.n_ctx_orig = config.context_length;
    rope.beta_fast = 0.0F;
    rope.beta_slow = 0.0F;
  }
  return rope;
}

RopeSetup MakeRopeSetup(const RopeParams& rope) {
  const std::uint32_t n_rot = rope.n_rot;
  RopeSetup setup;
  // ggml_rope_yarn_corr_dims
  if (rope.ext_factor != 0.0F) {
    const auto corr_dim = [&](float n_rot_beta) {
      return static_cast<float>(n_rot) *
             std::log(static_cast<float>(rope.n_ctx_orig) /
                      (n_rot_beta * 2.0F * kPi)) /
             (2.0F * std::log(rope.freq_base));
    };
    const float start = std::floor(corr_dim(rope.beta_fast));
    const float end = std::ceil(corr_dim(rope.beta_slow));
    setup.corr[0] = std::max(0.0F, start);
    setup.corr[1] = std::min(static_cast<float>(n_rot) - 1.0F, end);
  }
  setup.theta_scale =
      std::pow(rope.freq_base, -2.0F / static_cast<float>(n_rot));
  setup.mscale = rope.attn_factor;
  if (rope.ext_factor != 0.0F) {
    setup.mscale *= 1.0F + 0.1F * std::log(1.0F / rope.freq_scale);
  }
  return setup;
}

void ApplyRope(float* x, std::span<const std::int32_t> positions,
               std::uint32_t heads, std::uint32_t head_dim,
               const RopeParams& rope) {
  const std::uint32_t n_rot = rope.n_rot;
  if (n_rot > head_dim || n_rot % 2 != 0) {
    throw std::invalid_argument("laguna oracle: bad rotary dimension");
  }
  const RopeSetup setup = MakeRopeSetup(rope);
  const float* corr = setup.corr;
  const float theta_scale = setup.theta_scale;
  std::vector<float> cache(n_rot);
  for (std::size_t t = 0; t < positions.size(); ++t) {
    // ggml_rope_cache_init / rope_yarn, in the same float arithmetic.
    float theta = static_cast<float>(positions[t]);
    for (std::uint32_t i0 = 0; i0 < n_rot; i0 += 2) {
      const float theta_interp = rope.freq_scale * theta;
      float rotated = theta_interp;
      const float mscale = setup.mscale;
      if (rope.ext_factor != 0.0F) {
        const float y = (static_cast<float>(i0 / 2) - corr[0]) /
                        std::max(0.001F, corr[1] - corr[0]);
        const float ramp = 1.0F - std::min(1.0F, std::max(0.0F, y));
        const float ramp_mix = ramp * rope.ext_factor;
        rotated = theta_interp * (1.0F - ramp_mix) + theta * ramp_mix;
      }
      cache[i0] = std::cos(rotated) * mscale;
      cache[i0 + 1] = std::sin(rotated) * mscale;
      theta *= theta_scale;
    }
    for (std::uint32_t h = 0; h < heads; ++h) {
      float* head = x + (t * heads + h) * head_dim;
      for (std::uint32_t i0 = 0; i0 < n_rot; i0 += 2) {
        const std::uint32_t ic = i0 / 2;
        const float x0 = head[ic];
        const float x1 = head[ic + n_rot / 2];
        head[ic] = x0 * cache[i0] - x1 * cache[i0 + 1];
        head[ic + n_rot / 2] = x0 * cache[i0 + 1] + x1 * cache[i0];
      }
    }
  }
}

void GqaAttentionAt(const float* q, std::span<const std::int32_t> q_positions,
                    const float* k, const float* v,
                    std::span<const std::int32_t> k_positions,
                    std::uint32_t heads, std::uint32_t kv_heads,
                    std::uint32_t head_dim, std::uint32_t window, float scale,
                    float* out) {
  if (kv_heads == 0 || heads % kv_heads != 0) {
    throw std::invalid_argument(
        "laguna oracle: heads not a multiple of kv heads");
  }
  const std::uint32_t group = heads / kv_heads;
  const std::size_t q_row = static_cast<std::size_t>(heads) * head_dim;
  const std::size_t kv_row = static_cast<std::size_t>(kv_heads) * head_dim;
  std::vector<double> score;
  std::vector<std::size_t> visible;
  for (std::size_t p = 0; p < q_positions.size(); ++p) {
    // Visible keys: k_pos <= q_pos and q_pos - k_pos < window
    // (llama.cpp SWA_TYPE_STANDARD), in key order.
    const std::int64_t qp = q_positions[p];
    visible.clear();
    for (std::size_t j = 0; j < k_positions.size(); ++j) {
      const std::int64_t kp = k_positions[j];
      if (kp <= qp && (window == 0 || qp - kp < window)) {
        visible.push_back(j);
      }
    }
    score.assign(visible.size(), 0.0);
    for (std::uint32_t h = 0; h < heads; ++h) {
      float* oh = out + p * q_row + static_cast<std::size_t>(h) * head_dim;
      if (visible.empty()) {
        std::fill(oh, oh + head_dim, 0.0F);
        continue;
      }
      const float* qh = q + p * q_row + static_cast<std::size_t>(h) * head_dim;
      const std::size_t kvh = h / group;
      double top = -std::numeric_limits<double>::infinity();
      for (std::size_t i = 0; i < visible.size(); ++i) {
        const float* kj = k + visible[i] * kv_row + kvh * head_dim;
        double dot = 0.0;
        for (std::uint32_t d = 0; d < head_dim; ++d) {
          dot += static_cast<double>(qh[d]) * kj[d];
        }
        score[i] = dot * scale;
        top = std::max(top, score[i]);
      }
      double denom = 0.0;
      for (double& s : score) {
        s = std::exp(s - top);
        denom += s;
      }
      for (std::uint32_t d = 0; d < head_dim; ++d) {
        double acc = 0.0;
        for (std::size_t i = 0; i < visible.size(); ++i) {
          acc += score[i] * v[visible[i] * kv_row + kvh * head_dim + d];
        }
        oh[d] = static_cast<float>(acc / denom);
      }
    }
  }
}

void GqaAttention(const float* q, const float* k, const float* v,
                  std::size_t tokens, std::uint32_t heads,
                  std::uint32_t kv_heads, std::uint32_t head_dim,
                  std::uint32_t window, float scale, float* out) {
  std::vector<std::int32_t> positions(tokens);
  std::iota(positions.begin(), positions.end(), 0);
  GqaAttentionAt(q, positions, k, v, positions, heads, kv_heads, head_dim,
                 window, scale, out);
}

Routing Route(const float* logits, const float* bias, std::uint32_t experts,
              std::uint32_t used, bool normalize, float scale,
              RouterDenominator denominator) {
  if (experts == 0 || used == 0 || used > experts) {
    throw std::invalid_argument("laguna oracle: invalid router expert count");
  }
  std::vector<float> probs(experts);
  std::vector<float> choice(experts);
  for (std::uint32_t e = 0; e < experts; ++e) {
    probs[e] = Sigmoid(logits[e]);
    choice[e] = probs[e] + (bias != nullptr ? bias[e] : 0.0F);
    if (!std::isfinite(choice[e])) {
      throw std::invalid_argument("laguna oracle: nonfinite router score");
    }
  }
  std::vector<std::uint32_t> order(experts);
  std::iota(order.begin(), order.end(), 0U);
  std::partial_sort(order.begin(), order.begin() + used, order.end(),
                    [&](std::uint32_t a, std::uint32_t b) {
                      return choice[a] == choice[b] ? a < b
                                                    : choice[a] > choice[b];
                    });
  Routing routing;
  routing.ids.assign(order.begin(), order.begin() + used);
  for (const auto e : routing.ids) {
    routing.probs.push_back(probs[e]);
    routing.sum += probs[e];
  }
  routing.denominator = routing.sum;
  if (normalize) {
    routing.denominator = denominator == RouterDenominator::kLlamaClamp
                              ? std::max(routing.sum, kRouterSumClamp)
                              : routing.sum;
  }
  routing.weights.resize(used);
  for (std::uint32_t i = 0; i < used; ++i) {
    float w = routing.probs[i];
    if (normalize) {
      w /= routing.denominator;
    }
    if (scale != 0.0F && scale != 1.0F) {
      w *= scale;
    }
    routing.weights[i] = w;
  }
  return routing;
}

std::vector<float> FfnForward(const Config& c, const LayerWeights& lw,
                              const float* fn_ptr, std::size_t T,
                              const Options& options, const TapFn& tap,
                              int layer) {
  const std::size_t H = c.hidden_size;
  const auto emit = [&](const char* name, TapType type,
                        std::array<std::uint64_t, 3> ne, const void* data) {
    if (!tap) {
      return;
    }
    const std::string full =
        layer >= 0 ? std::string(name) + "-" + std::to_string(layer)
                   : std::string(name);
    tap(Tap{full, type, ne, data});
  };
  const auto emit_f32 = [&](const char* name, int,
                            std::array<std::uint64_t, 3> ne,
                            const std::vector<float>& data) {
    emit(name, TapType::kF32, ne, data.data());
  };
  const std::vector<float> fn(fn_ptr, fn_ptr + T * H);
  std::vector<float> ffn_out(T * H);
  if (!lw.moe) {
    const std::size_t ff = lw.ffn_gate.rows;
    std::vector<float> up(T * ff);
    std::vector<float> gt(T * ff);
    MatMul(lw.ffn_up, 0, fn.data(), T, up.data(), options);
    MatMul(lw.ffn_gate, 0, fn.data(), T, gt.data(), options);
    emit_f32("ffn_up", layer, {ff, T, 1}, up);
    emit_f32("ffn_gate", layer, {ff, T, 1}, gt);
    std::vector<float> act(T * ff);
    for (std::size_t i = 0; i < T * ff; ++i) {
      act[i] = Silu(gt[i]) * up[i];
    }
    emit_f32("ffn_swiglu", layer, {ff, T, 1}, act);
    MatMul(lw.ffn_down, 0, act.data(), T, ffn_out.data(), options);
    emit_f32("ffn_down", layer, {H, T, 1}, ffn_out);
  } else {
    const std::uint32_t E = c.num_experts;
    const std::uint32_t used = c.num_experts_used;
    const std::size_t eff = c.expert_ff;
    std::vector<float> logits(T * E);
    MatMul(lw.ffn_gate_inp, 0, fn.data(), T, logits.data(), options);
    emit_f32("ffn_moe_logits", layer, {E, T, 1}, logits);
    const std::vector<float> bias = LoadF32(lw.ffn_exp_probs_b);

    std::vector<float> probs(T * E);
    std::vector<float> biased(T * E);
    std::vector<std::int32_t> topk(T * used);
    std::vector<float> sel_probs(T * used);
    std::vector<float> sums(T);
    std::vector<float> clamped(T);
    std::vector<float> normed(T * used);
    std::vector<float> scaled(T * used);
    std::vector<std::vector<std::pair<std::size_t, std::uint32_t>>> by_expert(
        E);
    for (std::size_t t = 0; t < T; ++t) {
      for (std::uint32_t e = 0; e < E; ++e) {
        probs[t * E + e] = Sigmoid(logits[t * E + e]);
        biased[t * E + e] = probs[t * E + e] + bias[e];
      }
      const Routing r = Route(logits.data() + t * E, bias.data(), E, used,
                              c.expert_weights_norm, c.expert_weights_scale,
                              options.router_denominator);
      sums[t] = r.sum;
      clamped[t] = r.denominator;
      for (std::uint32_t s = 0; s < used; ++s) {
        topk[t * used + s] = static_cast<std::int32_t>(r.ids[s]);
        sel_probs[t * used + s] = r.probs[s];
        normed[t * used + s] =
            c.expert_weights_norm ? r.probs[s] / r.denominator : r.probs[s];
        scaled[t * used + s] = r.weights[s];
        by_expert[r.ids[s]].emplace_back(t, s);
      }
    }
    emit_f32("ffn_moe_probs", layer, {E, T, 1}, probs);
    emit_f32("ffn_moe_probs_biased", layer, {E, T, 1}, biased);
    emit("ffn_moe_topk", TapType::kI32, {used, T, 1}, topk.data());
    emit_f32("ffn_moe_weights", layer, {1, used, T}, sel_probs);
    if (c.expert_weights_norm) {
      emit_f32("ffn_moe_weights_sum", layer, {1, T, 1}, sums);
      emit_f32("ffn_moe_weights_sum_clamped", layer, {1, T, 1}, clamped);
      emit_f32("ffn_moe_weights_norm", layer, {used, T, 1}, normed);
    }
    if (c.expert_weights_scale != 0.0F && c.expert_weights_scale != 1.0F) {
      emit_f32("ffn_moe_weights_scaled", layer, {1, used, T}, scaled);
    }

    // Routed experts, grouped by expert so each is dequantized once.
    std::vector<float> e_gate(T * used * eff);
    std::vector<float> e_up(T * used * eff);
    std::vector<float> e_act(T * used * eff);
    std::vector<float> e_down(T * used * H);
    for (std::uint32_t e = 0; e < E; ++e) {
      const auto& list = by_expert[e];
      if (list.empty()) {
        continue;
      }
      const std::size_t m = list.size();
      std::vector<float> xin(m * H);
      for (std::size_t i = 0; i < m; ++i) {
        std::copy_n(fn.data() + list[i].first * H, H, xin.data() + i * H);
      }
      std::vector<float> g(m * eff);
      std::vector<float> u(m * eff);
      MatMul(lw.ffn_gate_exps, e, xin.data(), m, g.data(), options);
      MatMul(lw.ffn_up_exps, e, xin.data(), m, u.data(), options);
      std::vector<float> a(m * eff);
      for (std::size_t i = 0; i < m * eff; ++i) {
        a[i] = Silu(g[i]) * u[i];
      }
      std::vector<float> d(m * H);
      MatMul(lw.ffn_down_exps, e, a.data(), m, d.data(), options);
      for (std::size_t i = 0; i < m; ++i) {
        const std::size_t slot = list[i].first * used + list[i].second;
        std::copy_n(g.data() + i * eff, eff, e_gate.data() + slot * eff);
        std::copy_n(u.data() + i * eff, eff, e_up.data() + slot * eff);
        std::copy_n(a.data() + i * eff, eff, e_act.data() + slot * eff);
        std::copy_n(d.data() + i * H, H, e_down.data() + slot * H);
      }
    }
    emit_f32("ffn_moe_gate", layer, {eff, used, T}, e_gate);
    emit_f32("ffn_moe_up", layer, {eff, used, T}, e_up);
    emit_f32("ffn_moe_swiglu", layer, {eff, used, T}, e_act);
    emit_f32("ffn_moe_down", layer, {H, used, T}, e_down);

    std::vector<float> weighted(T * used * H);
    std::vector<float> moe_out(T * H);
    for (std::size_t t = 0; t < T; ++t) {
      for (std::uint32_t s = 0; s < used; ++s) {
        const std::size_t slot = t * used + s;
        for (std::size_t d = 0; d < H; ++d) {
          weighted[slot * H + d] = e_down[slot * H + d] * scaled[slot];
        }
      }
      // ggml sums the slots left to right in float.
      for (std::size_t d = 0; d < H; ++d) {
        float acc = weighted[(t * used) * H + d];
        for (std::uint32_t s = 1; s < used; ++s) {
          acc += weighted[(t * used + s) * H + d];
        }
        moe_out[t * H + d] = acc;
      }
    }
    emit_f32("ffn_moe_weighted", layer, {H, used, T}, weighted);
    emit_f32("ffn_moe_out", layer, {H, T, 1}, moe_out);

    // Shared expert: plain SwiGLU, no gate.
    const std::size_t sff = lw.ffn_gate_shexp.rows;
    std::vector<float> su(T * sff);
    std::vector<float> sg(T * sff);
    MatMul(lw.ffn_up_shexp, 0, fn.data(), T, su.data(), options);
    MatMul(lw.ffn_gate_shexp, 0, fn.data(), T, sg.data(), options);
    std::vector<float> sa(T * sff);
    for (std::size_t i = 0; i < T * sff; ++i) {
      sa[i] = Silu(sg[i]) * su[i];
    }
    std::vector<float> shexp(T * H);
    MatMul(lw.ffn_down_shexp, 0, sa.data(), T, shexp.data(), options);
    emit_f32("ffn_shexp", layer, {H, T, 1}, shexp);
    for (std::size_t i = 0; i < T * H; ++i) {
      ffn_out[i] = moe_out[i] + shexp[i];
    }
  }
  return ffn_out;
}

// ---- TapWriter ----

TapWriter::TapWriter(std::filesystem::path directory,
                     std::vector<std::string> filter)
    : directory_(std::move(directory)), filter_(std::move(filter)) {
  std::error_code ec;
  std::filesystem::create_directories(directory_, ec);
  if (ec) {
    error_ = "cannot create " + directory_.string() + ": " + ec.message();
  }
}

void TapWriter::operator()(const Tap& tap) {
  if (!filter_.empty()) {
    const bool keep = std::any_of(
        filter_.begin(), filter_.end(), [&](const std::string& needle) {
          return tap.name.find(needle) != std::string_view::npos;
        });
    if (!keep) {
      return;
    }
  }
  Entry entry{std::string(tap.name), "", tap.type, tap.ne};
  entry.file = entry.name + (tap.type == TapType::kF32 ? ".f32" : ".i32");
  std::ofstream file(directory_ / entry.file, std::ios::binary);
  file.write(static_cast<const char*>(tap.data),
             static_cast<std::streamsize>(tap.Count() * 4));
  if (!file && error_.empty()) {
    error_ = "write failed: " + entry.file;
  }
  entries_.push_back(std::move(entry));
}

bool TapWriter::Finish(std::string* error) {
  std::ofstream index(directory_ / "index.json");
  index << "{\n  \"layout\": \"raw little-endian, ggml ne order (ne[0] "
           "fastest, token axis last)\",\n  \"taps\": [\n";
  for (std::size_t i = 0; i < entries_.size(); ++i) {
    const Entry& e = entries_[i];
    index << "    {\"name\": \"" << e.name << "\", \"file\": \"" << e.file
          << "\", \"dtype\": \"" << (e.type == TapType::kF32 ? "f32" : "i32")
          << "\", \"ne\": [" << e.ne[0] << ", " << e.ne[1] << ", " << e.ne[2]
          << "]}" << (i + 1 < entries_.size() ? "," : "") << "\n";
  }
  index << "  ]\n}\n";
  index.flush();
  if (!index && error_.empty()) {
    error_ = "write failed: index.json";
  }
  if (!error_.empty() && error != nullptr) {
    *error = error_;
  }
  return error_.empty();
}

// ---- Model ----

std::vector<float> AttnForward(const Config& c, const LayerWeights& lw,
                               std::uint32_t il, const float* n_ptr,
                               std::size_t T, const Options& options,
                               const TapFn& tap, int layer) {
  const std::size_t H = c.hidden_size;
  const std::uint32_t hd = c.head_dim;
  const std::uint32_t kvh = c.num_kv_heads;
  const std::size_t kv_dim = c.kv_dim();
  const std::uint32_t heads = c.n_head(il);
  const std::size_t q_dim = c.q_dim(il);
  const float kq_scale = 1.0F / std::sqrt(static_cast<float>(hd));
  std::vector<std::int32_t> positions(T);
  std::iota(positions.begin(), positions.end(), 0);
  const auto emit_f32 = [&](const char* name, int,
                            std::array<std::uint64_t, 3> ne,
                            const std::vector<float>& data) {
    if (!tap) {
      return;
    }
    const std::string full =
        layer >= 0 ? std::string(name) + "-" + std::to_string(layer)
                   : std::string(name);
    tap(Tap{full, TapType::kF32, ne, data.data()});
  };
  const std::vector<float> n(n_ptr, n_ptr + T * H);
  std::vector<float> q(T * q_dim);
  std::vector<float> k(T * kv_dim);
  std::vector<float> v(T * kv_dim);
  MatMul(lw.attn_q, 0, n.data(), T, q.data(), options);
  MatMul(lw.attn_k, 0, n.data(), T, k.data(), options);
  MatMul(lw.attn_v, 0, n.data(), T, v.data(), options);
  emit_f32("Qcur", layer, {q_dim, T, 1}, q);
  emit_f32("Kcur", layer, {kv_dim, T, 1}, k);
  emit_f32("Vcur", layer, {kv_dim, T, 1}, v);

  // Output gate input: the same attn_norm output as q/k/v.
  std::vector<float> gate(T * heads);
  MatMul(lw.attn_gate, 0, n.data(), T, gate.data(), options);
  emit_f32("attn_gate_proj", layer, {heads, T, 1}, gate);

  const std::vector<float> q_norm_w = LoadF32(lw.attn_q_norm);
  const std::vector<float> k_norm_w = LoadF32(lw.attn_k_norm);
  for (std::size_t i = 0; i < T * heads; ++i) {
    RmsNormRow(q.data() + i * hd, q_norm_w.data(), hd, c.rms_eps,
               q.data() + i * hd);
  }
  for (std::size_t i = 0; i < T * kvh; ++i) {
    RmsNormRow(k.data() + i * hd, k_norm_w.data(), hd, c.rms_eps,
               k.data() + i * hd);
  }
  emit_f32("Qcur_normed", layer, {hd, heads, T}, q);
  emit_f32("Kcur_normed", layer, {hd, kvh, T}, k);

  const RopeParams rope = RopeFor(c, il);
  ApplyRope(q.data(), positions, heads, hd, rope);
  ApplyRope(k.data(), positions, kvh, hd, rope);
  emit_f32("Qcur_rope", layer, {hd, heads, T}, q);
  emit_f32("Kcur_rope", layer, {hd, kvh, T}, k);

  if (options.round_kv_f16) {
    for (float& value : k) {
      value = RoundToF16(value);
    }
    for (float& value : v) {
      value = RoundToF16(value);
    }
  }

  std::vector<float> attn(T * q_dim);
  GqaAttention(q.data(), k.data(), v.data(), T, heads, kvh, hd,
               c.is_full_attention(il) ? 0U : c.sliding_window, kq_scale,
               attn.data());
  emit_f32("attn_out", layer, {q_dim, T, 1}, attn);

  for (float& g : gate) {
    g = Softplus(g);
  }
  emit_f32("attn_gate_softplus", layer, {heads, T, 1}, gate);
  for (std::size_t t = 0; t < T; ++t) {
    for (std::uint32_t head = 0; head < heads; ++head) {
      float* out =
          attn.data() + t * q_dim + static_cast<std::size_t>(head) * hd;
      for (std::uint32_t d = 0; d < hd; ++d) {
        out[d] *= gate[t * heads + head];
      }
    }
  }
  emit_f32("attn_gated", layer, {q_dim, T, 1}, attn);

  std::vector<float> o(T * H);
  MatMul(lw.attn_output, 0, attn.data(), T, o.data(), options);
  emit_f32("attn_o_proj", layer, {H, T, 1}, o);
  return o;
}

Model::Model(const ModelWeights& weights, Options options)
    : weights_(weights), options_(options) {}

std::vector<float> Model::Forward(std::span<const std::int32_t> tokens,
                                  const TapFn& tap) const {
  const Config& c = weights_.config;
  const std::size_t T = tokens.size();
  if (T == 0) {
    throw std::invalid_argument("laguna oracle: empty token list");
  }
  const std::size_t H = c.hidden_size;

  const auto emit = [&](const char* name, int layer, TapType type,
                        std::array<std::uint64_t, 3> ne, const void* data) {
    if (!tap) {
      return;
    }
    const std::string full =
        layer >= 0 ? std::string(name) + "-" + std::to_string(layer)
                   : std::string(name);
    tap(Tap{full, type, ne, data});
  };
  const auto emit_f32 = [&](const char* name, int layer,
                            std::array<std::uint64_t, 3> ne,
                            const std::vector<float>& data) {
    emit(name, layer, TapType::kF32, ne, data.data());
  };

  // Embedding lookup.
  std::vector<float> h(T * H);
  for (std::size_t t = 0; t < T; ++t) {
    if (tokens[t] < 0 ||
        static_cast<std::uint64_t>(tokens[t]) >= weights_.token_embd.rows) {
      throw std::out_of_range("laguna oracle: token id outside the vocabulary");
    }
    DequantizeRow(weights_.token_embd.type,
                  static_cast<const std::uint8_t*>(weights_.token_embd.data) +
                      static_cast<std::size_t>(tokens[t]) *
                          weights_.token_embd.RowBytes(),
                  h.data() + t * H, H);
  }
  emit_f32("inp_embd", -1, {H, T, 1}, h);

  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    const LayerWeights& lw = weights_.layers[il];
    const int layer = static_cast<int>(il);

    // ---- attention ----
    const std::vector<float> attn_norm_w = LoadF32(lw.attn_norm);
    std::vector<float> n(T * H);
    for (std::size_t t = 0; t < T; ++t) {
      RmsNormRow(h.data() + t * H, attn_norm_w.data(), H, c.rms_eps,
                 n.data() + t * H);
    }
    emit_f32("attn_norm", layer, {H, T, 1}, n);

    const std::vector<float> o =
        AttnForward(c, lw, il, n.data(), T, options_, tap, layer);

    std::vector<float> ffn_inp(T * H);
    for (std::size_t i = 0; i < T * H; ++i) {
      ffn_inp[i] = o[i] + h[i];
    }
    emit_f32("ffn_inp", layer, {H, T, 1}, ffn_inp);

    // ---- MLP ----
    const std::vector<float> ffn_norm_w = LoadF32(lw.ffn_norm);
    std::vector<float> fn(T * H);
    for (std::size_t t = 0; t < T; ++t) {
      RmsNormRow(ffn_inp.data() + t * H, ffn_norm_w.data(), H, c.rms_eps,
                 fn.data() + t * H);
    }
    emit_f32("ffn_norm", layer, {H, T, 1}, fn);

    const std::vector<float> ffn_out =
        FfnForward(c, lw, fn.data(), T, options_, tap, layer);
    emit_f32("ffn_out", layer, {H, T, 1}, ffn_out);

    for (std::size_t i = 0; i < T * H; ++i) {
      h[i] = ffn_out[i] + ffn_inp[i];
    }
    emit_f32("l_out", layer, {H, T, 1}, h);
  }

  const std::vector<float> output_norm_w = LoadF32(weights_.output_norm);
  std::vector<float> normed(T * H);
  for (std::size_t t = 0; t < T; ++t) {
    RmsNormRow(h.data() + t * H, output_norm_w.data(), H, c.rms_eps,
               normed.data() + t * H);
  }
  emit_f32("result_norm", -1, {H, T, 1}, normed);

  const std::size_t rows = options_.last_row_logits_only ? 1 : T;
  std::vector<float> logits(rows * weights_.output.rows);
  MatMul(weights_.output, 0, normed.data() + (T - rows) * H, rows,
         logits.data(), options_);
  emit_f32("result_output", -1, {weights_.output.rows, rows, 1}, logits);
  return logits;
}

TargetTaps ForwardWithTaps(const Model& model,
                           std::span<const std::int32_t> tokens,
                           std::span<const std::uint32_t> output_layers) {
  const std::size_t T = tokens.size();
  TargetTaps out;
  out.layers.assign(output_layers.begin(), output_layers.end());
  std::vector<std::vector<float>> found(output_layers.size());
  std::vector<bool> seen(output_layers.size(), false);
  std::size_t hidden = 0;
  out.logits = model.Forward(tokens, [&](const Tap& tap) {
    for (std::size_t i = 0; i < output_layers.size(); ++i) {
      if (tap.name == "l_out-" + std::to_string(output_layers[i])) {
        hidden = tap.ne[0];
        found[i].assign(static_cast<const float*>(tap.data),
                        static_cast<const float*>(tap.data) + tap.Count());
        seen[i] = true;
      }
    }
  });
  for (std::size_t i = 0; i < output_layers.size(); ++i) {
    if (!seen[i] || found[i].size() != T * hidden) {
      throw std::invalid_argument("laguna oracle: no l_out tap for layer " +
                                  std::to_string(output_layers[i]));
    }
  }
  out.hidden_size = hidden;
  out.features.resize(T * output_layers.size() * hidden);
  for (std::size_t t = 0; t < T; ++t) {
    for (std::size_t i = 0; i < output_layers.size(); ++i) {
      std::copy_n(
          found[i].data() + t * hidden, hidden,
          out.features.data() + (t * output_layers.size() + i) * hidden);
    }
  }
  return out;
}

}  // namespace gufo::models::laguna::reference
