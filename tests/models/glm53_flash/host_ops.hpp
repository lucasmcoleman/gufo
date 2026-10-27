#ifndef GUFO_TESTS_MODELS_GLM53_FLASH_HOST_OPS_HPP_
#define GUFO_TESTS_MODELS_GLM53_FLASH_HOST_OPS_HPP_

// Host stand-ins for Lane B's MLA and FFN operators (layer_ops.hpp), bound
// into the executor through LayerOps until the GPU operators exist. Each call
// copies x to the host, synchronises ctx.stream, runs the scalar oracles of
// reference_mla.hpp / reference_moe.hpp, and copies y back. Weights are read
// back from the uploaded MlaLayerDevice / FfnLayerDevice (dequantised with
// gufo's CPU dequantisers, cached per layer), so a run also checks that the
// uploads Lane B will consume are populated as layer_ops.hpp documents. The
// math mirrors ReferenceModel::Mla / ::Ffn (reference.cpp). The MLA state
// lives on the host, keyed by layer; the device MlaCache is only checked for
// size. Not thread-safe.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/glm53_flash/config.hpp"
#include "src/models/glm53_flash/executor.hpp"
#include "src/models/glm53_flash/layer_ops.hpp"
#include "src/models/glm53_flash/reference_mla.hpp"
#include "src/models/glm53_flash/reference_moe.hpp"

namespace gufo::models::glm53_flash::stand_in {

/// One encoded row of `cols` elements as F32 (the formats ReferenceModel's
/// GgufWeightSource reads).
inline void DequantizeRow(core::GgmlType type, const std::uint8_t* src,
                          std::size_t cols, float* out) {
  namespace q = ::gufo::quant;
  using core::GgmlType;
  switch (type) {
    case GgmlType::kF32:
      std::memcpy(out, src, cols * sizeof(float));
      return;
    case GgmlType::kF16:
      for (std::size_t i = 0; i < cols; ++i) {
        std::uint16_t h = 0;
        std::memcpy(&h, src + (2 * i), 2);
        out[i] = q::Fp16ToFloat(h);
      }
      return;
    case GgmlType::kBF16:
      for (std::size_t i = 0; i < cols; ++i) {
        std::uint16_t h = 0;
        std::memcpy(&h, src + (2 * i), 2);
        const std::uint32_t bits = static_cast<std::uint32_t>(h) << 16;
        std::memcpy(out + i, &bits, 4);
      }
      return;
    case GgmlType::kQ8_0:
      return q::DequantizeQ8_0(src, out, cols);
    case GgmlType::kQ3_K:
      return q::DequantizeQ3_K(src, out, cols);
    case GgmlType::kQ4_K:
      return q::DequantizeQ4_K(src, out, cols);
    case GgmlType::kQ5_K:
      return q::DequantizeQ5_K(src, out, cols);
    case GgmlType::kQ6_K:
      return q::DequantizeQ6_K(src, out, cols);
    case GgmlType::kQ8_K:
      return q::DequantizeQ8_K(src, out, cols);
    case GgmlType::kIQ4_NL:
      return q::DequantizeIQ4_NL(src, out, cols);
    case GgmlType::kIQ4_XS:
      return q::DequantizeIQ4_XS(src, out, cols);
    case GgmlType::kIQ3_S:
      return q::DequantizeIQ3_S(src, out, cols);
    case GgmlType::kIQ2_XXS:
      return q::DequantizeIQ2_XXS(src, out, cols);
    case GgmlType::kIQ3_XXS:
      return q::DequantizeIQ3_XXS(src, out, cols);
    case GgmlType::kIQ2_S:
      return q::DequantizeIQ2_S(src, out, cols);
    default:
      throw std::runtime_error("stand-in: no CPU dequantizer for type " +
                               std::to_string(static_cast<int>(type)));
  }
}

/// A device matrix read back as F32 [experts][rows][cols]. Throws on an
/// empty matrix or a failed copy.
inline std::vector<float> DownloadMatrix(const DeviceMatrix& m) {
  if (m.empty() || m.row_bytes == 0) {
    throw std::runtime_error("stand-in: empty device matrix");
  }
  const std::size_t rows = m.rows * m.experts;
  std::vector<std::uint8_t> raw(rows * m.row_bytes);
  if (hipMemcpy(raw.data(), m.data, raw.size(), hipMemcpyDeviceToHost) !=
      hipSuccess) {
    throw std::runtime_error("stand-in: matrix download failed");
  }
  std::vector<float> out(rows * m.cols);
  for (std::size_t r = 0; r < rows; ++r) {
    DequantizeRow(m.type, raw.data() + (r * m.row_bytes), m.cols,
                  out.data() + (r * m.cols));
  }
  return out;
}

inline std::vector<float> DownloadVector(const float* p, std::size_t n) {
  if (p == nullptr) {
    throw std::runtime_error("stand-in: null device vector");
  }
  std::vector<float> out(n);
  if (hipMemcpy(out.data(), p, n * sizeof(float), hipMemcpyDeviceToHost) !=
      hipSuccess) {
    throw std::runtime_error("stand-in: vector download failed");
  }
  return out;
}

/// y[r] = sum_c W[r][c] x[c], F64 accumulation (MatrixView::MatVec).
inline void MatVec(const std::vector<float>& w, std::size_t cols,
                   const float* x, float* y) {
  const std::size_t rows = w.size() / cols;
  for (std::size_t r = 0; r < rows; ++r) {
    double acc = 0.0;
    for (std::size_t c = 0; c < cols; ++c) {
      acc += static_cast<double>(w[(r * cols) + c]) * x[c];
    }
    y[r] = static_cast<float>(acc);
  }
}

/// HF Glm5NextTextRMSNorm, as reference.cpp computes it.
inline void RmsNorm(const float* x, const float* weight, std::size_t n,
                    float eps, float* y) {
  double ss = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    ss += static_cast<double>(x[i]) * x[i];
  }
  const double inv = 1.0 / std::sqrt((ss / static_cast<double>(n)) + eps);
  for (std::size_t i = 0; i < n; ++i) {
    y[i] = static_cast<float>(static_cast<double>(x[i]) * inv * weight[i]);
  }
}

class HostOps {
public:
  explicit HostOps(const Config& config) : config_(config) {}

  /// The operator table; `this` must outlive every executor it is bound to.
  [[nodiscard]] LayerOps Bind() {
    LayerOps ops;
    // A latent-sized cache, so the executor's allocation and reset run; the
    // stand-in keeps its own state on the host.
    ops.mla_cache_bytes = [](const Config& c, std::uint32_t capacity) {
      return std::size_t{capacity} * c.kv_lora_rank * sizeof(float);
    };
    ops.op_scratch_bytes = [](const Config&, std::uint32_t) {
      return std::size_t{0};
    };
    ops.mla = [this](const OpContext& ctx, const Config& c, std::uint32_t il,
                     const MlaLayerDevice& w, MlaCache& cache, const float* x,
                     std::uint32_t n, std::uint32_t pos0, float* y) {
      return Call(ctx, c, x, n, y, [&](const float* hx, float* hy) {
        Mla(il, w, cache, hx, n, pos0, hy);
      });
    };
    ops.ffn = [this](const OpContext& ctx, const Config& c, std::uint32_t il,
                     const FfnLayerDevice& w, const float* x, std::uint32_t n,
                     float* y) {
      return Call(ctx, c, x, n, y,
                  [&](const float* hx, float* hy) { Ffn(il, w, hx, n, hy); });
    };
    return ops;
  }

  [[nodiscard]] std::uint32_t mla_calls() const noexcept { return mla_calls_; }
  [[nodiscard]] std::uint32_t ffn_calls() const noexcept { return ffn_calls_; }
  /// The last exception a call swallowed (it returned false).
  [[nodiscard]] const std::string& last_error() const noexcept {
    return last_error_;
  }

private:
  struct MlaHost {
    bool loaded{false};
    std::vector<float> q_a, q_b, kv_a, k_b, v_b, out;
    std::vector<float> idx_q_b, idx_k, idx_proj, idx_gate;
    std::vector<float> q_a_norm, kv_a_norm, k_norm, k_norm_bias, ape;
    // ReferenceModel's MlaLayerState.
    std::vector<float> latent, pooled, tail_keys, tail_gates;
    // Every token's indexer key and gate, so a position rewind (Executor
    // Rollback after a verify) can rebuild the pool tail.
    std::vector<float> keys, gates;
  };
  struct FfnHost {
    bool loaded{false};
    std::vector<float> gate, up, down;  ///< dense or shared expert
    std::vector<float> router, bias;
    std::vector<float> gate_up;  ///< [expert][2*expert_ff][hidden]
    std::vector<float> down_exps;
  };

  /// Copies x down, runs `body` on the host, copies y up.
  template<typename Body>
  bool Call(const OpContext& ctx, const Config& c, const float* x,
            std::uint32_t n, float* y, Body&& body) {
    const std::size_t count = std::size_t{n} * c.hidden_size;
    std::vector<float> hx(count);
    std::vector<float> hy(count);
    try {
      if (hipMemcpyAsync(hx.data(), x, count * sizeof(float),
                         hipMemcpyDeviceToHost, ctx.stream) != hipSuccess ||
          hipStreamSynchronize(ctx.stream) != hipSuccess) {
        throw std::runtime_error("activation download failed");
      }
      body(hx.data(), hy.data());
      if (hipMemcpyAsync(y, hy.data(), count * sizeof(float),
                         hipMemcpyHostToDevice, ctx.stream) != hipSuccess ||
          hipStreamSynchronize(ctx.stream) != hipSuccess) {
        throw std::runtime_error("activation upload failed");
      }
    } catch (const std::exception& e) {
      last_error_ = e.what();
      return false;
    }
    return true;
  }

  void Mla(std::uint32_t il, const MlaLayerDevice& w, const MlaCache& cache,
           const float* x, std::uint32_t n, std::uint32_t pos0, float* y) {
    ++mla_calls_;
    const Config& c = config_;
    const std::size_t hidden = c.hidden_size;
    const std::size_t heads = c.num_heads;
    const std::size_t q_lora = c.q_lora_rank;
    const std::size_t kv_lora = c.kv_lora_rank;
    const std::size_t nope = c.mla_head_dim;
    const std::size_t v_dim = c.mla_value_dim;
    const std::size_t ih = c.indexer_heads;
    const std::size_t idim = c.indexer_head_dim;
    const std::size_t kpool = c.indexer_kpool;
    MlaHost& h = mla_[il];
    if (!h.loaded) {
      h.q_a = DownloadMatrix(w.q_a);
      h.q_b = DownloadMatrix(w.q_b);
      h.kv_a = DownloadMatrix(w.kv_a);
      h.k_b = DownloadMatrix(w.k_b);
      h.v_b = DownloadMatrix(w.v_b);
      h.out = DownloadMatrix(w.out);
      h.idx_q_b = DownloadMatrix(w.idx_q_b);
      h.idx_k = DownloadMatrix(w.idx_k);
      h.idx_proj = DownloadMatrix(w.idx_proj);
      h.idx_gate = DownloadMatrix(w.idx_gate);
      h.q_a_norm = DownloadVector(w.q_a_norm, q_lora);
      h.kv_a_norm = DownloadVector(w.kv_a_norm, kv_lora);
      h.k_norm = DownloadVector(w.idx_k_norm, idim);
      h.k_norm_bias = DownloadVector(w.idx_k_norm_bias, idim);
      h.ape = DownloadVector(w.idx_ape, kpool * idim);
      h.loaded = true;
    }
    if (pos0 == 0) {
      h.latent.clear();
      h.pooled.clear();
      h.tail_keys.clear();
      h.tail_gates.clear();
      h.keys.clear();
      h.gates.clear();
    }
    if (h.latent.size() > std::size_t{pos0} * kv_lora) {
      // Rewound (Executor::Rollback): drop the later tokens. Completed pools
      // before pos0 are unchanged; the tail is rebuilt from the history.
      const std::size_t pools = pos0 / kpool;
      h.latent.resize(std::size_t{pos0} * kv_lora);
      h.keys.resize(std::size_t{pos0} * idim);
      h.gates.resize(std::size_t{pos0} * idim);
      h.pooled.resize(pools * idim);
      const auto tail = static_cast<std::ptrdiff_t>(pools * kpool * idim);
      h.tail_keys.assign(h.keys.begin() + tail, h.keys.end());
      h.tail_gates.assign(h.gates.begin() + tail, h.gates.end());
    }
    if (h.latent.size() != std::size_t{pos0} * kv_lora) {
      throw std::runtime_error("MLA stand-in: pos0 does not continue state");
    }
    if (cache.data == nullptr || pos0 + n > cache.capacity) {
      throw std::runtime_error("MLA stand-in: cache missing or too small");
    }
    using namespace ::gufo::models::glm53_flash::reference;
    std::vector<float> tmp(std::max({q_lora, kv_lora, idim}));
    std::vector<float> q_resid(q_lora);
    std::vector<float> q(heads * nope);
    std::vector<float> q_abs(heads * kv_lora);
    std::vector<float> ctx(heads * kv_lora);
    std::vector<float> attn(heads * v_dim);
    std::vector<float> iq(ih * idim);
    std::vector<float> iw(ih);
    std::vector<float> key(idim);
    std::vector<float> pooled(idim);
    std::vector<float> scores;
    for (std::uint32_t t = 0; t < n; ++t) {
      const float* xt = x + (t * hidden);
      const std::uint32_t pos = pos0 + t;
      MatVec(h.q_a, hidden, xt, tmp.data());
      RmsNorm(tmp.data(), h.q_a_norm.data(), q_lora, c.rms_eps, q_resid.data());
      MatVec(h.q_b, q_lora, q_resid.data(), q.data());
      MatVec(h.kv_a, hidden, xt, tmp.data());
      h.latent.resize((std::size_t{pos} + 1) * kv_lora);
      RmsNorm(tmp.data(), h.kv_a_norm.data(), kv_lora, c.rms_eps,
              h.latent.data() + (std::size_t{pos} * kv_lora));
      MatVec(h.idx_k, hidden, xt, tmp.data());
      LayerNorm(tmp.data(), h.k_norm.data(), h.k_norm_bias.data(), idim,
                c.norm_eps, key.data());
      h.tail_keys.insert(h.tail_keys.end(), key.begin(), key.end());
      h.keys.insert(h.keys.end(), key.begin(), key.end());
      MatVec(h.idx_gate, hidden, xt, tmp.data());
      h.tail_gates.insert(h.tail_gates.end(), tmp.begin(),
                          tmp.begin() + static_cast<std::ptrdiff_t>(idim));
      h.gates.insert(h.gates.end(), tmp.begin(),
                     tmp.begin() + static_cast<std::ptrdiff_t>(idim));
      if (h.tail_keys.size() == kpool * idim) {
        PoolKeys(h.tail_keys.data(), h.tail_gates.data(), h.ape.data(), kpool,
                 idim, pooled.data());
        h.pooled.insert(h.pooled.end(), pooled.begin(), pooled.end());
        h.tail_keys.clear();
        h.tail_gates.clear();
      }
      MatVec(h.idx_q_b, q_lora, q_resid.data(), iq.data());
      MatVec(h.idx_proj, hidden, xt, iw.data());
      const std::size_t pools = h.pooled.size() / idim;
      scores.assign(pools, 0.0F);
      IndexerScores(iq.data(), h.pooled.data(), iw.data(), ih, idim, pools,
                    scores.data());
      const Selection sel =
          SelectPools(scores.data(), pos + 1, 0, c.indexer_kpool,
                      c.IndexerPoolBudget(), /*select_tail=*/true);
      AbsorbQuery(q.data(), h.k_b.data(), heads, nope, kv_lora, q_abs.data());
      LatentAttention(q_abs.data(), h.latent.data(), std::size_t{pos} + 1,
                      heads, kv_lora, nope, sel.tokens, ctx.data());
      ProjectValues(ctx.data(), h.v_b.data(), heads, kv_lora, v_dim,
                    attn.data());
      MatVec(h.out, heads * v_dim, attn.data(), y + (t * hidden));
    }
  }

  void Ffn(std::uint32_t il, const FfnLayerDevice& w, const float* x,
           std::uint32_t n, float* y) {
    ++ffn_calls_;
    const Config& c = config_;
    const std::size_t hidden = c.hidden_size;
    using namespace ::gufo::models::glm53_flash::reference;
    FfnHost& h = ffn_[il];
    if (!h.loaded) {
      if (w.dense) {
        h.gate = DownloadMatrix(w.gate);
        h.up = DownloadMatrix(w.up);
        h.down = DownloadMatrix(w.down);
      } else {
        const std::size_t ne = c.num_experts;
        const std::size_t eff = c.expert_ff;
        h.gate = DownloadMatrix(w.shexp_gate);
        h.up = DownloadMatrix(w.shexp_up);
        h.down = DownloadMatrix(w.shexp_down);
        h.router = DownloadVector(w.router, ne * hidden);
        h.bias = DownloadVector(w.router_bias, ne);
        const std::vector<float> g = DownloadMatrix(w.gate_exps);
        const std::vector<float> u = DownloadMatrix(w.up_exps);
        h.down_exps = DownloadMatrix(w.down_exps);
        const std::size_t m = eff * hidden;
        h.gate_up.resize(ne * 2 * m);
        for (std::size_t e = 0; e < ne; ++e) {
          std::memcpy(h.gate_up.data() + (e * 2 * m), g.data() + (e * m),
                      m * sizeof(float));
          std::memcpy(h.gate_up.data() + (e * 2 * m) + m, u.data() + (e * m),
                      m * sizeof(float));
        }
      }
      h.loaded = true;
    }
    const std::size_t ff = h.gate.size() / hidden;
    for (std::uint32_t t = 0; t < n; ++t) {
      const float* xt = x + (t * hidden);
      float* yt = y + (t * hidden);
      if (w.dense) {
        DenseFfn(xt, h.gate.data(), h.up.data(), h.down.data(), hidden, ff,
                 w.clamp_exp, yt);
        continue;
      }
      const std::size_t ne = c.num_experts;
      std::vector<float> logits(ne);
      for (std::size_t e = 0; e < ne; ++e) {
        double acc = 0.0;
        for (std::size_t d = 0; d < hidden; ++d) {
          acc += static_cast<double>(h.router[(e * hidden) + d]) * xt[d];
        }
        logits[e] = static_cast<float>(acc);
      }
      const Routing routing =
          Route(logits.data(), h.bias.data(), static_cast<std::uint32_t>(ne),
                c.num_experts_used, c.expert_weights_scale,
                c.expert_weights_norm, RouterDenominator::kLlamaClamp);
      std::vector<float> routed(hidden);
      std::vector<float> shared(hidden);
      RoutedFfn(xt, h.gate_up.data(), h.down_exps.data(), hidden, c.expert_ff,
                routing, w.clamp_exp, routed.data());
      DenseFfn(xt, h.gate.data(), h.up.data(), h.down.data(), hidden, ff,
               w.clamp_shexp, shared.data());
      for (std::size_t d = 0; d < hidden; ++d) {
        yt[d] = routed[d] + shared[d];
      }
    }
  }

  const Config& config_;
  std::map<std::uint32_t, MlaHost> mla_;
  std::map<std::uint32_t, FfnHost> ffn_;
  std::uint32_t mla_calls_{0};
  std::uint32_t ffn_calls_{0};
  std::string last_error_;
};

}  // namespace gufo::models::glm53_flash::stand_in

#endif  // GUFO_TESTS_MODELS_GLM53_FLASH_HOST_OPS_HPP_
