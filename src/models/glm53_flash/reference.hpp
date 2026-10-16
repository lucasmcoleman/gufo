#ifndef GUFO_MODELS_GLM53_FLASH_REFERENCE_HPP_
#define GUFO_MODELS_GLM53_FLASH_REFERENCE_HPP_

// Full CPU reference forward ("A5") for GLM-5.3-Flash (glm5-next): one
// unpadded sequence, F32 storage with F64 accumulation, prefill of T tokens
// followed by any number of incremental calls on persistent state. Mirrors HF
// transformers 09281b8 modeling_glm5_next.py:
//   Glm5NextTextModel.forward     (HF:1447-1510): 4 embedding copies, layers,
//                                  mean head, final RMSNorm
//   Glm5NextTextDecoderLayer      (HF:1299-1348): HcPre -> RMSNorm -> mixer ->
//                                  HcPost, then the same with the FFN
//   Glm5NextTextLinearAttention   (HF:664-769)  KDA, on head_count_kv == 0
//   Glm5NextTextAttention+Indexer (HF:810-1060, 1174-1235) absorbed NoPE MLA
//                                  with the k-pool DSA indexer
//   Glm5NextTextMLP / MoE         (HF:97-206) clamped SwiGLU, sigmoid router
// The router denominator follows llama.cpp (max(sum, 2^-14)), the project
// contract; the KDA prefill uses the sequential recurrence (HF's chunk form is
// the same function to ~3e-7). The indexer always appends the tail, as
// Config::FromGguf requires; every MLA layer runs its own indexer (no
// cross-layer top-k sharing, indexer_types "full" in the checkpoint).
//
// Weights are read through WeightSource by GGUF tensor name (weights.cpp
// names and layouts), one F32 row at a time.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "src/models/glm53_flash/config.hpp"
#include "src/models/glm53_flash/weights.hpp"

namespace gufo::models::glm53_flash::reference {

/// A [experts][rows][cols] matrix whose rows are produced as F32 on demand.
/// `cols` is the contiguous reduction dimension (GGUF ne[0]).
class MatrixView {
public:
  using RowReader = void (*)(const void* ctx, std::uint64_t expert,
                             std::uint64_t row, float* out);

  MatrixView() = default;
  MatrixView(const void* ctx, RowReader reader, std::uint64_t cols,
             std::uint64_t rows, std::uint64_t experts = 1)
      : ctx_(ctx),
        reader_(reader),
        cols_(cols),
        rows_(rows),
        experts_(experts) {}

  [[nodiscard]] std::uint64_t cols() const noexcept { return cols_; }
  [[nodiscard]] std::uint64_t rows() const noexcept { return rows_; }
  [[nodiscard]] std::uint64_t experts() const noexcept { return experts_; }

  /// Writes row `row` of stacked matrix `expert` (cols floats) to `out`.
  void Row(std::uint64_t expert, std::uint64_t row, float* out) const;
  /// y[r] = sum_c W[expert][r][c] * x[c] with F64 accumulation.
  void MatVec(const float* x, float* y, std::uint64_t expert = 0) const;

private:
  const void* ctx_{nullptr};
  RowReader reader_{nullptr};
  std::uint64_t cols_{0};
  std::uint64_t rows_{0};
  std::uint64_t experts_{0};
};

/// Read-only weight access by GGUF tensor name (e.g. "blk.3.attn_q_a.weight").
/// Missing tensors and shape mismatches throw std::runtime_error.
class WeightSource {
public:
  virtual ~WeightSource() = default;
  [[nodiscard]] virtual bool Has(const std::string& name) const = 0;
  /// A matrix (projection, expert stack, embedding, output) in any format.
  [[nodiscard]] virtual MatrixView Matrix(const std::string& name) const = 0;
  /// A contiguous F32 tensor of exactly `count` elements (norms, biases,
  /// conv, hyper-connection, router and ape weights; F32 in every GGUF).
  [[nodiscard]] virtual const float* Vector(const std::string& name,
                                            std::size_t count) const = 0;
};

/// In-memory F32 tensors keyed by GGUF name, for fixtures. A tensor of shape
/// [experts][rows][cols] (C order) is a matrix of that shape; Vector accepts
/// any tensor whose element count matches.
class F32WeightSource final : public WeightSource {
public:
  void Add(std::string name, std::vector<float> data, std::uint64_t cols,
           std::uint64_t rows = 1, std::uint64_t experts = 1);

  [[nodiscard]] bool Has(const std::string& name) const override;
  [[nodiscard]] MatrixView Matrix(const std::string& name) const override;
  [[nodiscard]] const float* Vector(const std::string& name,
                                    std::size_t count) const override;

private:
  struct Tensor {
    std::vector<float> data;
    std::uint64_t cols{0};
    std::uint64_t rows{0};
    std::uint64_t experts{0};
  };
  std::map<std::string, Tensor, std::less<>> tensors_;
};

/// Tensors bound by ModelWeights::Bind (or built by hand), dequantized one row
/// at a time with gufo's CPU dequantizers: F32, F16, BF16, Q8_0, Q3_K, Q4_K,
/// Q5_K, Q6_K, IQ4_NL, IQ4_XS, IQ3_S, IQ2_XXS, IQ3_XXS, IQ2_S. Other formats
/// (Q2_K, used only by the NextN experts) throw on first use. The weights and
/// the mapped file must outlive this object.
class GgufWeightSource final : public WeightSource {
public:
  explicit GgufWeightSource(const ModelWeights& weights);

  [[nodiscard]] bool Has(const std::string& name) const override;
  [[nodiscard]] MatrixView Matrix(const std::string& name) const override;
  [[nodiscard]] const float* Vector(const std::string& name,
                                    std::size_t count) const override;

private:
  void Index(const TensorRef& t);
  std::unordered_map<std::string, const TensorRef*> tensors_;
};

/// Called once per tap with `n_tokens` rows of `width` floats (row-major).
/// `layer` is the decoder layer, or -1 for "hc_head.out" / "final_norm.out".
/// Layer tap names follow the HF module hooks of the fixture:
///   in, attn_hc.out0 (post [hc]), attn_hc.out1 (comb [hc*hc], [src][dst]),
///   attn_hc.out2 (collapsed), input_layernorm.out, self_attn.out,
///   ffn_hc.in, ffn_hc.out0/out1/out2, post_attention_layernorm.out, mlp.out,
///   out0. Stream taps (in, ffn_hc.in, out0) are [hc][hidden] per token.
using TapFn =
    std::function<void(std::string_view name, int layer, const float* data,
                       std::size_t width, std::uint32_t n_tokens)>;

/// Persistent per-sequence state.
struct KdaLayerState {
  std::vector<float> conv_history;  ///< [kernel-1][3*kda_dim], q|k|v
  std::vector<float> recurrent;     ///< [heads][Dk][Dv]
};
struct MlaLayerState {
  std::vector<float> latent;      ///< [position][kv_lora] (the KV cache)
  std::vector<float> pooled;      ///< [complete pools][indexer_dim]
  std::vector<float> tail_keys;   ///< [< kpool][indexer_dim] LayerNorm keys
  std::vector<float> tail_gates;  ///< [< kpool][indexer_dim] gate logits
};

class ReferenceModel {
public:
  struct Output {
    std::vector<float> hidden;  ///< [T][hidden] final normed hidden state
    std::vector<float> logits;  ///< [T][vocab] when requested and available
  };

  /// `config` and `weights` must outlive the model.
  ReferenceModel(const Config& config, const WeightSource& weights);

  /// Clears every cache; the next Forward is a fresh prefill.
  void Reset();
  [[nodiscard]] std::uint32_t Position() const noexcept { return position_; }

  /// Runs `tokens` at positions Position() .. Position()+T-1, advancing the
  /// state. Logits are produced only if `want_logits` and "output.weight"
  /// exists.
  Output Forward(std::span<const std::int32_t> tokens, bool want_logits = false,
                 const TapFn& tap = {});

private:
  void Layer(std::uint32_t il, std::uint32_t n_tokens, std::vector<float>& H,
             const TapFn& tap);
  void Kda(std::uint32_t il, const float* x, std::uint32_t n_tokens, float* y);
  void Mla(std::uint32_t il, const float* x, std::uint32_t n_tokens, float* y);
  void Ffn(std::uint32_t il, const float* x, float* y) const;

  const Config& config_;
  const WeightSource& weights_;
  std::uint32_t position_{0};
  std::vector<KdaLayerState> kda_;  ///< Indexed by layer; empty on MLA layers.
  std::vector<MlaLayerState> mla_;  ///< Indexed by layer; empty on KDA layers.
};

}  // namespace gufo::models::glm53_flash::reference

#endif  // GUFO_MODELS_GLM53_FLASH_REFERENCE_HPP_
