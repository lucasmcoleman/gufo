#ifndef GUFO_MODELS_LAGUNA_REFERENCE_HPP_
#define GUFO_MODELS_LAGUNA_REFERENCE_HPP_

// CPU reference forward ("oracle") for Laguna-S-2.1, written exactly as
// docs/models/laguna-s-2.1/ARCH.md specifies and with llama.cpp's arithmetic
// where ARCH.md names it (ggml RoPE/YaRN, RMSNorm, softplus, the 2^-14 router
// clamp). It is a correctness reference, not a fast path: full recompute of
// the whole sequence, no KV cache.
//
// Weights stay in the (mmapped) GGUF. Each matrix row is dequantized on the
// fly into a small per-thread buffer and consumed immediately, so the 74 GB
// model is never held in F32. MoE layers dequantize only the experts the
// router selected for the tokens at hand.
//
// Two modes, both kept:
//  - F32: activations stay exact F32 (dequantized weights, exact inputs).
//  - Q8-aware: every matmul input is rounded the way the llama.cpp vec-dot
//    path quantizes it for the weight type, so kernel arithmetic bugs are not
//    confused with expected quantization error. See ActivationQuant.

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/laguna/weights.hpp"

namespace gufo::models::laguna::reference {

/// How matmul inputs are rounded before the dot product.
enum class ActivationQuant {
  /// Exact F32 inputs.
  kNone,
  /// ggml-cpu `vec_dot_type` of the weight: Q8_0 (32-blocks, fp16 scale) for
  /// Q8_0/IQ4_NL weights, Q8_K (256-blocks, float scale) for Q3_K..Q6_K and
  /// IQ4_XS weights, F16 for F16 weights, BF16 for BF16 weights; F32 weights
  /// get exact inputs. Reference: quantize_row_q8_0_ref /
  /// quantize_row_q8_K_ref.
  kCpu,
  /// ggml HIP/CUDA mul_mat_vec_q path: Q8_1 (32-blocks, fp16 scale, same
  /// rounding as Q8_0) for every quantized weight type; F32 weights exact.
  kGpuQ8_1,
};

/// Router weight denominator. Production gufo follows llama.cpp.
enum class RouterDenominator {
  kLlamaClamp,  ///< w = s[sel] / max(sum, 2^-14)  (llama.cpp d854d98).
  kHfRaw,       ///< w = s[sel] / sum               (modeling_laguna.py:182).
};

struct Options {
  ActivationQuant activation_quant{ActivationQuant::kNone};
  RouterDenominator router_denominator{RouterDenominator::kLlamaClamp};
  /// Round K (after RoPE) and V through F16 before attention, as llama.cpp's
  /// default F16 KV cache does.
  bool round_kv_f16{false};
  /// Compute result_output only for the last token (the other rows are not
  /// returned). Saves most of the output projection for long prompts.
  bool last_row_logits_only{false};
  /// Worker threads for matmul rows. 0 picks the hardware concurrency. The
  /// result does not depend on the thread count.
  unsigned threads{0};
};

enum class TapType { kF32, kI32 };

/// One named intermediate tensor. `name` carries llama.cpp's "-<layer>" suffix
/// for per-layer tensors ("Qcur-3") and none for result_norm/result_output.
/// `ne` follows ggml: ne[0] is the contiguous (fastest) dimension and the
/// token axis is last, so a [width, T] tensor is T rows of `width` values.
struct Tap {
  std::string_view name;
  TapType type{TapType::kF32};
  std::array<std::uint64_t, 3> ne{1, 1, 1};
  const void* data{nullptr};
  [[nodiscard]] std::uint64_t Count() const noexcept {
    return ne[0] * ne[1] * ne[2];
  }
};
using TapFn = std::function<void(const Tap&)>;

/// Writes every tap as `<name>.f32` / `<name>.i32` (raw little-endian) and,
/// on Finish(), `index.json` listing name, file, dtype and ggml `ne`.
class TapWriter {
public:
  /// `filter` keeps only taps whose name contains one of the substrings
  /// (empty keeps all).
  explicit TapWriter(std::filesystem::path directory,
                     std::vector<std::string> filter = {});
  void operator()(const Tap& tap);
  /// Writes index.json. Returns false (message in `error`) on I/O failure.
  bool Finish(std::string* error = nullptr);
  [[nodiscard]] std::size_t written() const noexcept { return entries_.size(); }

private:
  struct Entry {
    std::string name;
    std::string file;
    TapType type;
    std::array<std::uint64_t, 3> ne;
  };
  std::filesystem::path directory_;
  std::vector<std::string> filter_;
  std::vector<Entry> entries_;
  std::string error_;
};

// ---- Building blocks, public so the unit tests can exercise them alone. ----

/// Round-trip through IEEE binary16 (round to nearest even).
[[nodiscard]] float RoundToF16(float value) noexcept;

/// Rounds `rows` x `cols` activations in place the way `mode` quantizes the
/// matmul input for a weight of type `weight_type`.
void RoundActivations(core::GgmlType weight_type, ActivationQuant mode,
                      float* x, std::size_t rows, std::size_t cols);

/// Dequantizes one logical row of `cols` elements (F32/F16/BF16/Q8_0/Q4_K/
/// Q5_K/Q6_K/IQ4_XS). Aborts on any other type.
void DequantizeRow(core::GgmlType type, const void* src, float* dst,
                   std::size_t cols);

/// ggml_rope_ext parameters of one layer, NeoX layout, leading `n_rot`
/// channels rotated.
struct RopeParams {
  std::uint32_t n_rot{0};
  float freq_base{10000.0F};
  float freq_scale{1.0F};  ///< 1 / factor with YaRN, else 1.
  float ext_factor{0.0F};  ///< 1 with YaRN, 0 without.
  float attn_factor{
      1.0F};  ///< ggml's `attn_factor` argument (not the net mscale).
  float beta_fast{32.0F};
  float beta_slow{1.0F};
  std::uint32_t n_ctx_orig{0};
};
/// Parameters llama.cpp passes for layer `il` (full layers: YaRN; sliding
/// layers: plain RoPE). With YaRN, ggml multiplies cos/sin by
/// attn_factor * (1 + 0.1 ln(1/freq_scale)), which equals Config::YarnMscale().
[[nodiscard]] RopeParams RopeFor(const Config& config, std::uint32_t il);
/// Rotates `heads` heads of `head_dim` channels per token in place. `x` holds
/// `positions.size()` rows of `heads * head_dim`.
void ApplyRope(float* x, std::span<const std::int32_t> positions,
               std::uint32_t heads, std::uint32_t head_dim,
               const RopeParams& rope);

/// Causal softmax attention with grouped-query heads. q is [T][heads*hd],
/// k and v are [T][kv_heads*hd]. Query head h reads kv head h / (heads /
/// kv_heads). `window` 0 means full causal; otherwise key j is visible from
/// position p iff j <= p and p - j < window.
void GqaAttention(const float* q, const float* k, const float* v,
                  std::size_t tokens, std::uint32_t heads,
                  std::uint32_t kv_heads, std::uint32_t head_dim,
                  std::uint32_t window, float scale, float* out);

/// Routing of one token. `probs` and `weights` are in selection order.
struct Routing {
  std::vector<std::uint32_t> ids;
  std::vector<float> probs;    ///< s[sel], unbiased sigmoid.
  std::vector<float> weights;  ///< Final weights, scale included.
  float sum{0.0F};             ///< sum of probs.
  float denominator{1.0F};     ///< value the sum was divided by.
};
/// Sigmoid routing with a selection-only bias. Ties pick the lower expert id.
/// `normalize` divides by the denominator, `scale` multiplies afterwards
/// (skipped when 0 or 1, as llama.cpp does).
[[nodiscard]] Routing Route(const float* logits, const float* bias,
                            std::uint32_t experts, std::uint32_t used,
                            bool normalize, float scale,
                            RouterDenominator denominator);

class Model {
public:
  /// `weights` must outlive the model (it points into the mapping).
  explicit Model(const ModelWeights& weights, Options options = {});

  /// Runs the whole network over `tokens` (positions 0..T-1) and returns the
  /// logits, T rows of vocab_size (or one row with last_row_logits_only).
  /// Emits taps when `tap` is set.
  [[nodiscard]] std::vector<float> Forward(std::span<const std::int32_t> tokens,
                                           const TapFn& tap = {}) const;

private:
  const ModelWeights& weights_;
  Options options_;
};

}  // namespace gufo::models::laguna::reference

#endif  // GUFO_MODELS_LAGUNA_REFERENCE_HPP_
