#ifndef GUFO_MODELS_LAGUNA_DFLASH_HPP_
#define GUFO_MODELS_LAGUNA_DFLASH_HPP_

// Laguna DFlash drafter (GGUF architecture `dflash`, `decoder_arch` laguna):
// metadata and weight index, CPU only. The contract is written down in
// docs/models/laguna-s-2.1/DFLASH.md; this file binds the artifact to it and
// computes nothing.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/laguna/config.hpp"
#include "src/models/laguna/weights.hpp"

namespace gufo::models::laguna {

/// Parameters of a Laguna DFlash drafter, read from the `dflash.*` keys. The
/// drafter has no embedding or output head: it reads the target's
/// `token_embd` and `output`, so hidden size and vocabulary must match the
/// target (ValidateAgainstTarget).
struct DFlashConfig {
  std::uint32_t num_layers{0};      ///< block_count, 6 (all sliding window).
  std::uint32_t hidden_size{0};     ///< embedding_length, 3072 (= target).
  std::uint32_t ffn_size{0};        ///< feed_forward_length, 12288, SwiGLU.
  std::uint32_t vocab_size{0};      ///< Length of tokenizer.ggml.tokens.
  std::uint32_t context_length{0};  ///< 1048576.
  float rms_eps{1e-6F};

  std::uint32_t num_heads{0};     ///< 72 query heads on every layer.
  std::uint32_t num_kv_heads{0};  ///< 8.
  std::uint32_t head_dim{0};      ///< 128.
  /// Every layer is sliding-window: key j is visible from p iff p - j < n.
  std::uint32_t sliding_window{0};
  /// One rope for all layers: NeoX, all `rotary_dim` (= head_dim) channels,
  /// no scaling.
  float rope_theta{0.0F};
  std::uint32_t rotary_dim{0};

  /// Block length including the anchor row: [last token, MASK x (n - 1)].
  std::uint32_t block_size{0};
  std::uint32_t mask_token_id{0};  ///< tokenizer.ggml.mask_token_id, 12.

  /// `dflash.target_layers` as written: target layer-INPUT numbering, so entry
  /// N is the residual stream entering target layer N (the output of layer
  /// N - 1), and an entry equal to the target's layer count is the stream
  /// after the last layer, before the final norm. Strictly increasing.
  std::vector<std::uint32_t> target_layers;

  /// Number of tapped target features per token (6).
  [[nodiscard]] std::uint32_t tap_count() const noexcept {
    return static_cast<std::uint32_t>(target_layers.size());
  }
  /// Zero-based index of the target layer whose OUTPUT entry `i` is:
  /// `target_layers[i] - 1`. This is the convention gufo's Qwen DFlash path
  /// normalizes to, and the `target_layer_ids` of the Hugging Face config.
  [[nodiscard]] std::uint32_t tap_layer(std::size_t i) const {
    return target_layers.at(i) - 1U;
  }
  /// Width of the encoder input per token: tap_count * hidden_size.
  [[nodiscard]] std::uint64_t fused_input_width() const noexcept {
    return std::uint64_t{tap_count()} * hidden_size;
  }
  [[nodiscard]] std::uint32_t q_dim() const noexcept {
    return num_heads * head_dim;
  }
  [[nodiscard]] std::uint32_t kv_dim() const noexcept {
    return num_kv_heads * head_dim;
  }
  /// Query heads per KV head (9).
  [[nodiscard]] std::uint32_t gqa_group() const noexcept {
    return num_heads / num_kv_heads;
  }
  /// Largest draft the block can carry: every row after the anchor.
  [[nodiscard]] std::uint32_t max_draft_tokens() const noexcept {
    return block_size - 1;
  }

  /// Reads and validates the `dflash.*` metadata and the vocabulary. Fails
  /// (nullopt and a message) on a missing key, a malformed value or a shape
  /// outside the profile this runtime implements (Laguna-S-2.1's drafter).
  [[nodiscard]] static std::optional<DFlashConfig> FromGguf(
      const core::GgufReader& reader, std::string* error_msg = nullptr);

  /// Checks that this drafter can read `target`: same hidden size and
  /// vocabulary, and taps that exist (the last tap is the post-last-layer
  /// stream, so `target_layers.back() == target.num_layers`).
  [[nodiscard]] bool ValidateAgainstTarget(const Config& target,
                                           std::string* error_msg) const;
};

/// Tensors of one drafter layer, by their GGUF names. Unlike the target's
/// v2 gate (F32), the drafter stores `attn_gate` quantized.
struct DFlashLayerWeights {
  TensorRef attn_norm;    ///< blk.N.attn_norm.weight   [hidden] F32
  TensorRef attn_q;       ///< [hidden -> heads * head_dim]
  TensorRef attn_k;       ///< [hidden -> kv_heads * head_dim]
  TensorRef attn_v;       ///< [hidden -> kv_heads * head_dim]
  TensorRef attn_output;  ///< [heads * head_dim -> hidden]
  TensorRef attn_q_norm;  ///< [head_dim] F32
  TensorRef attn_k_norm;  ///< [head_dim] F32
  TensorRef attn_gate;    ///< [hidden -> heads], per head softplus gate
  TensorRef ffn_norm;     ///< [hidden] F32
  TensorRef ffn_gate;     ///< [hidden -> ffn]
  TensorRef ffn_up;       ///< [hidden -> ffn]
  TensorRef ffn_down;     ///< [ffn -> hidden]
};

struct DFlashWeights {
  DFlashConfig config;
  /// enc.aux_norm.weight [hidden, taps] F32: tap k's RMSNorm weight is row k
  /// (contiguous `hidden` floats).
  TensorRef aux_norm;
  /// fc.weight [taps * hidden -> hidden]: the fusion projection.
  TensorRef fc;
  /// enc.output_norm.weight [hidden] F32: RMSNorm after `fc`.
  TensorRef enc_output_norm;
  /// output_norm.weight [hidden] F32: the drafter's final norm before the
  /// target's output head.
  TensorRef output_norm;
  std::vector<DFlashLayerWeights> layers;
  std::size_t tensor_count{0};
  std::uint64_t total_bytes{0};

  /// 4 global tensors plus 12 per layer (76 for six layers).
  [[nodiscard]] static std::size_t ExpectedTensorCount(
      const DFlashConfig& config);

  [[nodiscard]] static std::optional<DFlashWeights> Bind(
      const core::GgufReader& reader, std::string* error_msg = nullptr);

  /// Same binding over an explicit tensor list; fails on a missing, misshapen,
  /// mistyped or duplicate tensor and on any tensor the layout does not
  /// account for (an `output.weight` or `token_embd.weight` in a drafter
  /// would be one: it must not carry its own head).
  [[nodiscard]] static std::optional<DFlashWeights> Bind(
      const DFlashConfig& config, std::span<const core::GgufTensorInfo> tensors,
      std::span<const core::GgufMappedRegion> regions,
      std::string* error_msg = nullptr);
};

}  // namespace gufo::models::laguna

#endif  // GUFO_MODELS_LAGUNA_DFLASH_HPP_
