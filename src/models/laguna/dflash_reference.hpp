#ifndef GUFO_MODELS_LAGUNA_DFLASH_REFERENCE_HPP_
#define GUFO_MODELS_LAGUNA_DFLASH_REFERENCE_HPP_

// CPU reference forward ("oracle") for the Laguna DFlash drafter, written as
// docs/models/laguna-s-2.1/DFLASH.md reads Poolside's llama.cpp fork
// (src/models/dflash.cpp, decoder_arch laguna). It reuses the target oracle's
// building blocks (matmul, RoPE, RMSNorm, attention, SwiGLU) and keeps the
// same Options (F32 or activation-quant aware) and tap names.
//
// Nothing here has been compared against the fork's libllama yet: it is the
// spec's arithmetic, executed.

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "src/models/laguna/dflash.hpp"
#include "src/models/laguna/reference.hpp"
#include "src/models/laguna/weights.hpp"

namespace gufo::models::laguna::reference {

/// One drafter call: context features at target positions plus one block.
struct DFlashInputs {
  /// [n_ctx][taps * hidden]: per context token the target taps concatenated in
  /// DFlashConfig::target_layers order (TargetTaps::features layout).
  std::span<const float> features;
  /// Absolute position of each context row; all must be < block_position.
  std::span<const std::int32_t> context_positions;
  /// [anchor, MASK, ...], 2 to block_size entries; row r sits at
  /// block_position + r.
  std::span<const std::int32_t> block_tokens;
  std::int32_t block_position{0};
};

class DFlashModel {
public:
  /// `drafter` and `target` must outlive the model. The drafter reads the
  /// target's token_embd (noise embeddings) and output (head).
  DFlashModel(const DFlashWeights& drafter, const ModelWeights& target,
              Options options = {});

  /// Draft logits for block rows 1..B-1 ([B - 1][vocab], row 0 is discarded
  /// as in the fork). Emits the taps listed in DFLASH.md (enc_aux_norm,
  /// fc_out, enc_norm_out, kv_inp_normed-N, Kcur_injected-N, Vcur_injected-N,
  /// noise_norm-N, Qcur-N, Kcur-N, attn_gate_softplus-N, attn_gated-N,
  /// ffn_out-N, l_out-N, result_norm, result_output) when `tap` is set.
  [[nodiscard]] std::vector<float> Forward(const DFlashInputs& inputs,
                                           const TapFn& tap = {}) const;

private:
  const DFlashWeights& drafter_;
  const ModelWeights& target_;
  Options options_;
};

/// Index of the largest value of each of `rows` rows of `width` (first wins).
[[nodiscard]] std::vector<std::uint32_t> RowArgmax(std::span<const float> rows,
                                                   std::size_t width);

}  // namespace gufo::models::laguna::reference

#endif  // GUFO_MODELS_LAGUNA_DFLASH_REFERENCE_HPP_
