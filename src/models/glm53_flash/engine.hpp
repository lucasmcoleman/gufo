#ifndef GUFO_MODELS_GLM53_FLASH_ENGINE_HPP_
#define GUFO_MODELS_GLM53_FLASH_ENGINE_HPP_

// GLM-5.3-Flash runtime: a GGUF with general.architecture = glm5-next loads
// into one resident GPU Executor (executor.hpp) with its tokenizer, and
// hands out a ForwardSession for generator.hpp. The Executor owns weights and
// sequence state together, so a Model serves one session at a time.
//
// The MLA / MoE sub-layers are Lane B's GPU operators (layer_ops.hpp). A
// build without them (GLM53_LANE_B_OPS undefined) refuses every load with
// kLaneBMissingError before reading weights; nothing falls back to a slower
// or partial path.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

#include "src/core/gguf_reader.hpp"
#include "src/models/glm53_flash/config.hpp"
#include "src/models/glm53_flash/generator.hpp"
#include "src/models/glm53_flash/tokenizer.hpp"

namespace gufo::models::glm53_flash {

class Executor;

/// `general.architecture` of a GLM-5.3-Flash GGUF.
inline constexpr std::string_view kArchitecture = "glm5-next";

/// Why a build without Lane B's GPU operators cannot load the model.
inline constexpr std::string_view kLaneBMissingError =
    "GLM-5.3-Flash: GPU MLA/MoE operators not built yet (Lane B)";

/// True for a GGUF whose architecture is kArchitecture.
[[nodiscard]] bool IsGlm53Flash(const core::GgufReader& reader);

/// True when this build can run GLM-5.3-Flash: ROCm and Lane B's operators.
/// Otherwise false, with the reason (kLaneBMissingError first).
[[nodiscard]] bool RuntimeAvailable(std::string* error = nullptr);

struct ModelOptions {
  /// Positions per sequence (MLA cache capacity). 0 = the GGUF's context
  /// length.
  std::uint32_t max_context{4096};
  /// Tokens per executor pass; longer prefills run in several passes.
  std::uint32_t prefill_chunk{256};
  /// Compute the last prompt token's logits inside the final prefill pass
  /// (Executor::ForwardLast) rather than in an extra one-token pass. The
  /// environment variable GUFO_GLM53_FUSED_LAST_LOGITS=0 turns it off for A/B
  /// runs.
  bool fused_last_logits{true};
  /// Speculative decoding with the GGUF's NextN (MTP) block when it has one:
  /// one greedy draft token per step, verified by the trunk
  /// (sample-and-match, so outputs follow the sampler exactly). Costs the
  /// NextN weights (~2.8 GB) and a KDA state checkpoint per layer. The
  /// environment variable GUFO_GLM53_MTP=0 turns it off.
  bool mtp{true};
  /// With MTP, chain a second draft token when the first draft's probability
  /// under the draft block is at least this. glm53_flash_mtp_depth_bench (6
  /// prompts, two runs): gate 0.9-0.95 +1.5% to +2.7% over one draft, a
  /// fixed second draft -2.4% to +0.4%. End to end (gufo prompt, 5 prompts x
  /// 2 fresh runs) gate 0.95 averaged 16.8 tok/s against 17.1 for one draft,
  /// within fresh-process noise, so the default stays off. Above 1 keeps one
  /// draft per step. The environment variable GUFO_GLM53_MTP_GATE overrides it.
  float mtp_chain_gate{2.0F};
  /// Run a short prefill, a decode step and a draft cycle at the end of
  /// Load, so the first request does not pay the weights' first GPU touch
  /// and each kernel's first launch (about 1-4 s on a loaded MasterServer).
  /// Off by default: a one-shot run pays the cost either way.
  bool warm_up{false};
};

class Model final : public std::enable_shared_from_this<Model> {
public:
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;
  Model(Model&&) = delete;
  Model& operator=(Model&&) = delete;

  /// Opens `model_path`, checks the architecture and that this build has the
  /// runtime (RuntimeAvailable), then reads config, tokenizer and weights and
  /// uploads them. nullptr with `error` on any failure.
  [[nodiscard]] static std::shared_ptr<Model> Load(
      const std::string& model_path, const ModelOptions& options,
      std::string* error = nullptr);

  /// The model's only sequence. nullptr (with `error`) while another session
  /// is alive. The session keeps the model alive.
  [[nodiscard]] std::unique_ptr<ForwardSession> CreateSession(
      std::string* error = nullptr);

  [[nodiscard]] const Tokenizer& tokenizer() const noexcept {
    return *tokenizer_;
  }
  [[nodiscard]] const Config& config() const noexcept;
  [[nodiscard]] const StopTokens& stop_tokens() const noexcept {
    return stops_;
  }
  [[nodiscard]] std::uint32_t MaxContext() const noexcept {
    return max_context_;
  }
  [[nodiscard]] std::uint32_t VocabSize() const noexcept;
  /// `general.name`, else kArchitecture.
  [[nodiscard]] std::string ModelName() const { return name_; }
  [[nodiscard]] std::size_t DeviceBytes() const noexcept;
  /// Sessions draft with the NextN (MTP) block (ModelOptions::mtp).
  [[nodiscard]] bool Mtp() const noexcept;

private:
  struct Impl;
  class Session;
  Model();

  std::unique_ptr<Impl> impl_;
  std::unique_ptr<Tokenizer> tokenizer_;
  StopTokens stops_;
  std::uint32_t max_context_{0};
  std::string name_;
  std::atomic<bool> session_alive_{false};
};

#if defined(ENGINE_ENABLE_HIP)
/// `executor` (embedding and head uploaded) as a ForwardSession of
/// `capacity` positions and `vocab` logits. The executor must outlive it.
/// Used by Model and by GPU tests with stand-in operators. With
/// `fused_last_logits` a Forward that wants logits computes the last token's
/// inside the final pass (Executor::ForwardLast); without it (the default)
/// the last token gets its own one-token pass, as before.
/// A nonzero `draft_hidden` (the model's hidden size) enables
/// Draft/Verify/Accept; the executor must have the NextN block and
/// max_verify_tokens >= 1 + max_drafts. DraftChain chains up to
/// `max_drafts` drafts while the previous draft's probability is at least
/// `chain_gate`.
[[nodiscard]] std::unique_ptr<ForwardSession> MakeExecutorSession(
    Executor& executor, std::uint32_t capacity, std::uint32_t vocab,
    bool fused_last_logits = false, std::uint32_t draft_hidden = 0,
    std::uint32_t max_drafts = 1, float chain_gate = 2.0F);
#endif

}  // namespace gufo::models::glm53_flash

#endif  // GUFO_MODELS_GLM53_FLASH_ENGINE_HPP_
