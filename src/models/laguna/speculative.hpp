#ifndef GUFO_MODELS_LAGUNA_SPECULATIVE_HPP_
#define GUFO_MODELS_LAGUNA_SPECULATIVE_HPP_

// Laguna-S-2.1 DFlash speculative decoding (phase 6e of
// docs/models/laguna-s-2.1/DFLASH.md): the shared SpeculativeVerifier over one
// session's DFlashSession.
//
//  * LagunaSpeculativeTarget is the ISpeculativeTargetExecutor adapter. A
//    verify chunk runs on the pinned vector path (DFlashSession::Verify), so
//    its rows equal the same tokens decoded alone bit for bit; the commit of
//    a partly rejected chunk is a rewind of target and drafter
//    (DFlashSession::Commit, DFLASH rules 1 and 2), never a replay. The
//    target predictions are the greedy sampler's own argmax of each row, so
//    ties break exactly as plain decode breaks them.
//  * LagunaDraftBackend proposes the drafter's block (batched path) after the
//    anchor. Greedy proposals only: a request that samples at random does not
//    speculate (the caller decodes it plainly).
//  * SpeculativeDecoder runs one step from a session's frontier: it selects
//    the token at the frontier with the request's sampler (as plain decode
//    does), verifies it with up to block_rows - 1 drafts, and returns that
//    token and the accepted drafts, all of them consumed by the target. The
//    correction token is not emitted: the new frontier logits are the row
//    that predicted it, so the next step (or a plain decode, a snapshot, a
//    cancelled request) selects it again exactly as plain decode would. The
//    session state after a step is therefore the state plain decode reaches
//    after the same tokens. Constraints, penalties and reasoning budgets run
//    through the draft prefix in the verifier (deterministic requests only).

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/sampling.hpp"
#include "src/core/speculative/speculative_verifier.hpp"

namespace gufo::models::laguna {

class Model;

namespace rocm {
class DFlashSession;
}

/// ISpeculativeTargetExecutor over one DFlashSession (see the file comment).
/// Calls whose position is not the session's position (a wrong anchor) and
/// commits that are not a prefix of the last verified chunk are refused with
/// std::logic_error, touching nothing; a target failure throws
/// std::runtime_error with the executor poisoned.
class LagunaSpeculativeTarget final
    : public speculative::ISpeculativeTargetExecutor {
public:
  LagunaSpeculativeTarget(const Model& model, rocm::DFlashSession& session);

  void Reset() noexcept override;
  [[nodiscard]] tokenization::TokenId ForwardPromptBatch(
      std::span<const tokenization::TokenId> prompt_tokens) override;
  [[nodiscard]] tokenization::TokenId ForwardPromptSuffix(
      std::span<const tokenization::TokenId> tokens, std::uint32_t position,
      bool compute_logits) override;
  [[nodiscard]] tokenization::TokenId ForwardToken(
      tokenization::TokenId token_id, std::uint32_t pos,
      bool compute_logits) override;
  void SaveState(std::uint32_t valid_context) override;
  void RestoreState() override;
  void FinishVerification() override;
  [[nodiscard]] std::span<const float> CopyLastLogits() override;
  [[nodiscard]] std::span<const float> CopyVerificationLogits(
      std::size_t row) override;
  [[nodiscard]] std::size_t VocabularySize() const noexcept override;
  [[nodiscard]] speculative::VerificationChunkResult ForwardVerificationChunk(
      std::span<const tokenization::TokenId> candidate_tokens,
      std::uint32_t start_pos, bool capture_hidden,
      bool capture_logits) override;
  void CommitVerificationChunk(
      std::span<const tokenization::TokenId> committed_tokens,
      std::uint32_t start_pos) override;
  [[nodiscard]] tokenization::TokenId GetEosTokenId() const noexcept override;
  [[nodiscard]] bool IsStopToken(
      tokenization::TokenId token) const noexcept override;
  [[nodiscard]] std::string_view DecodeToken(
      tokenization::TokenId token_id) const noexcept override;

  /// The logits of the token at position() - 1 (the session's frontier) from
  /// the last call, or empty when the last call did not cover it.
  [[nodiscard]] std::span<const float> FrontierLogits() const;

private:
  /// Runs `tokens` at the session's position on the vector path, keeping all
  /// rows' logits. Throws (see the class comment).
  void Run(std::span<const tokenization::TokenId> tokens, std::uint32_t pos);
  [[nodiscard]] tokenization::TokenId Argmax(std::size_t row) const;

  const Model& model_;
  rocm::DFlashSession& session_;
  std::size_t vocab_{0};
  std::vector<float> logits_;           ///< [rows_][vocab] of the last call
  std::vector<std::int32_t> tokens_;    ///< inputs of the last call
  std::uint32_t begin_{0};              ///< position of its first row
  std::optional<std::uint32_t> saved_;  ///< SaveState's position
  mutable std::string piece_;
};

/// IDraftBackend over the session's drafter. Propose reads only the anchor,
/// the last token of the sequence it is given, which must sit at the
/// session's position. A drafter that is dropped or suspended proposes
/// nothing (the verifier then takes a target-only step).
class LagunaDraftBackend final : public speculative::IDraftBackend {
public:
  explicit LagunaDraftBackend(rocm::DFlashSession& session);

  [[nodiscard]] std::string_view Name() const noexcept override {
    return "laguna-dflash";
  }
  [[nodiscard]] speculative::DraftProposal Propose(
      std::span<const tokenization::TokenId> prompt_tokens,
      std::uint32_t current_pos, std::uint32_t max_tokens) override;

  /// Test seam: replaces the drafter's proposals (position, anchor, maximum
  /// count) -> drafts, at most maximum. The drafter is not run.
  using DraftOverride = std::function<std::vector<std::int32_t>(
      std::uint32_t position, std::int32_t anchor, std::uint32_t max_tokens)>;
  void SetDraftOverrideForTesting(DraftOverride override_fn) {
    override_ = std::move(override_fn);
  }

private:
  rocm::DFlashSession& session_;
  DraftOverride override_;
};

/// What one SpeculativeDecoder::Step emitted.
struct SpeculativeStepResult {
  /// The frontier token and the accepted drafts, in order, all consumed by
  /// the target (the session's position advanced by tokens.size()).
  std::vector<std::int32_t> tokens;
  std::uint32_t drafted{0};
  std::uint32_t accepted{0};
  /// The frontier token is a stop token (stop_at_eos) or the context limit
  /// was reached: nothing was emitted or run.
  bool stop{false};
};

/// One session's speculative decoder (see the file comment). Not thread
/// safe; one per session.
class SpeculativeDecoder {
public:
  /// `block_rows` (2..16): rows per verify, the anchor plus block_rows - 1
  /// drafts.
  SpeculativeDecoder(const Model& model, rocm::DFlashSession& session,
                     std::uint32_t block_rows);
  ~SpeculativeDecoder();
  SpeculativeDecoder(const SpeculativeDecoder&) = delete;
  SpeculativeDecoder& operator=(const SpeculativeDecoder&) = delete;

  /// One step from `frontier` (the logits of the session's last token) at
  /// the session's position: selects the frontier token with a copy of
  /// `sampler` (the caller's sampler is not advanced; its RNG is untouched
  /// since only deterministic requests speculate), then verifies it with
  /// drafts so that at most `max_tokens` tokens are emitted and the position
  /// stays within `context_limit`. `out_frontier` receives the new frontier
  /// logits. Throws std::invalid_argument for a random-sampling request or a
  /// bad budget, and std::runtime_error / std::logic_error on a failure, after
  /// which the session's state is undefined (reset it).
  void Step(std::span<const float> frontier, sampling::SamplerState& sampler,
            std::uint32_t max_tokens, std::uint32_t context_limit,
            bool stop_at_eos, SpeculativeStepResult* out,
            std::vector<float>* out_frontier);

  [[nodiscard]] std::uint32_t block_rows() const noexcept {
    return block_rows_;
  }
  void set_block_rows(std::uint32_t rows);
  /// Owned by the verifier.
  [[nodiscard]] LagunaDraftBackend& backend() noexcept { return *backend_; }
  [[nodiscard]] LagunaSpeculativeTarget& target() noexcept { return *target_; }
  [[nodiscard]] const speculative::SpeculativeStats& stats() const noexcept;

private:
  const Model& model_;
  rocm::DFlashSession& session_;
  std::uint32_t block_rows_;
  std::unique_ptr<LagunaSpeculativeTarget> target_;
  LagunaDraftBackend* backend_{nullptr};
  std::unique_ptr<speculative::SpeculativeVerifier> verifier_;
};

}  // namespace gufo::models::laguna

#endif  // GUFO_MODELS_LAGUNA_SPECULATIVE_HPP_
