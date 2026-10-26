// Laguna DFlash speculative decoding (phase 6e). See speculative.hpp.
#include "src/models/laguna/speculative.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

#include "src/models/laguna/engine.hpp"
#include "src/models/laguna/kernels/dflash_session.hpp"

namespace gufo::models::laguna {
namespace {

/// Rows the vector path takes in one verify.
constexpr std::size_t kMaxVerifyRows = 32;
/// Laguna's EOS (BOS) id; EOT 24 is a stop token through IsStopToken.
constexpr tokenization::TokenId kEos = 2;

}  // namespace

// ---- LagunaSpeculativeTarget ----------------------------------------------

LagunaSpeculativeTarget::LagunaSpeculativeTarget(const Model& model,
                                                 rocm::DFlashSession& session)
    : model_(model), session_(session), vocab_(model.VocabSize()) {}

void LagunaSpeculativeTarget::Reset() noexcept {
  try {
    (void)session_.Reset();
  } catch (...) {
  }
  logits_.clear();
  tokens_.clear();
  begin_ = 0;
  saved_.reset();
}

void LagunaSpeculativeTarget::Run(std::span<const tokenization::TokenId> tokens,
                                  std::uint32_t pos) {
  if (pos != session_.position()) {
    throw std::logic_error("laguna speculative target: call at position " +
                           std::to_string(pos) + " but the session is at " +
                           std::to_string(session_.position()));
  }
  if (tokens.empty() || tokens.size() > kMaxVerifyRows) {
    throw std::invalid_argument(
        "laguna speculative target: a verify takes 1 to 32 rows");
  }
  std::vector<std::int32_t> ids;
  ids.reserve(tokens.size());
  for (const auto t : tokens) {
    if (t >= vocab_) {
      throw std::invalid_argument(
          "laguna speculative target: token outside the vocabulary");
    }
    ids.push_back(static_cast<std::int32_t>(t));
  }
  std::vector<float> logits(tokens.size() * vocab_);
  std::string error;
  if (!session_.Verify(ids, logits.data(), &error)) {
    throw std::runtime_error("laguna speculative verify failed: " + error);
  }
  logits_ = std::move(logits);
  tokens_ = std::move(ids);
  begin_ = pos;
}

tokenization::TokenId LagunaSpeculativeTarget::Argmax(std::size_t row) const {
  // The greedy sampler's own selection (first maximum over finite logits),
  // so a verify predicts exactly what plain greedy decode would select.
  sampling::SamplerState greedy;
  return greedy.Sample(
      std::span<const float>(logits_).subspan(row * vocab_, vocab_));
}

tokenization::TokenId LagunaSpeculativeTarget::ForwardPromptBatch(
    std::span<const tokenization::TokenId> prompt_tokens) {
  if (session_.position() != 0) {
    throw std::logic_error(
        "laguna speculative target: prompt batch on a used session");
  }
  return ForwardPromptSuffix(prompt_tokens, 0, true);
}

tokenization::TokenId LagunaSpeculativeTarget::ForwardPromptSuffix(
    std::span<const tokenization::TokenId> tokens, std::uint32_t position,
    bool compute_logits) {
  (void)compute_logits;
  if (position != session_.position() || tokens.empty()) {
    throw std::logic_error(
        "laguna speculative target: prompt at the wrong position");
  }
  std::vector<std::int32_t> ids;
  ids.reserve(tokens.size());
  for (const auto t : tokens) {
    if (t >= vocab_) {
      throw std::invalid_argument(
          "laguna speculative target: token outside the vocabulary");
    }
    ids.push_back(static_cast<std::int32_t>(t));
  }
  std::vector<float> last(vocab_);
  std::string error;
  if (!session_.Prefill(ids, rocm::LogitsRows::kLast, last.data(), &error)) {
    throw std::runtime_error("laguna speculative prefill failed: " + error);
  }
  logits_ = std::move(last);
  tokens_.assign(1, ids.back());
  begin_ = session_.position() - 1;
  return Argmax(0);
}

tokenization::TokenId LagunaSpeculativeTarget::ForwardToken(
    tokenization::TokenId token_id, std::uint32_t pos, bool compute_logits) {
  (void)compute_logits;
  Run(std::span<const tokenization::TokenId>(&token_id, 1), pos);
  return Argmax(0);
}

void LagunaSpeculativeTarget::SaveState(std::uint32_t valid_context) {
  if (valid_context != session_.position()) {
    throw std::logic_error(
        "laguna speculative target: verification saved at another position");
  }
  saved_ = valid_context;
}

void LagunaSpeculativeTarget::RestoreState() {
  // Nothing to restore on the device: the commit that follows is a rewind of
  // the verified rows (DFLASH rule 2).
  if (!saved_.has_value()) {
    throw std::logic_error(
        "laguna speculative target: restore without a saved verification");
  }
}

void LagunaSpeculativeTarget::FinishVerification() {
  // The shared verifier closes a verification before it restores and
  // commits a partly rejected chunk, so the saved position stays until the
  // commit (or the next SaveState) consumes it.
}

std::span<const float> LagunaSpeculativeTarget::CopyLastLogits() {
  return FrontierLogits();
}

std::span<const float> LagunaSpeculativeTarget::CopyVerificationLogits(
    std::size_t row) {
  if (row >= tokens_.size()) {
    return {};
  }
  return std::span<const float>(logits_).subspan(row * vocab_, vocab_);
}

std::size_t LagunaSpeculativeTarget::VocabularySize() const noexcept {
  return vocab_;
}

speculative::VerificationChunkResult
LagunaSpeculativeTarget::ForwardVerificationChunk(
    std::span<const tokenization::TokenId> candidate_tokens,
    std::uint32_t start_pos, bool capture_hidden, bool capture_logits) {
  if (capture_hidden) {
    throw std::logic_error(
        "laguna speculative target: hidden states stay on the device");
  }
  Run(candidate_tokens, start_pos);
  speculative::VerificationChunkResult result;
  result.predictions.reserve(candidate_tokens.size());
  for (std::size_t row = 0; row < candidate_tokens.size(); ++row) {
    result.predictions.push_back(Argmax(row));
  }
  if (capture_logits) {
    result.logits = logits_;
    result.vocab_size = vocab_;
  }
  return result;
}

void LagunaSpeculativeTarget::CommitVerificationChunk(
    std::span<const tokenization::TokenId> committed_tokens,
    std::uint32_t start_pos) {
  // Only a prefix of the rows just verified, from where they started, can be
  // kept: anything else would keep K/V of rows that were never verified.
  const bool prefix =
      committed_tokens.size() <= tokens_.size() &&
      std::equal(committed_tokens.begin(), committed_tokens.end(),
                 tokens_.begin(), [](tokenization::TokenId a, std::int32_t b) {
                   return a == static_cast<tokenization::TokenId>(b);
                 });
  if (!saved_.has_value() || *saved_ != start_pos || start_pos != begin_ ||
      !prefix || session_.position() != begin_ + tokens_.size()) {
    throw std::logic_error(
        "laguna speculative target: commit is not a prefix of the verified "
        "rows");
  }
  std::string error;
  if (!session_.Commit(
          start_pos + static_cast<std::uint32_t>(committed_tokens.size()),
          &error)) {
    throw std::runtime_error("laguna speculative commit failed: " + error);
  }
  saved_.reset();
}

tokenization::TokenId LagunaSpeculativeTarget::GetEosTokenId() const noexcept {
  return kEos;
}

bool LagunaSpeculativeTarget::IsStopToken(
    tokenization::TokenId token) const noexcept {
  return token <= static_cast<tokenization::TokenId>(
                      std::numeric_limits<std::int32_t>::max()) &&
         model_.IsStopToken(static_cast<std::int32_t>(token));
}

std::string_view LagunaSpeculativeTarget::DecodeToken(
    tokenization::TokenId token_id) const noexcept {
  try {
    piece_ = model_.DecodeToken(static_cast<std::int32_t>(token_id));
  } catch (...) {
    piece_.clear();
  }
  return piece_;
}

std::span<const float> LagunaSpeculativeTarget::FrontierLogits() const {
  const std::uint32_t pos = session_.position();
  if (pos <= begin_ || pos - 1 - begin_ >= tokens_.size()) {
    return {};
  }
  return std::span<const float>(logits_).subspan(
      std::size_t{pos - 1 - begin_} * vocab_, vocab_);
}

// ---- LagunaDraftBackend -------------------------------------------------

LagunaDraftBackend::LagunaDraftBackend(rocm::DFlashSession& session)
    : session_(session) {}

speculative::DraftProposal LagunaDraftBackend::Propose(
    std::span<const tokenization::TokenId> prompt_tokens,
    std::uint32_t current_pos, std::uint32_t max_tokens) {
  speculative::DraftProposal proposal;
  proposal.start_pos = current_pos;
  if (prompt_tokens.empty() || max_tokens == 0) {
    return proposal;
  }
  if (current_pos != session_.position()) {
    throw std::logic_error(
        "laguna draft backend: anchor is not at the session's position");
  }
  const auto anchor = static_cast<std::int32_t>(prompt_tokens.back());
  std::vector<std::int32_t> drafts;
  if (override_) {
    drafts = override_(current_pos, anchor, max_tokens);
    drafts.resize(std::min<std::size_t>(drafts.size(), max_tokens));
  } else {
    if (!session_.capturing()) {
      return proposal;
    }
    std::string error;
    // A failure drops the drafter inside the session; this step and the
    // following ones are target-only.
    if (!session_.Propose(anchor, max_tokens + 1, &drafts, nullptr, &error)) {
      return proposal;
    }
  }
  proposal.tokens.reserve(drafts.size());
  for (const std::int32_t d : drafts) {
    proposal.tokens.push_back(static_cast<tokenization::TokenId>(d));
  }
  return proposal;
}

// ---- SpeculativeDecoder ---------------------------------------------------

SpeculativeDecoder::SpeculativeDecoder(const Model& model,
                                       rocm::DFlashSession& session,
                                       std::uint32_t block_rows)
    : model_(model), session_(session), block_rows_(0) {
  set_block_rows(block_rows);
  target_ = std::make_unique<LagunaSpeculativeTarget>(model, session);
  auto backend = std::make_unique<LagunaDraftBackend>(session);
  backend_ = backend.get();
  speculative::SpeculativeOptions options;
  // The block is bounded per step by block_rows_ (Step's budget); the
  // verifier's own adaptive length (tuned on the dense Qwen 27B) is off.
  options.max_draft_tokens = 15;
  options.min_draft_tokens = 1;
  options.initial_draft_tokens = 15;
  options.enable_adaptive_draft_length = false;
  options.use_batched_verification = true;
  options.retain_frontier_logits = false;
  options.draft_deterministic_sampled = true;
  verifier_ = std::make_unique<speculative::SpeculativeVerifier>(
      *target_, std::move(backend), options);
}

SpeculativeDecoder::~SpeculativeDecoder() = default;

void SpeculativeDecoder::set_block_rows(std::uint32_t rows) {
  if (rows < 2 || rows > 16) {
    throw std::invalid_argument(
        "laguna speculative block must have 2 to 16 rows");
  }
  block_rows_ = rows;
}

const speculative::SpeculativeStats& SpeculativeDecoder::stats()
    const noexcept {
  return verifier_->GetStats();
}

void SpeculativeDecoder::Step(std::span<const float> frontier,
                              sampling::SamplerState& sampler,
                              std::uint32_t max_tokens,
                              std::uint32_t context_limit, bool stop_at_eos,
                              SpeculativeStepResult* out,
                              std::vector<float>* out_frontier) {
  if (out == nullptr || out_frontier == nullptr || max_tokens == 0) {
    throw std::invalid_argument("laguna speculative step: bad arguments");
  }
  if (sampler.config().uses_random_sampling()) {
    throw std::invalid_argument(
        "laguna speculative step: random sampling does not speculate");
  }
  *out = {};
  const std::uint32_t P = session_.position();
  if (P >= std::min(context_limit, session_.target().context())) {
    out->stop = true;
    return;
  }
  if (frontier.size() != model_.VocabSize()) {
    throw std::logic_error("laguna speculative step: no frontier logits");
  }
  // The frontier token, exactly as plain decode selects it.
  sampling::SamplerState work = sampler;
  const auto t0 = work.Sample(frontier);
  if (stop_at_eos && model_.IsStopToken(static_cast<std::int32_t>(t0))) {
    out->stop = true;
    return;
  }
  work.Accept(t0);
  const std::uint32_t room =
      std::min(context_limit, session_.target().context()) - P;
  const std::uint32_t budget = std::min({max_tokens, block_rows_, room});
  const tokenization::TokenId anchor = t0;
  auto step =
      verifier_->VerifyStep(std::span<const tokenization::TokenId>(&anchor, 1),
                            P, t0, kEos, budget, work);
  const auto accepted = static_cast<std::uint32_t>(step.accepted_count);
  if (accepted > step.emitted_tokens.size() ||
      session_.position() != P + 1 + accepted) {
    throw std::logic_error(
        "laguna speculative step: the session did not commit the accepted "
        "rows");
  }
  out->tokens.reserve(accepted + 1);
  out->tokens.push_back(static_cast<std::int32_t>(t0));
  for (std::uint32_t i = 0; i < accepted; ++i) {
    out->tokens.push_back(static_cast<std::int32_t>(step.emitted_tokens[i]));
  }
  out->drafted = static_cast<std::uint32_t>(step.draft_count);
  out->accepted = accepted;
  const auto row = target_->FrontierLogits();
  if (row.size() != model_.VocabSize()) {
    throw std::logic_error("laguna speculative step: frontier row missing");
  }
  out_frontier->assign(row.begin(), row.end());
}

}  // namespace gufo::models::laguna
