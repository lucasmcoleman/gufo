#ifndef GUFO_MODELS_GLM53_FLASH_GENERATOR_HPP_
#define GUFO_MODELS_GLM53_FLASH_GENERATOR_HPP_

// Generation loop for GLM-5.3-Flash, independent of the device: tokenizer ->
// chat template -> ForwardSession prefill / decode -> gufo's sampler -> stop
// tokens -> ParseResponse. The engine (engine.hpp) binds the GPU Executor as
// the ForwardSession; tests bind a scripted one.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/sampling.hpp"
#include "src/models/glm53_flash/chat_template.hpp"
#include "src/models/glm53_flash/tokenizer.hpp"

namespace gufo::models::glm53_flash {

/// The forward state of one sequence: the executor seam. Forward runs tokens
/// at Position() .. Position()+n-1 and advances; a failed call leaves the
/// state unusable until Reset.
class ForwardSession {
public:
  ForwardSession() = default;
  virtual ~ForwardSession() = default;
  ForwardSession(const ForwardSession&) = delete;
  ForwardSession& operator=(const ForwardSession&) = delete;
  ForwardSession(ForwardSession&&) = delete;
  ForwardSession& operator=(ForwardSession&&) = delete;

  /// With `logits` set, writes the logits of the last token of `tokens`
  /// ([vocab]); without, computes none.
  [[nodiscard]] virtual bool Forward(std::span<const TokenId> tokens,
                                     std::vector<float>* logits,
                                     std::string* error) = 0;
  /// Empties the sequence; the next Forward starts at position 0.
  [[nodiscard]] virtual bool Reset(std::string* error) = 0;
  [[nodiscard]] virtual std::uint32_t Position() const noexcept = 0;
  /// Positions the sequence can hold.
  [[nodiscard]] virtual std::uint32_t Capacity() const noexcept = 0;

  // Speculative decoding with a draft model (GLM-5.3-Flash's NextN block).
  // One cycle: Draft(anchor), Verify({anchor, draft}), Accept(1 or 2), or
  // DraftChain(anchor), Verify({anchor, drafts...}), Accept(1..1 + drafts).
  /// True when Draft / Verify / Accept are available.
  [[nodiscard]] virtual bool CanDraft() const noexcept { return false; }
  /// The draft model's greedy guess for the token after `anchor`, the token
  /// just sampled from the last logits (not yet forwarded).
  [[nodiscard]] virtual bool Draft(TokenId anchor, TokenId* draft,
                                   std::string* error) {
    (void)anchor;
    (void)draft;
    if (error != nullptr)
      *error = "this session has no draft model";
    return false;
  }
  /// Like Draft, then more drafts chained on the previous one while the
  /// session's own gate allows (at least one; `drafts` receives them).
  [[nodiscard]] virtual bool DraftChain(TokenId anchor,
                                        std::vector<TokenId>* drafts,
                                        std::string* error) {
    drafts->assign(1, 0);
    return Draft(anchor, drafts->data(), error);
  }
  /// Forwards `tokens` (the anchor, then drafts) like Forward, returning
  /// every row's logits ([tokens][vocab]); Accept must follow.
  [[nodiscard]] virtual bool Verify(std::span<const TokenId> tokens,
                                    std::vector<float>* logits,
                                    std::string* error) {
    (void)tokens;
    (void)logits;
    if (error != nullptr)
      *error = "this session has no draft model";
    return false;
  }
  /// Keeps the first `keep` tokens of the last Verify; the sequence ends
  /// after them as if only they had been forwarded.
  [[nodiscard]] virtual bool Accept(std::size_t keep, std::string* error) {
    (void)keep;
    if (error != nullptr)
      *error = "this session has no draft model";
    return false;
  }
};

/// Tokens that end an assistant turn: the GGUF eos / eot / eom ids and the
/// role tokens the template opens the next turn with (<|user|>,
/// <|observation|>, <|system|>).
struct StopTokens {
  std::vector<TokenId> ids;
  [[nodiscard]] bool Contains(TokenId token) const noexcept;
};

/// The stop tokens of `tokenizer`'s vocabulary, without duplicates; ids the
/// vocabulary lacks are left out.
[[nodiscard]] StopTokens ResolveStopTokens(const Tokenizer& tokenizer);

struct GenerationOptions {
  std::size_t max_tokens{256};
  sampling::SamplingConfig sampling;
  /// False samples through stop tokens (benchmarks) until max_tokens.
  bool stop_at_eos{true};
};

enum class FinishReason : std::uint8_t {
  kStop,       ///< A stop token was sampled; it is not in the output.
  kLength,     ///< max_tokens were generated.
  kContext,    ///< The session is full.
  kCancelled,  ///< The callback returned false.
};

struct GenerationResult {
  std::vector<TokenId> tokens;  ///< Generated, without the stop token.
  std::string text;             ///< Decoded `tokens` (empty without tokenizer).
  FinishReason finish_reason{FinishReason::kLength};
  std::optional<TokenId> stop_token;
  std::size_t prompt_tokens{0};
  /// Leading prompt tokens the session already held (not prefilled again).
  std::size_t reused_tokens{0};
};

/// Called with every generated token and its decoded bytes (possibly a
/// partial UTF-8 sequence). Returning false stops generation.
using TokenCallback =
    std::function<bool(TokenId token, std::string_view piece)>;

/// A rendered, tokenized chat prompt.
struct ChatPrompt {
  std::vector<TokenId> tokens;
  /// The prompt ends in the generation prompt's `<think>`, so the reply
  /// starts inside reasoning (ParseResponse's `prompt_opened_think`).
  bool opened_think{false};
};

/// RenderChat + EncodeRendered. Nullopt, with `error`, when RenderChat fails
/// or the prompt is empty.
[[nodiscard]] std::optional<ChatPrompt> PrepareChat(
    const Tokenizer& tokenizer, std::span<const ChatMessage> messages,
    std::span<const ChatTool> tools, const ChatTemplateOptions& options,
    std::string* error = nullptr);

/// The part of a response one piece of text added. Reasoning and content are
/// byte suffixes; `tool_calls` are the calls completed by this piece.
struct ChatDelta {
  std::string reasoning;
  std::string content;
  std::vector<ParsedToolCall> tool_calls;
  [[nodiscard]] bool empty() const noexcept {
    return reasoning.empty() && content.empty() && tool_calls.empty();
  }
};

/// Incremental ParseResponse: Push appends generated text and returns what
/// became final, withholding a possible partial tag or an unfinished call;
/// Finish parses the whole text as complete and returns the rest.
class ResponseStream {
public:
  ResponseStream(std::vector<ChatTool> tools, bool prompt_opened_think);

  [[nodiscard]] ChatDelta Push(std::string_view piece);
  [[nodiscard]] ChatDelta Finish();
  /// Everything pushed so far.
  [[nodiscard]] const std::string& text() const noexcept { return text_; }
  /// The latest parse: partial after Push, complete after Finish.
  [[nodiscard]] const ParsedResponse& response() const noexcept {
    return parsed_;
  }

private:
  ChatDelta Advance(ParsedResponse next);

  std::vector<ChatTool> tools_;
  bool prompt_opened_think_;
  std::string text_;
  ParsedResponse parsed_;
  std::string reasoning_sent_;  ///< Reasoning returned so far.
  std::string content_sent_;    ///< Content returned so far.
  std::size_t calls_sent_{0};
};

struct ChatResult {
  GenerationResult generation;
  ParsedResponse response;  ///< ParseResponse of generation.text.
};

/// Called with each non-empty ChatDelta. Returning false stops generation.
using ChatDeltaCallback = std::function<bool(const ChatDelta& delta)>;

/// Drives one ForwardSession. Tracks the tokens the session holds, so a
/// prompt that extends them prefills only the new suffix (the recurrent KDA
/// state cannot rewind, so anything else resets the session first). Not
/// thread-safe.
class Generator {
public:
  /// `tokenizer` may be null: pieces and text are then empty and Chat is
  /// unavailable.
  Generator(ForwardSession& session, StopTokens stops,
            const Tokenizer* tokenizer = nullptr);

  /// Makes the session hold `prompt` with its last token's logits ready,
  /// reusing the held prefix when the prompt extends it. Sets
  /// `*reused_tokens` to the reused length.
  [[nodiscard]] bool Prefill(std::span<const TokenId> prompt,
                             std::string* error,
                             std::size_t* reused_tokens = nullptr);
  /// Appends `tokens` to the session (incremental prefill or a decoded
  /// token). Logits are computed only with `want_logits`.
  [[nodiscard]] bool Extend(std::span<const TokenId> tokens, bool want_logits,
                            std::string* error);
  /// Appends `token`, already Accepted by `sampler`, and its logits. When the
  /// session drafts (ForwardSession::DraftChain), the drafts for the tokens
  /// after it are forwarded with it and kept while `sampler` samples each from
  /// the verified row before it (sample-and-match), at most `max_drafts` and
  /// never a stop when `stop_at_eos`. Kept drafts are Accepted and appended to
  /// `kept`; the first mismatching sample is deferred in `sampler`
  /// (SamplerState::DeferSample), so the next Sample of logits() returns it.
  /// `*drafted` counts the drafts forwarded.
  [[nodiscard]] bool ExtendDrafted(TokenId token,
                                   sampling::SamplerState& sampler,
                                   std::size_t max_drafts, bool stop_at_eos,
                                   std::vector<TokenId>* kept,
                                   std::size_t* drafted, std::string* error);
  /// Empties the session.
  [[nodiscard]] bool Reset(std::string* error);
  /// Forgets the held tokens; the next Prefill or Extend from an empty
  /// history resets the session first.
  void Invalidate() noexcept;

  /// Prefills `prompt`, then samples until a stop token, max_tokens, a full
  /// session or a false callback. The final generated token is not run
  /// through the model (a later Prefill extends with it).
  [[nodiscard]] bool Generate(std::span<const TokenId> prompt,
                              const GenerationOptions& options,
                              const TokenCallback& on_token,
                              GenerationResult* result, std::string* error);

  /// PrepareChat + Generate + ParseResponse, streaming deltas through a
  /// ResponseStream. Needs the tokenizer.
  [[nodiscard]] bool Chat(std::span<const ChatMessage> messages,
                          std::span<const ChatTool> tools,
                          const ChatTemplateOptions& template_options,
                          const GenerationOptions& options,
                          const ChatDeltaCallback& on_delta, ChatResult* result,
                          std::string* error);

  /// The tokens the session holds.
  [[nodiscard]] std::span<const TokenId> tokens() const noexcept {
    return tokens_;
  }
  /// Logits after the last held token; empty when none were computed.
  [[nodiscard]] std::span<const float> logits() const noexcept {
    return logits_valid_ ? std::span<const float>(logits_)
                         : std::span<const float>{};
  }
  [[nodiscard]] bool IsStop(TokenId token) const noexcept {
    return stops_.Contains(token);
  }
  [[nodiscard]] const StopTokens& stop_tokens() const noexcept {
    return stops_;
  }
  [[nodiscard]] ForwardSession& session() noexcept { return session_; }
  [[nodiscard]] std::uint32_t Capacity() const noexcept {
    return session_.Capacity();
  }

private:
  ForwardSession& session_;
  StopTokens stops_;
  const Tokenizer* tokenizer_;
  std::vector<TokenId> tokens_;
  std::vector<float> logits_;
  std::vector<float> rows_;       ///< ExtendDrafted's verified rows.
  std::vector<TokenId> drafts_;   ///< ExtendDrafted's drafts.
  std::vector<TokenId> verify_;   ///< ExtendDrafted's forwarded tokens.
  bool logits_valid_{false};
  bool needs_reset_{false};  ///< After a failure or Invalidate.
};

}  // namespace gufo::models::glm53_flash

#endif  // GUFO_MODELS_GLM53_FLASH_GENERATOR_HPP_
