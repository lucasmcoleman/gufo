#include "src/models/glm53_flash/generator.hpp"

#include <algorithm>
#include <utility>

namespace gufo::models::glm53_flash {

namespace {

bool Fail(std::string* error, std::string message) {
  if (error != nullptr)
    *error = std::move(message);
  return false;
}

/// Appends to `out` the bytes `now` adds to `sent`, and records them. A parse
/// that no longer extends what was returned adds nothing.
void AppendSuffix(const std::string& now, std::string& sent, std::string& out) {
  if (now.size() <= sent.size() || !now.starts_with(sent))
    return;
  out.append(now, sent.size());
  sent = now;
}

}  // namespace

bool StopTokens::Contains(TokenId token) const noexcept {
  return std::ranges::find(ids, token) != ids.end();
}

StopTokens ResolveStopTokens(const Tokenizer& tokenizer) {
  const SpecialIds& special = tokenizer.special_ids();
  const ChatTokens& chat = tokenizer.chat_tokens();
  StopTokens stops;
  for (const TokenId id : {special.eos, special.eot, special.eom, chat.user,
                           chat.observation, chat.system}) {
    if (id != kInvalidTokenId && id < tokenizer.VocabSize() &&
        !stops.Contains(id))
      stops.ids.push_back(id);
  }
  return stops;
}

std::optional<ChatPrompt> PrepareChat(const Tokenizer& tokenizer,
                                      std::span<const ChatMessage> messages,
                                      std::span<const ChatTool> tools,
                                      const ChatTemplateOptions& options,
                                      std::string* error) {
  std::vector<ContentSpan> spans;
  const auto rendered = RenderChat(messages, tools, options, error, &spans);
  if (!rendered)
    return std::nullopt;
  ChatPrompt prompt;
  prompt.tokens = EncodeRendered(tokenizer, *rendered, spans);
  if (prompt.tokens.empty()) {
    Fail(error, "GLM-5.3-Flash chat prompt produced no tokens");
    return std::nullopt;
  }
  prompt.opened_think =
      options.add_generation_prompt && options.enable_thinking;
  return prompt;
}

// ---- ResponseStream ----

ResponseStream::ResponseStream(std::vector<ChatTool> tools,
                               bool prompt_opened_think)
    : tools_(std::move(tools)), prompt_opened_think_(prompt_opened_think) {}

ChatDelta ResponseStream::Push(std::string_view piece) {
  text_.append(piece);
  return Advance(ParseResponse(text_, tools_, prompt_opened_think_,
                               /*partial=*/true));
}

ChatDelta ResponseStream::Finish() {
  return Advance(ParseResponse(text_, tools_, prompt_opened_think_,
                               /*partial=*/false));
}

ChatDelta ResponseStream::Advance(ParsedResponse next) {
  ChatDelta delta;
  AppendSuffix(next.reasoning, reasoning_sent_, delta.reasoning);
  AppendSuffix(next.content, content_sent_, delta.content);
  for (std::size_t i = calls_sent_; i < next.tool_calls.size(); ++i)
    delta.tool_calls.push_back(next.tool_calls[i]);
  calls_sent_ = std::max(calls_sent_, next.tool_calls.size());
  parsed_ = std::move(next);
  return delta;
}

// ---- Generator ----

Generator::Generator(ForwardSession& session, StopTokens stops,
                     const Tokenizer* tokenizer)
    : session_(session), stops_(std::move(stops)), tokenizer_(tokenizer) {}

void Generator::Invalidate() noexcept {
  tokens_.clear();
  logits_valid_ = false;
  needs_reset_ = true;
}

bool Generator::Reset(std::string* error) {
  tokens_.clear();
  logits_valid_ = false;
  if (!session_.Reset(error)) {
    needs_reset_ = true;
    return false;
  }
  needs_reset_ = false;
  return true;
}

bool Generator::Extend(std::span<const TokenId> tokens, bool want_logits,
                       std::string* error) {
  if (tokens.empty())
    return Fail(error, "GLM-5.3-Flash forward needs at least one token");
  if ((needs_reset_ || (tokens_.empty() && session_.Position() != 0)) &&
      !Reset(error))
    return false;
  if (session_.Position() != tokens_.size()) {
    Invalidate();
    return Fail(error, "GLM-5.3-Flash session position " +
                           std::to_string(session_.Position()) +
                           " does not match its " +
                           std::to_string(tokens_.size()) + " held tokens");
  }
  if (tokens.size() > session_.Capacity() - tokens_.size()) {
    return Fail(error, "GLM-5.3-Flash context of " +
                           std::to_string(session_.Capacity()) +
                           " tokens is full");
  }
  logits_valid_ = false;
  if (!session_.Forward(tokens, want_logits ? &logits_ : nullptr, error)) {
    Invalidate();
    return false;
  }
  tokens_.insert(tokens_.end(), tokens.begin(), tokens.end());
  logits_valid_ = want_logits && !logits_.empty();
  if (want_logits && !logits_valid_)
    return Fail(error, "GLM-5.3-Flash forward returned no logits");
  return true;
}

bool Generator::Prefill(std::span<const TokenId> prompt, std::string* error,
                        std::size_t* reused_tokens) {
  if (reused_tokens != nullptr)
    *reused_tokens = 0;
  if (prompt.empty())
    return Fail(error, "GLM-5.3-Flash prompt is empty");
  if (prompt.size() > session_.Capacity()) {
    return Fail(error, "GLM-5.3-Flash prompt of " +
                           std::to_string(prompt.size()) +
                           " tokens exceeds the context of " +
                           std::to_string(session_.Capacity()));
  }
  const std::size_t held = tokens_.size();
  const bool extends = !needs_reset_ && held > 0 && held <= prompt.size() &&
                       session_.Position() == held &&
                       std::ranges::equal(tokens_, prompt.first(held));
  if (extends && held == prompt.size() && logits_valid_) {
    if (reused_tokens != nullptr)
      *reused_tokens = held;
    return true;
  }
  // The recurrent state cannot rewind, and an identical prompt without its
  // logits needs its last token again: start over.
  if (!extends || held == prompt.size()) {
    if (!Reset(error))
      return false;
    return Extend(prompt, /*want_logits=*/true, error);
  }
  if (!Extend(prompt.subspan(held), /*want_logits=*/true, error))
    return false;
  if (reused_tokens != nullptr)
    *reused_tokens = held;
  return true;
}

bool Generator::Generate(std::span<const TokenId> prompt,
                         const GenerationOptions& options,
                         const TokenCallback& on_token,
                         GenerationResult* result, std::string* error) {
  if (result == nullptr)
    return Fail(error, "GLM-5.3-Flash generation needs a result");
  *result = {};
  result->prompt_tokens = prompt.size();
  if (!Prefill(prompt, error, &result->reused_tokens))
    return false;
  sampling::SamplerState sampler(options.sampling, prompt);
  result->finish_reason = FinishReason::kLength;
  // Emits `token` (already Accepted by the sampler); false ends generation
  // with result->finish_reason set.
  const auto emit = [&](TokenId token) {
    result->tokens.push_back(token);
    const std::string piece =
        tokenizer_ != nullptr ? tokenizer_->DecodeToken(token) : std::string{};
    result->text += piece;
    if (on_token && !on_token(token, piece)) {
      result->finish_reason = FinishReason::kCancelled;
      return false;
    }
    return true;
  };
  // ExtendDrafted defers a mismatching draft's sample in `sampler`, so the
  // next Sample returns it.
  std::vector<TokenId> kept;
  for (std::size_t n = 0; n < options.max_tokens; ++n) {
    const auto frontier = logits();
    if (frontier.empty())
      return Fail(error, "GLM-5.3-Flash has no logits to sample");
    const TokenId token = sampler.Sample(frontier);
    if (options.stop_at_eos && IsStop(token)) {
      result->finish_reason = FinishReason::kStop;
      result->stop_token = token;
      return true;
    }
    sampler.Accept(token);
    if (!emit(token))
      return true;
    if (n + 1 == options.max_tokens)
      break;
    if (tokens_.size() >= session_.Capacity()) {
      result->finish_reason = FinishReason::kContext;
      return true;
    }
    std::size_t drafted = 0;
    if (!ExtendDrafted(token, sampler, options.max_tokens - n - 1,
                       options.stop_at_eos, &kept, &drafted, error))
      return false;
    for (const TokenId draft : kept) {
      ++n;
      if (!emit(draft))
        return true;
    }
  }
  return true;
}

bool Generator::ExtendDrafted(TokenId token, sampling::SamplerState& sampler,
                              std::size_t max_drafts, bool stop_at_eos,
                              std::vector<TokenId>* kept, std::size_t* drafted,
                              std::string* error) {
  kept->clear();
  *drafted = 0;
  if (max_drafts == 0 || !session_.CanDraft() ||
      tokens_.size() + 2 > session_.Capacity() || needs_reset_ ||
      session_.Position() != tokens_.size())
    return Extend(std::span<const TokenId>(&token, 1), /*want_logits=*/true,
                  error);
  if (!session_.DraftChain(token, &drafts_, error)) {
    Invalidate();
    return false;
  }
  if (drafts_.empty()) {
    Invalidate();
    return Fail(error, "GLM-5.3-Flash drafted nothing");
  }
  *drafted = drafts_.size();
  verify_.assign(1, token);
  verify_.insert(verify_.end(), drafts_.begin(), drafts_.end());
  logits_valid_ = false;
  if (!session_.Verify(verify_, &rows_, error)) {
    Invalidate();
    return false;
  }
  const std::size_t vocab = rows_.size() / verify_.size();
  const auto row = [&](std::size_t i) {
    return std::span<const float>(rows_.data() + (i * vocab), vocab);
  };
  // A stop is never forwarded; the next Sample reports it.
  const std::size_t limit = std::min(drafts_.size(), max_drafts);
  std::size_t count = 0;
  while (count < limit) {
    const TokenId next = sampler.Sample(row(count));
    if (next != drafts_[count] || (stop_at_eos && IsStop(next))) {
      sampler.DeferSample(next);
      break;
    }
    sampler.Accept(next);
    ++count;
  }
  if (!session_.Accept(count + 1, error)) {
    Invalidate();
    return false;
  }
  tokens_.push_back(token);
  kept->assign(drafts_.begin(),
               drafts_.begin() + static_cast<std::ptrdiff_t>(count));
  tokens_.insert(tokens_.end(), kept->begin(), kept->end());
  const auto frontier = row(count);
  logits_.assign(frontier.begin(), frontier.end());
  logits_valid_ = true;
  return true;
}

bool Generator::Chat(std::span<const ChatMessage> messages,
                     std::span<const ChatTool> tools,
                     const ChatTemplateOptions& template_options,
                     const GenerationOptions& options,
                     const ChatDeltaCallback& on_delta, ChatResult* result,
                     std::string* error) {
  if (result == nullptr)
    return Fail(error, "GLM-5.3-Flash chat needs a result");
  *result = {};
  if (tokenizer_ == nullptr)
    return Fail(error, "GLM-5.3-Flash chat needs the tokenizer");
  const auto prompt =
      PrepareChat(*tokenizer_, messages, tools, template_options, error);
  if (!prompt)
    return false;
  ResponseStream stream({tools.begin(), tools.end()}, prompt->opened_think);
  const TokenCallback forward_delta = [&](TokenId, std::string_view piece) {
    const ChatDelta delta = stream.Push(piece);
    return !on_delta || delta.empty() || on_delta(delta);
  };
  if (!Generate(prompt->tokens, options, forward_delta, &result->generation,
                error))
    return false;
  const ChatDelta tail = stream.Finish();
  if (on_delta && !tail.empty() &&
      result->generation.finish_reason != FinishReason::kCancelled)
    (void)on_delta(tail);
  result->response = stream.response();
  return true;
}

}  // namespace gufo::models::glm53_flash
