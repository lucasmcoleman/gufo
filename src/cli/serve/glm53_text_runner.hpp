#ifndef GUFO_SERVER_GLM53_TEXT_RUNNER_HPP_
#define GUFO_SERVER_GLM53_TEXT_RUNNER_HPP_

#include <cstdint>
#include <memory>

#include "src/cli/serve/text_model_runner.hpp"

namespace gufo::models::glm53_flash {
class Model;
}

namespace gufo::server {

/// GLM-5.3-Flash behind the common text runner: the model's single sequence
/// as the only request state, its chat template and tool dialect
/// (ToolFormat::kGlm), and reasoning controls mapped by
/// glm53_flash::ResolveChatOptions. Prefill runs in chunks of at most
/// `prefill_chunk` tokens. There are no snapshots: a live sequence is reused
/// only by a prompt that extends it.
[[nodiscard]] std::shared_ptr<TextModelRunner> MakeGlm53TextRunner(
    std::shared_ptr<models::glm53_flash::Model> model,
    std::uint32_t max_context, std::uint32_t prefill_chunk);

}  // namespace gufo::server

#endif  // GUFO_SERVER_GLM53_TEXT_RUNNER_HPP_
