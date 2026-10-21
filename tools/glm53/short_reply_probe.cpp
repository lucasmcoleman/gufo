// Short-reply latency probe for GLM-5.3-Flash: one model load, the same chat
// prompt generated `reps` times (session reset each time), timing prefill,
// the first token and the rest of decode separately. Run once with
// GUFO_GLM53_MTP=0 and once without to tell a one-time warm-up cost from a
// per-request one. PROBE_WARM_UP=1 loads with ModelOptions::warm_up.
//
// usage: glm53_flash_short_reply_probe <model.gguf> <prompt text> [reps] [max_tokens]

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "src/models/glm53_flash/chat_template.hpp"
#include "src/models/glm53_flash/engine.hpp"
#include "src/models/glm53_flash/generator.hpp"

namespace glm = gufo::models::glm53_flash;
using Clock = std::chrono::steady_clock;

namespace {
double Since(Clock::time_point t) {
  return std::chrono::duration<double>(Clock::now() - t).count();
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: " << argv[0]
              << " <model.gguf> <prompt text> [reps] [max_tokens]\n";
    return 2;
  }
  const int reps = argc > 3 ? std::atoi(argv[3]) : 4;
  const std::size_t max_tokens = argc > 4 ? std::strtoul(argv[4], nullptr, 10) : 256;
  std::string error;
  const auto load0 = Clock::now();
  const char* warm = std::getenv("PROBE_WARM_UP");
  auto model = glm::Model::Load(
      argv[1],
      glm::ModelOptions{.max_context = 4096,
                        .warm_up = warm != nullptr && *warm == '1'},
      &error);
  if (model == nullptr) {
    std::cerr << "load: " << error << '\n';
    return 1;
  }
  auto session = model->CreateSession(&error);
  if (session == nullptr) {
    std::cerr << "session: " << error << '\n';
    return 1;
  }
  std::cout << std::fixed << std::setprecision(3) << "load " << Since(load0)
            << " s, MTP " << (std::getenv("GUFO_GLM53_MTP") ? std::getenv("GUFO_GLM53_MTP") : "default")
            << ", warm-up " << (std::getenv("PROBE_WARM_UP") ? std::getenv("PROBE_WARM_UP") : "0")
            << '\n';
  glm::Generator generator(*session, model->stop_tokens(), &model->tokenizer());
  std::vector<glm::ChatMessage> messages(1);
  messages[0].role = glm::ChatRole::kUser;
  messages[0].content = argv[2];
  glm::ChatTemplateOptions chat;
  chat.enable_thinking = false;
  const auto prompt = glm::PrepareChat(model->tokenizer(), messages, {}, chat, &error);
  if (!prompt) {
    std::cerr << "template: " << error << '\n';
    return 1;
  }
  glm::GenerationOptions options;
  options.max_tokens = max_tokens;
  for (int r = 1; r <= reps; ++r) {
    if (!generator.Reset(&error)) {
      std::cerr << "reset: " << error << '\n';
      return 1;
    }
    const auto t0 = Clock::now();
    if (!generator.Prefill(prompt->tokens, &error)) {
      std::cerr << "prefill: " << error << '\n';
      return 1;
    }
    const double prefill_s = Since(t0);
    const auto t1 = Clock::now();
    double first_s = -1.0;
    glm::GenerationResult result;
    // Generate reuses the prefilled prompt (reused_tokens == prompt size).
    if (!generator.Generate(
            prompt->tokens, options,
            [&](glm::TokenId, std::string_view) {
              if (first_s < 0.0)
                first_s = Since(t1);
              return true;
            },
            &result, &error)) {
      std::cerr << "generate: " << error << '\n';
      return 1;
    }
    const double gen_s = Since(t1);
    const auto n = result.tokens.size();
    std::cout << "rep " << r << ": prompt " << prompt->tokens.size()
              << " tok, prefill " << prefill_s << " s, first token "
              << first_s << " s, " << n << " tokens in " << gen_s
              << " s, rest " << (n > 1 ? (n - 1) / (gen_s - first_s) : 0.0)
              << " tok/s, reused " << result.reused_tokens << '\n';
  }
  return 0;
}
