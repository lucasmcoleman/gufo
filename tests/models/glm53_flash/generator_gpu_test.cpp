// The generation loop (generator.hpp) on the real GPU Executor: the e2e
// fixture model of executor_loop_test.cpp (loop_model.hpp) with the host MLA /
// FFN stand-ins of host_ops.hpp, bound through MakeExecutorSession (the
// production session wrapper). Checks against the CPU ReferenceModel:
//   (a) greedy generation from a 12-token prompt (prefilled in executor passes
//       of 5) emits the reference's greedy tokens, with the reference's
//       logits after each of them;
//   (b) a follow-up prompt extending the held tokens prefills only the new
//       suffix and continues the reference's sequence;
//   (c) a stop token ends the turn before the model runs it;
//   (d) a diverging prompt resets the executor and matches a fresh reference.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/models/glm53_flash/engine.hpp"
#include "src/models/glm53_flash/executor.hpp"
#include "src/models/glm53_flash/generator.hpp"
#include "src/models/glm53_flash/reference.hpp"
#include "tests/models/glm53_flash/host_ops.hpp"
#include "tests/models/glm53_flash/loop_model.hpp"

namespace glm = gufo::models::glm53_flash;
namespace ref = gufo::models::glm53_flash::reference;
namespace fixture = gufo::models::glm53_flash::loop_model;

namespace {

constexpr double kTol = 1e-5;
constexpr std::uint32_t kPrompt = 12;
constexpr std::uint32_t kCapacity = 64;
int failures = 0;

void Check(const std::string& what, bool ok) {
  std::cout << (ok ? "ok   " : "FAIL ") << what << "\n";
  if (!ok)
    ++failures;
}

double RelL2(std::span<const float> got, std::span<const float> want) {
  if (got.size() != want.size() || want.empty())
    return INFINITY;
  double num = 0.0;
  double den = 0.0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    const double d = static_cast<double>(got[i]) - want[i];
    num += d * d;
    den += static_cast<double>(want[i]) * want[i];
  }
  return den == 0.0 ? std::sqrt(num) : std::sqrt(num / den);
}

/// Greedy reference continuation: feeds `prompt` (continuing the model's
/// state), then `steps` argmax tokens. Returns the tokens and the logits
/// after the prompt and after each token but the last.
struct Greedy {
  std::vector<glm::TokenId> tokens;
  std::vector<std::vector<float>> logits;
};

Greedy ReferenceGreedy(ref::ReferenceModel& model,
                       std::span<const glm::TokenId> prompt,
                       std::uint32_t steps, std::uint32_t vocab) {
  Greedy out;
  std::vector<std::int32_t> ids(prompt.begin(), prompt.end());
  for (std::uint32_t step = 0; step < steps; ++step) {
    const auto result = model.Forward(ids, true);
    if (result.logits.size() < vocab)
      throw std::runtime_error("reference returned no logits");
    std::vector<float> last(result.logits.end() - vocab, result.logits.end());
    const auto best = static_cast<glm::TokenId>(
        std::max_element(last.begin(), last.end()) - last.begin());
    out.logits.push_back(std::move(last));
    out.tokens.push_back(best);
    if (step + 1 < steps)
      ids.assign(1, static_cast<std::int32_t>(best));
  }
  return out;
}

/// Smallest gap between the best and second-best logit: greedy agreement is
/// only meaningful when it is well above the numerical tolerance.
float MinMargin(const std::vector<std::vector<float>>& rows) {
  float margin = INFINITY;
  for (const auto& row : rows) {
    std::vector<float> sorted(row);
    std::partial_sort(sorted.begin(), sorted.begin() + 2, sorted.end(),
                      std::greater<>());
    margin = std::min(margin, sorted[0] - sorted[1]);
  }
  return margin;
}

void Run(const fixture::Model& m, bool fused) {
  std::cout << (fused ? "-- fused last-token logits\n"
                      : "-- separate last-token pass\n");
  const glm::Config& c = m.config;
  const std::uint32_t vocab = c.vocab_size;
  glm::stand_in::HostOps ops(c);
  glm::ExecutorOptions options;
  options.max_tokens = 5;  // a 12-token prompt runs in several passes
  options.capacity = kCapacity;
  std::string error;
  auto executor = glm::Executor::Create(m.weights, options, ops.Bind(), &error);
  Check("create executor " + error, executor != nullptr);
  if (!executor)
    return;
  const auto session =
      glm::MakeExecutorSession(*executor, kCapacity, vocab, fused);
  glm::Generator generator(*session, glm::StopTokens{}, nullptr);

  const ref::GgufWeightSource source(m.weights);
  ref::ReferenceModel reference(c, source);
  const std::vector<glm::TokenId> prompt(m.ids.begin(),
                                         m.ids.begin() + kPrompt);
  constexpr std::uint32_t kSteps = 6;
  const Greedy want = ReferenceGreedy(reference, prompt, kSteps, vocab);
  std::cout << "reference greedy margin " << MinMargin(want.logits) << "\n";

  // (a) Greedy generation, logits checked after every token.
  glm::GenerationOptions greedy;
  greedy.max_tokens = kSteps;
  std::vector<double> rel;
  glm::GenerationResult result;
  // The logits sampled for token k are the generator's frontier when the
  // callback for token k runs.
  std::vector<std::vector<float>> seen;
  seen.emplace_back(generator.logits().begin(), generator.logits().end());
  bool ok = generator.Generate(
      prompt, greedy,
      [&](glm::TokenId, std::string_view) {
        seen.emplace_back(generator.logits().begin(), generator.logits().end());
        return true;
      },
      &result, &error);
  Check("generate " + error, ok);
  if (!ok)
    return;
  Check("greedy tokens match the reference",
        result.tokens == want.tokens &&
            result.finish_reason == glm::FinishReason::kLength);
  double worst = 0.0;
  for (std::uint32_t k = 0; k < kSteps; ++k)
    worst = std::max(worst, RelL2(seen[k + 1], want.logits[k]));
  std::cout << "  worst logits rel_l2 " << worst << "\n";
  Check("logits before each token match the reference", worst <= kTol);
  Check("session holds prompt + all but the final token",
        executor->Position() == kPrompt + kSteps - 1 &&
            generator.tokens().size() == kPrompt + kSteps - 1);

  // (b) Extend: the held tokens, the final generated token, two more.
  std::vector<glm::TokenId> next(prompt);
  next.insert(next.end(), result.tokens.begin(), result.tokens.end());
  next.push_back(m.ids[kPrompt]);
  next.push_back(m.ids[kPrompt + 1]);
  const std::uint32_t mla_before = ops.mla_calls();
  greedy.max_tokens = 3;
  // The reference continues from its own state (prompt + 5 tokens fed).
  const std::vector<glm::TokenId> suffix(next.begin() + kPrompt + kSteps - 1,
                                         next.end());
  const Greedy want_next = ReferenceGreedy(reference, suffix, 3, vocab);
  ok = generator.Generate(next, greedy, {}, &result, &error);
  Check("extended generate " + error, ok);
  Check("extension reuses " + std::to_string(result.reused_tokens) +
            " held tokens",
        ok && result.reused_tokens == kPrompt + kSteps - 1);
  Check("extension continues the reference's greedy sequence",
        ok && result.tokens == want_next.tokens);
  // MLA layer 3 runs once per pass: the 3-token suffix is one pass when the
  // last token's logits are fused into it, else 2 + 1; then two decodes.
  const std::uint32_t expect_calls = fused ? 3 : 4;
  Check("only the suffix was prefilled (" +
            std::to_string(ops.mla_calls() - mla_before) + " MLA calls, " +
            std::to_string(expect_calls) + " expected)",
        ops.mla_calls() - mla_before == expect_calls);

  // (c) A stop token ends the turn before it is run.
  const glm::TokenId stop = want.tokens[2];
  glm::Generator stopping(*session, glm::StopTokens{{stop}}, nullptr);
  greedy.max_tokens = kSteps;
  ok = stopping.Generate(prompt, greedy, {}, &result, &error);
  const std::size_t first_stop =
      std::find(want.tokens.begin(), want.tokens.end(), stop) -
      want.tokens.begin();
  Check("stop token " + std::to_string(stop) + " ends the turn",
        ok && result.finish_reason == glm::FinishReason::kStop &&
            result.stop_token == stop &&
            result.tokens == std::vector<glm::TokenId>(
                                 want.tokens.begin(),
                                 want.tokens.begin() +
                                     static_cast<std::ptrdiff_t>(first_stop)));
  Check("the stop token was not run",
        executor->Position() == kPrompt + first_stop);

  // (d) A diverging prompt resets and matches a fresh reference.
  std::vector<glm::TokenId> other(m.ids.begin() + 1,
                                  m.ids.begin() + kPrompt + 1);
  ref::ReferenceModel fresh(c, source);
  const Greedy want_other = ReferenceGreedy(fresh, other, 4, vocab);
  greedy.max_tokens = 4;
  ok = stopping.Generate(other, greedy, {}, &result, &error);
  Check("diverging prompt " + error, ok && result.reused_tokens == 0);
  // Tokens up to the first stop agree with the fresh reference.
  std::vector<glm::TokenId> expect_other;
  for (const glm::TokenId t : want_other.tokens) {
    if (t == stop)
      break;
    expect_other.push_back(t);
  }
  Check("diverging prompt matches a fresh reference",
        ok && result.tokens == expect_other);
}

}  // namespace

int main() {
  try {
    int devices = 0;
    if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
      std::cerr << "no HIP device\n";
      return 77;
    }
    std::cout << std::scientific << std::setprecision(2);
    const fixture::Fixture fx = fixture::Load(GLM53_E2E_FIXTURE);
    fixture::Model m;
    fixture::BuildModel(m, fx);
    Run(m, false);
    Run(m, true);
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << "\n";
    return 1;
  }
  if (failures != 0) {
    std::cerr << failures << " check(s) failed\n";
    return 1;
  }
  std::cout << "glm53_flash generator_gpu: PASS\n";
  return 0;
}
