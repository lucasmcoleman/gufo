#ifndef GUFO_TESTS_MODELS_LAGUNA_SPECULATIVE_FIXTURE_HPP_
#define GUFO_TESTS_MODELS_LAGUNA_SPECULATIVE_FIXTURE_HPP_

// Shared by the speculative-decoding checks (phase 6e): plain decode as the
// serve runner does it (select from the frontier, then advance), speculative
// decode through Session::DecodeSpeculative, and the exactness comparison
// (tokens, stop, position and bit-identical frontier logits).

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "src/core/sampling.hpp"
#include "src/models/laguna/engine.hpp"
#include "src/models/laguna/speculative.hpp"
#include "tests/models/laguna/support.hpp"

namespace gufo::testing::laguna {

namespace ml = gufo::models::laguna;
namespace smp = gufo::sampling;
using Tokens = std::vector<std::int32_t>;

inline bool SameBits(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() &&
         std::equal(a.begin(), a.end(), b.begin(), [](float x, float y) {
           return std::memcmp(&x, &y, sizeof(float)) == 0;
         });
}

inline std::vector<gufo::tokenization::TokenId> U32(const Tokens& t) {
  return {t.begin(), t.end()};
}

inline smp::SamplingConfig Greedy() {
  smp::SamplingConfig c;
  c.temperature = 0.0F;
  return c;
}

struct Run {
  Tokens tokens;
  std::vector<float> logits;  ///< frontier after the run
  std::uint32_t position{0};
  bool stopped{false};
  std::uint32_t steps{0};
  std::uint32_t drafted{0};
  std::uint32_t accepted{0};
};

inline void RequireSame(const Run& a, const Run& b, const std::string& what) {
  if (a.tokens != b.tokens) {
    std::size_t i = 0;
    while (i < a.tokens.size() && i < b.tokens.size() &&
           a.tokens[i] == b.tokens[i]) {
      ++i;
    }
    Require(false, what + ": tokens differ at index " + std::to_string(i) +
                       " (" + std::to_string(a.tokens.size()) + " vs " +
                       std::to_string(b.tokens.size()) + " tokens)");
  }
  Require(a.stopped == b.stopped && a.position == b.position,
          what + ": stop or position differs");
  Require(SameBits(a.logits, b.logits), what + ": frontier logits differ");
}

/// Plain decode as the serve runner does it: select from the frontier,
/// then advance (here every emitted token is advanced, so the end state is
/// the one after all of them).
inline Run Plain(ml::Session& s, const Tokens& prompt, std::uint32_t max,
                 const smp::SamplingConfig& config, bool stop_at_eos = true,
                 std::uint64_t seed = 0) {
  std::string error;
  Require(s.Prefill(prompt, &error), "plain prefill: " + error);
  // As the runner does for a request that samples: no speculation, and the
  // session leaves the drafter's taps (so batched decode could take it).
  s.SuspendDrafter();
  smp::SamplingConfig c = config;
  c.seed = static_cast<std::int64_t>(seed);
  smp::SamplerState sampler(c, U32(prompt));
  Run r;
  while (r.tokens.size() < max && s.Position() < s.Context()) {
    const auto t = static_cast<std::int32_t>(sampler.Sample(s.Logits()));
    if (stop_at_eos && (t == 2 || t == 24)) {
      r.stopped = true;
      break;
    }
    r.tokens.push_back(t);
    sampler.Accept(static_cast<std::uint32_t>(t));
    Require(s.Evaluate(t, &error), "plain evaluate: " + error);
  }
  r.logits = s.Logits();
  r.position = s.Position();
  return r;
}

inline Run Plain(ml::Model& m, const Tokens& prompt, std::uint32_t max,
                 const smp::SamplingConfig& config, bool stop_at_eos = true) {
  std::string error;
  auto s = m.CreateSession(&error);
  Require(s != nullptr, "plain session: " + error);
  return Plain(*s, prompt, max, config, stop_at_eos);
}

inline Run Spec(ml::Session& s, const Tokens& prompt, std::uint32_t max,
                const smp::SamplingConfig& config, bool stop_at_eos = true,
                bool prefill = true) {
  std::string error;
  if (prefill) {
    Require(s.Prefill(prompt, &error), "spec prefill: " + error);
  }
  smp::SamplerState sampler(config, U32(prompt));
  Run r;
  const std::uint32_t block = s.decoder()->block_rows();
  while (r.tokens.size() < max) {
    ml::SpeculativeStepResult step;
    const auto remaining = static_cast<std::uint32_t>(max - r.tokens.size());
    const bool ok = s.DecodeSpeculative(sampler, remaining, s.Context(),
                                        stop_at_eos, &step, &error);
    Require(ok, "speculative step: " + error);
    if (step.stop) {
      // Only a stop token at the frontier ends a greedy request early.
      r.stopped = s.Position() < s.Context();
      break;
    }
    Require(!step.tokens.empty() && step.tokens.size() <= remaining &&
                step.tokens.size() <= block &&
                step.tokens.size() == step.accepted + 1 &&
                step.accepted <= step.drafted && step.drafted < block,
            "speculative step size out of bounds");
    for (const auto t : step.tokens) {
      r.tokens.push_back(t);
      sampler.Accept(static_cast<std::uint32_t>(t));
    }
    ++r.steps;
    r.drafted += step.drafted;
    r.accepted += step.accepted;
  }
  r.logits = s.Logits();
  r.position = s.Position();
  return r;
}

}  // namespace gufo::testing::laguna

#endif  // GUFO_TESTS_MODELS_LAGUNA_SPECULATIVE_FIXTURE_HPP_
