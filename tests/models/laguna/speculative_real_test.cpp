// Laguna DFlash speculative decoding (phase 6e) on the real target (68.6 GiB)
// and drafter (0.61 GiB). Without the files it exits 77 (CTest: skipped,
// which verifies nothing). Run it under the GPU admission lock.
//
// Default mode, the exact-greedy matrix: every speculative run must equal
// plain greedy decode of the same model (a session without a drafter, the
// serve path's decode) token for token, end at the same position and leave
// bit-identical frontier logits. Covered: raw code prompts of 509..515, 2048
// and 8192 tokens (and --long N, e.g. 32768, with a matching --context);
// chat prompts with thinking on and off; the Laguna tool-call grammar run
// through the draft prefix (thinking on and off); a short answer that ends
// on EOT with the drafts watched for stop tokens; max_tokens of 1, 2, 3, 6
// and 11 inside a block of 8; a snapshot and restore mid-request (the drafter
// ring restored; drafts after restore equal the original's bit for bit); and
// 2 and 8 sessions on threads, speculating greedy and plain sampled mixed,
// each equal to its solo run. On a divergence it prints the index, both
// tokens and the plain run's top-2 logit gap there before failing. It also
// prints, per run, steps, drafted, accepted and the smallest top-2 gap the
// plain run decoded through (how close to a tie the exactness held).
//
// --ab: the smoke A/B, plain against speculative at blocks 4 and 8, in this
// one process: --rounds rounds (default 5), round 0 dropped as warm-up, the
// order of the three modes reversed every round (ABC, CBA, ...), each run a
// fresh request of --tokens tokens (default 160) on a reused session.
// Decode only (the prompt's prefill is excluded); unprofiled wall time.
// Prints per prompt and mode the median tok/s and mean accepted per step,
// and checks every speculative output against the plain one.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <thread>

#include "src/core/gguf_reader.hpp"
#include "src/core/json.hpp"
#include "src/core/json_constraint.hpp"
#include "src/models/laguna/engine.hpp"
#include "src/models/laguna/kernels/dflash_session.hpp"
#include "src/models/laguna/speculative.hpp"
#include "tests/models/laguna/speculative_fixture.hpp"
#include "tests/models/laguna/support.hpp"

using namespace gufo::testing::laguna;
namespace fs = std::filesystem;

namespace {

std::string Env(const char* name, const char* fallback) {
  const char* v = std::getenv(name);
  return v != nullptr && *v != '\0' ? v : fallback;
}

double Seconds(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
      .count();
}

/// Top-1 minus top-2 of a logit row.
float Gap(const std::vector<float>& l) {
  float a = -std::numeric_limits<float>::infinity();
  float b = a;
  for (const float v : l) {
    if (v > a) {
      b = a;
      a = v;
    } else if (v > b) {
      b = v;
    }
  }
  return a - b;
}

/// Plain greedy decode recording the top-2 gap of every selected position.
struct PlainRun {
  Run run;
  std::vector<float> gaps;
};
PlainRun PlainGaps(ml::Model& m, const Tokens& prompt, std::uint32_t max,
                   const smp::SamplingConfig& config, bool stop_at_eos = true) {
  std::string error;
  auto s = m.CreateSession(&error, /*with_drafter=*/false);
  Require(s != nullptr, "plain session: " + error);
  Require(s->Prefill(prompt, &error), "plain prefill: " + error);
  smp::SamplerState sampler(config, U32(prompt));
  PlainRun out;
  while (out.run.tokens.size() < max && s->Position() < s->Context()) {
    out.gaps.push_back(Gap(s->Logits()));
    const auto t = static_cast<std::int32_t>(sampler.Sample(s->Logits()));
    if (stop_at_eos && m.IsStopToken(t)) {
      out.run.stopped = true;
      break;
    }
    out.run.tokens.push_back(t);
    sampler.Accept(static_cast<std::uint32_t>(t));
    Require(s->Evaluate(t, &error), "plain evaluate: " + error);
  }
  out.run.logits = s->Logits();
  out.run.position = s->Position();
  return out;
}

struct Watch {
  std::uint32_t stop_drafts{0};  ///< blocks that proposed EOS or EOT
};

/// The real drafter, watched: proposes exactly what the backend would.
ml::LagunaDraftBackend::DraftOverride Spy(ml::Session& s, Watch* w) {
  return [&s, w](std::uint32_t pos, std::int32_t anchor, std::uint32_t max) {
    (void)pos;
    std::vector<std::int32_t> d;
    if (!s.dflash()->capturing() ||
        !s.dflash()->Propose(anchor, max + 1, &d, nullptr, nullptr)) {
      return std::vector<std::int32_t>{};
    }
    if (std::any_of(d.begin(), d.end(),
                    [](std::int32_t t) { return t == 2 || t == 24; })) {
      ++w->stop_drafts;
    }
    return d;
  };
}

void Report(const std::string& name, const PlainRun& ref, const Run& got,
            std::uint32_t block, const Watch* watch = nullptr) {
  float min_gap = std::numeric_limits<float>::infinity();
  for (std::size_t i = 0; i < ref.gaps.size() && i < ref.run.tokens.size();
       ++i) {
    min_gap = std::min(min_gap, ref.gaps[i]);
  }
  std::cout << "  " << std::left << std::setw(30) << name << " B" << block
            << ": " << got.tokens.size() << " tok, " << got.steps
            << " steps, accepted " << got.accepted << "/" << got.drafted
            << " (mean " << std::fixed << std::setprecision(2)
            << (got.steps ? static_cast<double>(got.accepted) / got.steps : 0)
            << "/step), min plain top-2 gap " << std::setprecision(4) << min_gap
            << (got.stopped ? ", stopped" : "");
  if (watch != nullptr) {
    std::cout << ", blocks proposing a stop token " << watch->stop_drafts;
  }
  std::cout << "\n" << std::flush;
  if (got.tokens != ref.run.tokens) {
    std::size_t i = 0;
    while (i < got.tokens.size() && i < ref.run.tokens.size() &&
           got.tokens[i] == ref.run.tokens[i]) {
      ++i;
    }
    std::cout << "  DIVERGENCE at " << i << ": plain "
              << (i < ref.run.tokens.size() ? ref.run.tokens[i] : -1)
              << " speculative " << (i < got.tokens.size() ? got.tokens[i] : -1)
              << ", plain top-2 gap there "
              << (i < ref.gaps.size() ? ref.gaps[i] : -1.0F) << "\n";
  }
  RequireSame(got, ref.run, name + " B" + std::to_string(block));
}

Run SpecRun(ml::Model& m, const Tokens& prompt, std::uint32_t max,
            const smp::SamplingConfig& config, std::uint32_t block,
            Watch* watch = nullptr, bool stop_at_eos = true) {
  std::string error;
  auto s = m.CreateSession(&error);
  Require(s != nullptr && s->speculative(), "spec session: " + error);
  s->decoder()->set_block_rows(block);
  if (watch != nullptr) {
    s->decoder()->backend().SetDraftOverrideForTesting(Spy(*s, watch));
  }
  return Spec(*s, prompt, max, config, stop_at_eos);
}

std::shared_ptr<const smp::TokenConstraint> ToolConstraint(const ml::Model& m,
                                                           bool reasoning) {
  const auto schema = gufo::json::parse(
      R"({"type":"object","properties":{"city":{"type":"string"},)"
      R"("unit":{"type":"string","enum":["celsius","fahrenheit"]}},)"
      R"("required":["city","unit"],"additionalProperties":false})");
  auto grammar = smp::JsonConstraint::WithTools(
      nullptr,
      {{"get_weather",
        smp::JsonConstraint::ToolParameters(
            schema, true, smp::JsonConstraint::ToolFormat::kLaguna)}},
      true, false, smp::JsonConstraint::ToolFormat::kLaguna);
  if (reasoning) {
    grammar = smp::JsonConstraint::WithReasoning(grammar);
  }
  auto binding = std::make_shared<smp::TokenConstraint>();
  binding->grammar = std::move(grammar);
  binding->vocabulary = std::make_shared<const smp::ConstraintVocabulary>(
      m.VocabSize(), [&m](std::uint32_t id) {
        return smp::ConstraintVocabulary::Piece{
            m.DecodeToken(static_cast<std::int32_t>(id)),
            m.IsStopToken(static_cast<std::int32_t>(id))};
      });
  return binding;
}

Tokens Chat(const ml::Model& m, const std::string& text, bool thinking,
            const std::vector<ml::ChatTool>& tools = {}) {
  ml::ChatTemplateOptions options;
  options.enable_thinking = thinking;
  const std::vector<ml::ChatMessage> messages{
      {.role = "user", .content = text}};
  return m.EncodeChat(messages, tools, options);
}

struct Texts {
  Tokens code;  ///< BOS + the 6b-0 code text
  std::string code_text;
  std::string prose_text;
};

// ---------------------------------------------------------------------------

void Matrix(ml::Model& m, const Texts& t, std::uint32_t long_prompt) {
  std::cout << "exact-greedy matrix\n";
  // Raw code prompts straddling the window, 2k, 8k (and --long).
  std::vector<std::size_t> lengths{509, 510, 511,  512, 513,
                                   514, 515, 2048, 8192};
  if (long_prompt != 0) {
    lengths.push_back(long_prompt);
  }
  for (const std::size_t n : lengths) {
    Require(t.code.size() >= n, "code text too short");
    const Tokens prompt(t.code.begin(), t.code.begin() + n);
    const std::uint32_t max = n >= 2048 ? 48 : 40;
    const PlainRun ref = PlainGaps(m, prompt, max, Greedy());
    for (const std::uint32_t block : n == 512 || n == 2048
                                         ? std::vector<std::uint32_t>{4, 8, 16}
                                         : std::vector<std::uint32_t>{4}) {
      Report("code " + std::to_string(n), ref,
             SpecRun(m, prompt, max, Greedy(), block), block);
    }
  }
  // Chat, thinking on and off.
  const std::string code_q =
      "Write a Python function that merges overlapping intervals in a list "
      "of [start, end] pairs, with a short docstring and two doctests.";
  const std::string chat_q =
      "My basil plant on the kitchen windowsill keeps wilting even though I "
      "water it every day. What could be going on?";
  for (const bool thinking : {true, false}) {
    for (const auto& [name, text] :
         {std::pair<std::string, std::string>{"code", code_q},
          {"chat", chat_q}}) {
      const Tokens prompt = Chat(m, text, thinking);
      const PlainRun ref = PlainGaps(m, prompt, 96, Greedy());
      for (const std::uint32_t block : {4U, 8U}) {
        Report("chat " + name + (thinking ? " think" : " nothink"), ref,
               SpecRun(m, prompt, 96, Greedy(), block), block);
      }
    }
  }
  // The tool-call grammar through the draft prefix.
  const std::vector<ml::ChatTool> tools{
      {.name = "get_weather",
       .description = "Get the current weather for a city.",
       .parameters_json =
           R"({"type":"object","properties":{"city":{"type":"string"},)"
           R"("unit":{"type":"string","enum":["celsius","fahrenheit"]}},)"
           R"("required":["city","unit"],"additionalProperties":false})",
       .definition_json = ""}};
  for (const bool thinking : {false, true}) {
    auto config = Greedy();
    config.constraint = ToolConstraint(m, thinking);
    const Tokens prompt =
        Chat(m, "What's the weather in Lisbon right now, in celsius?", thinking,
             tools);
    const PlainRun ref = PlainGaps(m, prompt, 96, config);
    for (const std::uint32_t block : {4U, 8U}) {
      Watch w;
      const Run got = SpecRun(m, prompt, 96, config, block, &w);
      Report(std::string("tool grammar") + (thinking ? " think" : " nothink"),
             ref, got, block, &w);
      Require(got.drafted > 0, "the constrained request did not draft");
    }
  }
  // Short answers that end on EOT; the drafts are watched for stops.
  for (const char* text :
       {"List the three primary colours of light, one per line, and nothing "
        "else.",
        "Reply with exactly: OK, done."}) {
    const Tokens prompt = Chat(m, text, false);
    const PlainRun ref = PlainGaps(m, prompt, 64, Greedy());
    for (const std::uint32_t block : {4U, 8U, 16U}) {
      Watch w;
      Report(std::string("short list/reply (EOT)"), ref,
             SpecRun(m, prompt, 64, Greedy(), block, &w), block, &w);
    }
  }
  {
    const Tokens prompt =
        Chat(m,
             "Answer with one word only: what colour is a clear daytime "
             "sky?",
             false);
    const PlainRun ref = PlainGaps(m, prompt, 64, Greedy());
    Require(ref.run.stopped, "the short answer did not stop");
    for (const std::uint32_t block : {4U, 8U, 16U}) {
      Watch w;
      Report("short answer (EOT)", ref,
             SpecRun(m, prompt, 64, Greedy(), block, &w), block, &w);
    }
    // The same without stopping at EOS/EOT (ignore_eos).
    const PlainRun all = PlainGaps(m, prompt, 24, Greedy(), false);
    Watch w;
    Report("short answer ignore_eos", all,
           SpecRun(m, prompt, 24, Greedy(), 8, &w, false), 8, &w);
  }
  // max_tokens inside a block of 8.
  {
    const Tokens prompt = Chat(m, code_q, false);
    for (const std::uint32_t max : {1U, 2U, 3U, 6U, 11U}) {
      const PlainRun ref = PlainGaps(m, prompt, max, Greedy());
      Report("max_tokens " + std::to_string(max), ref,
             SpecRun(m, prompt, max, Greedy(), 8), 8);
    }
  }
}

void Snapshots(ml::Model& m, const Texts& t) {
  std::cout << "snapshot/restore mid-request\n";
  std::string error;
  const Tokens prompt(t.code.begin(), t.code.begin() + 700);
  const Tokens other(t.code.begin() + 1000, t.code.begin() + 1600);
  auto a = m.CreateSession(&error);
  auto b = m.CreateSession(&error);
  Require(a != nullptr && b != nullptr, error);
  const Run head = Spec(*a, prompt, 20, Greedy());
  auto snap = a->SaveSnapshot(&error);
  Require(snap != nullptr && snap->drafter.has_value(),
          "snapshot without the drafter ring: " + error);
  smp::SamplerState greedy;
  const auto anchor = static_cast<std::int32_t>(greedy.Sample(a->Logits()));
  std::vector<std::int32_t> da, db;
  std::vector<float> la, lb;
  Require(a->dflash()->Propose(anchor, 16, &da, &la, &error), error);
  Tokens seq = prompt;
  seq.insert(seq.end(), head.tokens.begin(), head.tokens.end());
  const Run tail_a = Spec(*a, seq, 40, Greedy(), true, false);
  (void)Spec(*b, other, 9, Greedy());
  Require(b->RestoreSnapshot(*snap, &error), "restore: " + error);
  Require(b->dflash()->Propose(anchor, 16, &db, &lb, &error), error);
  Require(da == db && SameBits(la, lb),
          "drafts after restoring the drafter ring differ");
  const Run tail_b = Spec(*b, seq, 40, Greedy(), true, false);
  RequireSame(tail_b, tail_a, "continuation after restore");
  Require(tail_b.accepted == tail_a.accepted, "acceptance after restore");
  const PlainRun ref = PlainGaps(m, prompt, 60, Greedy());
  Tokens all = head.tokens;
  all.insert(all.end(), tail_a.tokens.begin(), tail_a.tokens.end());
  Require(all == ref.run.tokens && SameBits(tail_a.logits, ref.run.logits),
          "snapshot request differs from plain decode");
  std::cout << "  ok: 20 + 40 tokens, ring of " << snap->drafter->bytes()
            << " bytes restored; drafts after restore equal bit for bit; "
               "accepted "
            << tail_a.accepted << "/" << tail_a.drafted << " on both\n";
}

void Threads(ml::Model& m, const Texts& t, int count) {
  std::cout << count << " sessions on threads (speculating and plain mixed)\n";
  std::string error;
  struct Job {
    Tokens prompt;
    bool speculate;
  };
  std::vector<Job> jobs;
  for (int i = 0; i < count; ++i) {
    const std::size_t at = 2000 + 700 * i;
    jobs.push_back(
        {Tokens(t.code.begin() + at, t.code.begin() + at + 520 + 7 * i),
         i % 2 == 0});
    jobs.back().prompt.insert(jobs.back().prompt.begin(), 2);
  }
  smp::SamplingConfig hot;
  hot.temperature = 1.0F;
  hot.top_k = 20;
  const auto run = [&](ml::Session& s, const Job& j, int i) {
    if (j.speculate) {
      return Spec(s, j.prompt, 32, Greedy());
    }
    return Plain(s, j.prompt, 32, hot, true, 77 + i);
  };
  std::vector<Run> solo;
  for (int i = 0; i < count; ++i) {
    auto s = m.CreateSession(&error);
    Require(s != nullptr, error);
    solo.push_back(run(*s, jobs[i], i));
  }
  {
    std::vector<std::unique_ptr<ml::Session>> sessions;
    for (int i = 0; i < count; ++i) {
      sessions.push_back(m.CreateSession(&error));
      Require(sessions.back() != nullptr, error);
    }
    std::atomic<bool> go{false};
    std::atomic<int> failures{0};
    std::vector<Run> got(count);
    std::vector<std::thread> threads;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < count; ++i) {
      threads.emplace_back([&, i] {
        try {
          while (!go.load()) {
            std::this_thread::yield();
          }
          got[i] = run(*sessions[i], jobs[i], i);
        } catch (const std::exception& e) {
          std::cerr << "thread " << i << ": " << e.what() << '\n';
          ++failures;
        }
      });
    }
    go = true;
    for (auto& th : threads) {
      th.join();
    }
    Require(failures.load() == 0, "a concurrent session failed");
    for (int i = 0; i < count; ++i) {
      RequireSame(got[i], solo[i], "concurrent session " + std::to_string(i));
    }
    std::cout << "  ok: each equals its solo run (" << Seconds(t0) << " s)\n";
  }
}

// ---------------------------------------------------------------------------

struct AbPrompt {
  std::string name;
  Tokens tokens;
};

int Ab(ml::Model& m, const Texts& t, int rounds, std::uint32_t tokens,
       const std::string& csv_path) {
  std::string error;
  std::vector<AbPrompt> prompts;
  prompts.push_back(
      {"code", Chat(m,
                    "Write a Python module that parses ISO-8601 timestamps "
                    "into datetime objects, with input validation, clear "
                    "error messages and pytest unit tests.",
                    false)});
  prompts.push_back(
      {"prose",
       Chat(m,
            "Summarize the following document in three paragraphs:\n\n" +
                t.prose_text.substr(0, 6000),
            false)});
  prompts.push_back(
      {"chat",
       Chat(m,
            "I'm planning a three-day trip to Kyoto in late autumn. What "
            "should I see, and how should I plan each day?",
            true)});
  prompts.push_back(
      {"code-explain",
       Chat(m,
            "Explain what this code does and suggest two improvements:\n\n" +
                t.code_text.substr(20000, 5000),
            false)});
  auto plain = m.CreateSession(&error, false);
  auto spec = m.CreateSession(&error);
  Require(plain != nullptr && spec != nullptr, error);
  struct Sample {
    double tps;
    double accepted_per_step;
    double tokens_per_step;
  };
  std::map<std::pair<std::string, std::string>, std::vector<Sample>> samples;
  std::ofstream csv(csv_path);
  csv << "round,prompt,mode,prompt_tokens,tokens,decode_s,tok_s,steps,"
         "drafted,accepted\n";
  const std::vector<std::string> modes{"plain", "spec-b4", "spec-b8"};
  std::map<std::string, Tokens> reference;
  for (int round = 0; round < rounds; ++round) {
    for (std::size_t pi = 0; pi < prompts.size(); ++pi) {
      const auto& p = prompts[pi];
      std::vector<std::string> order = modes;
      if ((round + pi) % 2 == 1) {
        std::reverse(order.begin(), order.end());
      }
      for (const auto& mode : order) {
        ml::Session& s = mode == "plain" ? *plain : *spec;
        s.Invalidate();
        Require(s.Prefill(p.tokens, &error), "prefill: " + error);
        Run r;
        const auto t0 = std::chrono::steady_clock::now();
        if (mode == "plain") {
          smp::SamplerState sampler(Greedy(), U32(p.tokens));
          while (r.tokens.size() < tokens) {
            const auto tok =
                static_cast<std::int32_t>(sampler.Sample(s.Logits()));
            if (m.IsStopToken(tok)) {
              break;
            }
            r.tokens.push_back(tok);
            sampler.Accept(static_cast<std::uint32_t>(tok));
            Require(s.Evaluate(tok, &error), "evaluate: " + error);
            ++r.steps;
          }
        } else {
          s.decoder()->set_block_rows(mode == "spec-b4" ? 4 : 8);
          r = Spec(s, p.tokens, tokens, Greedy(), true, false);
        }
        const double secs = Seconds(t0);
        const double tps = r.tokens.size() / secs;
        csv << round << ',' << p.name << ',' << mode << ',' << p.tokens.size()
            << ',' << r.tokens.size() << ',' << secs << ',' << tps << ','
            << r.steps << ',' << r.drafted << ',' << r.accepted << '\n'
            << std::flush;
        auto& ref = reference[p.name];
        if (ref.empty() && mode == "plain") {
          ref = r.tokens;
        }
        if (round > 0) {
          samples[{p.name, mode}].push_back(
              {tps, r.steps ? static_cast<double>(r.accepted) / r.steps : 0.0,
               r.steps ? static_cast<double>(r.tokens.size()) / r.steps : 0.0});
        }
        std::cout << "round " << round << " " << p.name << " " << mode << ": "
                  << r.tokens.size() << " tok in " << std::fixed
                  << std::setprecision(3) << secs
                  << " s = " << std::setprecision(2) << tps << " tok/s"
                  << (mode == "plain"
                          ? std::string()
                          : ", accepted " + std::to_string(r.accepted) + "/" +
                                std::to_string(r.drafted) + " in " +
                                std::to_string(r.steps) + " steps")
                  << "\n"
                  << std::flush;
        if (!ref.empty() && mode != "plain") {
          Require(r.tokens == ref,
                  "A/B: speculative output differs from plain (" + p.name +
                      ", " + mode + ")");
        }
      }
    }
  }
  const auto median = [](std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const std::size_t n = v.size();
    return n == 0 ? 0.0 : n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
  };
  std::cout << "\nA/B (rounds 1.." << rounds - 1
            << ", decode only, median tok/s; spec outputs equal plain)\n"
            << std::left << std::setw(14) << "prompt" << std::setw(10)
            << "plain" << std::setw(22) << "spec B4 (x, acc/step)"
            << "spec B8 (x, acc/step)\n";
  for (const auto& p : prompts) {
    const auto stat = [&](const std::string& mode, auto field) {
      std::vector<double> v;
      for (const auto& sample : samples[{p.name, mode}]) {
        v.push_back(field(sample));
      }
      return median(v);
    };
    const auto tps = [](const Sample& x) { return x.tps; };
    const auto acc = [](const Sample& x) { return x.accepted_per_step; };
    const double base = stat("plain", tps);
    std::cout << std::setw(14) << p.name << std::fixed << std::setprecision(2)
              << std::setw(10) << base;
    for (const std::string mode : {"spec-b4", "spec-b8"}) {
      const double v = stat(mode, tps);
      std::ostringstream cell;
      cell << std::fixed << std::setprecision(2) << v << " (" << v / base
           << "x, " << stat(mode, acc) << ")";
      std::cout << std::setw(22) << cell.str();
    }
    std::cout << "  [" << p.tokens.size() << " prompt tokens]\n";
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    bool ab = false;
    int rounds = 5;
    std::uint32_t tokens = 160;
    std::uint32_t context = 9216;
    std::uint32_t long_prompt = 0;
    bool threads = true;
    std::string csv = "speculative-ab.csv";
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      const auto next = [&]() -> std::string {
        Require(i + 1 < argc, a + " needs a value");
        return argv[++i];
      };
      if (a == "--ab") {
        ab = true;
      } else if (a == "--rounds") {
        rounds = std::stoi(next());
      } else if (a == "--tokens") {
        tokens = static_cast<std::uint32_t>(std::stoul(next()));
      } else if (a == "--context") {
        context = static_cast<std::uint32_t>(std::stoul(next()));
      } else if (a == "--long") {
        long_prompt = static_cast<std::uint32_t>(std::stoul(next()));
      } else if (a == "--no-threads") {
        threads = false;
      } else if (a == "--csv") {
        csv = next();
      } else {
        Require(false, "unknown argument " + a);
      }
    }
    const fs::path drafter_path =
        Env("GUFO_LAGUNA_DFLASH_GGUF",
            "/server/ai/models/Myric/Laguna-S-2.1-APEX-GGUF/"
            "laguna-s-2.1-DFlash-Q4_K.gguf");
    const fs::path text_path =
        Env("GUFO_LAGUNA_6D_TEXT",
            "/server/staging/laguna/dflash-6b0/verify-context-code.txt");
    const fs::path prose_path =
        Env("GUFO_LAGUNA_6E_PROSE", (fs::path(__FILE__)
                                         .parent_path()
                                         .parent_path()
                                         .parent_path()
                                         .parent_path() /
                                     "README.md")
                                        .c_str());
    std::error_code ec;
    if (!fs::exists(ModelPath(), ec) || !fs::exists(drafter_path, ec) ||
        !fs::exists(text_path, ec) || !fs::exists(prose_path, ec)) {
      std::cout << "SKIP: laguna.speculative_real needs the target and "
                   "drafter GGUFs and the texts; nothing was verified\n";
      return kSkipExitCode;
    }
    std::string error;
    const auto t0 = std::chrono::steady_clock::now();
    std::shared_ptr<const gufo::core::GgufReader> reader =
        gufo::core::GgufReader::OpenFile(ModelPath(), &error);
    Require(reader != nullptr, "target: " + error);
    std::shared_ptr<const gufo::core::GgufReader> dreader =
        gufo::core::GgufReader::OpenFile(drafter_path, &error);
    Require(dreader != nullptr, "drafter: " + error);
    auto model = ml::Model::Load(
        reader,
        ml::ModelOptions{.max_context = context,
                         .prefill_chunk = 512,
                         .session_count = ab ? 2U : (threads ? 8U : 2U)},
        &error);
    Require(model != nullptr, "load: " + error);
    Require(model->AttachDrafter(dreader, ml::DrafterOptions{}, &error),
            "drafter: " + error);
    std::cout << "loaded target + drafter in " << Seconds(t0) << " s (context "
              << context << ", default block " << model->draft_block_rows()
              << ")\n"
              << std::flush;
    Texts t;
    {
      std::ifstream in(text_path, std::ios::binary);
      std::stringstream ss;
      ss << in.rdbuf();
      t.code_text = ss.str();
      t.code = model->Tokenize(t.code_text, /*add_bos=*/true);
      std::ifstream pin(prose_path, std::ios::binary);
      std::stringstream ps;
      ps << pin.rdbuf();
      t.prose_text = ps.str();
    }
    if (ab) {
      return Ab(*model, t, rounds, tokens, csv);
    }
    const auto t1 = std::chrono::steady_clock::now();
    Matrix(*model, t, long_prompt);
    Snapshots(*model, t);
    if (threads) {
      Threads(*model, t, 2);
      Threads(*model, t, 8);
    }
    std::cout << "checks took " << Seconds(t1) << " s\n";
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
  std::cout << "laguna.speculative_real: ok\n";
  return 0;
}
