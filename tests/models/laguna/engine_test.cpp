// Laguna serving engine on a tiny random model (no model file, nothing
// skipped): the Model and Session that `gufo serve` drives, on the GPU, with
// the real geometry (head_dim 128, GQA 6 and 9, partial RoPE with YaRN on the
// full layers, routed Q6_K/Q5_K/IQ4_XS), a 12-position window and prefill
// chunks of 16 (so the sliding rings hold 28 positions and wrap).
//
//  * determinism: equal sessions give bit-equal logits; Invalidate() starts
//    over exactly.
//  * snapshots: a state saved at positions around the window (11/12/13), the
//    ring capacity (27/28/29) and far past it, restored into a fresh session
//    or into the same one after it ran on, decodes bit-identically, and the
//    full-layer and ring bookkeeping (committed, retained_begin, valid) is
//    restored layer by layer. A snapshot that does not fit is refused without
//    harming the session.
//  * failure: a device error after a layer appended its K/V (prefill, decode
//    and restore) invalidates and resets the session; it then reproduces a
//    fresh session bit for bit, so nothing continued on a poisoned cache.
//    Calls rejected up front touch nothing.
//  * concurrency (untapped): several sessions on several threads, with
//    resets, snapshots and restores in the mix, equal their single-threaded
//    results (per-stream scratch).
//  * repeated load/unload of sessions and of the whole model.
#include <algorithm>
#include <atomic>
#include <cstring>
#include <iostream>
#include <thread>

#include "tests/models/laguna/engine_fixture.hpp"

using namespace gufo::testing::laguna;
namespace ml = gufo::models::laguna;

namespace {

using Tokens = std::vector<std::int32_t>;

Tokens RandomTokens(const ml::Model& model, std::size_t n, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  Tokens t(n);
  for (auto& v : t) {
    // Avoid the stop tokens so a sequence never looks finished.
    do {
      v = static_cast<std::int32_t>(rng() % model.VocabSize());
    } while (model.IsStopToken(v));
  }
  return t;
}

std::unique_ptr<ml::Session> NewSession(const ml::Model& model) {
  std::string error;
  auto session = model.CreateSession(&error);
  Require(session != nullptr, "CreateSession: " + error);
  return session;
}

void PrefillOk(ml::Session& s, const Tokens& tokens) {
  std::string error;
  Require(s.Prefill(tokens, &error), "Prefill: " + error);
}

void EvaluateOk(ml::Session& s, std::int32_t token) {
  std::string error;
  Require(s.Evaluate(token, &error), "Evaluate: " + error);
}

bool Same(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() &&
         (a.empty() ||
          std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

/// Logits after each of `steps` decode tokens (taken from `tokens`).
std::vector<std::vector<float>> Decode(ml::Session& s, const Tokens& tokens) {
  std::vector<std::vector<float>> out;
  for (const std::int32_t t : tokens) {
    EvaluateOk(s, t);
    out.push_back(s.Logits());
  }
  return out;
}

bool SameAll(const std::vector<std::vector<float>>& a,
             const std::vector<std::vector<float>>& b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (!Same(a[i], b[i])) {
      return false;
    }
  }
  return true;
}

void CheckKvBookkeeping(ml::Session& s, const ml::SessionSnapshot& snap,
                        const std::string& what) {
  for (std::uint32_t il = 0; il < snap.kv.layers.size(); ++il) {
    const auto& cache = s.executor().kv_cache(il);
    const auto& image = snap.kv.layers[il];
    Require(cache.valid() && cache.committed() == image.committed &&
                cache.retained_begin() == image.retained_begin,
            what + ": layer " + std::to_string(il) +
                " committed/retained_begin/valid differ from the snapshot");
  }
}

// ---- determinism ---------------------------------------------------------

void Determinism(const EngineFixture& f) {
  const Tokens prompt = RandomTokens(*f.model, 50, 1);
  auto a = NewSession(*f.model);
  auto b = NewSession(*f.model);
  PrefillOk(*a, prompt);
  PrefillOk(*b, prompt);
  Require(a->Position() == 50 && Same(a->Logits(), b->Logits()),
          "equal sessions must give bit-equal logits");
  Require(a->Logits().size() == f.model->VocabSize(), "frontier logits size");
  const Tokens more = RandomTokens(*f.model, 8, 2);
  const auto first = Decode(*a, more);
  a->Invalidate();
  Require(a->Position() == 0 && a->Logits().empty() && !a->invalid(),
          "Invalidate() starts the session over");
  PrefillOk(*a, prompt);
  Require(Same(a->Logits(), b->Logits()),
          "after Invalidate() the prompt again");
  Require(SameAll(first, Decode(*a, more)), "decode after Invalidate()");
  std::cout << "determinism ok\n";
}

// ---- snapshots -----------------------------------------------------------

void Snapshots(const EngineFixture& f) {
  const std::size_t kPositions[] = {1, 11, 12, 13, 27, 28, 29, 31, 60, 97};
  for (const std::size_t position : kPositions) {
    const Tokens prompt = RandomTokens(*f.model, position, 100 + position);
    const Tokens tail = RandomTokens(*f.model, 6, 200 + position);
    auto original = NewSession(*f.model);
    PrefillOk(*original, prompt);
    Require(original->SnapshotBytes() > 0, "snapshot size is known up front");
    std::string error;
    const auto snapshot = original->SaveSnapshot(&error);
    Require(snapshot != nullptr, "SaveSnapshot: " + error);
    Require(snapshot->position() == position &&
                snapshot->bytes() == original->SnapshotBytes(),
            "SnapshotBytes() equals the captured size");
    const auto reference = Decode(*original, tail);

    auto fresh = NewSession(*f.model);
    Require(fresh->RestoreSnapshot(*snapshot, &error), "Restore: " + error);
    Require(fresh->Position() == position &&
                Same(fresh->Logits(), snapshot->logits),
            "restore recovers the position and the frontier logits");
    CheckKvBookkeeping(*fresh, *snapshot,
                       "restored at " + std::to_string(position));
    Require(SameAll(reference, Decode(*fresh, tail)),
            "decode after restore into a fresh session at " +
                std::to_string(position));

    // Into a session that already ran on, and whose caches hold other data.
    PrefillOk(*original, RandomTokens(*f.model, 20, 300 + position));
    Require(original->RestoreSnapshot(*snapshot, &error),
            "Restore into used: " + error);
    CheckKvBookkeeping(*original, *snapshot, "restored over a used session");
    Require(SameAll(reference, Decode(*original, tail)),
            "decode after restore over a used session at " +
                std::to_string(position));
  }

  // The ring really wrapped for the long cases, and the snapshot kept only
  // the retained history of the rings.
  {
    auto s = NewSession(*f.model);
    PrefillOk(*s, RandomTokens(*f.model, 97, 7));
    std::string error;
    const auto snapshot = s->SaveSnapshot(&error);
    Require(snapshot != nullptr, error);
    const auto& swa = snapshot->kv.layers[1];
    const auto& full = snapshot->kv.layers[0];
    Require(full.committed == 97 && full.retained_begin == 0,
            "a full layer retains everything");
    Require(swa.committed == 97 && swa.retained_begin == 97 - 28,
            "a ring retains its capacity (window 12 + chunk 16)");
    Require(swa.keys.size() < full.keys.size(),
            "ring images are smaller than full-layer images");
  }

  // A snapshot from a different geometry is refused and harms nothing.
  {
    const auto other =
        MakeEngineFixture({.max_context = 96, .prefill_chunk = 16});
    auto small = NewSession(*other.model);
    PrefillOk(*small, RandomTokens(*other.model, 20, 5));
    std::string error;
    const auto snapshot = small->SaveSnapshot(&error);
    Require(snapshot != nullptr, error);
    auto s = NewSession(*f.model);
    PrefillOk(*s, RandomTokens(*f.model, 10, 6));
    const auto before = s->Logits();
    Require(!s->RestoreSnapshot(*snapshot, &error) && !error.empty(),
            "a snapshot of another geometry is refused");
    Require(s->failures() == 0 && !s->invalid() && s->Position() == 10 &&
                Same(s->Logits(), before),
            "a refused snapshot leaves the session as it was");
  }
  std::cout << "snapshots ok (" << std::size(kPositions) << " positions)\n";
}

// ---- failure -------------------------------------------------------------

void Failure(const EngineFixture& f) {
  const Tokens prompt = RandomTokens(*f.model, 40, 11);
  const Tokens more = RandomTokens(*f.model, 24, 12);
  const Tokens tail = RandomTokens(*f.model, 5, 13);

  // The reference: a fresh session over the same history.
  auto reference = NewSession(*f.model);
  PrefillOk(*reference, prompt);
  const auto expected_prompt_logits = reference->Logits();

  // Failure inside a prefill, after layer 0 appended its K/V of the chunk.
  {
    auto s = NewSession(*f.model);
    PrefillOk(*s, prompt);
    s->InjectFailureForTesting("attn");
    std::string error;
    Require(!s->Prefill(more, &error) && !error.empty(),
            "an injected failure fails the prefill");
    Require(s->failures() == 1 && s->rebuilds() == 1 && !s->invalid() &&
                !s->poisoned(),
            "the failure reset the session at once");
    Require(s->Position() == 0 && s->Logits().empty(),
            "a failed session starts over from position 0");
    Require(!s->SaveSnapshot(&error), "nothing to snapshot after a reset");
    PrefillOk(*s, prompt);
    Require(Same(s->Logits(), expected_prompt_logits),
            "after the reset the session equals a fresh one");
    const auto expected_tail = Decode(*reference, tail);
    Require(SameAll(expected_tail, Decode(*s, tail)),
            "decode after a failure equals a fresh session");
    // Rebuild the reference for the next scenarios.
    reference = NewSession(*f.model);
    PrefillOk(*reference, prompt);
  }

  // Failure in the middle of a decode step.
  {
    auto s = NewSession(*f.model);
    PrefillOk(*s, prompt);
    EvaluateOk(*s, tail[0]);
    s->InjectFailureForTesting("o");
    std::string error;
    Require(!s->Evaluate(tail[1], &error), "an injected decode failure");
    Require(s->Position() == 0 && s->rebuilds() == 1 && !s->poisoned(),
            "decode failure resets the session");
    PrefillOk(*s, prompt);
    Require(Same(s->Logits(), expected_prompt_logits),
            "state after a decode failure is clean");
  }

  // Failure while a multi-chunk prefill is already partly done.
  {
    f.model->InjectForwardFailureForTesting(2, "attn");
    auto s = NewSession(*f.model);
    std::string error;
    Require(!s->Prefill(RandomTokens(*f.model, 80, 14), &error),
            "the third chunk of a prefill fails");
    Require(s->Position() == 0 && s->failures() == 1 && !s->poisoned(),
            "a failure after earlier chunks still resets everything");
    PrefillOk(*s, prompt);
    Require(Same(s->Logits(), expected_prompt_logits),
            "state after a partly completed prefill failed is clean");
  }

  // Failure of the pass a restore is followed by: a restored session that
  // then fails is reset, not continued.
  {
    auto source = NewSession(*f.model);
    PrefillOk(*source, prompt);
    std::string error;
    const auto snapshot = source->SaveSnapshot(&error);
    Require(snapshot != nullptr, error);
    auto s = NewSession(*f.model);
    Require(s->RestoreSnapshot(*snapshot, &error), error);
    s->InjectFailureForTesting("attn");
    Require(!s->Evaluate(tail[0], &error), "failure after a restore");
    Require(s->Position() == 0 && s->Logits().empty(),
            "a restored session that fails is reset");
    Require(s->RestoreSnapshot(*snapshot, &error),
            "the snapshot restores again after the reset");
    Require(Same(s->Logits(), snapshot->logits), "restored frontier");
  }

  // Calls rejected up front touch nothing.
  {
    auto s = NewSession(*f.model);
    PrefillOk(*s, prompt);
    const auto before = s->Logits();
    std::string error;
    Require(!s->Evaluate(-1, &error) &&
                !s->Evaluate(static_cast<std::int32_t>(f.model->VocabSize()),
                             &error),
            "tokens outside the vocabulary are refused");
    Require(!s->Prefill(RandomTokens(*f.model, f.model->MaxContext() + 8, 15),
                        &error) &&
                error.find("context") != std::string::npos,
            "a context overrun is refused");
    Require(s->Prefill({}, &error), "an empty prefill is a no-op");
    Require(s->failures() == 0 && !s->invalid() && s->Position() == 40 &&
                Same(s->Logits(), before),
            "rejections leave the session untouched and valid");
    // The context fills exactly, then refuses.
    Require(s->Prefill(RandomTokens(*f.model, f.model->MaxContext() - 40, 16),
                       &error),
            "the context can be filled exactly: " + error);
    Require(!s->Evaluate(1, &error) && s->failures() == 0 &&
                s->Position() == f.model->MaxContext(),
            "a full context refuses one more token without failing");
  }
  std::cout << "failure and reset ok\n";
}

// ---- concurrency ---------------------------------------------------------

void Concurrency(const EngineFixture& f) {
  constexpr int kThreads = 4;
  constexpr int kRounds = 3;
  struct Work {
    Tokens prompt, tail;
    std::vector<std::vector<float>> expected;
    std::vector<float> expected_prompt;
  };
  std::vector<Work> work(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    work[t].prompt = RandomTokens(*f.model, 30 + 9 * t, 500 + t);
    work[t].tail = RandomTokens(*f.model, 12, 600 + t);
    auto s = NewSession(*f.model);
    PrefillOk(*s, work[t].prompt);
    work[t].expected_prompt = s->Logits();
    work[t].expected = Decode(*s, work[t].tail);
  }
  std::vector<std::unique_ptr<ml::Session>> sessions;
  for (int t = 0; t < kThreads; ++t) {
    sessions.push_back(NewSession(*f.model));
  }
  std::atomic<int> failures{0};
  std::atomic<bool> go{false};
  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t] {
      try {
        while (!go.load()) {
          std::this_thread::yield();
        }
        ml::Session& s = *sessions[t];
        for (int round = 0; round < kRounds; ++round) {
          s.Invalidate();
          PrefillOk(s, work[t].prompt);
          if (!Same(s.Logits(), work[t].expected_prompt)) {
            ++failures;
          }
          std::string error;
          const auto snapshot = s.SaveSnapshot(&error);
          Require(snapshot != nullptr, error);
          if (!SameAll(work[t].expected, Decode(s, work[t].tail))) {
            ++failures;
          }
          // Restore and decode again, while the other threads run.
          Require(s.RestoreSnapshot(*snapshot, &error), error);
          if (!SameAll(work[t].expected, Decode(s, work[t].tail))) {
            ++failures;
          }
        }
      } catch (const std::exception& e) {
        std::cerr << "thread " << t << ": " << e.what() << '\n';
        ++failures;
      }
    });
  }
  go = true;
  for (auto& th : threads) {
    th.join();
  }
  Require(failures.load() == 0,
          std::to_string(failures.load()) +
              " concurrent session result(s) differ from the single-threaded "
              "baseline");
  std::cout << "concurrency ok (" << kThreads << " threads x " << kRounds
            << " rounds)\n";
}

// ---- load / unload -------------------------------------------------------

void LoadUnload() {
  const auto fixture = MakeEngineFixture();
  const Tokens prompt = RandomTokens(*fixture.model, 33, 21);
  std::vector<float> first;
  for (int cycle = 0; cycle < 6; ++cycle) {
    std::vector<std::unique_ptr<ml::Session>> sessions;
    for (int i = 0; i < 3; ++i) {
      sessions.push_back(NewSession(*fixture.model));
    }
    PrefillOk(*sessions[cycle % 3], prompt);
    if (cycle == 0) {
      first = sessions[0]->Logits();
    } else if (cycle % 3 == 0) {
      Require(Same(first, sessions[0]->Logits()),
              "a session of cycle " + std::to_string(cycle) +
                  " differs from the first");
    }
    // Destroyed here in a different order every cycle.
    std::rotate(sessions.begin(), sessions.begin() + (cycle % 3),
                sessions.end());
    sessions.clear();
  }
  // A whole model, uploaded again each time.
  std::vector<float> logits_of_first_model;
  for (int cycle = 0; cycle < 3; ++cycle) {
    const auto again = MakeEngineFixture();
    auto s = NewSession(*again.model);
    PrefillOk(*s, prompt);
    if (cycle == 0) {
      logits_of_first_model = s->Logits();
    } else {
      Require(Same(logits_of_first_model, s->Logits()),
              "a re-uploaded model gives the same logits");
    }
  }
  // Sessions keep their model alive: the last reference may be a session.
  {
    auto doomed = MakeEngineFixture();
    auto s = NewSession(*doomed.model);
    doomed.model.reset();
    doomed.tiny.reset();
    PrefillOk(*s, prompt);
  }
  std::cout << "load/unload ok\n";
}

// ---- model api -----------------------------------------------------------

void ModelApi(const EngineFixture& f) {
  const auto& m = *f.model;
  Require(m.IsStopToken(2) && m.IsStopToken(24) && !m.IsStopToken(23) &&
              !m.IsStopToken(-1) && !m.IsStopToken(19),
          "stop tokens are EOS and EOT only");
  const std::string text = "the then there";
  Require(m.Decode(m.Tokenize(text)) == text, "tokenize/decode round trip");
  const auto with_bos = m.Tokenize(text, true);
  Require(with_bos.size() == m.Tokenize(text).size() + 1 && with_bos[0] == 2,
          "add_bos prepends exactly one BOS");
  Require(m.Tokenize("</think>", false, true) == Tokens{19} &&
              m.Tokenize("</assistant>", false, true) == Tokens{24} &&
              m.Tokenize("</assistant>", false, false) != Tokens{24},
          "control text parses as tokens only when asked");
  std::vector<ml::ChatMessage> messages(1);
  messages[0].role = "user";
  messages[0].content = "hello";
  const auto ids = m.EncodeChat(messages, {}, {});
  Require(ids.size() > 8 && ids.front() == 2 &&
              std::count(ids.begin(), ids.end(), 2) == 1,
          "a chat prompt starts with exactly one BOS");
  Require(ids.back() == 18 && ids[ids.size() - 2] == 23,
          "a thinking generation prompt ends with <assistant><think>");
  ml::ChatTemplateOptions off;
  off.enable_thinking = false;
  const auto plain = m.EncodeChat(messages, {}, off);
  Require(plain.back() == 19 && plain[plain.size() - 2] == 23,
          "without thinking it ends with <assistant></think>");
  Require(m.DecodeToken(24) == "</assistant>" && m.DecodeToken(-5).empty() &&
              m.DecodeToken(100000).empty(),
          "DecodeToken");

  // Options are validated against the artifact.
  std::string error;
  const auto bad = [&](ml::ModelOptions options) {
    std::vector<std::uint8_t> storage;
    const auto reader = OpenSynthetic(TinyTokenizerMetadata(640), &storage);
    auto tokenizer = ml::Tokenizer::FromGguf(*reader, &error);
    auto weights = std::make_unique<ml::ModelWeights>(f.tiny->weights());
    auto device = ml::rocm::DeviceModel::Upload(*weights, nullptr, &error);
    return ml::Model::FromParts(f.tiny, std::move(weights), std::move(device),
                                std::move(tokenizer), options, "x",
                                &error) == nullptr;
  };
  Require(bad({.max_context = 1}) && bad({.max_context = 262145}) &&
              bad({.prefill_chunk = 0}) && bad({.prefill_chunk = 4096}) &&
              bad({.session_count = 0}),
          "invalid model options are rejected");
  Require(!bad({.max_context = 512, .prefill_chunk = 16}),
          "valid model options are accepted");
  std::cout << "model api ok\n";
}

}  // namespace

int main() {
  try {
    const EngineFixture fixture = MakeEngineFixture();
    Determinism(fixture);
    Snapshots(fixture);
    Failure(fixture);
    Concurrency(fixture);
    ModelApi(fixture);
    LoadUnload();
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
  std::cout << "laguna.engine: ok\n";
  return 0;
}
