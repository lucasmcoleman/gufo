// Batched (multi-session) decode on tiny random Laguna models (no model file,
// nothing skipped): Session::EvaluateBatch / rocm::Executor::DecodeBatch,
// which advance several sessions by one token in shared forward passes. The
// models use the real geometry and a 12-position window, prefill chunks of 16
// (sliding rings of 28 positions), so sessions sit on both sides of the
// window and wrap their rings during the runs.
//
//  * identity: for N = 2, 3, 5, 8 and beyond one group (17, 32), sessions at
//    different positions around the window (11/12/13), the ring capacity
//    (27/28/29) and far past it, decoded through ring wrap, give the same
//    logits bit for bit and the same K/V bytes and bookkeeping as the same
//    sessions decoded one at a time.
//  * membership: a session finishing early, a peer rejoining later, a changing
//    first session (the one that lends its stream) and peers of different
//    lengths all keep every session equal to its solo twin.
//  * rejection: a session that cannot take its token (outside the vocabulary,
//    context full) is refused alone; it and its peers are untouched and the
//    peers still equal their solo twins.
//  * failure: an injected device error inside a shared pass (session-level on
//    the first or a later session, and model-wide) invalidates and resets
//    exactly the sessions of that pass; a session outside the batch is
//    untouched; afterwards every session reproduces a fresh one bit for bit.
//  * untapped: all of the above run untapped; a layer-tapped run and repeated
//    runs are bit-identical, and two batches on two threads (two streams) equal
//    their solo twins.
#include <algorithm>
#include <atomic>
#include <cstring>
#include <iostream>
#include <random>
#include <thread>

#include "tests/models/laguna/engine_fixture.hpp"

using namespace gufo::testing::laguna;
namespace ml = gufo::models::laguna;

namespace {

using Tokens = std::vector<std::int32_t>;
using SessionPtr = std::unique_ptr<ml::Session>;

Tokens RandomTokens(const ml::Model& model, std::size_t n, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  Tokens t(n);
  for (auto& v : t) {
    do {
      v = static_cast<std::int32_t>(rng() % model.VocabSize());
    } while (model.IsStopToken(v));
  }
  return t;
}

SessionPtr NewSession(const ml::Model& model) {
  std::string error;
  auto session = model.CreateSession(&error);
  Require(session != nullptr, "CreateSession: " + error);
  return session;
}

void PrefillOk(ml::Session& s, const Tokens& tokens) {
  std::string error;
  Require(s.Prefill(tokens, &error), "Prefill: " + error);
}

bool Same(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() &&
         (a.empty() ||
          std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

bool SameBytes(const std::vector<std::uint8_t>& a,
               const std::vector<std::uint8_t>& b) {
  return a.size() == b.size() &&
         (a.empty() || std::memcmp(a.data(), b.data(), a.size()) == 0);
}

/// Same position, same frontier logits, same K/V bytes and bookkeeping in
/// every layer.
void RequireEqualState(ml::Session& a, ml::Session& b,
                       const std::string& what) {
  Require(a.Position() == b.Position(), what + ": positions differ");
  Require(Same(a.Logits(), b.Logits()), what + ": logits differ");
  if (a.Position() == 0) {
    return;
  }
  std::string error;
  const auto sa = a.SaveSnapshot(&error);
  Require(sa != nullptr, what + ": snapshot: " + error);
  const auto sb = b.SaveSnapshot(&error);
  Require(sb != nullptr, what + ": snapshot: " + error);
  for (std::size_t il = 0; il < sa->kv.layers.size(); ++il) {
    const auto& x = sa->kv.layers[il];
    const auto& y = sb->kv.layers[il];
    Require(x.committed == y.committed && x.retained_begin == y.retained_begin,
            what + ": layer " + std::to_string(il) + " bookkeeping differs");
    Require(SameBytes(x.keys, y.keys) && SameBytes(x.values, y.values),
            what + ": layer " + std::to_string(il) + " K/V bytes differ");
  }
}

/// Steps `members` (indices into `batch`) together with EvaluateBatch and the
/// twins in `solo` one by one, with the same tokens, and compares after every
/// step.
void Step(std::vector<SessionPtr>& batch, std::vector<SessionPtr>& solo,
          const std::vector<std::size_t>& members, const Tokens& tokens,
          const std::string& what) {
  std::vector<ml::Session::BatchItem> items(members.size());
  for (std::size_t i = 0; i < members.size(); ++i) {
    items[i].session = batch[members[i]].get();
    items[i].token = tokens[members[i]];
  }
  ml::Session::EvaluateBatch(items);
  for (std::size_t i = 0; i < members.size(); ++i) {
    Require(items[i].completed,
            what + ": batched step failed: " + items[i].error);
    std::string error;
    Require(solo[members[i]]->Evaluate(tokens[members[i]], &error),
            what + ": solo step failed: " + error);
    Require(Same(batch[members[i]]->Logits(), solo[members[i]]->Logits()) &&
                batch[members[i]]->Position() == solo[members[i]]->Position(),
            what + ": session " + std::to_string(members[i]) + " at position " +
                std::to_string(solo[members[i]]->Position()) +
                " differs from its solo twin");
  }
}

struct Group {
  std::vector<SessionPtr> batch, solo;
  std::vector<Tokens> prompts;
};

Group MakeGroup(const ml::Model& model, const std::vector<std::size_t>& starts,
                std::uint64_t seed) {
  Group g;
  for (std::size_t i = 0; i < starts.size(); ++i) {
    g.prompts.push_back(RandomTokens(model, starts[i], seed + i));
    g.batch.push_back(NewSession(model));
    g.solo.push_back(NewSession(model));
    if (starts[i] > 0) {
      PrefillOk(*g.batch.back(), g.prompts.back());
      PrefillOk(*g.solo.back(), g.prompts.back());
    }
  }
  return g;
}

std::vector<std::size_t> Iota(std::size_t n) {
  std::vector<std::size_t> v(n);
  for (std::size_t i = 0; i < n; ++i) {
    v[i] = i;
  }
  return v;
}

// ---- identity -----------------------------------------------------------

void Identity(const ml::Model& model, const std::vector<std::size_t>& starts,
              int steps, std::uint64_t seed) {
  // Sessions need a first token: position 0 sessions are started solo-style
  // by the first step, so starts here are positive.
  Group g = MakeGroup(model, starts, seed);
  const Tokens feed = RandomTokens(model, steps * starts.size(), seed + 99);
  for (int s = 0; s < steps; ++s) {
    Tokens tokens(starts.size());
    for (std::size_t i = 0; i < starts.size(); ++i) {
      tokens[i] = feed[s * starts.size() + i];
    }
    Step(g.batch, g.solo, Iota(starts.size()), tokens,
         "N=" + std::to_string(starts.size()) + " step " + std::to_string(s));
  }
  for (std::size_t i = 0; i < starts.size(); ++i) {
    RequireEqualState(*g.batch[i], *g.solo[i],
                      "N=" + std::to_string(starts.size()) + " end state " +
                          std::to_string(i));
  }
}

void IdentityCases(const ml::Model& model) {
  const std::vector<std::vector<std::size_t>> cases = {
      {11, 12},
      {27, 28},
      {5, 90},
      {10, 13, 28},
      {1, 29, 60},
      {12, 27, 29},
      {3, 11, 12, 13, 28},
      {27, 29, 31, 60, 97},
      {1, 2, 3, 4, 5},
      {1, 9, 11, 12, 13, 26, 27, 28},
  };
  for (const auto& starts : cases) {
    // 40 steps: every session crosses the window and wraps its ring at least
    // once (28 positions) from the lowest start.
    Identity(model, starts, 40, 1000 + starts.size() * 7 + starts.front());
  }
  std::cout << "identity ok (" << cases.size()
            << " position sets, N = 2..8, 40 steps each)\n";
}

void WideIdentity(const ml::Model& model, std::size_t n) {
  std::vector<std::size_t> starts(n);
  for (std::size_t i = 0; i < n; ++i) {
    starts[i] = 1 + (i * 7) % 45;
  }
  Identity(model, starts, 8, 5000 + n);
  std::cout << "identity ok (N = " << n << ", chunk " << model.PrefillChunk()
            << ")\n";
}

// ---- membership ---------------------------------------------------------

void Membership(const ml::Model& model) {
  Group g = MakeGroup(model, {10, 26, 40, 13}, 2000);
  const Tokens feed = RandomTokens(model, 4 * 60, 2100);
  int cursor = 0;
  const auto next_tokens = [&] {
    Tokens t(4);
    for (auto& v : t) {
      v = feed[cursor++];
    }
    return t;
  };
  // Everybody, then the first session finishes (the lender changes), then
  // the second leaves and the first rejoins, then two alone, then one alone.
  for (int s = 0; s < 12; ++s) {
    Step(g.batch, g.solo, {0, 1, 2, 3}, next_tokens(),
         "all " + std::to_string(s));
  }
  for (int s = 0; s < 12; ++s) {
    Step(g.batch, g.solo, {1, 2, 3}, next_tokens(), "without 0");
  }
  for (int s = 0; s < 12; ++s) {
    Step(g.batch, g.solo, {3, 0, 2}, next_tokens(), "reordered, 1 gone");
  }
  for (int s = 0; s < 8; ++s) {
    Step(g.batch, g.solo, {2, 3}, next_tokens(), "two left");
  }
  for (int s = 0; s < 4; ++s) {
    Step(g.batch, g.solo, {3}, next_tokens(), "one left");
  }
  for (int s = 0; s < 6; ++s) {
    Step(g.batch, g.solo, {0, 1, 2, 3}, next_tokens(), "all again");
  }
  for (std::size_t i = 0; i < 4; ++i) {
    RequireEqualState(*g.batch[i], *g.solo[i],
                      "membership end " + std::to_string(i));
  }
  std::cout << "membership ok (early finish, rejoin, changing lender)\n";
}

// ---- rejection ----------------------------------------------------------

void Rejection(const ml::Model& model) {
  Group g = MakeGroup(model, {14, 30, 9}, 3000);
  const Tokens feed = RandomTokens(model, 3 * 10, 3100);
  // A session whose context is full, and one given a token outside the
  // vocabulary, are refused alone.
  auto full = NewSession(model);
  PrefillOk(*full, RandomTokens(model, model.MaxContext(), 3200));
  const auto full_logits = full->Logits();
  for (int s = 0; s < 10; ++s) {
    Tokens t = {feed[3 * s], feed[3 * s + 1], feed[3 * s + 2]};
    std::vector<ml::Session::BatchItem> items(5);
    items[0] = {g.batch[0].get(), t[0], false, {}};
    items[1] = {full.get(), 7, false, {}};
    items[2] = {g.batch[1].get(), t[1], false, {}};
    items[3] = {g.batch[2].get(), s == 4 ? -3 : t[2], false, {}};
    items[4] = {g.batch[2].get(),  // twice in one batch
                s == 4 ? -3 : t[2],
                false,
                {}};
    const auto before2 = g.batch[2]->Logits();
    const auto pos2 = g.batch[2]->Position();
    ml::Session::EvaluateBatch(items);
    Require(!items[1].completed && !items[1].error.empty() &&
                full->failures() == 0 && !full->invalid() &&
                full->Position() == model.MaxContext() &&
                Same(full->Logits(), full_logits),
            "a full-context session is refused and untouched");
    Require(!items[4].completed && !items[4].error.empty(),
            "a session twice in a batch: the second entry is refused");
    if (s == 4) {
      Require(!items[3].completed && g.batch[2]->Position() == pos2 &&
                  Same(g.batch[2]->Logits(), before2) &&
                  g.batch[2]->failures() == 0,
              "a bad token is refused and leaves its session untouched");
    } else {
      Require(items[3].completed, "valid session in a mixed batch");
    }
    Require(items[0].completed && items[2].completed,
            "peers of a refused session still run");
    std::string error;
    Require(g.solo[0]->Evaluate(t[0], &error), error);
    Require(g.solo[1]->Evaluate(t[1], &error), error);
    if (s != 4) {
      Require(g.solo[2]->Evaluate(t[2], &error), error);
    }
    for (std::size_t i = 0; i < 3; ++i) {
      Require(Same(g.batch[i]->Logits(), g.solo[i]->Logits()) &&
                  g.batch[i]->Position() == g.solo[i]->Position(),
              "peer " + std::to_string(i) + " differs from its solo twin");
    }
  }
  for (std::size_t i = 0; i < 3; ++i) {
    RequireEqualState(*g.batch[i], *g.solo[i],
                      "rejection end " + std::to_string(i));
  }
  // A batch that ends up with one runnable session runs it as a lone decode.
  {
    std::vector<ml::Session::BatchItem> items(2);
    items[0] = {g.batch[0].get(), 5, false, {}};
    items[1] = {full.get(), 5, false, {}};
    ml::Session::EvaluateBatch(items);
    std::string error;
    Require(items[0].completed && !items[1].completed &&
                g.solo[0]->Evaluate(5, &error) &&
                Same(g.batch[0]->Logits(), g.solo[0]->Logits()),
            "one runnable session equals a lone decode");
  }
  std::cout << "rejection ok\n";
}

// ---- failure ------------------------------------------------------------

enum class Fault { kLeaderSession, kLaterSession, kModelWide };

void FailureCase(const ml::Model& model, Fault fault, const char* stage) {
  const std::vector<std::size_t> starts = {13, 28, 40, 7};
  Group g = MakeGroup(model, starts, 4000);
  Group outside = MakeGroup(model, {22}, 4500);
  const Tokens feed = RandomTokens(model, 4 * 8, 4100);
  const auto tokens_at = [&](int s) {
    Tokens t(4);
    for (std::size_t i = 0; i < 4; ++i) {
      t[i] = feed[4 * s + i];
    }
    return t;
  };
  // A few clean shared steps first.
  for (int s = 0; s < 3; ++s) {
    Step(g.batch, g.solo, {0, 1, 2}, tokens_at(s), "before the fault");
  }
  const auto outside_logits = outside.batch[0]->Logits();
  const auto outside_position = outside.batch[0]->Position();

  switch (fault) {
    case Fault::kLeaderSession:
      g.batch[0]->InjectFailureForTesting(stage);
      break;
    case Fault::kLaterSession:
      g.batch[2]->InjectFailureForTesting(stage);
      break;
    case Fault::kModelWide:
      model.InjectForwardFailureForTesting(0, stage);
      break;
  }
  const Tokens t = tokens_at(3);
  std::vector<ml::Session::BatchItem> items(3);
  for (std::size_t i = 0; i < 3; ++i) {
    items[i] = {g.batch[i].get(), t[i], false, {}};
  }
  ml::Session::EvaluateBatch(items);
  for (std::size_t i = 0; i < 3; ++i) {
    Require(!items[i].completed && !items[i].error.empty(),
            "every session of the failed pass fails");
    Require(g.batch[i]->failures() == 1 && g.batch[i]->rebuilds() == 1 &&
                !g.batch[i]->invalid() && !g.batch[i]->poisoned() &&
                g.batch[i]->Position() == 0 && g.batch[i]->Logits().empty(),
            "every session of the failed pass was reset at once");
  }
  // The session that sat the pass out was never touched.
  Require(outside.batch[0]->failures() == 0 &&
              outside.batch[0]->Position() == outside_position &&
              Same(outside.batch[0]->Logits(), outside_logits),
          "a session outside the batch is untouched");
  // A bystander of the same model that was not in the failed pass, and the
  // twins, then keep working: replay everything on the reset sessions and
  // require the fresh twins' results, shared steps included.
  Group again = MakeGroup(model, starts, 4000);
  for (std::size_t i = 0; i < 3; ++i) {
    PrefillOk(*g.batch[i], g.prompts[i]);
    Require(Same(g.batch[i]->Logits(), again.solo[i]->Logits()),
            "after the reset the prompt matches a fresh session");
  }
  for (int s = 0; s < 6; ++s) {
    Step(g.batch, again.solo, {0, 1, 2}, tokens_at(s), "after the fault");
  }
  for (std::size_t i = 0; i < 3; ++i) {
    RequireEqualState(*g.batch[i], *again.solo[i],
                      "after the fault, session " + std::to_string(i));
  }
  // The outside session joins a batch with a recovered one.
  {
    std::vector<ml::Session::BatchItem> mixed(2);
    mixed[0] = {outside.batch[0].get(), 5, false, {}};
    mixed[1] = {g.batch[1].get(), 6, false, {}};
    ml::Session::EvaluateBatch(mixed);
    std::string error;
    Require(mixed[0].completed && mixed[1].completed &&
                outside.solo[0]->Evaluate(5, &error) &&
                again.solo[1]->Evaluate(6, &error) &&
                Same(outside.batch[0]->Logits(), outside.solo[0]->Logits()) &&
                Same(g.batch[1]->Logits(), again.solo[1]->Logits()),
            "the untouched session and a recovered one share a pass");
  }
}

void Failure(const ml::Model& model) {
  const char* const stages[] = {"attn", "o", "gate_proj"};
  for (const char* stage : stages) {
    FailureCase(model, Fault::kLeaderSession, stage);
    FailureCase(model, Fault::kLaterSession, stage);
    FailureCase(model, Fault::kModelWide, stage);
  }
  std::cout << "failure ok (leader, later and model-wide faults x "
            << std::size(stages) << " stages)\n";
}

// ---- untapped, tapped, threads ------------------------------------------

std::vector<std::vector<float>> RunScript(const ml::Model& model, bool tapped,
                                          std::size_t* tap_rows) {
  Group g = MakeGroup(model, {11, 29, 50}, 6000);
  if (tapped) {
    g.batch[0]->executor().SetLayerTap(
        [tap_rows](std::uint32_t, const float*, std::size_t count) {
          *tap_rows += count;
        });
  }
  const Tokens feed = RandomTokens(model, 3 * 30, 6100);
  std::vector<std::vector<float>> out;
  for (int s = 0; s < 30; ++s) {
    std::vector<ml::Session::BatchItem> items(3);
    for (std::size_t i = 0; i < 3; ++i) {
      items[i] = {g.batch[i].get(), feed[3 * s + i], false, {}};
    }
    ml::Session::EvaluateBatch(items);
    for (std::size_t i = 0; i < 3; ++i) {
      Require(items[i].completed, "scripted step: " + items[i].error);
      out.push_back(g.batch[i]->Logits());
    }
  }
  return out;
}

void UntappedAndTapped(const ml::Model& model) {
  std::size_t none = 0;
  std::size_t tap_rows = 0;
  const auto first = RunScript(model, false, &none);
  const auto second = RunScript(model, false, &none);
  const auto tapped = RunScript(model, true, &tap_rows);
  Require(first.size() == second.size() && first.size() == tapped.size(),
          "script size");
  for (std::size_t i = 0; i < first.size(); ++i) {
    Require(Same(first[i], second[i]), "an untapped rerun is bit-identical");
    Require(Same(first[i], tapped[i]), "a tapped run is bit-identical");
  }
  // The first session lent its stream: its tap saw all three rows of every
  // layer of every step (a solo decode would show one row per layer).
  Require(tap_rows == std::size_t{30} * model.config().num_layers * 3 *
                          model.config().hidden_size,
          "the layer tap saw the batched rows");
  std::cout << "untapped reruns and tapped run bit-identical\n";
}

void Threads(const ml::Model& model) {
  constexpr int kThreads = 2;
  std::vector<Group> groups;
  for (int t = 0; t < kThreads; ++t) {
    groups.push_back(
        MakeGroup(model, {9 + t, 27 + t, 31 + 3 * t}, 7000 + t * 10));
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
        const Tokens feed = RandomTokens(model, 3 * 40, 7500 + t);
        for (int s = 0; s < 40; ++s) {
          Tokens tokens = {feed[3 * s], feed[3 * s + 1], feed[3 * s + 2]};
          Step(groups[t].batch, groups[t].solo, {0, 1, 2}, tokens,
               "thread " + std::to_string(t));
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
  Require(failures.load() == 0, "concurrent batches equal their solo twins");
  std::cout << "concurrent batches ok (" << kThreads << " threads)\n";
}

}  // namespace

int main() {
  try {
    const EngineFixture fixture = MakeEngineFixture();
    IdentityCases(*fixture.model);
    // Wider than one group of this fixture's chunk (16): 17 splits into 16+1.
    WideIdentity(*fixture.model, 16);
    WideIdentity(*fixture.model, 17);
    const EngineFixture wide =
        MakeEngineFixture({.max_context = 512, .prefill_chunk = 32});
    WideIdentity(*wide.model, 32);
    Membership(*fixture.model);
    Rejection(*fixture.model);
    Failure(*fixture.model);
    UntappedAndTapped(*fixture.model);
    Threads(*fixture.model);
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
  std::cout << "laguna.batch: ok\n";
  return 0;
}
