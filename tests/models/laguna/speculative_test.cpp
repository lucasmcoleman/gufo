// Laguna DFlash speculative decoding (phase 6e of DFLASH.md) on tiny random
// models (no model file, nothing skipped): a six-layer target with the real
// head geometry and a 512-position window (so 509..515 straddle the real
// boundary), a three-layer drafter of the real head geometry, and the serving
// engine's Session::DecodeSpeculative over the shared SpeculativeVerifier.
//
// The reference is plain greedy decode on a second Model of the same weights
// that has no drafter at all (Session::Evaluate, the serve path's decode).
// Every speculative run must equal it token for token, end at the same
// position and leave bit-identical frontier logits.
//
//  * exact greedy: prompts of 9, 509..515, 600, 2048 and 8192 tokens, chat
//    prompts with thinking on and off; the random tiny drafter (low
//    acceptance), an oracle drafter (the reference continuation: everything
//    accepted) and a corrupted oracle (partial acceptance); blocks of 4, 8, 16.
//  * stop tokens inside a draft: EOS 2 and EOT 24 are made the greedy winner
//    where another token would win (aliased head rows), with the oracle
//    drafting through them; stop_at_eos on and off.
//  * max_tokens inside a draft: budgets 1, 2, 3, 5 and 9 at block 8.
//  * deterministic requests that are not plain argmax: a Laguna tool-call
//    grammar run through the draft prefix (drafted and accepted, so the
//    opt-in verifier path is exercised), repeat penalties, top_k 1.
//  * snapshot and restore mid-request: the drafter ring travels with the
//    snapshot (drafts after restore equal the original's bit for bit); a
//    snapshot without the ring restores a cold ring, never the stale one.
//  * negative: a verify at the wrong anchor position, a commit past or off
//    the verified rows, random sampling, a target failure mid-step (session
//    reset, drafter re-armed), the drafter on a poisoned target, a drafter
//    failure mid-request (decoding continues target-only, still exact).
//  * sessions: EvaluateBatch with drafting and suspended sessions mixed;
//    two and eight sessions on threads (speculating greedy and plain sampled
//    mixed) equal their solo runs.
#include "src/models/laguna/speculative.hpp"

#include <atomic>
#include <cstring>
#include <iostream>
#include <map>
#include <optional>
#include <random>
#include <thread>

#include "src/core/json.hpp"
#include "src/core/json_constraint.hpp"
#include "src/models/laguna/engine.hpp"
#include "src/models/laguna/kernels/dflash_session.hpp"
#include "tests/models/laguna/dflash_gpu.hpp"
#include "tests/models/laguna/engine_fixture.hpp"
#include "tests/models/laguna/speculative_fixture.hpp"

using namespace gufo::testing::laguna;

namespace {

/// The real head geometry at hidden 512; vocabulary of the tiny target.
constexpr DrafterShape kShape{"spec", 512, 768, 18,  2,
                              3,      512, 3,   640, 500000.0F};
constexpr std::uint32_t kContext = 9216;

/// A Model over `tiny`'s weights (its own device copy).
std::shared_ptr<ml::Model> BuildModel(const std::shared_ptr<TinyModel>& tiny,
                                      ml::ModelOptions options) {
  std::vector<std::uint8_t> storage;
  const auto reader =
      OpenSynthetic(TinyTokenizerMetadata(tiny->spec().vocab), &storage);
  std::string error;
  auto tokenizer = ml::Tokenizer::FromGguf(*reader, &error);
  Require(tokenizer != nullptr, "tiny tokenizer: " + error);
  auto weights = std::make_unique<ml::ModelWeights>(tiny->weights());
  auto device = gpu::DeviceModel::Upload(*weights, nullptr, &error);
  Require(device != nullptr, "tiny upload: " + error);
  auto model = ml::Model::FromParts(tiny, std::move(weights), std::move(device),
                                    std::move(tokenizer), options,
                                    "Laguna-tiny", &error);
  Require(model != nullptr, "tiny model: " + error);
  return model;
}

struct Fixture {
  std::shared_ptr<TinyModel> tiny;
  std::unique_ptr<TinyDrafter> drafter;
  std::shared_ptr<ml::Model> plain;  ///< no drafter: the reference
  std::shared_ptr<ml::Model> spec;   ///< drafter attached
};

ModelSpec TargetSpec() {
  ModelSpec spec;
  spec.window = 512;
  spec.hidden = 512;
  return spec;
}

/// `alias`: head rows (dst, src) made equal before upload.
Fixture MakeFixture(
    std::uint32_t block, std::uint32_t chunk,
    const std::vector<std::pair<std::uint32_t, std::uint32_t>>& alias = {},
    std::uint32_t sessions = 8) {
  Fixture f;
  f.tiny = std::make_shared<TinyModel>(TargetSpec());
  for (const auto& [dst, src] : alias) {
    f.tiny->AliasHeadRow(dst, src);
  }
  const ml::ModelOptions options{.max_context = kContext,
                                 .prefill_chunk = chunk,
                                 .session_count = sessions};
  f.plain = BuildModel(f.tiny, options);
  f.spec = BuildModel(f.tiny, options);
  f.drafter = std::make_unique<TinyDrafter>(kShape, 0x5EC6);
  // Layer-input numbering ending at the target's layer count (6): the
  // drafter reads the outputs of layers 1, 3 and 5.
  f.drafter->drafter.config.target_layers = {2, 4, 6};
  std::string error;
  Require(
      f.spec->AttachDrafterWeights(
          f.drafter->drafter, ml::DrafterOptions{.block_rows = block}, &error),
      "attach drafter: " + error);
  Require(!f.spec->AttachDrafterWeights(f.drafter->drafter, {}, &error),
          "a second drafter was attached");
  return f;
}

enum class Drafts { kDrafter, kOracle, kCorrupt };

/// Drafts continuing `reference` (the tokens after the prompt, plus what
/// follows them) from position p: exact, or with every third block corrupted
/// at a pseudo-random row.
ml::LagunaDraftBackend::DraftOverride Oracle(std::size_t prompt_len,
                                             Tokens reference, bool corrupt) {
  return [prompt_len, reference = std::move(reference), corrupt](
             std::uint32_t pos, std::int32_t anchor, std::uint32_t max) {
    (void)anchor;
    Tokens d;
    for (std::uint32_t i = 0; i < max; ++i) {
      const std::size_t k = pos - prompt_len + 1 + i;
      d.push_back(k < reference.size() ? reference[k] : 300 + (i % 7));
    }
    if (corrupt && pos % 3 == 0 && !d.empty()) {
      const std::size_t row = (pos / 3) % d.size();
      d[row] = d[row] == 77 ? 78 : 77;
    }
    return d;
  };
}

Run SpecRun(ml::Model& m, const Tokens& prompt, std::uint32_t max,
            const smp::SamplingConfig& config, std::uint32_t block,
            Drafts drafts, const Tokens& reference, bool stop_at_eos = true) {
  std::string error;
  auto s = m.CreateSession(&error);
  Require(s != nullptr && s->speculative(), "spec session: " + error);
  s->decoder()->set_block_rows(block);
  if (drafts != Drafts::kDrafter) {
    s->decoder()->backend().SetDraftOverrideForTesting(
        Oracle(prompt.size(), reference, drafts == Drafts::kCorrupt));
  }
  return Spec(*s, prompt, max, config, stop_at_eos);
}

const char* Name(Drafts d) {
  return d == Drafts::kDrafter  ? "drafter"
         : d == Drafts::kOracle ? "oracle"
                                : "corrupt";
}

// ---------------------------------------------------------------------------

void ExactGreedy(Fixture& f) {
  std::vector<std::pair<std::string, Tokens>> prompts;
  for (const std::size_t n :
       {9U, 509U, 510U, 511U, 512U, 513U, 514U, 515U, 600U, 2048U, 8192U}) {
    prompts.emplace_back(std::to_string(n), TokensFor(*f.tiny, n, 900 + n));
  }
  for (const bool thinking : {true, false}) {
    ml::ChatTemplateOptions options;
    options.enable_thinking = thinking;
    const std::vector<ml::ChatMessage> messages{
        {.role = "user", .content = "Write the quick brown fox twice."}};
    prompts.emplace_back(thinking ? "chat-think" : "chat-nothink",
                         f.spec->EncodeChat(messages, {}, options));
  }
  std::size_t runs = 0;
  std::uint32_t drafted = 0, accepted = 0;
  for (const auto& [name, prompt] : prompts) {
    const std::uint32_t max = prompt.size() >= 2048 ? 24 : 48;
    // The reference continues 16 tokens past `max` so the oracle can draft
    // through the budget's end.
    const Run ref = Plain(*f.plain, prompt, max + 16, Greedy());
    Run ref_cut = Plain(*f.plain, prompt, max, Greedy());
    Require(!ref.stopped, "fixture: the reference stopped early");
    for (const Drafts d :
         {Drafts::kDrafter, Drafts::kOracle, Drafts::kCorrupt}) {
      for (const std::uint32_t block : {4U, 8U, 16U}) {
        if (prompt.size() >= 2048 && block != 8) {
          continue;  // the long prompts run one block size
        }
        const Run got =
            SpecRun(*f.spec, prompt, max, Greedy(), block, d, ref.tokens);
        RequireSame(got, ref_cut,
                    "prompt " + name + " drafts " + Name(d) + " block " +
                        std::to_string(block));
        if (d == Drafts::kOracle) {
          Require(got.accepted == got.drafted && got.drafted > 0,
                  "oracle drafts were not all accepted");
        }
        if (d == Drafts::kCorrupt) {
          Require(got.accepted < got.drafted && got.accepted > 0,
                  "corrupted oracle did not give partial acceptance");
        }
        drafted += got.drafted;
        accepted += got.accepted;
        ++runs;
      }
    }
  }
  std::cout << "exact greedy ok (" << runs << " runs over " << prompts.size()
            << " prompts incl. 509..515, 2048, 8192, chat thinking on/off; "
            << accepted << "/" << drafted << " drafts accepted)\n";
}

void StopTokens() {
  // Find two tokens the greedy continuations reach, then make EOS 2 and
  // EOT 24 tie with them (first maximum: the stop token wins).
  Fixture probe = MakeFixture(8, 64, {}, 2);
  const Tokens p1 = TokensFor(*probe.tiny, 520, 77);
  const Tokens p2 = TokensFor(*probe.tiny, 40, 78);
  const Run c1 = Plain(*probe.plain, p1, 30, Greedy());
  const Run c2 = Plain(*probe.plain, p2, 30, Greedy());
  const auto pick = [](const Tokens& t, std::size_t at,
                       const std::vector<std::int32_t>& avoid) {
    for (std::size_t i = at; i < t.size(); ++i) {
      if (t[i] > 24 &&
          std::find(t.begin(), t.begin() + i, t[i]) == t.begin() + i &&
          std::find(avoid.begin(), avoid.end(), t[i]) == avoid.end()) {
        return t[i];
      }
    }
    Require(false, "fixture: no token to alias");
    return 0;
  };
  const std::int32_t x = pick(c1.tokens, 6, {});
  const std::int32_t y = pick(c2.tokens, 9, {x});
  Fixture f = MakeFixture(
      8, 64,
      {{2, static_cast<std::uint32_t>(x)}, {24, static_cast<std::uint32_t>(y)}},
      2);
  std::size_t checked = 0;
  std::vector<std::int32_t> seen;
  for (const Tokens& prompt : {p1, p2}) {
    const Run ref = Plain(*f.plain, prompt, 60, Greedy());
    // Without stopping, the reference emits the stop token and goes on: the
    // oracle drafts the stop token and what follows it.
    const Run ref_all = Plain(*f.plain, prompt, 40, Greedy(), false);
    Require(ref.stopped && ref.tokens.size() < ref_all.tokens.size(),
            "fixture: the aliased reference did not stop");
    const std::int32_t stop = ref_all.tokens[ref.tokens.size()];
    Require(stop == 2 || stop == 24, "fixture: not a stop token");
    seen.push_back(stop);
    Tokens oracle = ref_all.tokens;
    for (int i = 0; i < 16; ++i) {
      oracle.push_back(300 + i);
    }
    for (const Drafts d : {Drafts::kDrafter, Drafts::kOracle}) {
      for (const std::uint32_t block : {4U, 8U, 16U}) {
        const Run got =
            SpecRun(*f.spec, prompt, 60, Greedy(), block, d, oracle);
        RequireSame(got, ref,
                    "stop " + std::to_string(stop) + " drafts " + Name(d) +
                        " block " + std::to_string(block));
        ++checked;
      }
    }
    // ignore_eos: the stop token is emitted and decoding goes on.
    for (const std::uint32_t block : {4U, 8U}) {
      const Run got = SpecRun(*f.spec, prompt, 40, Greedy(), block,
                              Drafts::kOracle, oracle, false);
      RequireSame(got, ref_all, "stop " + std::to_string(stop) + " ignore_eos");
      ++checked;
    }
  }
  Require(std::find(seen.begin(), seen.end(), 2) != seen.end() &&
              std::find(seen.begin(), seen.end(), 24) != seen.end(),
          "fixture: EOS and EOT were not both reached");
  std::cout << "stop tokens inside drafts ok (EOS 2 and EOT 24, " << checked
            << " runs, stop_at_eos on and off)\n";
}

void BudgetInsideDraft(Fixture& f) {
  const Tokens prompt = TokensFor(*f.tiny, 530, 31);
  const Run ref = Plain(*f.plain, prompt, 40, Greedy());
  for (const std::uint32_t max : {1U, 2U, 3U, 5U, 9U}) {
    const Run cut = Plain(*f.plain, prompt, max, Greedy());
    const Run got =
        SpecRun(*f.spec, prompt, max, Greedy(), 8, Drafts::kOracle, ref.tokens);
    RequireSame(got, cut, "max_tokens " + std::to_string(max));
    Require(got.tokens.size() == max, "budget not used up");
  }
  std::cout << "max_tokens inside a draft ok (1, 2, 3, 5, 9 at block 8)\n";
}

std::shared_ptr<const smp::TokenConstraint> ToolConstraint(const ml::Model& m,
                                                           bool reasoning) {
  const auto schema = gufo::json::parse(
      R"({"type":"object","properties":{"city":{"type":"string"},)"
      R"("days":{"type":"integer"}},"required":["city","days"],"additionalProperties":false})");
  auto params = smp::JsonConstraint::ToolParameters(
      schema, true, smp::JsonConstraint::ToolFormat::kLaguna);
  auto grammar = smp::JsonConstraint::WithTools(
      nullptr, {{"get_weather", params}}, true, false,
      smp::JsonConstraint::ToolFormat::kLaguna);
  if (reasoning) {
    grammar = smp::JsonConstraint::WithReasoning(grammar);
  }
  auto vocabulary = std::make_shared<const smp::ConstraintVocabulary>(
      m.VocabSize(), [&m](std::uint32_t id) {
        return smp::ConstraintVocabulary::Piece{
            m.DecodeToken(static_cast<std::int32_t>(id)),
            m.IsStopToken(static_cast<std::int32_t>(id))};
      });
  auto binding = std::make_shared<smp::TokenConstraint>();
  binding->grammar = std::move(grammar);
  binding->vocabulary = std::move(vocabulary);
  return binding;
}

void DeterministicSampled(Fixture& f) {
  const Tokens prompt = TokensFor(*f.tiny, 513, 44);
  std::vector<std::pair<std::string, smp::SamplingConfig>> configs;
  {
    auto c = Greedy();
    c.constraint = ToolConstraint(*f.spec, false);
    configs.emplace_back("tool grammar", c);
    auto r = Greedy();
    r.constraint = ToolConstraint(*f.spec, true);
    configs.emplace_back("tool grammar after reasoning", r);
    auto p = Greedy();
    p.repeat_penalty = 1.3F;
    configs.emplace_back("repeat penalty", p);
    auto k = Greedy();
    k.temperature = 0.7F;
    k.top_k = 1;
    configs.emplace_back("top_k 1", k);
  }
  for (const auto& [name, config] : configs) {
    Require(!config.uses_random_sampling(), "fixture: config samples");
    const Run ref = Plain(*f.plain, prompt, 56, config);
    const Run cut = Plain(*f.plain, prompt, 40, config);
    for (const Drafts d :
         {Drafts::kDrafter, Drafts::kOracle, Drafts::kCorrupt}) {
      const Run got = SpecRun(*f.spec, prompt, 40, config, 8, d, ref.tokens);
      RequireSame(got, cut, name + " drafts " + Name(d));
      if (d == Drafts::kOracle) {
        // The constraint runs through accepted drafts: nothing is lost to
        // target-only steps.
        Require(got.drafted > 0 && got.accepted == got.drafted,
                name + ": the oracle was not drafted and accepted");
      }
    }
  }
  std::cout << "deterministic sampled ok (tool grammar with and without "
               "reasoning, repeat penalty, top_k 1; oracle drafts accepted "
               "through the constraint)\n";
}

std::vector<std::int32_t> ProposeNow(ml::Session& s,
                                     std::vector<float>* logits) {
  smp::SamplerState greedy;
  const auto anchor = static_cast<std::int32_t>(greedy.Sample(s.Logits()));
  std::vector<std::int32_t> d;
  std::string error;
  Require(s.dflash()->Propose(anchor, 16, &d, logits, &error),
          "propose: " + error);
  return d;
}

void Snapshots(Fixture& f) {
  std::string error;
  const Tokens prompt = TokensFor(*f.tiny, 600, 55);
  const Tokens other = TokensFor(*f.tiny, 700, 56);
  for (const std::uint32_t split : {3U, 17U}) {
    auto a = f.spec->CreateSession(&error);
    Require(a != nullptr, error);
    const Run head = Spec(*a, prompt, split, Greedy());
    auto snap = a->SaveSnapshot(&error);
    Require(snap != nullptr && snap->drafter.has_value() &&
                snap->bytes() > snap->kv.bytes() + snap->logits.size() * 4,
            "snapshot without the drafter ring: " + error);
    std::vector<float> la;
    const auto da = ProposeNow(*a, &la);
    Tokens seq = prompt;
    seq.insert(seq.end(), head.tokens.begin(), head.tokens.end());
    const Run tail_a = Spec(*a, seq, 30, Greedy(), true, false);

    // Restored into a session that holds another context.
    auto b = f.spec->CreateSession(&error);
    Require(b != nullptr, error);
    (void)Spec(*b, other, 5, Greedy());
    Require(b->RestoreSnapshot(*snap, &error), "restore: " + error);
    std::vector<float> lb;
    Require(ProposeNow(*b, &lb) == da && BitIdentical(la, lb),
            "drafts after restoring the drafter ring differ");
    const Run tail_b = Spec(*b, seq, 30, Greedy(), true, false);
    RequireSame(tail_b, tail_a, "continuation after restore");
    Require(tail_b.accepted == tail_a.accepted,
            "acceptance after restore differs");

    // The same snapshot without the ring: a cold ring at the position,
    // never the rows the session held before (stale).
    ml::SessionSnapshot cold = *snap;
    cold.drafter.reset();
    auto c = f.spec->CreateSession(&error);
    auto d = f.spec->CreateSession(&error);
    Require(c != nullptr && d != nullptr, error);
    (void)Spec(*c, other, 9, Greedy());
    Require(
        c->RestoreSnapshot(cold, &error) && d->RestoreSnapshot(cold, &error),
        "cold restore: " + error);
    const auto& kv = c->dflash()->drafter().kv();
    Require(kv.floor() == c->Position() && kv.committed() == c->Position(),
            "a restore without the ring kept the stale ring");
    std::vector<float> lc, ld;
    Require(ProposeNow(*c, &lc) == ProposeNow(*d, &ld) && BitIdentical(lc, ld),
            "a cold restore drafts differently from a fresh cold session");
    Require(!BitIdentical(lc, la), "a cold ring drafts like the full ring");
    RequireSame(Spec(*c, seq, 30, Greedy(), true, false), tail_a,
                "continuation after a cold restore");
    // Plain reference for the whole request.
    const Run ref = Plain(*f.plain, prompt, split + 30, Greedy());
    Require(ref.tokens.size() == head.tokens.size() + tail_a.tokens.size() &&
                std::equal(head.tokens.begin(), head.tokens.end(),
                           ref.tokens.begin()) &&
                std::equal(tail_a.tokens.begin(), tail_a.tokens.end(),
                           ref.tokens.begin() + head.tokens.size()),
            "snapshot request differs from plain decode");
  }
  std::cout << "snapshot/restore mid-request ok (ring restored bit for bit; "
               "cold restore never stale)\n";
}

void Negative(Fixture& f) {
  std::string error;
  const Tokens prompt = TokensFor(*f.tiny, 520, 66);
  const Run ref = Plain(*f.plain, prompt, 40, Greedy());
  // (1) A verify at the wrong anchor position is refused, touching nothing.
  {
    auto s = f.spec->CreateSession(&error);
    auto twin = f.plain->CreateSession(&error);
    Require(s != nullptr && twin != nullptr, error);
    Require(s->Prefill(prompt, &error) && twin->Prefill(prompt, &error), error);
    auto& target = s->decoder()->target();
    const std::vector<gufo::tokenization::TokenId> block{5, 6, 7, 8};
    const std::uint32_t pos = s->Position();
    for (const std::uint32_t bad : {pos + 1, pos - 1}) {
      bool threw = false;
      try {
        (void)target.ForwardVerificationChunk(block, bad, false, false);
      } catch (const std::logic_error&) {
        threw = true;
      }
      Require(threw && s->Position() == pos,
              "a verify at the wrong anchor position was run");
    }
    // (2) A commit past, or off, the verified rows is refused.
    target.SaveState(pos);
    (void)target.ForwardVerificationChunk(block, pos, false, false);
    const std::vector<gufo::tokenization::TokenId> past{5, 6, 7, 8, 9};
    const std::vector<gufo::tokenization::TokenId> off{5, 9};
    for (const auto* bad : {&past, &off}) {
      bool threw = false;
      try {
        target.CommitVerificationChunk(*bad, pos);
      } catch (const std::logic_error&) {
        threw = true;
      }
      Require(threw && s->Position() == pos + 4,
              "a commit past or off the verified rows was accepted");
    }
    bool threw = false;
    try {
      target.CommitVerificationChunk(
          std::vector<gufo::tokenization::TokenId>{5, 6}, pos + 1);
    } catch (const std::logic_error&) {
      threw = true;
    }
    Require(threw, "a commit from another start was accepted");
    target.CommitVerificationChunk(
        std::vector<gufo::tokenization::TokenId>{5, 6}, pos);
    target.FinishVerification();
    Require(s->Position() == pos + 2 &&
                BitIdentical(std::vector<float>(target.FrontierLogits().begin(),
                                                target.FrontierLogits().end()),
                             [&] {
                               Require(twin->Evaluate(5, &error) &&
                                           twin->Evaluate(6, &error),
                                       error);
                               return twin->Logits();
                             }()),
            "a valid commit after refused ones differs from plain decode");
    // (3) Random sampling does not speculate.
    smp::SamplingConfig hot;
    hot.temperature = 1.0F;
    hot.top_k = 20;
    smp::SamplerState sampler(hot);
    ml::SpeculativeStepResult out;
    Require(!s->DecodeSpeculative(sampler, 4, kContext, true, &out, &error),
            "a sampled request speculated");
  }
  // (4) A target failure inside a step: the session resets, the drafter is
  // re-armed, and the next request equals a cold one.
  {
    auto s = f.spec->CreateSession(&error);
    Require(s != nullptr && s->Prefill(prompt, &error), error);
    s->InjectFailureForTesting("attn");
    smp::SamplerState sampler(Greedy());
    ml::SpeculativeStepResult out;
    Require(!s->DecodeSpeculative(sampler, 8, kContext, true, &out, &error) &&
                s->Position() == 0 && s->failures() == 1 && !s->invalid() &&
                s->drafting(),
            "a failed step did not reset the session and re-arm the drafter");
    RequireSame(Spec(*s, prompt, 40, Greedy()), ref,
                "request after a failed step");
  }
  // (5) The drafter on a poisoned target: dropped, and refuses to draft.
  {
    auto s = f.spec->CreateSession(&error);
    Require(s != nullptr && s->Prefill(prompt, &error), error);
    s->executor().attention_ops().SetFailAfterStage("attn");
    std::vector<float> l(4 * f.spec->VocabSize());
    Require(!s->dflash()->Verify(Tokens{5, 6, 7, 8}, l.data(), &error) &&
                s->poisoned() && !s->dflash()->capturing(),
            "a target failure did not drop the drafter");
    s->executor().attention_ops().SetFailAfterStage("");
    std::vector<std::int32_t> d;
    Require(!s->dflash()->Propose(5, 4, &d, nullptr, &error),
            "the drafter drafted on a poisoned target");
    s->Invalidate();
    Require(!s->invalid() && s->drafting() && s->Position() == 0,
            "invalidate did not recover the session");
    RequireSame(Spec(*s, prompt, 40, Greedy()), ref,
                "request after a poisoned target");
  }
  // (6) A drafter failure mid-request: the drafter is dropped, decoding goes
  // on target-only and stays exact.
  {
    auto s = f.spec->CreateSession(&error);
    Require(s != nullptr, error);
    const Run head = Spec(*s, prompt, 9, Greedy());
    s->dflash()->drafter().ops().SetFailAfterStage("Qcur-1");
    Tokens seq = prompt;
    seq.insert(seq.end(), head.tokens.begin(), head.tokens.end());
    const Run tail = Spec(*s, seq, 31, Greedy(), true, false);
    s->dflash()->drafter().ops().SetFailAfterStage("");
    Require(!s->dflash()->capturing() && tail.drafted == 0 && tail.steps == 31,
            "a drafter failure did not leave target-only decoding");
    Tokens all = head.tokens;
    all.insert(all.end(), tail.tokens.begin(), tail.tokens.end());
    Require(all == ref.tokens && BitIdentical(tail.logits, ref.logits),
            "decoding after a drafter failure differs from plain decode");
  }
  std::cout << "negative checks ok (wrong anchor, commit past/off/elsewhere, "
               "random sampling refused, failed step resets, poisoned target "
               "drops the drafter, drafter failure continues exactly)\n";
}

void BatchMixed(Fixture& f) {
  std::string error;
  // Sessions 0 and 2 draft; 1 and 3 are suspended (sampled requests). One
  // EvaluateBatch: the drafting ones run alone, the others share a pass.
  std::vector<std::unique_ptr<ml::Session>> s, twin;
  for (int i = 0; i < 4; ++i) {
    s.push_back(f.spec->CreateSession(&error));
    twin.push_back(f.plain->CreateSession(&error));
    Require(s.back() != nullptr && twin.back() != nullptr, error);
    const Tokens p = TokensFor(*f.tiny, 505 + 4 * i, 70 + i);
    Require(s.back()->Prefill(p, &error) && twin.back()->Prefill(p, &error),
            error);
    if (i % 2 == 1) {
      s.back()->SuspendDrafter();
    }
  }
  for (int step = 0; step < 12; ++step) {
    std::vector<ml::Session::BatchItem> items(4);
    for (int i = 0; i < 4; ++i) {
      items[i].session = s[i].get();
      items[i].token = 30 + step * 7 + i;
      Require(twin[i]->Evaluate(items[i].token, &error), error);
    }
    ml::Session::EvaluateBatch(items);
    for (int i = 0; i < 4; ++i) {
      Require(
          items[i].completed && BitIdentical(s[i]->Logits(), twin[i]->Logits()),
          "mixed batch row differs from solo decode: " + items[i].error);
    }
  }
  Require(s[0]->drafting() && s[2]->drafting() && !s[1]->drafting(),
          "batch changed the drafting state");
  // The drafting sessions stayed in step: a speculative continuation is exact.
  smp::SamplerState sampler(Greedy());
  ml::SpeculativeStepResult out;
  Require(s[0]->DecodeSpeculative(sampler, 8, kContext, true, &out, &error),
          error);
  std::cout << "batched decode with drafting and suspended sessions ok\n";
}

void Threads(Fixture& f, int count) {
  std::string error;
  struct Job {
    Tokens prompt;
    bool speculate;
    std::uint64_t seed;
  };
  std::vector<Job> jobs;
  for (int i = 0; i < count; ++i) {
    jobs.push_back(
        {TokensFor(*f.tiny, 500 + 9 * i, 4000 + i), i % 2 == 0, 1234U + i});
  }
  smp::SamplingConfig hot;
  hot.temperature = 0.8F;
  hot.top_k = 20;
  const auto run = [&](ml::Session& s, const Job& j) {
    if (j.speculate) {
      return Spec(s, j.prompt, 40, Greedy());
    }
    return Plain(s, j.prompt, 40, hot, true, j.seed);
  };
  std::vector<Run> solo;
  for (const Job& j : jobs) {
    auto s = f.spec->CreateSession(&error);
    Require(s != nullptr, error);
    solo.push_back(run(*s, j));
    if (j.speculate) {
      RequireSame(solo.back(), Plain(*f.plain, j.prompt, 40, Greedy()),
                  "solo speculative run");
    } else {
      auto p = f.plain->CreateSession(&error);
      RequireSame(solo.back(), Plain(*p, j.prompt, 40, hot, true, j.seed),
                  "solo sampled run on the drafter model");
    }
  }
  std::vector<std::unique_ptr<ml::Session>> sessions;
  for (int i = 0; i < count; ++i) {
    sessions.push_back(f.spec->CreateSession(&error));
    Require(sessions.back() != nullptr, error);
  }
  std::atomic<bool> go{false};
  std::atomic<int> failures{0};
  std::vector<Run> got(count);
  std::vector<std::thread> threads;
  for (int i = 0; i < count; ++i) {
    threads.emplace_back([&, i] {
      try {
        while (!go.load()) {
          std::this_thread::yield();
        }
        got[i] = run(*sessions[i], jobs[i]);
      } catch (const std::exception& e) {
        std::cerr << "thread " << i << ": " << e.what() << '\n';
        ++failures;
      }
    });
  }
  go = true;
  for (auto& t : threads) {
    t.join();
  }
  Require(failures.load() == 0, "a concurrent session failed");
  for (int i = 0; i < count; ++i) {
    RequireSame(got[i], solo[i], "concurrent session " + std::to_string(i));
  }
  std::cout << count
            << " concurrent sessions ok (speculating greedy and "
               "plain sampled mixed; each equals its solo run)\n";
}

}  // namespace

int main() {
  try {
    {
      Fixture f = MakeFixture(4, 64);
      ExactGreedy(f);
      BudgetInsideDraft(f);
      DeterministicSampled(f);
      Snapshots(f);
      Negative(f);
      BatchMixed(f);
      Threads(f, 2);
      Threads(f, 8);
    }
    // The smallest chunk that admits a block of 16 (the rings' spare rows).
    {
      Fixture f = MakeFixture(16, 16, {}, 2);
      const Tokens prompt = TokensFor(*f.tiny, 515, 91);
      const Run ref = Plain(*f.plain, prompt, 60, Greedy());
      const Run cut = Plain(*f.plain, prompt, 44, Greedy());
      for (const Drafts d : {Drafts::kOracle, Drafts::kCorrupt}) {
        RequireSame(SpecRun(*f.spec, prompt, 44, Greedy(), 16, d, ref.tokens),
                    cut, std::string("chunk 16 block 16 ") + Name(d));
      }
      std::cout << "chunk 16 with block 16 ok\n";
    }
    StopTokens();
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
  std::cout << "laguna.speculative: ok\n";
  return 0;
}
