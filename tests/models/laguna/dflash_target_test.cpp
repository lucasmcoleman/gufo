// The target side of Laguna DFlash (phase 6d of DFLASH.md) on tiny random
// models (no model file, nothing skipped): a six-layer target with the real
// geometry and a 512-position window (so 511/512/513 and the ring wrap are the
// real boundaries), and a three-tap, three-layer drafter of the real head
// geometry (hidden 512, GQA 9, window 512).
//
//  * capture: Executor feature taps (device-side, no synchronize) equal the
//    debug layer tap's hidden rows bit for bit for prefill (batched), decode
//    and verify (vector) rows; calls with capture give the same logits as
//    untapped calls and untapped reruns; taps one layer early differ (must
//    fail); bad tap lists and capture without taps are refused.
//  * path: Forward with a per-call vector path on a kAuto executor of chunk
//    16 or 512 equals the same tokens decoded alone, bit for bit, for every
//    verify block crossing 511/512/513 and the ring wrap; a batched verify
//    does not (a wrong path must fail); 33 vector rows are refused.
//  * rewind: RewindTo at every depth 0..16 after a 16-row verify at those
//    boundaries, at the smallest chunk that admits the verify (16) and at 512:
//    the continuation equals a twin that never saw the rejected rows (logits
//    bitwise, retained K/V bytes equal). A rewind past retained history
//    (single, chained, beyond the window), ahead of the position, or on a
//    poisoned executor is refused and leaves everything untouched.
//  * drafter ring: DrafterKv::RewindTo trims rejected rows (a rewound ring
//    drafts exactly as one that never held them) and refuses positions
//    outside [floor, committed] or with evicted context.
//  * DFlashSession: drafts from device-captured taps equal drafts from the
//    same taps uploaded from the host (different injection chunking); a
//    verify equals solo decode; commit equals a fresh injection; failure
//    rules (a target failure after its K/V append poisons the target and
//    drops the drafter; drafter failures in injection, in a block and from a
//    non-finite tap leave the target usable and bit-identical to its twin;
//    a refused commit touches nothing); DecodeBatch refuses a tapped
//    executor; two sessions on two threads equal their solo runs.
#include <atomic>
#include <cstring>
#include <iostream>
#include <map>
#include <optional>
#include <thread>
#include <tuple>

#include "src/models/laguna/kernels/dflash_session.hpp"
#include "tests/models/laguna/dflash_gpu.hpp"
#include "tests/models/laguna/executor_gpu.hpp"

using namespace gufo::testing::laguna;

namespace {

using Tokens = std::vector<std::int32_t>;
const std::vector<std::uint32_t> kTaps = {1, 3, 5};
const std::vector<std::uint32_t> kEarly = {0, 2, 4};
constexpr std::int32_t kMask = 12;

ModelSpec TargetSpec() {
  ModelSpec spec;
  spec.window = 512;
  spec.hidden = 512;
  return spec;
}

/// Sequence with no token equal to `avoid` at the same index.
Tokens Junk(const Tokens& avoid, std::size_t vocab, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  Tokens t(avoid.size());
  for (std::size_t i = 0; i < t.size(); ++i) {
    do {
      t[i] = static_cast<std::int32_t>(rng() % vocab);
    } while (t[i] == avoid[i]);
  }
  return t;
}

std::vector<float> Slice(const std::vector<float>& v, std::size_t begin,
                         std::size_t count) {
  return {v.begin() + static_cast<std::ptrdiff_t>(begin),
          v.begin() + static_cast<std::ptrdiff_t>(begin + count)};
}

/// One Forward on `ex` with the debug layer tap collecting the rows of the
/// `layers` outputs as features [T][layers][H] (the reference for capture).
struct Tapped {
  std::vector<float> logits;
  std::vector<float> features;
};
Tapped DebugForward(exec::Executor& ex, const Tokens& tokens,
                    std::optional<gpu::Path> path,
                    const std::vector<std::uint32_t>& layers) {
  const std::size_t H = ex.model().config().hidden_size;
  const std::size_t V = ex.model().config().vocab_size;
  const std::size_t T = tokens.size();
  Tapped out;
  out.logits.resize(T * V);
  out.features.resize(T * layers.size() * H);
  ex.SetLayerTap([&](std::uint32_t il, const float* h, std::size_t count) {
    for (std::size_t k = 0; k < layers.size(); ++k) {
      if (layers[k] != il) {
        continue;
      }
      Require(count == T * H, "debug tap row count");
      const auto rows = ToHost(h, count);
      for (std::size_t t = 0; t < T; ++t) {
        std::copy_n(rows.data() + t * H, H,
                    out.features.data() + (t * layers.size() + k) * H);
      }
    }
  });
  std::string error;
  gpu::ForwardArgs args;
  args.path = path;
  const bool ok = ex.Forward(tokens, exec::LogitsRows::kAll, out.logits.data(),
                             args, &error);
  ex.SetLayerTap({});
  Require(ok, "debug forward: " + error);
  return out;
}

/// One Forward capturing features into a device buffer; returns both.
Tapped CaptureForward(exec::Executor& ex, const Tokens& tokens,
                      std::optional<gpu::Path> path, float* device_features) {
  const std::size_t V = ex.model().config().vocab_size;
  Tapped out;
  out.logits.resize(tokens.size() * V);
  std::string error;
  gpu::ForwardArgs args;
  args.path = path;
  args.features = device_features;
  Require(ex.Forward(tokens, exec::LogitsRows::kAll, out.logits.data(), args,
                     &error),
          "capture forward: " + error);
  CheckHip(hipStreamSynchronize(ex.stream()), "sync");
  if (device_features != nullptr) {
    out.features = ToHost(device_features, tokens.size() * ex.feature_width());
  }
  return out;
}

std::vector<float> Plain(exec::Executor& ex, const Tokens& tokens,
                         std::optional<gpu::Path> path) {
  return CaptureForward(ex, tokens, path, nullptr).logits;
}

/// Decodes `tokens` one at a time; all logits.
std::vector<float> DecodeEach(exec::Executor& ex, const Tokens& tokens) {
  const std::size_t V = ex.model().config().vocab_size;
  std::vector<float> all(tokens.size() * V);
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    const auto l = Plain(ex, Tokens{tokens[i]}, std::nullopt);
    std::copy(l.begin(), l.end(),
              all.begin() + static_cast<std::ptrdiff_t>(i * V));
  }
  return all;
}

void PrefillTo(exec::Executor& ex, const Tokens& seq, std::uint32_t to) {
  std::string error;
  while (ex.position() < to) {
    const std::uint32_t n = std::min(ex.max_chunk(), to - ex.position());
    Require(
        ex.Forward(std::span<const std::int32_t>(seq).subspan(ex.position(), n),
                   exec::LogitsRows::kNone, nullptr, &error),
        "prefill: " + error);
  }
}

exec::ExecutorSnapshot Snap(exec::Executor& ex) {
  exec::ExecutorSnapshot s;
  std::string error;
  Require(ex.SaveSnapshot(&s, &error), "snapshot: " + error);
  return s;
}

void Restore(exec::Executor& ex, const exec::ExecutorSnapshot& s) {
  std::string error;
  Require(ex.RestoreSnapshot(s, &error), "restore: " + error);
}

/// The two executors hold the same K/V for every position both retain, at
/// the same position; `a` retains no more than `b` (a rewound ring evicted
/// more). Returns nothing; throws.
void RequireSameRetained(exec::Executor& a, exec::Executor& b,
                         const std::string& what) {
  Require(a.position() == b.position(), what + ": positions differ");
  const auto sa = Snap(a);
  const auto sb = Snap(b);
  for (std::size_t il = 0; il < sa.layers.size(); ++il) {
    const auto& x = sa.layers[il];
    const auto& y = sb.layers[il];
    Require(x.committed == y.committed && x.retained_begin >= y.retained_begin,
            what + ": layer " + std::to_string(il) + " bookkeeping");
    const std::size_t row =
        x.committed == x.retained_begin
            ? 0
            : x.keys.size() / (x.committed - x.retained_begin);
    const std::size_t skip = (x.retained_begin - y.retained_begin) * row;
    Require(y.keys.size() >= skip + x.keys.size() &&
                std::memcmp(x.keys.data(), y.keys.data() + skip,
                            x.keys.size()) == 0 &&
                std::memcmp(x.values.data(), y.values.data() + skip,
                            x.values.size()) == 0,
            what + ": layer " + std::to_string(il) + " K/V bytes differ");
  }
}

struct DeviceFloats {
  explicit DeviceFloats(std::size_t n) {
    CheckHip(hipMalloc(&ptr, n * sizeof(float)), "hipMalloc");
  }
  ~DeviceFloats() { (void)hipFree(ptr); }
  DeviceFloats(const DeviceFloats&) = delete;
  DeviceFloats& operator=(const DeviceFloats&) = delete;
  float* ptr{nullptr};
};

// ---- capture --------------------------------------------------------------

void CaptureChecks(const TinyModel& tm, const exec::DeviceModel& dev) {
  const std::size_t H = tm.config().hidden_size;
  const std::size_t V = tm.config().vocab_size;
  auto cap = MakeExecutor(dev, 2048, 64, gpu::Path::kAuto);
  auto dbg = MakeExecutor(dev, 2048, 64, gpu::Path::kAuto);
  auto plain = MakeExecutor(dev, 2048, 64, gpu::Path::kAuto);
  auto early = MakeExecutor(dev, 2048, 64, gpu::Path::kAuto);
  std::string error;
  Require(cap->SetFeatureTaps(kTaps, &error), error);
  Require(early->SetFeatureTaps(kEarly, &error), error);
  Require(cap->feature_width() == 3 * H, "feature width");
  // Bad lists are refused and keep the previous list.
  const std::vector<std::uint32_t> out_of_range = {1, 6};
  const std::vector<std::uint32_t> twice = {3, 1, 3};
  Require(!cap->SetFeatureTaps(out_of_range, &error) &&
              !cap->SetFeatureTaps(twice, &error) &&
              cap->feature_taps() == kTaps,
          "bad tap lists are refused and keep the previous list");
  DeviceFloats buf(64 * 3 * H);
  // Capture without taps is refused before anything runs.
  {
    gpu::ForwardArgs args;
    args.features = buf.ptr;
    std::vector<float> l(V);
    Require(!plain->Forward(Tokens{5}, exec::LogitsRows::kLast, l.data(), args,
                            &error) &&
                plain->position() == 0 && !plain->poisoned(),
            "capture without feature taps is refused, untouched");
  }
  const Tokens seq = TokensFor(tm, 700, 61);
  std::size_t calls = 0;
  std::size_t p = 0;
  // Prefill chunks (batched), single decode rows, verify blocks (vector),
  // short default-path blocks, crossing 511/512/513.
  const std::vector<std::pair<std::uint32_t, std::optional<gpu::Path>>> plan = {
      {64, std::nullopt},       {64, std::nullopt},       {64, std::nullopt},
      {64, std::nullopt},       {64, std::nullopt},       {64, std::nullopt},
      {64, std::nullopt},       {63, std::nullopt},       {1, std::nullopt},
      {16, gpu::Path::kVector}, {1, std::nullopt},        {5, std::nullopt},
      {16, gpu::Path::kVector}, {32, gpu::Path::kVector}, {64, std::nullopt},
      {1, std::nullopt}};
  for (const auto& [n, path] : plan) {
    const Tokens chunk(seq.begin() + static_cast<std::ptrdiff_t>(p),
                       seq.begin() + static_cast<std::ptrdiff_t>(p + n));
    const Tapped c = CaptureForward(*cap, chunk, path, buf.ptr);
    const Tapped d = DebugForward(*dbg, chunk, path, kTaps);
    const auto u = Plain(*plain, chunk, path);
    const Tapped e = CaptureForward(*early, chunk, path, buf.ptr);
    const std::string at = "rows " + std::to_string(p) + ".." +
                           std::to_string(p + n - 1) +
                           (path ? " vector" : " default path");
    Require(BitIdentical(c.features, d.features),
            at + ": captured taps differ from the debug layer tap");
    Require(BitIdentical(c.logits, d.logits) && BitIdentical(c.logits, u) &&
                BitIdentical(c.logits, e.logits),
            at + ": capture changed the logits");
    // Negative: one layer early is a different tensor everywhere.
    Require(!BitIdentical(e.features, d.features) &&
                RelRms(e.features, d.features) > 0.05,
            at + ": taps one layer early were not told apart");
    p += n;
    ++calls;
  }
  // An untapped rerun of the whole plan on a fresh executor is bit-identical.
  {
    auto again = MakeExecutor(dev, 2048, 64, gpu::Path::kAuto);
    auto first = MakeExecutor(dev, 2048, 64, gpu::Path::kAuto);
    std::size_t q = 0;
    for (const auto& [n, path] : plan) {
      const Tokens chunk(seq.begin() + static_cast<std::ptrdiff_t>(q),
                         seq.begin() + static_cast<std::ptrdiff_t>(q + n));
      Require(
          BitIdentical(Plain(*again, chunk, path), Plain(*first, chunk, path)),
          "untapped rerun differs");
      q += n;
    }
  }
  std::cout << "capture ok (" << calls
            << " calls: prefill, decode and verify rows bitwise equal to the "
               "debug tap; logits unchanged; one layer early fails)\n";
}

// ---- per-call path, bit identity and rewind ---------------------------------

struct RewindStats {
  std::size_t cases{0};
  std::size_t rows{0};
};

/// For each verify start in `starts`: a 16-row vector verify whose first
/// k = c - n rows are the real tokens, then RewindTo(c) for every c in
/// n..n+16, then four decode rows; against a twin that decodes the real
/// tokens alone.
void RewindCases(const TinyModel& tm, const exec::DeviceModel& dev,
                 std::uint32_t chunk, const std::vector<std::uint32_t>& starts,
                 RewindStats* stats) {
  const std::size_t V = tm.config().vocab_size;
  const std::uint32_t end = starts.back() + 40;
  const Tokens seq = TokensFor(tm, end, 7000 + chunk);
  auto base = MakeExecutor(dev, end + 64, chunk, gpu::Path::kAuto);
  auto sub = MakeExecutor(dev, end + 64, chunk, gpu::Path::kAuto);
  auto twin = MakeExecutor(dev, end + 64, chunk, gpu::Path::kAuto);
  std::string error;
  for (const std::uint32_t n : starts) {
    PrefillTo(*base, seq, n);
    const auto snap = Snap(*base);
    const Tokens real(seq.begin() + n, seq.begin() + n + 16);
    const Tokens junk = Junk(real, V, n);
    // The twin decodes the real block alone once: its rows are the reference.
    Restore(*twin, snap);
    const auto solo = DecodeEach(*twin, real);
    // A verify of the real block equals the solo rows bit for bit; a batched
    // verify does not (the wrong path must be caught).
    Restore(*sub, snap);
    Require(BitIdentical(Plain(*sub, real, gpu::Path::kVector), solo),
            "chunk " + std::to_string(chunk) + ": verify at " +
                std::to_string(n) + " differs from solo decode");
    Restore(*sub, snap);
    Require(!BitIdentical(Plain(*sub, real, gpu::Path::kBatched), solo),
            "a batched verify was not told apart from solo decode at " +
                std::to_string(n));
    for (std::uint32_t depth = 0; depth <= 16; ++depth) {
      const std::uint32_t k = 16 - depth;
      const std::uint32_t c = n + k;
      Tokens block(real.begin(), real.begin() + k);
      block.insert(block.end(), junk.begin() + k, junk.end());
      Restore(*sub, snap);
      const auto verified = Plain(*sub, block, gpu::Path::kVector);
      Require(BitIdentical(Slice(verified, 0, std::size_t{k} * V),
                           Slice(solo, 0, std::size_t{k} * V)),
              "accepted verify rows differ from solo decode");
      const std::string at = "chunk " + std::to_string(chunk) + " verify at " +
                             std::to_string(n) + ", rewind to " +
                             std::to_string(c);
      Require(sub->RewindTo(c, &error), at + ": " + error);
      Require(sub->position() == c, at + ": position");
      // Continue: four decode rows, then a 4-row verify; the twin decodes
      // the same real tokens from its own state.
      const Tokens next(seq.begin() + c, seq.begin() + c + 8);
      Restore(*twin, snap);
      (void)DecodeEach(*twin, Tokens(real.begin(), real.begin() + k));
      const auto want = DecodeEach(*twin, next);
      const auto got_a =
          DecodeEach(*sub, Tokens(next.begin(), next.begin() + 4));
      const auto got_b =
          Plain(*sub, Tokens(next.begin() + 4, next.end()), gpu::Path::kVector);
      Require(BitIdentical(got_a, Slice(want, 0, 4 * V)) &&
                  BitIdentical(got_b, Slice(want, 4 * V, 4 * V)),
              at + ": continuation differs from the twin");
      RequireSameRetained(*sub, *twin, at);
      ++stats->cases;
      stats->rows += 16 + 8;
    }
  }
}

void RewindChecks(const TinyModel& tm, const exec::DeviceModel& dev) {
  RewindStats stats;
  // Chunk 16, the smallest that admits a 16-row verify: ring 528, wraps at
  // 528 and 1056.
  RewindCases(tm, dev, 16,
              {480, 495, 496, 497, 500, 505, 510, 511, 512, 513, 520, 527, 528,
               1040, 1041, 1056},
              &stats);
  // Chunk 512 (serve): ring 1024.
  RewindCases(tm, dev, 512, {495, 496, 497, 511, 512, 513, 1008, 1016, 1024},
              &stats);
  std::cout << "verify bit identity and rewind ok (" << stats.cases
            << " cases, depths 0..16 at 511/512/513 and ring wraps, chunks 16 "
               "and 512)\n";
}

/// Ring and bookkeeping of every layer, for "untouched" checks.
struct State {
  std::uint32_t position{0};
  std::vector<std::pair<std::uint32_t, std::uint32_t>> books;
  exec::ExecutorSnapshot snap;
};
State Observe(exec::Executor& ex) {
  State s;
  s.position = ex.position();
  for (std::uint32_t il = 0; il < ex.model().config().num_layers; ++il) {
    s.books.emplace_back(ex.kv_cache(il).committed(),
                         ex.kv_cache(il).retained_begin());
  }
  s.snap = Snap(ex);
  return s;
}
void RequireUntouched(exec::Executor& ex, const State& before,
                      const std::string& what) {
  const State now = Observe(ex);
  Require(now.position == before.position && now.books == before.books,
          what + ": bookkeeping changed");
  for (std::size_t il = 0; il < now.snap.layers.size(); ++il) {
    Require(now.snap.layers[il].keys == before.snap.layers[il].keys &&
                now.snap.layers[il].values == before.snap.layers[il].values,
            what + ": K/V bytes changed");
  }
}

void RefusalChecks(const TinyModel& tm, const exec::DeviceModel& dev) {
  std::size_t refused = 0;
  std::string error;
  for (const std::uint32_t chunk : {16U, 512U}) {
    const std::uint32_t P = chunk == 16 ? 700 : 1100;
    const Tokens seq = TokensFor(tm, P + 40, 9100 + chunk);
    auto ex = MakeExecutor(dev, P + 64, chunk, gpu::Path::kAuto);
    auto twin = MakeExecutor(dev, P + 64, chunk, gpu::Path::kAuto);
    PrefillTo(*ex, seq, P - 16);
    // The last 16 rows as one vector verify: the ring holds exactly
    // [P - (window + chunk), P).
    (void)Plain(*ex, Tokens(seq.begin() + P - 16, seq.begin() + P),
                gpu::Path::kVector);
    PrefillTo(*twin, seq, P - 16);
    (void)Plain(*twin, Tokens(seq.begin() + P - 16, seq.begin() + P),
                gpu::Path::kVector);
    const State before = Observe(*ex);
    const auto refuse = [&](std::uint32_t to, const std::string& why) {
      Require(!ex->RewindTo(to, &error),
              "chunk " + std::to_string(chunk) + ": rewind to " +
                  std::to_string(to) + " (" + why + ") was accepted");
      RequireUntouched(*ex, before, "refused rewind (" + why + ")");
      ++refused;
    };
    refuse(P + 1, "ahead of the position");
    refuse(P - chunk - 1, "one past the ring's rewind depth");
    refuse(P - 600, "keys evicted");
    refuse(0 + 1, "far past retained history");
    // Chained: rewinds of 10 and then 8 more need a key the ring lost.
    if (chunk == 16) {
      const auto snap = Snap(*ex);
      Require(ex->RewindTo(P - 10, &error), "rewind 10: " + error);
      const State mid = Observe(*ex);
      Require(!ex->RewindTo(P - 18, &error),
              "chained rewind past retained history was accepted");
      RequireUntouched(*ex, mid, "refused chained rewind");
      // ...while 7 more (key P - 528 still in its slot) is fine.
      Require(ex->RewindTo(P - 17, &error), "chained rewind 17: " + error);
      Restore(*ex, snap);
      ++refused;
    }
    // After every refusal the executor still equals its twin.
    const Tokens next(seq.begin() + P, seq.begin() + P + 6);
    Require(BitIdentical(DecodeEach(*ex, next), DecodeEach(*twin, next)),
            "after refused rewinds the executor differs from its twin");
    // A poisoned executor refuses a rewind.
    ex->attention_ops().SetFailAfterStage("attn");
    std::vector<float> l(tm.config().vocab_size);
    Require(
        !ex->Forward(Tokens{3}, exec::LogitsRows::kLast, l.data(), &error) &&
            ex->poisoned(),
        "injected failure poisons");
    ex->attention_ops().SetFailAfterStage("");
    Require(!ex->RewindTo(ex->position() - 1, &error),
            "a poisoned executor accepted a rewind");
    ++refused;
    // 33 rows on the vector path are refused up front on a chunk-512
    // executor (nothing touched).
    if (chunk == 512) {
      Require(twin->RewindTo(twin->position(), &error), error);
      const State t0 = Observe(*twin);
      std::vector<float> big(33 * tm.config().vocab_size);
      gpu::ForwardArgs args;
      args.path = gpu::Path::kVector;
      Require(
          !twin->Forward(Tokens(seq.begin(), seq.begin() + 33),
                         exec::LogitsRows::kAll, big.data(), args, &error) &&
              !twin->poisoned(),
          "33 vector rows were accepted");
      RequireUntouched(*twin, t0, "refused 33-row vector call");
      ++refused;
    }
  }
  std::cout << "refusals ok (" << refused
            << " refused rewinds and calls, state untouched)\n";
}

// ---- drafter ring -----------------------------------------------------------

void DrafterRingChecks() {
  TinyDrafter fx(kDraftGeom, 0xD6D);
  DrafterCtx ctx(fx.drafter);
  DeviceHead head(fx.target.output);
  const std::size_t width = fx.config().fused_input_width();
  const auto feats = RandomFeatures(640, fx.config().tap_count(),
                                    fx.config().hidden_size, fx.rng());
  std::string error;
  const auto make = [&] {
    auto s = gpu::DrafterSession::Create(ctx.weights(), head.matrix(),
                                         fx.target.token_embd, kMask, 32,
                                         ctx.stream(), &error);
    Require(s != nullptr, "drafter session: " + error);
    return s;
  };
  const auto inject = [&](gpu::DrafterSession& s, std::uint32_t from,
                          std::uint32_t to) {
    for (std::uint32_t p = from; p < to; p += 32) {
      const std::uint32_t rows = std::min(32U, to - p);
      Require(s.Inject(std::span<const float>(feats).subspan(p * width,
                                                             rows * width),
                       p, gpu::Path::kVector, &error),
              "inject: " + error);
    }
  };
  std::size_t checks = 0;
  for (const std::uint32_t n : {500U, 511U, 512U, 513U, 600U}) {
    for (const std::uint32_t keep : {0U, 1U, 7U, 15U, 16U}) {
      auto a = make();
      a->kv().Reset(n - 511 > n ? 0 : n - 511);
      inject(*a, a->kv().floor(), n);
      inject(*a, n, n + 16);  // a verify block's rows
      Require(a->kv().RewindTo(n + keep, &error), "drafter rewind: " + error);
      auto b = make();
      b->kv().Reset(a->kv().floor());
      inject(*b, b->kv().floor(), n + keep);
      std::vector<std::int32_t> da, db;
      std::vector<float> la, lb;
      Require(a->Propose(7, 16, &da, &la, gpu::Path::kVector, &error) &&
                  b->Propose(7, 16, &db, &lb, gpu::Path::kVector, &error),
              "propose: " + error);
      Require(da == db && BitIdentical(la, lb),
              "a rewound drafter ring drafts differently from one that never "
              "held the rejected rows");
      ++checks;
    }
  }
  // Refusals leave the ring untouched.
  auto r = make();
  r->kv().Reset(40);
  inject(*r, 40, 616);  // committed 616; ring 544 holds [72, 616)
  const auto books = [&] {
    return std::tuple(r->kv().committed(), r->kv().retained_begin(),
                      r->kv().floor(), r->kv().valid());
  };
  const auto before = books();
  for (const std::uint32_t to : {617U, 39U, 582U, 100U}) {
    Require(!r->kv().RewindTo(to, &error) && books() == before,
            "drafter rewind to " + std::to_string(to) + " was accepted");
    ++checks;
  }
  Require(r->kv().RewindTo(583, &error), "drafter rewind to 583: " + error);
  // An invalid ring refuses.
  r->ops().SetFailAfterStage("kv_stored-0");
  Require(!r->Inject(std::span<const float>(feats).subspan(583 * width, width),
                     583, gpu::Path::kVector, &error) &&
              !r->kv().valid(),
          "a failed injection invalidates the ring");
  r->ops().SetFailAfterStage("");
  Require(!r->kv().RewindTo(r->kv().committed(), &error),
          "an invalid ring accepted a rewind");
  std::cout << "drafter ring rewind ok (" << checks << " checks)\n";
}

// ---- DFlashSession ----------------------------------------------------------

struct SessionRig {
  std::unique_ptr<exec::Executor> ex;
  std::unique_ptr<gpu::DFlashSession> s;
};

SessionRig MakeSession(const exec::DeviceModel& dev,
                       const gpu::DrafterWeights& w, std::uint32_t chunk) {
  SessionRig r;
  r.ex = MakeExecutor(dev, 2048, chunk, gpu::Path::kAuto);
  std::string error;
  r.s = gpu::DFlashSession::Create(*r.ex, w, kTaps, kMask, &error);
  Require(r.s != nullptr, "DFlashSession::Create: " + error);
  return r;
}

/// Reference drafter fed host features from the debug layer tap of a twin
/// executor that runs the same calls.
struct HostRef {
  std::unique_ptr<exec::Executor> ex;
  std::unique_ptr<gpu::DrafterSession> d;
  hipStream_t stream{};
  std::size_t width{0};
  HostRef(const exec::DeviceModel& dev, const gpu::DrafterWeights& w,
          std::uint32_t chunk) {
    ex = MakeExecutor(dev, 2048, chunk, gpu::Path::kAuto);
    CheckHip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "stream");
    std::string error;
    d = gpu::DrafterSession::Create(w, dev.output(), dev.weights().token_embd,
                                    kMask, 32, stream, &error);
    Require(d != nullptr, "reference drafter: " + error);
    width = kTaps.size() * dev.config().hidden_size;
  }
  ~HostRef() {
    d.reset();
    qfn_mmq_release_stream(stream);
    (void)hipStreamDestroy(stream);
  }
  HostRef(const HostRef&) = delete;
  HostRef& operator=(const HostRef&) = delete;
  /// Runs `tokens` at the twin's position; injects rows >= from.
  std::vector<float> Run(const Tokens& tokens, std::optional<gpu::Path> path,
                         std::uint32_t from) {
    const std::uint32_t start = ex->position();
    const Tapped t = DebugForward(*ex, tokens, path, kTaps);
    std::string error;
    for (std::uint32_t p = std::max(start, from); p < start + tokens.size();
         p += 32) {
      const std::uint32_t rows = std::min<std::uint32_t>(
          32, static_cast<std::uint32_t>(start + tokens.size() - p));
      Require(d->Inject(std::span<const float>(t.features)
                            .subspan((p - start) * width, rows * width),
                        p, gpu::Path::kVector, &error),
              "reference inject: " + error);
    }
    return t.logits;
  }
};

std::vector<float> SessionPrefill(gpu::DFlashSession& s, const Tokens& t) {
  std::vector<float> l(t.size() * s.target().model().config().vocab_size);
  std::string error;
  Require(s.Prefill(t, exec::LogitsRows::kAll, l.data(), &error),
          "session prefill: " + error);
  return l;
}
std::vector<float> SessionVerify(gpu::DFlashSession& s, const Tokens& t) {
  std::vector<float> l(t.size() * s.target().model().config().vocab_size);
  std::string error;
  Require(s.Verify(t, l.data(), &error), "session verify: " + error);
  return l;
}
std::vector<std::int32_t> SessionPropose(gpu::DFlashSession& s,
                                         std::int32_t anchor,
                                         std::vector<float>* logits) {
  std::vector<std::int32_t> d;
  std::string error;
  Require(s.Propose(anchor, 16, &d, logits, &error),
          "session propose: " + error);
  return d;
}

/// The scripted request used by the equivalence and concurrency checks:
/// prompt, propose, verify, commit after `accept`, propose, verify, commit.
struct Script {
  std::vector<std::vector<float>> logits;
  std::vector<std::vector<std::int32_t>> drafts;
};
Script RunScript(gpu::DFlashSession& s, const Tokens& prompt,
                 std::uint32_t accept, std::size_t vocab) {
  Script out;
  out.logits.push_back(SessionPrefill(s, prompt));
  std::int32_t anchor = 9;
  for (int step = 0; step < 3; ++step) {
    std::vector<float> dl;
    auto d = SessionPropose(s, anchor, &dl);
    out.drafts.push_back(d);
    out.logits.push_back(dl);
    Tokens block{anchor};
    for (auto t : d) {
      block.push_back(t < 0 ? 0 : t % static_cast<std::int32_t>(vocab));
    }
    const std::uint32_t n = s.position();
    out.logits.push_back(SessionVerify(s, block));
    std::string error;
    Require(s.Commit(n + accept + 1, &error), "commit: " + error);
    anchor = static_cast<std::int32_t>((anchor * 31 + step + 5) % vocab);
  }
  return out;
}

void SessionEquivalence(const TinyModel& tm, const exec::DeviceModel& dev,
                        const gpu::DrafterWeights& w) {
  const std::size_t V = tm.config().vocab_size;
  for (const std::uint32_t prompt_len : {9U, 600U, 1030U}) {
    const Tokens prompt = TokensFor(tm, prompt_len, 300 + prompt_len);
    SessionRig rig = MakeSession(dev, w, 64);
    HostRef ref(dev, w, 64);
    auto plain = MakeExecutor(dev, 2048, 64, gpu::Path::kAuto);
    const std::uint32_t from = prompt_len >= 512 ? prompt_len - 511 : 0;
    ref.d->kv().Reset(from);
    // Prompt in chunks of 64 on both, the reference with host features.
    const auto sl = SessionPrefill(*rig.s, prompt);
    std::vector<float> rl;
    for (std::uint32_t p = 0; p < prompt_len; p += 64) {
      const Tokens c(prompt.begin() + p,
                     prompt.begin() + std::min(prompt_len, p + 64));
      const auto l = ref.Run(c, std::nullopt, from);
      rl.insert(rl.end(), l.begin(), l.end());
      Require(BitIdentical(Plain(*plain, c, std::nullopt), l),
              "untapped executor differs");
    }
    Require(BitIdentical(sl, rl), "session prefill logits differ");
    Require(rig.s->drafter().kv().floor() == from &&
                rig.s->drafter().kv().committed() == prompt_len,
            "the session injected other prompt rows than the last 511");
    std::int32_t anchor = 11;
    for (int step = 0; step < 4; ++step) {
      std::vector<float> dl, rdl;
      const auto d = SessionPropose(*rig.s, anchor, &dl);
      std::vector<std::int32_t> rd;
      std::string error;
      Require(
          ref.d->Propose(anchor, 16, &rd, &rdl, gpu::Path::kBatched, &error),
          "reference propose: " + error);
      Require(
          d == rd && BitIdentical(dl, rdl),
          "drafts from device-captured taps differ from host taps (prompt " +
              std::to_string(prompt_len) + ", step " + std::to_string(step) +
              ")");
      Tokens block{anchor};
      block.insert(block.end(), d.begin(), d.end());
      const std::uint32_t n = rig.s->position();
      const auto vl = SessionVerify(*rig.s, block);
      const auto rvl = ref.Run(block, gpu::Path::kVector, 0);
      Require(BitIdentical(vl, rvl), "session verify logits differ");
      Require(BitIdentical(vl, DecodeEach(*plain, block)),
              "session verify differs from solo decode");
      const std::uint32_t accept = static_cast<std::uint32_t>(step * 5) % 16;
      Require(rig.s->Commit(n + accept + 1, &error), "commit: " + error);
      Require(ref.ex->RewindTo(n + accept + 1, &error) &&
                  ref.d->kv().RewindTo(n + accept + 1, &error) &&
                  plain->RewindTo(n + accept + 1, &error),
              "reference rewind: " + error);
      anchor = block[accept + 1 < block.size() ? accept + 1 : 0] ^ 1;
      anchor %= static_cast<std::int32_t>(V);
    }
    Require(rig.s->drafter_active(), "drafter dropped without a failure");
  }
  std::cout << "session ok (prompts 9, 600, 1030: device-captured taps draft "
               "exactly as host taps; verify equals solo decode; commit)\n";
}

void SessionFailures(const TinyModel& tm, const exec::DeviceModel& dev,
                     const gpu::DrafterWeights& w) {
  const std::size_t V = tm.config().vocab_size;
  const Tokens prompt = TokensFor(tm, 560, 4242);
  const Tokens block = TokensFor(tm, 16, 4343);
  std::string error;
  // (1) The target fails after its K/V append: poisoned, drafter dropped.
  {
    SessionRig rig = MakeSession(dev, w, 64);
    (void)SessionPrefill(*rig.s, prompt);
    rig.ex->attention_ops().SetFailAfterStage("attn");
    std::vector<float> l(16 * V);
    Require(!rig.s->Verify(block, l.data(), &error) && rig.ex->poisoned() &&
                !rig.s->drafter_active(),
            "a target failure must poison the target and drop the drafter");
    rig.ex->attention_ops().SetFailAfterStage("");
    std::vector<std::int32_t> d;
    Require(!rig.s->Propose(3, 16, &d, nullptr, &error),
            "a dropped drafter still proposes");
    // Reset re-arms both and reproduces a fresh session.
    Require(rig.s->Reset(&error), "reset: " + error);
    SessionRig fresh = MakeSession(dev, w, 64);
    Require(BitIdentical(SessionPrefill(*rig.s, prompt),
                         SessionPrefill(*fresh.s, prompt)) &&
                SessionPropose(*rig.s, 3, nullptr) ==
                    SessionPropose(*fresh.s, 3, nullptr),
            "after Reset the session differs from a fresh one");
  }
  // (2) A drafter failure in the injection of a verify: target result stands,
  // drafter dropped, the target continues bit-identical to a plain twin.
  // (3) A drafter failure inside a block. (4) A non-finite tap row.
  for (int kind = 2; kind <= 4; ++kind) {
    SessionRig rig = MakeSession(dev, w, 64);
    auto twin = MakeExecutor(dev, 2048, 64, gpu::Path::kAuto);
    (void)SessionPrefill(*rig.s, prompt);
    PrefillTo(*twin, prompt, static_cast<std::uint32_t>(prompt.size()));
    std::vector<float> l(16 * V);
    if (kind == 2) {
      rig.s->drafter().ops().SetFailAfterStage("kv_stored-1");
      Require(rig.s->Verify(block, l.data(), &error),
              "a drafter failure failed the verify: " + error);
      Require(BitIdentical(l, DecodeEach(*twin, block)),
              "verify logits after a drafter failure differ");
    } else if (kind == 3) {
      rig.s->drafter().ops().SetFailAfterStage("Qcur-1");
      const std::uint32_t pos = rig.s->position();
      std::vector<std::int32_t> d;
      Require(!rig.s->Propose(5, 16, &d, nullptr, &error) &&
                  rig.s->position() == pos,
              "a failed block must fail the proposal only");
    } else {
      rig.s->PoisonNextCaptureForTesting();
      Require(rig.s->Verify(block, l.data(), &error), "verify: " + error);
      Require(BitIdentical(l, DecodeEach(*twin, block)),
              "verify logits with a poisoned tap differ");
      std::vector<std::int32_t> d;
      Require(!rig.s->Propose(5, 16, &d, nullptr, &error),
              "a non-finite tap row produced drafts");
    }
    rig.s->drafter().ops().SetFailAfterStage("");
    Require(!rig.s->drafter_active() && !rig.ex->poisoned() &&
                !rig.s->drafter_error().empty(),
            "a drafter failure must drop the drafter only (kind " +
                std::to_string(kind) + ")");
    // The target keeps going as plain decode, equal to its twin.
    const Tokens more = TokensFor(tm, 20, 4444 + kind);
    Require(BitIdentical(
                SessionVerify(*rig.s, Tokens(more.begin(), more.begin() + 12)),
                Plain(*twin, Tokens(more.begin(), more.begin() + 12),
                      gpu::Path::kVector)),
            "the target after a drafter failure differs from its twin");
    Require(rig.s->Commit(rig.s->position() - 3, &error) &&
                twin->RewindTo(twin->position() - 3, &error),
            "commit without a drafter: " + error);
    Require(BitIdentical(
                SessionVerify(*rig.s, Tokens(more.begin() + 12, more.end())),
                DecodeEach(*twin, Tokens(more.begin() + 12, more.end()))),
            "plain decode after a drafter failure differs from its twin");
  }
  // (5) A refused commit touches neither side.
  {
    SessionRig rig = MakeSession(dev, w, 16);
    (void)SessionPrefill(*rig.s, prompt);
    (void)SessionVerify(*rig.s, block);
    const State before = Observe(*rig.ex);
    const auto dk = std::tuple(rig.s->drafter().kv().committed(),
                               rig.s->drafter().kv().retained_begin());
    Require(!rig.s->Commit(rig.s->position() - 17, &error) &&
                rig.s->drafter_active(),
            "a commit past retained history was accepted");
    RequireUntouched(*rig.ex, before, "refused commit");
    Require(dk == std::tuple(rig.s->drafter().kv().committed(),
                             rig.s->drafter().kv().retained_begin()),
            "a refused commit moved the drafter");
  }
  // (6) Batched decode refuses a speculating (tapped) executor, and takes
  // it again once its DFlashSession is gone.
  {
    SessionRig rig = MakeSession(dev, w, 64);
    auto other = MakeExecutor(dev, 2048, 64, gpu::Path::kAuto);
    (void)SessionPrefill(*rig.s, prompt);
    PrefillTo(*other, prompt, 30);
    std::vector<float> l0(V), l1(V);
    std::vector<exec::DecodeItem> items(2);
    items[0].executor = rig.ex.get();
    items[0].token = 4;
    items[0].logits = l0.data();
    items[1].executor = other.get();
    items[1].token = 4;
    items[1].logits = l1.data();
    const std::uint32_t pos = rig.ex->position();
    exec::Executor::DecodeBatch(items);
    Require(
        !items[0].completed && items[1].completed && rig.ex->position() == pos,
        "DecodeBatch took a speculating executor");
    rig.s.reset();
    exec::Executor::DecodeBatch(std::span(items).subspan(0, 1));
    Require(items[0].completed, "DecodeBatch refused a plain executor");
  }
  std::cout << "session failure rules ok (target failure poisons and drops; "
               "drafter failures in injection, block and non-finite taps drop "
               "only the drafter; refused commit; batched decode excluded)\n";
}

void SessionThreads(const TinyModel& tm, const exec::DeviceModel& dev,
                    const gpu::DrafterWeights& w) {
  const std::size_t V = tm.config().vocab_size;
  constexpr int kThreads = 2;
  std::vector<Tokens> prompts;
  std::vector<Script> solo;
  for (int t = 0; t < kThreads; ++t) {
    prompts.push_back(TokensFor(tm, 520 + 30 * t, 8000 + t));
    SessionRig rig = MakeSession(dev, w, 64);
    solo.push_back(RunScript(*rig.s, prompts[t], 3 + 4 * t, V));
  }
  std::vector<SessionRig> rigs;
  for (int t = 0; t < kThreads; ++t) {
    rigs.push_back(MakeSession(dev, w, 64));
  }
  std::atomic<int> failures{0};
  std::atomic<bool> go{false};
  std::vector<Script> got(kThreads);
  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t] {
      try {
        while (!go.load()) {
          std::this_thread::yield();
        }
        got[t] = RunScript(*rigs[t].s, prompts[t], 3 + 4 * t, V);
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
  Require(failures.load() == 0, "a concurrent session failed");
  for (int t = 0; t < kThreads; ++t) {
    Require(got[t].drafts == solo[t].drafts &&
                got[t].logits.size() == solo[t].logits.size(),
            "concurrent drafts differ from the solo run");
    for (std::size_t i = 0; i < got[t].logits.size(); ++i) {
      Require(BitIdentical(got[t].logits[i], solo[t].logits[i]),
              "concurrent logits differ from the solo run");
    }
  }
  std::cout << "concurrent sessions ok (" << kThreads
            << " threads, two streams, equal to their solo runs)\n";
}

}  // namespace

int main() {
  try {
    const TinyModel tm(TargetSpec());
    const auto dev = tm.Upload();
    CaptureChecks(tm, *dev);
    RewindChecks(tm, *dev);
    RefusalChecks(tm, *dev);
    DrafterRingChecks();
    TinyDrafter fx(kDraftGeom, 0x6D6D);
    Require(fx.config().hidden_size == tm.config().hidden_size &&
                fx.config().tap_count() == kTaps.size(),
            "fixture geometry");
    DrafterCtx ctx(fx.drafter);
    SessionEquivalence(tm, *dev, ctx.weights());
    SessionFailures(tm, *dev, ctx.weights());
    SessionThreads(tm, *dev, ctx.weights());
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
  std::cout << "laguna.dflash_target: ok\n";
  return 0;
}
