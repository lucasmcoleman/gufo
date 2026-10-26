// GPU attention operator checks against the phase 2 CPU oracle, on random tiny
// layers: no model file, nothing skipped. attention_real_test.cpp repeats the
// checks on real GGUF layers.
//
// Every case runs a chunk plan through the operator tapped, then compares:
//
//  * each projection (q, k, v, o): the packed Q8 activation buffer the kernel
//    reads is captured through qfn_mmq's capture hook and validated against
//    the float input it was quantized from (ValidateQuantRows: every code
//    within the derived quantizer error of the nearest, one shared multiplier
//    per block, stored scales and sums the fp16 or float rounding of the exact
//    ones); only a valid capture feeds the oracle's exact product, which the
//    GPU output must match within the float accumulation bound
//    (CheckCapturedProjection, as in the FFN tests). Nothing is searched or
//    nudged, and q, k and v must have been given one quantization of their
//    shared input;
//  * the F32 gate projection and its softplus (2e-5 and 2e-6 relative);
//  * q after per-head RMSNorm and RoPE against the oracle's own norm and
//    ApplyRope applied to the GPU's projection (1e-6 of the head's largest
//    element, derived in AttnTolerances), and the length of the rotary channels
//    against mscale (YaRN's 1.3466 on full layers, 1 on sliding layers);
//  * every cached K and V row against the oracle's value: the stored binary16
//    must be within half an ulp of it plus the float disagreement before the
//    rounding (K: 3e-6 relative plus the rope tolerance of its head, because a
//    rotated element near zero is a difference of terms of the head's size; V
//    is stored from the v_proj values themselves, so half an ulp exactly);
//  * attention against reference::GqaAttention (double softmax) run on the
//    GPU's own q and the cache as stored. The GPU sums in float, so the bound
//    is three times the distance of a plain float CPU implementation of the
//    same arithmetic from the double result (the float noise of that input),
//    floored at 2e-6 of rms; both are printed. Gating is the single float
//    product attn * softplus (bitwise);
//  * for plans using one projection mode, the whole chain before Wo: the gated
//    attention rows against the oracle recomposed from its public blocks
//    (MatMulRows, RmsNorm, ApplyRope, GqaAttention) in that mode, fed the GPU's
//    stored K/V rows (the F16 rounding of a float within noise of a boundary
//    can differ between GPU and oracle; the stored rows remove that). The bound
//    is four times the float noise a plain float attention adds to those rows
//    (the attention bound above, scaled by the gate) plus 5e-6 of rms for the
//    projections' summation order.
//    The recomposition is proved bit-identical to reference::AttnForward on
//    the same input. The Q8 rounding of Wo's input is the "o" projection stage
//    above; and the final output is compared with AttnForward itself at 3e-2 of
//    rms, a sanity bound: there the two sides round their own Wo input, a
//    near-tie element takes a different Q8 code on each, and one flip moves an
//    output by up to a few 1e-3 of rms (row 0 attends to a single key, so a KV
//    group's heads are copies that flip together, up to 1e-2), against Q8
//    noise of 8e-2 or more (printed); a wiring defect is far larger;
//  * the window boundary: where asked, the rows are also compared with the
//    same attention using a window one key narrower and one wider, which must
//    be at least 100 times further from the GPU than the correct window is.
//
// Each case then reruns untapped (no synchronization between chunks) twice;
// both must reproduce the tapped output bit for bit. Plans of one path over
// the same sequence must agree bit for bit with each other whatever the split.
#include <atomic>
#include <iostream>
#include <memory>
#include <set>
#include <thread>

#include "tests/models/laguna/attention_gpu.hpp"

using namespace gufo::testing::laguna;

namespace {

std::map<std::string, GroupWorst> g_groups;
std::size_t g_cases = 0;

struct Fixture {
  Fixture(const AttnShape& shape, std::uint64_t seed) : tiny(shape, seed) {
    for (std::uint32_t il = 0; il < 2; ++il) {
      std::string error;
      auto up =
          gpu::LayerAttn::Upload(tiny.config(), tiny.layer(il), il, &error);
      Require(up.has_value(), "upload: " + error);
      dev[il] = std::move(*up);
    }
  }
  TinyAttn tiny;
  gpu::LayerAttn dev[2];
  AttnContext ctx;
};

/// Runs `plan` tapped and checks it, then reruns untapped twice and requires
/// identical bits. Returns the tapped output.
std::vector<float> RunCase(const std::string& group, const std::string& label,
                           Fixture& fx, std::uint32_t il,
                           const std::vector<float>& n, const Plan& plan,
                           CheckOptions opt) {
  const ml::Config& config = fx.tiny.config();
  const GpuRun tapped =
      RunPlan(fx.ctx, fx.dev[il].weights(), config, il, n, plan, true);
  const AttnReport rep =
      CheckRun(label, config, fx.tiny.layer(il), il, n, plan, tapped, opt);
  PrintAttnReport(rep);
  g_groups[group].Add(rep);
  ++g_cases;
  for (int rerun = 0; rerun < 2; ++rerun) {
    const GpuRun again =
        RunPlan(fx.ctx, fx.dev[il].weights(), config, il, n, plan, false);
    Require(BitIdentical(again.out, tapped.out),
            label + ": untapped rerun differs from the tapped run");
  }
  return tapped.out;
}

std::string Label(const AttnShape& s, std::uint32_t il,
                  const std::string& what) {
  return std::string(s.name) + (il == 0 ? " full" : " swa") + " " + what;
}

// ---------------------------------------------------------------------------

/// Full and sliding layers on every shape (GQA 6/9, 2/4, 8/12), prefill in one
/// chunk or two, on both projection paths, and a decode tail.
void TestBasic() {
  std::cout << "basic\n";
  for (const AttnShape* shape : {&kModelShape, &kOddShape, &kWideShape}) {
    Fixture fx(*shape, 100 + shape->hidden);
    for (std::uint32_t il = 0; il < 2; ++il) {
      const std::uint32_t window = il == 0 ? 0 : shape->window;
      // Cross the window of the short-window shapes, stay small otherwise.
      const std::uint32_t T = window != 0 && window < 100 ? 2 * window + 9 : 45;
      const auto n = TinyRows(T, shape->hidden, fx.tiny.rng());
      std::map<int, OracleOut> oracle;
      CheckOptions opt;
      opt.oracle_cache = &oracle;
      opt.f32_reference = true;
      const std::string group =
          std::string(shape->name) + (il == 0 ? " full" : " swa");
      std::vector<float> first_mmq;
      std::vector<float> first_vec;
      const std::vector<std::pair<std::string, Plan>> plans = {
          {"one chunk", UniformPlan({T}, gpu::Path::kBatched)},
          {"two chunks", UniformPlan({T / 2, T - T / 2}, gpu::Path::kBatched)},
          {"prefill + 5 decode rows",
           UniformPlan(Cat({T - 5}, Singles(5)), gpu::Path::kBatched)},
          {"vector chunks of 32",
           UniformPlan(Cat(std::vector<std::uint32_t>(T / 32, 32U),
                           T % 32 != 0 ? std::vector<std::uint32_t>{T % 32}
                                       : std::vector<std::uint32_t>{}),
                       gpu::Path::kVector)},
          {"vector decode singles",
           UniformPlan(Singles(T), gpu::Path::kVector)}};
      for (const auto& [what, plan] : plans) {
        const bool vec = plan.paths[0] == gpu::Path::kVector;
        const auto out =
            RunCase(group, Label(*shape, il, what + " T=" + std::to_string(T)),
                    fx, il, n, plan, opt);
        auto& first = vec ? first_vec : first_mmq;
        if (first.empty()) {
          first = out;
        } else {
          Require(
              BitIdentical(first, out),
              Label(*shape, il, what) + ": output depends on the chunk split");
        }
      }
    }
  }
}

/// Query rows 511, 512 and 513 of a 512-window layer, in a prefill chunk and
/// as decode rows, with the window probe.
void TestBoundary() {
  std::cout << "window boundary (window 512)\n";
  Fixture fx(kModelShape, 7001);
  const std::uint32_t il = 1;
  const std::uint32_t T = 520;
  const auto n = TinyRows(T, kModelShape.hidden, fx.tiny.rng());
  std::map<int, OracleOut> oracle;
  CheckOptions opt;
  opt.oracle_cache = &oracle;
  opt.boundary_rows = {511, 512, 513};
  std::vector<float> first;
  for (const auto& [what, plan] : std::vector<std::pair<std::string, Plan>>{
           {"prefill 520", UniformPlan({T}, gpu::Path::kBatched)},
           {"prefill 511 then decode rows 511..519",
            UniformPlan(Cat({511}, Singles(9)), gpu::Path::kBatched)},
           {"prefill 512 then decode rows 512..519",
            UniformPlan(Cat({512}, Singles(8)), gpu::Path::kBatched)},
           {"prefill 513 + 7 rows", UniformPlan({513, 7}, gpu::Path::kBatched)},
           {"vector chunks of 29 across the boundary",
            UniformPlan(Cat(std::vector<std::uint32_t>(17, 29U), {27}),
                        gpu::Path::kVector)}}) {
    const auto out = RunCase("window boundary", Label(kModelShape, il, what),
                             fx, il, n, plan, opt);
    if (plan.paths[0] == gpu::Path::kBatched) {
      if (first.empty()) {
        first = out;
      } else {
        Require(BitIdentical(first, out),
                what + ": output depends on the chunk split");
      }
    }
  }
}

/// A chunk that crosses the window, split in many ways: identical outputs.
void TestChunkSplits() {
  std::cout << "chunk splits crossing the window\n";
  for (std::uint32_t il = 0; il < 2; ++il) {
    Fixture fx(kModelShape, 7100 + il);
    const std::uint32_t T = 600;
    const auto n = TinyRows(T, kModelShape.hidden, fx.tiny.rng());
    std::map<int, OracleOut> oracle;
    CheckOptions opt;
    opt.oracle_cache = &oracle;
    std::vector<float> first;
    const std::vector<std::vector<std::uint32_t>> splits = {
        {600},
        {300, 300},
        {512, 88},
        {513, 87},
        {511, 89},
        {1, 599},
        {511, 1, 88},
        {97, 97, 97, 97, 97, 115},
        {200, 1, 1, 1, 397},
        {64, 64, 64, 64, 64, 64, 64, 64, 64, 24}};
    for (const auto& chunks : splits) {
      std::string what = "split";
      for (const auto c : chunks) {
        what += " " + std::to_string(c);
      }
      const auto out =
          RunCase("chunk splits", Label(kModelShape, il, what), fx, il, n,
                  UniformPlan(chunks, gpu::Path::kBatched), opt);
      if (first.empty()) {
        first = out;
      } else {
        Require(BitIdentical(first, out),
                what + ": output depends on the chunk split");
      }
    }
  }
}

/// More than 1024 tokens through a ring of window + chunk: positions wrap.
void TestRingWrap() {
  std::cout << "ring wrap (T=1120)\n";
  for (std::uint32_t il = 1; il < 2; ++il) {
    Fixture fx(kModelShape, 7200);
    const std::uint32_t T = 1120;
    const auto n = TinyRows(T, kModelShape.hidden, fx.tiny.rng());
    std::map<int, OracleOut> oracle;
    CheckOptions opt;
    opt.oracle_cache = &oracle;
    std::vector<float> first_mmq;
    for (const auto& [what, plan] : std::vector<std::pair<std::string, Plan>>{
             {"chunks of 128 (ring 640)",
              UniformPlan(Cat(std::vector<std::uint32_t>(8, 128U), {96}),
                          gpu::Path::kBatched)},
             {"chunks of 1 (ring 513)",
              UniformPlan(Singles(T), gpu::Path::kBatched)},
             {"one chunk (ring 1632)", UniformPlan({T}, gpu::Path::kBatched)},
             {"vector 32 then 64 decode rows (ring 544)",
              UniformPlan(Cat(std::vector<std::uint32_t>(33, 32U), Singles(64)),
                          gpu::Path::kVector)}}) {
      const auto out = RunCase("ring wrap", Label(kModelShape, il, what), fx,
                               il, n, plan, opt);
      if (plan.paths[0] == gpu::Path::kBatched) {
        if (first_mmq.empty()) {
          first_mmq = out;
        } else {
          Require(BitIdentical(first_mmq, out),
                  what + ": ring output differs from the other splits");
        }
      }
    }
  }
  // The same sequence on a full layer (its capacity is the context).
  {
    Fixture fx(kModelShape, 7201);
    const std::uint32_t T = 1120;
    const auto n = TinyRows(T, kModelShape.hidden, fx.tiny.rng());
    std::map<int, OracleOut> oracle;
    CheckOptions opt;
    opt.oracle_cache = &oracle;
    RunCase("ring wrap", Label(kModelShape, 0, "chunks of 128 T=1120"), fx, 0,
            n,
            UniformPlan(Cat(std::vector<std::uint32_t>(8, 128U), {96}),
                        gpu::Path::kBatched),
            opt);
  }
}

/// A prefill followed by decode rows equals the all-prefill run exactly when
/// the decode rows use the same projection path; the automatic path (decode on
/// the vector kernels) is checked against its own modes.
void TestPrefillThenDecode() {
  std::cout << "prefill then decode\n";
  for (std::uint32_t il = 0; il < 2; ++il) {
    Fixture fx(kModelShape, 7300 + il);
    const std::uint32_t T = 530;
    const auto n = TinyRows(T, kModelShape.hidden, fx.tiny.rng());
    std::map<int, OracleOut> oracle;
    CheckOptions opt;
    opt.oracle_cache = &oracle;
    const auto all =
        RunCase("prefill then decode", Label(kModelShape, il, "all prefill"),
                fx, il, n, UniformPlan({T}, gpu::Path::kBatched), opt);
    const auto split =
        RunCase("prefill then decode",
                Label(kModelShape, il, "524 + 6 decode rows (mmq)"), fx, il, n,
                UniformPlan(Cat({524}, Singles(6)), gpu::Path::kBatched), opt);
    Require(BitIdentical(all, split),
            "prefill then decode (same path) differs from all-prefill");
    // Mixed projection paths: stage checks only (the oracle has one mode).
    CheckOptions mixed = opt;
    mixed.e2e = false;
    Plan vec_decode = UniformPlan(Cat({524}, Singles(6)), gpu::Path::kBatched);
    for (std::size_t i = 1; i < vec_decode.paths.size(); ++i) {
      vec_decode.paths[i] = gpu::Path::kVector;
    }
    const auto a = RunCase("prefill then decode",
                           Label(kModelShape, il, "524 mmq + 6 decode vec"), fx,
                           il, n, vec_decode, mixed);
    Plan automatic = UniformPlan(Cat({524}, Singles(6)), gpu::Path::kAuto);
    const auto b = RunCase("prefill then decode",
                           Label(kModelShape, il, "524 + 6 decode rows (auto)"),
                           fx, il, n, automatic, mixed);
    Require(BitIdentical(a, b),
            "automatic path is not batched prefill + vector decode");
    // The decode rows of the mixed run stay close to the all-prefill rows:
    // different Q8 layouts, so only Q8-level agreement is expected. Report it.
    double worst = 0;
    const std::size_t H = kModelShape.hidden;
    const double rms = Rms(all);
    for (std::size_t i = 524 * H; i < all.size(); ++i) {
      worst = std::max(worst, std::fabs(static_cast<double>(a[i]) - all[i]));
    }
    std::cout << "    decode rows, vector vs batched projections: max |diff| "
              << worst / rms << " of rms (Q8 layout noise, informational)\n";
  }
}

/// Two streams, each with its own operator and cache, share the device
/// weights and run at once; every result must equal the solo run's bits.
void TestConcurrent() {
  std::cout << "two concurrent streams\n";
  Fixture fx(kModelShape, 7400);
  const std::uint32_t T = 600;
  const auto n = TinyRows(T, kModelShape.hidden, fx.tiny.rng());
  const Plan plans[2] = {
      UniformPlan({300, 300}, gpu::Path::kBatched),
      UniformPlan({200, 1, 199, 100, 100}, gpu::Path::kBatched)};
  std::vector<float> solo[2];
  for (std::uint32_t il = 0; il < 2; ++il) {
    solo[il] = RunPlan(fx.ctx, fx.dev[il].weights(), fx.tiny.config(), il, n,
                       plans[0], false)
                   .out;
  }
  for (std::uint32_t il = 0; il < 2; ++il) {
    Require(
        BitIdentical(solo[il], RunPlan(fx.ctx, fx.dev[il].weights(),
                                       fx.tiny.config(), il, n, plans[1], false)
                                   .out),
        "solo runs of two plans differ on layer " + std::to_string(il));
  }
  std::atomic<int> failures{0};
  std::string messages[2];
  std::vector<std::thread> threads;
  for (int t = 0; t < 2; ++t) {
    threads.emplace_back([&, t] {
      try {
        AttnContext ctx;  // own stream, own operator scratch
        for (int round = 0; round < 6; ++round) {
          for (std::uint32_t il = 0; il < 2; ++il) {
            const auto out = RunPlan(ctx, fx.dev[il].weights(),
                                     fx.tiny.config(), il, n, plans[t], false)
                                 .out;
            Require(
                BitIdentical(out, solo[il]),
                "stream " + std::to_string(t) + " differs from the solo run");
          }
        }
      } catch (const std::exception& e) {
        messages[t] = e.what();
        ++failures;
      }
    });
  }
  for (auto& th : threads) {
    th.join();
  }
  Require(failures == 0, "concurrent streams: " + messages[0] + messages[1]);
  std::cout << "  2 streams x 6 rounds x 2 layers: bit-identical to solo\n";
  ++g_cases;
}

// ---- allocation failure
// ------------------------------------------------------

struct MockState {
  int calls{0};
  int fail_at{0};  // 1-based call to fail; 0 never
  std::set<void*> live;
  int bad_frees{0};
};
MockState g_mock;

hipError_t MockAllocate(void** ptr, std::size_t bytes) {
  ++g_mock.calls;
  if (g_mock.calls == g_mock.fail_at) {
    *ptr = nullptr;
    return hipErrorOutOfMemory;
  }
  const hipError_t status = hipMalloc(ptr, bytes);
  if (status == hipSuccess) {
    g_mock.live.insert(*ptr);
  }
  return status;
}
hipError_t MockRelease(void* ptr) {
  if (g_mock.live.erase(ptr) == 0) {
    ++g_mock.bad_frees;  // freed twice, or never allocated here
    return hipSuccess;
  }
  return hipFree(ptr);
}

/// A failed scratch allocation must leave the operator and the cache as they
/// were (nothing freed early, nothing double freed) and later calls must work.
void TestAllocationFailure() {
  std::cout << "scratch allocation failure\n";
  Fixture fx(kModelShape, 7500);
  const std::uint32_t il = 1;
  const std::size_t H = kModelShape.hidden;
  const auto n = TinyRows(48, H, fx.tiny.rng());
  const auto& w = fx.dev[il].weights();
  const Plan reference_plan = UniformPlan({8, 8, 32}, gpu::Path::kBatched);
  const auto reference =
      RunPlan(fx.ctx, w, fx.tiny.config(), il, n, reference_plan, false).out;
  float* dn = ToDevice(n);
  float* dout = nullptr;
  CheckHip(hipMalloc(&dout, 48 * H * sizeof(float)), "alloc out");
  const auto host_out = [&](std::size_t first_row, std::size_t rows) {
    CheckHip(hipStreamSynchronize(fx.ctx.stream()), "sync");
    return ToHost(dout + first_row * H, rows * H);
  };
  const auto rows_of = [&](std::size_t first_row, std::size_t rows) {
    return std::vector<float>(reference.begin() + first_row * H,
                              reference.begin() + (first_row + rows) * H);
  };
  std::string error;
  // 1. A failure while growing the scratch of the very first call, at each of
  // the six scratch buffers.
  for (int fail_at = 1; fail_at <= 6; ++fail_at) {
    g_mock = {};
    {
      auto cache = gpu::KvCache::Create(fx.tiny.config(), il, 48, 32, &error);
      Require(cache.has_value(), error);
      gpu::AttnOps ops(fx.ctx.stream());
      ops.SetAllocator({MockAllocate, MockRelease});
      g_mock.fail_at = fail_at;
      Require(
          !ops.Forward(w, *cache, dn, dout, 8, 0, gpu::Path::kBatched, &error),
          "Forward survived a failed scratch allocation");
      Require(error.find("hipMalloc scratch") != std::string::npos,
              "unexpected error text: " + error);
      Require(cache->committed() == 0, "failed call changed the cache");
      Require(g_mock.live.size() == static_cast<std::size_t>(fail_at - 1),
              "failed growth freed or leaked scratch");
      g_mock.fail_at = 0;
      Require(
          ops.Forward(w, *cache, dn, dout, 8, 0, gpu::Path::kBatched, &error),
          "retry after a failed allocation: " + error);
      Require(BitIdentical(host_out(0, 8), rows_of(0, 8)),
              "retry after a failed allocation gave different rows");
    }
    Require(g_mock.live.empty(), "scratch leaked after destruction");
    Require(g_mock.bad_frees == 0, "scratch freed twice or never allocated");
  }
  // 2. A failure while growing existing buffers: the old buffers stay owned
  // and valid, so a call that fits them still works, and growth works later.
  for (int fail_at = 1; fail_at <= 6; ++fail_at) {
    g_mock = {};
    {
      auto cache = gpu::KvCache::Create(fx.tiny.config(), il, 48, 32, &error);
      Require(cache.has_value(), error);
      gpu::AttnOps ops(fx.ctx.stream());
      ops.SetAllocator({MockAllocate, MockRelease});
      Require(
          ops.Forward(w, *cache, dn, dout, 8, 0, gpu::Path::kBatched, &error),
          error);
      const std::size_t live_after_first = g_mock.live.size();
      g_mock.calls = 0;
      g_mock.fail_at = fail_at;
      Require(!ops.Forward(w, *cache, dn + 8 * H, dout + 8 * H, 32, 8,
                           gpu::Path::kBatched, &error),
              "growth succeeded despite the injected failure");
      Require(cache->committed() == 8, "failed growth changed the cache");
      Require(g_mock.live.size() == live_after_first,
              "failed growth changed the set of live buffers");
      g_mock.fail_at = 0;
      Require(ops.Forward(w, *cache, dn + 8 * H, dout + 8 * H, 8, 8,
                          gpu::Path::kBatched, &error),
              "call that fits the old scratch failed: " + error);
      Require(BitIdentical(host_out(8, 8), rows_of(8, 8)),
              "rows after a failed growth differ");
      Require(ops.Forward(w, *cache, dn + 16 * H, dout + 16 * H, 32, 16,
                          gpu::Path::kBatched, &error),
              "growth after the failure: " + error);
      Require(BitIdentical(host_out(16, 32), rows_of(16, 32)),
              "rows after the retried growth differ");
    }
    Require(g_mock.live.empty(), "scratch leaked after destruction");
    Require(g_mock.bad_frees == 0, "scratch freed twice or never allocated");
  }
  (void)hipFree(dn);
  (void)hipFree(dout);
  std::cout << "  12 injected failures: no leak, no double free, no state "
               "change, retries bit-identical\n";
  ++g_cases;
}

/// Bad arguments are rejected before anything runs, and a rewind recomputes
/// the same rows on a ring that has wrapped.
void TestArguments() {
  std::cout << "argument checks and ring rewind\n";
  Fixture fx(kOddShape, 7600);  // window 37
  const std::uint32_t il = 1;
  const std::size_t H = kOddShape.hidden;
  const auto& w = fx.dev[il].weights();
  const auto n = TinyRows(100, H, fx.tiny.rng());
  float* dn = ToDevice(n);
  float* dout = nullptr;
  CheckHip(hipMalloc(&dout, 100 * H * sizeof(float)), "alloc out");
  std::string error;
  auto cache = gpu::KvCache::Create(fx.tiny.config(), il, 100, 16, &error);
  Require(cache.has_value(), error);
  Require(cache->capacity() == 37 + 16, "ring capacity is window + chunk");
  auto& ops = fx.ctx.ops();
  const auto fails = [&](std::uint32_t rows, std::uint32_t start,
                         gpu::Path path, const char* what) {
    const std::uint32_t before = cache->committed();
    Require(!ops.Forward(w, *cache, dn, dout, rows, start, path, &error),
            std::string("accepted: ") + what);
    Require(cache->committed() == before, std::string("mutated on: ") + what);
  };
  fails(0, 0, gpu::Path::kBatched, "zero rows");
  fails(17, 0, gpu::Path::kBatched, "chunk larger than the ring slack");
  fails(33, 0, gpu::Path::kVector, "33 rows on the vector path");
  fails(4, 3, gpu::Path::kBatched, "start beyond the cached rows");
  // Run 100 positions in chunks of 10, then rewind within and beyond the slack.
  std::vector<float> original;
  for (std::uint32_t pos = 0; pos < 100; pos += 10) {
    Require(ops.Forward(w, *cache, dn + pos * H, dout + pos * H, 10, pos,
                        gpu::Path::kBatched, &error),
            error);
  }
  CheckHip(hipStreamSynchronize(fx.ctx.stream()), "sync");
  original = ToHost(dout, 100 * H);
  fails(10, 70, gpu::Path::kBatched, "rewind by 30 on a ring with 16 slack");
  Require(ops.Forward(w, *cache, dn + 90 * H, dout + 90 * H, 10, 90,
                      gpu::Path::kBatched, &error),
          error);  // rewind by 10 and recompute positions 90..99
  CheckHip(hipStreamSynchronize(fx.ctx.stream()), "sync");
  const auto again = ToHost(dout, 100 * H);
  Require(BitIdentical(again, original),
          "rewound rows differ from the first computation");
  Require(ops.Forward(w, *cache, dn + 92 * H, dout + 92 * H, 8, 92,
                      gpu::Path::kBatched, &error),
          error);
  CheckHip(hipStreamSynchronize(fx.ctx.stream()), "sync");
  Require(cache->committed() == 100, "committed after a shorter rewind");
  const auto tail = ToHost(dout + 92 * H, 8 * H);
  Require(std::memcmp(tail.data(), original.data() + 92 * H,
                      8 * H * sizeof(float)) == 0,
          "mid-chunk rewind differs");
  // Full layer: context overflow.
  auto full = gpu::KvCache::Create(fx.tiny.config(), 0, 20, 16, &error);
  Require(full.has_value(), error);
  Require(!ops.Forward(fx.dev[0].weights(), *full, dn, dout, 21, 0,
                       gpu::Path::kBatched, &error),
          "accepted rows past the context of a full layer");
  (void)hipFree(dn);
  (void)hipFree(dout);
  std::cout << "  rejected bad calls untouched; rewind on a wrapped ring "
               "reproduces the rows\n";
  ++g_cases;
}

}  // namespace

int main() {
  try {
    TestBasic();
    TestBoundary();
    TestChunkSplits();
    TestRingWrap();
    TestPrefillThenDecode();
    TestConcurrent();
    TestAllocationFailure();
    TestArguments();
    std::cout << "SUMMARY " << g_cases << " attention operator cases passed\n";
    for (const auto& [name, g] : g_groups) {
      PrintGroup(name, g);
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << "\n";
    return 1;
  }
}
