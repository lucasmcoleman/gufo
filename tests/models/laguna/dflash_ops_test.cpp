// Laguna DFlash drafter GPU operators (phase 6c) against the phase 6b CPU
// oracle (reference::DFlashModel), on tiny random drafters: no model file,
// nothing skipped. dflash_gpu_real_test.cpp runs the composed forward on the
// real drafter.
//
// Fixtures (dflash_gpu.hpp): every tensor random, dimensions unequal (hidden
// 512 or 768, FFN 768 or 512, GQA 9 over 2 KV heads or GQA 4 over 3, two or
// three taps, window 512 or 37, theta 500000 or 10000, vocabulary 1000 or
// 777), Q4_K projections with the last layer's attn_v and ffn_down in Q6_K.
// So a transposed projection, a swapped tap or a swapped K/V cannot pass.
//
// Every case runs the operator tapped, with the packed Q8 activation buffer of
// every projection captured through qfn_mmq's hook, and checks:
//
//  * each projection against the oracle's exact product of its captured,
//    independently validated quantized input, within the float accumulation
//    bound (CheckCapturedProjection, as in the target's FFN and attention
//    tests): kernel arithmetic is judged against the expected quantization,
//    not against a loose tolerance;
//  * norms and the other float stages against the oracle's building blocks
//    applied to the GPU's own upstream stage;
//  * the operator end to end against DFlashModel's taps in the same
//    activation mode (Q8_1 for the vector path, MMQ for the batched path), and
//    that this distance is far below the quantization noise (the distance to
//    the F32 oracle, printed);
//  * an untapped rerun (no synchronization between stages; taps hide races)
//    that must reproduce the tapped output bit for bit.
//
// Negative tests mutate the input or the device view (swapped taps, wrong
// norm row) and require the same checks to fail.
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "tests/models/laguna/dflash_gpu.hpp"

using namespace gufo::testing::laguna;

namespace {

std::size_t g_cases = 0;

struct EncoderRun {
  std::vector<float> fused;
  Taps taps;
};

EncoderRun RunEncoder(DrafterCtx& ctx, const gpu::DrafterWeights& w,
                      const std::vector<float>& features, std::uint32_t T,
                      gpu::Path path, bool tapped) {
  EncoderRun run;
  float* df = ToDevice(features);
  float* dout = nullptr;
  CheckHip(hipMalloc(&dout, std::size_t{T} * w.hidden * sizeof(float)),
           "alloc fused");
  std::string error;
  bool ok = false;
  if (tapped) {
    TapScope scope(ctx.ops(), &run.taps);
    ok = ctx.ops().Encode(w, df, dout, T, path, &error);
    CheckHip(hipStreamSynchronize(ctx.stream()), "sync");
  } else {
    ok = ctx.ops().Encode(w, df, dout, T, path, &error);
    CheckHip(hipStreamSynchronize(ctx.stream()), "sync");
  }
  Require(ok, "Encode failed: " + error);
  Require(run.taps.capture_error.empty(),
          "activation capture failed: " + run.taps.capture_error);
  Require(run.taps.pending.empty(),
          "activation buffers captured after the last stage");
  run.fused = ToHost(dout, std::size_t{T} * w.hidden);
  (void)hipFree(df);
  (void)hipFree(dout);
  return run;
}

/// Runs the encoder tapped and checked, then untapped, which must be
/// bit-identical.
void EncoderCase(TinyDrafter& fx, DrafterCtx& ctx, std::uint32_t T,
                 gpu::Path path) {
  const ml::DFlashConfig& c = fx.config();
  const auto features =
      RandomFeatures(T, c.tap_count(), c.hidden_size, fx.rng());
  const bool vector = UsesVector(path, T);
  const std::string label =
      std::string(fx.shape().name) + " encoder " + CaseLabel("", T, path);
  const EncoderRun tapped =
      RunEncoder(ctx, ctx.weights(), features, T, path, true);
  PrintEncoder(
      CheckEncoder(label, fx, features, T, vector, tapped.taps, tapped.fused));
  const EncoderRun plain =
      RunEncoder(ctx, ctx.weights(), features, T, path, false);
  Require(BitIdentical(plain.fused, tapped.fused),
          label + ": untapped run differs from the tapped run");
  ++g_cases;
}

void TestEncoder() {
  std::cout << "encoder\n";
  for (const DrafterShape* shape : {&kDraftGeom, &kDraftOdd}) {
    TinyDrafter fx(*shape, 0xE0C0DE + shape->hidden);
    DrafterCtx ctx(fx.drafter);
    for (const std::uint32_t T : {1U, 5U, 16U, 32U}) {
      EncoderCase(fx, ctx, T, gpu::Path::kVector);
    }
    for (const std::uint32_t T : {1U, 7U, 40U, 129U}) {
      EncoderCase(fx, ctx, T, gpu::Path::kBatched);
    }
  }
}

/// A check that must throw: returns its message.
template<class Fn>
std::string MustFail(const std::string& what, const Fn& fn) {
  try {
    fn();
  } catch (const std::exception& e) {
    std::cout << "  negative " << what << ": fails as expected (" << e.what()
              << ")\n";
    return e.what();
  }
  Require(false, "negative test passed although it must fail: " + what);
  return {};
}

void TestEncoderNegative() {
  std::cout << "encoder negative\n";
  TinyDrafter fx(kDraftGeom, 0xBAD0E0C);
  DrafterCtx ctx(fx.drafter);
  const ml::DFlashConfig& c = fx.config();
  const std::size_t H = c.hidden_size;
  const std::uint32_t T = 6;
  const auto features = RandomFeatures(T, c.tap_count(), H, fx.rng());
  for (const gpu::Path path : {gpu::Path::kVector, gpu::Path::kBatched}) {
    const bool vector = path == gpu::Path::kVector;
    // Taps 0 and 1 swapped in the GPU's input.
    std::vector<float> swapped = features;
    for (std::uint32_t t = 0; t < T; ++t) {
      std::swap_ranges(swapped.begin() + (t * c.tap_count()) * H,
                       swapped.begin() + (t * c.tap_count() + 1) * H,
                       swapped.begin() + (t * c.tap_count() + 1) * H);
    }
    const EncoderRun a = RunEncoder(ctx, ctx.weights(), swapped, T, path, true);
    MustFail(std::string("swapped taps, ") + PathName(path), [&] {
      (void)CheckEncoder("swapped taps", fx, features, T, vector, a.taps,
                         a.fused);
    });
    // The operator must refuse a geometry its fc does not match.
    {
      gpu::DrafterWeights inconsistent = ctx.weights();
      inconsistent.taps -= 1;
      float* df = ToDevice(features);
      float* dout = nullptr;
      CheckHip(hipMalloc(&dout, T * H * sizeof(float)), "alloc");
      std::string error;
      Require(!ctx.ops().Encode(inconsistent, df, dout, T, path, &error),
              "Encode accepted an fc that does not match the taps");
      (void)hipFree(df);
      (void)hipFree(dout);
    }
    // Tap k normalized with aux_norm row k + 1 (rows rotated on the device).
    {
      const auto aux = Values(fx.drafter.aux_norm);
      std::vector<float> rotated(aux.size());
      for (std::size_t k = 0; k < c.tap_count(); ++k) {
        const std::size_t from = (k + 1) % c.tap_count();
        std::copy_n(aux.begin() + from * H, H, rotated.begin() + k * H);
      }
      float* daux = ToDevice(rotated);
      gpu::DrafterWeights view = ctx.weights();
      view.aux_norm = daux;
      const EncoderRun r = RunEncoder(ctx, view, features, T, path, true);
      MustFail(std::string("aux_norm rows rotated, ") + PathName(path), [&] {
        (void)CheckEncoder("rotated aux rows", fx, features, T, vector, r.taps,
                           r.fused);
      });
      (void)hipFree(daux);
    }
    gpu::DrafterWeights wrong_norm = ctx.weights();
    wrong_norm.enc_norm = ctx.weights().output_norm;
    const EncoderRun b = RunEncoder(ctx, wrong_norm, features, T, path, true);
    MustFail(std::string("output norm weight swapped, ") + PathName(path), [&] {
      (void)CheckEncoder("wrong norm", fx, features, T, vector, b.taps,
                         b.fused);
    });
    ++g_cases;
  }
}

/// Scratch growth that fails must leave the operator usable and its results
/// unchanged (allocate before free).
hipError_t FailingAllocate(void**, std::size_t) {
  return hipErrorOutOfMemory;
}
hipError_t PlainRelease(void* p) {
  return hipFree(p);
}
hipError_t PlainAllocate(void** p, std::size_t bytes) {
  return hipMalloc(p, bytes);
}

void TestEncoderAllocationFailure() {
  std::cout << "encoder allocation failure\n";
  TinyDrafter fx(kDraftOdd, 0xA110C);
  DrafterCtx ctx(fx.drafter);
  const ml::DFlashConfig& c = fx.config();
  const auto small = RandomFeatures(4, c.tap_count(), c.hidden_size, fx.rng());
  const auto big = RandomFeatures(64, c.tap_count(), c.hidden_size, fx.rng());
  const EncoderRun before =
      RunEncoder(ctx, ctx.weights(), small, 4, gpu::Path::kBatched, false);
  const std::size_t held = ctx.ops().scratch_bytes();
  ctx.ops().SetAllocator({FailingAllocate, PlainRelease});
  float* df = ToDevice(big);
  float* dout = nullptr;
  CheckHip(hipMalloc(&dout, 64 * c.hidden_size * sizeof(float)), "alloc");
  std::string error;
  Require(!ctx.ops().Encode(ctx.weights(), df, dout, 64, gpu::Path::kBatched,
                            &error),
          "Encode succeeded although every allocation fails");
  Require(ctx.ops().scratch_bytes() == held,
          "a failed growth changed the scratch held");
  std::cout << "  growth refused: " << error << "\n";
  const EncoderRun after =
      RunEncoder(ctx, ctx.weights(), small, 4, gpu::Path::kBatched, false);
  Require(BitIdentical(before.fused, after.fused),
          "result changed after a failed scratch growth");
  ctx.ops().SetAllocator({PlainAllocate, PlainRelease});
  (void)hipFree(df);
  (void)hipFree(dout);
  ++g_cases;
}

// ---------------------------------------------------------------------------
// K/V injection
// ---------------------------------------------------------------------------

struct InjectPlanRun {
  KvRows hist;   ///< positions [start, start + N) as each chunk stored them
  KvRows final;  ///< the rings after the plan: [retained, committed)
  InjectReport rep;
};

/// Injects `features` ([N][taps*hidden], positions start ..) chunk by chunk
/// into `kv` (Reset(start) first). Tapped runs synchronize per chunk, check
/// every chunk's stages and stored rows (when `check`) and record them;
/// untapped runs queue every chunk back to back and synchronize once.
InjectPlanRun RunInjectPlan(const std::string& label, TinyDrafter& fx,
                            DrafterCtx& ctx, const gpu::DrafterWeights& w,
                            gpu::DrafterKv& kv,
                            const std::vector<float>& features,
                            const Plan& plan, std::uint32_t start, bool tapped,
                            bool check = true) {
  const ml::DFlashConfig& c = fx.config();
  const std::size_t width = c.fused_input_width();
  const std::uint32_t N = plan.total();
  Require(features.size() == N * width, "RunInjectPlan: features");
  InjectPlanRun run;
  run.rep.label = label;
  run.hist.begin = start;
  run.hist.end = start + N;
  run.hist.layers = c.num_layers;
  run.hist.kvd = c.kv_dim();
  run.hist.k.assign(std::size_t{N} * c.num_layers * c.kv_dim(), 0);
  run.hist.v = run.hist.k;
  kv.Reset(start);
  float* df = ToDevice(features);
  std::uint32_t pos = start;
  for (std::size_t i = 0; i < plan.chunks.size(); ++i) {
    const std::uint32_t rows = plan.chunks[i];
    const float* chunk = df + std::size_t{pos - start} * width;
    std::string error;
    if (tapped) {
      Taps taps;
      bool ok = false;
      {
        TapScope scope(ctx.ops(), &taps);
        ok = ctx.ops().Inject(w, kv, chunk, rows, pos, plan.paths[i], &error);
        CheckHip(hipStreamSynchronize(ctx.stream()), "sync");
      }
      Require(ok, label + ": Inject failed: " + error);
      Require(taps.capture_error.empty() && taps.pending.empty(),
              label + ": activation capture failed");
      const KvRows ring = ReadRing(kv, pos, pos + rows);
      if (check) {
        const std::vector<float> f(
            features.begin() + std::size_t{pos - start} * width,
            features.begin() + std::size_t{pos - start + rows} * width);
        CheckInjectChunk(label, fx, f, rows, pos,
                         UsesVector(plan.paths[i], rows), taps, ring, &run.rep);
      }
      for (std::uint32_t p = pos; p < pos + rows; ++p) {
        for (std::size_t l = 0; l < c.num_layers; ++l) {
          std::copy_n(
              ring.K(p, l), ring.kvd,
              run.hist.k.data() + ((p - start) * c.num_layers + l) * ring.kvd);
          std::copy_n(
              ring.V(p, l), ring.kvd,
              run.hist.v.data() + ((p - start) * c.num_layers + l) * ring.kvd);
        }
      }
    } else {
      Require(ctx.ops().Inject(w, kv, chunk, rows, pos, plan.paths[i], &error),
              label + ": Inject failed: " + error);
    }
    pos += rows;
  }
  CheckHip(hipStreamSynchronize(ctx.stream()), "sync");
  (void)hipFree(df);
  Require(kv.valid() && kv.committed() == start + N,
          label + ": ring bookkeeping after the plan");
  run.final = ReadRing(kv, kv.retained_begin(), kv.committed());
  return run;
}

/// The rings after a plan hold exactly what each chunk stored, for every
/// retained position (later chunks did not clobber them through the wrap).
void CheckRetained(const std::string& label, const InjectPlanRun& run) {
  for (std::uint32_t p = run.final.begin; p < run.final.end; ++p) {
    for (std::size_t l = 0; l < run.final.layers; ++l) {
      Require(std::memcmp(run.final.K(p, l), run.hist.K(p, l),
                          run.final.kvd * 2) == 0 &&
                  std::memcmp(run.final.V(p, l), run.hist.V(p, l),
                              run.final.kvd * 2) == 0,
              label + ": ring slot of position " + std::to_string(p) +
                  " was overwritten");
    }
  }
}

bool SameRows(const KvRows& a, const KvRows& b) {
  return a.begin == b.begin && a.end == b.end && a.k == b.k && a.v == b.v;
}

void InjectCase(TinyDrafter& fx, DrafterCtx& ctx, const Plan& plan,
                std::uint32_t max_rows, const std::string& what) {
  const ml::DFlashConfig& c = fx.config();
  const std::uint32_t N = plan.total();
  const auto features =
      RandomFeatures(N, c.tap_count(), c.hidden_size, fx.rng());
  const std::string label = std::string(fx.shape().name) + " inject " + what;
  std::string error;
  auto kv = gpu::DrafterKv::Create(ctx.weights(), max_rows, &error);
  Require(kv.has_value(), "kv: " + error);
  InjectPlanRun run = RunInjectPlan(label, fx, ctx, ctx.weights(), *kv,
                                    features, plan, 0, true);
  const std::uint32_t cap = kv->capacity();
  Require(cap == c.sliding_window + std::max(max_rows, c.block_size),
          label + ": ring capacity");
  Require(kv->retained_begin() == (N > cap ? N - cap : 0U),
          label + ": retained_begin " + std::to_string(kv->retained_begin()));
  CheckRetained(label, run);
  const bool vector = UsesVector(plan.paths[0], plan.chunks[0]);
  CheckInjectEndToEnd(label, fx, features, N, vector, run.hist, &run.rep);
  PrintInject(run.rep);
  std::cout << "    ring capacity " << cap << ", retained ["
            << kv->retained_begin() << ", " << kv->committed() << ")"
            << (N > cap ? ", wrapped" : "") << "\n";
  // Untapped: same rings, same bookkeeping.
  auto kv2 = gpu::DrafterKv::Create(ctx.weights(), max_rows, &error);
  Require(kv2.has_value(), "kv: " + error);
  const InjectPlanRun plain = RunInjectPlan(label, fx, ctx, ctx.weights(), *kv2,
                                            features, plan, 0, false);
  Require(SameRows(plain.final, run.final) &&
              kv2->retained_begin() == kv->retained_begin(),
          label + ": untapped rings differ from the tapped rings");
  ++g_cases;
}

std::vector<std::uint32_t> Repeat(std::uint32_t rows, std::uint32_t total) {
  std::vector<std::uint32_t> chunks;
  while (total > 0) {
    chunks.push_back(std::min(rows, total));
    total -= chunks.back();
  }
  return chunks;
}

void TestInject() {
  std::cout << "kv injection\n";
  {
    TinyDrafter fx(kDraftGeom, 0x1A7EC7);
    DrafterCtx ctx(fx.drafter);
    // Window 512: capacity 544 with 32-row chunks, so 600 rows cover
    // positions 511/512/513 and wrap the ring.
    InjectCase(fx, ctx, UniformPlan(Repeat(32, 600), gpu::Path::kVector), 32,
               "600 rows, 32-row vector chunks");
    InjectCase(fx, ctx,
               UniformPlan({129, 200, 200, 200, 71}, gpu::Path::kBatched), 200,
               "800 rows, batched chunks up to 200");
    InjectCase(fx, ctx, UniformPlan(Repeat(5, 40), gpu::Path::kVector), 5,
               "40 rows, 5-row chunks (capacity 512 + block 16)");
  }
  {
    TinyDrafter fx(kDraftOdd, 0x0DD1A7);
    DrafterCtx ctx(fx.drafter);
    InjectCase(fx, ctx, UniformPlan(Repeat(16, 100), gpu::Path::kVector), 16,
               "100 rows, 16-row vector chunks");
    InjectCase(fx, ctx, UniformPlan({20, 33, 47}, gpu::Path::kBatched), 47,
               "100 rows, batched chunks");
  }
}

/// Overwriting rows (as trimming rejected rows does) equals injecting the
/// final sequence directly, bit for bit, and leaves earlier rows untouched.
void TestInjectOverwrite() {
  std::cout << "kv injection overwrite\n";
  TinyDrafter fx(kDraftGeom, 0x0E7E);
  DrafterCtx ctx(fx.drafter);
  const ml::DFlashConfig& c = fx.config();
  const std::size_t width = c.fused_input_width();
  const auto a = RandomFeatures(40, c.tap_count(), c.hidden_size, fx.rng());
  const auto b = RandomFeatures(15, c.tap_count(), c.hidden_size, fx.rng());
  for (const gpu::Path path : {gpu::Path::kVector, gpu::Path::kBatched}) {
    std::string error;
    auto kv = gpu::DrafterKv::Create(ctx.weights(), 16, &error);
    auto ref_kv = gpu::DrafterKv::Create(ctx.weights(), 16, &error);
    Require(kv.has_value() && ref_kv.has_value(), "kv: " + error);
    (void)RunInjectPlan("overwrite", fx, ctx, ctx.weights(), *kv, a,
                        UniformPlan({16, 16, 8}, path), 0, false);
    // Rows 30..44 replaced: inject b at 30 (below committed 40).
    float* db = ToDevice(b);
    Require(ctx.ops().Inject(ctx.weights(), *kv, db, 15, 30, path, &error),
            "overwrite inject: " + error);
    CheckHip(hipStreamSynchronize(ctx.stream()), "sync");
    (void)hipFree(db);
    Require(kv->committed() == 45, "overwrite: committed");
    std::vector<float> combined(a.begin(), a.begin() + 30 * width);
    combined.insert(combined.end(), b.begin(), b.end());
    (void)RunInjectPlan("overwrite reference", fx, ctx, ctx.weights(), *ref_kv,
                        combined, UniformPlan({16, 14, 15}, path), 0, false);
    Require(SameRows(ReadRing(*kv, 0, 45), ReadRing(*ref_kv, 0, 45)),
            std::string("overwrite differs from direct injection, ") +
                PathName(path));
    std::cout << "  " << PathName(path)
              << ": rows 30..44 rewritten equal a direct injection\n";
    ++g_cases;
  }
}

/// Mutations of the device view or the position must fail the injection
/// checks; bad arguments are refused before anything is written; a failure
/// after a store invalidates the ring until Reset.
void TestInjectNegative() {
  std::cout << "kv injection negative\n";
  TinyDrafter fx(kDraftGeom, 0x5EED1A);
  DrafterCtx ctx(fx.drafter);
  const ml::DFlashConfig& c = fx.config();
  const std::uint32_t T = 6;
  const std::uint32_t pos = 509;  // rows 509..514 straddle 511/512/513
  const auto features =
      RandomFeatures(T, c.tap_count(), c.hidden_size, fx.rng());
  const auto check = [&](const std::string& what, const gpu::DrafterWeights& w,
                         gpu::Path path, std::uint32_t gpu_pos) {
    std::string error;
    auto kv = gpu::DrafterKv::Create(ctx.weights(), 16, &error);
    Require(kv.has_value(), "kv: " + error);
    kv->Reset(gpu_pos);
    float* df = ToDevice(features);
    Taps taps;
    bool ok = false;
    {
      TapScope scope(ctx.ops(), &taps);
      ok = ctx.ops().Inject(w, *kv, df, T, gpu_pos, path, &error);
      CheckHip(hipStreamSynchronize(ctx.stream()), "sync");
    }
    (void)hipFree(df);
    Require(ok, what + ": Inject failed: " + error);
    KvRows ring = ReadRing(*kv, gpu_pos, gpu_pos + T);
    ring.end = pos + T;  // read as if the rows sat at `pos`
    ring.begin = pos;
    InjectReport rep;
    CheckInjectChunk(what, fx, features, T, pos, path == gpu::Path::kVector,
                     taps, ring, &rep);
  };
  for (const gpu::Path path : {gpu::Path::kVector, gpu::Path::kBatched}) {
    const std::string p = std::string(", ") + PathName(path);
    check("baseline" + p, ctx.weights(), path, pos);  // must pass
    gpu::DrafterWeights theta = ctx.weights();
    ref::RopeParams rp;
    rp.n_rot = 128;
    rp.freq_base = c.rope_theta * 0.5F;
    theta.rope.theta_scale = ref::MakeRopeSetup(rp).theta_scale;
    MustFail("rope theta halved" + p,
             [&] { check("theta", theta, path, pos); });
    gpu::DrafterWeights swapped = ctx.weights();
    std::swap(swapped.layers[0].k, swapped.layers[0].v);
    MustFail("K and V projections swapped" + p,
             [&] { check("swapped", swapped, path, pos); });
    MustFail("rows written one position late" + p,
             [&] { check("late", ctx.weights(), path, pos + 1); });
    ++g_cases;
  }

  // Refusals: nothing is written, the ring stays as it was.
  std::string error;
  auto kv = gpu::DrafterKv::Create(ctx.weights(), 16, &error);
  Require(kv.has_value(), "kv: " + error);
  float* df =
      ToDevice(RandomFeatures(40, c.tap_count(), c.hidden_size, fx.rng()));
  Require(ctx.ops().Inject(ctx.weights(), *kv, df, 8, 0, gpu::Path::kVector,
                           &error),
          "inject: " + error);
  CheckHip(hipStreamSynchronize(ctx.stream()), "sync");
  const KvRows before = ReadRing(*kv, 0, 8);
  std::uint32_t committed = 8;
  std::uint32_t retained = 0;
  const auto refuse = [&](const std::string& what, std::uint32_t rows,
                          std::uint32_t start, gpu::Path path,
                          const gpu::DrafterWeights& w) {
    std::string why;
    Require(!ctx.ops().Inject(w, *kv, df, rows, start, path, &why),
            "accepted: " + what);
    Require(kv->valid() && kv->committed() == committed &&
                kv->retained_begin() == retained,
            what + ": refusal changed the bookkeeping");
    std::cout << "  refused " << what << ": " << why << "\n";
  };
  refuse("a gap after committed", 4, 9, gpu::Path::kVector, ctx.weights());
  refuse("more rows than the ring takes", 17, 8, gpu::Path::kBatched,
         ctx.weights());
  refuse("33 rows on the vector path", 33, 8, gpu::Path::kVector,
         ctx.weights());
  {
    gpu::DrafterWeights other = ctx.weights();
    other.window = 511;
    refuse("a ring of another window", 4, 8, gpu::Path::kVector, other);
  }
  kv->Reset(100);
  committed = retained = 100;
  refuse("a start below floor", 4, 99, gpu::Path::kVector, ctx.weights());
  kv->Reset(0);
  Require(ctx.ops().Inject(ctx.weights(), *kv, df, 8, 0, gpu::Path::kVector,
                           &error),
          "inject: " + error);
  CheckHip(hipStreamSynchronize(ctx.stream()), "sync");
  Require(SameRows(ReadRing(*kv, 0, 8), before), "re-injection differs");
  // A failure before any store leaves the ring valid and unchanged.
  ctx.ops().SetFailAfterStage("fc_out");
  Require(!ctx.ops().Inject(ctx.weights(), *kv, df, 4, 8, gpu::Path::kVector,
                            &error),
          "injected failure did not fail");
  Require(kv->valid() && kv->committed() == 8,
          "failure before the store "
          "changed the ring");
  // A failure after layer 0's store invalidates it until Reset.
  ctx.ops().SetFailAfterStage("Vcur_injected-1");
  Require(!ctx.ops().Inject(ctx.weights(), *kv, df, 4, 8, gpu::Path::kVector,
                            &error),
          "injected failure did not fail");
  ctx.ops().SetFailAfterStage("");
  CheckHip(hipStreamSynchronize(ctx.stream()), "sync");
  Require(!kv->valid(), "a failure after a store left the ring valid");
  Require(!ctx.ops().Inject(ctx.weights(), *kv, df, 4, 0, gpu::Path::kVector,
                            &error),
          "an invalid ring accepted an injection");
  std::cout << "  invalidated ring refuses: " << error << "\n";
  kv->Reset(0);
  Require(ctx.ops().Inject(ctx.weights(), *kv, df, 8, 0, gpu::Path::kVector,
                           &error),
          "inject after reset: " + error);
  CheckHip(hipStreamSynchronize(ctx.stream()), "sync");
  Require(SameRows(ReadRing(*kv, 0, 8), before),
          "rows after Reset differ from the first injection");
  (void)hipFree(df);
  ++g_cases;
}

}  // namespace

int main() {
  try {
    TestEncoder();
    TestEncoderNegative();
    TestEncoderAllocationFailure();
    TestInject();
    TestInjectOverwrite();
    TestInjectNegative();
    std::cout << "PASS: laguna.dflash_ops (" << g_cases << " cases)\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << "\n";
    return 1;
  }
}
