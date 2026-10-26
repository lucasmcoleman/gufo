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

}  // namespace

int main() {
  try {
    TestEncoder();
    TestEncoderNegative();
    TestEncoderAllocationFailure();
    std::cout << "PASS: laguna.dflash_ops (" << g_cases << " cases)\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << "\n";
    return 1;
  }
}
