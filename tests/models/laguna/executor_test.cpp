// Laguna GPU executor checks on tiny random models (no model file, nothing
// skipped). The models use the real geometry and storage formats (see
// executor_gpu.hpp), a 12-position window and six layers: full+dense, three
// sliding (routed Q6_K, Q5_K, IQ4_XS), full, sliding.
//
//  * oracle: the executor's logits match the CPU oracle in the activation and
//    KV mode of the path taken (kGpuQ8_1 for the vector path, kGpuMmq for the
//    batched path, F16-rounded K/V), over sequences that cross the window,
//    prefilled in chunks and decoded one token at a time. Per-layer hidden
//    rows are compared too, to localize any difference.
//  * prefill-then-decode == all-prefill, and different chunk splits, are
//    bit-identical on a pinned path.
//  * a cache Reset between sequences reproduces the first run exactly.
//  * a tapped run and untapped reruns are bit-identical.
//  * allocation failure: Create and Forward fail cleanly (no leaked or
//    double-freed allocation through the allocator seam), a failed Forward
//    poisons the executor until Reset, and calls rejected up front do not.
//  * the #969 logit dump (shared with laguna_logits): files, sizes, executed
//    tokens, manifest published last, nonfinite failure, determinism, and the
//    real scorer accepting a pair of dumps.
//  * the budget's weight and KV figures are exact and its scratch figure
//    covers what the warm-up pass measured.
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>

#include "tests/models/laguna/executor_gpu.hpp"
#include "tests/models/laguna/logits_dump.hpp"

using namespace gufo::testing::laguna;

namespace {

constexpr std::uint32_t kVocabCheckTokens = 40;
// One layer on the executor's own input rows: the operators' end-to-end bound
// (ffn_ops_test: 5e-3 of rms; observed 3e-3 or less).
constexpr double kLocalBound = 6e-3;

// ---- allocator seam mock
// -----------------------------------------------------

std::set<void*> g_live;
int g_bad_frees = 0;
long g_fail_after = -1;  // fail the allocation after this many more (-1: never)
long g_allocations = 0;

hipError_t MockAlloc(void** ptr, std::size_t bytes) {
  if (g_fail_after == 0) {
    return hipErrorOutOfMemory;
  }
  if (g_fail_after > 0) {
    --g_fail_after;
  }
  const hipError_t status = hipMalloc(ptr, bytes);
  if (status == hipSuccess) {
    g_live.insert(*ptr);
    ++g_allocations;
  }
  return status;
}
hipError_t MockFree(void* ptr) {
  if (g_live.erase(ptr) == 0) {
    ++g_bad_frees;
  }
  return hipFree(ptr);
}
exec::DeviceAllocator Mock() {
  return exec::DeviceAllocator{MockAlloc, MockFree};
}

// ---- oracle comparison
// -------------------------------------------------------

struct Worst {
  double rel_rms{0};
  double max_abs_over_rms{0};
  double layer_rel_rms{0};
};
Worst g_worst;

double Rms(const std::vector<float>& v) {
  double s = 0;
  for (float x : v) {
    s += static_cast<double>(x) * x;
  }
  return std::sqrt(s / static_cast<double>(std::max<std::size_t>(v.size(), 1)));
}

struct OracleResult {
  std::vector<float> logits;
};

OracleResult RunOracleModel(const TinyModel& m,
                            const std::vector<std::int32_t>& tokens,
                            ref::ActivationQuant mode) {
  ref::Options options;
  options.activation_quant = mode;
  options.round_kv_f16 = true;
  ref::Model model(m.weights(), options);
  return OracleResult{model.Forward(tokens)};
}

/// The oracle's RmsNormRow (double sum, float scale), applied to T rows.
std::vector<float> RmsNormRows(const std::vector<float>& x,
                               const ml::TensorRef& weight, std::size_t T,
                               std::size_t H, float eps) {
  const auto* w = static_cast<const float*>(weight.data);
  std::vector<float> y(T * H);
  for (std::size_t t = 0; t < T; ++t) {
    double sum = 0;
    for (std::size_t i = 0; i < H; ++i) {
      sum += static_cast<double>(x[t * H + i]) * x[t * H + i];
    }
    const float mean = static_cast<float>(sum / static_cast<double>(H));
    const float scale = 1.0F / std::sqrt(mean + eps);
    for (std::size_t i = 0; i < H; ++i) {
      y[t * H + i] = (x[t * H + i] * scale) * w[i];
    }
  }
  return y;
}

double RelRmsDiff(const std::vector<float>& a, const std::vector<float>& b) {
  Require(a.size() == b.size(), "RelRmsDiff size");
  double num = 0;
  double den = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = static_cast<double>(a[i]) - b[i];
    num += d * d;
    den += static_cast<double>(b[i]) * b[i];
  }
  return std::sqrt(num / den);
}

/// Local check: the oracle runs ONE layer (or the output head) on the
/// executor's own input rows, so Q8 rounding differences between GPU and
/// oracle do not compound through the depth (a deep random network amplifies
/// them to about the size of the Q8 noise itself, see CompareLogits). Each
/// layer must reproduce the executor's layer output to the operators' own
/// end-to-end bound; a wiring defect in any layer is O(1).
double LocalLayerCheck(const std::string& label, const TinyModel& m,
                       ref::ActivationQuant mode,
                       const std::vector<std::int32_t>& tokens,
                       const std::map<std::string, std::vector<float>>& taps,
                       const std::vector<float>& logits) {
  const ml::Config& c = m.config();
  const ml::ModelWeights& w = m.weights();
  const std::size_t T = tokens.size();
  const std::size_t H = c.hidden_size;
  const std::size_t V = c.vocab_size;
  ref::Options options;
  options.activation_quant = mode;
  options.round_kv_f16 = true;
  std::vector<float> prev(T * H);
  for (std::size_t t = 0; t < T; ++t) {
    ref::DequantizeRow(
        w.token_embd.type,
        static_cast<const std::uint8_t*>(w.token_embd.data) +
            static_cast<std::size_t>(tokens[t]) * w.token_embd.RowBytes(),
        prev.data() + t * H, H);
  }
  double worst = 0;
  std::string detail;
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    const ml::LayerWeights& lw = w.layers[il];
    const auto n = RmsNormRows(prev, lw.attn_norm, T, H, c.rms_eps);
    const auto o = ref::AttnForward(c, lw, il, n.data(), T, options);
    std::vector<float> x(T * H);
    for (std::size_t i = 0; i < x.size(); ++i) {
      x[i] = o[i] + prev[i];
    }
    const auto fn = RmsNormRows(x, lw.ffn_norm, T, H, c.rms_eps);
    const auto f = ref::FfnForward(c, lw, fn.data(), T, options);
    std::vector<float> out(T * H);
    for (std::size_t i = 0; i < out.size(); ++i) {
      out[i] = f[i] + x[i];
    }
    const auto& got = taps.at("l_out-" + std::to_string(il));
    const double err = RelRmsDiff(got, out);
    char text[48];
    std::snprintf(text, sizeof(text), " L%u %.1e", il, err);
    detail += text;
    worst = std::max(worst, err);
    prev = got;
  }
  // Output head on the executor's last hidden rows.
  const auto normed = RmsNormRows(prev, w.output_norm, T, H, c.rms_eps);
  std::vector<float> want(T * V);
  ref::MatMulRows(w.output, 0, normed.data(), T, want.data(), options);
  const double head = RelRmsDiff(logits, want);
  worst = std::max(worst, head);
  std::size_t flips = 0;
  for (std::size_t t = 0; t < T; ++t) {
    std::vector<float> row(want.begin() + t * V, want.begin() + (t + 1) * V);
    std::partial_sort(row.begin(), row.begin() + 2, row.end(),
                      std::greater<float>());
    const auto top = [&](const std::vector<float>& v) {
      return std::max_element(v.begin() + t * V, v.begin() + (t + 1) * V) -
             (v.begin() + t * V);
    };
    if (row[0] - row[1] > 1e-3 * Rms(want) && top(logits) != top(want)) {
      ++flips;
    }
  }
  std::printf("  %-22s local rel-rms:%s head %.1e\n", label.c_str(),
              detail.c_str(), head);
  Require(flips == 0, label + ": a clear top-1 differs from the local oracle");
  return worst;
}

void CompareLogits(const std::string& label, const TinyModel& m,
                   const std::vector<float>& got,
                   const std::vector<float>& want) {
  Require(got.size() == want.size(), label + ": size");
  for (float v : got) {
    Require(std::isfinite(v), label + ": nonfinite logit");
  }
  const double rel = RelRmsDiff(got, want);
  const double rms = Rms(want);
  double max_abs = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    max_abs =
        std::max(max_abs, std::fabs(static_cast<double>(got[i]) - want[i]));
  }
  (void)m;
  std::printf("  %-22s whole-model rel-rms %.2e, max|d|/rms %.2e\n",
              label.c_str(), rel, max_abs / rms);
  g_worst.rel_rms = std::max(g_worst.rel_rms, rel);
  g_worst.max_abs_over_rms = std::max(g_worst.max_abs_over_rms, max_abs / rms);
  // Whole-model agreement is limited by Q8 noise amplified through six random
  // layers (switching the oracle between the vector and tile activation modes
  // moves the final hidden rows by 5% rms). It only catches gross defects.
  Require(rel < 0.1, label + ": logits differ grossly from the oracle");
}

void CheckAgainstOracle(const TinyModel& m, exec::DeviceModel& dm,
                        const std::string& label,
                        const std::vector<std::int32_t>& tokens,
                        const std::vector<std::uint32_t>& chunks,
                        gpu::Path path, ref::ActivationQuant mode) {
  const std::uint32_t max_chunk =
      *std::max_element(chunks.begin(), chunks.end());
  auto ex = MakeExecutor(dm, static_cast<std::uint32_t>(tokens.size()),
                         max_chunk, path);
  const std::size_t V = m.config().vocab_size;
  std::map<std::string, std::vector<float>> taps;
  ex->SetLayerTap([&](std::uint32_t il, const float* h, std::size_t count) {
    const auto rows = ToHost(h, count);
    auto& dst = taps["l_out-" + std::to_string(il)];
    dst.insert(dst.end(), rows.begin(), rows.end());
  });
  const auto tapped = RunChunks(*ex, tokens, chunks, V);
  ex->SetLayerTap({});
  Require(ex->Reset(), "reset");
  const auto untapped = RunChunks(*ex, tokens, chunks, V);
  Require(BitIdentical(tapped, untapped),
          label + ": tapped and untapped runs differ");
  const double local = LocalLayerCheck(label, m, mode, tokens, taps, untapped);
  g_worst.layer_rel_rms = std::max(g_worst.layer_rel_rms, local);
  Require(local < kLocalBound,
          label + ": a layer or the head differs from the oracle");
  CompareLogits(label, m, untapped, RunOracleModel(m, tokens, mode).logits);
}

void TestOracle(const TinyModel& m, exec::DeviceModel& dm) {
  std::printf("oracle comparison (window %u):\n", m.spec().window);
  const auto t40 = TokensFor(m, kVocabCheckTokens, 11);
  // Batched path: chunks 16,16,8 cross the 12-position window and wrap the
  // ring (capacity window + max_chunk = 28).
  CheckAgainstOracle(m, dm, "batched 16+16+8", t40, {16, 16, 8},
                     gpu::Path::kBatched, ref::ActivationQuant::kGpuMmq);
  CheckAgainstOracle(m, dm, "batched 40", t40, {40}, gpu::Path::kBatched,
                     ref::ActivationQuant::kGpuMmq);
  // Vector path with 20-row chunks, and pure decode (auto path: one row).
  CheckAgainstOracle(m, dm, "vector 20+20", t40, {20, 20}, gpu::Path::kVector,
                     ref::ActivationQuant::kGpuQ8_1);
  CheckAgainstOracle(m, dm, "decode 1x40 (auto)", t40,
                     std::vector<std::uint32_t>(40, 1U), gpu::Path::kAuto,
                     ref::ActivationQuant::kGpuQ8_1);
  // Auto path prefill of 6 rows (vector) then 6 decode rows.
  const auto t12 = TokensFor(m, 12, 12);
  CheckAgainstOracle(m, dm, "auto 6 + 6x1", t12, {6, 1, 1, 1, 1, 1, 1},
                     gpu::Path::kAuto, ref::ActivationQuant::kGpuQ8_1);
}

// ---- bit-identity
// ------------------------------------------------------------

/// Bit-identical chunk splits on the batched path. Routed Q6_K and IQ4_XS
/// only: the batched Q5_K routed kernel is not split invariant (see
/// TestQ5KSplits).
void TestBatchedSplits(const TinyModel& m, exec::DeviceModel& dm) {
  std::printf("batched chunk splits (routed Q6_K and IQ4_XS):\n");
  const std::size_t V = m.config().vocab_size;
  const auto tokens = TokensFor(m, 40, 21);
  auto ex = MakeExecutor(dm, 40, 40, gpu::Path::kBatched);
  const auto whole = RunChunks(*ex, tokens, {40}, V);
  for (const auto& split : std::vector<std::vector<std::uint32_t>>{
           {16, 16, 8}, {7, 13, 20}, {1, 39}, {10, 10, 10, 10}, {39, 1}}) {
    Require(ex->Reset(), "reset");
    const auto got = RunChunks(*ex, tokens, split, V);
    Require(BitIdentical(whole, got), "batched chunk split differs");
  }
  std::printf("  5 splits identical to one 40-row chunk\n");
}

/// The batched routed Q5_K kernel gives results that depend on how many rows
/// share the call at the 1e-6 level (a different tile or summation order for
/// the same row), so splits are not bit-identical; require them to agree to
/// 1e-4 after the first routed layer and the head.
void TestQ5KSplits() {
  std::printf("batched chunk splits (routed Q5_K, not bit-identical):\n");
  ModelSpec spec;
  spec.layers = 2;
  spec.routed = {GgmlType::kQ5_K};
  const TinyModel m(spec);
  auto dm = m.Upload();
  const std::size_t V = m.config().vocab_size;
  const auto tokens = TokensFor(m, 40, 22);
  auto ex = MakeExecutor(*dm, 40, 40, gpu::Path::kBatched);
  const auto whole = RunChunks(*ex, tokens, {40}, V);
  double worst = 0;
  bool identical = true;
  for (const auto& split :
       std::vector<std::vector<std::uint32_t>>{{16, 16, 8}, {7, 13, 20}}) {
    Require(ex->Reset(), "reset");
    const auto got = RunChunks(*ex, tokens, split, V);
    identical = identical && BitIdentical(whole, got);
    worst = std::max(worst, RelRmsDiff(got, whole));
  }
  std::printf("  worst rel-rms vs one chunk %.2e (%s)\n", worst,
              identical ? "bit-identical" : "not bit-identical");
  Require(worst < 1e-4, "Q5_K batched splits differ by more than 1e-4");
}

void TestChunkSplits(const TinyModel& m, exec::DeviceModel& dm) {
  std::printf("prefill-then-decode (vector path):\n");
  const std::size_t V = m.config().vocab_size;
  const auto tokens = TokensFor(m, 40, 21);
  {
    auto ex = MakeExecutor(dm, 40, 20, gpu::Path::kVector);
    const auto all = RunChunks(*ex, tokens, {20, 20}, V);
    Require(ex->Reset(), "reset");
    std::vector<std::uint32_t> mixed = {12};
    mixed.insert(mixed.end(), 28, 1U);
    const auto got = RunChunks(*ex, tokens, mixed, V);
    Require(BitIdentical(all, got), "prefill-then-decode differs");
    Require(ex->Reset(), "reset");
    const auto singles =
        RunChunks(*ex, tokens, std::vector<std::uint32_t>(40, 1U), V);
    Require(BitIdentical(all, singles), "all-decode differs from prefill");
    Require(ex->Reset(), "reset");
    const auto odd = RunChunks(*ex, tokens, {3, 17, 9, 11}, V);
    Require(BitIdentical(all, odd), "vector chunk split differs");
    std::printf(
        "  prefill 12 then 28 decode steps == prefill 20+20 == 40 "
        "decode steps == 3+17+9+11 (all routed types)\n");
  }
  {
    // Under kAuto, last-row logits after a prefill equal that row of the
    // all-rows result (the same path for the output projection).
    auto ex = MakeExecutor(dm, 40, 40, gpu::Path::kAuto);
    const auto all = RunChunks(*ex, tokens, {40}, V);
    Require(ex->Reset(), "reset");
    std::vector<float> last(V);
    std::string error;
    Require(ex->Forward(tokens, exec::LogitsRows::kLast, last.data(), &error),
            error);
    Require(
        std::memcmp(last.data(), all.data() + 39 * V, V * sizeof(float)) == 0,
        "last-row logits differ from the all-row logits");
    std::printf("  auto: kLast row == kAll row 39\n");
  }
}

void TestReset(const TinyModel& m, exec::DeviceModel& dm) {
  std::printf("cache reset:\n");
  const std::size_t V = m.config().vocab_size;
  const auto a = TokensFor(m, 30, 31);
  const auto b = TokensFor(m, 30, 32);
  auto ex = MakeExecutor(dm, 30, 15, gpu::Path::kBatched);
  const auto first = RunChunks(*ex, a, {15, 15}, V);
  Require(ex->position() == 30, "position after sequence");
  Require(ex->Reset() && ex->position() == 0, "reset");
  const auto other = RunChunks(*ex, b, {15, 15}, V);
  Require(!BitIdentical(first, other), "different sequences gave equal logits");
  Require(ex->Reset(), "reset");
  const auto again = RunChunks(*ex, a, {15, 15}, V);
  Require(BitIdentical(first, again), "reset did not restore the first run");
  // A fresh executor agrees too.
  auto fresh = MakeExecutor(dm, 30, 15, gpu::Path::kBatched);
  Require(BitIdentical(first, RunChunks(*fresh, a, {15, 15}, V)),
          "fresh executor differs from a reset one");
  std::printf(
      "  sequence A, reset, B, reset, A: identical; fresh executor "
      "identical\n");
}

// ---- failure paths
// -------------------------------------------------------------

void TestRejectedCalls(const TinyModel& m, exec::DeviceModel& dm) {
  std::printf("calls rejected up front:\n");
  const std::size_t V = m.config().vocab_size;
  auto ex = MakeExecutor(dm, 8, 4, gpu::Path::kAuto);
  std::vector<float> logits(8 * V);
  std::string error;
  const auto ok_tokens = TokensFor(m, 4, 41);
  const std::int32_t bad[] = {1, static_cast<std::int32_t>(V)};
  Require(!ex->Forward(bad, exec::LogitsRows::kAll, logits.data(), &error) &&
              !ex->poisoned(),
          "an out-of-range token was accepted or poisoned");
  const auto five = TokensFor(m, 5, 42);
  Require(!ex->Forward(five, exec::LogitsRows::kAll, logits.data(), &error) &&
              !ex->poisoned(),
          "a chunk above max_chunk was accepted or poisoned");
  Require(ex->Forward(ok_tokens, exec::LogitsRows::kAll, logits.data()),
          "valid chunk");
  Require(ex->Forward(ok_tokens, exec::LogitsRows::kAll, logits.data()),
          "second valid chunk");
  Require(ex->position() == 8, "position");
  Require(
      !ex->Forward(ok_tokens, exec::LogitsRows::kAll, logits.data(), &error) &&
          !ex->poisoned() && ex->position() == 8,
      "exhausted context was accepted or poisoned");
  std::printf(
      "  bad token, oversize chunk, exhausted context: refused, not "
      "poisoned (%s)\n",
      error.c_str());
}

void TestAllocationFailure(const TinyModel& m, exec::DeviceModel& dm) {
  std::printf("allocation failure:\n");
  const std::size_t V = m.config().vocab_size;
  const auto tokens = TokensFor(m, 20, 51);
  // Reference result from an ordinary executor.
  auto base = MakeExecutor(dm, 20, 10, gpu::Path::kBatched);
  const auto want = RunChunks(*base, tokens, {10, 10}, V);

  // Create fails at every allocation it makes (executor buffers, then the
  // operators' scratch during the warm-up pass): clean nullptr, nothing
  // leaked, nothing freed twice.
  std::size_t failed_creates = 0;
  for (long k = 0;; ++k) {
    g_live.clear();
    g_bad_frees = 0;
    g_allocations = 0;
    g_fail_after = k;
    exec::ExecutorOptions options;
    options.context = 20;
    options.max_chunk = 10;
    options.path = gpu::Path::kBatched;
    options.allocator = Mock();
    std::string error;
    auto ex = exec::Executor::Create(dm, options, &error);
    g_fail_after = -1;
    if (ex != nullptr) {
      Require(g_allocations > 6, "warm-up made no operator allocations");
      Require(BitIdentical(want, RunChunks(*ex, tokens, {10, 10}, V)),
              "executor created after the failing range differs");
      ex.reset();
      Require(g_live.empty() && g_bad_frees == 0,
              "leak or double free after a successful create");
      break;
    }
    Require(!error.empty(), "failed Create gave no message");
    Require(g_live.empty() && g_bad_frees == 0,
            "failed Create leaked or double-freed (allocation " +
                std::to_string(k) + ")");
    ++failed_creates;
    Require(k < 200, "Create never succeeded");
  }
  std::printf(
      "  Create failing at each of %zu allocation points: clean, no "
      "leak\n",
      failed_creates);

  // Forward fails mid-chunk on a scratch growth: poisoned until Reset, and a
  // Reset executor reproduces the baseline exactly.
  g_live.clear();
  g_bad_frees = 0;
  {
    auto ex =
        MakeExecutor(dm, 20, 10, gpu::Path::kBatched, /*warmup=*/false, Mock());
    std::vector<float> logits(10 * V);
    std::string error;
    for (const long after : {0L, 3L, 9L}) {
      g_fail_after = after;
      const bool ok =
          ex->Forward(std::span<const std::int32_t>(tokens.data(), 10),
                      exec::LogitsRows::kAll, logits.data(), &error);
      g_fail_after = -1;
      Require(!ok && ex->poisoned() && ex->position() == 0,
              "a failing Forward did not poison the executor");
      Require(!error.empty(), "failing Forward gave no message");
      Require(!ex->Forward(std::span<const std::int32_t>(tokens.data(), 10),
                           exec::LogitsRows::kAll, logits.data(), &error) &&
                  error.find("Reset") != std::string::npos,
              "a poisoned executor accepted a call");
      Require(ex->Reset() && !ex->poisoned(), "Reset after failure");
    }
    Require(BitIdentical(want, RunChunks(*ex, tokens, {10, 10}, V)),
            "run after failure and Reset differs from the baseline");
  }
  Require(g_live.empty() && g_bad_frees == 0,
          "leak or double free after Forward failures");
  std::printf(
      "  Forward failing mid-chunk: poisoned, Reset recovers "
      "bit-identically, no leak\n");

  // A context that cannot fit is refused with the budget, touching nothing.
  exec::ExecutorOptions huge;
  huge.context = 0x7fffffffU;
  huge.max_chunk = 10;
  std::string error;
  auto none = exec::Executor::Create(dm, huge, &error);
  Require(
      none == nullptr && error.find("lower the context") != std::string::npos,
      "an impossible context was not refused with the budget: " + error);
  std::printf("  context 2^31-1: refused (%.60s...)\n", error.c_str());
}

// ---- logit dump ------------------------------------------------------------

std::vector<char> ReadBytes(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  Require(static_cast<bool>(in), "cannot read " + path.string());
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

void TestDump(const TinyModel& m, exec::DeviceModel& dm) {
  std::printf("logit dump:\n");
  const std::size_t V = m.config().vocab_size;
  constexpr std::size_t T = 24;
  constexpr std::size_t S = 3;
  const fs::path root = fs::temp_directory_path() /
                        ("laguna_executor_dump_" + std::to_string(::getpid()));
  fs::remove_all(root);
  fs::create_directories(root / "ref0");
  struct Cleanup {
    fs::path path;
    ~Cleanup() {
      std::error_code ec;
      fs::remove_all(path, ec);
    }
  } cleanup{root};
  // A reference directory with only the inputs (tokens and manifest).
  const auto tokens = TokensFor(m, S * T, 61);
  WriteAll(root / "ref0" / "tokens.i32", tokens.data(),
           tokens.size() * sizeof(std::int32_t));
  {
    std::string manifest =
        "{\"complete\":true,\"n_vocab\":" + std::to_string(V) +
        ",\"T\":24,\"seqs\":[";
    const char* names[] = {"00-chat.txt", "01-code.txt", "02-tool.txt"};
    for (std::size_t s = 0; s < S; ++s) {
      manifest += std::string(s ? "," : "") + "{\"file\":\"" + names[s] +
                  "\",\"n_tokens\":" + std::to_string(100 + s) +
                  ",\"tail\":" + (s == 2 ? "true" : "false") + "}";
    }
    manifest += "]}";
    WriteAll(root / "ref0" / "index.json", manifest.data(), manifest.size());
  }
  const Reference ref = ReadReference(root / "ref0");
  Require(ref.seqs.size() == S && ref.T == T && ref.n_vocab == V,
          "reference parse");

  const auto dump = [&](const std::string& name, std::uint32_t ubatch,
                        std::size_t sequences = 0,
                        std::function<void(float*, std::size_t)> hook = {}) {
    auto ex = MakeExecutor(dm, T, ubatch, gpu::Path::kAuto);
    DumpOptions o;
    o.output = root / name;
    o.reference = root / "ref0";
    o.model = "tiny";
    o.ubatch = ubatch;
    o.context = T;
    o.sequences = sequences;
    o.test_hook = std::move(hook);
    DumpLogits(ref, *ex, o);
  };
  dump("a", 24);
  dump("b", 8);
  dump("b2", 8);
  for (const char* name : {"a", "b"}) {
    const fs::path dir = root / name;
    Require(
        !fs::exists(dir / ".incomplete") && !fs::exists(dir / "index.json.tmp"),
        "marker or temp file left behind");
    Require(ReadBytes(dir / "tokens.i32").size() == S * T * 4 &&
                std::memcmp(ReadBytes(dir / "tokens.i32").data(), tokens.data(),
                            S * T * 4) == 0,
            "executed tokens differ from the input");
    for (std::size_t s = 0; s < S; ++s) {
      char file[16];
      std::snprintf(file, sizeof(file), "%02zu.f32", s);
      Require(fs::file_size(dir / file) == T * V * sizeof(float),
              "logit file size");
    }
    const auto index = gufo::json::parse(ReadText(dir / "index.json"));
    Require(index.find("complete") != nullptr &&
                index.find("complete")->as_bool() &&
                index.member_size("T") == T &&
                index.member_size("n_vocab") == V &&
                index.find("seqs")->items().size() == S,
            "manifest fields");
    const auto& seq = index.find("seqs")->items()[2];
    Require(seq.member_str("file") == "02-tool.txt" &&
                seq.member_size("n_tokens") == 102 &&
                seq.find("tail")->as_bool() &&
                seq.member_str("logits") == "02.f32",
            "manifest sequence fields");
  }
  Require(ReadBytes(root / "b" / "01.f32") == ReadBytes(root / "b2" / "01.f32"),
          "two dumps of one sequence differ");
  // The dump equals chunked Forward calls, from a cleared cache each time.
  {
    auto ex = MakeExecutor(dm, T, 8, gpu::Path::kAuto);
    const std::vector<std::int32_t> seq1(tokens.begin() + T,
                                         tokens.begin() + 2 * T);
    const auto direct = RunChunks(*ex, seq1, {8, 8, 8}, V);
    const auto bytes = ReadBytes(root / "b" / "01.f32");
    Require(bytes.size() == direct.size() * 4 &&
                std::memcmp(bytes.data(), direct.data(), bytes.size()) == 0,
            "dumped logits differ from Forward");
  }
  // A new directory is required; --limit style partial runs list only what ran.
  bool threw = false;
  try {
    dump("b", 8);
  } catch (const std::exception&) {
    threw = true;
  }
  Require(threw, "a populated output directory was accepted");
  dump("pilot", 8, 2);
  Require(gufo::json::parse(ReadText(root / "pilot" / "index.json"))
                  .find("seqs")
                  ->items()
                  .size() == 2,
          "pilot manifest");
  // Nonfinite logits fail the run and leave no manifest.
  threw = false;
  try {
    dump("nan", 8, 0, [](float* logits, std::size_t) { logits[5] = NAN; });
  } catch (const std::exception& e) {
    threw = std::string(e.what()).find("nonfinite") != std::string::npos;
  }
  Require(threw && fs::exists(root / "nan" / ".incomplete") &&
              !fs::exists(root / "nan" / "index.json"),
          "nonfinite logits did not fail the dump cleanly");
  std::printf(
      "  files, sizes, executed tokens, manifest-last, determinism, "
      "populated-dir and nonfinite refusals: ok\n");

  // The real scorer accepts the pair (needs python3 with numpy).
  const char* scorer = "/server/staging/laguna-codexms-reference/score.py";
  if (fs::exists(scorer) &&
      std::system("python3 -I -c 'import numpy' >/dev/null 2>&1") == 0) {
    const std::string command =
        std::string("python3 -I ") + scorer + " " + (root / "a").string() +
        " gufo=" + (root / "b").string() + " --output " +
        (root / "score.json").string() + " >/dev/null 2>&1";
    Require(std::system(command.c_str()) == 0, "score.py rejected the dumps");
    Require(fs::exists(root / "score.json"), "score.py wrote no result");
    std::printf("  score.py accepted the dump pair\n");
  } else {
    std::printf("  score.py or numpy unavailable: scorer check not run\n");
  }
}

// ---- budget
// ------------------------------------------------------------------

void TestBudget(const TinyModel& m, exec::DeviceModel& dm) {
  std::printf("budget:\n");
  const std::uint32_t context = 64;
  const std::uint32_t chunk = 24;
  const auto budget = ml::ComputeBudget(m.weights(), {context, chunk, true});
  Require(dm.resident_bytes() == budget.weights,
          "weight budget " + std::to_string(budget.weights) + " != uploaded " +
              std::to_string(dm.resident_bytes()));
  std::size_t before = 0;
  std::size_t total = 0;
  CheckHip(hipMemGetInfo(&before, &total), "meminfo");
  auto ex = MakeExecutor(dm, context, chunk, gpu::Path::kAuto);
  Require(ex->kv_bytes() == budget.kv(),
          "kv budget " + std::to_string(budget.kv()) + " != allocated " +
              std::to_string(ex->kv_bytes()));
  const std::size_t measured = ex->measured_warmup_bytes();
  std::printf(
      "  weights %zu B and kv %zu B exact; scratch estimate %zu B, "
      "measured over the warm-up pass %zu B\n",
      budget.weights, budget.kv(), budget.scratch(), measured);
  // Other processes allocating during the pass add noise; allow 64 MiB.
  Require(measured <= budget.scratch() + (std::size_t{64} << 20),
          "the warm-up pass allocated more than the scratch estimate");
  Require(ex->scratch_bytes() <= budget.scratch(),
          "executor and ffn scratch exceed the estimate");
  std::printf("%s", budget.Format().c_str());
}

}  // namespace

int main() {
  try {
    const TinyModel model;
    auto dm = model.Upload();
    TestOracle(model, *dm);
    TestChunkSplits(model, *dm);
    {
      ModelSpec inv;
      inv.routed = {GgmlType::kQ6_K, GgmlType::kIQ4_XS};
      const TinyModel inv_model(inv);
      auto inv_dm = inv_model.Upload();
      TestBatchedSplits(inv_model, *inv_dm);
    }
    TestQ5KSplits();
    TestReset(model, *dm);
    TestRejectedCalls(model, *dm);
    TestAllocationFailure(model, *dm);
    TestDump(model, *dm);
    TestBudget(model, *dm);
    std::printf(
        "MAX_ORACLE_ERROR local layer/head rel-rms %.3e (bound %.0e); "
        "whole-model logits rel-rms %.3e, max|d|/rms %.3e\n",
        g_worst.layer_rel_rms, kLocalBound, g_worst.rel_rms,
        g_worst.max_abs_over_rms);
    std::cout << "executor tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << "\n";
    return 1;
  }
}
