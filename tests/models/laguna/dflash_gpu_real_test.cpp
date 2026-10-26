// The composed GPU drafter forward (phase 6c) on the real Laguna-S-2.1 DFlash
// drafter (laguna-s-2.1-DFlash-Q4_K.gguf, 0.61 GiB on the device) and the
// target's head (Q6_K, 0.24 GiB) and token_embd (read from the mapping; the
// 68.9 GiB target is never loaded), against the drafter's CPU oracle
// (reference::DFlashModel). No target forward runs anywhere: the context
// features are the CPU oracle's recorded l_out-1, 10, 19, 29, 38, 47 taps
// of the 9-token prompt in /server/staging/laguna/oracle-20261025-capital-cpuq8
// (GUFO_LAGUNA_DFLASH_TAPS overrides), and the anchor is that run's top-1.
//
// Cases, each on the vector and the batched path:
//  * fixture: 9 context rows, block of 16 (anchor + 15 MASK);
//  * long: 600 context rows built from the fixture rows (cycled, with a few
//    percent of noise), so the 512 window and the ring wrap are in play on
//    real weights.
// Each runs Block tapped through DFlashOps and checks every stage as
// laguna.dflash_ops does (projections against captured activations, norms,
// rope, stored K/V, attention against a double oracle, residuals bitwise,
// head, argmax), then end to end against the oracle in the same activation
// mode: layer outputs and logits (with the F32 oracle's distance as the
// quantization noise and the oracle's own Q8_1-vs-MMQ distance), and the
// 15 drafts' top-1 agreement with every disagreement explained as a near tie.
// The same block through DrafterSession (host features in, drafts out,
// untapped) must reproduce the tapped logits bit for bit, and blocks of 4
// and 8 must equal the prefix of 16.
//
// Skips (status 77, nothing verified) without the drafter, the target or the
// tap fixture. Touches the GPU: run it under the GPU admission lock.
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "tests/models/laguna/dflash_gpu.hpp"

using namespace gufo::testing::laguna;
namespace core = gufo::core;

namespace {

std::string DrafterPath() {
  if (const char* env = std::getenv("GUFO_LAGUNA_DFLASH_GGUF");
      env != nullptr && *env != '\0') {
    return env;
  }
  return "/server/ai/models/Myric/Laguna-S-2.1-APEX-GGUF/"
         "laguna-s-2.1-DFlash-Q4_K.gguf";
}

std::filesystem::path TapsDir() {
  if (const char* env = std::getenv("GUFO_LAGUNA_DFLASH_TAPS");
      env != nullptr && *env != '\0') {
    return env;
  }
  return "/server/staging/laguna/oracle-20261025-capital-cpuq8";
}

int Skip(const std::string& why) {
  std::cout << "SKIP: laguna.dflash_gpu_real " << why
            << "; nothing was verified\n";
  return kSkipExitCode;
}

std::vector<float> ReadF32(const std::filesystem::path& path,
                           std::size_t count) {
  std::ifstream in(path, std::ios::binary);
  Require(static_cast<bool>(in), "cannot read " + path.string());
  std::vector<float> v(count);
  in.read(reinterpret_cast<char*>(v.data()),
          static_cast<std::streamsize>(count * sizeof(float)));
  Require(in.gcount() == static_cast<std::streamsize>(count * sizeof(float)),
          path.string() + " is shorter than expected");
  return v;
}

double Seconds(std::chrono::steady_clock::time_point since) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - since)
      .count();
}

struct Oracles {
  OracleTaps q8_1;
  OracleTaps mmq;
  OracleTaps f32;
};

Oracles RunOracles(const DraftModel& fx, const std::vector<float>& features,
                   std::uint32_t n, const std::vector<std::int32_t>& block) {
  const auto t0 = std::chrono::steady_clock::now();
  const auto pos = Iota(0, n);
  Oracles o;
  o.q8_1 =
      RunDrafterOracle(fx, features, pos, block, static_cast<std::int32_t>(n),
                       ref::ActivationQuant::kGpuQ8_1);
  o.mmq =
      RunDrafterOracle(fx, features, pos, block, static_cast<std::int32_t>(n),
                       ref::ActivationQuant::kGpuMmq);
  o.f32 =
      RunDrafterOracle(fx, features, pos, block, static_cast<std::int32_t>(n),
                       ref::ActivationQuant::kNone);
  std::cout << "  CPU oracle (Q8_1, MMQ, F32 modes), " << n
            << " context rows: " << Seconds(t0) << " s\n"
            << std::flush;
  return o;
}

void PrintDrafts(const char* what, const std::vector<std::int32_t>& d) {
  std::cout << "    " << what << ":";
  for (const auto t : d) {
    std::cout << " " << t;
  }
  std::cout << "\n";
}

/// One case on one path: inject `features` through a DrafterSession in
/// chunks of `chunk`, then the tapped Block via DFlashOps (stage and end to
/// end checks), the session's untapped Propose (bitwise equal), and shorter
/// blocks (prefix bitwise equal).
void RunCase(const std::string& name, const DraftModel& fx, DrafterCtx& ctx,
             const DeviceHead& head, const std::vector<float>& features,
             std::uint32_t n, std::int32_t anchor, std::uint32_t chunk,
             gpu::Path path, const Oracles& oracles) {
  const ml::DFlashConfig& c = fx.config();
  const bool vector = path == gpu::Path::kVector;
  const std::size_t width = c.fused_input_width();
  const std::size_t V = fx.target.output.rows;
  const std::uint32_t B = c.block_size;
  const auto block =
      BlockTokens(anchor, B, static_cast<std::int32_t>(c.mask_token_id));
  std::string error;
  hipStream_t stream{};
  CheckHip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "stream");
  auto session = gpu::DrafterSession::Create(
      ctx.weights(), head.matrix(), fx.target.token_embd,
      static_cast<std::int32_t>(c.mask_token_id), chunk, stream, &error);
  Require(session != nullptr, "session: " + error);
  const auto t0 = std::chrono::steady_clock::now();
  for (std::uint32_t p = 0; p < n; p += chunk) {
    const std::uint32_t rows = std::min(chunk, n - p);
    Require(session->Inject(std::span<const float>(features).subspan(
                                std::size_t{p} * width, rows * width),
                            p, path, &error),
            "inject: " + error);
  }
  std::vector<std::int32_t> drafts;
  std::vector<float> logits;
  Require(session->Propose(anchor, B, &drafts, &logits, path, &error),
          "propose: " + error);
  const double gpu_s = Seconds(t0);
  // The same block, tapped, on the test's own DFlashOps over the session's
  // ring (the session's stream is idle after Propose).
  const BlockRun run =
      RunBlock(ctx, ctx.weights(), head.matrix(), session->kv(),
               Embeddings(fx.target, block), B, path, true);
  const std::string label =
      "real " + name + " n=" + std::to_string(n) + " " + PathName(path);
  const OracleTaps& matched = vector ? oracles.q8_1 : oracles.mmq;
  const OracleTaps& other = vector ? oracles.mmq : oracles.q8_1;
  BlockSpec spec{0, 0, n, B, vector};
  const auto t1 = std::chrono::steady_clock::now();
  const BlockReport rep =
      CheckBlock(label, fx, spec, Embeddings(fx.target, block), run.taps,
                 run.ring, run.logits, run.drafts, matched, oracles.f32, other);
  PrintBlock(rep);
  std::cout << "    stage checks " << Seconds(t1) << " s; session inject + "
            << "propose " << gpu_s << " s (first call, includes setup)\n";
  Require(BitIdentical(logits, run.logits) && drafts == run.drafts,
          label +
              ": the session's untapped Propose differs from the tapped "
              "Block");
  PrintDrafts("gpu drafts   ", drafts);
  std::vector<std::int32_t> oracle_drafts;
  for (const auto t : ref::RowArgmax(matched.logits, V)) {
    oracle_drafts.push_back(static_cast<std::int32_t>(t));
  }
  PrintDrafts("oracle drafts", oracle_drafts);
  // Top-1 against the F32 oracle and the other mode, for scale (not
  // asserted: those are other quantizations).
  for (const auto& [what, o] : {std::pair{"F32 oracle", &oracles.f32},
                                std::pair{"other-mode oracle", &other}}) {
    const auto best = ref::RowArgmax(o->logits, V);
    std::size_t agree = 0;
    for (std::size_t r = 0; r < best.size(); ++r) {
      agree += static_cast<std::int32_t>(best[r]) == drafts[r] ? 1 : 0;
    }
    std::cout << "    top-1 vs the " << what << ": " << agree << "/"
              << best.size() << " (logits " << RelRms(logits, o->logits)
              << " rms)\n";
  }
  // Shorter blocks equal the prefix.
  for (const std::uint32_t sb : {4U, 8U}) {
    std::vector<std::int32_t> d;
    std::vector<float> l;
    Require(session->Propose(anchor, sb, &d, &l, path, &error),
            "propose: " + error);
    Require(
        std::memcmp(l.data(), logits.data(), l.size() * sizeof(float)) == 0 &&
            std::equal(d.begin(), d.end(), drafts.begin()),
        label + ": a block of " + std::to_string(sb) +
            " differs from the prefix of 16");
  }
  std::cout << "    blocks of 4 and 8 equal the prefix of 16 bit for bit\n";
  session.reset();
  qfn_mmq_release_stream(stream);
  (void)hipStreamDestroy(stream);
}

}  // namespace

int main() {
  try {
    std::error_code ec;
    const auto taps_dir = TapsDir();
    if (!std::filesystem::exists(ModelPath(), ec) ||
        !std::filesystem::exists(DrafterPath(), ec) ||
        !std::filesystem::exists(taps_dir / "l_out-1.f32", ec)) {
      return Skip(
          "needs the target and drafter GGUF files and the tap "
          "fixture " +
          taps_dir.string());
    }
    std::string error;
    const auto target_reader = core::GgufReader::OpenFile(ModelPath(), &error);
    Require(target_reader != nullptr, "target does not open: " + error);
    const auto drafter_reader =
        core::GgufReader::OpenFile(DrafterPath(), &error);
    Require(drafter_reader != nullptr, "drafter does not open: " + error);
    const auto target = ml::ModelWeights::Bind(*target_reader, &error);
    Require(target.has_value(), "target bind: " + error);
    const auto drafter = ml::DFlashWeights::Bind(*drafter_reader, &error);
    Require(drafter.has_value(), "drafter bind: " + error);
    const ml::DFlashConfig& c = drafter->config;
    Require(c.ValidateAgainstTarget(target->config, &error),
            "pairing: " + error);
    const DraftModel fx{*drafter, *target};
    const std::size_t H = c.hidden_size;
    const std::size_t taps = c.tap_count();
    const std::size_t V = target->output.rows;

    // Fixture: features [9][6 * 3072] from l_out-<tap_layer(i)>, the anchor
    // is the recorded run's top-1 after the prompt.
    constexpr std::uint32_t kRows = 9;
    std::vector<float> features(kRows * taps * H);
    for (std::size_t k = 0; k < taps; ++k) {
      const auto tap = ReadF32(
          taps_dir / ("l_out-" + std::to_string(c.tap_layer(k)) + ".f32"),
          kRows * H);
      for (std::uint32_t t = 0; t < kRows; ++t) {
        std::copy_n(tap.data() + t * H, H,
                    features.data() + (t * taps + k) * H);
      }
    }
    const auto last = ReadF32(taps_dir / "logits.f32", V);
    const auto anchor = static_cast<std::int32_t>(ref::RowArgmax(last, V)[0]);
    std::cout << "drafter " << DrafterPath() << "\ntaps " << taps_dir.string()
              << " (l_out-1, 10, 19, 29, 38, 47, 9 rows), anchor " << anchor
              << ", block " << c.block_size << "\n"
              << std::flush;

    // Long context: the fixture rows cycled with 3% noise, 600 rows.
    constexpr std::uint32_t kLong = 600;
    std::vector<float> long_features(kLong * taps * H);
    {
      std::mt19937_64 rng(0x10C);
      std::normal_distribution<float> noise(0.0F, 1.0F);
      for (std::uint32_t t = 0; t < kLong; ++t) {
        const std::uint32_t src = t % kRows;
        for (std::size_t k = 0; k < taps; ++k) {
          const float* row = features.data() + (src * taps + k) * H;
          double sq = 0;
          for (std::size_t i = 0; i < H; ++i) {
            sq += static_cast<double>(row[i]) * row[i];
          }
          const auto rms = static_cast<float>(std::sqrt(sq / H));
          float* dst = long_features.data() + (t * taps + k) * H;
          for (std::size_t i = 0; i < H; ++i) {
            dst[i] = row[i] + 0.03F * rms * noise(rng);
          }
        }
      }
    }

    // CPU oracles first (no GPU work runs meanwhile).
    const auto block = BlockTokens(anchor, c.block_size,
                                   static_cast<std::int32_t>(c.mask_token_id));
    std::cout << "fixture oracles\n";
    const Oracles fixture_oracles = RunOracles(fx, features, kRows, block);
    std::cout << "long-context oracles\n";
    const Oracles long_oracles = RunOracles(fx, long_features, kLong, block);

    const auto t0 = std::chrono::steady_clock::now();
    DrafterCtx ctx(*drafter);
    DeviceHead head(target->output);
    std::cout << "upload (drafter + head): " << Seconds(t0) << " s\n";
    for (const gpu::Path path : {gpu::Path::kVector, gpu::Path::kBatched}) {
      RunCase("fixture", fx, ctx, head, features, kRows, anchor, 32, path,
              fixture_oracles);
      RunCase("long", fx, ctx, head, long_features, kLong, anchor,
              path == gpu::Path::kVector ? 32U : 128U, path, long_oracles);
    }
    std::cout << "PASS: laguna.dflash_gpu_real\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << "\n";
    return 1;
  }
}
