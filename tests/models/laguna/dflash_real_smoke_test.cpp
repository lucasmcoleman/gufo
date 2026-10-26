// Opt-in smoke test: the real Laguna-S-2.1 DFlash drafter (Q4_K, 0.61 GiB) on
// the CPU oracle, fed by the target CPU oracle's taps for an 8-token prompt.
//
// Opt-in because it maps the 74 GB target (CPU/mmap, ~13 s for 9 tokens in an
// earlier run): set GUFO_LAGUNA_DFLASH_REAL=1. Without it, or with
// MemAvailable under 90 GiB, or without both files, it ends with the CTest
// skip status (77) and verifies nothing. No GPU is touched.
//
// What it checks and prints:
//  - the drafter binds, pairs with the target, and its logits are finite;
//  - the drafter's top-1 proposals for the 15 masked positions;
//  - target greedy verification: one teacher-forced target forward over
//    [prompt, anchor, drafts...] gives the target's argmax at every draft
//    position, so the leading-match count is the acceptance length a greedy
//    speculative step would see, and draft 1 vs the target's token after the
//    anchor is the "does the first proposal match" answer.
// Neither number is asserted beyond sanity: the oracle has not been compared
// with Poolside's fork yet, so a low acceptance is a finding, not a failure.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <optional>
#include <string>
#include <vector>

#include "src/models/laguna/dflash_reference.hpp"
#include "src/models/laguna/tokenizer.hpp"
#include "tests/models/laguna/support.hpp"

namespace laguna = gufo::models::laguna;
namespace rf = gufo::models::laguna::reference;
namespace core = gufo::core;
using gufo::testing::laguna::Require;

namespace {

constexpr std::uint64_t kMinAvailableKiB = 90ULL * 1024 * 1024;

std::uint64_t MemAvailableKiB() {
  std::ifstream meminfo("/proc/meminfo");
  std::string key;
  std::uint64_t value = 0;
  std::string unit;
  while (meminfo >> key >> value >> unit) {
    if (key == "MemAvailable:") {
      return value;
    }
  }
  return 0;
}

std::string DrafterPath() {
  if (const char* env = std::getenv("GUFO_LAGUNA_DFLASH_GGUF");
      env != nullptr && *env != '\0') {
    return env;
  }
  return "/server/ai/models/Myric/Laguna-S-2.1-APEX-GGUF/"
         "laguna-s-2.1-DFlash-Q4_K.gguf";
}

int Skip(const std::string& why) {
  std::cout << "SKIP: laguna.dflash_real_smoke " << why
            << "; nothing was verified\n";
  return gufo::testing::laguna::kSkipExitCode;
}

std::uint32_t Argmax(const float* row, std::size_t n) {
  return static_cast<std::uint32_t>(std::max_element(row, row + n) - row);
}

}  // namespace

int main() {
  try {
    const char* opt = std::getenv("GUFO_LAGUNA_DFLASH_REAL");
    if (opt == nullptr || std::string(opt) != "1") {
      return Skip(
          "is opt-in (GUFO_LAGUNA_DFLASH_REAL=1; loads the 74 GB target)");
    }
    std::error_code ec;
    if (!std::filesystem::exists(gufo::testing::laguna::ModelPath(), ec) ||
        !std::filesystem::exists(DrafterPath(), ec)) {
      return Skip("needs the target and drafter GGUF files");
    }
    if (const auto avail = MemAvailableKiB(); avail < kMinAvailableKiB) {
      return Skip("MemAvailable is " + std::to_string(avail / 1024 / 1024) +
                  " GiB, under the 90 GiB floor");
    }

    std::string error;
    const auto target_reader =
        core::GgufReader::OpenFile(gufo::testing::laguna::ModelPath(), &error);
    Require(target_reader != nullptr, "target does not open: " + error);
    const auto drafter_reader =
        core::GgufReader::OpenFile(DrafterPath(), &error);
    Require(drafter_reader != nullptr, "drafter does not open: " + error);
    const auto target = laguna::ModelWeights::Bind(*target_reader, &error);
    Require(target.has_value(), "target bind: " + error);
    const auto drafter = laguna::DFlashWeights::Bind(*drafter_reader, &error);
    Require(drafter.has_value(), "drafter bind: " + error);
    const laguna::DFlashConfig& dc = drafter->config;
    Require(dc.ValidateAgainstTarget(target->config, &error),
            "pairing: " + error);
    const auto tokenizer = laguna::Tokenizer::FromGguf(*target_reader, &error);
    Require(tokenizer != nullptr, "tokenizer: " + error);

    // 8-token prompt: BOS plus the first seven tokens of a plain sentence.
    laguna::EncodeOptions encode;
    encode.add_bos = true;
    // GUFO_LAGUNA_DFLASH_PROMPT_FILE=path replaces the sentence with the file's
    // text and GUFO_LAGUNA_DFLASH_NCTX=n keeps the first n tokens (default 8),
    // so a prompt longer than the 512-token window can be compared too.
    std::string text =
        "The capital of France is Paris, and the capital of Germany is Berlin.";
    if (const char* pf = std::getenv("GUFO_LAGUNA_DFLASH_PROMPT_FILE");
        pf != nullptr && *pf != '\0') {
      std::ifstream in(pf);
      Require(static_cast<bool>(in), "cannot read the prompt file");
      text.assign(std::istreambuf_iterator<char>(in),
                  std::istreambuf_iterator<char>());
    }
    std::size_t n_ctx = 8;
    if (const char* nc = std::getenv("GUFO_LAGUNA_DFLASH_NCTX");
        nc != nullptr && *nc != '\0') {
      n_ctx = static_cast<std::size_t>(std::stoul(nc));
    }
    auto prompt = tokenizer->Encode(text, encode);
    Require(prompt.size() >= n_ctx, "prompt has fewer tokens than NCTX");
    prompt.resize(n_ctx);
    std::vector<std::int32_t> tokens(prompt.begin(), prompt.end());
    std::cout << "prompt (" << n_ctx << "):";
    for (const auto t : tokens) {
      std::cout << " " << t << "<" << tokenizer->DecodeToken(t) << ">";
    }
    std::cout << "\n" << std::flush;

    std::vector<std::uint32_t> layers;
    for (std::size_t i = 0; i < dc.tap_count(); ++i) {
      layers.push_back(dc.tap_layer(i));  // 1, 10, 19, 29, 38, 47
    }
    rf::Options options;  // F32, exact activations
    const rf::Model target_model(*target, options);
    const std::size_t vocab = target->output.rows;

    // Run 1: the prompt. Its taps are the drafter's context; its last-row
    // argmax is the anchor (the sampled token not yet fed to the target).
    const auto t0 = std::chrono::steady_clock::now();
    const rf::TargetTaps taps =
        rf::ForwardWithTaps(target_model, tokens, layers);
    const std::uint32_t anchor =
        Argmax(taps.logits.data() + (tokens.size() - 1) * vocab, vocab);
    const auto t1 = std::chrono::steady_clock::now();
    std::cout << "target prompt forward: "
              << std::chrono::duration<double>(t1 - t0).count()
              << " s; anchor (target greedy after prompt) = " << anchor << "<"
              << tokenizer->DecodeToken(static_cast<int>(anchor)) << ">\n"
              << std::flush;
    for (const float v : taps.features) {
      Require(std::isfinite(v), "tap features are finite");
    }

    // Drafter: context = the 8 prompt rows at positions 0..7, block at 8.
    std::vector<std::int32_t> ctx_pos;
    for (std::int32_t p = 0; p < static_cast<std::int32_t>(n_ctx); ++p) {
      ctx_pos.push_back(p);
    }
    std::vector<std::int32_t> block(
        dc.block_size, static_cast<std::int32_t>(dc.mask_token_id));
    block[0] = static_cast<std::int32_t>(anchor);
    // Drafter-only knobs, to mimic llama.cpp's CPU arithmetic when diffing
    // against the fork: GUFO_LAGUNA_DFLASH_ACT=cpu (Q8_K/Q8_0 activations) and
    // GUFO_LAGUNA_DFLASH_KVF16=1 (F16 K/V cache) and
    // GUFO_LAGUNA_DFLASH_FA16ACC=1 (the fork's CPU flash-attention F16 V
    // accumulator).
    rf::Options drafter_options = options;
    if (const char* act = std::getenv("GUFO_LAGUNA_DFLASH_ACT");
        act != nullptr && std::string(act) == "cpu") {
      drafter_options.activation_quant = rf::ActivationQuant::kCpu;
    }
    if (const char* kv = std::getenv("GUFO_LAGUNA_DFLASH_KVF16");
        kv != nullptr && std::string(kv) == "1") {
      drafter_options.round_kv_f16 = true;
    }
    if (const char* fa = std::getenv("GUFO_LAGUNA_DFLASH_FA16ACC");
        fa != nullptr && std::string(fa) == "1") {
      drafter_options.fa_f16_accumulator = true;  // CPU fork's flash attention
    }
    const rf::DFlashModel drafter_model(*drafter, *target, drafter_options);
    const auto t2 = std::chrono::steady_clock::now();
    // GUFO_LAGUNA_DFLASH_TAPS=DIR also writes every drafter tap there, for
    // layer-by-layer diffs against the fork (fork_compare).
    std::optional<rf::TapWriter> tap_writer;
    rf::TapFn tap_fn;
    if (const char* td = std::getenv("GUFO_LAGUNA_DFLASH_TAPS");
        td != nullptr && *td != '\0') {
      tap_writer.emplace(td);
      tap_fn = std::ref(*tap_writer);
    }
    const std::vector<float> logits = drafter_model.Forward(
        {taps.features, ctx_pos, block, static_cast<std::int32_t>(n_ctx)},
        tap_fn);
    if (tap_writer.has_value()) {
      Require(tap_writer->Finish(&error), "tap write: " + error);
    }
    const auto t3 = std::chrono::steady_clock::now();
    Require(logits.size() == std::size_t{dc.max_draft_tokens()} * vocab,
            "15 draft rows");
    for (const float v : logits) {
      Require(std::isfinite(v), "draft logits are finite");
    }
    if (const char* dump = std::getenv("GUFO_LAGUNA_DFLASH_DUMP");
        dump != nullptr && *dump != '\0') {
      // Input of tests/models/laguna/fork_compare (the fork's libllama).
      const std::filesystem::path dir = dump;
      std::filesystem::create_directories(dir);
      const auto write = [&](const char* name, const std::vector<float>& v) {
        std::ofstream f(dir / name, std::ios::binary);
        f.write(reinterpret_cast<const char*>(v.data()),
                static_cast<std::streamsize>(v.size() * sizeof(float)));
      };
      write("features.f32", taps.features);
      write("logits.f32", logits);
      std::ofstream meta(dir / "meta.txt");
      meta << ctx_pos.size() << " " << anchor << " " << dc.block_size << " "
           << dc.mask_token_id << " " << vocab << "\n";
    }
    if (std::getenv("GUFO_LAGUNA_DFLASH_DUMP_ONLY") != nullptr) {
      std::cout << "DUMP_ONLY: stopping before the teacher-forced check\n";
      return 0;
    }
    const auto drafts = rf::RowArgmax(logits, vocab);
    std::cout << "drafter forward: "
              << std::chrono::duration<double>(t3 - t2).count() << " s\n"
              << "drafter top-1 for positions 9..23 (draft[r] predicts the "
                 "token at position n_ctx + r + 1):\n";
    for (std::size_t i = 0; i < drafts.size(); ++i) {
      std::cout << "  " << (9 + i) << ": " << drafts[i] << "<"
                << tokenizer->DecodeToken(static_cast<int>(drafts[i])) << ">\n";
    }

    // Run 2: teacher-forced verification, [prompt, anchor, d1..d14].
    std::vector<std::int32_t> verify = tokens;
    verify.push_back(static_cast<std::int32_t>(anchor));
    for (std::size_t i = 0; i + 1 < drafts.size(); ++i) {
      verify.push_back(static_cast<std::int32_t>(drafts[i]));
    }
    rf::Options last_only = options;
    const rf::Model verify_model(*target, last_only);
    const rf::TargetTaps v = rf::ForwardWithTaps(verify_model, verify, layers);
    std::size_t accepted = 0;
    bool still = true;
    std::cout << "target greedy at the same positions (row 8 + i, given the "
                 "draft prefix):\n";
    std::vector<std::uint32_t> greedy;
    for (std::size_t i = 0; i < drafts.size(); ++i) {
      greedy.push_back(Argmax(v.logits.data() + (n_ctx + i) * vocab, vocab));
      const bool match = greedy[i] == drafts[i];
      still = still && match;
      accepted += still ? 1 : 0;
      std::cout << "  " << (9 + i) << ": target " << greedy[i] << "<"
                << tokenizer->DecodeToken(static_cast<int>(greedy[i]))
                << "> draft " << drafts[i] << (match ? "  match" : "  miss")
                << "\n";
    }
    // The prompt rows' taps must not depend on what follows (causal target).
    for (std::size_t i = 0; i < taps.features.size(); ++i) {
      Require(std::abs(taps.features[i] - v.features[i]) <=
                  1e-3F * (1.0F + std::abs(taps.features[i])),
              "prompt taps are the same in the longer forward");
    }
    std::size_t matches = 0;
    for (std::size_t i = 0; i < drafts.size(); ++i) {
      matches += greedy[i] == drafts[i] ? 1 : 0;
    }
    std::cout << "RESULT first draft matches target greedy: "
              << (greedy[0] == drafts[0] ? "yes" : "no")
              << "; leading accepted " << accepted << " of 15; position "
              << "matches " << matches << " of 15\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << "\n";
    return 1;
  }
}
