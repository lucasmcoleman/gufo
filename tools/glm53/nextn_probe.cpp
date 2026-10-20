// NextN (MTP) block probe (not a test). On a real GGUF:
//   1. Greedy-generates kGen tokens with the GPU trunk after a text prompt,
//      keeping every position's final_norm.out (h) and logits.
//   2. Runs Executor::ForwardNextn over the whole sequence, token p paired
//      with h[p - 1] (zeros at p = 0), as llama.cpp's draft-mtp catch-up
//      does. Row p's argmax drafts token p + 1; over the generated part, its
//      agreement with the trunk's greedy token is the 1-token draft
//      acceptance a greedy MTP decode would see.
//   3. Checks the GPU NextN rows against ReferenceModel::ForwardNextn (CPU,
//      same h) on the last kCpu positions: hidden rel L2 and logits argmax.
//
// usage: glm53_flash_nextn_probe MODEL.gguf
// Exit 0 when the CPU check passes (hidden rel <= kTol on every row).

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/glm53_flash/executor.hpp"
#include "src/models/glm53_flash/reference.hpp"
#include "src/models/glm53_flash/tokenizer.hpp"
#include "src/models/glm53_flash/weights.hpp"

namespace glm = gufo::models::glm53_flash;
namespace ref = gufo::models::glm53_flash::reference;

namespace {

constexpr std::uint32_t kGen = 48;
constexpr std::uint32_t kCpu = 4;
constexpr double kTol = 3e-2;

std::size_t ArgMax(const float* v, std::size_t n) {
  return static_cast<std::size_t>(std::max_element(v, v + n) - v);
}

double RelL2(const float* a, const float* b, std::size_t n) {
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double d = static_cast<double>(a[i]) - b[i];
    num += d * d;
    den += static_cast<double>(b[i]) * b[i];
  }
  return std::sqrt(num / den);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: glm53_flash_nextn_probe MODEL.gguf\n";
    return 2;
  }
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(argv[1], &error);
  if (!reader) {
    std::cerr << "open failed: " << error << "\n";
    return 1;
  }
  auto weights = glm::ModelWeights::Bind(*reader, /*bind_nextn=*/true, &error);
  auto tokenizer = glm::Tokenizer::CreateFromGguf(*reader, &error);
  if (!weights || !tokenizer) {
    std::cerr << "bind failed: " << error << "\n";
    return 1;
  }
  const glm::Config& c = weights->config;
  const std::size_t H = c.hidden_size;
  const std::size_t V = c.vocab_size;
  glm::ExecutorOptions options;
  options.max_tokens = 64;
  options.capacity = 512;
  options.nextn = true;
  options.shards = reader->GetMappedRegions();
  auto exec = glm::Executor::Create(*weights, options, glm::LaneBOps(), &error);
  if (!exec) {
    std::cerr << "create failed: " << error << "\n";
    return 1;
  }

  // 1. Prompt + greedy continuation, every position's h and logits.
  std::vector<std::int32_t> seq;
  for (const auto id : tokenizer->Encode(
           "Write a detailed story about a lighthouse keeper.", false)) {
    seq.push_back(static_cast<std::int32_t>(id));
  }
  const std::size_t prompt = seq.size();
  std::vector<float> h_all, logits_all;
  glm::ForwardResult out;
  if (!exec->Forward(seq, true, &out, nullptr, &error)) {
    std::cerr << "prompt failed: " << error << "\n";
    return 1;
  }
  h_all = out.hidden;
  logits_all = out.logits;
  for (std::uint32_t g = 0; g < kGen; ++g) {
    const auto next = static_cast<std::int32_t>(
        ArgMax(logits_all.data() + ((seq.size() - 1) * V), V));
    seq.push_back(next);
    const std::int32_t one[1] = {next};
    if (!exec->Forward(one, true, &out, nullptr, &error)) {
      std::cerr << "decode failed: " << error << "\n";
      return 1;
    }
    h_all.insert(h_all.end(), out.hidden.begin(), out.hidden.end());
    logits_all.insert(logits_all.end(), out.logits.begin(), out.logits.end());
  }
  const std::size_t T = seq.size();

  // 2. NextN over the sequence: token p with h[p - 1].
  std::vector<float> h_shift(T * H, 0.0F);
  std::copy(h_all.begin(), h_all.begin() + ((T - 1) * H), h_shift.begin() + H);
  glm::ForwardResult mtp;
  if (!exec->ForwardNextn(seq, h_shift, true, &mtp, &error,
                          /*all_rows=*/true)) {
    std::cerr << "FAIL ForwardNextn: " << error << "\n";
    return 1;
  }
  std::uint32_t agree = 0, total = 0;
  for (std::size_t p = prompt - 1; p + 1 < T; ++p) {
    ++total;
    agree += ArgMax(mtp.logits.data() + (p * V), V) ==
             static_cast<std::size_t>(seq[p + 1]);
  }
  std::cout << "1-token draft agreement over " << total
            << " greedy positions: " << agree << "/" << total << " ("
            << (100.0 * agree / total) << "%)\n";
  std::cout << "text: " << tokenizer->Decode(std::vector<glm::TokenId>(
                               seq.begin() + prompt, seq.end()))
            << "\n";

  // 3. CPU reference on the last kCpu rows. The reference NextN cache must
  // see every earlier row too, so it runs the whole sequence (CPU, slow).
  ref::GgufWeightSource source(*weights);
  ref::ReferenceModel model(c, source);
  const auto t0 = std::chrono::steady_clock::now();
  const auto want = model.ForwardNextn(
      std::span(seq), std::span<const float>(h_shift), true);
  std::cout << "CPU NextN " << T << " rows in "
            << std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                             t0)
                   .count()
            << " s\n";
  int bad = 0;
  for (std::size_t p = T - kCpu; p < T; ++p) {
    const double rel = RelL2(mtp.hidden.data() + (p * H),
                             want.hidden.data() + (p * H), H);
    const bool same_top = ArgMax(mtp.logits.data() + (p * V), V) ==
                          ArgMax(want.logits.data() + (p * V), V);
    const bool ok = std::isfinite(rel) && rel <= kTol;
    std::cout << (ok ? "ok   " : "FAIL ") << "row " << p << ": h_nextn rel "
              << rel << " (<= " << kTol << "), argmax "
              << (same_top ? "same" : "differs") << "\n";
    bad += ok ? 0 : 1;
  }
  std::cout << (bad == 0 ? "PASS" : "FAIL") << "\n";
  return bad == 0 ? 0 : 1;
}
