// Greedy MTP decode benchmark (not a test): the cycle the generator will
// run, on the Executor directly.
//   anchor a = argmax(trunk logits at the last accepted position n)
//   draft  d = argmax(ForwardNextn(a at n + 1, h_n))
//   verify ForwardVerify([a, d]); accept d when argmax(row 0) == d, else
//          Rollback(1)
//   then ForwardNextn over the accepted rows the draft block has not seen,
//   the last of which drafts the next cycle.
// Plain greedy decode of the same prompt runs first in the same process;
// prints both rates, the acceptance, and where the two token streams first
// differ (verify rows run in batches of 2, so greedy may drift).
//
// usage: glm53_flash_mtp_bench MODEL.gguf [tokens=256]

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/glm53_flash/executor.hpp"
#include "src/models/glm53_flash/tokenizer.hpp"
#include "src/models/glm53_flash/weights.hpp"

namespace glm = gufo::models::glm53_flash;
using Clock = std::chrono::steady_clock;

namespace {

std::int32_t ArgMax(const float* v, std::size_t n) {
  return static_cast<std::int32_t>(std::max_element(v, v + n) - v);
}

double Since(Clock::time_point t) {
  return std::chrono::duration<double>(Clock::now() - t).count();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: glm53_flash_mtp_bench MODEL.gguf [tokens=256]\n";
    return 2;
  }
  const std::uint32_t want = argc > 2 ? std::atoi(argv[2]) : 256;
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
  const std::size_t H = weights->config.hidden_size;
  const std::size_t V = weights->config.vocab_size;
  glm::ExecutorOptions options;
  options.max_tokens = 64;
  options.capacity = 4096;
  options.nextn = true;
  options.max_verify_tokens = 2;
  options.shards = reader->GetMappedRegions();
  auto exec = glm::Executor::Create(*weights, options, glm::LaneBOps(), &error);
  if (!exec) {
    std::cerr << "create failed: " << error << "\n";
    return 1;
  }
  std::vector<std::int32_t> prompt;
  for (const auto id : tokenizer->Encode(
           "Write a detailed story about a lighthouse keeper.", false)) {
    prompt.push_back(static_cast<std::int32_t>(id));
  }
  const auto fail = [&](const char* what) {
    std::cerr << "FAIL " << what << ": " << error << "\n";
    return 1;
  };

  // Plain greedy decode.
  std::vector<std::int32_t> plain;
  glm::ForwardResult out;
  if (!exec->Reset() || !exec->Forward(prompt, true, &out, nullptr, &error))
    return fail("prompt");
  std::int32_t next = ArgMax(out.logits.data() + ((prompt.size() - 1) * V), V);
  auto t = Clock::now();
  while (plain.size() < want) {
    plain.push_back(next);
    const std::int32_t one[1] = {next};
    if (!exec->Forward(one, true, &out, nullptr, &error)) return fail("decode");
    next = ArgMax(out.logits.data(), V);
  }
  const double plain_s = Since(t);

  // MTP decode.
  std::vector<std::int32_t> mtp;
  if (!exec->Reset() || !exec->Forward(prompt, true, &out, nullptr, &error))
    return fail("prompt");
  // Draft-block catch-up over the prompt: token p with h[p - 1].
  const std::size_t P = prompt.size();
  std::vector<float> h_prev(P * H, 0.0F);
  std::copy(out.hidden.begin(), out.hidden.end() - H, h_prev.begin() + H);
  std::vector<float> h_last(out.hidden.end() - H, out.hidden.end());
  std::int32_t anchor = ArgMax(out.logits.data() + ((P - 1) * V), V);
  glm::ForwardResult nx;
  if (!exec->ForwardNextn(prompt, h_prev, false, &nx, &error))
    return fail("nextn prompt");
  std::uint32_t cycles = 0, accepted = 0;
  double nextn_s = 0, verify_s = 0;
  t = Clock::now();
  // Each cycle: the draft block sees `anchor` paired with h_last.
  std::vector<std::int32_t> nx_tokens = {anchor};
  std::vector<float> nx_h = h_last;
  while (mtp.size() < want) {
    auto tc = Clock::now();
    if (!exec->ForwardNextn(nx_tokens, nx_h, true, &nx, &error))
      return fail("nextn");
    nextn_s += Since(tc);
    const std::int32_t draft = ArgMax(nx.logits.data(), V);
    const std::int32_t verify[2] = {anchor, draft};
    tc = Clock::now();
    if (!exec->ForwardVerify(verify, true, &out, &error)) return fail("verify");
    verify_s += Since(tc);
    ++cycles;
    const std::int32_t first = ArgMax(out.logits.data(), V);
    mtp.push_back(anchor);
    if (first == draft) {
      ++accepted;
      mtp.push_back(draft);
      if (!exec->Rollback(2)) return fail("rollback(2)");
      anchor = ArgMax(out.logits.data() + V, V);
      // The draft block saw `anchor` (row 0's token) last cycle; now the
      // accepted draft (with h of row 0), then the new anchor (with row 1).
      nx_tokens = {draft, anchor};
      nx_h.assign(out.hidden.begin(), out.hidden.end());
    } else {
      if (!exec->Rollback(1)) return fail("rollback(1)");
      anchor = first;
      nx_tokens = {anchor};
      nx_h.assign(out.hidden.begin(), out.hidden.begin() + H);
    }
  }
  const double mtp_s = Since(t);
  mtp.resize(want);
  std::size_t diff = 0;
  while (diff < want && plain[diff] == mtp[diff]) ++diff;
  std::cout << "plain greedy " << want << " tokens: " << want / plain_s
            << " tok/s\n";
  std::cout << "MTP greedy   " << want << " tokens: " << want / mtp_s
            << " tok/s, " << cycles << " cycles, draft accepted " << accepted
            << "/" << cycles << " (" << 100.0 * accepted / cycles << "%)\n";
  std::cout << "per cycle: NextN " << 1e3 * nextn_s / cycles << " ms, verify "
            << 1e3 * verify_s / cycles << " ms, plain decode "
            << 1e3 * plain_s / want << " ms/token\n";
  std::cout << "streams identical for the first " << diff << " of " << want
            << " tokens\n";
  return 0;
}
