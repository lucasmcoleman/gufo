// MTP draft-depth benchmark (not a test). For each prompt (one per line of
// PROMPTS.txt, wrapped as a single user turn with thinking off):
//   - verify cost: ForwardVerify of 1..max_depth+1 tokens, then Rollback(1)
//   - plain greedy decode, the reference stream and ms/token
//   - greedy MTP at draft depth 1..max_depth: drafts chain through the NextN
//     block (draft i + 1 = argmax(ForwardNextn(draft i, h_nextn of draft i)),
//     verify [anchor, drafts...], keep the matching prefix, rewind the draft
//     block to the anchor and catch it up on the kept rows with trunk h.
//   - gated depth 2 for each threshold in GATES: the second draft is chained
//     only when the first draft's softmax probability is at least the gate.
// Prints tok/s, acceptance per draft slot and where each stream first differs
// from plain greedy.
//
// usage: glm53_flash_mtp_depth_bench MODEL.gguf PROMPTS.txt [tokens=256]
//                                     [max_depth=3] [gates=0.5,0.7,0.8,0.9]

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
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

// Softmax probability of the argmax entry.
float TopProb(const float* v, std::size_t n) {
  const float top = *std::max_element(v, v + n);
  double sum = 0;
  for (std::size_t i = 0; i < n; ++i) sum += std::exp(double(v[i]) - top);
  return static_cast<float>(1.0 / sum);
}

double Since(Clock::time_point t) {
  return std::chrono::duration<double>(Clock::now() - t).count();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: glm53_flash_mtp_depth_bench MODEL.gguf PROMPTS.txt "
                 "[tokens=256] [max_depth=3] [gates=0.5,0.7,0.8,0.9]\n";
    return 2;
  }
  const std::uint32_t want = argc > 3 ? std::atoi(argv[3]) : 256;
  const std::uint32_t max_depth =
      std::max<std::uint32_t>(argc > 4 ? std::atoi(argv[4]) : 3, 2);
  // Runs: fixed depth 1..max_depth (gate < 0), then gated depth 2.
  struct Run {
    std::uint32_t depth;
    float gate;
  };
  std::vector<Run> runs;
  for (std::uint32_t d = 1; d <= max_depth; ++d) runs.push_back({d, -1.0F});
  {
    std::string list = argc > 5 ? argv[5] : "0.5,0.7,0.8,0.9";
    for (std::size_t at = 0; at < list.size();) {
      const std::size_t end = std::min(list.find(',', at), list.size());
      runs.push_back({2, std::stof(list.substr(at, end - at))});
      at = end + 1;
    }
  }
  std::vector<std::string> prompts;
  {
    std::ifstream in(argv[2]);
    for (std::string line; std::getline(in, line);)
      if (!line.empty()) prompts.push_back(line);
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
  const std::size_t H = weights->config.hidden_size;
  const std::size_t V = weights->config.vocab_size;
  glm::ExecutorOptions options;
  options.capacity = 4096;
  options.max_tokens = 256;
  options.nextn = true;
  options.max_verify_tokens = max_depth + 1;
  options.shards = reader->GetMappedRegions();
  auto exec = glm::Executor::Create(*weights, options, glm::LaneBOps(), &error);
  if (!exec) {
    std::cerr << "create failed: " << error << "\n";
    return 1;
  }
  const auto fail = [&](const char* what) {
    std::cerr << "FAIL " << what << ": " << error << "\n";
    return 1;
  };
  std::cout << std::fixed << std::setprecision(2);

  for (std::size_t pi = 0; pi < prompts.size(); ++pi) {
    std::vector<std::int32_t> prompt;
    for (const auto id : tokenizer->Encode(
             "[gMASK]<sop><|user|>" + prompts[pi] +
                 "<|assistant|><think></think>",
             true)) {
      prompt.push_back(static_cast<std::int32_t>(id));
    }
    const std::size_t P = prompt.size();
    std::cout << "== prompt " << pi + 1 << " (" << P
              << " tokens): " << prompts[pi].substr(0, 60) << "\n";
    glm::ForwardResult out;

    // Plain greedy decode (stops early on the end-of-turn token's argmax
    // repeating is not handled: the stream simply runs `want` tokens).
    std::vector<std::int32_t> plain;
    if (!exec->Reset() || !exec->Forward(prompt, true, &out, nullptr, &error))
      return fail("prompt");
    std::int32_t next = ArgMax(out.logits.data() + ((P - 1) * V), V);
    auto t = Clock::now();
    while (plain.size() < want) {
      plain.push_back(next);
      const std::int32_t one[1] = {next};
      if (!exec->Forward(one, true, &out, nullptr, &error))
        return fail("decode");
      next = ArgMax(out.logits.data(), V);
    }
    const double plain_s = Since(t);
    std::cout << "plain        " << want / plain_s << " tok/s ("
              << 1e3 * plain_s / want << " ms/token)\n";

    // Verify cost at the current depth of the plain run.
    std::cout << "verify ms:";
    for (std::uint32_t n = 1; n <= max_depth + 1; ++n) {
      std::vector<std::int32_t> ids(plain.begin(), plain.begin() + n);
      double s = 0;
      constexpr int kReps = 8;
      for (int r = 0; r < kReps + 1; ++r) {
        const auto tv = Clock::now();
        if (!exec->ForwardVerify(ids, true, &out, &error))
          return fail("verify cost");
        if (r > 0) s += Since(tv);  // first rep warms up
        if (!exec->Rollback(1)) return fail("verify cost rollback");
        // Rollback(1) keeps ids[0]; drop it too by re-running from scratch
        // is not needed: the cost only depends on n and the depth.
      }
      std::cout << " n=" << n << " " << 1e3 * s / kReps;
    }
    std::cout << "\n";

    for (const Run& run : runs) {
      const std::uint32_t depth = run.depth;
      std::vector<std::int32_t> mtp;
      if (!exec->Reset() ||
          !exec->Forward(prompt, true, &out, nullptr, &error))
        return fail("prompt");
      std::vector<float> h_prev(P * H, 0.0F);
      std::copy(out.hidden.begin(), out.hidden.end() - H, h_prev.begin() + H);
      std::vector<float> nx_h(out.hidden.end() - H, out.hidden.end());
      std::int32_t anchor = ArgMax(out.logits.data() + ((P - 1) * V), V);
      glm::ForwardResult nx;
      if (!exec->ForwardNextn(prompt, h_prev, false, &nx, &error))
        return fail("nextn prompt");
      std::vector<std::int32_t> nx_tokens = {anchor};
      std::uint32_t cycles = 0;
      std::vector<std::uint32_t> slot_ok(depth, 0), slot_tried(depth, 0);
      double nextn_s = 0, verify_s = 0;
      t = Clock::now();
      while (mtp.size() < want) {
        // Catch-up plus the first draft, then chained drafts.
        auto tc = Clock::now();
        if (!exec->ForwardNextn(nx_tokens, nx_h, true, &nx, &error))
          return fail("nextn");
        const std::uint32_t anchor_end = exec->NextnPosition();
        std::vector<std::int32_t> verify = {anchor};
        verify.push_back(ArgMax(nx.logits.data(), V));
        const bool chain =
            run.gate < 0 || TopProb(nx.logits.data(), V) >= run.gate;
        for (std::uint32_t d = 1; chain && d < depth; ++d) {
          const std::int32_t prev[1] = {verify.back()};
          const std::vector<float> h(nx.hidden.end() - H, nx.hidden.end());
          if (!exec->ForwardNextn(prev, h, true, &nx, &error))
            return fail("nextn chain");
          verify.push_back(ArgMax(nx.logits.data(), V));
        }
        const auto drafts = static_cast<std::uint32_t>(verify.size() - 1);
        if (drafts > 1 && !exec->RewindNextn(anchor_end))
          return fail("nextn rewind");
        nextn_s += Since(tc);
        tc = Clock::now();
        if (!exec->ForwardVerify(verify, true, &out, &error))
          return fail("verify");
        verify_s += Since(tc);
        ++cycles;
        std::uint32_t m = 0;  // accepted drafts
        while (m < drafts) {
          ++slot_tried[m];
          if (ArgMax(out.logits.data() + (m * V), V) != verify[m + 1]) break;
          ++slot_ok[m];
          ++m;
        }
        for (std::uint32_t i = 0; i <= m; ++i) mtp.push_back(verify[i]);
        if (!exec->Rollback(m + 1)) return fail("rollback");
        anchor = ArgMax(out.logits.data() + (m * V), V);
        nx_tokens.assign(verify.begin() + 1, verify.begin() + 1 + m);
        nx_tokens.push_back(anchor);
        nx_h.assign(out.hidden.begin(), out.hidden.begin() + ((m + 1) * H));
      }
      const double mtp_s = Since(t);
      mtp.resize(want);
      std::size_t diff = 0;
      while (diff < want && plain[diff] == mtp[diff]) ++diff;
      if (run.gate < 0)
        std::cout << "depth " << depth << "      ";
      else
        std::cout << "gate " << run.gate << "  ";
      std::cout << want / mtp_s << " tok/s, "
                << cycles << " cycles, " << static_cast<double>(want) / cycles
                << " tok/cycle, accept";
      for (std::uint32_t d = 0; d < depth; ++d)
        std::cout << " " << slot_ok[d] << "/" << slot_tried[d];
      std::cout << ", per cycle NextN " << 1e3 * nextn_s / cycles
                << " ms verify " << 1e3 * verify_s / cycles
                << " ms, same as plain for " << diff << "/" << want << "\n";
    }
  }
  return 0;
}
