// Speculative verify/rollback probe (not a test): Executor::ForwardVerify +
// Rollback on a real GGUF. For prefix lengths P that put the verify pass at
// each position mod kpool (so a dropped token can complete an indexer pool),
// verify sizes n and kept counts k:
//   A  prefix, ForwardVerify(n tokens), Rollback(k), ForwardVerify(2 more)
//   B  as A with the n - k dropped tokens replaced by other ids
//   C  prefix, ForwardVerify(the k kept tokens only), ForwardVerify(2 more)
// A and B must be bit-identical (a dropped token leaves no trace); A and C
// agree to F32 rounding only (the kept tokens ran in batches of n vs k).
// Prints C's max rel and argmax agreement.
//
// usage: glm53_flash_rollback_probe MODEL.gguf [long=1]
// Exit 0 when every A == B.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/glm53_flash/executor.hpp"
#include "src/models/glm53_flash/weights.hpp"

namespace glm = gufo::models::glm53_flash;

namespace {

std::int32_t Id(std::uint32_t i, std::uint32_t salt) {
  return static_cast<std::int32_t>(1000 + ((i * 7919U + salt * 104729U) % 90000U));
}

struct Case {
  std::uint32_t prefix;
  std::uint32_t n;
  std::uint32_t keep;
};

// Logits of the 2 continuation tokens after prefix + verify(n) + rollback.
bool RunA(glm::Executor& e, const Case& c, std::uint32_t drop_salt,
          std::vector<float>* logits, std::string* error) {
  std::vector<std::int32_t> prefix(c.prefix);
  for (std::uint32_t i = 0; i < c.prefix; ++i) prefix[i] = Id(i, 0);
  std::vector<std::int32_t> verify(c.n);
  for (std::uint32_t i = 0; i < c.n; ++i)
    verify[i] = i < c.keep ? Id(c.prefix + i, 0) : Id(c.prefix + i, drop_salt);
  const std::vector<std::int32_t> more = {Id(c.prefix + c.keep, 7),
                                          Id(c.prefix + c.keep + 1, 7)};
  glm::ForwardResult out;
  if (!e.Reset() || !e.Forward(prefix, false, &out, nullptr, error) ||
      !e.ForwardVerify(verify, true, &out, error) || !e.Rollback(c.keep) ||
      e.Position() != c.prefix + c.keep ||
      !e.ForwardVerify(more, true, &out, error)) {
    if (error->empty()) *error = "verify/rollback sequence failed";
    return false;
  }
  *logits = std::move(out.logits);
  return true;
}

bool RunC(glm::Executor& e, const Case& c, std::vector<float>* logits,
          std::string* error) {
  std::vector<std::int32_t> prefix(c.prefix);
  for (std::uint32_t i = 0; i < c.prefix; ++i) prefix[i] = Id(i, 0);
  std::vector<std::int32_t> kept(c.keep);
  for (std::uint32_t i = 0; i < c.keep; ++i) kept[i] = Id(c.prefix + i, 0);
  const std::vector<std::int32_t> more = {Id(c.prefix + c.keep, 7),
                                          Id(c.prefix + c.keep + 1, 7)};
  glm::ForwardResult out;
  if (!e.Reset() || !e.Forward(prefix, false, &out, nullptr, error) ||
      !e.ForwardVerify(kept, true, &out, error) ||
      !e.ForwardVerify(more, true, &out, error)) {
    return false;
  }
  *logits = std::move(out.logits);
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: glm53_flash_rollback_probe MODEL.gguf [long=1]\n";
    return 2;
  }
  const bool long_case = argc < 3 || std::atoi(argv[2]) != 0;
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(argv[1], &error);
  if (!reader) {
    std::cerr << "open failed: " << error << "\n";
    return 1;
  }
  auto weights = glm::ModelWeights::Bind(*reader, /*bind_nextn=*/false, &error);
  if (!weights) {
    std::cerr << "bind failed: " << error << "\n";
    return 1;
  }
  glm::ExecutorOptions options;
  options.max_tokens = 64;
  options.capacity = 4096;
  options.max_verify_tokens = 5;
  options.shards = reader->GetMappedRegions();
  auto exec = glm::Executor::Create(*weights, options, glm::LaneBOps(), &error);
  if (!exec) {
    std::cerr << "create failed: " << error << "\n";
    return 1;
  }
  const std::size_t V = weights->config.vocab_size;
  std::vector<Case> cases;
  for (std::uint32_t p = 64; p < 68; ++p) {
    cases.push_back({p, 2, 1});
    cases.push_back({p, 5, 1});
    cases.push_back({p, 5, 3});
    cases.push_back({p, 5, 4});
  }
  if (long_case) {
    // Past top_k raw tokens: sparse selection, pools dropped.
    cases.push_back({2050, 2, 1});
    cases.push_back({2050, 5, 2});
  }
  int bad = 0;
  for (const Case& c : cases) {
    std::vector<float> a, b, cc;
    if (!RunA(*exec, c, 1, &a, &error) || !RunA(*exec, c, 2, &b, &error) ||
        !RunC(*exec, c, &cc, &error)) {
      std::cout << "FAIL P=" << c.prefix << " n=" << c.n << " k=" << c.keep
                << ": " << error << "\n";
      ++bad;
      continue;
    }
    const bool same = a.size() == b.size() && a.size() == 2 * V &&
                      std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
    double num = 0.0, den = 0.0;
    std::uint32_t agree = 0;
    for (std::size_t r = 0; r < 2; ++r) {
      const float* x = a.data() + (r * V);
      const float* y = cc.data() + (r * V);
      for (std::size_t i = 0; i < V; ++i) {
        const double d = static_cast<double>(x[i]) - y[i];
        num += d * d;
        den += static_cast<double>(y[i]) * y[i];
      }
      agree += std::max_element(x, x + V) - x == std::max_element(y, y + V) - y;
    }
    std::cout << (same ? "ok   " : "FAIL ") << "P=" << c.prefix << " (mod 4 = "
              << c.prefix % 4 << ") n=" << c.n << " k=" << c.keep
              << ": dropped tokens leave no trace " << (same ? "yes" : "NO")
              << "; vs verify of kept only rel " << std::sqrt(num / den)
              << ", argmax " << agree << "/2\n";
    bad += same ? 0 : 1;
  }
  std::cout << (bad == 0 ? "PASS" : "FAIL") << "\n";
  return bad == 0 ? 0 : 1;
}
