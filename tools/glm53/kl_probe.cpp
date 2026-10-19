// V6 KL probe (#675): full-vocabulary logits of the gufo GPU Executor for a
// fixed token set, for comparison with llama.cpp's (moe-compare/v6/).
//
// usage: glm53_flash_kl_probe MODEL.gguf TOKENS.i32 OUTDIR
//   TOKENS.i32 is [N][256] int32 (written by v6/llama_logits). For each
//   sequence: Reset, one 256-token Forward with logits at every position,
//   write OUTDIR/NN.f32 ([256][vocab] F32). GUFO_GLM53_SCALAR_MOE=1 selects
//   the scalar MoE path, as for the other probes.
// Memory and upload guard as in model_probe.cpp: ~87 GiB of managed memory;
// run under the GPU admission lock.

#include <hip/hip_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <sys/mman.h>
#include <thread>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/glm53_flash/executor.hpp"
#include "src/models/glm53_flash/weights.hpp"

namespace glm = gufo::models::glm53_flash;

namespace {

constexpr std::size_t kT = 256;

double Now() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

std::uint64_t GttUsed() {
  std::ifstream f("/sys/class/drm/card0/device/mem_info_gtt_used");
  std::uint64_t v = 0;
  f >> v;
  return v;
}

/// Same as model_probe.cpp's UploadGuard: drop mapped shard pages while the
/// upload runs, SIGTERM if GTT stops growing for `stall_s`.
void UploadGuard(const std::atomic<bool>& stop, double stall_s) {
  std::uint64_t best = 0;
  double best_at = Now();
  while (!stop.load()) {
    std::ifstream maps("/proc/self/maps");
    std::string line;
    while (std::getline(maps, line)) {
      if (line.find(".gguf") == std::string::npos ||
          line.find(" r--") == std::string::npos)
        continue;
      unsigned long lo = 0, hi = 0;
      if (std::sscanf(line.c_str(), "%lx-%lx", &lo, &hi) == 2)
        (void)madvise(reinterpret_cast<void*>(lo), hi - lo, MADV_DONTNEED);
    }
    const std::uint64_t g = GttUsed();
    if (g > best + (64ULL << 20)) {
      best = g;
      best_at = Now();
    }
    if (Now() - best_at > stall_s) {
      std::fprintf(stderr, "[upload] GTT stalled, sending SIGTERM\n");
      std::raise(SIGTERM);
      return;
    }
    for (int i = 0; i < 50 && !stop.load(); ++i)
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: " << argv[0] << " MODEL.gguf TOKENS.i32 OUTDIR\n";
    return 2;
  }
#if !defined(GLM53_LANE_B_OPS)
  std::cerr << "built without GLM53_LANE_B_OPS\n";
  return 3;
#else
  const std::string out_dir = argv[3];
  std::vector<std::int32_t> tokens;
  {
    std::ifstream in(argv[2], std::ios::binary | std::ios::ate);
    const auto bytes = static_cast<std::size_t>(in.tellg());
    if (!in || bytes == 0 || bytes % (kT * sizeof(std::int32_t)) != 0) {
      std::cerr << "bad token file " << argv[2] << "\n";
      return 2;
    }
    tokens.resize(bytes / sizeof(std::int32_t));
    in.seekg(0);
    in.read(reinterpret_cast<char*>(tokens.data()),
            static_cast<std::streamsize>(bytes));
  }
  const std::size_t n_seq = tokens.size() / kT;
  try {
    std::string error;
    auto reader = gufo::core::GgufReader::OpenFile(argv[1], &error);
    if (!reader) {
      std::cerr << "open failed: " << error << "\n";
      return 1;
    }
    auto weights =
        glm::ModelWeights::Bind(*reader, /*bind_nextn=*/false, &error);
    if (!weights) {
      std::cerr << "bind failed: " << error << "\n";
      return 1;
    }
    const std::size_t V = weights->config.vocab_size;
    glm::ExecutorOptions options;
    options.first_layer = 0;
    options.last_layer = 0;  // every trunk layer
    options.embed = true;
    options.head = true;
    // KL_MAX_TOKENS splits each sequence into passes of that many tokens
    // (determinism/chunking checks); default one 256-token pass.
    const char* mt = std::getenv("KL_MAX_TOKENS");
    options.max_tokens = mt != nullptr ? static_cast<std::uint32_t>(std::atoi(mt)) : kT;
    options.capacity = kT;
    const double up0 = Now();
    std::atomic<bool> upload_done{false};
    std::thread guard(UploadGuard, std::cref(upload_done), 300.0);
    auto exec = glm::Executor::Create(*weights, options, glm::LaneBOps(), &error);
    upload_done = true;
    guard.join();
    if (!exec) {
      std::cout << "FAIL create: " << error << "\n";
      return 1;
    }
    std::cout << "uploaded in " << Now() - up0 << " s, " << n_seq
              << " sequences\n" << std::flush;
    for (std::size_t s = 0; s < n_seq; ++s) {
      const double t0 = Now();
      glm::ForwardResult out;
      const std::span<const std::int32_t> ids(tokens.data() + s * kT, kT);
      if (!exec->Reset() || !exec->Forward(ids, true, &out, nullptr, &error)) {
        std::cout << "FAIL seq " << s << ": " << error << "\n";
        return 1;
      }
      if (out.logits.size() != kT * V) {
        std::cout << "FAIL seq " << s << ": logits size " << out.logits.size()
                  << "\n";
        return 1;
      }
      char name[32];
      std::snprintf(name, sizeof name, "/%02zu.f32", s);
      std::ofstream f(out_dir + name, std::ios::binary);
      f.write(reinterpret_cast<const char*>(out.logits.data()),
              static_cast<std::streamsize>(out.logits.size() * sizeof(float)));
      std::cout << "[" << s + 1 << "/" << n_seq << "] " << Now() - t0 << " s\n"
                << std::flush;
    }
    exec->PrintProfile(std::cout);  // GUFO_GLM53_PROFILE=1 only
    return 0;
  } catch (const std::exception& e) {
    std::cout << "FAIL: " << e.what() << "\n";
    return 1;
  }
#endif
}
