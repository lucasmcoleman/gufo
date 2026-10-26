// laguna_dflash_cost: wall time of one DFlash drafter pass on the real drafter
// (DFLASH.md 6d item 6, replacing 6b-0's 6.2 ms bandwidth estimate).
//
//   laguna_dflash_cost --out FILE.csv [--features FILE] [--reps 30]
//                      [--warmup 5] [--blocks 4,8,16]
//
// Loads the drafter (0.61 GiB) and the target's head (Q6_K, 0.24 GiB) on the
// device and maps the target's token_embd; the 68.6 GiB target is not loaded.
// A DrafterSession injects 511 context rows at positions 189..699 (the long
// check's real GPU taps from --features, rows x 6 x 3072 F32; random rows
// without it) on the vector path, so a block at 700 attends a full window.
// Timed, one CSV row per call:
//  * propose: DrafterSession::Propose(anchor, B) on the vector and the batched
//    path (host embedding gather, upload, the block forward, head, argmax and
//    the drafts' download, ending with a stream synchronize). The ring is not
//    advanced by a block, so every rep starts from the same state.
//  * inject: the verify rows' injection, InjectDevice of 16 (and 1) rows at
//    700 from device features plus a stream synchronize, then a rewind to 700.
// One rep runs every (kind, path, B) once in an order rotated by the rep; the
// first --warmup reps are discarded. Unprofiled. Prints medians.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>

#include "src/core/gguf_reader.hpp"
#include "tests/models/laguna/dflash_gpu.hpp"

using namespace gufo::testing::laguna;
namespace core = gufo::core;

namespace {

[[noreturn]] void Fail(const std::string& m) {
  std::cerr << "laguna_dflash_cost: " << m << "\n";
  std::exit(1);
}

std::vector<std::uint32_t> ParseList(const std::string& text) {
  std::vector<std::uint32_t> v;
  std::stringstream s(text);
  for (std::string item; std::getline(s, item, ',');) {
    v.push_back(static_cast<std::uint32_t>(std::stoul(item)));
  }
  return v;
}

double Median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const std::size_t n = v.size();
  return n % 2 == 1 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

struct Item {
  std::string kind;  // propose | inject
  gpu::Path path;
  std::uint32_t rows;
};

}  // namespace

int main(int argc, char** argv) {
  std::string out_path;
  std::string features_path;
  std::uint32_t reps = 30;
  std::uint32_t warmup = 5;
  std::vector<std::uint32_t> blocks{4, 8, 16};
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const auto value = [&]() -> std::string {
      if (i + 1 >= argc) {
        Fail(a + " needs a value");
      }
      return argv[++i];
    };
    if (a == "--out") {
      out_path = value();
    } else if (a == "--features") {
      features_path = value();
    } else if (a == "--reps") {
      reps = static_cast<std::uint32_t>(std::stoul(value()));
    } else if (a == "--warmup") {
      warmup = static_cast<std::uint32_t>(std::stoul(value()));
    } else if (a == "--blocks") {
      blocks = ParseList(value());
    } else {
      Fail("unknown option " + a);
    }
  }
  if (out_path.empty()) {
    Fail("--out is required");
  }
  try {
    std::string error;
    const std::string drafter_path =
        "/server/ai/models/Myric/Laguna-S-2.1-APEX-GGUF/"
        "laguna-s-2.1-DFlash-Q4_K.gguf";
    const auto treader = core::GgufReader::OpenFile(ModelPath(), &error);
    const auto dreader = core::GgufReader::OpenFile(drafter_path, &error);
    if (treader == nullptr || dreader == nullptr) {
      Fail("cannot open the GGUFs: " + error);
    }
    const auto target = ml::ModelWeights::Bind(*treader, &error);
    const auto drafter = ml::DFlashWeights::Bind(*dreader, &error);
    if (!target || !drafter) {
      Fail("bind: " + error);
    }
    const ml::DFlashConfig& c = drafter->config;
    const std::size_t width = c.fused_input_width();
    constexpr std::uint32_t kFloor = 189;
    constexpr std::uint32_t kRows = 511;
    std::vector<float> feats(kRows * width);
    if (!features_path.empty()) {
      std::ifstream in(features_path, std::ios::binary);
      in.read(reinterpret_cast<char*>(feats.data()),
              static_cast<std::streamsize>(feats.size() * sizeof(float)));
      if (in.gcount() !=
          static_cast<std::streamsize>(feats.size() * sizeof(float))) {
        Fail("--features is not 511 rows of " + std::to_string(width));
      }
    } else {
      std::mt19937_64 rng(6);
      feats = RandomFeatures(kRows, c.tap_count(), c.hidden_size, rng);
    }
    DrafterCtx ctx(*drafter);
    DeviceHead head(target->output);
    hipStream_t stream{};
    CheckHip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "stream");
    auto session = gpu::DrafterSession::Create(
        ctx.weights(), head.matrix(), target->token_embd,
        static_cast<std::int32_t>(c.mask_token_id), 32, stream, &error);
    if (session == nullptr) {
      Fail("session: " + error);
    }
    session->kv().Reset(kFloor);
    for (std::uint32_t p = 0; p < kRows; p += 32) {
      const std::uint32_t n = std::min(32U, kRows - p);
      if (!session->Inject(
              std::span<const float>(feats).subspan(p * width, n * width),
              kFloor + p, gpu::Path::kVector, &error)) {
        Fail("inject: " + error);
      }
    }
    const std::uint32_t n0 = session->kv().committed();
    float* dfeat = ToDevice(std::vector<float>(
        feats.end() - static_cast<std::ptrdiff_t>(16 * width), feats.end()));
    std::vector<Item> items;
    for (const gpu::Path path : {gpu::Path::kVector, gpu::Path::kBatched}) {
      for (const std::uint32_t b : blocks) {
        items.push_back({"propose", path, b});
      }
    }
    items.push_back({"inject", gpu::Path::kVector, 16});
    items.push_back({"inject", gpu::Path::kVector, 1});
    std::ofstream csv(out_path);
    csv << "kind,path,rows,rep,ms\n";
    std::map<std::string, std::vector<double>> times;
    std::vector<std::int32_t> drafts;
    const std::int32_t anchor = 340;
    for (std::uint32_t rep = 0; rep < warmup + reps; ++rep) {
      for (std::size_t k = 0; k < items.size(); ++k) {
        const Item& it = items[(k + rep) % items.size()];
        CheckHip(hipStreamSynchronize(stream), "sync");
        const auto t0 = std::chrono::steady_clock::now();
        bool ok = false;
        if (it.kind == "propose") {
          ok = session->Propose(anchor, it.rows, &drafts, nullptr, it.path,
                                &error);
        } else {
          ok = session->InjectDevice(dfeat, it.rows, n0, it.path, &error) &&
               hipStreamSynchronize(stream) == hipSuccess;
        }
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0)
                              .count();
        if (!ok) {
          Fail(it.kind + ": " + error);
        }
        if (it.kind == "inject" && !session->kv().RewindTo(n0, &error)) {
          Fail("rewind: " + error);
        }
        const std::string key =
            it.kind + "," + PathName(it.path) + "," + std::to_string(it.rows);
        if (rep >= warmup) {
          times[key].push_back(ms);
          csv << key << "," << rep - warmup << "," << std::setprecision(6) << ms
              << "\n";
        }
      }
    }
    for (auto& [key, v] : times) {
      const auto [lo, hi] = std::minmax_element(v.begin(), v.end());
      std::cout << std::fixed << std::setprecision(3) << key << ": median "
                << Median(v) << " ms [min " << *lo << ", max " << *hi
                << "] n=" << v.size() << "\n";
    }
    (void)hipFree(dfeat);
    session.reset();
    qfn_mmq_release_stream(stream);
    (void)hipStreamDestroy(stream);
    return 0;
  } catch (const std::exception& e) {
    Fail(e.what());
  }
}
