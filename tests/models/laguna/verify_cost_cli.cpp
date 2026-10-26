// laguna_verify_cost: times one target Forward of w rows at a fixed context
// depth, the verify cost that decides whether DFlash can pay (DFLASH.md,
// step 6b-0 (c)). No drafter is involved.
//
//   laguna_verify_cost --text FILE[,FILE...] --out FILE.csv
//                      [--depths 512,2048,8192] [--widths 1,2,4,8,16]
//                      [--batched-widths 8,16] [--reps 20] [--warmup 3]
//                      [--model GGUF]
//
// The texts are tokenized (BOS first, then each file in order) and give both
// the context and the verified rows: at depth d a call runs tokens d..d+w-1,
// so the rows are the real continuation of the context.
//
// Two executors share the uploaded weights:
//   vector:  Path::kVector, max_chunk 32 (DFLASH.md rule 3), every width;
//   batched: Path::kBatched, max_chunk 512, --batched-widths only.
// Per depth each executor is prefilled to d (vector in chunks of 32, batched
// in chunks of 512) and snapshotted; every timed call is preceded by an
// untimed RestoreSnapshot, so all reps start from the same state. One rep
// runs every width once, in an order rotated by the rep index; the first
// --warmup reps are discarded. Timed: Forward(kAll logits to the host), which
// ends with a stream synchronize. One CSV row per timed call.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#include "src/models/laguna/config.hpp"
#include "src/models/laguna/kernels/device_model.hpp"
#include "src/models/laguna/kernels/executor.hpp"
#include "src/models/laguna/tokenizer.hpp"
#include "src/models/laguna/weights.hpp"
#include "tests/models/laguna/support.hpp"

namespace ml = gufo::models::laguna;
namespace gpu = gufo::models::laguna::rocm;

namespace {

[[noreturn]] void Fail(const std::string& message) {
  std::cerr << "laguna_verify_cost: " << message << "\n";
  std::exit(1);
}

std::vector<std::string> Split(const std::string& text) {
  std::vector<std::string> parts;
  std::stringstream stream(text);
  for (std::string item; std::getline(stream, item, ',');) {
    if (!item.empty()) {
      parts.push_back(item);
    }
  }
  return parts;
}

std::vector<std::uint32_t> ParseList(const std::string& flag,
                                     const std::string& text) {
  std::vector<std::uint32_t> values;
  for (const std::string& part : Split(text)) {
    std::uint32_t v = 0;
    const auto [end, ec] =
        std::from_chars(part.data(), part.data() + part.size(), v);
    if (ec != std::errc{} || end != part.data() + part.size() || v == 0) {
      Fail(flag + " needs positive integers, got '" + text + "'");
    }
    values.push_back(v);
  }
  return values;
}

struct Args {
  std::filesystem::path model = gufo::testing::laguna::ModelPath();
  std::vector<std::string> texts;
  std::filesystem::path out;
  std::vector<std::uint32_t> depths{512, 2048, 8192};
  std::vector<std::uint32_t> widths{1, 2, 4, 8, 16};
  std::vector<std::uint32_t> batched{8, 16};
  std::uint32_t reps{20};
  std::uint32_t warmup{3};
};

Args Parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    const auto value = [&]() -> std::string {
      if (i + 1 >= argc) {
        Fail(flag + " needs a value");
      }
      return argv[++i];
    };
    if (flag == "--model") {
      a.model = value();
    } else if (flag == "--text") {
      a.texts = Split(value());
    } else if (flag == "--out") {
      a.out = value();
    } else if (flag == "--depths") {
      a.depths = ParseList(flag, value());
    } else if (flag == "--widths") {
      a.widths = ParseList(flag, value());
    } else if (flag == "--batched-widths") {
      const std::string v = value();
      a.batched =
          v == "none" ? std::vector<std::uint32_t>{} : ParseList(flag, v);
    } else if (flag == "--reps") {
      a.reps = ParseList(flag, value()).at(0);
    } else if (flag == "--warmup") {
      a.warmup = ParseList(flag, value()).at(0);
    } else {
      Fail("unknown option " + flag);
    }
  }
  if (a.texts.empty() || a.out.empty()) {
    Fail("--text and --out are required");
  }
  std::sort(a.depths.begin(), a.depths.end());
  for (std::uint32_t w : a.widths) {
    if (w > 32) {
      Fail("vector widths are at most 32");
    }
  }
  return a;
}

double Seconds(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
}

struct Arm {
  const char* name;
  gpu::Executor* ex;
  std::uint32_t chunk;
  std::vector<std::uint32_t> widths;
};

/// Advances `ex` from its position to `depth` with `chunk`-row calls.
void PrefillTo(gpu::Executor& ex, const std::vector<std::int32_t>& tokens,
               std::uint32_t depth, std::uint32_t chunk) {
  std::string error;
  while (ex.position() < depth) {
    const std::uint32_t n = std::min(chunk, depth - ex.position());
    if (!ex.Forward(std::span(tokens).subspan(ex.position(), n),
                    gpu::LogitsRows::kNone, nullptr, &error)) {
      Fail("prefill: " + error);
    }
  }
}

int Run(int argc, char** argv) {
  const Args args = Parse(argc, argv);
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(args.model, &error);
  if (reader == nullptr) {
    Fail("cannot open " + args.model.string() + ": " + error);
  }
  auto weights = ml::ModelWeights::Bind(*reader, &error);
  if (!weights) {
    Fail("weights: " + error);
  }
  auto tokenizer = ml::Tokenizer::FromGguf(*reader, &error);
  if (tokenizer == nullptr) {
    Fail("tokenizer: " + error);
  }
  std::vector<std::int32_t> tokens{2};
  for (const std::string& path : args.texts) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
      Fail("cannot read " + path);
    }
    std::stringstream text;
    text << file.rdbuf();
    for (const auto id : tokenizer->Encode(text.str(), {false, false})) {
      tokens.push_back(static_cast<std::int32_t>(id));
    }
  }
  const std::uint32_t max_w = 32;
  const std::uint32_t context = args.depths.back() + max_w;
  if (tokens.size() < context) {
    Fail("the texts give " + std::to_string(tokens.size()) +
         " tokens, the deepest depth needs " + std::to_string(context));
  }
  std::cout << "tokens " << tokens.size() << ", context " << context << "\n";

  const auto load_start = std::chrono::steady_clock::now();
  auto model = gpu::DeviceModel::Upload(*weights, reader.get(), &error);
  if (model == nullptr) {
    Fail("upload: " + error);
  }
  gpu::ExecutorOptions vopt;
  vopt.context = context;
  vopt.max_chunk = 32;
  vopt.path = gpu::Path::kVector;
  auto vec = gpu::Executor::Create(*model, vopt, &error);
  if (vec == nullptr) {
    Fail("vector executor: " + error);
  }
  std::unique_ptr<gpu::Executor> bat;
  if (!args.batched.empty()) {
    gpu::ExecutorOptions bopt;
    bopt.context = context;
    bopt.max_chunk = 512;
    bopt.path = gpu::Path::kBatched;
    bat = gpu::Executor::Create(*model, bopt, &error);
    if (bat == nullptr) {
      Fail("batched executor: " + error);
    }
  }
  std::cout << "loaded in " << Seconds(load_start) << " s\n" << std::flush;

  std::ofstream csv(args.out);
  if (!csv) {
    Fail("cannot write " + args.out.string());
  }
  csv << "depth,path,w,rep,ms\n";
  std::vector<Arm> arms{{"vector", vec.get(), 32, args.widths}};
  if (bat) {
    arms.push_back({"batched", bat.get(), 512, args.batched});
  }
  std::vector<float> logits(std::size_t{max_w} * weights->config.vocab_size);
  for (const std::uint32_t depth : args.depths) {
    for (const Arm& arm : arms) {
      const auto prefill_start = std::chrono::steady_clock::now();
      PrefillTo(*arm.ex, tokens, depth, arm.chunk);
      const double prefill_s = Seconds(prefill_start);
      gpu::ExecutorSnapshot snap;
      if (!arm.ex->SaveSnapshot(&snap, &error)) {
        Fail("snapshot: " + error);
      }
      std::cout << "depth " << depth << " " << arm.name << ": prefill "
                << prefill_s << " s (chunk " << arm.chunk << "), snapshot "
                << snap.bytes() << " B\n"
                << std::flush;
      std::map<std::uint32_t, std::vector<double>> times;
      const std::size_t n = arm.widths.size();
      for (std::uint32_t rep = 0; rep < args.warmup + args.reps; ++rep) {
        for (std::size_t k = 0; k < n; ++k) {
          const std::uint32_t w = arm.widths[(k + rep) % n];
          if (!arm.ex->RestoreSnapshot(snap, &error)) {
            Fail("restore: " + error);
          }
          (void)hipDeviceSynchronize();
          const auto start = std::chrono::steady_clock::now();
          if (!arm.ex->Forward(std::span(tokens).subspan(depth, w),
                               gpu::LogitsRows::kAll, logits.data(), &error)) {
            Fail("forward: " + error);
          }
          const double ms = Seconds(start) * 1e3;
          for (std::size_t i = 0;
               i < std::size_t{w} * weights->config.vocab_size; ++i) {
            if (!std::isfinite(logits[i])) {
              Fail("nonfinite logit");
            }
          }
          if (rep >= args.warmup) {
            times[w].push_back(ms);
            csv << depth << ',' << arm.name << ',' << w << ','
                << rep - args.warmup << ',' << std::setprecision(6) << ms
                << '\n';
          }
        }
      }
      csv << std::flush;
      // Leave the executor at `depth` for the next, deeper prefill.
      if (!arm.ex->RestoreSnapshot(snap, &error)) {
        Fail("restore: " + error);
      }
      const auto median = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v.size() % 2 == 1
                   ? v[v.size() / 2]
                   : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
      };
      const double base = times.count(1) != 0 ? median(times[1]) : 0.0;
      for (auto& [w, v] : times) {
        const double m = median(v);
        const auto [lo, hi] = std::minmax_element(v.begin(), v.end());
        std::cout << std::fixed << std::setprecision(2) << "  " << arm.name
                  << " d=" << depth << " w=" << w << ": median " << m
                  << " ms [min " << *lo << ", max " << *hi << "]";
        if (base > 0.0) {
          std::cout << " x" << std::setprecision(3) << m / base;
        }
        std::cout << "\n";
      }
      std::cout << std::flush;
    }
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return Run(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << "laguna_verify_cost: " << e.what() << "\n";
    return 1;
  }
}
