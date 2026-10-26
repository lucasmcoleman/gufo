// laguna_oracle: runs the CPU reference forward on the real Laguna-S-2.1 GGUF
// for a token list and writes the final logits plus llama.cpp-named taps.
//
//   laguna_oracle --tokens 2,1234,5678 --out DIR [--gguf PATH]
//                 [--act-quant none|cpu|gpu] [--taps name,name]
//                 [--hf-denominator] [--kv-f16] [--last-row-only]
//                 [--threads N]
//
// The model is 74 GB quantized and the oracle reads all of it through the
// mmap, so the tool refuses to start while a `llama_logits` process runs or
// while MemAvailable is under 90 GiB. There is no override.
#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "src/models/laguna/reference.hpp"
#include "tests/models/laguna/support.hpp"

namespace laguna = gufo::models::laguna;
namespace ref = gufo::models::laguna::reference;

namespace {

constexpr std::uint64_t kMinAvailableKiB = 90ULL * 1024 * 1024;

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

/// Returns an empty string when the memory/GPU-job guard passes.
std::string MemoryGuard() {
  std::error_code ec;
  for (const auto& entry : std::filesystem::directory_iterator("/proc", ec)) {
    const std::string name = entry.path().filename().string();
    if (name.empty() ||
        !std::all_of(name.begin(), name.end(),
                     [](unsigned char ch) { return std::isdigit(ch) != 0; })) {
      continue;
    }
    std::ifstream comm(entry.path() / "comm");
    std::string text;
    std::getline(comm, text);
    if (text == "llama_logits") {
      return "a llama_logits process (pid " + name + ") is running";
    }
  }
  std::ifstream meminfo("/proc/meminfo");
  std::string key;
  std::uint64_t value = 0;
  std::string unit;
  while (meminfo >> key >> value >> unit) {
    if (key == "MemAvailable:") {
      if (value < kMinAvailableKiB) {
        return "MemAvailable is " + std::to_string(value / 1024 / 1024) +
               " GiB, under the 90 GiB floor";
      }
      return "";
    }
  }
  return "cannot read MemAvailable from /proc/meminfo";
}

int Usage(const char* why) {
  if (why != nullptr) {
    std::cerr << "laguna_oracle: " << why << "\n";
  }
  std::cerr
      << "usage: laguna_oracle --tokens ID,ID,... --out DIR [--gguf "
         "PATH] [--act-quant none|cpu|gpu] [--taps SUBSTR,...] "
         "[--hf-denominator] [--kv-f16] [--last-row-only] [--threads N]\n";
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::int32_t> tokens;
  std::filesystem::path out_dir;
  std::filesystem::path gguf = gufo::testing::laguna::ModelPath();
  std::vector<std::string> tap_filter;
  ref::Options options;
  bool help = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto value = [&]() -> const char* {
      return i + 1 < argc ? argv[++i] : nullptr;
    };
    if (arg == "--help") {
      help = true;
    } else if (arg == "--tokens") {
      const char* v = value();
      if (v == nullptr) {
        return Usage("--tokens needs a value");
      }
      for (const auto& part : Split(v)) {
        std::int32_t id = 0;
        const auto [end, ec] =
            std::from_chars(part.data(), part.data() + part.size(), id);
        if (ec != std::errc{} || end != part.data() + part.size() || id < 0) {
          return Usage("bad token id");
        }
        tokens.push_back(id);
      }
    } else if (arg == "--out") {
      const char* v = value();
      if (v == nullptr) {
        return Usage("--out needs a value");
      }
      out_dir = v;
    } else if (arg == "--gguf") {
      const char* v = value();
      if (v == nullptr) {
        return Usage("--gguf needs a value");
      }
      gguf = v;
    } else if (arg == "--act-quant") {
      const char* v = value();
      const std::string mode = v != nullptr ? v : "";
      if (mode == "none") {
        options.activation_quant = ref::ActivationQuant::kNone;
      } else if (mode == "cpu") {
        options.activation_quant = ref::ActivationQuant::kCpu;
      } else if (mode == "gpu") {
        options.activation_quant = ref::ActivationQuant::kGpuQ8_1;
      } else {
        return Usage("--act-quant must be none, cpu or gpu");
      }
    } else if (arg == "--taps") {
      const char* v = value();
      if (v == nullptr) {
        return Usage("--taps needs a value");
      }
      tap_filter = Split(v);
    } else if (arg == "--hf-denominator") {
      options.router_denominator = ref::RouterDenominator::kHfRaw;
    } else if (arg == "--kv-f16") {
      options.round_kv_f16 = true;
    } else if (arg == "--last-row-only") {
      options.last_row_logits_only = true;
    } else if (arg == "--threads") {
      const char* v = value();
      options.threads = v != nullptr ? static_cast<unsigned>(std::atoi(v)) : 0;
    } else {
      return Usage(("unknown argument " + arg).c_str());
    }
  }
  if (help) {
    Usage(nullptr);
    return 0;
  }
  if (tokens.empty() || out_dir.empty()) {
    return Usage("--tokens and --out are required");
  }
  if (const std::string blocked = MemoryGuard(); !blocked.empty()) {
    std::cerr << "laguna_oracle: refusing to start: " << blocked
              << " (the oracle maps the whole 74 GB model)\n";
    return 3;
  }

  std::string error;
  const auto reader = gufo::core::GgufReader::OpenFile(gguf, &error);
  if (reader == nullptr) {
    std::cerr << "laguna_oracle: cannot open " << gguf << ": " << error << "\n";
    return 1;
  }
  const auto weights = laguna::ModelWeights::Bind(*reader, &error);
  if (!weights) {
    std::cerr << "laguna_oracle: bind failed: " << error << "\n";
    return 1;
  }

  ref::TapWriter writer(out_dir, tap_filter);
  const ref::Model model(*weights, options);
  std::vector<float> logits;
  try {
    logits = model.Forward(tokens, std::ref(writer));
  } catch (const std::exception& e) {
    std::cerr << "laguna_oracle: forward failed: " << e.what() << "\n";
    return 1;
  }
  if (!writer.Finish(&error)) {
    std::cerr << "laguna_oracle: tap write failed: " << error << "\n";
    return 1;
  }

  const std::size_t vocab = weights->output.rows;
  const float* last = logits.data() + logits.size() - vocab;
  {
    std::ofstream file(out_dir / "logits.f32", std::ios::binary);
    file.write(reinterpret_cast<const char*>(last),
               static_cast<std::streamsize>(vocab * sizeof(float)));
  }
  std::vector<std::uint32_t> order(vocab);
  for (std::uint32_t i = 0; i < vocab; ++i) {
    order[i] = i;
  }
  std::partial_sort(
      order.begin(), order.begin() + 5, order.end(),
      [&](std::uint32_t a, std::uint32_t b) { return last[a] > last[b]; });
  std::cout << "tokens=" << tokens.size() << " taps=" << writer.written()
            << " out=" << out_dir.string() << "\ntop5 (final row):\n";
  for (int i = 0; i < 5; ++i) {
    std::cout << "  id=" << order[i] << " logit=" << last[order[i]] << "\n";
  }
  return 0;
}
