#ifndef GUFO_TESTS_MODELS_LAGUNA_SUPPORT_HPP_
#define GUFO_TESTS_MODELS_LAGUNA_SUPPORT_HPP_

// Shared helpers for the Laguna tests: locating the real GGUF (and skipping
// loudly without it), and a metadata-only GGUF writer for the model-free
// checks.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace gufo::testing::laguna {

/// Exit status CTest reports as "Skipped" (SKIP_RETURN_CODE in CMake).
inline constexpr int kSkipExitCode = 77;

inline void Require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

/// GUFO_LAGUNA_GGUF overrides the default MasterServer location.
inline std::filesystem::path ModelPath() {
  if (const char* env = std::getenv("GUFO_LAGUNA_GGUF");
      env != nullptr && *env != '\0') {
    return env;
  }
  return "/server/ai/models/Myric/Laguna-S-2.1-APEX-GGUF/"
         "Laguna-S-2.1-APEX-i-quality-v2.gguf";
}

/// Opens the real GGUF (memory-mapped, headers only) or ends the process with
/// the CTest skip status and an unmistakable message. A skip is not a pass.
inline std::unique_ptr<core::GgufReader> OpenModelOrSkip(const char* test) {
  const auto path = ModelPath();
  std::error_code ec;
  if (!std::filesystem::exists(path, ec)) {
    std::cout << "SKIP: " << test << " needs the Laguna-S-2.1 GGUF at "
              << path.string()
              << " (set GUFO_LAGUNA_GGUF); nothing was verified\n";
    std::exit(kSkipExitCode);
  }
  std::string error;
  auto reader = core::GgufReader::OpenFile(path, &error);
  Require(reader != nullptr, "GGUF exists but does not open: " + error);
  return reader;
}

using MetaValue =
    std::variant<std::uint32_t, float, bool, std::string,
                 std::vector<std::uint32_t>, std::vector<std::string>>;
using Metadata = std::map<std::string, MetaValue>;

/// Serializes `fields` as a tensorless GGUF v3 image and opens it in memory.
inline std::unique_ptr<core::GgufReader> OpenSynthetic(
    const Metadata& fields, std::vector<std::uint8_t>* storage) {
  auto& bytes = *storage;
  bytes.clear();
  const auto pod = [&]<class T>(T value) {
    const auto begin = bytes.size();
    bytes.resize(begin + sizeof(value));
    std::memcpy(bytes.data() + begin, &value, sizeof(value));
  };
  const auto text = [&](const std::string& value) {
    pod(std::uint64_t{value.size()});
    bytes.insert(bytes.end(), value.begin(), value.end());
  };
  pod(std::uint32_t{0x46554747});
  pod(std::uint32_t{3});
  pod(std::uint64_t{0});
  pod(static_cast<std::uint64_t>(fields.size()));
  for (const auto& [key, value] : fields) {
    text(key);
    std::visit(
        [&](const auto& item) {
          using T = std::decay_t<decltype(item)>;
          if constexpr (std::is_same_v<T, std::uint32_t>) {
            pod(std::uint32_t{4});
            pod(item);
          } else if constexpr (std::is_same_v<T, float>) {
            pod(std::uint32_t{6});
            pod(item);
          } else if constexpr (std::is_same_v<T, bool>) {
            pod(std::uint32_t{7});
            pod(static_cast<std::uint8_t>(item ? 1 : 0));
          } else if constexpr (std::is_same_v<T, std::string>) {
            pod(std::uint32_t{8});
            text(item);
          } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
            pod(std::uint32_t{9});
            pod(std::uint32_t{8});
            pod(static_cast<std::uint64_t>(item.size()));
            for (const auto& entry : item) {
              text(entry);
            }
          } else {
            pod(std::uint32_t{9});
            pod(std::uint32_t{4});
            pod(static_cast<std::uint64_t>(item.size()));
            for (const auto entry : item) {
              pod(entry);
            }
          }
        },
        value);
  }
  bytes.resize((bytes.size() + 31) / 32 * 32);
  std::string error;
  auto reader =
      core::GgufReader::OpenMemory(bytes.data(), bytes.size(), &error);
  Require(reader != nullptr, "synthetic GGUF does not open: " + error);
  return reader;
}

/// Metadata of Laguna-S-2.1 exactly as the real GGUF records it (minus the
/// vocabulary arrays), the base for every mutation.
inline Metadata ValidLagunaMetadata() {
  Metadata m;
  m["general.architecture"] = std::string{"laguna"};
  const auto u32 = [&](const char* key, std::uint32_t value) {
    m[std::string{"laguna."} + key] = value;
  };
  const auto f32 = [&](const char* key, float value) {
    m[std::string{"laguna."} + key] = value;
  };
  u32("block_count", 48);
  u32("context_length", 262144);
  u32("embedding_length", 3072);
  u32("feed_forward_length", 12288);
  std::vector<std::uint32_t> heads;
  for (std::uint32_t il = 0; il < 48; ++il) {
    heads.push_back(il % 4 == 0 ? 48 : 72);
  }
  m["laguna.attention.head_count"] = heads;
  u32("attention.head_count_kv", 8);
  m["laguna.rope.scaling.type"] = std::string{"yarn"};
  f32("rope.scaling.factor", 32.0F);
  u32("rope.scaling.original_context_length", 8192);
  f32("rope.scaling.yarn_attn_factor", 1.0F);
  f32("rope.scaling.yarn_beta_fast", 32.0F);
  f32("rope.scaling.yarn_beta_slow", 1.0F);
  f32("rope.freq_base", 500000.0F);
  f32("rope.freq_base_swa", 10000.0F);
  f32("attention.layer_norm_rms_epsilon", 1e-6F);
  u32("expert_count", 256);
  u32("expert_used_count", 10);
  u32("attention.key_length", 128);
  u32("attention.value_length", 128);
  u32("vocab_size", 100352);
  u32("attention.sliding_window", 512);
  u32("expert_feed_forward_length", 1024);
  u32("expert_shared_feed_forward_length", 1024);
  m["laguna.expert_weights_norm"] = true;
  f32("expert_weights_scale", 2.5F);
  u32("expert_gating_func", 2);
  u32("leading_dense_block_count", 1);
  u32("rope.dimension_count", 64);
  u32("rope.dimension_count_swa", 128);
  return m;
}

}  // namespace gufo::testing::laguna

#endif  // GUFO_TESTS_MODELS_LAGUNA_SUPPORT_HPP_
