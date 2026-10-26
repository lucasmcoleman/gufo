#ifndef GUFO_TESTS_MODELS_LAGUNA_LOGITS_DUMP_HPP_
#define GUFO_TESTS_MODELS_LAGUNA_LOGITS_DUMP_HPP_

// The codexms #969 logit-dump contract (see logits_cli.cpp): reading a
// reference directory and writing the matching gufo directory. Shared by the
// laguna_logits CLI and the executor test, which drives it with a tiny model
// and the real scorer. Every error throws std::runtime_error.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#include "src/core/json.hpp"
#include "src/models/laguna/kernels/executor.hpp"

namespace gufo::testing::laguna {

namespace fs = std::filesystem;

struct SeqInfo {
  std::string file;
  std::size_t n_tokens{0};
  bool tail{false};
  bool has_tail{false};
};

struct Reference {
  std::uint32_t T{0};
  std::uint32_t n_vocab{0};
  std::vector<SeqInfo> seqs;
  std::vector<std::int32_t> tokens;  // [S][T]
};

[[noreturn]] inline void DumpFail(const std::string& message) {
  throw std::runtime_error(message);
}

inline std::string ReadText(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    DumpFail("cannot read " + path.string());
  }
  std::stringstream text;
  text << in.rdbuf();
  return text.str();
}

inline Reference ReadReference(const fs::path& dir) {
  Reference ref;
  if (fs::exists(dir / ".incomplete")) {
    DumpFail(dir.string() + " has an .incomplete marker");
  }
  gufo::json::Value index;
  try {
    index = gufo::json::parse(ReadText(dir / "index.json"));
  } catch (const std::exception& e) {
    DumpFail("reference index.json is not valid JSON: " +
             std::string(e.what()));
  }
  const auto* complete = index.find("complete");
  if (complete != nullptr && complete->is_bool() && !complete->as_bool()) {
    DumpFail("reference manifest is explicitly incomplete");
  }
  const auto* tokenize_only = index.find("tokenize_only");
  if (tokenize_only != nullptr && tokenize_only->is_bool() &&
      tokenize_only->as_bool()) {
    DumpFail("reference manifest is tokenize-only");
  }
  ref.T = static_cast<std::uint32_t>(index.member_size("T"));
  ref.n_vocab = static_cast<std::uint32_t>(index.member_size("n_vocab"));
  const auto* seqs = index.find("seqs");
  if (ref.T < 2 || ref.n_vocab < 2 || seqs == nullptr || !seqs->is_array() ||
      seqs->items().empty()) {
    DumpFail("reference index.json needs T, n_vocab and a nonempty seqs list");
  }
  for (const auto& item : seqs->items()) {
    SeqInfo s;
    s.file = item.member_str("file");
    s.n_tokens = item.member_size("n_tokens");
    if (const auto* tail = item.find("tail");
        tail != nullptr && tail->is_bool()) {
      s.tail = tail->as_bool();
      s.has_tail = true;
    }
    if (s.file.empty()) {
      DumpFail("a reference sequence has no file name");
    }
    ref.seqs.push_back(std::move(s));
  }
  const std::string raw = ReadText(dir / "tokens.i32");
  const std::size_t expect = ref.seqs.size() * ref.T * sizeof(std::int32_t);
  if (raw.size() != expect) {
    DumpFail("reference tokens.i32 has " + std::to_string(raw.size()) +
             " bytes, expected " + std::to_string(expect));
  }
  ref.tokens.resize(ref.seqs.size() * ref.T);
  std::memcpy(ref.tokens.data(), raw.data(), raw.size());  // little endian host
  for (const std::int32_t t : ref.tokens) {
    if (t < 0 || static_cast<std::uint32_t>(t) >= ref.n_vocab) {
      DumpFail("reference token outside the vocabulary");
    }
  }
  return ref;
}

inline void WriteAll(const fs::path& path, const void* data,
                     std::size_t bytes) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(static_cast<const char*>(data),
            static_cast<std::streamsize>(bytes));
  out.close();
  if (!out) {
    DumpFail("writing " + path.string() + " failed");
  }
}

struct DumpOptions {
  fs::path output;     ///< Must be new or empty.
  fs::path reference;  ///< Recorded in the manifest.
  std::string model;   ///< Recorded in the manifest.
  std::uint32_t ubatch{256};
  std::uint32_t context{0};
  std::size_t sequences{0};  ///< Run the first N (0: all).
  /// Test hook: called on each chunk's logits before the finite check.
  std::function<void(float* logits, std::size_t count)> test_hook;
};

/// Runs each sequence from position 0 on a cleared cache in chunks of
/// `ubatch`, writes NN.f32, tokens.i32 (the tokens actually executed) and,
/// last, index.json (by rename); a `.incomplete` marker exists until then.
/// Fails on nonfinite logits and any I/O or executor error.
inline void DumpLogits(const Reference& ref,
                       gufo::models::laguna::rocm::Executor& ex,
                       const DumpOptions& o, std::ostream* log = nullptr) {
  namespace gpu = gufo::models::laguna::rocm;
  const std::size_t S = o.sequences == 0
                            ? ref.seqs.size()
                            : std::min(o.sequences, ref.seqs.size());
  std::error_code ec;
  fs::create_directories(o.output, ec);
  if (ec || !fs::is_directory(o.output) ||
      fs::directory_iterator(o.output) != fs::directory_iterator()) {
    DumpFail(o.output.string() + " must be a new or empty directory");
  }
  const fs::path marker = o.output / ".incomplete";
  WriteAll(marker, "laguna_logits run in progress\n", 30);
  const std::size_t T = ref.T;
  const std::size_t V = ref.n_vocab;
  std::vector<std::int32_t> executed(S * T);
  std::vector<float> logits(static_cast<std::size_t>(o.ubatch) * V);
  gufo::json::Value seqs = gufo::json::Value::array();
  std::string error;
  for (std::size_t s = 0; s < S; ++s) {
    const auto start = std::chrono::steady_clock::now();
    char name[16];
    std::snprintf(name, sizeof(name), "%02zu.f32", s);
    std::ofstream out(o.output / name, std::ios::binary | std::ios::trunc);
    if (!out) {
      DumpFail(std::string("cannot create ") + name);
    }
    if (!ex.Reset(&error)) {
      DumpFail("cache reset failed: " + error);
    }
    std::copy_n(ref.tokens.begin() + static_cast<std::ptrdiff_t>(s * T), T,
                executed.begin() + static_cast<std::ptrdiff_t>(s * T));
    for (std::size_t pos = 0; pos < T; pos += o.ubatch) {
      const std::size_t n = std::min<std::size_t>(o.ubatch, T - pos);
      if (!ex.Forward(
              std::span<const std::int32_t>(executed.data() + s * T + pos, n),
              gpu::LogitsRows::kAll, logits.data(), &error)) {
        DumpFail("sequence " + std::to_string(s) + " position " +
                 std::to_string(pos) + ": " + error);
      }
      if (o.test_hook) {
        o.test_hook(logits.data(), n * V);
      }
      for (std::size_t i = 0; i < n * V; ++i) {
        if (!std::isfinite(logits[i])) {
          DumpFail("nonfinite logit in sequence " + std::to_string(s) +
                   " at position " + std::to_string(pos + i / V) + ", token " +
                   std::to_string(i % V));
        }
      }
      out.write(reinterpret_cast<const char*>(logits.data()),
                static_cast<std::streamsize>(n * V * sizeof(float)));
    }
    out.close();
    if (!out) {
      DumpFail(std::string("writing ") + name + " failed");
    }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    gufo::json::Value entry = gufo::json::Value::object();
    entry["file"] = ref.seqs[s].file;
    entry["n_tokens"] = static_cast<long long>(ref.seqs[s].n_tokens);
    if (ref.seqs[s].has_tail) {
      entry["tail"] = ref.seqs[s].tail;
    }
    entry["logits"] = std::string(name);
    entry["prefill_and_write_s"] = seconds;
    seqs.push_back(std::move(entry));
    if (log != nullptr) {
      *log << "sequence " << (s + 1) << "/" << S << " (" << ref.seqs[s].file
           << ") " << seconds << " s\n";
    }
  }
  WriteAll(o.output / "tokens.i32", executed.data(),
           executed.size() * sizeof(std::int32_t));
  gufo::json::Value index = gufo::json::Value::object();
  index["complete"] = true;
  index["engine"] = "gufo laguna executor";
  index["model"] = o.model;
  index["n_vocab"] = static_cast<long long>(V);
  index["T"] = static_cast<long long>(T);
  index["ubatch"] = static_cast<long long>(o.ubatch);
  index["context"] = static_cast<long long>(o.context);
  index["tokenize_only"] = false;
  index["reference"] = o.reference.string();
  index["seqs"] = std::move(seqs);
  const std::string text = index.dump() + "\n";
  WriteAll(o.output / "index.json.tmp", text.data(), text.size());
  fs::rename(o.output / "index.json.tmp", o.output / "index.json", ec);
  if (ec) {
    DumpFail("publishing index.json failed: " + ec.message());
  }
  fs::remove(marker, ec);
}

}  // namespace gufo::testing::laguna

#endif  // GUFO_TESTS_MODELS_LAGUNA_LOGITS_DUMP_HPP_
