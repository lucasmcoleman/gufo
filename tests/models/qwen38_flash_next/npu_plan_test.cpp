// CPU checks of the NPU planning helpers (src/models/qwen38_flash_next/
// npu_plan.hpp) and of the shipped instruction streams under tools/npu. No
// NPU, XRT or GPU is touched: the streams are read as data.

#include "src/models/qwen38_flash_next/npu_plan.hpp"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace qfn = gufo::models::qwen38_flash_next;

namespace {

void Require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

std::vector<std::uint32_t> ReadWords(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  Require(static_cast<bool>(file), "missing instruction stream " + path);
  const std::vector<char> bytes((std::istreambuf_iterator<char>(file)),
                                std::istreambuf_iterator<char>());
  Require(!bytes.empty() && bytes.size() % 4 == 0,
          "instruction stream is empty or not word aligned: " + path);
  std::vector<std::uint32_t> words(bytes.size() / 4);
  for (std::size_t i = 0; i < words.size(); ++i) {
    words[i] =
        static_cast<std::uint32_t>(static_cast<unsigned char>(bytes[4 * i])) |
        static_cast<std::uint32_t>(static_cast<unsigned char>(bytes[4 * i + 1]))
            << 8 |
        static_cast<std::uint32_t>(static_cast<unsigned char>(bytes[4 * i + 2]))
            << 16 |
        static_cast<std::uint32_t>(static_cast<unsigned char>(bytes[4 * i + 3]))
            << 24;
  }
  return words;
}

void CheckSwitches() {
  Require(!qfn::EnvSwitchOn(nullptr), "unset switch is on");
  Require(!qfn::EnvSwitchOn(""), "empty switch is on");
  Require(!qfn::EnvSwitchOn("0"), "0 is on");
  Require(!qfn::EnvSwitchOn("00"), "00 is on");
  Require(qfn::EnvSwitchOn("1"), "1 is off");
  Require(qfn::EnvSwitchOn("on"), "on is off");
}

void CheckProgramParsing() {
  Require(qfn::ParseNpuProgram(nullptr) == qfn::NpuProgram::kFixed,
          "unset program is not the fixed one");
  Require(qfn::ParseNpuProgram("") == qfn::NpuProgram::kFixed,
          "empty program is not the fixed one");
  Require(qfn::ParseNpuProgram("fixed") == qfn::NpuProgram::kFixed,
          "fixed not parsed");
  Require(qfn::ParseNpuProgram("runtime_k") == qfn::NpuProgram::kRuntimeK,
          "runtime_k not parsed");
  Require(qfn::ParseNpuProgram("runtime_mk") == qfn::NpuProgram::kRuntimeMK,
          "runtime_mk not parsed");
  for (const char* bad : {"runtime", "RUNTIME_K", "1", "runtime_k ", "mk"}) {
    Require(!qfn::ParseNpuProgram(bad).has_value(),
            std::string("accepted program ") + bad);
  }
}

void CheckPaths() {
  // The default must name exactly the PR #568 pair the live build loads.
  const auto fixed =
      qfn::NpuProgramPaths("/d", qfn::NpuProgram::kFixed, 4096, 2560);
  Require(fixed.has_value(), "fixed 4096x2560 rejected");
  Require(fixed->xclbin == "/d/final_4096x2560x1024_128x64x128.xclbin",
          "fixed xclbin path changed: " + fixed->xclbin);
  Require(fixed->insts == "/d/insts_4096x2560x1024_128x64x128.txt",
          "fixed insts path changed: " + fixed->insts);
  Require(!qfn::NpuProgramPaths("/d", qfn::NpuProgram::kFixed, 4096, 2048),
          "fixed program accepted K=2048");
  Require(!qfn::NpuProgramPaths("/d", qfn::NpuProgram::kFixed, 2048, 2560),
          "fixed program accepted M=2048");

  const auto rk =
      qfn::NpuProgramPaths("/d", qfn::NpuProgram::kRuntimeK, 4096, 5120);
  Require(rk.has_value(), "runtime_k 4096x5120 rejected");
  Require(rk->xclbin == "/d/runtime_k/final_4096xKx1024_128x64x128.xclbin",
          "runtime_k xclbin path: " + rk->xclbin);
  Require(rk->insts == "/d/runtime_k/insts_4096x5120x1024_128x64x128.txt",
          "runtime_k insts path: " + rk->insts);
  Require(!qfn::NpuProgramPaths("/d", qfn::NpuProgram::kRuntimeK, 4096, 2600),
          "runtime_k accepted K not a multiple of 64");
  Require(!qfn::NpuProgramPaths("/d", qfn::NpuProgram::kRuntimeK, 2048, 2560),
          "runtime_k accepted M != 4096");
  Require(!qfn::NpuProgramPaths("/d", qfn::NpuProgram::kRuntimeK, 4096, 0),
          "runtime_k accepted K=0");

  const auto rmk =
      qfn::NpuProgramPaths("/d", qfn::NpuProgram::kRuntimeMK, 1536, 2048);
  Require(rmk.has_value(), "runtime_mk 1536x2048 rejected");
  Require(rmk->xclbin == "/d/runtime_mk/final_4096xKx1024_128x64x128.xclbin",
          "runtime_mk xclbin path: " + rmk->xclbin);
  Require(rmk->insts == "/d/runtime_mk/insts_1536x2048x1024_128x64x128.txt",
          "runtime_mk insts path: " + rmk->insts);
  for (const std::uint32_t m : {0U, 1000U, 4608U, 8192U}) {
    Require(!qfn::NpuProgramPaths("/d", qfn::NpuProgram::kRuntimeMK, m, 2048),
            "runtime_mk accepted M=" + std::to_string(m));
  }
}

/// Every runtime-K stream writes K / 64 into word 0 of all 32 cores' rtp
/// buffers before the DMA program (one 12-word write and lock per core).
bool RuntimeKWordsMatch(const std::vector<std::uint32_t>& words,
                        std::uint32_t k) {
  if (words.size() <= 8 + 12 * 32) {
    return false;
  }
  for (std::size_t core = 0; core < 32; ++core) {
    if (words[8 + 12 * core] != k / 64) {
      return false;
    }
  }
  return true;
}

/// Runtime-M/K streams write K / 64 and the per-core output tile count
/// (M / 128 * 1024 / 128 / 32 = M / 512) into the two rtp words.
bool RuntimeMkWordsMatch(const std::vector<std::uint32_t>& words,
                         std::uint32_t m, std::uint32_t k) {
  if (words.size() <= 14 + 18 * 32) {
    return false;
  }
  for (std::size_t core = 0; core < 32; ++core) {
    if (words[8 + 18 * core] != k / 64 || words[14 + 18 * core] != m / 512) {
      return false;
    }
  }
  return true;
}

/// A stream whose file name and contents disagree is caught here. The
/// negative cases keep the predicate honest: a K=2560 stream must not pass
/// as K=2048, and an M=1024 stream must not pass as M=1536.
void CheckRuntimeKStreams(const std::string& npu_dir) {
  for (const std::uint32_t k : {2048U, 2560U, 4096U, 5120U, 6144U, 17408U}) {
    const auto files =
        qfn::NpuProgramPaths(npu_dir, qfn::NpuProgram::kRuntimeK, 4096, k);
    Require(files.has_value(), "runtime_k path rejected");
    Require(std::filesystem::exists(files->xclbin), "missing " + files->xclbin);
    Require(RuntimeKWordsMatch(ReadWords(files->insts), k),
            "runtime_k K=" + std::to_string(k) + " rtp words disagree");
  }
  const auto k2560 =
      qfn::NpuProgramPaths(npu_dir, qfn::NpuProgram::kRuntimeK, 4096, 2560);
  Require(!RuntimeKWordsMatch(ReadWords(k2560->insts), 2048),
          "negative: a K=2560 stream passed as K=2048");
  // The fixed PR #568 stream has no rtp writes at all.
  const auto fixed =
      qfn::NpuProgramPaths(npu_dir, qfn::NpuProgram::kFixed, 4096, 2560);
  const auto fixed_words = ReadWords(fixed->insts);
  Require(!RuntimeKWordsMatch(fixed_words, 2560),
          "negative: the fixed stream passed as a runtime-K stream");

  // At K = 2560 the runtime-K stream is the fixed stream's DMA program behind
  // a 384-word prologue (32 cores x 12 words of rtp write and lock release):
  // both programs move exactly the same data, only the cores' K source
  // differs. A one-word shift must not match.
  const auto k2560_words = ReadWords(k2560->insts);
  constexpr std::size_t kHeader = 4;
  constexpr std::size_t kPrologue = 32 * 12;
  const auto body_matches = [&](std::size_t shift) {
    if (k2560_words.size() != fixed_words.size() + kPrologue) {
      return false;
    }
    for (std::size_t i = kHeader; i + shift < fixed_words.size(); ++i) {
      if (fixed_words[i + shift] != k2560_words[i + kPrologue]) {
        return false;
      }
    }
    return true;
  };
  Require(body_matches(0),
          "runtime_k K=2560 DMA program differs from the fixed program");
  Require(!body_matches(1), "negative: a shifted DMA program matched");
}

void CheckRuntimeMkStreams(const std::string& npu_dir) {
  struct Shape {
    std::uint32_t m;
    std::uint32_t k;
  };
  std::vector<Shape> shapes;
  for (std::uint32_t m = 1024; m <= 4096; m += 512) {
    shapes.push_back({m, 2048});
  }
  shapes.push_back({4096, 4096});
  for (const Shape& s : shapes) {
    const auto files =
        qfn::NpuProgramPaths(npu_dir, qfn::NpuProgram::kRuntimeMK, s.m, s.k);
    Require(files.has_value(), "runtime_mk path rejected");
    Require(std::filesystem::exists(files->xclbin), "missing " + files->xclbin);
    Require(RuntimeMkWordsMatch(ReadWords(files->insts), s.m, s.k),
            "runtime_mk " + std::to_string(s.m) + "x" + std::to_string(s.k) +
                " rtp words disagree");
  }
  const auto m1024 =
      qfn::NpuProgramPaths(npu_dir, qfn::NpuProgram::kRuntimeMK, 1024, 2048);
  Require(!RuntimeMkWordsMatch(ReadWords(m1024->insts), 1536, 2048),
          "negative: an M=1024 stream passed as M=1536");
  Require(!RuntimeMkWordsMatch(ReadWords(m1024->insts), 1024, 4096),
          "negative: a K=2048 stream passed as K=4096");
  // Flash-Next's K = 2560 has no runtime-M/K stream yet; the program must be
  // generated before GUFO_NPU_PROGRAM=runtime_mk can serve it.
  const auto flash =
      qfn::NpuProgramPaths(npu_dir, qfn::NpuProgram::kRuntimeMK, 4096, 2560);
  Require(flash.has_value() && !std::filesystem::exists(flash->insts),
          "runtime_mk 4096x2560 stream now exists: update NPU-ROUND notes");
}

/// Two-bank split (mikealanni/gufo 58fac455): whole 4096-token groups, bank
/// A takes the extra group of an odd count, bank B fits max_batch / 2.
void CheckTwoBankSplit() {
  constexpr std::uint32_t kTile16k = 16384 + 128;  // PrefillThroughCapacity
  struct Case {
    std::size_t n;
    std::uint32_t max_batch;
    bool relaxed;
    bool ok;
    std::uint32_t a;
    std::uint32_t b;
  };
  const Case cases[] = {
      {8192, kTile16k, false, true, 4096, 4096},
      {12288, kTile16k, false, true, 8192, 4096},
      {16384, kTile16k, false, true, 8192, 8192},
      {24576, 24576 + 128, false, true, 12288, 12288},
      {20480, 24576 + 128, false, true, 12288, 8192},
      // Too short, not whole groups, wider than the batch, bank B too wide.
      {4096, kTile16k, false, false, 0, 0},
      {8191, kTile16k, false, false, 0, 0},
      {10000, kTile16k, false, false, 0, 0},
      {20480, kTile16k, false, false, 0, 0},
      {8192, 8192 + 128, false, true, 4096, 4096},
      // Bank A (8192) would exceed the 8192 + 128 batch's bank B limit only
      // for B; here B = 4096 fits, so 12288 rows are accepted.
      {12288, 12288 + 128, false, true, 8192, 4096},
      // Bank B (relaxed remainder) wider than half the batch.
      {12000, 12100, true, false, 0, 0},
      // The ordinary 4096-token tile never qualifies.
      {4096 + 128, 4096 + 128, false, false, 0, 0},
      // Relaxed: bank B takes the remainder.
      {10000, kTile16k, true, true, 4096, 5904},
      {12000, kTile16k, true, true, 4096, 7904},
  };
  for (const Case& c : cases) {
    const auto split = qfn::PlanTwoBank(c.n, c.max_batch, c.relaxed);
    const std::string where = "PlanTwoBank(" + std::to_string(c.n) + ", " +
                              std::to_string(c.max_batch) + ")";
    if (!c.ok) {
      Require(!split.has_value(), where + " accepted");
      continue;
    }
    Require(split.has_value(), where + " rejected");
    Require(split->a == c.a && split->b == c.b,
            where + " split " + std::to_string(split->a) + "+" +
                std::to_string(split->b));
    Require(split->a + split->b == c.n, where + " lost tokens");
  }
}

void CheckTwoBankChunking() {
  Require(qfn::TwoBankChunk(16384) == 16384, "16384 rounded");
  Require(qfn::TwoBankChunk(16384 + 128) == 16384, "16512 not rounded down");
  Require(qfn::TwoBankChunk(12288) == 8192, "12288 not rounded down");
  Require(qfn::TwoBankChunk(8192) == 8192, "8192 rounded");
  Require(qfn::TwoBankChunk(4096) == 4096, "4096 rounded");
  Require(qfn::TwoBankChunk(3000) == 3000, "a short chunk rounded");
  Require(qfn::WholeGroupsPerBank(16384 + 128, 4096) == 2,
          "16K tile: two whole groups per bank");
  Require(qfn::WholeGroupsPerBank(24576 + 128, 4096) == 3,
          "24K tile: three whole groups per bank");
  Require(qfn::WholeGroupsPerBank(65536 + 128, 4096) == 3,
          "whole groups per bank not capped at three");
  Require(qfn::WholeGroupsPerBank(4096 + 128, 4096) == 0,
          "the 4096 tile gets whole groups");
  Require(qfn::WholeGroupsPerBank(16384, 0) == 0, "zero group size");
  // Up to a 24576 tile every bank PlanTwoBank produces from an engine chunk
  // fits the staged NPU groups; a wider tile's 16384-row banks do not, and
  // ForwardPair then runs lock-step without the whole-projection mode.
  for (const std::uint32_t tile : {8192U, 16384U, 24576U}) {
    const std::uint32_t max_batch = tile + 128;
    const std::uint32_t groups = qfn::WholeGroupsPerBank(max_batch, 4096);
    for (std::size_t n = 8192; n <= max_batch; n += 4096) {
      const auto split =
          qfn::PlanTwoBank(qfn::TwoBankChunk(n), max_batch, false);
      if (split) {
        Require(split->a / 4096 <= groups && split->b / 4096 <= groups,
                "tile " + std::to_string(tile) + ": bank wider than the " +
                    "staged NPU groups at n=" + std::to_string(n));
      }
    }
  }
}

void CheckWideTileFallsBack() {
  const std::uint32_t max_batch = 32768 + 128;
  const auto split = qfn::PlanTwoBank(32768, max_batch, false);
  Require(split.has_value() && split->a == 16384, "32768 split");
  Require(split->a / 4096 > qfn::WholeGroupsPerBank(max_batch, 4096),
          "a 16384-row bank now fits the staged groups: update the notes");
}

void CheckPrefillTile() {
  Require(qfn::ParsePrefillTile(nullptr, 4096) == 4096, "unset tile");
  Require(qfn::ParsePrefillTile("", 4096) == 4096, "empty tile");
  Require(qfn::ParsePrefillTile("16384", 4096) == 16384, "16384 rejected");
  Require(qfn::ParsePrefillTile("128", 4096) == 128, "128 rejected");
  Require(qfn::ParsePrefillTile("65536", 4096) == 65536, "65536 rejected");
  for (const char* bad :
       {"127", "65537", "16k", "-1", "1e4", " 8192", "99999999999999999999"}) {
    Require(qfn::ParsePrefillTile(bad, 4096) == 4096,
            std::string("accepted tile ") + bad);
  }
}

void CheckFixedPair(const std::string& npu_dir) {
  const auto files =
      qfn::NpuProgramPaths(npu_dir, qfn::NpuProgram::kFixed, 4096, 2560);
  Require(files.has_value() && std::filesystem::exists(files->xclbin) &&
              std::filesystem::exists(files->insts),
          "fixed PR #568 pair missing under tools/npu");
}

}  // namespace

int main() {
  try {
    const std::string npu_dir = std::string(GUFO_SOURCE_DIR) + "/tools/npu";
    CheckSwitches();
    CheckProgramParsing();
    CheckPaths();
    CheckFixedPair(npu_dir);
    CheckRuntimeKStreams(npu_dir);
    CheckRuntimeMkStreams(npu_dir);
    CheckTwoBankSplit();
    CheckTwoBankChunking();
    CheckWideTileFallsBack();
    CheckPrefillTile();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "npu_plan_test: %s\n", e.what());
    return 1;
  }
  std::puts("npu_plan_test: PASS");
  return 0;
}
