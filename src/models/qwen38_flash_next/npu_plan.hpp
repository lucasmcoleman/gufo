#pragma once

// Host-side planning for the XDNA2 NPU path of Qwen3.8 Flash-Next: which
// instruction program to load and how the two-bank prefill splits a chunk.
// Pure functions with no HIP or XRT dependency, so the CPU tests exercise
// exactly the code the device model, executor and engine call.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace gufo::models::qwen38_flash_next {

/// The value of a default-off switch variable: unset, empty or starting with
/// '0' is off; anything else is on.
[[nodiscard]] inline bool EnvSwitchOn(const char* value) noexcept {
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

/// The NPU GEMM program family (all N = 1024, tile 128 x 64 x 128).
enum class NpuProgram {
  kFixed,     ///< PR #568: M = 4096 and K = 2560 compiled into the cores
  kRuntimeK,  ///< tools/npu/runtime_k: K at run time, M = 4096
  kRuntimeMK  ///< tools/npu/runtime_mk: M and K at run time
};

/// GUFO_NPU_PROGRAM: unset, empty or "fixed" selects the PR #568 program;
/// "runtime_k" or "runtime_mk" a runtime program. Anything else is rejected.
[[nodiscard]] inline std::optional<NpuProgram> ParseNpuProgram(
    const char* value) {
  const std::string_view v = value == nullptr ? std::string_view{} : value;
  if (v.empty() || v == "fixed") {
    return NpuProgram::kFixed;
  }
  if (v == "runtime_k") {
    return NpuProgram::kRuntimeK;
  }
  if (v == "runtime_mk") {
    return NpuProgram::kRuntimeMK;
  }
  return std::nullopt;
}

struct NpuProgramFiles {
  std::string xclbin;
  std::string insts;
};

/// The XCLBIN and instruction stream for an M x K x 1024 GEMM of `program`
/// under `dir` (GUFO_NPU_DIR), or nullopt when that program cannot serve the
/// shape. The fixed program serves only 4096 x 2560; runtime-K needs M = 4096
/// and K a multiple of 64; runtime-M/K needs M a multiple of 512 (at most
/// 4096) and K a multiple of 64. Only the instruction files differ per shape.
[[nodiscard]] inline std::optional<NpuProgramFiles> NpuProgramPaths(
    std::string_view dir, NpuProgram program, std::uint32_t m,
    std::uint32_t k) {
  const std::string base(dir);
  const std::string shape =
      std::to_string(m) + "x" + std::to_string(k) + "x1024_128x64x128";
  switch (program) {
    case NpuProgram::kFixed:
      if (m != 4096 || k != 2560) {
        return std::nullopt;
      }
      return NpuProgramFiles{base + "/final_" + shape + ".xclbin",
                             base + "/insts_" + shape + ".txt"};
    case NpuProgram::kRuntimeK:
      if (m != 4096 || k == 0 || k % 64 != 0) {
        return std::nullopt;
      }
      return NpuProgramFiles{
          base + "/runtime_k/final_4096xKx1024_128x64x128.xclbin",
          base + "/runtime_k/insts_" + shape + ".txt"};
    case NpuProgram::kRuntimeMK:
      if (m == 0 || m % 512 != 0 || m > 4096 || k == 0 || k % 64 != 0) {
        return std::nullopt;
      }
      return NpuProgramFiles{
          base + "/runtime_mk/final_4096xKx1024_128x64x128.xclbin",
          base + "/runtime_mk/insts_" + shape + ".txt"};
  }
  return std::nullopt;
}

/// A prefill chunk of n tokens split into two banks of whole 4096-token
/// groups (two-bank prefill, mikealanni/gufo 58fac455). Bank A runs on the
/// executor's own scratch (max_batch rows), bank B on a second scratch of
/// max_batch / 2 rows. A multiple of 8192 splits evenly; otherwise bank A
/// takes the extra group. With `relaxed` (GUFO_TWO_BANK=2) n need not be a
/// multiple of 4096 and bank B takes the remainder (then the NPU
/// whole-projection mode cannot run on it). Nullopt when the chunk does not
/// qualify.
struct TwoBankSplit {
  std::uint32_t a{0};
  std::uint32_t b{0};
};
[[nodiscard]] inline std::optional<TwoBankSplit> PlanTwoBank(
    std::size_t n, std::uint32_t max_batch, bool relaxed) {
  constexpr std::size_t kGroup = 4096;
  if (n < 2 * kGroup || n > max_batch || (n % kGroup != 0 && !relaxed)) {
    return std::nullopt;
  }
  const std::size_t groups = n / kGroup;
  const std::size_t a_groups =
      n % (2 * kGroup) == 0 ? groups / 2 : (groups + 1) / 2;
  const std::size_t a = a_groups * kGroup;
  const std::size_t b = n - a;
  if (a < kGroup || b < kGroup || a > max_batch || b > max_batch / 2) {
    return std::nullopt;
  }
  return TwoBankSplit{static_cast<std::uint32_t>(a),
                      static_cast<std::uint32_t>(b)};
}

/// The engine feeds a long prefill chunk to the two-bank path as the largest
/// multiple of 8192 tokens (equal banks; 8192 + 4096 runs slower than one
/// bank) and the rest as its own chunk.
[[nodiscard]] constexpr std::size_t TwoBankChunk(std::size_t n) noexcept {
  return n > 8192 && n % 8192 != 0 ? n / 8192 * 8192 : n;
}

/// Whole-projection NPU groups (4096 tokens each) per bank: enough for the
/// largest bank a max_batch chunk can produce, at most three.
[[nodiscard]] constexpr std::uint32_t WholeGroupsPerBank(
    std::uint32_t max_batch, std::uint32_t group_tokens) noexcept {
  if (group_tokens == 0) {
    return 0;
  }
  const std::uint32_t groups = max_batch / 2 / group_tokens;
  return groups < 3 ? groups : 3;
}

/// GUFO_PREFILL_TILE: the prefill chunk in tokens, 128 to 65536, or
/// `fallback` when unset or out of range (mikealanni/gufo f4502187).
[[nodiscard]] inline std::uint32_t ParsePrefillTile(const char* value,
                                                    std::uint32_t fallback) {
  if (value == nullptr || value[0] == '\0') {
    return fallback;
  }
  std::uint64_t tile = 0;
  for (const char* p = value; *p != '\0'; ++p) {
    if (*p < '0' || *p > '9' || tile > 65536) {
      return fallback;
    }
    tile = tile * 10 + static_cast<std::uint64_t>(*p - '0');
  }
  return tile >= 128 && tile <= 65536 ? static_cast<std::uint32_t>(tile)
                                      : fallback;
}

}  // namespace gufo::models::qwen38_flash_next
