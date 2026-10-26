#pragma once

// Host-side planning for the XDNA2 NPU path of Qwen3.8 Flash-Next: which
// instruction program to load. Pure functions with no HIP or XRT dependency,
// so the CPU tests exercise exactly the code the device model calls.

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

}  // namespace gufo::models::qwen38_flash_next
