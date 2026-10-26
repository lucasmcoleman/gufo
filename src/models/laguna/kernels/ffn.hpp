#ifndef GUFO_MODELS_LAGUNA_KERNELS_FFN_HPP_
#define GUFO_MODELS_LAGUNA_KERNELS_FFN_HPP_

// GPU FFN operators for Laguna-S-2.1 (gfx1151): the dense SwiGLU of layer 0
// and the sigmoid-routed MoE of layers 1 to 47. They follow
// docs/models/laguna-s-2.1/ARCH.md and are checked against the phase 2 CPU
// oracle (`reference::FfnForward`).
//
//   logits = x Wr (F32)            s = sigmoid(logits)
//   sel    = topk(s + bias, k)     ties pick the lower expert id
//   w      = s[sel] / max(sum s[sel], 2^-14) * scale
//   out    = sum_j w_j SwiGLU-expert_sel_j(x) + SwiGLU-shared(x)   (no gate)
//
// The routed projections run on the quantized GEMM tier vendored with
// qwen38_flash_next (qfn_mmq): Q6_K, Q5_K and IQ4_XS routed experts, Q8_0
// shared and dense matrices. Few rows use the MMVQ vector kernels (Q8_1
// activations in 32-blocks with an fp16 scale), many rows the MMQ tile
// kernels (Q8_1 MMQ layout), so the two paths round activations differently.
//
// Scratch is owned by the FfnOps instance and every launch is asynchronous on
// the instance's stream: use one FfnOps per stream, and do not share an
// instance between streams. Device inputs and outputs must stay valid until
// the stream has run past the call.

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/laguna/config.hpp"
#include "src/models/laguna/weights.hpp"

namespace gufo::models::laguna::rocm {

/// A device-resident tensor in its GGUF encoding. `cols` is the contiguous
/// reduction dimension, `rows` the outputs, `experts` stacks whole matrices.
struct DeviceMatrix {
  const void* data{nullptr};
  core::GgmlType type{core::GgmlType::kF32};
  std::uint32_t cols{0};
  std::uint32_t rows{0};
  std::uint32_t experts{1};
};

/// Dense SwiGLU: down(silu(x gate) * (x up)). Layer 0 (Q8_0, ff 12288), and
/// the shared expert inside the MoE (Q8_0, ff 1024).
struct DenseFfnWeights {
  DeviceMatrix gate;
  DeviceMatrix up;
  DeviceMatrix down;
};

struct MoeFfnWeights {
  const float* router{nullptr};  ///< F32 [experts][hidden].
  const float* bias{nullptr};    ///< F32 [experts], selection only.
  DeviceMatrix gate_exps;        ///< [hidden -> expert_ff] x experts
  DeviceMatrix up_exps;          ///< same shape and type as gate_exps
  DeviceMatrix down_exps;        ///< [expert_ff -> hidden] x experts
  DenseFfnWeights shared;        ///< Shared expert, always added, no gate.
  std::uint32_t experts{0};
  std::uint32_t used{0};  ///< Experts per token (top-k), at most 32.
  /// Divide by max(sum, 2^-14) after selection (expert_weights_norm).
  bool normalize{true};
  /// Applied after normalizing; 0 or 1 means unscaled (expert_weights_scale).
  float scale{1.0F};
};

/// Which kernels a call uses. kAuto picks kVector up to kVectorMaxTokens rows
/// and kBatched above. The explicit values let tests exercise both paths at
/// every row count (kVector accepts up to 32 rows).
enum class Path { kAuto, kVector, kBatched };
inline constexpr std::uint32_t kVectorMaxTokens = 8;

/// Owning device copy of one layer's FFN tensors, uploaded from the mapped
/// GGUF (or any host TensorRefs). Move-only.
class LayerFfn {
public:
  LayerFfn() = default;
  LayerFfn(const LayerFfn&) = delete;
  LayerFfn& operator=(const LayerFfn&) = delete;
  LayerFfn(LayerFfn&& other) noexcept;
  LayerFfn& operator=(LayerFfn&& other) noexcept;
  ~LayerFfn();

  /// Copies the tensors `layer` uses (dense or MoE per `layer.moe`). Returns
  /// nullopt with a message on an unsupported type, a shape that disagrees
  /// with `config`, or a HIP failure (partial uploads are freed).
  [[nodiscard]] static std::optional<LayerFfn> Upload(
      const Config& config, const LayerWeights& layer,
      std::string* error_msg = nullptr);

  [[nodiscard]] bool moe() const noexcept { return moe_; }
  [[nodiscard]] const DenseFfnWeights& dense() const noexcept { return dense_; }
  [[nodiscard]] const MoeFfnWeights& routed() const noexcept { return routed_; }
  /// Replaces the F32 router (a test hook: same shape as the uploaded one).
  [[nodiscard]] bool ReplaceRouter(const std::vector<float>& values,
                                   std::string* error_msg = nullptr);
  [[nodiscard]] std::size_t device_bytes() const noexcept { return bytes_; }

private:
  void Release() noexcept;
  bool moe_{false};
  DenseFfnWeights dense_;
  MoeFfnWeights routed_;
  std::vector<void*> allocations_;
  std::size_t bytes_{0};
};

class FfnOps {
public:
  /// `stream` must outlive this object. Calls qfn_mmq_init lazily.
  explicit FfnOps(hipStream_t stream);
  FfnOps(const FfnOps&) = delete;
  FfnOps& operator=(const FfnOps&) = delete;
  ~FfnOps();

  /// out[t][hidden] = dense SwiGLU of x[t][hidden]. Q8_0 weights only.
  [[nodiscard]] bool Dense(const DenseFfnWeights& w, const float* x, float* out,
                           std::uint32_t tokens, Path path = Path::kAuto,
                           std::string* error_msg = nullptr);

  /// Router only: fills the routing scratch (see ids()/weights()/logits()).
  [[nodiscard]] bool Route(const MoeFfnWeights& w, const float* x,
                           std::uint32_t hidden, std::uint32_t tokens,
                           std::string* error_msg = nullptr);

  /// out[t][hidden] = routed experts + shared expert of x[t][hidden]. Leaves
  /// the routing in the scratch.
  [[nodiscard]] bool Moe(const MoeFfnWeights& w, const float* x, float* out,
                         std::uint32_t hidden, std::uint32_t tokens,
                         Path path = Path::kAuto,
                         std::string* error_msg = nullptr);

  /// Debug tap for stage-by-stage checks. When set, the operator synchronizes
  /// the stream after each stage and calls `tap` with the stage name, a device
  /// pointer and an element count (int32 for "topk", float otherwise). Stages:
  /// router_logits, topk, weights, gate, up, act, down (routed experts, one
  /// row per (token, slot)), then shared_gate, shared_up, shared_act and
  /// shared_out (the shared expert; a dense layer reports "dense_gate",
  /// "dense_up", "dense_act", "dense_out"). Taps hide stream races because of
  /// the synchronization, so every check also runs untapped.
  using TapFn = std::function<void(std::string_view name, const void* device,
                                   std::size_t count, bool is_int32)>;
  void SetTap(TapFn tap) { tap_ = std::move(tap); }

  /// Routing of the last Route/Moe call, valid until the next one on this
  /// instance: logits [tokens][experts] F32, ids [tokens][used] int32 and
  /// weights [tokens][used] F32 (scale included), in selection order.
  [[nodiscard]] const float* logits() const noexcept { return logits_; }
  [[nodiscard]] const std::int32_t* ids() const noexcept { return ids_; }
  [[nodiscard]] const float* weights() const noexcept { return weights_; }

  /// Total bytes of scratch currently held (grows monotonically).
  [[nodiscard]] std::size_t scratch_bytes() const noexcept;

  /// Test seam for the scratch allocator: every scratch allocation and free
  /// (including the destructor's) goes through these. Null restores
  /// hipMalloc/hipFree. Swap them only while no work is queued that would
  /// free scratch through the other pair.
  using DeviceAlloc = hipError_t (*)(void** ptr, std::size_t bytes);
  using DeviceFree = hipError_t (*)(void* ptr);
  void SetAllocatorForTesting(DeviceAlloc alloc, DeviceFree release);

private:
  /// Grows `*ptr` to at least `bytes`. The replacement is allocated before
  /// the old buffer is freed, so on failure `*ptr` and `*capacity` keep their
  /// old (still valid) values; on success both are updated together.
  template<class T>
  bool Reserve(T** ptr, std::size_t* capacity, std::size_t bytes,
               std::string* error_msg);
  bool ProjectDense(const DeviceMatrix& w, const float* x, float* out,
                    std::uint32_t tokens, bool vector, std::string* error_msg);
  bool DenseImpl(const DenseFfnWeights& w, const float* x, float* out,
                 std::uint32_t tokens, bool vector, const char* prefix,
                 std::string* error_msg);
  bool Emit(const char* prefix, const char* name, const void* device,
            std::size_t count, bool is_int32, std::string* error_msg);
  bool Experts(const DeviceMatrix& w, const float* x, const std::int32_t* ids,
               float* out, std::uint32_t rows, std::uint32_t used, bool vector,
               const DeviceMatrix* pair, float* pair_out,
               std::string* error_msg);

  hipStream_t stream_;
  TapFn tap_;
  DeviceAlloc alloc_;
  DeviceFree free_;
  bool initialized_{false};
  float* logits_{nullptr};
  std::int32_t* ids_{nullptr};
  float* weights_{nullptr};
  std::size_t logits_cap_{0}, ids_cap_{0}, weights_cap_{0};
  float* gate_{nullptr};
  float* up_{nullptr};
  float* act_{nullptr};
  float* down_{nullptr};
  float* shared_{nullptr};
  void* q8_{nullptr};
  std::size_t gate_cap_{0}, up_cap_{0}, act_cap_{0}, down_cap_{0},
      shared_cap_{0}, q8_cap_{0};
};

}  // namespace gufo::models::laguna::rocm

#endif  // GUFO_MODELS_LAGUNA_KERNELS_FFN_HPP_
