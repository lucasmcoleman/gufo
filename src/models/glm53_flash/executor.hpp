#ifndef GUFO_MODELS_GLM53_FLASH_EXECUTOR_HPP_
#define GUFO_MODELS_GLM53_FLASH_EXECUTOR_HPP_

// GPU executor for GLM-5.3-Flash (Lane A). First milestone: the KDA mixer
// sub-layer. The full executor (layer loop, hyper-connections, embedding,
// head, Lane B's MlaForward / FfnForward) grows from the pieces here:
//   DeviceArena     owns every device allocation (weights, state, scratch);
//   UploadMatrix    copies one GGUF tensor to the device in its encoding;
//   UploadKdaLayer  fills a KdaLayerDevice the way MLA / FFN layers will fill
//                   MlaLayerDevice / FfnLayerDevice (layer_ops.hpp);
//   KdaMixerForward is the KDA counterpart of MlaForward, with the same
//                   OpContext and call conventions (layer_ops.hpp).
// The scalar contract is ReferenceModel::Kda (reference.cpp) on top of
// reference_kda.hpp; the kernels between the projections are
// kernels/kda.hpp.
//
// Projections run through gufo's GEMMs with F32 activations: Q8_0, Q8_K and
// the in-kernel K-quant/IQ formats (Q3_K..Q6_K, IQ4_NL, IQ4_XS, IQ3_S) via
// LaunchBatchedQuantGEMMFp32, F32 and BF16 via LaunchGEMV per token. Other
// encodings are refused at upload. Every KDA projection of the Maya-S GGUF is
// Q8_0 or Q6_K.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "src/models/glm53_flash/config.hpp"
#include "src/models/glm53_flash/layer_ops.hpp"
#include "src/models/glm53_flash/weights.hpp"

namespace gufo::models::glm53_flash {

/// Owns device allocations; frees them on destruction. Not thread-safe.
class DeviceArena {
public:
  DeviceArena() = default;
  ~DeviceArena();
  DeviceArena(const DeviceArena&) = delete;
  DeviceArena& operator=(const DeviceArena&) = delete;
  DeviceArena(DeviceArena&& other) noexcept;
  DeviceArena& operator=(DeviceArena&& other) noexcept;

  /// A new uninitialised allocation of `bytes` (256-byte aligned), or nullptr
  /// if the device allocation fails.
  [[nodiscard]] void* Allocate(std::size_t bytes);
  /// Bytes allocated so far.
  [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

private:
  std::vector<void*> blocks_;
  std::size_t bytes_{0};
};

/// True if `type` with `cols` reduction elements has a projection route
/// (see the file comment and the block size of the format).
[[nodiscard]] bool SupportsProjection(core::GgmlType type,
                                      std::uint64_t cols) noexcept;

/// Copies `t` (all experts) to the device unchanged and describes it in
/// `out`. Synchronous. Fails on an empty tensor, an encoding without a
/// projection route, or an allocation / copy failure.
[[nodiscard]] bool UploadMatrix(DeviceArena& arena, const TensorRef& t,
                                DeviceMatrix* out, std::string* error);

/// Y[tokens x w.rows] = X[tokens x w.cols] * W^T for a single-expert matrix,
/// enqueued on `stream`. False on a matrix without a projection route.
[[nodiscard]] bool LaunchProjection(const DeviceMatrix& w, const float* x,
                                    float* y, std::uint32_t tokens,
                                    Stream stream);

/// Device weights of one KDA layer. Projections keep their GGUF encoding;
/// q|k|v are stacked into one [3*kda_dim x hidden] matrix (the conv channel
/// order), which needs a common encoding. F32 vectors are plain floats.
struct KdaLayerDevice {
  DeviceMatrix qkv;       ///< [hidden -> 3*kda_dim], attn_q|attn_k|attn_v
  DeviceMatrix f_a, f_b;  ///< forget gate [hidden -> D], [D -> kda_dim]
  DeviceMatrix g_a, g_b;  ///< output gate [hidden -> D], [D -> kda_dim]
  DeviceMatrix beta;      ///< [hidden -> heads]
  DeviceMatrix out;       ///< attn_output [kda_dim -> hidden]
  const float* conv_w{nullptr};   ///< [3*kda_dim][4], ssm_conv1d_q|k|v
  const float* dt_bias{nullptr};  ///< [kda_dim] ssm_dt.bias
  const float* a{nullptr};        ///< [heads] ssm_a = -exp(A_log)
  const float* norm{nullptr};     ///< [D] ssm_norm, shared by the heads
};

/// Per-layer KDA state of one sequence: conv history and recurrent state.
/// A zero state is a fresh sequence.
struct KdaState {
  float* conv{nullptr};       ///< [3][3*kda_dim] last three q|k|v inputs
  float* recurrent{nullptr};  ///< [heads][D value][D key] (kernels/kda.hpp)
};

/// Uploads layer `layer`'s KDA weights. Fails (with `error`) on a non-KDA
/// layer, a shape that does not match `config`, a head dim other than 128,
/// a conv kernel other than 4, q/k/v in different encodings, an encoding
/// without a projection route, or an allocation failure. Synchronous.
[[nodiscard]] bool UploadKdaLayer(DeviceArena& arena, const Config& config,
                                  const LayerWeights& layer, std::uint32_t il,
                                  KdaLayerDevice* out, std::string* error);

/// Bytes of one layer's KdaState.
[[nodiscard]] std::size_t KdaStateBytes(const Config& config) noexcept;

/// Allocates one layer's KdaState from `arena` and zeroes it (synchronous).
[[nodiscard]] bool AllocateKdaState(DeviceArena& arena, const Config& config,
                                    KdaState* out);

/// Zeroes `state` on `stream` (start of a new sequence).
[[nodiscard]] bool ResetKdaState(const Config& config, const KdaState& state,
                                 Stream stream);

/// Scratch KdaMixerForward needs to run `max_tokens` tokens in one pass.
/// Calls with more tokens than the scratch holds run in chunks, which is
/// exact (the state carries across chunks).
[[nodiscard]] std::size_t KdaScratchBytes(const Config& config,
                                          std::uint32_t max_tokens) noexcept;

/// KDA mixer (Glm5NextTextLinearAttention.forward) on `tokens` consecutive
/// positions of one sequence, continuing from `state` and advancing it:
/// prefill is one call, decode a call per token. `x` is the RMS-normalised
/// mixer input [tokens x hidden], `y` the output [tokens x hidden], both
/// device F32; they may not alias each other or the scratch. Uses
/// ctx.scratch (at least KdaScratchBytes(config, 1)); with ctx.tap set,
/// reports `y` as "kda_out" (llama.cpp's name). Returns false without
/// touching `state` on a bad argument (no KDA layer, null pointer, scratch
/// too small); false after a failed launch leaves `state` unspecified.
[[nodiscard]] bool KdaMixerForward(const OpContext& ctx, const Config& config,
                                   std::uint32_t layer, const KdaLayerDevice& w,
                                   KdaState& state, const float* x,
                                   std::uint32_t tokens, float* y);

}  // namespace gufo::models::glm53_flash

#endif  // GUFO_MODELS_GLM53_FLASH_EXECUTOR_HPP_
