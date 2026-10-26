#ifndef GUFO_MODELS_LAGUNA_KERNELS_DFLASH_OPS_HPP_
#define GUFO_MODELS_LAGUNA_KERNELS_DFLASH_OPS_HPP_

// GPU operators of the Laguna DFlash drafter (gfx1151), phase 6c of
// docs/models/laguna-s-2.1/DFLASH.md. Each is checked against the phase 6b CPU
// oracle (`reference::DFlashModel`, src/models/laguna/dflash_reference.*).
//
// Encoder, per context row t (no dependence between rows):
//
//   a_k   = RMSNorm(feat[t][k*H .. (k+1)*H)) * aux_norm[k]     k < taps
//   u     = fc . [a_0 | ... | a_{taps-1}]                      Q4_K
//   fused = RMSNorm(u) * enc_output_norm
//
// The projections reuse the qfn_mmq kernels vendored with qwen38_flash_next
// with one expert and zero ids (the target's head does the same): the vector
// (MMVQ) path quantizes activations to Q8_1, the batched (MMQ tile) path to the
// MMQ layout, so the two paths round differently; compare runs of one Path.
//
// Scratch is owned by the DFlashOps instance and every launch is asynchronous
// on the instance's stream: use one DFlashOps per stream. Growing a scratch
// buffer allocates the replacement first and swaps it in only on success, so a
// failed allocation leaves the instance exactly as it was.

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/laguna/dflash.hpp"
#include "src/models/laguna/kernels/attention.hpp"
#include "src/models/laguna/kernels/ffn.hpp"
#include "src/models/laguna/weights.hpp"

namespace gufo::models::laguna::rocm {

/// Device view of one drafter layer. Projections are Q4_K or Q6_K.
struct DrafterLayer {
  const float* attn_norm{nullptr};  ///< F32 [hidden]
  DeviceMatrix q;                   ///< [hidden -> heads*128]
  DeviceMatrix k;                   ///< [hidden -> kv_heads*128]
  DeviceMatrix v;                   ///< [hidden -> kv_heads*128]
  DeviceMatrix o;                   ///< [heads*128 -> hidden]
  DeviceMatrix gate;                ///< [hidden -> heads], softplus gate
  const float* q_norm{nullptr};     ///< F32 [128]
  const float* k_norm{nullptr};     ///< F32 [128]
  const float* ffn_norm{nullptr};   ///< F32 [hidden]
  DeviceMatrix ffn_gate;            ///< [hidden -> ffn]
  DeviceMatrix ffn_up;              ///< [hidden -> ffn]
  DeviceMatrix ffn_down;            ///< [ffn -> hidden]
};

/// Device view of the whole drafter. Plain data: tests may copy it and change
/// a field to check that the checks notice (wrong theta, window, swapped K/V).
struct DrafterWeights {
  std::uint32_t hidden{0};
  std::uint32_t ffn{0};
  std::uint32_t heads{0};
  std::uint32_t kv_heads{0};
  std::uint32_t head_dim{128};
  std::uint32_t taps{0};
  /// Every layer is sliding: key j is visible from p iff p - j < window.
  std::uint32_t window{0};
  std::uint32_t block_size{0};  ///< Rows per block including the anchor.
  float eps{1e-6F};
  RopeSpec rope;  ///< NeoX, no YaRN (ext_factor 0, mscale 1).
  /// Block rows see block rows 0..r only (Laguna's decoder). False makes the
  /// block non-causal (generic DFlash); kept as data so a test can flip it.
  bool causal_block{true};
  const float* aux_norm{nullptr};     ///< F32 [taps][hidden]
  DeviceMatrix fc;                    ///< [taps*hidden -> hidden]
  const float* enc_norm{nullptr};     ///< F32 [hidden]
  const float* output_norm{nullptr};  ///< F32 [hidden]
  std::vector<DrafterLayer> layers;
};

/// Owning device copy of a drafter (DFlashWeights). Move-only.
class DeviceDrafter {
public:
  DeviceDrafter() = default;
  DeviceDrafter(const DeviceDrafter&) = delete;
  DeviceDrafter& operator=(const DeviceDrafter&) = delete;
  DeviceDrafter(DeviceDrafter&& other) noexcept;
  DeviceDrafter& operator=(DeviceDrafter&& other) noexcept;
  ~DeviceDrafter();

  /// Copies every tensor from TensorRef::data (the mapping or a fixture).
  /// Nullopt with a message on a type other than Q4_K/Q6_K (projections) or
  /// F32 (norms), a shape that disagrees with the config, head_dim != 128, or
  /// a HIP failure (everything allocated is freed).
  [[nodiscard]] static std::optional<DeviceDrafter> Upload(
      const DFlashWeights& weights, std::string* error_msg = nullptr);

  [[nodiscard]] const DrafterWeights& weights() const noexcept { return w_; }
  [[nodiscard]] std::size_t device_bytes() const noexcept { return bytes_; }

private:
  void Release() noexcept;
  DrafterWeights w_;
  std::vector<void*> allocations_;
  std::size_t bytes_{0};
};

class DFlashOps {
public:
  /// `stream` must outlive this object. Calls qfn_mmq_init.
  explicit DFlashOps(hipStream_t stream);
  DFlashOps(const DFlashOps&) = delete;
  DFlashOps& operator=(const DFlashOps&) = delete;
  ~DFlashOps();

  /// Test hook: replaces the scratch allocator. Never with work in flight.
  void SetAllocator(const DeviceAllocator& allocator) {
    allocator_ = allocator;
  }

  /// fused[t][hidden] = encoder of features[t][taps*hidden] (the target taps
  /// concatenated in DFlashConfig::target_layers order). Rows: at most 32 on
  /// Path::kVector.
  [[nodiscard]] bool Encode(const DrafterWeights& w, const float* features,
                            float* fused, std::uint32_t rows,
                            Path path = Path::kAuto,
                            std::string* error_msg = nullptr);

  /// Debug tap for stage-by-stage checks: the operator synchronizes the stream
  /// after each stage and calls `tap` with the stage name (the oracle's tap
  /// names, see DFlashModel::Forward, plus the raw projections), a device
  /// pointer and a float count. Taps hide stream races, so every check also
  /// runs untapped. Encoder stages: enc_aux_norm, fc_out, enc_norm_out.
  using TapFn = std::function<void(std::string_view name, const float* device,
                                   std::size_t count)>;
  void SetTap(TapFn tap) { tap_ = std::move(tap); }

  /// Test hook: the call fails right after stage `stage` (a tap name with its
  /// layer suffix; empty disables), whether or not a tap is set.
  void SetFailAfterStage(std::string stage) { fail_after_ = std::move(stage); }

  /// Bytes of scratch currently held.
  [[nodiscard]] std::size_t scratch_bytes() const noexcept;

private:
  struct Scratch {
    void* ptr{nullptr};
    std::size_t capacity{0};
  };
  bool Reserve(Scratch* buffer, std::size_t bytes, std::string* error_msg);
  void Free(Scratch* buffer) noexcept;
  bool Project(const DeviceMatrix& w, const float* x, float* out,
               std::uint32_t rows, bool vector, std::string* error_msg);
  bool Emit(const char* name, int layer, const float* device, std::size_t count,
            std::string* error_msg);
  bool Ready(Path path, std::uint32_t rows, bool* vector,
             std::string* error_msg) const;

  hipStream_t stream_;
  DeviceAllocator allocator_;
  TapFn tap_;
  std::string fail_after_;
  bool initialized_{false};
  Scratch ids_, aux_, u_;
};

}  // namespace gufo::models::laguna::rocm

#endif  // GUFO_MODELS_LAGUNA_KERNELS_DFLASH_OPS_HPP_
