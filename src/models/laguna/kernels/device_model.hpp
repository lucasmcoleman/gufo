#ifndef GUFO_MODELS_LAGUNA_KERNELS_DEVICE_MODEL_HPP_
#define GUFO_MODELS_LAGUNA_KERNELS_DEVICE_MODEL_HPP_

// The Laguna-S-2.1 trunk resident on the GPU: per layer the attention weights
// (AttnWeights, built here so the payloads can stream from the file), the FFN
// (LayerFfn) and the two RMSNorm weights, plus the final norm and the untied
// Q6_K output. token_embd stays in the mapping: the executor gathers the few
// rows a call needs on the host.

#include <hip/hip_runtime.h>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/laguna/kernels/attention.hpp"
#include "src/models/laguna/kernels/ffn.hpp"
#include "src/models/laguna/weights.hpp"

namespace gufo::models::laguna::rocm {

struct DeviceLayer {
  AttnWeights attn;
  LayerFfn ffn;
  const float* attn_norm{nullptr};  ///< F32 [hidden]
  const float* ffn_norm{nullptr};   ///< F32 [hidden]
};

class DeviceModel {
public:
  ~DeviceModel();
  DeviceModel(const DeviceModel&) = delete;
  DeviceModel& operator=(const DeviceModel&) = delete;

  /// Uploads every tensor but token_embd. With a `reader` the payloads stream
  /// from its file through a bounded staging pipeline (no second host copy of
  /// the model, the mapping is never read); without one (synthetic models)
  /// they are copied from TensorRef::data. `weights` must outlive the model.
  /// Nullptr with a message on any failure; everything allocated is freed.
  [[nodiscard]] static std::unique_ptr<DeviceModel> Upload(
      const ModelWeights& weights, const core::GgufReader* reader,
      std::string* error_msg = nullptr);

  [[nodiscard]] const ModelWeights& weights() const noexcept {
    return *weights_;
  }
  [[nodiscard]] const Config& config() const noexcept {
    return weights_->config;
  }
  [[nodiscard]] const std::vector<DeviceLayer>& layers() const noexcept {
    return layers_;
  }
  [[nodiscard]] const float* output_norm() const noexcept {
    return output_norm_;
  }
  [[nodiscard]] const DeviceMatrix& output() const noexcept { return output_; }
  /// Device bytes held, padding included (equals WeightDeviceBytes).
  [[nodiscard]] std::size_t resident_bytes() const noexcept { return bytes_; }

private:
  DeviceModel() = default;
  const ModelWeights* weights_{nullptr};
  std::vector<DeviceLayer> layers_;
  const float* output_norm_{nullptr};
  DeviceMatrix output_;
  std::vector<void*> allocations_;
  std::size_t bytes_{0};
};

}  // namespace gufo::models::laguna::rocm

#endif  // GUFO_MODELS_LAGUNA_KERNELS_DEVICE_MODEL_HPP_
