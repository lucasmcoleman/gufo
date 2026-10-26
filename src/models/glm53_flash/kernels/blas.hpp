#ifndef GUFO_MODELS_GLM53_FLASH_KERNELS_BLAS_HPP_
#define GUFO_MODELS_GLM53_FLASH_KERNELS_BLAS_HPP_

// One hipBLAS handle per stream for the GLM-5.3-Flash operators (dense
// prefill attention in mla.hip.cpp, the router GEMM in moe.hip.cpp).
// hipblasSetStream changes handle state, so a shared handle would race
// between streams. Handles live for the process.

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <map>
#include <mutex>

namespace gufo::models::glm53_flash {

/// The stream's handle, created on first use; nullptr if creation fails.
inline hipblasHandle_t BlasHandle(hipStream_t stream) {
  static std::mutex mutex;
  static std::map<hipStream_t, hipblasHandle_t> handles;
  const std::lock_guard<std::mutex> lock(mutex);
  auto it = handles.find(stream);
  if (it == handles.end()) {
    hipblasHandle_t h = nullptr;
    if (hipblasCreate(&h) != HIPBLAS_STATUS_SUCCESS ||
        hipblasSetStream(h, stream) != HIPBLAS_STATUS_SUCCESS) {
      return nullptr;
    }
    it = handles.emplace(stream, h).first;
  }
  return it->second;
}

}  // namespace gufo::models::glm53_flash

#endif  // GUFO_MODELS_GLM53_FLASH_KERNELS_BLAS_HPP_
