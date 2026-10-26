// Development check: encodes one 1024-row slice of a K-quant GGUF projection as
// NPU block-float weights, runs it on the NPU with random activations, and
// compares with an fp32 GPU product of the same decoded weights.
// usage: npu_quant_check model.gguf tensor_name xclbin insts [k] [row0]
//        [min_cosine]
// Exits 3 when the cosine falls below min_cosine (default 0.999).
//
// It drives the NPU and the GPU: run it only under the NPU-round prerequisites
// (fabric clock held, memlock unlimited, GPU admission lock; see
// /server/staging/npu-borrow/NPU-ROUND.md), never as a casual probe.
//
// Ported from mikealanni/gufo 4f346915 (MIT, Mike Alani); the min_cosine gate
// is added here.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen/hip/quant_ops.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/npu_gemm.hpp"

namespace {

using gufo::models::qwen38_flash_next::rocm::NpuGemm;
namespace rocm = gufo::models::qwen38_flash_next::rocm;

constexpr std::uint32_t kTokens = 4096;
constexpr std::uint32_t kCols = 1024;

#define CHECK(x)                                             \
  do {                                                       \
    const hipError_t e_ = (x);                               \
    if (e_ != hipSuccess) {                                  \
      std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, \
                   hipGetErrorString(e_));                   \
      std::exit(1);                                          \
    }                                                        \
  } while (0)

__global__ void DecodeWeightsKernel(const std::uint8_t* w, std::uint32_t type,
                                    std::size_t row_bytes, std::uint32_t k,
                                    float* out) {
  const std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (i >= std::size_t{kCols} * k) {
    return;
  }
  const std::size_t row = i / k;
  const std::size_t col = i % k;
  out[i] = gufo::hip::QuantBlockElement(static_cast<gufo::core::GgmlType>(type),
                                        w + row * row_bytes, col);
}

__global__ void ReferenceKernel(const __half* x, const float* w,
                                std::uint32_t k, float* out) {
  const std::uint32_t n = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t t = blockIdx.y;
  if (n >= kCols) {
    return;
  }
  float acc = 0.0F;
  for (std::uint32_t i = 0; i < k; ++i) {
    acc += __half2float(x[std::size_t{t} * k + i]) * w[std::size_t{n} * k + i];
  }
  out[std::size_t{t} * kCols + n] = acc;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr,
                 "usage: %s model.gguf tensor xclbin insts [k] [row0] "
                 "[min_cosine]\n",
                 argv[0]);
    return 2;
  }
  const std::uint32_t k = argc > 5 ? std::atoi(argv[5]) : 5120;
  const std::uint32_t row0 = argc > 6 ? std::atoi(argv[6]) : 0;
  const double min_cosine = argc > 7 ? std::atof(argv[7]) : 0.999;
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(argv[1], &error);
  if (!reader) {
    std::fprintf(stderr, "open: %s\n", error.c_str());
    return 1;
  }
  const auto* tensor = reader->FindTensor(argv[2]);
  if (tensor == nullptr) {
    std::fprintf(stderr, "no tensor %s\n", argv[2]);
    return 1;
  }
  const auto type = tensor->type;
  const std::size_t row_bytes = gufo::quant::QuantizedRowBytes(type, k);
  std::printf("%s type=%s row_bytes=%zu\n", argv[2],
              std::string(gufo::core::ToString(type)).c_str(), row_bytes);

  std::uint8_t* d_w = nullptr;
  CHECK(hipMalloc(&d_w, row_bytes * kCols));
  CHECK(hipMemcpy(
      d_w, static_cast<const std::uint8_t*>(tensor->data) + row_bytes * row0,
      row_bytes * kCols, hipMemcpyHostToDevice));

  NpuGemm::Config cfg;
  cfg.xclbin = argv[3];
  cfg.insts = argv[4];
  cfg.tokens = kTokens;
  cfg.k = k;
  cfg.n = kCols;
  auto npu = NpuGemm::Create(cfg, &error);
  if (!npu) {
    std::fprintf(stderr, "NPU: %s\n", error.c_str());
    return 1;
  }
  const int a_id = npu->NewBuffer(npu->ABytes(), &error);
  const int b_id = npu->NewBuffer(npu->BBytes(), &error);
  const int c_id = npu->NewBuffer(npu->CBytes(), &error);
  if (a_id < 0 || b_id < 0 || c_id < 0) {
    std::fprintf(stderr, "buffers: %s\n", error.c_str());
    return 1;
  }

  hipStream_t stream;
  CHECK(hipStreamCreate(&stream));
  rocm::NpuEncodeWeightsQuant(d_w, static_cast<std::uint32_t>(type), k, 0,
                              kCols, k, 0, k, npu->Device(b_id), stream);
  CHECK(hipStreamSynchronize(stream));
  npu->FinalizeWeights(b_id);

  std::vector<__half> host_x(std::size_t{kTokens} * k);
  std::mt19937 rng(7);
  std::normal_distribution<float> dist(0.0F, 1.0F);
  for (auto& v : host_x)
    v = __float2half(dist(rng));
  __half* d_x = nullptr;
  CHECK(hipMalloc(&d_x, host_x.size() * sizeof(__half)));
  CHECK(hipMemcpy(d_x, host_x.data(), host_x.size() * sizeof(__half),
                  hipMemcpyHostToDevice));
  rocm::NpuEncodeActivations(d_x, k, kTokens, k, 0, k, npu->Device(a_id),
                             stream);

  float* d_out = nullptr;
  float* d_ref = nullptr;
  float* d_wf = nullptr;
  CHECK(hipMalloc(&d_out, std::size_t{kTokens} * kCols * sizeof(float)));
  CHECK(hipMalloc(&d_ref, std::size_t{kTokens} * kCols * sizeof(float)));
  CHECK(hipMalloc(&d_wf, std::size_t{kCols} * k * sizeof(float)));

  hipEvent_t t0, t1;
  CHECK(hipEventCreate(&t0));
  CHECK(hipEventCreate(&t1));
  for (int rep = 0; rep < 3; ++rep) {
    CHECK(hipEventRecord(t0, stream));
    if (!npu->Submit(stream, a_id, &b_id, &c_id, 1) || !npu->Wait(stream)) {
      std::fprintf(stderr, "NPU run failed\n");
      return 1;
    }
    CHECK(hipEventRecord(t1, stream));
    CHECK(hipEventSynchronize(t1));
    float ms = 0.0F;
    CHECK(hipEventElapsedTime(&ms, t0, t1));
    std::printf("npu run %d: %.3f ms (%.1f TFLOPS)\n", rep, ms,
                2.0 * kTokens * kCols * k / (ms * 1e-3) / 1e12);
  }
  rocm::NpuDecodeOutput(npu->Staged(stream, c_id), d_out, kTokens, kCols, kCols,
                        0, false, stream);

  hipLaunchKernelGGL(DecodeWeightsKernel,
                     dim3((std::size_t{kCols} * k + 255) / 256), dim3(256), 0,
                     stream, d_w, static_cast<std::uint32_t>(type), row_bytes,
                     k, d_wf);
  hipLaunchKernelGGL(ReferenceKernel, dim3((kCols + 127) / 128, kTokens),
                     dim3(128), 0, stream, d_x, d_wf, k, d_ref);
  CHECK(hipStreamSynchronize(stream));

  std::vector<float> out(std::size_t{kTokens} * kCols);
  std::vector<float> ref(out.size());
  CHECK(hipMemcpy(out.data(), d_out, out.size() * sizeof(float),
                  hipMemcpyDeviceToHost));
  CHECK(hipMemcpy(ref.data(), d_ref, ref.size() * sizeof(float),
                  hipMemcpyDeviceToHost));
  double dot = 0.0, na = 0.0, nb = 0.0, err = 0.0;
  for (std::size_t i = 0; i < out.size(); ++i) {
    dot += double(out[i]) * ref[i];
    na += double(out[i]) * out[i];
    nb += double(ref[i]) * ref[i];
    const double d = double(out[i]) - ref[i];
    err += d * d;
  }
  const double cosine = dot / std::sqrt(na * nb);
  std::printf("cosine %.6f  rel RMS error %.4f  (ref RMS %.3f)\n", cosine,
              std::sqrt(err / nb), std::sqrt(nb / out.size()));
  if (!(cosine >= min_cosine)) {
    std::printf("FAIL: cosine below %.6f\n", min_cosine);
    return 3;
  }
  std::printf("PASS\n");
  return 0;
}
