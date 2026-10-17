// Full FFN parity against the independent scalar routing/SwiGLU oracle.
// Exact scratch and activation allocations expose ragged batch overruns.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/glm53_flash/kernels/moe.hpp"
#include "src/models/glm53_flash/layer_ops.hpp"
#include "src/models/glm53_flash/reference_moe.hpp"

namespace {
namespace glm = gufo::models::glm53_flash;
using gufo::core::GgmlType;
void Require(bool ok, const char* message) {
  if (!ok)
    throw std::runtime_error(message);
}
void Hip(hipError_t error) {
  if (error != hipSuccess)
    throw std::runtime_error(hipGetErrorString(error));
}
struct Buffer {
  void* p{};
  explicit Buffer(std::size_t bytes) { Hip(hipMalloc(&p, bytes)); }
  ~Buffer() { (void)hipFree(p); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  float* f() { return static_cast<float*>(p); }
};
struct Stream {
  hipStream_t value{};
  Stream() { Hip(hipStreamCreateWithFlags(&value, hipStreamNonBlocking)); }
  ~Stream() { (void)hipStreamDestroy(value); }
};
struct Weight {
  std::vector<float> host;
  std::unique_ptr<Buffer> device;
  glm::DeviceMatrix matrix;
  Weight(std::uint32_t rows, std::uint32_t cols, std::uint32_t experts,
         GgmlType type, std::mt19937& rng) {
    host.resize(std::size_t(rows) * cols * experts);
    const auto row_bytes = gufo::quant::EncodedSizeBytes(type, cols);
    std::vector<std::uint8_t> bytes(std::size_t(rows) * experts * row_bytes);
    if (type == GgmlType::kF32) {
      std::uniform_real_distribution<float> dist(-0.6F, 0.6F);
      for (auto& v : host)
        v = dist(rng);
      std::memcpy(bytes.data(), host.data(), bytes.size());
    } else {
      for (auto& v : bytes)
        v = static_cast<std::uint8_t>(rng());
      const auto block_bytes = row_bytes / (cols / 256);
      for (std::size_t offset = 0; offset < bytes.size();
           offset += block_bytes) {
        // IQ block scale is the first fp16. Keep it finite and small while
        // exercising independent random codebook indices, signs and scales.
        const auto scale = static_cast<std::uint16_t>(0x1400 + rng() % 1024);
        std::memcpy(bytes.data() + offset, &scale, sizeof(scale));
      }
      for (std::size_t row = 0; row < std::size_t(rows) * experts; ++row) {
        const auto* src = bytes.data() + row * row_bytes;
        auto* dst = host.data() + row * cols;
        switch (type) {
          case GgmlType::kIQ2_XXS:
            gufo::quant::DequantizeIQ2_XXS(src, dst, cols);
            break;
          case GgmlType::kIQ2_S:
            gufo::quant::DequantizeIQ2_S(src, dst, cols);
            break;
          case GgmlType::kIQ3_XXS:
            gufo::quant::DequantizeIQ3_XXS(src, dst, cols);
            break;
          default:
            throw std::runtime_error("unsupported fixture type");
        }
      }
    }
    device = std::make_unique<Buffer>(bytes.size());
    Hip(hipMemcpy(device->p, bytes.data(), bytes.size(),
                  hipMemcpyHostToDevice));
    matrix = {device->p, type, cols, rows, experts, row_bytes};
  }
};

void Run(bool dense, bool shared, bool clamp, bool normalize, int routing_case,
         std::uint32_t tokens, GgmlType type) {
  std::cout << "case dense=" << dense << " shared=" << shared
            << " clamp=" << clamp << " norm=" << normalize
            << " routing=" << routing_case << " tokens=" << tokens
            << " type=" << static_cast<int>(type) << std::endl;
  const bool iq = type != GgmlType::kF32;
  glm::Config config;
  config.num_layers = 2;
  config.leading_dense_layers = 1;
  config.hidden_size = iq ? 256 : 19;
  config.dense_ff = iq ? 256 : 37;
  config.expert_ff = iq ? 256 : 23;
  config.num_experts = 5;
  config.num_experts_used = 3;
  config.shared_expert_ff = shared ? (iq ? 256 : 29) : 0;
  config.num_shared_experts = shared ? 1 : 0;
  config.expert_weights_norm = normalize;
  config.expert_weights_scale = normalize ? 2.5F : 0.7F;
  const auto hidden = config.hidden_size;
  const auto mid = dense ? config.dense_ff : config.expert_ff;
  const auto experts = dense ? 1U : config.num_experts;
  std::mt19937 rng(1907 + tokens);
  Weight gate(mid, hidden, experts, type, rng);
  Weight up(mid, hidden, experts, type, rng);
  Weight down(hidden, mid, experts, type, rng);
  std::unique_ptr<Weight> sg, su, sd;
  glm::FfnLayerDevice weights;
  weights.dense = dense;
  weights.clamp_exp = clamp ? 0.2F : 0.0F;
  weights.clamp_shexp = clamp ? 0.15F : 0.0F;
  if (dense) {
    weights.gate = gate.matrix;
    weights.up = up.matrix;
    weights.down = down.matrix;
  } else {
    weights.gate_exps = gate.matrix;
    weights.up_exps = up.matrix;
    weights.down_exps = down.matrix;
    if (shared) {
      sg = std::make_unique<Weight>(config.shared_expert_ff, hidden, 1,
                                    GgmlType::kF32, rng);
      su = std::make_unique<Weight>(config.shared_expert_ff, hidden, 1,
                                    GgmlType::kF32, rng);
      sd = std::make_unique<Weight>(hidden, config.shared_expert_ff, 1,
                                    GgmlType::kF32, rng);
      weights.shexp_gate = sg->matrix;
      weights.shexp_up = su->matrix;
      weights.shexp_down = sd->matrix;
    }
  }
  std::uniform_real_distribution<float> dist(-0.7F, 0.7F);
  std::vector<float> x(std::size_t(tokens) * hidden);
  for (auto& v : x)
    v = dist(rng);
  std::vector<float> router(std::size_t(config.num_experts) * hidden, 0.0F);
  std::vector<float> bias(config.num_experts, 0.0F);
  if (routing_case == 0)
    for (auto& v : router)
      v = dist(rng);
  if (routing_case == 2)
    bias = {-0.9F, 0.8F, 0.3F, 0.9F, 0.6F};
  Buffer dr(router.size() * sizeof(float)), db(bias.size() * sizeof(float));
  Hip(hipMemcpy(dr.p, router.data(), router.size() * sizeof(float),
                hipMemcpyHostToDevice));
  Hip(hipMemcpy(db.p, bias.data(), bias.size() * sizeof(float),
                hipMemcpyHostToDevice));
  weights.router = dr.f();
  // Case 1 exercises ties and the optional null bias. Case 2 selects solely
  // by correction bias; that bias must not change mixture weights.
  weights.router_bias = routing_case == 1 ? nullptr : db.f();
  std::vector<float> gate_up(std::size_t(experts) * 2 * mid * hidden);
  for (std::uint32_t e = 0; e < experts; ++e) {
    const auto n = std::size_t(mid) * hidden;
    std::copy_n(gate.host.data() + e * n, n, gate_up.data() + e * 2 * n);
    std::copy_n(up.host.data() + e * n, n, gate_up.data() + (e * 2 + 1) * n);
  }
  std::vector<float> want(x.size()), got(x.size()), chunks(x.size());
  std::vector<float> logits(config.num_experts), shared_y(hidden);
  for (std::uint32_t t = 0; t < tokens; ++t) {
    const auto* input = x.data() + std::size_t(t) * hidden;
    auto* output = want.data() + std::size_t(t) * hidden;
    if (dense) {
      glm::reference::DenseFfn(input, gate.host.data(), up.host.data(),
                               down.host.data(), hidden, mid, weights.clamp_exp,
                               output);
      continue;
    }
    for (std::uint32_t e = 0; e < config.num_experts; ++e) {
      double sum = 0;
      for (std::uint32_t d = 0; d < hidden; ++d)
        sum += double(router[std::size_t(e) * hidden + d]) * input[d];
      logits[e] = static_cast<float>(sum);
    }
    const auto routing = glm::reference::Route(
        logits.data(), weights.router_bias ? bias.data() : nullptr,
        config.num_experts, config.num_experts_used,
        config.expert_weights_scale, normalize);
    if (routing_case == 1)
      Require(routing.ids == std::vector<std::uint32_t>({0, 1, 2}),
              "fixture tie ordering changed");
    if (routing_case == 2)
      Require(routing.ids == std::vector<std::uint32_t>({3, 1, 4}),
              "fixture bias selection changed");
    glm::reference::RoutedFfn(input, gate_up.data(), down.host.data(), hidden,
                              mid, routing, weights.clamp_exp, output);
    if (shared) {
      glm::reference::DenseFfn(input, sg->host.data(), su->host.data(),
                               sd->host.data(), hidden, config.shared_expert_ff,
                               weights.clamp_shexp, shared_y.data());
      for (std::uint32_t d = 0; d < hidden; ++d)
        output[d] += shared_y[d];
    }
  }
  Stream stream;
  const auto scratch_bytes = glm::FfnScratchBytes(config, tokens);
  Require(scratch_bytes > 0, "scratch query returned zero");
  Buffer scratch(scratch_bytes), dx(x.size() * sizeof(float)),
      dy(x.size() * sizeof(float));
  glm::OpContext ctx{stream.value, scratch.p, scratch_bytes, nullptr};
  const auto layer = dense ? 0U : 1U;
  Hip(hipMemcpyAsync(dx.p, x.data(), x.size() * sizeof(float),
                     hipMemcpyHostToDevice, stream.value));
  Hip(hipMemsetAsync(dy.p, 0xff, x.size() * sizeof(float), stream.value));
  Require(glm::FfnForward(ctx, config, layer, weights, dx.f(), tokens, dy.f()),
          "FfnForward rejected valid fixture");
  Hip(hipMemcpyAsync(got.data(), dy.p, got.size() * sizeof(float),
                     hipMemcpyDeviceToHost, stream.value));
  Hip(hipStreamSynchronize(stream.value));
  Hip(hipMemsetAsync(dy.p, 0xff, x.size() * sizeof(float), stream.value));
  for (std::uint32_t t = 0; t < tokens; t += 6) {
    const auto count = std::min(6U, tokens - t);
    // Each chunk advertises exactly its own required scratch size.
    ctx.scratch_bytes = glm::FfnScratchBytes(config, count);
    Require(glm::FfnForward(ctx, config, layer, weights,
                            dx.f() + std::size_t(t) * hidden, count,
                            dy.f() + std::size_t(t) * hidden),
            "FfnForward rejected chunk");
  }
  Hip(hipMemcpyAsync(chunks.data(), dy.p, chunks.size() * sizeof(float),
                     hipMemcpyDeviceToHost, stream.value));
  Hip(hipStreamSynchronize(stream.value));
  double diff = 0, norm = 0, max_abs = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    Require(std::isfinite(got[i]) && std::isfinite(chunks[i]) &&
                std::isfinite(want[i]),
            "nonfinite FFN output");
    const double error = std::abs(double(got[i]) - want[i]);
    max_abs = std::max(max_abs, error);
    diff += error * error;
    norm += double(want[i]) * want[i];
    if (error > 2e-5 + 2e-4 * std::abs(want[i]))
      throw std::runtime_error("FFN pointwise CPU parity failed at " +
                               std::to_string(i) +
                               " got=" + std::to_string(got[i]) +
                               " want=" + std::to_string(want[i]));
  }
  const auto rel = std::sqrt(diff / std::max(norm, 1e-30));
  const bool exact =
      std::memcmp(got.data(), chunks.data(), got.size() * sizeof(float)) == 0;
  std::cout << "FFN dense=" << dense << " shared=" << shared
            << " clamp=" << clamp << " norm=" << normalize
            << " routing=" << routing_case << " tokens=" << tokens
            << " type=" << static_cast<int>(type) << " rel_l2=" << rel
            << " max_abs=" << max_abs << " chunk_exact=" << exact << '\n';
  Require(rel <= 5e-5, "FFN relative CPU parity failed");
  Require(exact, "FFN chunk parity failed");

  ctx.scratch_bytes = scratch_bytes;
  auto invalid = weights;
  auto& matrix = dense ? invalid.gate : invalid.gate_exps;
  matrix.row_bytes += sizeof(float);
  Require(!glm::FfnForward(ctx, config, layer, invalid, dx.f(), tokens, dy.f()),
          "FFN accepted invalid row stride");
  invalid = weights;
  (dense ? invalid.down : invalid.down_exps).cols += 1;
  Require(!glm::FfnForward(ctx, config, layer, invalid, dx.f(), tokens, dy.f()),
          "FFN accepted invalid matrix shape");
  invalid = weights;
  (dense ? invalid.gate : invalid.gate_exps).type =
      static_cast<GgmlType>(0xffffffffU);
  Require(!glm::FfnForward(ctx, config, layer, invalid, dx.f(), tokens, dy.f()),
          "FFN accepted unsupported weight type");
  ctx.scratch_bytes = scratch_bytes - 1;
  Require(!glm::FfnForward(ctx, config, layer, weights, dx.f(), tokens, dy.f()),
          "FFN accepted undersized scratch");
  Hip(hipStreamSynchronize(stream.value));
  if (dense && clamp && tokens == 1 && !iq) {
    ctx.scratch_bytes = scratch_bytes;
    std::fill(x.begin(), x.end(), std::numeric_limits<float>::quiet_NaN());
    Hip(hipMemcpyAsync(dx.p, x.data(), x.size() * sizeof(float),
                       hipMemcpyHostToDevice, stream.value));
    Require(
        glm::FfnForward(ctx, config, layer, weights, dx.f(), tokens, dy.f()),
        "nonfinite dense launch rejected");
    Hip(hipMemcpyAsync(got.data(), dy.p, got.size() * sizeof(float),
                       hipMemcpyDeviceToHost, stream.value));
    Hip(hipStreamSynchronize(stream.value));
    Require(std::all_of(got.begin(), got.end(),
                        [](float v) { return std::isnan(v); }),
            "nonfinite dense activation was silently sanitized by clamp");
  }
  if (!dense && clamp && tokens == 1 && !iq && routing_case == 0) {
    ctx.scratch_bytes = scratch_bytes;
    std::fill(bias.begin(), bias.end(),
              std::numeric_limits<float>::quiet_NaN());
    Hip(hipMemcpyAsync(db.p, bias.data(), bias.size() * sizeof(float),
                       hipMemcpyHostToDevice, stream.value));
    Require(
        glm::FfnForward(ctx, config, layer, weights, dx.f(), tokens, dy.f()),
        "nonfinite routing launch rejected");
    Hip(hipMemcpyAsync(got.data(), dy.p, got.size() * sizeof(float),
                       hipMemcpyDeviceToHost, stream.value));
    Hip(hipStreamSynchronize(stream.value));
    Require(std::all_of(got.begin(), got.end(),
                        [](float v) { return std::isnan(v); }),
            "nonfinite routing bias was silently sanitized");
    std::fill(bias.begin(), bias.end(), 0.0F);
    Hip(hipMemcpyAsync(db.p, bias.data(), bias.size() * sizeof(float),
                       hipMemcpyHostToDevice, stream.value));
    std::fill(x.begin(), x.end(), std::numeric_limits<float>::quiet_NaN());
    Hip(hipMemcpyAsync(dx.p, x.data(), x.size() * sizeof(float),
                       hipMemcpyHostToDevice, stream.value));
    Require(
        glm::FfnForward(ctx, config, layer, weights, dx.f(), tokens, dy.f()),
        "nonfinite activation launch rejected");
    Hip(hipMemcpyAsync(got.data(), dy.p, got.size() * sizeof(float),
                       hipMemcpyDeviceToHost, stream.value));
    Hip(hipStreamSynchronize(stream.value));
    Require(std::all_of(got.begin(), got.end(),
                        [](float v) { return std::isnan(v); }),
            "nonfinite activation was silently sanitized by clamp");
  }
}
}  // namespace
int main() {
  try {
    for (auto tokens : {1U, 8U, 9U, 32U, 33U}) {
      for (bool clamp : {false, true}) {
        Run(true, false, clamp, true, 0, tokens, GgmlType::kF32);
        Run(false, true, clamp, true, 0, tokens, GgmlType::kF32);
      }
      Run(false, false, false, false, 1, tokens, GgmlType::kF32);
      Run(false, true, true, true, 2, tokens, GgmlType::kF32);
    }
    for (auto type :
         {GgmlType::kIQ2_XXS, GgmlType::kIQ2_S, GgmlType::kIQ3_XXS}) {
      Run(false, true, true, true, 2, 9, type);
      Run(false, false, false, false, 1, 1, type);
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
