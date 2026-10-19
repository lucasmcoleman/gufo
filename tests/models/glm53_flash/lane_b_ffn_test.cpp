// Full FFN parity against the independent scalar routing/SwiGLU oracle.
// Exact scratch and activation allocations expose ragged batch overruns.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <hip/hip_fp16.h>
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
    Require(row_bytes > 0, "invalid fixture row encoding");
    if (type == GgmlType::kF32 || type == GgmlType::kBF16) {
      std::uniform_real_distribution<float> dist(-0.6F, 0.6F);
      for (auto& v : host)
        v = dist(rng);
      if (type == GgmlType::kF32) {
        std::memcpy(bytes.data(), host.data(), bytes.size());
      } else {
        // Independently round finite F32 fixture weights to BF16, nearest
        // with ties to even. The oracle consumes precisely those rounded bits.
        for (std::size_t i = 0; i < host.size(); ++i) {
          const auto bits = std::bit_cast<std::uint32_t>(host[i]);
          const auto rounded = static_cast<std::uint16_t>(
              (bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
          std::memcpy(bytes.data() + i * sizeof(rounded), &rounded,
                      sizeof(rounded));
          host[i] = std::bit_cast<float>(std::uint32_t{rounded} << 16);
        }
      }
    } else {
      for (auto& v : bytes)
        v = static_cast<std::uint8_t>(rng());
      const auto block_elements = type == GgmlType::kQ8_0 ? 32U : 256U;
      const auto block_bytes = row_bytes / (cols / block_elements);
      for (std::size_t offset = 0; offset < bytes.size();
           offset += block_bytes) {
        // Q6_K stores its fp16 scale last; IQ and Q8_0 store it first.
        // Payload bytes remain independent, including signs and subscales.
        const auto scale = static_cast<std::uint16_t>(
            (type == GgmlType::kQ6_K ? 0x0800 : 0x1400) + rng() % 1024);
        const auto scale_offset = type == GgmlType::kQ6_K ? block_bytes - 2 : 0;
        std::memcpy(bytes.data() + offset + scale_offset, &scale,
                    sizeof(scale));
      }
      for (std::size_t row = 0; row < std::size_t(rows) * experts; ++row) {
        const auto* src = bytes.data() + row * row_bytes;
        auto* dst = host.data() + row * cols;
        switch (type) {
          case GgmlType::kQ6_K:
            gufo::quant::DequantizeQ6_K(src, dst, cols);
            break;
          case GgmlType::kQ8_0:
            gufo::quant::DequantizeQ8_0(src, dst, cols);
            break;
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

// The routed IQ projections run on DS4's vendored llama.cpp kernels unless
// GUFO_GLM53_SCALAR_MOE is set (moe.hip.cpp Ds4Enabled). Those kernels round
// their activations to Q8_1 first, so the oracle applies the same rounding at
// the same points and keeps the exact-path tolerances.
bool Ds4Active() {
  const char* v = std::getenv("GUFO_GLM53_SCALAR_MOE");
  return !(v && *v && *v != '0');
}

// Q8_1 activation rounding, 32 values per scale. Prefill (MMQ, tokens > 8)
// keeps d = 1 / (127 / amax) in F32 (quantize_mmq_q8_1, D4 layout); decode
// (MMVQ) uses d = amax / 127 and stores it as fp16 (quantize_q8_1).
void Q8Round(float* x, std::size_t n, bool vec) {
  for (std::size_t b = 0; b < n; b += 32) {
    const auto end = std::min(n, b + 32);
    float amax = 0;
    for (auto i = b; i < end; ++i)
      amax = std::max(amax, std::abs(x[i]));
    if (amax == 0) {
      std::fill(x + b, x + end, 0.0F);
      continue;
    }
    const float d_inv = 127.0F / amax;
    const float d = vec ? amax / 127.0F : 1.0F / d_inv;
    const float scale = vec ? __half2float(__float2half(d)) : d;
    for (auto i = b; i < end; ++i)
      x[i] = std::round(vec ? x[i] / d : x[i] * d_inv) * scale;
  }
}

double Dot(const float* a, const float* b, std::size_t n) {
  double sum = 0;
  for (std::size_t i = 0; i < n; ++i)
    sum += double(a[i]) * b[i];
  return sum;
}

// reference::RoutedFfn with optional Q8_1 rounding of the gate/up input and
// of each route's down input (the SwiGLU output). The gate/up input is the
// host fixture, identical on both sides. For the down input, `gpu_act`
// ([routes][mid] for this token, from the ffn_moe_act tap) replaces the CPU's
// SwiGLU output when given: a value on a .5 boundary may otherwise round
// differently on the two sides. The CPU and GPU activations are compared on
// their own (CheckAct) so the substitution hides nothing.
void RoutedFfnQ8(const float* x, const float* gate_up, const float* down,
                 std::size_t hidden, std::size_t mid,
                 const glm::reference::Routing& routing, float clamp,
                 bool q_gate_up, bool q_down, bool vec, const float* gpu_act,
                 float* y) {
  std::vector<float> xin(x, x + hidden), act(mid);
  if (q_gate_up)
    Q8Round(xin.data(), hidden, vec);
  std::fill(y, y + hidden, 0.0F);
  for (std::size_t r = 0; r < routing.ids.size(); ++r) {
    const auto e = routing.ids[r];
    const float* gate = gate_up + e * 2 * mid * hidden;
    const float* up = gate + mid * hidden;
    if (gpu_act)
      std::copy_n(gpu_act + r * mid, mid, act.data());
    else
      for (std::size_t i = 0; i < mid; ++i)
        act[i] = glm::reference::ClampedSwiGlu(
            static_cast<float>(Dot(gate + i * hidden, xin.data(), hidden)),
            static_cast<float>(Dot(up + i * hidden, xin.data(), hidden)),
            clamp);
    if (q_down)
      Q8Round(act.data(), mid, vec);
    const float* w = down + e * hidden * mid;
    for (std::size_t d = 0; d < hidden; ++d)
      y[d] += routing.weights[r] *
              static_cast<float>(Dot(w + d * mid, act.data(), mid));
  }
}

void Run(bool dense, bool shared, bool clamp, bool normalize, int routing_case,
         std::uint32_t tokens, GgmlType type,
         GgmlType shared_type = GgmlType::kF32, std::uint32_t dimension = 0) {
  std::cout << "case dense=" << dense << " shared=" << shared
            << " clamp=" << clamp << " norm=" << normalize
            << " routing=" << routing_case << " tokens=" << tokens
            << " type=" << static_cast<int>(type)
            << " shared_type=" << static_cast<int>(shared_type)
            << " dimension=" << dimension << std::endl;
  const bool packed = type != GgmlType::kF32 && type != GgmlType::kBF16;
  glm::Config config;
  config.num_layers = 2;
  config.leading_dense_layers = 1;
  config.hidden_size = dimension ? dimension : (packed ? 256 : 19);
  config.dense_ff = dimension ? dimension : (packed ? 256 : 37);
  config.expert_ff = dimension ? dimension : (packed ? 256 : 23);
  config.num_experts = 5;
  config.num_experts_used = 3;
  config.shared_expert_ff =
      shared ? (dimension ? dimension : (packed ? 256 : 29)) : 0;
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
                                    shared_type, rng);
      su = std::make_unique<Weight>(config.shared_expert_ff, hidden, 1,
                                    shared_type, rng);
      sd = std::make_unique<Weight>(hidden, config.shared_expert_ff, 1,
                                    shared_type, rng);
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
  // Which projections take DS4's Q8_1 path (moe.hip.cpp UseDs4/UseDs4Down).
  const bool q_gate_up = !dense && Ds4Active() && type == GgmlType::kIQ2_XXS;
  const bool q_down = !dense && Ds4Active() &&
                      (type == GgmlType::kIQ2_S || type == GgmlType::kIQ3_XXS);
  const bool q8 = q_gate_up || q_down;
  // The 6-token chunk pass runs the decode kernels; with Q8_1 rounding its
  // oracle differs from the full pass whenever that pass is a prefill.
  const bool chunk_differs = q8 && tokens > 8;
  const auto routes = config.num_experts_used;

  // GPU first: with q_down the oracle rounds the GPU's own down input, so the
  // ffn_moe_act and ffn_moe_topk taps are captured (in token order, across
  // the chunked calls too).
  std::vector<float> got(x.size()), chunks(x.size()), act_full, act_chunks;
  std::vector<std::int32_t> ids_full, ids_chunks;
  std::vector<float>* act_sink = &act_full;
  std::vector<std::int32_t>* ids_sink = &ids_full;
  Stream stream;
  const glm::TapFn capture = [&](std::string_view name, std::uint32_t,
                                 const glm::TapView& view) {
    if (name != "ffn_moe_act" && name != "ffn_moe_topk")
      return;
    Hip(hipStreamSynchronize(stream.value));
    if (name == "ffn_moe_act") {
      const auto at = act_sink->size();
      act_sink->resize(at + view.count());
      Hip(hipMemcpy(act_sink->data() + at, view.device,
                    view.count() * sizeof(float), hipMemcpyDeviceToHost));
    } else {
      const auto at = ids_sink->size();
      ids_sink->resize(at + view.count());
      Hip(hipMemcpy(ids_sink->data() + at, view.device,
                    view.count() * sizeof(std::int32_t),
                    hipMemcpyDeviceToHost));
    }
  };
  const auto scratch_bytes = glm::FfnScratchBytes(config, tokens);
  Require(scratch_bytes > 0, "scratch query returned zero");
  Buffer scratch(scratch_bytes), dx(x.size() * sizeof(float)),
      dy(x.size() * sizeof(float));
  glm::OpContext ctx{stream.value, scratch.p, scratch_bytes,
                     q_down ? &capture : nullptr};
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
  act_sink = &act_chunks;
  ids_sink = &ids_chunks;
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
  ctx.tap = nullptr;
  if (q_down) {
    const auto rows = std::size_t(tokens) * routes;
    Require(act_full.size() == rows * mid && act_chunks.size() == rows * mid &&
                ids_full.size() == rows && ids_chunks.size() == rows,
            "ffn_moe_act/topk taps missing");
  }

  // Exact-path tolerances, used for every comparison below.
  struct Gate {
    double diff = 0, norm = 0, max_abs = 0;
    void Add(const char* what, std::size_t i, float got, float want) {
      if (!std::isfinite(got) || !std::isfinite(want))
        throw std::runtime_error(std::string("nonfinite FFN ") + what);
      const double error = std::abs(double(got) - want);
      max_abs = std::max(max_abs, error);
      diff += error * error;
      norm += double(want) * want;
      if (error > 2e-5 + 2e-4 * std::abs(want))
        throw std::runtime_error(std::string("FFN ") + what +
                                 " pointwise parity failed at " +
                                 std::to_string(i) +
                                 " got=" + std::to_string(got) +
                                 " want=" + std::to_string(want));
    }
    double Rel() const { return std::sqrt(diff / std::max(norm, 1e-30)); }
  } full, chunk, act_gate, act_chunk_gate;

  std::vector<float> want(x.size()), want_chunks(x.size());
  std::vector<float> logits(config.num_experts), shared_y(hidden), cpu_act(mid);
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
    const auto row = std::size_t(t) * routes;
    if (q_down) {
      // The substituted GPU activations must belong to the same experts in
      // the same order, and match the CPU's own SwiGLU to exact tolerance.
      for (std::uint32_t r = 0; r < routes; ++r)
        Require(ids_full[row + r] == std::int32_t(routing.ids[r]) &&
                    ids_chunks[row + r] == std::int32_t(routing.ids[r]),
                "GPU expert selection differs from the oracle");
      for (std::uint32_t r = 0; r < routes; ++r) {
        const float* gw = gate_up.data() + routing.ids[r] * 2 * mid * hidden;
        for (std::size_t i = 0; i < mid; ++i)
          cpu_act[i] = glm::reference::ClampedSwiGlu(
              static_cast<float>(Dot(gw + i * hidden, input, hidden)),
              static_cast<float>(Dot(gw + (mid + i) * hidden, input, hidden)),
              weights.clamp_exp);
        for (std::size_t i = 0; i < mid; ++i) {
          act_gate.Add("act", (row + r) * mid + i, act_full[(row + r) * mid + i],
                       cpu_act[i]);
          act_chunk_gate.Add("chunk act", (row + r) * mid + i,
                             act_chunks[(row + r) * mid + i], cpu_act[i]);
        }
      }
    }
    if (q8)
      RoutedFfnQ8(input, gate_up.data(), down.host.data(), hidden, mid,
                  routing, weights.clamp_exp, q_gate_up, q_down, tokens <= 8,
                  q_down ? act_full.data() + row * mid : nullptr, output);
    else
      glm::reference::RoutedFfn(input, gate_up.data(), down.host.data(),
                                hidden, mid, routing, weights.clamp_exp,
                                output);
    auto* output_chunks = want_chunks.data() + std::size_t(t) * hidden;
    if (chunk_differs)
      RoutedFfnQ8(input, gate_up.data(), down.host.data(), hidden, mid,
                  routing, weights.clamp_exp, q_gate_up, q_down, /*vec=*/true,
                  q_down ? act_chunks.data() + row * mid : nullptr,
                  output_chunks);
    if (shared) {
      glm::reference::DenseFfn(input, sg->host.data(), su->host.data(),
                               sd->host.data(), hidden, config.shared_expert_ff,
                               weights.clamp_shexp, shared_y.data());
      for (std::uint32_t d = 0; d < hidden; ++d) {
        output[d] += shared_y[d];
        if (chunk_differs)
          output_chunks[d] += shared_y[d];
      }
    }
  }
  for (std::size_t i = 0; i < got.size(); ++i) {
    full.Add("CPU", i, got[i], want[i]);
    if (chunk_differs)
      chunk.Add("chunk Q8_1", i, chunks[i], want_chunks[i]);
    else
      Require(std::isfinite(chunks[i]), "nonfinite FFN chunk output");
  }
  const bool exact =
      std::memcmp(got.data(), chunks.data(), got.size() * sizeof(float)) == 0;
  std::cout << "FFN dense=" << dense << " shared=" << shared
            << " clamp=" << clamp << " norm=" << normalize
            << " routing=" << routing_case << " tokens=" << tokens
            << " type=" << static_cast<int>(type) << " q8=" << q8
            << " rel_l2=" << full.Rel() << " max_abs=" << full.max_abs;
  if (q_down)
    std::cout << " act_rel_l2=" << act_gate.Rel()
              << " chunk_act_rel_l2=" << act_chunk_gate.Rel();
  if (chunk_differs)
    std::cout << " chunk_rel_l2=" << chunk.Rel();
  std::cout << " chunk_exact=" << exact << '\n';
  Require(full.Rel() <= 5e-5, "FFN relative CPU parity failed");
  if (q_down)
    Require(act_gate.Rel() <= 5e-5 && act_chunk_gate.Rel() <= 5e-5,
            "FFN activation relative parity failed");
  if (chunk_differs)
    Require(chunk.Rel() <= 5e-5, "FFN chunk relative Q8_1 parity failed");
  else  // Both passes run the same kernels: bitwise chunk parity.
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
  invalid = weights;
  invalid.clamp_exp = -0.1F;
  Require(!glm::FfnForward(ctx, config, layer, invalid, dx.f(), tokens, dy.f()),
          "FFN accepted negative dense/routed clamp");
  if (!dense && shared) {
    invalid = weights;
    invalid.clamp_shexp = -0.1F;
    Require(
        !glm::FfnForward(ctx, config, layer, invalid, dx.f(), tokens, dy.f()),
        "FFN accepted negative shared clamp");
  }
  ctx.scratch_bytes = scratch_bytes - 1;
  Require(!glm::FfnForward(ctx, config, layer, weights, dx.f(), tokens, dy.f()),
          "FFN accepted undersized scratch");
  Hip(hipStreamSynchronize(stream.value));
  if (dense && clamp && tokens == 1 && !packed) {
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
  if (!dense && clamp && tokens == 1 && !packed && routing_case == 0) {
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
    for (auto tokens : {1U, 8U, 9U, 33U}) {
      for (bool clamp : {false, true}) {
        Run(true, false, clamp, true, 0, tokens, GgmlType::kBF16);
        Run(false, true, clamp, true, 0, tokens, GgmlType::kBF16,
            GgmlType::kBF16);
        Run(true, false, clamp, true, 0, tokens, GgmlType::kQ6_K);
        Run(false, true, clamp, true, 2, tokens, GgmlType::kF32,
            GgmlType::kQ6_K, 256);
        for (auto dimension : {32U, 128U, 256U}) {
          Run(true, false, clamp, true, 0, tokens, GgmlType::kQ8_0,
              GgmlType::kF32, dimension);
          Run(false, true, clamp, true, 2, tokens, GgmlType::kF32,
              GgmlType::kQ8_0, dimension);
        }
      }
      // Real mixed encoding: packed IQ routed gate/up/down plus quantized
      // shared gate/up/down. Each format is decoded independently on CPU.
      for (auto type :
           {GgmlType::kIQ2_XXS, GgmlType::kIQ2_S, GgmlType::kIQ3_XXS}) {
        for (auto shared_type : {GgmlType::kQ6_K, GgmlType::kQ8_0})
          Run(false, true, true, true, 2, tokens, type, shared_type);
      }
    }
    // Cooperative projection lanes must mask both reduction tails and the
    // final partial wave group without changing per-token arithmetic.
    for (auto tokens : {1U, 8U, 33U}) {
      for (auto type : {GgmlType::kF32, GgmlType::kBF16}) {
        Run(true, false, false, true, 0, tokens, type, type, 65);
        Run(false, true, false, true, 0, tokens, type, type, 129);
      }
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
