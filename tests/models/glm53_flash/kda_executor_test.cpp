// GPU KDA mixer (executor.hpp KdaMixerForward) against the CPU
// ReferenceModel on a synthetic one-layer model: hidden 256, 2 heads of 128,
// 21 tokens. The oracle is ReferenceModel::Kda itself: a tap captures layer 0
// `input_layernorm.out` (fed to the GPU) and `self_attn.out` (the expected
// output), then stops the forward. Both sides read the same hand-built
// TensorRefs (GgufWeightSource dequantizes Q8_0 on the CPU). Two encodings:
// all F32, and Q8_0 q|k|v, ssm_f_b (reduction 128), ssm_beta and attn_output
// with F32 for the rest. Per encoding, at rel-L2 <= 1e-5: a 21-token prefill
// (GEMM batch > 8); after ResetKdaState, prefill(19) + 2 single-token decodes
// on the same state; and the 21-token call with scratch for 6 tokens (chunks
// 6+6+6+3, GEMM batch <= 8).
// Also: q|k|v in different encodings and a non-KDA layer are refused.
// Allocates under 16 MB of device memory.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/glm53_flash/config.hpp"
#include "src/models/glm53_flash/executor.hpp"
#include "src/models/glm53_flash/reference.hpp"
#include "src/models/glm53_flash/weights.hpp"

namespace glm = gufo::models::glm53_flash;
namespace ref = gufo::models::glm53_flash::reference;
using gufo::core::GgmlType;

namespace {

constexpr std::uint32_t kHidden = 256;
constexpr std::uint32_t kHeads = 2;
constexpr std::uint32_t kD = 128;
constexpr std::uint32_t kVocab = 16;
constexpr std::uint32_t kPrefill = 19;
constexpr std::uint32_t kChunk = 6;
constexpr double kTol = 1e-5;
const std::vector<std::int32_t> kIds = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5,
                                        8, 9, 7, 9, 3, 2, 3, 8, 4, 6};
int failures = 0;

void CheckHip(hipError_t error, const char* what) {
  if (error != hipSuccess) {
    throw std::runtime_error(std::string(what) + ": " +
                             hipGetErrorString(error));
  }
}

double RelL2(const std::vector<float>& got, const std::vector<float>& want,
             std::size_t begin = 0, std::size_t end = SIZE_MAX) {
  end = std::min(end, want.size());
  double num = 0.0;
  double den = 0.0;
  for (std::size_t i = begin; i < end; ++i) {
    const double d = static_cast<double>(got[i]) - want[i];
    num += d * d;
    den += static_cast<double>(want[i]) * want[i];
  }
  return den == 0.0 ? std::sqrt(num) : std::sqrt(num / den);
}

void Check(const std::string& what, bool ok) {
  std::cout << (ok ? "ok   " : "FAIL ") << what << "\n";
  if (!ok)
    ++failures;
}

void CheckRel(const std::string& what, const std::vector<float>& got,
              const std::vector<float>& want, std::size_t begin = 0,
              std::size_t end = SIZE_MAX) {
  const double rel = RelL2(got, want, begin, end);
  const bool ok =
      got.size() == want.size() && std::isfinite(rel) && rel <= kTol;
  std::cout << (ok ? "ok   " : "FAIL ") << what << " rel_l2 " << rel << "\n";
  if (!ok)
    ++failures;
}

/// Q8_0 with power-of-two scales (exact fp16), as reference_e2e_test does.
std::vector<std::uint8_t> EncodeQ8_0(const std::vector<float>& m,
                                     std::size_t rows, std::size_t cols) {
  const std::size_t row_bytes =
      gufo::quant::QuantizedRowBytes(GgmlType::kQ8_0, cols);
  std::vector<std::uint8_t> q8(rows * row_bytes);
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t b = 0; b < cols / 32; ++b) {
      const float* x = m.data() + (r * cols) + (b * 32);
      float amax = 0.0F;
      for (int i = 0; i < 32; ++i)
        amax = std::max(amax, std::fabs(x[i]));
      const int e = static_cast<int>(std::ceil(std::log2(amax / 127.0F)));
      if (e < -14 || e > 15)
        throw std::runtime_error("Q8_0 scale outside fp16 normal range");
      const float d = std::ldexp(1.0F, e);
      std::uint8_t* blk = q8.data() + (r * row_bytes) + (b * 34);
      const auto dh = static_cast<std::uint16_t>((e + 15) << 10);
      std::memcpy(blk, &dh, 2);
      for (int i = 0; i < 32; ++i) {
        const auto q = static_cast<std::int8_t>(std::lround(x[i] / d));
        std::memcpy(blk + 2 + i, &q, 1);
      }
    }
  }
  return q8;
}

glm::Config MakeConfig() {
  glm::Config c;
  c.num_layers = 1;
  c.num_layers_all = 1;
  c.hidden_size = kHidden;
  c.vocab_size = kVocab;
  c.context_length = 64;
  c.mixers = {glm::Config::Mixer::kKda};
  c.hc_count = 4;
  c.hc_sinkhorn_iters = 20;
  c.num_heads = kHeads;
  c.kda_head_dim = kD;
  c.ssm_conv_kernel = 4;
  c.kda_gate_lower_bound = -5.0F;
  return c;
}

/// Host storage for the synthetic weights; TensorRefs point into it.
struct Model {
  glm::Config config = MakeConfig();
  std::deque<std::string> names;
  std::deque<std::vector<float>> f32;
  std::deque<std::vector<std::uint8_t>> q8;
  glm::ModelWeights weights;
};

void BuildModel(Model& m, bool quantized, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> n01(0.0F, 1.0F);
  const glm::Config& c = m.config;
  const auto ref_of = [&](const std::string& name, std::uint64_t cols,
                          std::uint64_t rows, float mean, float sd,
                          bool q8 = false) {
    std::vector<float> data(cols * rows);
    for (auto& x : data)
      x = mean + (sd * n01(rng));
    m.names.push_back(name);
    glm::TensorRef t;
    t.cols = cols;
    t.rows = rows;
    t.name = m.names.back();
    if (q8) {
      m.q8.push_back(EncodeQ8_0(data, rows, cols));
      t.data = m.q8.back().data();
      t.type = GgmlType::kQ8_0;
    } else {
      m.f32.push_back(std::move(data));
      t.data = m.f32.back().data();
      t.type = GgmlType::kF32;
    }
    return t;
  };
  const std::uint64_t dim = c.KdaDim();
  const std::uint64_t mix = c.HcMixDim();
  const float proj_sd = 1.0F / std::sqrt(static_cast<float>(kHidden));
  m.weights.config = c;
  m.weights.token_embd = ref_of("token_embd.weight", kHidden, kVocab, 0, 1);
  glm::LayerWeights l;
  l.mixer = glm::Config::Mixer::kKda;
  l.dense_ffn = true;
  l.hc_attn = {ref_of("blk.0.hc_attn_fn.weight", c.HcDim(), mix, 0, 0.05F),
               ref_of("blk.0.hc_attn_base.weight", mix, 1, 0, 0.5F),
               ref_of("blk.0.hc_attn_scale.weight", 3, 1, 1, 0.2F)};
  l.attn_norm = ref_of("blk.0.attn_norm.weight", kHidden, 1, 1, 0.1F);
  const bool q = quantized;
  l.kda_q = ref_of("blk.0.attn_q.weight", kHidden, dim, 0, proj_sd, q);
  l.kda_k = ref_of("blk.0.attn_k.weight", kHidden, dim, 0, proj_sd, q);
  l.kda_v = ref_of("blk.0.attn_v.weight", kHidden, dim, 0, proj_sd, q);
  const auto conv = [&](const char* part) {
    return ref_of(std::string("blk.0.ssm_conv1d_") + part + ".weight",
                  c.ssm_conv_kernel, dim, 0, 0.5F);
  };
  l.kda_conv_q = conv("q");
  l.kda_conv_k = conv("k");
  l.kda_conv_v = conv("v");
  l.kda_f_a = ref_of("blk.0.ssm_f_a.weight", kHidden, kD, 0, proj_sd);
  l.kda_f_b = ref_of("blk.0.ssm_f_b.weight", kD, dim, 0, 0.15F, q);
  l.kda_dt_bias = ref_of("blk.0.ssm_dt.bias", dim, 1, 0, 0.5F);
  l.kda_a = ref_of("blk.0.ssm_a", kHeads, 1, 0, 0.5F);
  for (float& a : m.f32.back())
    a = -std::exp(a);  // ssm_a = -exp(A_log)
  l.kda_beta = ref_of("blk.0.ssm_beta.weight", kHidden, kHeads, 0, proj_sd, q);
  l.kda_g_a = ref_of("blk.0.ssm_g_a.weight", kHidden, kD, 0, proj_sd);
  l.kda_g_b = ref_of("blk.0.ssm_g_b.weight", kD, dim, 0, 0.15F);
  l.kda_norm = ref_of("blk.0.ssm_norm.weight", kD, 1, 1, 0.2F);
  l.attn_out = ref_of("blk.0.attn_output.weight", dim, kHidden, 0,
                      1.0F / std::sqrt(static_cast<float>(dim)), q);
  m.weights.layers = {l};
}

struct StopForward {};

/// Layer-0 mixer input and output of the CPU ReferenceModel for kIds.
void RunOracle(const Model& m, std::vector<float>* x, std::vector<float>* y) {
  ref::GgufWeightSource source(m.weights);
  ref::ReferenceModel model(m.config, source);
  const ref::TapFn tap = [&](std::string_view name, int layer,
                             const float* data, std::size_t width,
                             std::uint32_t n) {
    if (layer != 0)
      return;
    if (name == "input_layernorm.out")
      x->assign(data, data + (width * n));
    if (name == "self_attn.out") {
      y->assign(data, data + (width * n));
      throw StopForward{};
    }
  };
  try {
    (void)model.Forward(kIds, false, tap);
    throw std::runtime_error("oracle did not reach self_attn.out");
  } catch (const StopForward&) {
  }
}

class DeviceFloats {
public:
  explicit DeviceFloats(std::size_t n) : n_(n) {
    CheckHip(hipMalloc(&ptr_, n * sizeof(float)), "hipMalloc");
  }
  ~DeviceFloats() { (void)hipFree(ptr_); }
  DeviceFloats(const DeviceFloats&) = delete;
  DeviceFloats& operator=(const DeviceFloats&) = delete;
  [[nodiscard]] float* get() const noexcept { return ptr_; }
  void Upload(const std::vector<float>& h) {
    CheckHip(
        hipMemcpy(ptr_, h.data(), n_ * sizeof(float), hipMemcpyHostToDevice),
        "hipMemcpy H2D");
  }
  [[nodiscard]] std::vector<float> Download() const {
    std::vector<float> h(n_);
    CheckHip(
        hipMemcpy(h.data(), ptr_, n_ * sizeof(float), hipMemcpyDeviceToHost),
        "hipMemcpy D2H");
    return h;
  }

private:
  float* ptr_{nullptr};
  std::size_t n_{0};
};

void RunCase(bool quantized, unsigned seed) {
  const std::string name = quantized ? "q8_0" : "f32";
  Model m;
  BuildModel(m, quantized, seed);
  const glm::Config& c = m.config;
  std::vector<float> x;
  std::vector<float> want;
  RunOracle(m, &x, &want);
  const auto T = static_cast<std::uint32_t>(kIds.size());

  glm::DeviceArena arena;
  glm::KdaLayerDevice w;
  std::string error;
  if (!glm::UploadKdaLayer(arena, c, m.weights.layers[0], 0, &w, &error)) {
    Check(name + " upload: " + error, false);
    return;
  }
  Check(name + " qkv stacked as one " +
            std::to_string(static_cast<int>(w.qkv.type)) + " matrix of " +
            std::to_string(w.qkv.rows) + " rows",
        w.qkv.rows == 3ULL * c.KdaDim() && w.qkv.cols == kHidden);
  glm::KdaState state;
  if (!glm::AllocateKdaState(arena, c, &state)) {
    Check(name + " state allocation", false);
    return;
  }
  const std::size_t scratch_bytes = glm::KdaScratchBytes(c, T);
  void* scratch = arena.Allocate(scratch_bytes);
  DeviceFloats dx(x.size());
  DeviceFloats dy(want.size());
  dx.Upload(x);
  hipStream_t stream = nullptr;
  CheckHip(hipStreamCreate(&stream), "hipStreamCreate");
  glm::OpContext ctx{.stream = stream,
                     .scratch = scratch,
                     .scratch_bytes = scratch_bytes,
                     .tap = nullptr};
  const auto run = [&](std::uint32_t t0, std::uint32_t n) {
    return glm::KdaMixerForward(ctx, c, 0, w, state, dx.get() + (t0 * kHidden),
                                n, dy.get() + (t0 * kHidden));
  };

  // Full prefill.
  bool ok = run(0, T);
  CheckHip(hipStreamSynchronize(stream), "sync");
  Check(name + " prefill(21) launch", ok);
  CheckRel(name + " prefill(21) vs oracle", dy.Download(), want);

  // Prefill then two decodes from a fresh state.
  ok = glm::ResetKdaState(c, state, stream);
  CheckHip(hipMemsetAsync(dy.get(), 0, want.size() * sizeof(float), stream),
           "memset");
  ok = ok && run(0, kPrefill);
  for (std::uint32_t t = kPrefill; t < T; ++t)
    ok = ok && run(t, 1);
  CheckHip(hipStreamSynchronize(stream), "sync");
  Check(name + " prefill(19)+decode(2) launch", ok);
  const auto split = dy.Download();
  CheckRel(name + " prefill(19) rows vs oracle", split, want, 0,
           kPrefill * kHidden);
  for (std::uint32_t t = kPrefill; t < T; ++t) {
    CheckRel(name + " decode token " + std::to_string(t) + " vs oracle", split,
             want, t * kHidden, (t + 1) * kHidden);
  }

  // Scratch for six tokens: the call runs in chunks.
  ok = glm::ResetKdaState(c, state, stream);
  CheckHip(hipMemsetAsync(dy.get(), 0, want.size() * sizeof(float), stream),
           "memset");
  ctx.scratch_bytes = glm::KdaScratchBytes(c, kChunk);
  ok = ok && run(0, T);
  CheckHip(hipStreamSynchronize(stream), "sync");
  Check(name + " chunked prefill launch", ok);
  CheckRel(name + " chunked prefill(21) vs oracle", dy.Download(), want);
  ctx.scratch_bytes = glm::KdaScratchBytes(c, 1) - 1;
  Check(name + " scratch below one token refused", !run(0, 1));
  ctx.scratch_bytes = scratch_bytes;
  Check(name + " non-KDA layer index refused",
        !glm::KdaMixerForward(ctx, c, 1, w, state, dx.get(), 1, dy.get()));
  CheckHip(hipStreamDestroy(stream), "hipStreamDestroy");
  std::cout << name << " device bytes " << arena.bytes() << "\n";
}

void CheckRefusals() {
  Model m;
  BuildModel(m, true, 31);
  glm::LayerWeights l = m.weights.layers[0];
  // F32 k next to Q8_0 q / v cannot be stacked.
  Model f;
  BuildModel(f, false, 31);
  l.kda_k = f.weights.layers[0].kda_k;
  glm::DeviceArena arena;
  glm::KdaLayerDevice w;
  std::string error;
  const bool mixed = glm::UploadKdaLayer(arena, m.config, l, 0, &w, &error);
  Check("mixed q|k|v encodings refused (" + error + ")", !mixed);
  glm::Config mla = m.config;
  mla.mixers = {glm::Config::Mixer::kMla};
  const bool not_kda =
      glm::UploadKdaLayer(arena, mla, m.weights.layers[0], 0, &w, &error);
  Check("MLA layer refused (" + error + ")", !not_kda);
}

}  // namespace

int main() {
  try {
    int devices = 0;
    if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
      std::cerr << "no HIP device\n";
      return 77;
    }
    RunCase(false, 21);
    RunCase(true, 22);
    CheckRefusals();
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << "\n";
    return 1;
  }
  if (failures != 0) {
    std::cerr << failures << " check(s) failed\n";
    return 1;
  }
  std::cout << "glm53_flash kda_executor: PASS\n";
  return 0;
}
