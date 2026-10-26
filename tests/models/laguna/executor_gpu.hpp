#ifndef GUFO_TESTS_MODELS_LAGUNA_EXECUTOR_GPU_HPP_
#define GUFO_TESTS_MODELS_LAGUNA_EXECUTOR_GPU_HPP_

// Tiny random Laguna models in the GPU's storage formats for the executor
// tests: the real model's geometry (head_dim 128, GQA 6 and 9, partial RoPE
// with YaRN on full layers, routed Q6_K/Q5_K/IQ4_XS, Q8_0 dense and shared
// experts, Q6_K attention and output) at a small size. Every weight is
// random. Reuses the phase 3 helpers (RandomQuant, RandomF32, HostMat, device
// helpers) from ffn_gpu.hpp and attention_gpu.hpp.

#include <deque>
#include <memory>
#include <span>

#include "src/models/laguna/budget.hpp"
#include "src/models/laguna/kernels/device_model.hpp"
#include "src/models/laguna/kernels/executor.hpp"
#include "tests/models/laguna/attention_gpu.hpp"

namespace gufo::testing::laguna {

namespace exec = gufo::models::laguna::rocm;

struct ModelSpec {
  std::uint32_t layers{6};  // L0 full+dense, L1-3 sliding, L4 full, L5 sliding
  std::uint32_t hidden{512};
  std::uint32_t kv_heads{2};
  std::uint32_t heads_full{12};  // GQA 6
  std::uint32_t heads_swa{18};   // GQA 9
  std::uint32_t rotary_full{64};
  std::uint32_t rotary_swa{128};
  std::uint32_t window{12};
  std::uint32_t experts{32};
  std::uint32_t used{10};
  std::uint32_t expert_ff{256};
  std::uint32_t shared_ff{320};
  std::uint32_t dense_ff{768};
  std::uint32_t vocab{640};
  std::uint64_t seed{4242};
  /// Routed expert type of layers 1.. in turn (cycled).
  std::vector<GgmlType> routed{GgmlType::kQ6_K, GgmlType::kQ5_K,
                               GgmlType::kIQ4_XS};
};

class TinyModel {
public:
  explicit TinyModel(const ModelSpec& spec = {})
      : spec_(spec), rng_(spec.seed) {
    ml::Config& c = weights_.config;
    c.num_layers = spec.layers;
    c.hidden_size = spec.hidden;
    c.ffn_size = spec.dense_ff;
    c.vocab_size = spec.vocab;
    c.context_length = 262144;
    c.rms_eps = 1e-6F;
    c.num_kv_heads = spec.kv_heads;
    c.head_dim = 128;
    c.sliding_window = spec.window;
    c.rope_theta_full = 500000.0F;
    c.rope_theta_swa = 10000.0F;
    c.rotary_dim_full = spec.rotary_full;
    c.rotary_dim_swa = spec.rotary_swa;
    c.yarn.factor = 32.0F;
    c.yarn.original_context = 8192;
    c.yarn.beta_fast = 32.0F;
    c.yarn.beta_slow = 1.0F;
    c.rope_attn_factor = 1.0F;
    c.num_experts = spec.experts;
    c.num_experts_used = spec.used;
    c.expert_ff = spec.expert_ff;
    c.shared_expert_ff = spec.shared_ff;
    c.expert_weights_norm = true;
    c.expert_weights_scale = 2.5F;
    c.leading_dense_layers = 1;
    for (std::uint32_t il = 0; il < spec.layers; ++il) {
      c.heads_per_layer.push_back(il % 4 == 0 ? spec.heads_full
                                              : spec.heads_swa);
    }
    const std::size_t H = spec.hidden;
    embd_ = RandomQuant(GgmlType::kQ8_0, H, spec.vocab, 1, rng_);
    onorm_ = RandomUniformF32(H, 0.5F, 1.5F, rng_);
    output_ = RandomQuant(GgmlType::kQ6_K, H, spec.vocab, 1, rng_);
    mats_.resize(spec.layers);
    weights_.layers.resize(spec.layers);
    for (std::uint32_t il = 0; il < spec.layers; ++il) {
      Layer& m = mats_[il];
      const std::size_t heads = c.n_head(il);
      const std::size_t kv = std::size_t{spec.kv_heads} * 128;
      m.an = RandomUniformF32(H, 0.5F, 1.5F, rng_);
      m.fn = RandomUniformF32(H, 0.5F, 1.5F, rng_);
      m.q = RandomQuant(GgmlType::kQ6_K, H, heads * 128, 1, rng_);
      m.k = RandomQuant(GgmlType::kQ6_K, H, kv, 1, rng_);
      m.v = RandomQuant(GgmlType::kQ6_K, H, kv, 1, rng_);
      m.o = RandomQuant(GgmlType::kQ6_K, heads * 128, H, 1, rng_);
      m.gate =
          RandomF32(H, heads, 1.5F / std::sqrt(static_cast<float>(H)), rng_);
      m.qn = RandomUniformF32(128, 0.5F, 1.5F, rng_);
      m.kn = RandomUniformF32(128, 0.5F, 1.5F, rng_);
      ml::LayerWeights& lw = weights_.layers[il];
      lw.full_attention = c.is_full_attention(il);
      lw.moe = c.is_moe(il);
      const std::string p = "blk." + std::to_string(il) + ".";
      lw.attn_norm = Ref(m.an, p + "attn_norm.weight");
      lw.ffn_norm = Ref(m.fn, p + "ffn_norm.weight");
      lw.attn_q = Ref(m.q, p + "attn_q.weight");
      lw.attn_k = Ref(m.k, p + "attn_k.weight");
      lw.attn_v = Ref(m.v, p + "attn_v.weight");
      lw.attn_output = Ref(m.o, p + "attn_output.weight");
      lw.attn_gate = Ref(m.gate, p + "attn_gate.weight");
      lw.attn_q_norm = Ref(m.qn, p + "attn_q_norm.weight");
      lw.attn_k_norm = Ref(m.kn, p + "attn_k_norm.weight");
      if (!lw.moe) {
        m.g = RandomQuant(GgmlType::kQ8_0, H, spec.dense_ff, 1, rng_);
        m.u = RandomQuant(GgmlType::kQ8_0, H, spec.dense_ff, 1, rng_);
        m.d = RandomQuant(GgmlType::kQ8_0, spec.dense_ff, H, 1, rng_);
        lw.ffn_gate = Ref(m.g, p + "ffn_gate.weight");
        lw.ffn_up = Ref(m.u, p + "ffn_up.weight");
        lw.ffn_down = Ref(m.d, p + "ffn_down.weight");
        continue;
      }
      const GgmlType type = spec.routed[(il - 1) % spec.routed.size()];
      const std::size_t E = spec.experts;
      m.router = RandomF32(H, E, 3.0F / std::sqrt(static_cast<float>(H)), rng_);
      m.bias = RandomF32(E, 1, 0.25F, rng_);
      m.g = RandomQuant(type, H, spec.expert_ff, E, rng_);
      m.u = RandomQuant(type, H, spec.expert_ff, E, rng_);
      m.d = RandomQuant(type, spec.expert_ff, H, E, rng_);
      m.sg = RandomQuant(GgmlType::kQ8_0, H, spec.shared_ff, 1, rng_);
      m.su = RandomQuant(GgmlType::kQ8_0, H, spec.shared_ff, 1, rng_);
      m.sd = RandomQuant(GgmlType::kQ8_0, spec.shared_ff, H, 1, rng_);
      lw.ffn_gate_inp = Ref(m.router, p + "ffn_gate_inp.weight");
      lw.ffn_exp_probs_b = Ref(m.bias, p + "exp_probs_b.bias");
      lw.ffn_gate_exps = Ref(m.g, p + "ffn_gate_exps.weight");
      lw.ffn_up_exps = Ref(m.u, p + "ffn_up_exps.weight");
      lw.ffn_down_exps = Ref(m.d, p + "ffn_down_exps.weight");
      lw.ffn_gate_shexp = Ref(m.sg, p + "ffn_gate_shexp.weight");
      lw.ffn_up_shexp = Ref(m.su, p + "ffn_up_shexp.weight");
      lw.ffn_down_shexp = Ref(m.sd, p + "ffn_down_shexp.weight");
    }
    weights_.token_embd = Ref(embd_, "token_embd.weight");
    weights_.output_norm = Ref(onorm_, "output_norm.weight");
    weights_.output = Ref(output_, "output.weight");
  }
  TinyModel(const TinyModel&) = delete;
  TinyModel& operator=(const TinyModel&) = delete;

  [[nodiscard]] const ml::ModelWeights& weights() const { return weights_; }
  [[nodiscard]] const ml::Config& config() const { return weights_.config; }
  [[nodiscard]] const ModelSpec& spec() const { return spec_; }

  /// Uploads the model; never null (the test fails instead).
  [[nodiscard]] std::unique_ptr<exec::DeviceModel> Upload() const {
    std::string error;
    auto model = exec::DeviceModel::Upload(weights_, nullptr, &error);
    Require(model != nullptr, "DeviceModel::Upload: " + error);
    return model;
  }

private:
  struct Layer {
    HostMat an, fn, q, k, v, o, gate, qn, kn, g, u, d, router, bias, sg, su, sd;
  };
  ml::TensorRef Ref(const HostMat& m, const std::string& name) {
    names_.push_back(name);
    ml::TensorRef t;
    t.data = m.bytes.data();
    t.type = m.type;
    t.cols = m.cols;
    t.rows = m.rows;
    t.experts = m.experts;
    t.name = names_.back();
    Require(t.SizeBytes() == m.bytes.size(), "fixture byte size mismatch");
    return t;
  }
  ModelSpec spec_;
  std::mt19937_64 rng_;
  ml::ModelWeights weights_;
  HostMat embd_, onorm_, output_;
  std::vector<Layer> mats_;
  std::deque<std::string> names_;
};

inline std::vector<std::int32_t> TokensFor(const TinyModel& m, std::size_t n,
                                           std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::vector<std::int32_t> t(n);
  for (auto& v : t) {
    v = static_cast<std::int32_t>(rng() % m.config().vocab_size);
  }
  return t;
}

/// Runs `tokens` through `ex` in chunks of `chunks` (summing to the token
/// count) and returns all [T][vocab] logits. Fails the test on any error.
inline std::vector<float> RunChunks(exec::Executor& ex,
                                    const std::vector<std::int32_t>& tokens,
                                    const std::vector<std::uint32_t>& chunks,
                                    std::size_t vocab) {
  std::vector<float> logits(tokens.size() * vocab);
  std::size_t pos = 0;
  for (const std::uint32_t c : chunks) {
    std::string error;
    const bool ok =
        ex.Forward(std::span<const std::int32_t>(tokens.data() + pos, c),
                   exec::LogitsRows::kAll, logits.data() + pos * vocab, &error);
    Require(ok, "Forward failed: " + error);
    pos += c;
  }
  Require(pos == tokens.size(), "RunChunks: chunks do not cover the tokens");
  return logits;
}

inline std::unique_ptr<exec::Executor> MakeExecutor(
    const exec::DeviceModel& model, std::uint32_t context,
    std::uint32_t max_chunk, gpu::Path path, bool warmup = true,
    exec::DeviceAllocator allocator = {}) {
  exec::ExecutorOptions options;
  options.context = context;
  options.max_chunk = max_chunk;
  options.path = path;
  options.warmup = warmup;
  options.allocator = allocator;
  std::string error;
  auto ex = exec::Executor::Create(model, options, &error);
  Require(ex != nullptr, "Executor::Create: " + error);
  return ex;
}

}  // namespace gufo::testing::laguna

#endif  // GUFO_TESTS_MODELS_LAGUNA_EXECUTOR_GPU_HPP_
