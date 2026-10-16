#ifndef GUFO_TESTS_MODELS_GLM53_FLASH_LOOP_MODEL_HPP_
#define GUFO_TESTS_MODELS_GLM53_FLASH_LOOP_MODEL_HPP_

// The e2e fixture model (fixtures/e2e_fixture.bin) in the shape the GPU
// executor runs: its four KDA layers redrawn (seeded) at head dim 128, every
// other tensor the fixture's, plus a synthetic output.weight; some tensors
// Q8_0 with exact power-of-two scales. Shared by executor_loop_test.cpp and
// generator_gpu_test.cpp; see the former for the details.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fstream>
#include <iterator>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/glm53_flash/config.hpp"
#include "src/models/glm53_flash/weights.hpp"

namespace gufo::models::glm53_flash::loop_model {

namespace glm = ::gufo::models::glm53_flash;
using ::gufo::core::GgmlType;

/// KDA head dim of the redrawn layers (the GPU kernels need 128).
inline constexpr std::uint32_t kD = 128;

struct Array {
  std::vector<std::uint32_t> shape;
  std::vector<float> data;
};
using Fixture = std::map<std::string, Array>;

/// fixtures/e2e_fixture.bin (format of reference_e2e_test.cpp).
inline Fixture Load(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw std::runtime_error("cannot open fixture " + path);
  const std::vector<char> bytes((std::istreambuf_iterator<char>(in)),
                                std::istreambuf_iterator<char>());
  std::size_t pos = 0;
  auto take = [&](void* dst, std::size_t n) {
    if (pos + n > bytes.size())
      throw std::runtime_error("truncated fixture");
    std::memcpy(dst, bytes.data() + pos, n);
    pos += n;
  };
  auto u32 = [&] {
    std::uint32_t v = 0;
    take(&v, sizeof(v));
    return v;
  };
  char magic[8];
  take(magic, sizeof(magic));
  if (std::memcmp(magic, "GLME2EF1", 8) != 0)
    throw std::runtime_error("bad fixture magic in " + path);
  Fixture out;
  const std::uint32_t count = u32();
  for (std::uint32_t i = 0; i < count; ++i) {
    std::string name(u32(), '\0');
    take(name.data(), name.size());
    Array a;
    a.shape.resize(u32());
    std::size_t n = 1;
    for (auto& d : a.shape) {
      d = u32();
      n *= d;
    }
    a.data.resize(n);
    take(a.data.data(), n * sizeof(float));
    out.emplace(std::move(name), std::move(a));
  }
  return out;
}

/// The fixture's config (reference_e2e_test.cpp TinyConfig) with the KDA
/// head dim the GPU kernels need.
inline glm::Config LoopConfig() {
  glm::Config c;
  c.num_layers = 5;
  c.num_layers_all = 5;
  c.nextn_layers = 0;
  c.hidden_size = 32;
  c.vocab_size = 64;
  c.context_length = 1024;
  c.rms_eps = 1e-5F;
  c.norm_eps = 1e-6F;
  using M = glm::Config::Mixer;
  c.mixers = {M::kKda, M::kKda, M::kKda, M::kMla, M::kKda};
  c.hc_count = 4;
  c.hc_sinkhorn_iters = 20;
  c.hc_eps = 1e-6F;
  c.num_heads = 4;
  c.kda_head_dim = kD;
  c.ssm_conv_kernel = 4;
  c.kda_gate_lower_bound = -5.0F;
  c.q_lora_rank = 24;
  c.kv_lora_rank = 16;
  c.mla_head_dim = 8;
  c.mla_value_dim = 8;
  c.indexer_heads = 4;
  c.indexer_head_dim = 8;
  c.indexer_top_k = 8;
  c.indexer_kpool = 4;
  c.leading_dense_layers = 3;
  c.dense_ff = 48;
  c.num_experts = 8;
  c.num_experts_used = 2;
  c.expert_ff = 16;
  c.shared_expert_ff = 16;
  c.num_shared_experts = 1;
  c.expert_weights_scale = 2.5F;
  c.expert_weights_norm = true;
  c.swiglu_clamp_exp.assign(5, 10.0F);
  c.swiglu_clamp_shexp.assign(5, 10.0F);
  return c;
}

/// Q8_0 with power-of-two scales (exact fp16), as reference_e2e_test does.
inline std::vector<std::uint8_t> EncodeQ8_0(const std::vector<float>& m,
                                            std::size_t rows,
                                            std::size_t cols) {
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

/// Host storage for the model; TensorRefs point into it.
struct Model {
  glm::Config config = LoopConfig();
  std::deque<std::string> names;
  std::deque<std::vector<float>> f32;
  std::deque<std::vector<std::uint8_t>> q8;
  glm::ModelWeights weights;
  std::vector<std::int32_t> ids;
};

inline void BuildModel(Model& m, const Fixture& fx) {
  std::mt19937 rng(53);
  std::normal_distribution<float> n01(0.0F, 1.0F);
  const glm::Config& c = m.config;
  const auto add = [&](const std::string& name, std::vector<float> data,
                       std::uint64_t cols, std::uint64_t rows,
                       std::uint64_t experts, bool q8) {
    m.names.push_back(name);
    glm::TensorRef t;
    t.cols = cols;
    t.rows = rows;
    t.experts = experts;
    t.name = m.names.back();
    if (q8) {
      m.q8.push_back(EncodeQ8_0(data, rows * experts, cols));
      t.data = m.q8.back().data();
      t.type = GgmlType::kQ8_0;
    } else {
      m.f32.push_back(std::move(data));
      t.data = m.f32.back().data();
      t.type = GgmlType::kF32;
    }
    return t;
  };
  // Shape [.., rows, cols] -> (cols, rows, experts = product of the rest).
  const auto fixture = [&](const std::string& name, bool q8 = false) {
    const Array& a = fx.at(name);
    const auto& s = a.shape;
    const std::uint64_t cols = s.empty() ? 1 : s.back();
    const std::uint64_t rows = s.size() >= 2 ? s[s.size() - 2] : 1;
    std::uint64_t experts = 1;
    for (std::size_t i = 0; i + 2 < s.size(); ++i)
      experts *= s[i];
    return add(name, a.data, cols, rows, experts, q8);
  };
  const auto random = [&](const std::string& name, std::uint64_t cols,
                          std::uint64_t rows, float mean, float sd,
                          bool q8 = false) {
    std::vector<float> data(cols * rows);
    for (auto& x : data)
      x = mean + (sd * n01(rng));
    return add(name, std::move(data), cols, rows, 1, q8);
  };

  const std::uint64_t hidden = c.hidden_size;
  const std::uint64_t dim = c.KdaDim();
  const float proj_sd = 1.0F / std::sqrt(static_cast<float>(hidden));
  m.weights.config = c;
  m.weights.token_embd = fixture("token_embd.weight", /*q8=*/true);
  m.weights.output_norm = fixture("output_norm.weight");
  m.weights.output =
      random("output.weight", hidden, c.vocab_size, 0, 0.2F, /*q8=*/true);
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    const auto fx_l = [&](const char* n, bool q8 = false) {
      return fixture(p + n, q8);
    };
    glm::LayerWeights l;
    l.mixer = c.mixers[il];
    l.dense_ffn = c.IsDenseFfnLayer(il);
    l.hc_attn = {fx_l("hc_attn_fn.weight"), fx_l("hc_attn_base.weight"),
                 fx_l("hc_attn_scale.weight")};
    l.hc_ffn = {fx_l("hc_ffn_fn.weight"), fx_l("hc_ffn_base.weight"),
                fx_l("hc_ffn_scale.weight")};
    l.attn_norm = fx_l("attn_norm.weight");
    l.ffn_norm = fx_l("ffn_norm.weight");
    if (c.IsKdaLayer(il)) {
      // Redrawn at head dim 128 (kda_executor_test.cpp scales).
      const bool q = il == 0;
      l.kda_q = random(p + "attn_q.weight", hidden, dim, 0, proj_sd, q);
      l.kda_k = random(p + "attn_k.weight", hidden, dim, 0, proj_sd, q);
      l.kda_v = random(p + "attn_v.weight", hidden, dim, 0, proj_sd, q);
      l.kda_conv_q = random(p + "ssm_conv1d_q.weight", 4, dim, 0, 0.5F);
      l.kda_conv_k = random(p + "ssm_conv1d_k.weight", 4, dim, 0, 0.5F);
      l.kda_conv_v = random(p + "ssm_conv1d_v.weight", 4, dim, 0, 0.5F);
      l.kda_f_a = random(p + "ssm_f_a.weight", hidden, kD, 0, proj_sd);
      l.kda_f_b = random(p + "ssm_f_b.weight", kD, dim, 0, 0.15F);
      l.kda_dt_bias = random(p + "ssm_dt.bias", dim, 1, 0, 0.5F);
      l.kda_a = random(p + "ssm_a", c.num_heads, 1, 0, 0.5F);
      for (float& a : m.f32.back())
        a = -std::exp(a);  // ssm_a = -exp(A_log)
      l.kda_beta =
          random(p + "ssm_beta.weight", hidden, c.num_heads, 0, proj_sd);
      l.kda_g_a = random(p + "ssm_g_a.weight", hidden, kD, 0, proj_sd);
      l.kda_g_b = random(p + "ssm_g_b.weight", kD, dim, 0, 0.15F);
      l.kda_norm = random(p + "ssm_norm.weight", kD, 1, 1, 0.2F);
      l.attn_out = random(p + "attn_output.weight", dim, hidden, 0,
                          1.0F / std::sqrt(static_cast<float>(dim)), q);
    } else {
      l.attn_out = fx_l("attn_output.weight");
      l.mla_q_a = fx_l("attn_q_a.weight");
      l.mla_q_a_norm = fx_l("attn_q_a_norm.weight");
      l.mla_q_b = fx_l("attn_q_b.weight");
      l.mla_kv_a = fx_l("attn_kv_a_mqa.weight");
      l.mla_kv_a_norm = fx_l("attn_kv_a_norm.weight");
      l.mla_k_b = fx_l("attn_k_b.weight");
      l.mla_v_b = fx_l("attn_v_b.weight");
      l.idx_q_b = fx_l("indexer.attn_q_b.weight");
      l.idx_k = fx_l("indexer.attn_k.weight");
      l.idx_k_norm = fx_l("indexer.k_norm.weight");
      l.idx_k_norm_bias = fx_l("indexer.k_norm.bias");
      l.idx_proj = fx_l("indexer.proj.weight");
      l.idx_gate = fx_l("indexer_compressor_gate.weight");
      l.idx_ape = fx_l("indexer_compressor_ape.weight");
    }
    if (l.dense_ffn) {
      const bool q = il == 2;
      l.ffn_gate = fx_l("ffn_gate.weight", q);
      l.ffn_up = fx_l("ffn_up.weight", q);
      l.ffn_down = fx_l("ffn_down.weight");
    } else {
      l.router = fx_l("ffn_gate_inp.weight");
      l.router_bias = fx_l("exp_probs_b.bias");
      l.ffn_gate_exps = fx_l("ffn_gate_exps.weight", il == 4);
      l.ffn_up_exps = fx_l("ffn_up_exps.weight");
      l.ffn_down_exps = fx_l("ffn_down_exps.weight");
      l.shexp_gate = fx_l("ffn_gate_shexp.weight");
      l.shexp_up = fx_l("ffn_up_shexp.weight");
      l.shexp_down = fx_l("ffn_down_shexp.weight");
    }
    m.weights.layers.push_back(l);
  }
  for (float v : fx.at("input_ids").data)
    m.ids.push_back(static_cast<std::int32_t>(v));
}

}  // namespace gufo::models::glm53_flash::loop_model

#endif  // GUFO_TESTS_MODELS_GLM53_FLASH_LOOP_MODEL_HPP_
