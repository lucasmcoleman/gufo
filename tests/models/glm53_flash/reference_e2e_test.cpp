// End-to-end CPU reference check for GLM-5.3-Flash (reference.hpp, "A5")
// against HF transformers 09281b8 Glm5NextTextModel on the tiny e2e fixture
// (fixtures/e2e_fixture.bin, made by fixtures/make_e2e_fixture.py from
// e2e_tiny.npz: hidden 32, 5 layers = KDA+dense x3, MLA/DSA+MoE, KDA+MoE,
// 14 tokens). Checks, all at rel-L2 <= 1e-5:
//   (a) every per-layer and sub-layer tap of a 14-token prefill, in forward
//       order, reporting the first failure;
//   (b) the final normed hidden state;
//   (c) prefill(10) + 4 single-token decodes vs HF's cached incremental run;
// plus (d) GgufWeightSource over hand-built TensorRefs (all F32, with the
// embedding Q8_0-encoded) reproducing the F32-source run exactly, and (e) the
// logits path on a synthetic output matrix (the fixture has no LM head).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/glm53_flash/config.hpp"
#include "src/models/glm53_flash/reference.hpp"
#include "src/models/glm53_flash/weights.hpp"

namespace glm = gufo::models::glm53_flash;
namespace ref = gufo::models::glm53_flash::reference;

namespace {

constexpr double kTol = 1e-5;

struct Array {
  std::vector<std::uint32_t> shape;
  std::vector<float> data;
};
using Fixture = std::map<std::string, Array>;

Fixture Load(const std::string& path) {
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

double RelL2(const float* got, const float* want, std::size_t n) {
  double num = 0.0;
  double den = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double d = static_cast<double>(got[i]) - want[i];
    num += d * d;
    den += static_cast<double>(want[i]) * want[i];
  }
  return den == 0.0 ? std::sqrt(num) : std::sqrt(num / den);
}

/// The fixture's global.config (manifest.json), as Config::FromGguf would
/// read it from a glm5-next GGUF of that model.
glm::Config TinyConfig() {
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
  c.kda_head_dim = 8;
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

bool IsWeight(const std::string& name) {
  return name.rfind("tap.", 0) != 0 && name != "input_ids" &&
         name.rfind("final_hidden", 0) != 0;
}

/// Shape [.., rows, cols] -> (cols, rows, experts = product of the rest).
void Dims(const Array& a, std::uint64_t& cols, std::uint64_t& rows,
          std::uint64_t& experts) {
  const auto& s = a.shape;
  cols = s.empty() ? 1 : s.back();
  rows = s.size() >= 2 ? s[s.size() - 2] : 1;
  experts = 1;
  for (std::size_t i = 0; i + 2 < s.size(); ++i)
    experts *= s[i];
}

ref::F32WeightSource MakeSource(const Fixture& fx) {
  ref::F32WeightSource src;
  for (const auto& [name, a] : fx) {
    if (!IsWeight(name))
      continue;
    std::uint64_t cols = 0, rows = 0, experts = 0;
    Dims(a, cols, rows, experts);
    src.Add(name, a.data, cols, rows, experts);
  }
  return src;
}

std::vector<std::int32_t> Tokens(const Fixture& fx) {
  std::vector<std::int32_t> ids;
  for (float v : fx.at("input_ids").data)
    ids.push_back(static_cast<std::int32_t>(v));
  return ids;
}

// (a) + (b): prefill taps and the final hidden state.
int CheckPrefill(const glm::Config& config, const Fixture& fx,
                 const ref::WeightSource& src, std::vector<float>& hidden) {
  std::map<std::string, std::vector<float>> got;
  std::vector<std::string> order;
  const ref::TapFn tap = [&](std::string_view name, int layer,
                             const float* data, std::size_t width,
                             std::uint32_t n) {
    const std::string key = layer < 0 ? "tap." + std::string(name)
                                      : "tap.layer" + std::to_string(layer) +
                                            "." + std::string(name);
    got[key].assign(data, data + (width * n));
    order.push_back(key);
  };
  ref::ReferenceModel model(config, src);
  const auto ids = Tokens(fx);
  hidden = model.Forward(ids, false, tap).hidden;

  // Fixture taps that duplicate a computed one (the input of a module is the
  // output of the previous one).
  const std::string last = "tap.layer" + std::to_string(config.num_layers - 1);
  const auto computed_name = [&](const std::string& key) -> std::string {
    const auto dot = key.find('.', 4);
    const std::string prefix = key.substr(0, dot + 1);
    const std::string suffix = key.substr(dot + 1);
    if (key == "tap.hc_head.in")
      return last + ".out0";
    if (key == "tap.final_norm.in")
      return "tap.hc_head.out";
    static const std::map<std::string, std::string> alias = {
        {"attn_hc.in", "in"},
        {"input_layernorm.in", "attn_hc.out2"},
        {"self_attn.in", "input_layernorm.out"},
        {"self_attn.out0", "self_attn.out"},
        {"post_attention_layernorm.in", "ffn_hc.out2"},
        {"mlp.in", "post_attention_layernorm.out"}};
    const auto it = alias.find(suffix);
    return it == alias.end() ? key : prefix + it->second;
  };

  // Group fixture taps by the computed tap they check, in forward order.
  std::map<std::string, std::vector<std::string>> by_computed;
  std::size_t fixture_taps = 0;
  for (const auto& [key, a] : fx) {
    if (key.rfind("tap.", 0) != 0)
      continue;
    ++fixture_taps;
    by_computed[computed_name(key)].push_back(key);
  }
  int failures = 0;
  std::string first_fail;
  std::size_t checked = 0;
  std::cout << std::scientific << std::setprecision(2);
  std::map<std::string, double> worst_by_kind;
  for (const std::string& key : order) {
    const auto it = by_computed.find(key);
    if (it == by_computed.end()) {
      std::cout << "  (no fixture tap for " << key << ")\n";
      continue;
    }
    const auto& mine = got.at(key);
    for (const std::string& fkey : it->second) {
      const Array& want = fx.at(fkey);
      if (want.data.size() != mine.size()) {
        std::cout << "FAIL " << fkey << ": size " << mine.size() << " vs "
                  << want.data.size() << "\n";
        ++failures;
        if (first_fail.empty())
          first_fail = fkey;
        continue;
      }
      const double r = RelL2(mine.data(), want.data.data(), mine.size());
      ++checked;
      const std::string kind = fkey.substr(fkey.find('.', 4) + 1);
      double& w =
          worst_by_kind[fkey.rfind("tap.layer", 0) == 0
                            ? fkey.substr(4, fkey.find('.', 4) - 4) + ":" + kind
                            : fkey.substr(4)];
      w = std::max(w, r);
      if (!(r <= kTol)) {
        std::cout << "FAIL " << fkey << " rel_l2 " << r << "\n";
        ++failures;
        if (first_fail.empty())
          first_fail = fkey;
      }
    }
  }
  // Worst rel-L2 per (layer, tap).
  for (std::uint32_t il = 0; il < config.num_layers; ++il) {
    const std::string l = "layer" + std::to_string(il) + ":";
    double worst = 0.0;
    std::string worst_name;
    for (const auto& [k, v] : worst_by_kind) {
      if (k.rfind(l, 0) == 0 && v >= worst) {
        worst = v;
        worst_name = k.substr(l.size());
      }
    }
    std::cout << "layer" << il << ": worst " << worst << " (" << worst_name
              << "), out0 " << worst_by_kind[l + "out0"] << ", self_attn "
              << std::max(worst_by_kind[l + "self_attn.out"],
                          worst_by_kind[l + "self_attn.out0"])
              << ", mlp " << worst_by_kind[l + "mlp.out"] << "\n";
  }
  std::cout << "hc_head.out " << worst_by_kind["hc_head.out"]
            << ", final_norm.out " << worst_by_kind["final_norm.out"] << "\n";
  if (checked != fixture_taps) {
    std::cout << "FAIL: checked " << checked << " of " << fixture_taps
              << " fixture taps\n";
    ++failures;
  }
  std::cout << "taps checked: " << checked << "\n";

  const Array& fh = fx.at("final_hidden");
  const double r = RelL2(hidden.data(), fh.data.data(), fh.data.size());
  std::cout << "final_hidden rel_l2 " << r << "\n";
  if (!(r <= kTol) || hidden.size() != fh.data.size()) {
    ++failures;
    if (first_fail.empty())
      first_fail = "final_hidden";
  }
  if (!first_fail.empty())
    std::cout << "first failing tap: " << first_fail << "\n";
  return failures;
}

// (c): prefill(10) then four single-token decodes.
int CheckIncremental(const glm::Config& config, const Fixture& fx,
                     const ref::WeightSource& src,
                     const std::vector<float>& full) {
  const auto ids = Tokens(fx);
  ref::ReferenceModel model(config, src);
  std::vector<float> hidden =
      model.Forward(std::span(ids).first(10), false).hidden;
  for (std::size_t t = 10; t < ids.size(); ++t) {
    const auto step = model.Forward(std::span(ids).subspan(t, 1), false);
    hidden.insert(hidden.end(), step.hidden.begin(), step.hidden.end());
  }
  const Array& want = fx.at("final_hidden_incremental");
  const std::size_t h = config.hidden_size;
  const double r = RelL2(hidden.data(), want.data.data(), want.data.size());
  const double r_dec =
      RelL2(hidden.data() + (10 * h), want.data.data() + (10 * h), 4 * h);
  const double r_self = RelL2(hidden.data(), full.data(), full.size());
  std::cout << "incremental vs HF incremental rel_l2 " << r << " (decoded rows "
            << r_dec << "), vs own prefill " << r_self << ", position "
            << model.Position() << "\n";
  int failures = 0;
  if (!(r <= kTol) || !(r_dec <= kTol) || model.Position() != ids.size()) {
    std::cout << "FAIL incremental\n";
    ++failures;
  }
  return failures;
}

// (d): GgufWeightSource over hand-built TensorRefs.
std::uint16_t PowerOfTwoHalf(int exponent) {
  if (exponent < -14 || exponent > 15)
    throw std::runtime_error("Q8_0 scale outside fp16 normal range");
  return static_cast<std::uint16_t>((exponent + 15) << 10);
}

int CheckGgufSource(const glm::Config& config, const Fixture& fx,
                    const std::vector<float>& f32_hidden) {
  // The embedding as Q8_0 with power-of-two scales, and its exact
  // dequantization for the F32 comparison run.
  const Array& embd = fx.at("token_embd.weight");
  const std::size_t cols = embd.shape[1];
  const std::size_t rows = embd.shape[0];
  const std::size_t row_bytes =
      gufo::quant::QuantizedRowBytes(gufo::core::GgmlType::kQ8_0, cols);
  std::vector<std::uint8_t> q8(rows * row_bytes);
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t b = 0; b < cols / 32; ++b) {
      const float* x = embd.data.data() + (r * cols) + (b * 32);
      float amax = 0.0F;
      for (int i = 0; i < 32; ++i)
        amax = std::max(amax, std::fabs(x[i]));
      const int e = static_cast<int>(std::ceil(std::log2(amax / 127.0F)));
      const float d = std::ldexp(1.0F, e);
      std::uint8_t* blk = q8.data() + (r * row_bytes) + (b * 34);
      const std::uint16_t dh = PowerOfTwoHalf(e);
      std::memcpy(blk, &dh, 2);
      for (int i = 0; i < 32; ++i) {
        const auto q = static_cast<std::int8_t>(std::lround(x[i] / d));
        std::memcpy(blk + 2 + i, &q, 1);
      }
    }
  }
  std::vector<float> deq(rows * cols);
  for (std::size_t r = 0; r < rows; ++r)
    gufo::quant::DequantizeQ8_0(q8.data() + (r * row_bytes),
                                deq.data() + (r * cols), cols);

  // TensorRefs over the fixture arrays (names must outlive the source).
  std::vector<std::string> names;
  names.reserve(fx.size());
  const auto ref_of = [&](const std::string& name) {
    const Array& a = fx.at(name);
    glm::TensorRef t;
    t.data = a.data.data();
    t.type = gufo::core::GgmlType::kF32;
    Dims(a, t.cols, t.rows, t.experts);
    names.push_back(name);
    t.name = names.back();
    return t;
  };
  glm::ModelWeights w;
  w.config = config;
  w.token_embd = ref_of("token_embd.weight");
  w.token_embd.data = q8.data();
  w.token_embd.type = gufo::core::GgmlType::kQ8_0;
  w.output_norm = ref_of("output_norm.weight");
  for (std::uint32_t il = 0; il < config.num_layers; ++il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    const auto r = [&](const char* n) { return ref_of(p + n); };
    glm::LayerWeights l;
    l.mixer = config.mixers[il];
    l.dense_ffn = config.IsDenseFfnLayer(il);
    l.hc_attn = {r("hc_attn_fn.weight"), r("hc_attn_base.weight"),
                 r("hc_attn_scale.weight")};
    l.hc_ffn = {r("hc_ffn_fn.weight"), r("hc_ffn_base.weight"),
                r("hc_ffn_scale.weight")};
    l.attn_norm = r("attn_norm.weight");
    l.ffn_norm = r("ffn_norm.weight");
    l.attn_out = r("attn_output.weight");
    if (config.IsKdaLayer(il)) {
      l.kda_q = r("attn_q.weight");
      l.kda_k = r("attn_k.weight");
      l.kda_v = r("attn_v.weight");
      l.kda_conv_q = r("ssm_conv1d_q.weight");
      l.kda_conv_k = r("ssm_conv1d_k.weight");
      l.kda_conv_v = r("ssm_conv1d_v.weight");
      l.kda_f_a = r("ssm_f_a.weight");
      l.kda_f_b = r("ssm_f_b.weight");
      l.kda_dt_bias = r("ssm_dt.bias");
      l.kda_a = r("ssm_a");
      l.kda_beta = r("ssm_beta.weight");
      l.kda_g_a = r("ssm_g_a.weight");
      l.kda_g_b = r("ssm_g_b.weight");
      l.kda_norm = r("ssm_norm.weight");
    } else {
      l.mla_q_a = r("attn_q_a.weight");
      l.mla_q_a_norm = r("attn_q_a_norm.weight");
      l.mla_q_b = r("attn_q_b.weight");
      l.mla_kv_a = r("attn_kv_a_mqa.weight");
      l.mla_kv_a_norm = r("attn_kv_a_norm.weight");
      l.mla_k_b = r("attn_k_b.weight");
      l.mla_v_b = r("attn_v_b.weight");
      l.idx_q_b = r("indexer.attn_q_b.weight");
      l.idx_k = r("indexer.attn_k.weight");
      l.idx_k_norm = r("indexer.k_norm.weight");
      l.idx_k_norm_bias = r("indexer.k_norm.bias");
      l.idx_proj = r("indexer.proj.weight");
      l.idx_gate = r("indexer_compressor_gate.weight");
      l.idx_ape = r("indexer_compressor_ape.weight");
    }
    if (l.dense_ffn) {
      l.ffn_gate = r("ffn_gate.weight");
      l.ffn_up = r("ffn_up.weight");
      l.ffn_down = r("ffn_down.weight");
    } else {
      l.router = r("ffn_gate_inp.weight");
      l.router_bias = r("exp_probs_b.bias");
      l.ffn_gate_exps = r("ffn_gate_exps.weight");
      l.ffn_up_exps = r("ffn_up_exps.weight");
      l.ffn_down_exps = r("ffn_down_exps.weight");
      l.shexp_gate = r("ffn_gate_shexp.weight");
      l.shexp_up = r("ffn_up_shexp.weight");
      l.shexp_down = r("ffn_down_shexp.weight");
    }
    w.layers.push_back(l);
  }
  const ref::GgufWeightSource gguf(w);

  ref::F32WeightSource f32 = MakeSource(fx);
  f32.Add("token_embd.weight", deq, cols, rows);
  const auto ids = Tokens(fx);
  ref::ReferenceModel a(config, gguf);
  ref::ReferenceModel b(config, f32);
  const auto ha = a.Forward(ids).hidden;
  const auto hb = b.Forward(ids).hidden;
  const bool exact = ha == hb;
  const double r_hf =
      RelL2(ha.data(), fx.at("final_hidden").data.data(), ha.size());
  const double r_f32 = RelL2(ha.data(), f32_hidden.data(), ha.size());
  std::cout << "gguf source (Q8_0 embedding): "
            << (exact ? "bit-exact" : "MISMATCH")
            << " vs F32 source on the dequantized embedding; vs HF " << r_hf
            << ", vs F32 embedding run " << r_f32 << "\n";
  return exact ? 0 : 1;
}

// (e): logits = output . final_hidden.
int CheckLogits(const glm::Config& config, const Fixture& fx) {
  ref::F32WeightSource src = MakeSource(fx);
  const std::size_t h = config.hidden_size;
  const std::size_t v = config.vocab_size;
  std::mt19937 rng(77);
  std::normal_distribution<float> dist(0.0F, 0.2F);
  std::vector<float> out_w(v * h);
  for (auto& x : out_w)
    x = dist(rng);
  src.Add("output.weight", out_w, h, v);
  ref::ReferenceModel model(config, src);
  const auto ids = Tokens(fx);
  const auto out = model.Forward(ids, true);
  if (out.logits.size() != ids.size() * v) {
    std::cout << "FAIL logits size\n";
    return 1;
  }
  std::vector<float> want(out.logits.size());
  for (std::size_t t = 0; t < ids.size(); ++t)
    for (std::size_t j = 0; j < v; ++j) {
      double acc = 0.0;
      for (std::size_t d = 0; d < h; ++d)
        acc +=
            static_cast<double>(out_w[(j * h) + d]) * out.hidden[(t * h) + d];
      want[(t * v) + j] = static_cast<float>(acc);
    }
  const double r = RelL2(out.logits.data(), want.data(), want.size());
  std::cout << "logits rel_l2 vs direct product " << r << "\n";
  return r <= 1e-7 ? 0 : 1;
}

}  // namespace

int main() {
  try {
    const glm::Config config = TinyConfig();
    const Fixture fx = Load(GLM53_E2E_FIXTURE);
    const ref::F32WeightSource src = MakeSource(fx);
    std::vector<float> hidden;
    int failures = CheckPrefill(config, fx, src, hidden);
    failures += CheckIncremental(config, fx, src, hidden);
    failures += CheckGgufSource(config, fx, hidden);
    failures += CheckLogits(config, fx);
    if (failures != 0) {
      std::cout << failures << " check(s) failed\n";
      return 1;
    }
    std::cout << "PASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
