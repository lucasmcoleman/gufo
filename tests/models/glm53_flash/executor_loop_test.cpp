// GPU layer loop (executor.hpp Executor) against the CPU ReferenceModel on
// the e2e fixture model (fixtures/e2e_fixture.bin: hidden 32, 5 layers =
// KDA+dense x3, MLA/DSA+MoE, KDA+MoE). The GPU KDA kernels need head dim 128
// and the fixture's is 8, so its four KDA layers are redrawn (seeded) at
// 4 heads x 128; every other tensor (embedding, hyper-connections, norms,
// MLA + indexer, dense FFN, MoE) is the fixture's, plus a synthetic
// output.weight for the logits path. Q8_0 (power-of-two scales, exact):
// token_embd, output, layer 0 q|k|v and attn_output, layer 2 ffn_gate /
// ffn_up, layer 4 ffn_gate_exps. Both sides read the same TensorRefs.
// MLA and FFN run through the host stand-ins of host_ops.hpp, which read
// their weights back from the uploaded MlaLayerDevice / FfnLayerDevice.
// Checks, at rel-L2 <= kTol:
//   (a) prefill(12) + 2 single-token decodes: every ReferenceModel tap per
//       layer, the final hidden state and the logits of each call;
//   (b) the 14 tokens in one call with max_tokens 5 (passes 5+5+4);
//   (c) Reset() then the 12-token prefill again reproduces (a);
//   (d) a [0,3) range with embedding and no head, then [3,5) with the head
//       from its streams (ForwardStreams), vs the full 14-token reference;
//   (e) refusals: an MLA range without an MLA operator, the embedding with a
//       range past layer 0, and a failing operator leaving the executor
//       Failed() until Reset().

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/glm53_flash/config.hpp"
#include "src/models/glm53_flash/executor.hpp"
#include "src/models/glm53_flash/reference.hpp"
#include "src/models/glm53_flash/weights.hpp"
#include "tests/models/glm53_flash/host_ops.hpp"

namespace glm = gufo::models::glm53_flash;
namespace ref = gufo::models::glm53_flash::reference;
using gufo::core::GgmlType;

namespace {

constexpr double kTol = 1e-5;
constexpr std::uint32_t kPrefill = 12;
constexpr std::uint32_t kD = 128;
int failures = 0;

struct Array {
  std::vector<std::uint32_t> shape;
  std::vector<float> data;
};
using Fixture = std::map<std::string, Array>;

/// fixtures/e2e_fixture.bin (format of reference_e2e_test.cpp).
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

void Check(const std::string& what, bool ok) {
  std::cout << (ok ? "ok   " : "FAIL ") << what << "\n";
  if (!ok)
    ++failures;
}

void CheckRel(const std::string& what, const std::vector<float>& got,
              const std::vector<float>& want, double tol = kTol) {
  const double rel = got.size() == want.size()
                         ? RelL2(got.data(), want.data(), want.size())
                         : INFINITY;
  const bool ok = !want.empty() && std::isfinite(rel) && rel <= tol;
  std::cout << (ok ? "ok   " : "FAIL ") << what << " rel_l2 " << rel << "\n";
  if (!ok)
    ++failures;
}

/// The fixture's config (reference_e2e_test.cpp TinyConfig) with the KDA
/// head dim the GPU kernels need.
glm::Config LoopConfig() {
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

/// Host storage for the model; TensorRefs point into it.
struct Model {
  glm::Config config = LoopConfig();
  std::deque<std::string> names;
  std::deque<std::vector<float>> f32;
  std::deque<std::vector<std::uint8_t>> q8;
  glm::ModelWeights weights;
  std::vector<std::int32_t> ids;
};

void BuildModel(Model& m, const Fixture& fx) {
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

/// Taps keyed "L<layer>.<name>" (or the bare name outside a layer), with
/// the order they arrived in.
struct Taps {
  std::map<std::string, std::vector<float>> data;
  std::vector<std::string> order;
  void Put(const std::string& key, const float* p, std::size_t n) {
    if (!data.contains(key))
      order.push_back(key);
    data[key].assign(p, p + n);
  }
};

std::string Key(std::string_view name, long long layer) {
  return layer < 0 ? std::string(name)
                   : "L" + std::to_string(layer) + "." + std::string(name);
}

ref::TapFn RefTap(Taps& taps) {
  return [&taps](std::string_view name, int layer, const float* data,
                 std::size_t width, std::uint32_t n) {
    taps.Put(Key(name, layer), data, width * n);
  };
}

glm::TapFn GpuTap(Taps& taps) {
  return [&taps](std::string_view name, std::uint32_t layer,
                 const glm::TapView& v) {
    if (name == "kda_out")
      return;  // operator tap, no ReferenceModel counterpart
    std::vector<float> host(v.count());
    if (hipDeviceSynchronize() != hipSuccess ||
        hipMemcpy(host.data(), v.device, host.size() * sizeof(float),
                  hipMemcpyDeviceToHost) != hipSuccess) {
      throw std::runtime_error("tap download failed");
    }
    taps.Put(Key(name, layer == glm::kNoLayer ? -1LL : layer), host.data(),
             host.size());
  };
}

/// Every reference tap against the GPU's, with the worst rel-L2 per layer.
void CompareTaps(const std::string& label, const Taps& gpu, const Taps& cpu,
                 std::uint32_t layers) {
  std::map<std::string, std::pair<double, std::string>> worst;
  std::size_t checked = 0;
  for (const std::string& key : cpu.order) {
    const auto it = gpu.data.find(key);
    const std::vector<float>& want = cpu.data.at(key);
    const std::string group =
        key[0] == 'L' ? key.substr(0, key.find('.')) : "head";
    if (it == gpu.data.end() || it->second.size() != want.size()) {
      std::cout << "FAIL " << label << " tap " << key
                << (it == gpu.data.end() ? " missing" : " size mismatch")
                << "\n";
      ++failures;
      continue;
    }
    const double r = RelL2(it->second.data(), want.data(), want.size());
    ++checked;
    auto& w = worst[group];
    if (!(r <= w.first)) {
      w = {r, group == "head" ? key : key.substr(key.find('.') + 1)};
    }
    if (!(r <= kTol)) {
      std::cout << "FAIL " << label << " tap " << key << " rel_l2 " << r
                << "\n";
      ++failures;
    }
  }
  for (std::uint32_t il = 0; il < layers; ++il) {
    const std::string l = "L" + std::to_string(il);
    const auto out0 = [&](const char* name) {
      const std::string key = l + "." + name;
      return gpu.data.contains(key) && cpu.data.contains(key)
                 ? RelL2(gpu.data.at(key).data(), cpu.data.at(key).data(),
                         cpu.data.at(key).size())
                 : INFINITY;
    };
    std::cout << "  " << label << " layer " << il << ": worst "
              << worst[l].first << " (" << worst[l].second << "), out0 "
              << out0("out0") << ", self_attn " << out0("self_attn.out")
              << ", mlp " << out0("mlp.out") << "\n";
  }
  if (worst.contains("head")) {
    std::cout << "  " << label << " head: worst " << worst["head"].first << " ("
              << worst["head"].second << ")\n";
  }
  Check(label + ": " + std::to_string(checked) + " of " +
            std::to_string(cpu.order.size()) + " taps matched",
        checked == cpu.order.size() && checked > 0);
}

glm::ExecutorOptions Options(std::uint32_t max_tokens) {
  glm::ExecutorOptions o;
  o.max_tokens = max_tokens;
  o.capacity = 64;
  return o;
}

// (a) + (c).
void CheckIncremental(const Model& m) {
  const glm::Config& c = m.config;
  const ref::GgufWeightSource source(m.weights);
  ref::ReferenceModel model(c, source);
  glm::stand_in::HostOps ops(c);
  std::string error;
  auto exec = glm::Executor::Create(m.weights, Options(16), ops.Bind(), &error);
  Check("create full executor " + error, exec != nullptr);
  if (!exec)
    return;
  std::cout << "device bytes " << exec->DeviceBytes() << "\n";
  const std::span<const std::int32_t> ids(m.ids);
  std::vector<float> first_hidden;
  std::vector<float> first_logits;
  for (std::uint32_t step = 0; step < 3; ++step) {
    const auto span =
        step == 0 ? ids.first(kPrefill) : ids.subspan(kPrefill + step - 1, 1);
    const std::string label =
        step == 0 ? "prefill(12)"
                  : "decode token " + std::to_string(kPrefill + step - 1);
    Taps cpu_taps;
    Taps gpu_taps;
    const auto want = model.Forward(span, true, RefTap(cpu_taps));
    glm::ForwardResult got;
    const glm::TapFn tap = GpuTap(gpu_taps);
    const bool ok = exec->Forward(span, true, &got, &tap, &error);
    Check(label + " forward " + error, ok);
    if (!ok)
      return;
    CompareTaps(label, gpu_taps, cpu_taps, c.num_layers);
    CheckRel(label + " final hidden", got.hidden, want.hidden);
    CheckRel(label + " logits", got.logits, want.logits);
    if (step == 0) {
      first_hidden = got.hidden;
      first_logits = got.logits;
    }
  }
  Check("position after prefill + 2 decodes", exec->Position() == 14);
  Check("stand-ins ran (mla " + std::to_string(ops.mla_calls()) + ", ffn " +
            std::to_string(ops.ffn_calls()) + ")",
        ops.mla_calls() == 3 && ops.ffn_calls() == 15);

  // (c) Reset, then the same prefill without taps.
  Check("reset", exec->Reset() && exec->Position() == 0);
  glm::ForwardResult again;
  Check("prefill after reset " + error,
        exec->Forward(ids.first(kPrefill), true, &again, nullptr, &error));
  CheckRel("prefill after reset vs first prefill (hidden)", again.hidden,
           first_hidden, 1e-6);
  CheckRel("prefill after reset vs first prefill (logits)", again.logits,
           first_logits, 1e-6);
}

// (b) + (d).
void CheckPassesAndRanges(const Model& m) {
  const glm::Config& c = m.config;
  const ref::GgufWeightSource source(m.weights);
  ref::ReferenceModel model(c, source);
  Taps cpu_taps;
  const auto want = model.Forward(m.ids, true, RefTap(cpu_taps));
  std::string error;

  {
    glm::stand_in::HostOps ops(c);
    auto exec =
        glm::Executor::Create(m.weights, Options(5), ops.Bind(), &error);
    glm::ForwardResult got;
    const bool ok = exec && exec->Forward(m.ids, true, &got, nullptr, &error);
    Check("14 tokens in passes of 5 " + error, ok);
    CheckRel("passes of 5: final hidden", got.hidden, want.hidden);
    CheckRel("passes of 5: logits", got.logits, want.logits);
  }

  glm::stand_in::HostOps ops(c);
  glm::ExecutorOptions front = Options(16);
  front.last_layer = 3;
  front.head = false;
  auto a = glm::Executor::Create(m.weights, front, ops.Bind(), &error);
  Check("create [0,3) with embedding, no head " + error, a != nullptr);
  glm::ExecutorOptions back = Options(16);
  back.first_layer = 3;
  back.embed = false;
  auto b = glm::Executor::Create(m.weights, back, ops.Bind(), &error);
  Check("create [3,5) with head " + error, b != nullptr);
  if (!a || !b)
    return;
  std::cout << "device bytes [0,3) " << a->DeviceBytes() << ", [3,5) "
            << b->DeviceBytes() << "\n";
  glm::ForwardResult mid;
  Taps gpu_taps;
  const glm::TapFn tap = GpuTap(gpu_taps);
  Check("[0,3) forward " + error, a->Forward(m.ids, false, &mid, &tap, &error));
  CheckRel("[0,3) streams vs reference layer 2 out0", mid.streams,
           cpu_taps.data.at("L2.out0"));
  Check("[0,3) reports no head", mid.hidden.empty() && mid.logits.empty());
  Check("[0,3) taps only its layers",
        !gpu_taps.data.contains("L3.in") && gpu_taps.data.contains("L2.out0"));
  glm::ForwardResult end;
  Check("[3,5) forward from streams " + error,
        b->ForwardStreams(mid.streams.data(),
                          static_cast<std::uint32_t>(m.ids.size()), true, &end,
                          nullptr, &error));
  CheckRel("[3,5) final hidden", end.hidden, want.hidden);
  CheckRel("[3,5) logits", end.logits, want.logits);
}

// (e).
void CheckRefusals(const Model& m) {
  std::string error;
  glm::stand_in::HostOps ops(m.config);
  glm::LayerOps no_mla = ops.Bind();
  no_mla.mla = nullptr;
  const bool refused_mla =
      glm::Executor::Create(m.weights, Options(16), no_mla, &error) == nullptr;
  Check("MLA range without an MLA operator refused (" + error + ")",
        refused_mla);
  glm::ExecutorOptions late = Options(16);
  late.first_layer = 3;
  const bool refused_embed =
      glm::Executor::Create(m.weights, late, ops.Bind(), &error) == nullptr;
  Check("embedding with a range past layer 0 refused (" + error + ")",
        refused_embed);
  glm::ExecutorOptions kda_only = Options(16);
  kda_only.last_layer = 3;
  Check("KDA-only range needs no MLA operator",
        glm::Executor::Create(m.weights, kda_only, no_mla, &error) != nullptr);

  // An FFN failure at layer 4 marks the executor failed until Reset().
  glm::LayerOps failing = ops.Bind();
  const auto ffn = failing.ffn;
  bool fail = true;
  failing.ffn = [&](const glm::OpContext& ctx, const glm::Config& c,
                    std::uint32_t il, const glm::FfnLayerDevice& w,
                    const float* x, std::uint32_t n, float* y) {
    return !(fail && il == 4) && ffn(ctx, c, il, w, x, n, y);
  };
  auto exec = glm::Executor::Create(m.weights, Options(16), failing, &error);
  glm::ForwardResult out;
  const bool first = exec && exec->Forward(m.ids, true, &out, nullptr, &error);
  Check("failing operator reported (" + error + ")",
        exec && !first && exec->Failed());
  fail = false;
  Check("call after a failure refused",
        exec && !exec->Forward(m.ids, true, &out, nullptr, &error));
  const ref::GgufWeightSource source(m.weights);
  ref::ReferenceModel model(m.config, source);
  const auto want = model.Forward(m.ids, true);
  error.clear();
  const bool again = exec && exec->Reset() &&
                     exec->Forward(m.ids, true, &out, nullptr, &error);
  Check("forward after Reset " + error, again);
  CheckRel("forward after failure + Reset: logits", out.logits, want.logits);
}

}  // namespace

int main() {
  try {
    int devices = 0;
    if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
      std::cerr << "no HIP device\n";
      return 77;
    }
    std::cout << std::scientific << std::setprecision(2);
    const Fixture fx = Load(GLM53_E2E_FIXTURE);
    Model m;
    BuildModel(m, fx);
    CheckIncremental(m);
    CheckPassesAndRanges(m);
    CheckRefusals(m);
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << "\n";
    return 1;
  }
  if (failures != 0) {
    std::cerr << failures << " check(s) failed\n";
    return 1;
  }
  std::cout << "glm53_flash executor_loop: PASS\n";
  return 0;
}
