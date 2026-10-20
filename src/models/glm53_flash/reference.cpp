#include "src/models/glm53_flash/reference.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/glm53_flash/reference_hc.hpp"
#include "src/models/glm53_flash/reference_kda.hpp"
#include "src/models/glm53_flash/reference_mla.hpp"
#include "src/models/glm53_flash/reference_moe.hpp"

namespace gufo::models::glm53_flash::reference {
namespace {

using core::GgmlType;

// q/k l2norm eps inside the KDA kernel call (HF:458), independent of rms_eps.
constexpr float kKdaL2Eps = 1e-6F;

/// y = x / sqrt(mean(x^2) + eps) * weight (HF Glm5NextTextRMSNorm, HF:74-79).
void RmsNorm(const float* x, const float* weight, std::size_t n, float eps,
             float* y) {
  double ss = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    ss += static_cast<double>(x[i]) * x[i];
  }
  const double inv = 1.0 / std::sqrt((ss / static_cast<double>(n)) + eps);
  for (std::size_t i = 0; i < n; ++i) {
    y[i] = static_cast<float>(static_cast<double>(x[i]) * inv * weight[i]);
  }
}

void DequantizeRow(const TensorRef& t, std::uint64_t expert, std::uint64_t row,
                   float* out) {
  const std::uint8_t* src = t.Expert(expert) + (row * t.RowBytes());
  const std::size_t k = t.cols;
  switch (t.type) {
    case GgmlType::kF32:
      std::memcpy(out, src, k * sizeof(float));
      return;
    case GgmlType::kF16:
      for (std::size_t i = 0; i < k; ++i) {
        std::uint16_t h = 0;
        std::memcpy(&h, src + (2 * i), 2);
        out[i] = quant::Fp16ToFloat(h);
      }
      return;
    case GgmlType::kBF16:
      for (std::size_t i = 0; i < k; ++i) {
        std::uint16_t h = 0;
        std::memcpy(&h, src + (2 * i), 2);
        const std::uint32_t bits = static_cast<std::uint32_t>(h) << 16;
        std::memcpy(out + i, &bits, 4);
      }
      return;
    case GgmlType::kQ8_0:
      quant::DequantizeQ8_0(src, out, k);
      return;
    case GgmlType::kQ2_K:
      quant::DequantizeQ2_K(src, out, k);
      return;
    case GgmlType::kQ3_K:
      quant::DequantizeQ3_K(src, out, k);
      return;
    case GgmlType::kQ4_K:
      quant::DequantizeQ4_K(src, out, k);
      return;
    case GgmlType::kQ5_K:
      quant::DequantizeQ5_K(src, out, k);
      return;
    case GgmlType::kQ6_K:
      quant::DequantizeQ6_K(src, out, k);
      return;
    case GgmlType::kQ8_K:
      quant::DequantizeQ8_K(src, out, k);
      return;
    case GgmlType::kIQ4_NL:
      quant::DequantizeIQ4_NL(src, out, k);
      return;
    case GgmlType::kIQ4_XS:
      quant::DequantizeIQ4_XS(src, out, k);
      return;
    case GgmlType::kIQ3_S:
      quant::DequantizeIQ3_S(src, out, k);
      return;
    case GgmlType::kIQ2_XXS:
      quant::DequantizeIQ2_XXS(src, out, k);
      return;
    case GgmlType::kIQ3_XXS:
      quant::DequantizeIQ3_XXS(src, out, k);
      return;
    case GgmlType::kIQ2_S:
      quant::DequantizeIQ2_S(src, out, k);
      return;
    default:
      throw std::runtime_error("reference: no CPU dequantizer for " +
                               std::string(t.name) + " (" +
                               std::string(core::ToString(t.type)) + ")");
  }
}

void ReadGgufRow(const void* ctx, std::uint64_t expert, std::uint64_t row,
                 float* out) {
  DequantizeRow(*static_cast<const TensorRef*>(ctx), expert, row, out);
}

[[noreturn]] void Missing(const std::string& name) {
  throw std::runtime_error("reference: missing tensor " + name);
}

void CheckShape(const MatrixView& m, const std::string& name,
                std::uint64_t cols, std::uint64_t rows,
                std::uint64_t experts = 1) {
  if (m.cols() != cols || m.rows() != rows || m.experts() != experts) {
    throw std::runtime_error(
        "reference: tensor " + name + " is [" + std::to_string(m.cols()) +
        ", " + std::to_string(m.rows()) + ", " + std::to_string(m.experts()) +
        "], expected [" + std::to_string(cols) + ", " + std::to_string(rows) +
        ", " + std::to_string(experts) + "]");
  }
}

/// Every row of a stacked matrix, contiguous [experts][rows][cols].
std::vector<float> Dense(const MatrixView& m) {
  std::vector<float> out(m.experts() * m.rows() * m.cols());
  for (std::uint64_t e = 0; e < m.experts(); ++e) {
    for (std::uint64_t r = 0; r < m.rows(); ++r) {
      m.Row(e, r, out.data() + (((e * m.rows()) + r) * m.cols()));
    }
  }
  return out;
}

}  // namespace

// --- MatrixView --------------------------------------------------------------

void MatrixView::Row(std::uint64_t expert, std::uint64_t row,
                     float* out) const {
  if (reader_ == nullptr || expert >= experts_ || row >= rows_) {
    throw std::out_of_range("reference: matrix row out of range");
  }
  reader_(ctx_, expert, row, out);
}

void MatrixView::MatVec(const float* x, float* y, std::uint64_t expert) const {
  std::vector<float> w(cols_);
  for (std::uint64_t r = 0; r < rows_; ++r) {
    Row(expert, r, w.data());
    double acc = 0.0;
    for (std::uint64_t c = 0; c < cols_; ++c) {
      acc += static_cast<double>(w[c]) * x[c];
    }
    y[r] = static_cast<float>(acc);
  }
}

// --- F32WeightSource ---------------------------------------------------------

void F32WeightSource::Add(std::string name, std::vector<float> data,
                          std::uint64_t cols, std::uint64_t rows,
                          std::uint64_t experts) {
  if (data.size() != cols * rows * experts) {
    throw std::invalid_argument("F32WeightSource: size mismatch for " + name);
  }
  tensors_[std::move(name)] = Tensor{std::move(data), cols, rows, experts};
}

bool F32WeightSource::Has(const std::string& name) const {
  return tensors_.contains(name);
}

MatrixView F32WeightSource::Matrix(const std::string& name) const {
  const auto it = tensors_.find(name);
  if (it == tensors_.end()) {
    Missing(name);
  }
  const Tensor& t = it->second;
  const MatrixView::RowReader read = [](const void* ctx, std::uint64_t expert,
                                        std::uint64_t row, float* out) {
    const auto* src = static_cast<const Tensor*>(ctx);
    const float* r =
        src->data.data() + (((expert * src->rows) + row) * src->cols);
    std::copy(r, r + src->cols, out);
  };
  return MatrixView(&t, read, t.cols, t.rows, t.experts);
}

const float* F32WeightSource::Vector(const std::string& name,
                                     std::size_t count) const {
  const auto it = tensors_.find(name);
  if (it == tensors_.end()) {
    Missing(name);
  }
  if (it->second.data.size() != count) {
    throw std::runtime_error("reference: tensor " + name + " has " +
                             std::to_string(it->second.data.size()) +
                             " elements, expected " + std::to_string(count));
  }
  return it->second.data.data();
}

// --- GgufWeightSource --------------------------------------------------------

GgufWeightSource::GgufWeightSource(const ModelWeights& w) {
  Index(w.token_embd);
  Index(w.output);
  Index(w.output_norm);
  for (const LayerWeights& l : w.layers) {
    for (const HcMixer* m : {&l.hc_attn, &l.hc_ffn}) {
      Index(m->fn);
      Index(m->base);
      Index(m->scale);
    }
    for (const TensorRef* t :
         {&l.attn_norm,       &l.ffn_norm,    &l.kda_q,
          &l.kda_k,           &l.kda_v,       &l.kda_conv_q,
          &l.kda_conv_k,      &l.kda_conv_v,  &l.kda_f_a,
          &l.kda_f_b,         &l.kda_dt_bias, &l.kda_a,
          &l.kda_beta,        &l.kda_g_a,     &l.kda_g_b,
          &l.kda_norm,        &l.attn_out,    &l.mla_q_a,
          &l.mla_q_a_norm,    &l.mla_q_b,     &l.mla_kv_a,
          &l.mla_kv_a_norm,   &l.mla_k_b,     &l.mla_v_b,
          &l.idx_q_b,         &l.idx_k,       &l.idx_k_norm,
          &l.idx_k_norm_bias, &l.idx_proj,    &l.idx_gate,
          &l.idx_ape,         &l.ffn_gate,    &l.ffn_up,
          &l.ffn_down,        &l.router,      &l.router_bias,
          &l.ffn_gate_exps,   &l.ffn_up_exps, &l.ffn_down_exps,
          &l.shexp_gate,      &l.shexp_up,    &l.shexp_down,
          &l.nextn_eh_proj,   &l.nextn_enorm, &l.nextn_hnorm,
          &l.nextn_head_norm}) {
      Index(*t);
    }
  }
}

void GgufWeightSource::Index(const TensorRef& t) {
  if (!t.empty() && !t.name.empty()) {
    tensors_.emplace(std::string(t.name), &t);
  }
}

bool GgufWeightSource::Has(const std::string& name) const {
  return tensors_.contains(name);
}

MatrixView GgufWeightSource::Matrix(const std::string& name) const {
  const auto it = tensors_.find(name);
  if (it == tensors_.end()) {
    Missing(name);
  }
  const TensorRef& t = *it->second;
  return MatrixView(&t, &ReadGgufRow, t.cols, t.rows, t.experts);
}

const float* GgufWeightSource::Vector(const std::string& name,
                                      std::size_t count) const {
  const auto it = tensors_.find(name);
  if (it == tensors_.end()) {
    Missing(name);
  }
  const TensorRef& t = *it->second;
  if (t.type != GgmlType::kF32 || t.ElementCount() != count) {
    throw std::runtime_error("reference: tensor " + name +
                             " is not an F32 tensor of " +
                             std::to_string(count) + " elements");
  }
  return static_cast<const float*>(t.data);
}

// --- ReferenceModel ----------------------------------------------------------

ReferenceModel::ReferenceModel(const Config& config,
                               const WeightSource& weights)
    : config_(config), weights_(weights) {
  if (config_.mixers.size() < config_.num_layers || config_.hc_count == 0 ||
      config_.hidden_size == 0) {
    throw std::invalid_argument("ReferenceModel: incomplete config");
  }
  Reset();
}

void ReferenceModel::Reset() {
  position_ = 0;
  failed_ = false;
  kda_.assign(config_.num_layers, {});
  mla_.assign(config_.num_layers, {});
  const std::size_t kda_dim = config_.KdaDim();
  for (std::uint32_t il = 0; il < config_.num_layers; ++il) {
    if (config_.IsKdaLayer(il)) {
      kda_[il].conv_history.assign(
          std::size_t{config_.ssm_conv_kernel - 1} * 3 * kda_dim, 0.0F);
      kda_[il].recurrent.assign(std::size_t{config_.num_heads} *
                                    config_.kda_head_dim * config_.kda_head_dim,
                                0.0F);
    }
  }
}

ReferenceModel::Output ReferenceModel::Forward(
    std::span<const std::int32_t> tokens, bool want_logits, const TapFn& tap) {
  const std::uint32_t n = static_cast<std::uint32_t>(tokens.size());
  const std::size_t hidden = config_.hidden_size;
  const std::size_t hc_dim = config_.HcDim();
  Output out;
  if (failed_) {
    throw std::logic_error(
        "ReferenceModel: a previous Forward failed partway; call Reset()");
  }
  if (n == 0) {
    return out;
  }

  // Four copies of the token embedding (HF:1493).
  const MatrixView embd = weights_.Matrix("token_embd.weight");
  CheckShape(embd, "token_embd.weight", hidden, config_.vocab_size);
  std::vector<float> H(n * hc_dim);
  std::vector<float> e(hidden);
  for (std::uint32_t t = 0; t < n; ++t) {
    if (tokens[t] < 0 || static_cast<std::uint32_t>(tokens[t]) >= embd.rows()) {
      throw std::out_of_range("ReferenceModel: token id out of range");
    }
    embd.Row(0, static_cast<std::uint64_t>(tokens[t]), e.data());
    HcInit(config_, e.data(), H.data() + (t * hc_dim));
  }

  // The layers append to the KDA and MLA caches before position_ advances.
  // If anything below throws, those caches no longer match position_, so the
  // model is marked failed instead of silently computing from mixed state.
  failed_ = true;
  for (std::uint32_t il = 0; il < config_.num_layers; ++il) {
    Layer(il, n, H, tap);
  }

  // Mean head, then the final RMSNorm (HF:1509).
  std::vector<float> head(n * hidden);
  for (std::uint32_t t = 0; t < n; ++t) {
    HcHeadMean(config_, H.data() + (t * hc_dim), head.data() + (t * hidden));
  }
  if (tap) {
    tap("hc_head.out", -1, head.data(), hidden, n);
  }
  const float* out_norm = weights_.Vector("output_norm.weight", hidden);
  out.hidden.resize(n * hidden);
  for (std::uint32_t t = 0; t < n; ++t) {
    RmsNorm(head.data() + (t * hidden), out_norm, hidden, config_.rms_eps,
            out.hidden.data() + (t * hidden));
  }
  if (tap) {
    tap("final_norm.out", -1, out.hidden.data(), hidden, n);
  }

  if (want_logits && weights_.Has("output.weight")) {
    const MatrixView output = weights_.Matrix("output.weight");
    CheckShape(output, "output.weight", hidden, config_.vocab_size);
    out.logits.resize(std::size_t{n} * config_.vocab_size);
    for (std::uint32_t t = 0; t < n; ++t) {
      output.MatVec(out.hidden.data() + (t * hidden),
                    out.logits.data() + (std::size_t{t} * config_.vocab_size));
    }
  }
  position_ += n;
  failed_ = false;
  return out;
}

void ReferenceModel::Layer(std::uint32_t il, std::uint32_t n,
                           std::vector<float>& H, const TapFn& tap) {
  const std::size_t hc = config_.hc_count;
  const std::size_t hidden = config_.hidden_size;
  const std::size_t hc_dim = config_.HcDim();
  const std::size_t mix_dim = config_.HcMixDim();
  const std::string p = "blk." + std::to_string(il) + ".";
  const auto emit = [&](const char* name, const std::vector<float>& v,
                        std::size_t width) {
    if (tap) {
      tap(name, static_cast<int>(il), v.data(), width, n);
    }
  };
  emit("in", H, hc_dim);

  std::vector<HcMix> mix(n);
  std::vector<float> x(n * hidden);
  std::vector<float> xn(n * hidden);
  std::vector<float> a(n * hidden);
  std::vector<float> H_new(n * hc_dim);
  std::vector<float> post(n * hc);
  std::vector<float> comb(n * hc * hc);

  // One half-layer: HcPre -> RMSNorm -> sub-layer -> HcPost (HF:1324-1346).
  const auto half = [&](const char* hc_name, const char* norm_name,
                        const char* tap_hc, const char* tap_norm,
                        const char* tap_out, auto&& sublayer) {
    const std::string hp = p + hc_name;
    const float* fn = weights_.Vector(hp + "_fn.weight", mix_dim * hc_dim);
    const float* base = weights_.Vector(hp + "_base.weight", mix_dim);
    const float* scale = weights_.Vector(hp + "_scale.weight", 3);
    for (std::uint32_t t = 0; t < n; ++t) {
      HcPre(config_, fn, base, scale, H.data() + (t * hc_dim),
            x.data() + (t * hidden), &mix[t]);
      std::copy(mix[t].post.begin(), mix[t].post.end(),
                post.begin() + static_cast<std::ptrdiff_t>(t * hc));
      std::copy(mix[t].comb.begin(), mix[t].comb.end(),
                comb.begin() + static_cast<std::ptrdiff_t>(t * hc * hc));
    }
    if (tap) {
      const std::string base_name(tap_hc);
      tap(base_name + ".out0", static_cast<int>(il), post.data(), hc, n);
      tap(base_name + ".out1", static_cast<int>(il), comb.data(), hc * hc, n);
      tap(base_name + ".out2", static_cast<int>(il), x.data(), hidden, n);
    }
    const float* norm_w = weights_.Vector(p + norm_name, hidden);
    for (std::uint32_t t = 0; t < n; ++t) {
      RmsNorm(x.data() + (t * hidden), norm_w, hidden, config_.rms_eps,
              xn.data() + (t * hidden));
    }
    emit(tap_norm, xn, hidden);
    sublayer();
    emit(tap_out, a, hidden);
    for (std::uint32_t t = 0; t < n; ++t) {
      HcPost(mix[t], a.data() + (t * hidden), H.data() + (t * hc_dim),
             H_new.data() + (t * hc_dim));
    }
    H.swap(H_new);
  };

  half("hc_attn", "attn_norm.weight", "attn_hc", "input_layernorm.out",
       "self_attn.out", [&] {
         if (config_.IsKdaLayer(il)) {
           Kda(il, xn.data(), n, a.data());
         } else {
           Mla(il, xn.data(), n, a.data());
         }
       });
  emit("ffn_hc.in", H, hc_dim);
  half("hc_ffn", "ffn_norm.weight", "ffn_hc", "post_attention_layernorm.out",
       "mlp.out", [&] {
         for (std::uint32_t t = 0; t < n; ++t) {
           Ffn(il, xn.data() + (t * hidden), a.data() + (t * hidden));
         }
       });
  emit("out0", H, hc_dim);
}

void ReferenceModel::Kda(std::uint32_t il, const float* x, std::uint32_t n,
                         float* y) {
  // Glm5NextTextLinearAttention.forward (HF:664-769).
  const std::size_t hidden = config_.hidden_size;
  const std::size_t heads = config_.num_heads;
  const std::size_t hd = config_.kda_head_dim;
  const std::size_t dim = config_.KdaDim();
  const std::size_t kernel = config_.ssm_conv_kernel;
  const std::string p = "blk." + std::to_string(il) + ".";
  KdaLayerState& st = kda_[il];

  const auto mat = [&](const std::string& name, std::uint64_t cols,
                       std::uint64_t rows) {
    MatrixView m = weights_.Matrix(p + name);
    CheckShape(m, p + name, cols, rows);
    return m;
  };

  // q|k|v projections, concatenated per token as the conv channels.
  const MatrixView wq = mat("attn_q.weight", hidden, dim);
  const MatrixView wk = mat("attn_k.weight", hidden, dim);
  const MatrixView wv = mat("attn_v.weight", hidden, dim);
  std::vector<float> qkv(n * 3 * dim);
  for (std::uint32_t t = 0; t < n; ++t) {
    float* row = qkv.data() + (t * 3 * dim);
    wq.MatVec(x + (t * hidden), row);
    wk.MatVec(x + (t * hidden), row + dim);
    wv.MatVec(x + (t * hidden), row + (2 * dim));
  }
  // Depthwise causal conv + SiLU over the 3*dim channels (HF:688-716).
  std::vector<float> conv_w(3 * dim * kernel);
  const char* parts[3] = {"ssm_conv1d_q.weight", "ssm_conv1d_k.weight",
                          "ssm_conv1d_v.weight"};
  for (std::size_t i = 0; i < 3; ++i) {
    const float* w = weights_.Vector(p + parts[i], dim * kernel);
    std::copy(w, w + (dim * kernel),
              conv_w.begin() + static_cast<std::ptrdiff_t>(i * dim * kernel));
  }
  std::vector<float> conv(n * 3 * dim);
  KdaConv(qkv.data(), conv_w.data(), st.conv_history.data(), conv.data(), n,
          static_cast<std::uint32_t>(3 * dim),
          static_cast<std::uint32_t>(kernel));
  std::vector<float> q(n * dim);
  std::vector<float> k(n * dim);
  std::vector<float> v(n * dim);
  for (std::uint32_t t = 0; t < n; ++t) {
    const float* row = conv.data() + (t * 3 * dim);
    std::copy(row, row + dim, q.begin() + static_cast<std::ptrdiff_t>(t * dim));
    std::copy(row + dim, row + (2 * dim),
              k.begin() + static_cast<std::ptrdiff_t>(t * dim));
    std::copy(row + (2 * dim), row + (3 * dim),
              v.begin() + static_cast<std::ptrdiff_t>(t * dim));
  }

  // Forget gate (HF:353-369) and beta (HF:733).
  const MatrixView f_a = mat("ssm_f_a.weight", hidden, hd);
  const MatrixView f_b = mat("ssm_f_b.weight", hd, dim);
  const MatrixView g_a = mat("ssm_g_a.weight", hidden, hd);
  const MatrixView g_b = mat("ssm_g_b.weight", hd, dim);
  const MatrixView w_beta = mat("ssm_beta.weight", hidden, heads);
  std::vector<float> f(n * dim);
  std::vector<float> gate(n * dim);
  std::vector<float> beta(n * heads);
  std::vector<float> low(hd);
  for (std::uint32_t t = 0; t < n; ++t) {
    f_a.MatVec(x + (t * hidden), low.data());
    f_b.MatVec(low.data(), f.data() + (t * dim));
    g_a.MatVec(x + (t * hidden), low.data());
    g_b.MatVec(low.data(), gate.data() + (t * dim));
    w_beta.MatVec(x + (t * hidden), beta.data() + (t * heads));
  }
  std::vector<float> g(n * dim);
  KdaGate(f.data(), weights_.Vector(p + "ssm_dt.bias", dim),
          weights_.Vector(p + "ssm_a", heads), config_.kda_gate_lower_bound,
          g.data(), n, static_cast<std::uint32_t>(heads),
          static_cast<std::uint32_t>(hd));
  KdaPrep(q.data(), k.data(), beta.data(), n, static_cast<std::uint32_t>(heads),
          static_cast<std::uint32_t>(hd), kKdaL2Eps);

  std::vector<float> o(n * dim);
  KdaRecurrent(q.data(), k.data(), v.data(), g.data(), beta.data(),
               st.recurrent.data(), o.data(), n,
               static_cast<std::uint32_t>(heads),
               static_cast<std::uint32_t>(hd), static_cast<std::uint32_t>(hd));
  KdaOutput(o.data(), gate.data(), weights_.Vector(p + "ssm_norm.weight", hd),
            config_.rms_eps, o.data(), n, static_cast<std::uint32_t>(heads),
            static_cast<std::uint32_t>(hd));

  const MatrixView wo = mat("attn_output.weight", dim, hidden);
  for (std::uint32_t t = 0; t < n; ++t) {
    wo.MatVec(o.data() + (t * dim), y + (t * hidden));
  }
}

void ReferenceModel::Mla(std::uint32_t il, const float* x, std::uint32_t n,
                         float* y) {
  // Glm5NextTextAttention.forward (HF:1174-1235), absorbed as llama.cpp does,
  // with the k-pool indexer (HF:810-1060).
  const std::size_t hidden = config_.hidden_size;
  const std::size_t heads = config_.num_heads;
  const std::size_t q_lora = config_.q_lora_rank;
  const std::size_t kv_lora = config_.kv_lora_rank;
  const std::size_t nope = config_.mla_head_dim;
  const std::size_t v_dim = config_.mla_value_dim;
  const std::size_t ih = config_.indexer_heads;
  const std::size_t idim = config_.indexer_head_dim;
  const std::size_t kpool = config_.indexer_kpool;
  const std::string p = "blk." + std::to_string(il) + ".";
  MlaLayerState& st = mla_[il];

  const auto mat = [&](const std::string& name, std::uint64_t cols,
                       std::uint64_t rows, std::uint64_t experts = 1) {
    MatrixView m = weights_.Matrix(p + name);
    CheckShape(m, p + name, cols, rows, experts);
    return m;
  };
  const MatrixView q_a = mat("attn_q_a.weight", hidden, q_lora);
  const MatrixView q_b = mat("attn_q_b.weight", q_lora, heads * nope);
  const MatrixView kv_a = mat("attn_kv_a_mqa.weight", hidden, kv_lora);
  const std::vector<float> k_b =
      Dense(mat("attn_k_b.weight", nope, kv_lora, heads));
  const std::vector<float> v_b =
      Dense(mat("attn_v_b.weight", kv_lora, v_dim, heads));
  const MatrixView wo = mat("attn_output.weight", heads * v_dim, hidden);
  const MatrixView idx_q_b = mat("indexer.attn_q_b.weight", q_lora, ih * idim);
  const MatrixView idx_k = mat("indexer.attn_k.weight", hidden, idim);
  const MatrixView idx_proj = mat("indexer.proj.weight", hidden, ih);
  const MatrixView idx_gate =
      mat("indexer_compressor_gate.weight", hidden, idim);
  const float* q_a_norm = weights_.Vector(p + "attn_q_a_norm.weight", q_lora);
  const float* kv_a_norm =
      weights_.Vector(p + "attn_kv_a_norm.weight", kv_lora);
  const float* k_norm_w = weights_.Vector(p + "indexer.k_norm.weight", idim);
  const float* k_norm_b = weights_.Vector(p + "indexer.k_norm.bias", idim);
  const float* ape =
      weights_.Vector(p + "indexer_compressor_ape.weight", kpool * idim);

  std::vector<float> tmp(std::max({q_lora, kv_lora, idim}));
  std::vector<float> q_resid(q_lora);
  std::vector<float> q(heads * nope);
  std::vector<float> q_abs(heads * kv_lora);
  std::vector<float> ctx(heads * kv_lora);
  std::vector<float> attn(heads * v_dim);
  std::vector<float> iq(ih * idim);
  std::vector<float> iw(ih);
  std::vector<float> key(idim);
  std::vector<float> pooled(idim);
  std::vector<float> scores;

  for (std::uint32_t t = 0; t < n; ++t) {
    const float* xt = x + (t * hidden);
    const std::uint32_t pos = position_ + t;

    // Query: q_resid = RMSNorm(q_a x), q = q_b q_resid (HF:1181-1182).
    q_a.MatVec(xt, tmp.data());
    RmsNorm(tmp.data(), q_a_norm, q_lora, config_.rms_eps, q_resid.data());
    q_b.MatVec(q_resid.data(), q.data());

    // Latent cache row: RMSNorm(kv_a x) (HF:1184-1186).
    kv_a.MatVec(xt, tmp.data());
    st.latent.resize((std::size_t{pos} + 1) * kv_lora);
    RmsNorm(tmp.data(), kv_a_norm, kv_lora, config_.rms_eps,
            st.latent.data() + (std::size_t{pos} * kv_lora));

    // Indexer key and pooling gate for this token (HF:829-832); a pool is
    // compressed once its kpool-th token arrives (HF:942-994).
    idx_k.MatVec(xt, tmp.data());
    LayerNorm(tmp.data(), k_norm_w, k_norm_b, idim, config_.norm_eps,
              key.data());
    st.tail_keys.insert(st.tail_keys.end(), key.begin(), key.end());
    idx_gate.MatVec(xt, tmp.data());
    st.tail_gates.insert(st.tail_gates.end(), tmp.begin(),
                         tmp.begin() + static_cast<std::ptrdiff_t>(idim));
    if (st.tail_keys.size() == kpool * idim) {
      PoolKeys(st.tail_keys.data(), st.tail_gates.data(), ape, kpool, idim,
               pooled.data());
      st.pooled.insert(st.pooled.end(), pooled.begin(), pooled.end());
      st.tail_keys.clear();
      st.tail_gates.clear();
    }

    // Pool scores and selection: top budget complete visible pools plus the
    // raw tail (HF:851-905, 1009-1060).
    idx_q_b.MatVec(q_resid.data(), iq.data());
    idx_proj.MatVec(xt, iw.data());
    const std::size_t pools = st.pooled.size() / idim;
    scores.assign(pools, 0.0F);
    IndexerScores(iq.data(), st.pooled.data(), iw.data(), ih, idim, pools,
                  scores.data());
    const Selection sel =
        SelectPools(scores.data(), pos + 1, 0, config_.indexer_kpool,
                    config_.IndexerPoolBudget(), /*select_tail=*/true);

    // Absorbed attention over the latent rows (K = V = latent), scale
    // 1/sqrt(nope) (HF:1158 scaling = qk_head_dim**-0.5), then v_b and wo.
    AbsorbQuery(q.data(), k_b.data(), heads, nope, kv_lora, q_abs.data());
    LatentAttention(q_abs.data(), st.latent.data(), std::size_t{pos} + 1, heads,
                    kv_lora, nope, sel.tokens, ctx.data());
    ProjectValues(ctx.data(), v_b.data(), heads, kv_lora, v_dim, attn.data());
    wo.MatVec(attn.data(), y + (t * hidden));
  }
}

void ReferenceModel::Ffn(std::uint32_t il, const float* x, float* y) const {
  const std::size_t hidden = config_.hidden_size;
  const std::string p = "blk." + std::to_string(il) + ".";
  const auto clamp_of = [&](const std::vector<float>& v) {
    return il < v.size() ? v[il] : 0.0F;
  };
  // Clamped SwiGLU MLP (HF:97-103); llama.cpp clamps the dense FFN with the
  // routed value.
  const auto swiglu = [&](const MatrixView& gate, const MatrixView& up,
                          const MatrixView& down, std::uint64_t expert,
                          float clamp, std::vector<float>& out) {
    const std::size_t ff = gate.rows();
    std::vector<float> g(ff);
    std::vector<float> u(ff);
    gate.MatVec(x, g.data(), expert);
    up.MatVec(x, u.data(), expert);
    for (std::size_t i = 0; i < ff; ++i) {
      g[i] = ClampedSwiGlu(g[i], u[i], clamp);
    }
    out.resize(hidden);
    down.MatVec(g.data(), out.data(), expert);
  };
  const auto mat = [&](const std::string& name, std::uint64_t cols,
                       std::uint64_t rows, std::uint64_t experts = 1) {
    MatrixView m = weights_.Matrix(p + name);
    CheckShape(m, p + name, cols, rows, experts);
    return m;
  };

  std::vector<float> part;
  if (config_.IsDenseFfnLayer(il)) {
    const std::size_t ff = config_.dense_ff;
    swiglu(mat("ffn_gate.weight", hidden, ff), mat("ffn_up.weight", hidden, ff),
           mat("ffn_down.weight", ff, hidden), 0,
           clamp_of(config_.swiglu_clamp_exp), part);
    std::copy(part.begin(), part.end(), y);
    return;
  }

  // Sigmoid router with selection-only bias (HF:157-182), llama.cpp
  // denominator.
  const std::size_t ne = config_.num_experts;
  const float* router = weights_.Vector(p + "ffn_gate_inp.weight", ne * hidden);
  const float* bias = weights_.Vector(p + "exp_probs_b.bias", ne);
  std::vector<float> logits(ne);
  for (std::size_t e = 0; e < ne; ++e) {
    double acc = 0.0;
    for (std::size_t d = 0; d < hidden; ++d) {
      acc += static_cast<double>(router[(e * hidden) + d]) * x[d];
    }
    logits[e] = static_cast<float>(acc);
  }
  const Routing routing =
      Route(logits.data(), bias, static_cast<std::uint32_t>(ne),
            config_.num_experts_used, config_.expert_weights_scale,
            config_.expert_weights_norm, RouterDenominator::kLlamaClamp);

  const std::size_t eff = config_.expert_ff;
  const MatrixView gate_exps = mat("ffn_gate_exps.weight", hidden, eff, ne);
  const MatrixView up_exps = mat("ffn_up_exps.weight", hidden, eff, ne);
  const MatrixView down_exps = mat("ffn_down_exps.weight", eff, hidden, ne);
  std::vector<double> acc(hidden, 0.0);
  for (std::size_t i = 0; i < routing.ids.size(); ++i) {
    swiglu(gate_exps, up_exps, down_exps, routing.ids[i],
           clamp_of(config_.swiglu_clamp_exp), part);
    for (std::size_t d = 0; d < hidden; ++d) {
      acc[d] += static_cast<double>(routing.weights[i]) * part[d];
    }
  }
  // Shared expert, added unweighted (HF:204).
  const std::size_t sff = config_.shared_expert_ff;
  swiglu(mat("ffn_gate_shexp.weight", hidden, sff),
         mat("ffn_up_shexp.weight", hidden, sff),
         mat("ffn_down_shexp.weight", sff, hidden), 0,
         clamp_of(config_.swiglu_clamp_shexp), part);
  for (std::size_t d = 0; d < hidden; ++d) {
    y[d] = static_cast<float>(acc[d] + part[d]);
  }
}

}  // namespace gufo::models::glm53_flash::reference
