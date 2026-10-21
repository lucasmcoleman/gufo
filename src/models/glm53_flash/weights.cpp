#include "src/models/glm53_flash/weights.hpp"

#include <string>
#include <utility>

#include "src/core/quant/ggml_dequant.hpp"

namespace gufo::models::glm53_flash {
namespace {

using core::GgmlType;

/// Bytes per element of the dense formats; 0 for block-quantized types.
[[nodiscard]] std::size_t DenseBytes(GgmlType type) noexcept {
  switch (type) {
    case GgmlType::kF32:
      return 4;
    case GgmlType::kF16:
    case GgmlType::kBF16:
      return 2;
    default:
      return 0;
  }
}

/// What a tensor may be stored as. Matrices accept any block format the core
/// can size (kernel coverage is checked by the executor, not the loader);
/// norms, biases, router, mixing and convolution weights must be F32.
enum class Kind : std::uint8_t { kMatrix, kF32 };

struct Binder {
  const core::GgufReader& reader;
  std::string* error;
  bool ok{true};

  void Fail(const std::string& message) {
    if (ok && error != nullptr) {
      *error = message;
    }
    ok = false;
  }

  /// Binds `name` with the exact shape [cols, rows, experts]. A 3-D GGUF
  /// shape [a, 1, b] (the depthwise conv weights) is written as rows=1.
  TensorRef Get(const std::string& name, std::uint64_t cols, std::uint64_t rows,
                std::uint64_t experts, Kind kind, bool required = true) {
    TensorRef t;
    const auto* info = reader.FindTensor(name);
    if (info == nullptr) {
      if (required) {
        Fail("missing tensor " + name);
      }
      return t;
    }
    const auto& d = info->dimensions;
    const std::uint64_t got_cols = d.size() > 0 ? d[0] : 1;
    const std::uint64_t got_rows = d.size() > 1 ? d[1] : 1;
    const std::uint64_t got_experts = d.size() > 2 ? d[2] : 1;
    if (got_cols != cols || got_rows != rows || got_experts != experts ||
        d.size() > 3) {
      Fail("tensor " + name + " has shape [" + std::to_string(got_cols) + ", " +
           std::to_string(got_rows) + ", " + std::to_string(got_experts) +
           "], expected [" + std::to_string(cols) + ", " +
           std::to_string(rows) + ", " + std::to_string(experts) + "]");
      return t;
    }
    const std::size_t dense = DenseBytes(info->type);
    const std::size_t block = ::gufo::quant::QuantizedBlockElements(info->type);
    const bool format_ok =
        kind == Kind::kF32
            ? info->type == GgmlType::kF32
            : (dense != 0 ||
               (block != 0 && cols % block == 0 &&
                ::gufo::quant::QuantizedRowBytes(info->type, cols) != 0));
    if (!format_ok) {
      Fail("tensor " + name + " has unsupported format " +
           std::string(core::ToString(info->type)));
      return t;
    }
    t.data = info->data;
    t.type = info->type;
    t.cols = cols;
    t.rows = rows;
    t.experts = experts;
    t.name = info->name;
    // Locate the shard so disk readers can address the payload directly, and
    // so a payload running past its shard is caught here, not as a fault.
    const auto regions = reader.GetMappedRegions();
    const auto address = reinterpret_cast<std::uintptr_t>(info->data);
    bool inside = false;
    for (std::uint32_t i = 0; i < regions.size(); ++i) {
      const auto base = reinterpret_cast<std::uintptr_t>(regions[i].data);
      if (address >= base && address < base + regions[i].size) {
        t.shard = i;
        t.file_offset = address - base;
        inside = t.SizeBytes() <= regions[i].size - t.file_offset;
        break;
      }
    }
    if (!inside) {
      Fail("tensor " + name + " is truncated");
      return TensorRef{};
    }
    return t;
  }

  TensorRef Matrix(const std::string& name, std::uint64_t cols,
                   std::uint64_t rows, std::uint64_t stacked = 1) {
    return Get(name, cols, rows, stacked, Kind::kMatrix);
  }
  TensorRef Vector(const std::string& name, std::uint64_t n,
                   std::uint64_t rows = 1) {
    return Get(name, n, rows, 1, Kind::kF32);
  }

  HcMixer Mixer(const std::string& prefix, const Config& c) {
    HcMixer m;
    m.fn = Get(prefix + "_fn.weight", c.HcDim(), c.HcMixDim(), 1, Kind::kF32);
    m.base = Vector(prefix + "_base.weight", c.HcMixDim());
    m.scale = Vector(prefix + "_scale.weight", 3);
    return m;
  }

  LayerWeights Layer(const Config& c, std::uint32_t il) {
    LayerWeights l;
    l.mixer = c.mixers[il];
    l.nextn = il >= c.num_layers;
    l.dense_ffn = c.IsDenseFfnLayer(il);
    const std::string p = "blk." + std::to_string(il) + ".";
    const std::uint64_t h = c.hidden_size;

    if (!l.nextn) {
      l.hc_attn = Mixer(p + "hc_attn", c);
      l.hc_ffn = Mixer(p + "hc_ffn", c);
    }
    l.attn_norm = Vector(p + "attn_norm.weight", h);
    l.ffn_norm = Vector(p + "ffn_norm.weight", h);

    if (l.mixer == Config::Mixer::kKda) {
      const std::uint64_t dim = c.KdaDim();
      const std::uint64_t hd = c.kda_head_dim;
      l.kda_q = Matrix(p + "attn_q.weight", h, dim);
      l.kda_k = Matrix(p + "attn_k.weight", h, dim);
      l.kda_v = Matrix(p + "attn_v.weight", h, dim);
      // GGUF [kernel, 1, channels]: one depthwise filter per channel.
      l.kda_conv_q =
          Get(p + "ssm_conv1d_q.weight", c.ssm_conv_kernel, 1, dim, Kind::kF32);
      l.kda_conv_k =
          Get(p + "ssm_conv1d_k.weight", c.ssm_conv_kernel, 1, dim, Kind::kF32);
      l.kda_conv_v =
          Get(p + "ssm_conv1d_v.weight", c.ssm_conv_kernel, 1, dim, Kind::kF32);
      l.kda_f_a = Matrix(p + "ssm_f_a.weight", h, hd);
      l.kda_f_b = Matrix(p + "ssm_f_b.weight", hd, dim);
      l.kda_dt_bias = Vector(p + "ssm_dt.bias", dim);
      l.kda_a = Vector(p + "ssm_a", c.num_heads);
      l.kda_beta = Matrix(p + "ssm_beta.weight", h, c.num_heads);
      l.kda_g_a = Matrix(p + "ssm_g_a.weight", h, hd);
      l.kda_g_b = Matrix(p + "ssm_g_b.weight", hd, dim);
      l.kda_norm = Vector(p + "ssm_norm.weight", hd);
      l.attn_out = Matrix(p + "attn_output.weight", dim, h);
    } else {
      const std::uint64_t heads = c.num_heads;
      l.mla_q_a = Matrix(p + "attn_q_a.weight", h, c.q_lora_rank);
      l.mla_q_a_norm = Vector(p + "attn_q_a_norm.weight", c.q_lora_rank);
      l.mla_q_b = Matrix(p + "attn_q_b.weight", c.q_lora_rank, c.MlaQDim());
      l.mla_kv_a = Matrix(p + "attn_kv_a_mqa.weight", h, c.kv_lora_rank);
      l.mla_kv_a_norm = Vector(p + "attn_kv_a_norm.weight", c.kv_lora_rank);
      l.mla_k_b =
          Matrix(p + "attn_k_b.weight", c.mla_head_dim, c.kv_lora_rank, heads);
      l.mla_v_b =
          Matrix(p + "attn_v_b.weight", c.kv_lora_rank, c.mla_value_dim, heads);
      l.attn_out = Matrix(p + "attn_output.weight", c.MlaOutDim(), h);
      l.idx_q_b = Matrix(p + "indexer.attn_q_b.weight", c.q_lora_rank,
                         std::uint64_t{c.indexer_heads} * c.indexer_head_dim);
      l.idx_k = Matrix(p + "indexer.attn_k.weight", h, c.indexer_head_dim);
      l.idx_k_norm = Vector(p + "indexer.k_norm.weight", c.indexer_head_dim);
      l.idx_k_norm_bias = Vector(p + "indexer.k_norm.bias", c.indexer_head_dim);
      l.idx_proj = Matrix(p + "indexer.proj.weight", h, c.indexer_heads);
      l.idx_gate =
          Matrix(p + "indexer_compressor_gate.weight", h, c.indexer_head_dim);
      l.idx_ape = Vector(p + "indexer_compressor_ape.weight",
                         c.indexer_head_dim, c.indexer_kpool);
    }

    if (l.dense_ffn) {
      l.ffn_gate = Matrix(p + "ffn_gate.weight", h, c.dense_ff);
      l.ffn_up = Matrix(p + "ffn_up.weight", h, c.dense_ff);
      l.ffn_down = Matrix(p + "ffn_down.weight", c.dense_ff, h);
    } else {
      const std::uint64_t e = c.num_experts;
      l.router = Get(p + "ffn_gate_inp.weight", h, e, 1, Kind::kF32);
      l.router_bias = Vector(p + "exp_probs_b.bias", e);
      l.ffn_gate_exps = Matrix(p + "ffn_gate_exps.weight", h, c.expert_ff, e);
      l.ffn_up_exps = Matrix(p + "ffn_up_exps.weight", h, c.expert_ff, e);
      l.ffn_down_exps = Matrix(p + "ffn_down_exps.weight", c.expert_ff, h, e);
      l.shexp_gate = Matrix(p + "ffn_gate_shexp.weight", h, c.shared_expert_ff);
      l.shexp_up = Matrix(p + "ffn_up_shexp.weight", h, c.shared_expert_ff);
      l.shexp_down = Matrix(p + "ffn_down_shexp.weight", c.shared_expert_ff, h);
    }

    if (l.nextn) {
      l.nextn_eh_proj = Matrix(p + "nextn.eh_proj.weight", 2 * h, h);
      l.nextn_enorm = Vector(p + "nextn.enorm.weight", h);
      l.nextn_hnorm = Vector(p + "nextn.hnorm.weight", h);
      l.nextn_head_norm = Vector(p + "nextn.shared_head_norm.weight", h);
    }
    return l;
  }
};

}  // namespace

std::size_t TensorRef::RowBytes() const noexcept {
  const std::size_t dense = DenseBytes(type);
  if (dense != 0) {
    return dense * cols;
  }
  return ::gufo::quant::QuantizedRowBytes(type, cols);
}

std::optional<ModelWeights> ModelWeights::Bind(const core::GgufReader& reader,
                                               bool bind_nextn,
                                               std::string* error_msg) {
  auto config = Config::FromGguf(reader, error_msg);
  if (!config) {
    return std::nullopt;
  }
  ModelWeights w;
  w.config = std::move(*config);
  const Config& c = w.config;
  Binder b{reader, error_msg};

  w.token_embd = b.Matrix("token_embd.weight", c.hidden_size, c.vocab_size);
  w.output = b.Matrix("output.weight", c.hidden_size, c.vocab_size);
  w.output_norm = b.Vector("output_norm.weight", c.hidden_size);

  const std::uint32_t n = bind_nextn ? c.num_layers_all : c.num_layers;
  w.layers.reserve(n);
  for (std::uint32_t il = 0; il < n && b.ok; ++il) {
    w.layers.push_back(b.Layer(c, il));
  }
  if (!b.ok) {
    return std::nullopt;
  }
  return w;
}

}  // namespace gufo::models::glm53_flash
