#include "src/models/laguna/weights.hpp"

#include <algorithm>
#include <initializer_list>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace gufo::models::laguna {
namespace {

using core::GgmlType;

struct Format {
  std::uint32_t block;
  std::uint32_t bytes;
};

/// Storage geometry of the formats a Laguna artifact may carry. Unknown types
/// get a zero block, which every validation below rejects.
[[nodiscard]] Format FormatOf(GgmlType type) noexcept {
  switch (type) {
    case GgmlType::kF32:
      return {1, 4};
    case GgmlType::kF16:
    case GgmlType::kBF16:
      return {1, 2};
    case GgmlType::kQ8_0:
      return {32, 34};
    case GgmlType::kQ4_K:
      return {256, 144};
    case GgmlType::kQ5_K:
      return {256, 176};
    case GgmlType::kQ6_K:
      return {256, 210};
    case GgmlType::kIQ4_XS:
      return {256, 136};
    default:
      return {0, 0};
  }
}

constexpr std::initializer_list<GgmlType> kF32Only = {GgmlType::kF32};
/// Projection and expert matrices: whichever quantization the artifact picked.
constexpr std::initializer_list<GgmlType> kMatrix = {
    GgmlType::kQ8_0,   GgmlType::kQ6_K, GgmlType::kQ5_K, GgmlType::kQ4_K,
    GgmlType::kIQ4_XS, GgmlType::kF16,  GgmlType::kBF16, GgmlType::kF32};

}  // namespace

std::size_t TensorRef::RowBytes() const noexcept {
  const Format format = FormatOf(type);
  return format.block == 0 ? 0 : (cols / format.block) * format.bytes;
}

namespace {

class TensorBinder {
public:
  TensorBinder(std::span<const core::GgufTensorInfo> tensors,
               std::span<const core::GgufMappedRegion> regions,
               std::string* error)
      : tensors_(tensors), regions_(regions), error_(error) {
    by_name_.reserve(tensors.size());
    for (const auto& info : tensors) {
      if (!by_name_.emplace(info.name, &info).second) {
        Fail("duplicate tensor " + std::string(info.name));
      }
    }
  }

  [[nodiscard]] bool ok() const noexcept { return ok_; }
  [[nodiscard]] std::size_t claimed() const noexcept { return claimed_.size(); }
  [[nodiscard]] std::uint64_t claimed_bytes() const noexcept {
    return claimed_bytes_;
  }

  void Fail(const std::string& message) {
    if (ok_ && error_ != nullptr) {
      *error_ = message;
    }
    ok_ = false;
  }

  /// Binds `name` with the exact shape and one of the accepted formats. The
  /// expected rank follows from the shape: 1D, 2D, or 3D when stacked.
  TensorRef Get(const std::string& name, std::uint64_t cols, std::uint64_t rows,
                std::uint64_t experts, std::initializer_list<GgmlType> types) {
    const auto found = by_name_.find(name);
    if (found == by_name_.end()) {
      Fail("missing tensor " + name);
      return {};
    }
    const core::GgufTensorInfo& info = *found->second;
    if (!claimed_.insert(info.name).second) {
      Fail("tensor claimed twice: " + name);
      return {};
    }
    const std::size_t rank = experts > 1 ? 3 : (rows > 1 ? 2 : 1);
    const auto& d = info.dimensions;
    const bool shape_ok = d.size() == rank && d[0] == cols &&
                          (rank < 2 || d[1] == rows) &&
                          (rank < 3 || d[2] == experts);
    if (!shape_ok) {
      std::string got = "[";
      for (std::size_t i = 0; i < d.size(); ++i) {
        got += (i ? ", " : "") + std::to_string(d[i]);
      }
      Fail("tensor " + name + " has shape " + got + "], expected [" +
           std::to_string(cols) +
           (rank > 1 ? ", " + std::to_string(rows) : "") +
           (rank > 2 ? ", " + std::to_string(experts) : "") + "]");
      return {};
    }
    const bool type_ok = std::ranges::find(types, info.type) != types.end();
    const Format format = FormatOf(info.type);
    if (!type_ok || format.block == 0 || cols % format.block != 0) {
      Fail("tensor " + name + " has unsupported format " +
           std::string(core::ToString(info.type)));
      return {};
    }
    TensorRef t;
    t.data = info.data;
    t.type = info.type;
    t.cols = cols;
    t.rows = rows;
    t.experts = experts;
    t.name = info.name;
    if (info.size_bytes != t.SizeBytes()) {
      Fail("tensor " + name + " records " + std::to_string(info.size_bytes) +
           " bytes, its shape and format need " +
           std::to_string(t.SizeBytes()));
      return {};
    }
    if (!regions_.empty()) {
      // Locate the mapping so disk readers can address the payload directly,
      // and so a payload running past its mapping is caught here rather than
      // as a fault later.
      const auto address = reinterpret_cast<std::uintptr_t>(info.data);
      bool inside = false;
      for (std::uint32_t i = 0; i < regions_.size(); ++i) {
        const auto base = reinterpret_cast<std::uintptr_t>(regions_[i].data);
        if (address >= base && address < base + regions_[i].size) {
          t.shard = i;
          t.file_offset = address - base;
          inside = t.SizeBytes() <= regions_[i].size - t.file_offset;
          break;
        }
      }
      if (!inside) {
        Fail("tensor " + name + " is truncated");
        return {};
      }
    }
    claimed_bytes_ += t.SizeBytes();
    return t;
  }

  /// Fails when the list holds a tensor no Get() claimed.
  void RequireNoLeftovers() {
    if (!ok_ || claimed_.size() == tensors_.size()) {
      return;
    }
    std::string names;
    std::size_t count = 0;
    for (const auto& info : tensors_) {
      if (!claimed_.contains(info.name)) {
        if (count++ < 8) {
          names += (names.empty() ? "" : ", ") + std::string(info.name);
        }
      }
    }
    Fail(std::to_string(count) + " tensor(s) not part of the Laguna layout: " +
         names + (count > 8 ? ", ..." : ""));
  }

private:
  std::span<const core::GgufTensorInfo> tensors_;
  std::span<const core::GgufMappedRegion> regions_;
  std::string* error_;
  bool ok_{true};
  std::unordered_map<std::string_view, const core::GgufTensorInfo*> by_name_;
  std::unordered_set<std::string_view> claimed_;
  std::uint64_t claimed_bytes_{0};
};

LayerWeights BindLayer(TensorBinder& b, const Config& c, std::uint32_t il) {
  LayerWeights l;
  l.full_attention = c.is_full_attention(il);
  l.moe = c.is_moe(il);
  const std::string p = "blk." + std::to_string(il) + ".";
  const std::uint64_t hidden = c.hidden_size;
  const std::uint64_t q_dim = c.q_dim(il);
  const std::uint64_t kv_dim = c.kv_dim();

  l.attn_norm = b.Get(p + "attn_norm.weight", hidden, 1, 1, kF32Only);
  l.attn_q = b.Get(p + "attn_q.weight", hidden, q_dim, 1, kMatrix);
  l.attn_k = b.Get(p + "attn_k.weight", hidden, kv_dim, 1, kMatrix);
  l.attn_v = b.Get(p + "attn_v.weight", hidden, kv_dim, 1, kMatrix);
  l.attn_output = b.Get(p + "attn_output.weight", q_dim, hidden, 1, kMatrix);
  l.attn_q_norm = b.Get(p + "attn_q_norm.weight", c.head_dim, 1, 1, kF32Only);
  l.attn_k_norm = b.Get(p + "attn_k_norm.weight", c.head_dim, 1, 1, kF32Only);
  // One scalar per head. llama.cpp also accepts a per-element gate
  // (heads * head_dim rows, Laguna-M.1); this runtime does not implement it.
  l.attn_gate =
      b.Get(p + "attn_gate.weight", hidden, c.n_head(il), 1, kF32Only);
  l.ffn_norm = b.Get(p + "ffn_norm.weight", hidden, 1, 1, kF32Only);

  if (l.moe) {
    l.ffn_gate_inp =
        b.Get(p + "ffn_gate_inp.weight", hidden, c.num_experts, 1, kF32Only);
    l.ffn_exp_probs_b =
        b.Get(p + "exp_probs_b.bias", c.num_experts, 1, 1, kF32Only);
    l.ffn_gate_exps = b.Get(p + "ffn_gate_exps.weight", hidden, c.expert_ff,
                            c.num_experts, kMatrix);
    l.ffn_up_exps = b.Get(p + "ffn_up_exps.weight", hidden, c.expert_ff,
                          c.num_experts, kMatrix);
    l.ffn_down_exps = b.Get(p + "ffn_down_exps.weight", c.expert_ff, hidden,
                            c.num_experts, kMatrix);
    l.ffn_gate_shexp = b.Get(p + "ffn_gate_shexp.weight", hidden,
                             c.shared_expert_ff, 1, kMatrix);
    l.ffn_up_shexp = b.Get(p + "ffn_up_shexp.weight", hidden,
                           c.shared_expert_ff, 1, kMatrix);
    l.ffn_down_shexp = b.Get(p + "ffn_down_shexp.weight", c.shared_expert_ff,
                             hidden, 1, kMatrix);
  } else {
    l.ffn_gate = b.Get(p + "ffn_gate.weight", hidden, c.ffn_size, 1, kMatrix);
    l.ffn_up = b.Get(p + "ffn_up.weight", hidden, c.ffn_size, 1, kMatrix);
    l.ffn_down = b.Get(p + "ffn_down.weight", c.ffn_size, hidden, 1, kMatrix);
  }
  return l;
}

}  // namespace

std::size_t ModelWeights::ExpectedTensorCount(const Config& config) {
  const std::size_t dense = config.leading_dense_layers;
  const std::size_t moe = config.num_layers - dense;
  return 3 + 9 * static_cast<std::size_t>(config.num_layers) + 3 * dense +
         8 * moe;
}

std::optional<ModelWeights> ModelWeights::Bind(
    const Config& config, std::span<const core::GgufTensorInfo> tensors,
    std::span<const core::GgufMappedRegion> regions, std::string* error_msg) {
  TensorBinder b(tensors, regions, error_msg);
  ModelWeights w;
  w.config = config;
  const std::uint64_t hidden = config.hidden_size;
  w.token_embd =
      b.Get("token_embd.weight", hidden, config.vocab_size, 1, kMatrix);
  w.output_norm = b.Get("output_norm.weight", hidden, 1, 1, kF32Only);
  w.output = b.Get("output.weight", hidden, config.vocab_size, 1, kMatrix);
  w.layers.reserve(config.num_layers);
  for (std::uint32_t il = 0; il < config.num_layers && b.ok(); ++il) {
    w.layers.push_back(BindLayer(b, config, il));
  }
  b.RequireNoLeftovers();
  if (!b.ok()) {
    return std::nullopt;
  }
  w.tensor_count = b.claimed();
  w.total_bytes = b.claimed_bytes();
  if (w.tensor_count != ExpectedTensorCount(config)) {
    if (error_msg != nullptr) {
      *error_msg = "bound " + std::to_string(w.tensor_count) +
                   " tensors, the layout needs " +
                   std::to_string(ExpectedTensorCount(config));
    }
    return std::nullopt;
  }
  return w;
}

std::optional<ModelWeights> ModelWeights::Bind(const core::GgufReader& reader,
                                               std::string* error_msg) {
  const auto config = Config::FromGguf(reader, error_msg);
  if (!config.has_value()) {
    return std::nullopt;
  }
  return Bind(*config, reader.GetTensors(), reader.GetMappedRegions(),
              error_msg);
}

}  // namespace gufo::models::laguna
