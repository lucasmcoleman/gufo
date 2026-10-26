#include "src/models/laguna/dflash.hpp"

#include <cmath>
#include <string_view>
#include <utility>
#include <variant>

#include "src/models/laguna/tensor_binder.hpp"

namespace gufo::models::laguna {
namespace {

constexpr std::string_view kArchitecture = "dflash";
constexpr std::string_view kDecoderArch = "laguna";

/// First-failure metadata reader over the `dflash.` keys, as config.cpp's.
struct Reader {
  const core::GgufReader& gguf;
  std::string* error;
  bool ok{true};

  void Fail(std::string message) {
    if (ok && error != nullptr) {
      *error = std::move(message);
    }
    ok = false;
  }

  static std::string Key(std::string_view suffix) {
    return "dflash." + std::string(suffix);
  }

  std::uint32_t U32(std::string_view suffix) {
    const auto key = Key(suffix);
    const auto value = gguf.GetMetadataUint32(key);
    if (!value.has_value()) {
      Fail("missing or invalid GGUF integer " + key);
      return 0;
    }
    return *value;
  }

  float F32(std::string_view suffix) {
    const auto key = Key(suffix);
    const auto value = gguf.GetMetadataFloat32(key);
    if (!value.has_value() || !std::isfinite(*value)) {
      Fail("missing or invalid GGUF float " + key);
      return 0.0F;
    }
    return *value;
  }

  /// Integer array with every entry in [0, UINT32_MAX]; empty on failure.
  std::vector<std::uint32_t> U32Array(std::string_view suffix) {
    const auto key = Key(suffix);
    const auto* meta = gguf.FindMetadata(key);
    std::vector<std::uint32_t> out;
    if (meta == nullptr) {
      Fail("missing GGUF array " + key);
      return out;
    }
    const auto push = [&](std::uint64_t v) {
      if (v > UINT32_MAX) {
        Fail("GGUF array entry out of range in " + key);
        return;
      }
      out.push_back(static_cast<std::uint32_t>(v));
    };
    if (const auto* u = std::get_if<std::vector<std::uint64_t>>(&meta->value)) {
      for (const auto v : *u) {
        push(v);
      }
    } else if (const auto* s =
                   std::get_if<std::vector<std::int64_t>>(&meta->value)) {
      for (const auto v : *s) {
        if (v < 0) {
          Fail("negative GGUF array entry in " + key);
          break;
        }
        push(static_cast<std::uint64_t>(v));
      }
    } else {
      Fail("GGUF value must be an integer array: " + key);
    }
    return ok ? out : std::vector<std::uint32_t>{};
  }
};

}  // namespace

std::optional<DFlashConfig> DFlashConfig::FromGguf(const core::GgufReader& gguf,
                                                   std::string* error_msg) {
  Reader r{gguf, error_msg};
  if (gguf.GetMetadataString("general.architecture") != kArchitecture) {
    r.Fail("GGUF architecture is not dflash");
    return std::nullopt;
  }
  // The Laguna draft-layer contract (norm placement, gate, causal block) is
  // selected by this key in Poolside's llama.cpp; a drafter for another
  // decoder is a different program.
  if (gguf.GetMetadataString(Reader::Key("decoder_arch")) != kDecoderArch) {
    r.Fail("dflash.decoder_arch must be laguna");
    return std::nullopt;
  }

  DFlashConfig c;
  c.num_layers = r.U32("block_count");
  c.hidden_size = r.U32("embedding_length");
  c.ffn_size = r.U32("feed_forward_length");
  c.context_length = r.U32("context_length");
  c.rms_eps = r.F32("attention.layer_norm_rms_epsilon");
  c.num_heads = r.U32("attention.head_count");
  c.num_kv_heads = r.U32("attention.head_count_kv");
  c.head_dim = r.U32("attention.key_length");
  if (r.U32("attention.value_length") != c.head_dim) {
    r.Fail("attention.value_length must equal attention.key_length");
  }
  c.sliding_window = r.U32("attention.sliding_window");
  c.rope_theta = r.F32("rope.freq_base");
  c.rotary_dim = r.U32("rope.dimension_count");
  c.block_size = r.U32("block_size");
  c.target_layers = r.U32Array("target_layers");
  const auto pattern = r.U32Array("attention.sliding_window_pattern");
  if (gguf.GetMetadataString(Reader::Key("rope.scaling.type")) != "none") {
    r.Fail("dflash.rope.scaling.type must be none");
  }
  if (const auto mask = gguf.GetMetadataUint32("tokenizer.ggml.mask_token_id");
      mask.has_value()) {
    c.mask_token_id = *mask;
  } else {
    r.Fail("missing or invalid GGUF integer tokenizer.ggml.mask_token_id");
  }
  // The drafter carries the target's vocabulary and has no vocab_size key.
  if (const auto* tokens = gguf.FindMetadata("tokenizer.ggml.tokens");
      tokens != nullptr) {
    if (const auto* list =
            std::get_if<std::vector<std::string_view>>(&tokens->value)) {
      c.vocab_size = static_cast<std::uint32_t>(list->size());
    }
  }
  if (c.vocab_size == 0) {
    r.Fail("missing tokenizer.ggml.tokens (the drafter's vocabulary)");
  }
  if (!r.ok) {
    return std::nullopt;
  }

  if (c.num_layers == 0 || c.num_layers > 256 || c.hidden_size == 0 ||
      c.ffn_size == 0 || c.context_length == 0 || c.rms_eps <= 0 ||
      c.num_heads == 0 || c.num_kv_heads == 0 || c.head_dim == 0 ||
      c.num_heads % c.num_kv_heads != 0 || c.sliding_window == 0 ||
      c.rope_theta <= 0 || c.rotary_dim == 0 || c.rotary_dim % 2 != 0 ||
      c.rotary_dim > c.head_dim || c.block_size < 2 ||
      c.mask_token_id >= c.vocab_size) {
    r.Fail("dflash metadata describes an invalid shape");
    return std::nullopt;
  }
  // The window must hold the block, or the block's own rows could not all see
  // each other.
  if (c.sliding_window < c.block_size) {
    r.Fail("dflash.attention.sliding_window is smaller than the block");
    return std::nullopt;
  }

  // Per-layer attention kind: every drafter layer is sliding-window.
  if (pattern.size() != c.num_layers) {
    r.Fail("dflash.attention.sliding_window_pattern has " +
           std::to_string(pattern.size()) + " entries but block_count is " +
           std::to_string(c.num_layers));
    return std::nullopt;
  }
  for (const auto entry : pattern) {
    if (entry != 1) {
      r.Fail(
          "dflash.attention.sliding_window_pattern must mark every layer "
          "sliding-window");
      return std::nullopt;
    }
  }

  // Tap list: layer-input numbering, 1..target layers, strictly increasing.
  if (c.target_layers.empty() || c.target_layers.size() > 16) {
    r.Fail("dflash.target_layers must hold 1 to 16 entries");
    return std::nullopt;
  }
  for (std::size_t i = 0; i < c.target_layers.size(); ++i) {
    if (c.target_layers[i] == 0 ||
        (i > 0 && c.target_layers[i] <= c.target_layers[i - 1])) {
      r.Fail(
          "dflash.target_layers must be strictly increasing positive "
          "layer-input numbers");
      return std::nullopt;
    }
  }

  // Profile of this runtime: the Laguna-S-2.1 drafter.
  if (c.num_layers != 6 || c.hidden_size != 3072 || c.ffn_size != 12288 ||
      c.vocab_size != 100352 || c.num_heads != 72 || c.num_kv_heads != 8 ||
      c.head_dim != 128 || c.sliding_window != 512 || c.rotary_dim != 128 ||
      c.block_size != 16) {
    r.Fail("unsupported Laguna drafter geometry (only Laguna-S-2.1's)");
    return std::nullopt;
  }
  return c;
}

bool DFlashConfig::ValidateAgainstTarget(const Config& target,
                                         std::string* error_msg) const {
  const auto fail = [&](std::string message) {
    if (error_msg != nullptr) {
      *error_msg = std::move(message);
    }
    return false;
  };
  if (hidden_size != target.hidden_size) {
    return fail("drafter hidden size " + std::to_string(hidden_size) +
                " differs from the target's " +
                std::to_string(target.hidden_size));
  }
  if (vocab_size != target.vocab_size) {
    return fail("drafter vocabulary " + std::to_string(vocab_size) +
                " differs from the target's " +
                std::to_string(target.vocab_size));
  }
  if (target_layers.back() != target.num_layers) {
    return fail("the last tap (" + std::to_string(target_layers.back()) +
                ") must be the target's post-last-layer stream, layer count " +
                std::to_string(target.num_layers));
  }
  if (mask_token_id >= target.vocab_size) {
    return fail("mask token is outside the target vocabulary");
  }
  return true;
}

std::size_t DFlashWeights::ExpectedTensorCount(const DFlashConfig& config) {
  return 4 + 12 * static_cast<std::size_t>(config.num_layers);
}

std::optional<DFlashWeights> DFlashWeights::Bind(
    const DFlashConfig& config, std::span<const core::GgufTensorInfo> tensors,
    std::span<const core::GgufMappedRegion> regions, std::string* error_msg) {
  using detail::kF32Only;
  using detail::kMatrix;
  detail::TensorBinder b(tensors, regions, error_msg);
  DFlashWeights w;
  w.config = config;
  const std::uint64_t hidden = config.hidden_size;
  const std::uint64_t q_dim = config.q_dim();
  const std::uint64_t kv_dim = config.kv_dim();

  w.aux_norm =
      b.Get("enc.aux_norm.weight", hidden, config.tap_count(), 1, kF32Only);
  w.enc_output_norm = b.Get("enc.output_norm.weight", hidden, 1, 1, kF32Only);
  w.fc = b.Get("fc.weight", config.fused_input_width(), hidden, 1, kMatrix);
  w.output_norm = b.Get("output_norm.weight", hidden, 1, 1, kF32Only);
  w.layers.reserve(config.num_layers);
  for (std::uint32_t il = 0; il < config.num_layers && b.ok(); ++il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    DFlashLayerWeights l;
    l.attn_norm = b.Get(p + "attn_norm.weight", hidden, 1, 1, kF32Only);
    l.attn_q = b.Get(p + "attn_q.weight", hidden, q_dim, 1, kMatrix);
    l.attn_k = b.Get(p + "attn_k.weight", hidden, kv_dim, 1, kMatrix);
    l.attn_v = b.Get(p + "attn_v.weight", hidden, kv_dim, 1, kMatrix);
    l.attn_output = b.Get(p + "attn_output.weight", q_dim, hidden, 1, kMatrix);
    l.attn_q_norm =
        b.Get(p + "attn_q_norm.weight", config.head_dim, 1, 1, kF32Only);
    l.attn_k_norm =
        b.Get(p + "attn_k_norm.weight", config.head_dim, 1, 1, kF32Only);
    l.attn_gate =
        b.Get(p + "attn_gate.weight", hidden, config.num_heads, 1, kMatrix);
    l.ffn_norm = b.Get(p + "ffn_norm.weight", hidden, 1, 1, kF32Only);
    l.ffn_gate =
        b.Get(p + "ffn_gate.weight", hidden, config.ffn_size, 1, kMatrix);
    l.ffn_up = b.Get(p + "ffn_up.weight", hidden, config.ffn_size, 1, kMatrix);
    l.ffn_down =
        b.Get(p + "ffn_down.weight", config.ffn_size, hidden, 1, kMatrix);
    w.layers.push_back(std::move(l));
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

std::optional<DFlashWeights> DFlashWeights::Bind(const core::GgufReader& reader,
                                                 std::string* error_msg) {
  const auto config = DFlashConfig::FromGguf(reader, error_msg);
  if (!config.has_value()) {
    return std::nullopt;
  }
  return Bind(*config, reader.GetTensors(), reader.GetMappedRegions(),
              error_msg);
}

}  // namespace gufo::models::laguna
