// Model-free Laguna DFlash drafter checks: the config parser against a
// synthetic header carrying the real drafter's metadata, and the weight index
// over a synthetic 76-tensor list (positive and negative). Runs in the hosted
// suite; the real-artifact check is laguna.dflash.
#include <iostream>
#include <string>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/laguna/config.hpp"
#include "src/models/laguna/dflash.hpp"
#include "tests/models/laguna/support.hpp"

namespace laguna = gufo::models::laguna;
namespace core = gufo::core;
namespace test = gufo::testing::laguna;
using test::Require;

namespace {

/// Metadata of Laguna-S-2.1-DFlash-Q4_K as the real file records it, with a
/// placeholder vocabulary of the right length.
test::Metadata ValidDFlashMetadata() {
  test::Metadata m;
  m["general.architecture"] = std::string{"dflash"};
  const auto u32 = [&](const char* key, std::uint32_t value) {
    m[std::string{"dflash."} + key] = value;
  };
  u32("block_count", 6);
  u32("context_length", 1048576);
  u32("embedding_length", 3072);
  u32("feed_forward_length", 12288);
  u32("attention.head_count", 72);
  u32("attention.head_count_kv", 8);
  u32("attention.key_length", 128);
  u32("attention.value_length", 128);
  u32("attention.sliding_window", 512);
  u32("rope.dimension_count", 128);
  u32("block_size", 16);
  m["dflash.attention.layer_norm_rms_epsilon"] = 1e-6F;
  m["dflash.rope.freq_base"] = 500000.0F;
  m["dflash.rope.scaling.type"] = std::string{"none"};
  m["dflash.decoder_arch"] = std::string{"laguna"};
  m["dflash.target_layers"] = std::vector<std::uint32_t>{2, 11, 20, 30, 39, 48};
  m["dflash.attention.sliding_window_pattern"] =
      std::vector<std::uint32_t>{1, 1, 1, 1, 1, 1};
  m["tokenizer.ggml.mask_token_id"] = std::uint32_t{12};
  m["tokenizer.ggml.tokens"] = std::vector<std::string>(100352, "t");
  return m;
}

std::optional<laguna::DFlashConfig> ParseConfig(const test::Metadata& fields,
                                                std::string* error) {
  std::vector<std::uint8_t> storage;
  const auto reader = test::OpenSynthetic(fields, &storage);
  return laguna::DFlashConfig::FromGguf(*reader, error);
}

void CheckConfig() {
  const auto valid = ValidDFlashMetadata();
  std::string error;
  const auto parsed = ParseConfig(valid, &error);
  Require(parsed.has_value(), "valid synthetic metadata rejected: " + error);
  const laguna::DFlashConfig& c = *parsed;
  Require(c.num_layers == 6 && c.hidden_size == 3072 && c.ffn_size == 12288 &&
              c.vocab_size == 100352,
          "dimensions");
  Require(c.num_heads == 72 && c.num_kv_heads == 8 && c.head_dim == 128 &&
              c.gqa_group() == 9,
          "attention geometry");
  Require(c.q_dim() == 9216 && c.kv_dim() == 1024, "projection widths");
  Require(c.sliding_window == 512 && c.rotary_dim == 128 &&
              c.rope_theta == 500000.0F,
          "window and rope");
  Require(c.block_size == 16 && c.max_draft_tokens() == 15, "block");
  Require(c.mask_token_id == 12, "mask token");
  Require(c.tap_count() == 6 && c.fused_input_width() == 18432, "taps");
  // GGUF numbering is layer-input; the tapped layer outputs are one less.
  const std::vector<std::uint32_t> outputs{1, 10, 19, 29, 38, 47};
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    Require(c.tap_layer(i) == outputs[i], "tap layer output index");
  }

  const auto expect_fail = [&](test::Metadata m, const std::string& needle,
                               const char* what) {
    error.clear();
    Require(!ParseConfig(m, &error), std::string("accepted ") + what);
    Require(
        error.find(needle) != std::string::npos,
        std::string(what) + ": message '" + error + "' lacks '" + needle + "'");
  };
  {
    auto m = valid;
    m["general.architecture"] = std::string{"laguna"};
    expect_fail(m, "not dflash", "target architecture");
  }
  {
    auto m = valid;
    m.erase("dflash.decoder_arch");
    expect_fail(m, "decoder_arch", "missing decoder_arch");
    m["dflash.decoder_arch"] = std::string{"qwen3"};
    expect_fail(m, "decoder_arch", "foreign decoder_arch");
  }
  for (const char* key :
       {"dflash.block_count", "dflash.block_size", "dflash.target_layers",
        "dflash.rope.freq_base", "tokenizer.ggml.mask_token_id",
        "tokenizer.ggml.tokens", "dflash.attention.sliding_window_pattern",
        "dflash.attention.sliding_window"}) {
    auto m = valid;
    m.erase(key);
    expect_fail(m, "", key);
  }
  {
    auto m = valid;
    m["dflash.rope.scaling.type"] = std::string{"yarn"};
    expect_fail(m, "rope.scaling.type", "scaled rope");
  }
  {
    auto m = valid;
    m["dflash.attention.sliding_window_pattern"] =
        std::vector<std::uint32_t>{1, 1, 1, 1, 1, 0};
    expect_fail(m, "sliding_window_pattern", "a full-attention layer");
    m["dflash.attention.sliding_window_pattern"] =
        std::vector<std::uint32_t>{1, 1, 1};
    expect_fail(m, "sliding_window_pattern", "short pattern");
  }
  {
    auto m = valid;
    m["dflash.target_layers"] = std::vector<std::uint32_t>{2, 11, 11, 30};
    expect_fail(m, "strictly increasing", "repeated tap");
    m["dflash.target_layers"] = std::vector<std::uint32_t>{0, 11};
    expect_fail(m, "strictly increasing", "zero tap");
    m["dflash.target_layers"] = std::vector<std::uint32_t>{};
    expect_fail(m, "target_layers", "no taps");
  }
  {
    auto m = valid;
    m["dflash.block_size"] = std::uint32_t{1};
    expect_fail(m, "invalid shape", "one-row block");
    m["dflash.block_size"] = std::uint32_t{8};
    expect_fail(m, "unsupported", "other block size");
    m = valid;
    m["dflash.attention.sliding_window"] = std::uint32_t{8};
    expect_fail(m, "smaller than the block", "window below block");
    m = valid;
    m["dflash.attention.head_count"] = std::uint32_t{70};
    expect_fail(m, "invalid shape", "heads not a multiple of KV heads");
    m = valid;
    m["tokenizer.ggml.mask_token_id"] = std::uint32_t{100352};
    expect_fail(m, "invalid shape", "mask outside the vocabulary");
    m = valid;
    m["tokenizer.ggml.tokens"] = std::vector<std::string>(1000, "t");
    expect_fail(m, "unsupported", "a vocabulary outside the profile");
  }
}

void CheckAgainstTarget(const laguna::DFlashConfig& d) {
  std::vector<std::uint8_t> storage;
  const auto reader =
      test::OpenSynthetic(test::ValidLagunaMetadata(), &storage);
  std::string error;
  const auto target = laguna::Config::FromGguf(*reader, &error);
  Require(target.has_value(), "synthetic target config: " + error);
  Require(d.ValidateAgainstTarget(*target, &error),
          "matching target rejected: " + error);
  // The last tap is the post-last-layer stream, so it equals the layer count.
  Require(d.target_layers.back() == target->num_layers, "last tap");
  Require(d.tap_layer(d.tap_count() - 1) == target->num_layers - 1,
          "last tap is the output of the last layer");

  auto shorter = *target;
  shorter.num_layers = 47;
  Require(!d.ValidateAgainstTarget(shorter, &error) &&
              error.find("post-last-layer") != std::string::npos,
          "last tap beyond a shorter target");
  auto wider = *target;
  wider.hidden_size = 2048;
  Require(!d.ValidateAgainstTarget(wider, &error) &&
              error.find("hidden size") != std::string::npos,
          "hidden size mismatch");
  auto vocab = *target;
  vocab.vocab_size = 100000;
  Require(!d.ValidateAgainstTarget(vocab, &error) &&
              error.find("vocabulary") != std::string::npos,
          "vocabulary mismatch");
}

struct Layout {
  std::vector<std::string> names;  // stable storage for the views
  std::vector<core::GgufTensorInfo> tensors;
  Layout() { names.reserve(256); }
  void Add(std::string name, std::vector<std::uint64_t> dims,
           core::GgmlType type) {
    names.push_back(std::move(name));
    core::GgufTensorInfo info;
    info.name = names.back();
    info.dimensions = std::move(dims);
    info.type = type;
    info.size_bytes = gufo::quant::EncodedSizeBytes(type, info.ElementCount());
    Require(info.size_bytes != 0,
            "synthetic tensor has no size: " + names.back());
    tensors.push_back(std::move(info));
  }
};

/// The real drafter's tensors: Q4_K projections, Q6_K for v/down of layers 2
/// and 5, F32 norms.
Layout BuildLayout(const laguna::DFlashConfig& c) {
  using T = core::GgmlType;
  Layout l;
  const std::uint64_t H = c.hidden_size;
  l.Add("enc.aux_norm.weight", {H, c.tap_count()}, T::kF32);
  l.Add("enc.output_norm.weight", {H}, T::kF32);
  l.Add("fc.weight", {c.fused_input_width(), H}, T::kQ4_K);
  l.Add("output_norm.weight", {H}, T::kF32);
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    const T vd = (il == 2 || il == 5) ? T::kQ6_K : T::kQ4_K;
    l.Add(p + "attn_gate.weight", {H, c.num_heads}, T::kQ4_K);
    l.Add(p + "attn_k.weight", {H, c.kv_dim()}, T::kQ4_K);
    l.Add(p + "attn_k_norm.weight", {c.head_dim}, T::kF32);
    l.Add(p + "attn_norm.weight", {H}, T::kF32);
    l.Add(p + "attn_output.weight", {c.q_dim(), H}, T::kQ4_K);
    l.Add(p + "attn_q.weight", {H, c.q_dim()}, T::kQ4_K);
    l.Add(p + "attn_q_norm.weight", {c.head_dim}, T::kF32);
    l.Add(p + "attn_v.weight", {H, c.kv_dim()}, vd);
    l.Add(p + "ffn_down.weight", {c.ffn_size, H}, vd);
    l.Add(p + "ffn_gate.weight", {H, c.ffn_size}, T::kQ4_K);
    l.Add(p + "ffn_norm.weight", {H}, T::kF32);
    l.Add(p + "ffn_up.weight", {H, c.ffn_size}, T::kQ4_K);
  }
  return l;
}

std::optional<laguna::DFlashWeights> Bind(const laguna::DFlashConfig& c,
                                          const Layout& l, std::string* error) {
  return laguna::DFlashWeights::Bind(c, l.tensors, {}, error);
}

std::size_t IndexOf(const Layout& l, std::string_view name) {
  for (std::size_t i = 0; i < l.tensors.size(); ++i) {
    if (l.tensors[i].name == name) {
      return i;
    }
  }
  throw std::runtime_error("no such synthetic tensor: " + std::string(name));
}

void CheckWeights(const laguna::DFlashConfig& c) {
  const Layout layout = BuildLayout(c);
  Require(layout.tensors.size() == 76, "synthetic layout has 76 tensors");
  Require(laguna::DFlashWeights::ExpectedTensorCount(c) == 76,
          "expected tensor count");
  std::string error;
  const auto weights = Bind(c, layout, &error);
  Require(weights.has_value(), "valid layout rejected: " + error);
  Require(weights->tensor_count == 76 && weights->layers.size() == 6, "bound");
  Require(weights->fc.cols == 18432 && weights->fc.rows == 3072, "fc shape");
  Require(weights->aux_norm.cols == 3072 && weights->aux_norm.rows == 6,
          "aux_norm shape");
  Require(weights->layers[2].attn_v.type == core::GgmlType::kQ6_K &&
              weights->layers[3].attn_v.type == core::GgmlType::kQ4_K,
          "per-layer value projection types");
  Require(weights->layers[0].attn_gate.rows == 72, "per-head gate");

  const auto expect_fail = [&](const Layout& broken, const std::string& needle,
                               const char* what) {
    error.clear();
    Require(!Bind(c, broken, &error), std::string("accepted ") + what);
    Require(
        error.find(needle) != std::string::npos,
        std::string(what) + ": message '" + error + "' lacks '" + needle + "'");
  };
  {  // Missing tensor, one of every kind.
    for (const char* name :
         {"enc.aux_norm.weight", "enc.output_norm.weight", "fc.weight",
          "output_norm.weight", "blk.3.attn_gate.weight", "blk.3.attn_k.weight",
          "blk.3.attn_k_norm.weight", "blk.3.attn_norm.weight",
          "blk.3.attn_output.weight", "blk.3.attn_q.weight",
          "blk.3.attn_q_norm.weight", "blk.3.attn_v.weight",
          "blk.3.ffn_down.weight", "blk.3.ffn_gate.weight",
          "blk.3.ffn_norm.weight", "blk.3.ffn_up.weight"}) {
      Layout broken = layout;
      broken.tensors.erase(broken.tensors.begin() + IndexOf(broken, name));
      expect_fail(broken, std::string("missing tensor ") + name, name);
    }
  }
  {  // The drafter shares the target's head and embeddings: its own copies,
     // like a seventh layer, are leftovers.
    Layout broken = layout;
    broken.Add("output.weight", {3072, 100352}, core::GgmlType::kQ6_K);
    expect_fail(broken, "output.weight", "own output head");
    Layout embd = layout;
    embd.Add("token_embd.weight", {3072, 100352}, core::GgmlType::kQ6_K);
    expect_fail(embd, "token_embd.weight", "own embeddings");
    Layout seventh = layout;
    seventh.Add("blk.6.attn_norm.weight", {3072}, core::GgmlType::kF32);
    expect_fail(seventh, "blk.6.attn_norm.weight", "seventh layer");
    Layout moe = layout;
    moe.Add("blk.1.ffn_gate_inp.weight", {3072, 256}, core::GgmlType::kF32);
    expect_fail(moe, "blk.1.ffn_gate_inp.weight", "router in a dense drafter");
  }
  {  // Duplicate name.
    Layout broken = layout;
    broken.tensors.push_back(broken.tensors[IndexOf(broken, "fc.weight")]);
    expect_fail(broken, "duplicate tensor fc.weight", "duplicate");
  }
  {  // Wrong shapes and formats.
    Layout broken = layout;
    broken.tensors[IndexOf(broken, "fc.weight")].dimensions = {3072, 3072};
    expect_fail(broken, "fc.weight has shape", "single-tap fc");
    broken = layout;
    broken.tensors[IndexOf(broken, "enc.aux_norm.weight")].dimensions = {3072 *
                                                                         6};
    expect_fail(broken, "enc.aux_norm.weight has shape", "flattened aux norm");
    broken = layout;
    broken.tensors[IndexOf(broken, "blk.1.attn_gate.weight")].dimensions = {
        3072, 72 * 128};
    expect_fail(broken, "blk.1.attn_gate.weight has shape", "per-element gate");
    broken = layout;
    broken.tensors[IndexOf(broken, "blk.4.attn_q.weight")].dimensions = {
        3072, 6144};  // the target's full-layer width
    expect_fail(broken, "blk.4.attn_q.weight has shape", "48-head query");
    broken = layout;
    broken.tensors[IndexOf(broken, "blk.0.attn_norm.weight")].type =
        core::GgmlType::kQ8_0;
    expect_fail(broken, "blk.0.attn_norm.weight has unsupported format",
                "quantized norm");
    broken = layout;
    auto& norm = broken.tensors[IndexOf(broken, "enc.aux_norm.weight")];
    norm.type = core::GgmlType::kF16;
    norm.size_bytes /= 2;
    expect_fail(broken, "enc.aux_norm.weight has unsupported format",
                "F16 aux norm");
    broken = layout;
    broken.tensors[IndexOf(broken, "blk.2.ffn_up.weight")].size_bytes -= 1;
    expect_fail(broken, "blk.2.ffn_up.weight records", "wrong byte count");
  }
}

}  // namespace

int main() {
  try {
    CheckConfig();
    std::string error;
    const auto config = ParseConfig(ValidDFlashMetadata(), &error);
    Require(config.has_value(), "config: " + error);
    CheckAgainstTarget(*config);
    CheckWeights(*config);
    std::cout << "PASS laguna.dflash_contract\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL laguna.dflash_contract: " << e.what() << '\n';
    return 1;
  }
}
