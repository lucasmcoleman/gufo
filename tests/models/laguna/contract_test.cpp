// Model-free Laguna checks: config validation against a synthetic header,
// weight-index binding over a synthetic 814-tensor list (positive and
// negative), and the pre-tokenizer split. Nothing here needs the GGUF, so this
// runs in the hosted suite; the real-artifact checks are laguna.config,
// laguna.weights and laguna.tokenizer_golden.
#include <deque>
#include <iostream>
#include <string>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/laguna/config.hpp"
#include "src/models/laguna/tokenizer.hpp"
#include "src/models/laguna/weights.hpp"
#include "tests/models/laguna/support.hpp"

namespace laguna = gufo::models::laguna;
namespace core = gufo::core;
namespace test = gufo::testing::laguna;
using test::Require;

namespace {

std::optional<laguna::Config> ParseConfig(const test::Metadata& fields,
                                          std::string* error = nullptr) {
  std::vector<std::uint8_t> storage;
  const auto reader = test::OpenSynthetic(fields, &storage);
  std::string local;
  const auto config = laguna::Config::FromGguf(*reader, error ? error : &local);
  return config;
}

void CheckConfig() {
  const auto valid = test::ValidLagunaMetadata();
  std::string error;
  const auto parsed = ParseConfig(valid, &error);
  Require(parsed.has_value(), "valid synthetic metadata rejected: " + error);
  const laguna::Config& c = *parsed;
  Require(c.is_full_attention(0) && c.is_full_attention(44) &&
              !c.is_full_attention(1) && !c.is_full_attention(47),
          "layer pattern");
  Require(c.n_head(0) == 48 && c.n_head(1) == 72 && c.n_head(4) == 48,
          "n_head");
  Require(c.rotary_dim(0) == 64 && c.rotary_dim(3) == 128, "rotary_dim");
  Require(c.rope_theta(8) == 500000.0F && c.rope_theta(9) == 10000.0F,
          "rope_theta");
  Require(c.gqa_group(0) == 6 && c.gqa_group(1) == 9, "gqa group");
  Require(c.kv_dim() == 1024 && c.q_dim(0) == 6144 && c.q_dim(1) == 9216,
          "projection widths");
  Require(c.YarnMscale() > 1.3465F && c.YarnMscale() < 1.3467F,
          "yarn magnitude scale");
  Require(c.shared_expert_count == 1, "shared expert count defaults to 1");

  // A uniform head count may be written as a scalar; it must then fail the
  // profile (full and sliding layers differ) rather than parse silently.
  {
    auto m = valid;
    m["laguna.attention.head_count"] = std::uint32_t{48};
    Require(!ParseConfig(m, &error), "uniform head count accepted");
  }

  // Every required key, removed on its own, must fail with a message.
  std::size_t removed = 0;
  for (const auto& [key, value] : valid) {
    if (key == "general.architecture" || key == "laguna.rope.scaling.type" ||
        key == "general.architecture") {
      continue;
    }
    auto m = valid;
    m.erase(key);
    error.clear();
    Require(!ParseConfig(m, &error), "config accepted without " + key);
    Require(!error.empty(), "no error message without " + key);
    ++removed;
  }
  Require(removed >= 28, "expected to exercise every laguna.* key");
  {
    auto m = valid;
    m.erase("laguna.rope.scaling.type");
    Require(!ParseConfig(m), "config accepted without rope scaling type");
    m["laguna.rope.scaling.type"] = std::string{"linear"};
    Require(!ParseConfig(m), "non-yarn scaling accepted");
    m = valid;
    m["general.architecture"] = std::string{"qwen35"};
    Require(!ParseConfig(m), "foreign architecture accepted");
  }

  // Wrong values: each mutation must be refused.
  const auto mutate = [&](const char* key, test::MetaValue value,
                          const char* what) {
    auto m = valid;
    m[key] = std::move(value);
    error.clear();
    Require(!ParseConfig(m, &error), std::string("accepted ") + what);
    Require(!error.empty(), std::string("no message for ") + what);
  };
  mutate("laguna.block_count", std::uint32_t{40}, "40 layers");
  mutate("laguna.block_count", std::uint32_t{0}, "0 layers");
  mutate("laguna.embedding_length", std::uint32_t{2048}, "hidden 2048");
  mutate("laguna.attention.head_count_kv", std::uint32_t{4}, "4 kv heads");
  mutate("laguna.attention.head_count_kv", std::uint32_t{7}, "7 kv heads");
  mutate("laguna.attention.key_length", std::uint32_t{64}, "key length 64");
  mutate("laguna.attention.value_length", std::uint32_t{64},
         "value length differing from key length");
  mutate("laguna.attention.sliding_window", std::uint32_t{0}, "no window");
  mutate("laguna.attention.sliding_window", std::uint32_t{1024}, "window 1024");
  mutate("laguna.rope.dimension_count", std::uint32_t{63}, "odd rotary dim");
  mutate("laguna.rope.dimension_count", std::uint32_t{256}, "rotary > head");
  mutate("laguna.rope.dimension_count_swa", std::uint32_t{64}, "swa rotary 64");
  mutate("laguna.rope.freq_base", 0.0F, "zero theta");
  mutate("laguna.rope.freq_base_swa", -1.0F, "negative theta");
  mutate("laguna.rope.scaling.factor", 0.5F, "yarn factor below 1");
  mutate("laguna.rope.scaling.yarn_beta_fast", 0.5F, "beta_fast < beta_slow");
  mutate("laguna.attention.layer_norm_rms_epsilon", 0.0F, "zero eps");
  mutate("laguna.expert_count", std::uint32_t{128}, "128 experts");
  mutate("laguna.expert_used_count", std::uint32_t{300}, "used > count");
  mutate("laguna.expert_used_count", std::uint32_t{8}, "8 used experts");
  mutate("laguna.expert_gating_func", std::uint32_t{1}, "softmax gating");
  mutate("laguna.expert_weights_norm", false, "no weight normalization");
  mutate("laguna.expert_weights_scale", 0.0F, "zero weights scale");
  mutate("laguna.leading_dense_block_count", std::uint32_t{48}, "all dense");
  mutate("laguna.leading_dense_block_count", std::uint32_t{3}, "3 dense");
  mutate("laguna.expert_shared_count", std::uint32_t{2}, "2 shared experts");
  mutate("laguna.vocab_size", std::uint32_t{1}, "tiny vocab");
  mutate("laguna.feed_forward_length", std::uint32_t{0}, "zero dense ff");
  {
    auto heads = std::get<std::vector<std::uint32_t>>(
        valid.at("laguna.attention.head_count"));
    heads.pop_back();
    mutate("laguna.attention.head_count", heads, "short head_count array");
    heads = std::get<std::vector<std::uint32_t>>(
        valid.at("laguna.attention.head_count"));
    heads[5] = 70;  // not a multiple of 8 KV heads
    mutate("laguna.attention.head_count", heads, "ragged GQA group");
    heads[5] = 64;  // multiple of 8, but breaks the 72-head sliding profile
    mutate("laguna.attention.head_count", heads, "inconsistent sliding heads");
  }
  {
    // A vocabulary that disagrees with laguna.vocab_size.
    auto m = valid;
    m["tokenizer.ggml.tokens"] = std::vector<std::string>{"a", "b"};
    Require(!ParseConfig(m), "vocabulary length mismatch accepted");
  }
}

// Synthetic tensor list matching the real artifact layout and quantization.
struct Layout {
  std::deque<std::string> names;
  std::vector<core::GgufTensorInfo> tensors;

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

Layout BuildLayout(const laguna::Config& c) {
  using T = core::GgmlType;
  Layout l;
  const std::uint64_t H = c.hidden_size;
  l.Add("token_embd.weight", {H, c.vocab_size}, T::kQ6_K);
  l.Add("output_norm.weight", {H}, T::kF32);
  l.Add("output.weight", {H, c.vocab_size}, T::kQ6_K);
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    const std::uint64_t q = c.q_dim(il);
    l.Add(p + "attn_norm.weight", {H}, T::kF32);
    l.Add(p + "attn_q.weight", {H, q}, T::kQ6_K);
    l.Add(p + "attn_k.weight", {H, c.kv_dim()}, T::kQ6_K);
    l.Add(p + "attn_v.weight", {H, c.kv_dim()}, T::kQ6_K);
    l.Add(p + "attn_output.weight", {q, H}, T::kQ6_K);
    l.Add(p + "attn_q_norm.weight", {c.head_dim}, T::kF32);
    l.Add(p + "attn_k_norm.weight", {c.head_dim}, T::kF32);
    l.Add(p + "attn_gate.weight", {H, c.n_head(il)}, T::kF32);
    l.Add(p + "ffn_norm.weight", {H}, T::kF32);
    if (il < c.leading_dense_layers) {
      l.Add(p + "ffn_gate.weight", {H, c.ffn_size}, T::kQ8_0);
      l.Add(p + "ffn_up.weight", {H, c.ffn_size}, T::kQ8_0);
      l.Add(p + "ffn_down.weight", {c.ffn_size, H}, T::kQ8_0);
      continue;
    }
    const T exps = (il <= 4 || il >= 43)   ? T::kQ6_K
                   : (il <= 9 || il >= 38) ? T::kQ5_K
                                           : T::kIQ4_XS;
    l.Add(p + "ffn_gate_inp.weight", {H, c.num_experts}, T::kF32);
    l.Add(p + "exp_probs_b.bias", {c.num_experts}, T::kF32);
    l.Add(p + "ffn_gate_exps.weight", {H, c.expert_ff, c.num_experts}, exps);
    l.Add(p + "ffn_up_exps.weight", {H, c.expert_ff, c.num_experts}, exps);
    l.Add(p + "ffn_down_exps.weight", {c.expert_ff, H, c.num_experts}, exps);
    l.Add(p + "ffn_gate_shexp.weight", {H, c.shared_expert_ff}, T::kQ8_0);
    l.Add(p + "ffn_up_shexp.weight", {H, c.shared_expert_ff}, T::kQ8_0);
    l.Add(p + "ffn_down_shexp.weight", {c.shared_expert_ff, H}, T::kQ8_0);
  }
  return l;
}

std::optional<laguna::ModelWeights> Bind(const laguna::Config& c,
                                         const Layout& l,
                                         std::string* error = nullptr) {
  std::string local;
  return laguna::ModelWeights::Bind(c, l.tensors, {}, error ? error : &local);
}

std::size_t IndexOf(const Layout& l, std::string_view name) {
  for (std::size_t i = 0; i < l.tensors.size(); ++i) {
    if (l.tensors[i].name == name) {
      return i;
    }
  }
  throw std::runtime_error("no such synthetic tensor: " + std::string(name));
}

void CheckWeights() {
  const auto config = ParseConfig(test::ValidLagunaMetadata());
  Require(config.has_value(), "synthetic config");
  const laguna::Config& c = *config;
  const Layout layout = BuildLayout(c);
  Require(layout.tensors.size() == 814, "synthetic layout has 814 tensors");
  Require(laguna::ModelWeights::ExpectedTensorCount(c) == 814,
          "expected tensor count");

  std::string error;
  const auto weights = Bind(c, layout, &error);
  Require(weights.has_value(), "valid layout rejected: " + error);
  Require(weights->tensor_count == 814, "bound count");
  Require(weights->layers[0].ffn_gate.rows == 12288 &&
              weights->layers[0].ffn_gate_inp.empty(),
          "dense layer 0");
  Require(weights->layers[1].ffn_gate_inp.rows == 256 &&
              weights->layers[1].ffn_gate.empty(),
          "MoE layer 1");
  Require(weights->layers[1].ffn_exp_probs_b.name == "blk.1.exp_probs_b.bias",
          "bias tensor name");
  Require(weights->layers[12].ffn_up_exps.type == core::GgmlType::kIQ4_XS,
          "IQ4_XS experts at layer 12");
  Require(weights->layers[12].ffn_up_exps.Expert(2) ==
              static_cast<const std::uint8_t*>(
                  weights->layers[12].ffn_up_exps.data) +
                  2 * weights->layers[12].ffn_up_exps.rows *
                      weights->layers[12].ffn_up_exps.RowBytes(),
          "expert stride");

  const auto expect_fail = [&](const Layout& broken, const std::string& needle,
                               const char* what) {
    error.clear();
    Require(!Bind(c, broken, &error), std::string("accepted ") + what);
    Require(
        error.find(needle) != std::string::npos,
        std::string(what) + ": message '" + error + "' lacks '" + needle + "'");
  };

  {  // Missing tensor, one of every kind.
    for (const char* name : {"token_embd.weight",
                             "output_norm.weight",
                             "output.weight",
                             "blk.7.attn_norm.weight",
                             "blk.7.attn_q.weight",
                             "blk.7.attn_k.weight",
                             "blk.7.attn_v.weight",
                             "blk.7.attn_output.weight",
                             "blk.7.attn_q_norm.weight",
                             "blk.7.attn_k_norm.weight",
                             "blk.7.attn_gate.weight",
                             "blk.7.ffn_norm.weight",
                             "blk.7.ffn_gate_inp.weight",
                             "blk.7.exp_probs_b.bias",
                             "blk.7.ffn_gate_exps.weight",
                             "blk.7.ffn_up_exps.weight",
                             "blk.7.ffn_down_exps.weight",
                             "blk.7.ffn_gate_shexp.weight",
                             "blk.7.ffn_up_shexp.weight",
                             "blk.7.ffn_down_shexp.weight",
                             "blk.0.ffn_gate.weight",
                             "blk.0.ffn_up.weight",
                             "blk.0.ffn_down.weight"}) {
      Layout broken = layout;
      broken.tensors.erase(broken.tensors.begin() + IndexOf(broken, name));
      expect_fail(broken, std::string("missing tensor ") + name, name);
    }
  }
  {  // The llama.cpp enum name is not the GGUF name: ffn_exp_probs_b is
     // unknown.
    Layout broken = layout;
    broken.names.push_back("blk.7.ffn_exp_probs_b.bias");
    broken.tensors[IndexOf(broken, "blk.7.exp_probs_b.bias")].name =
        broken.names.back();
    expect_fail(broken, "missing tensor blk.7.exp_probs_b.bias",
                "misnamed bias");
  }
  {  // Unknown leftovers.
    Layout broken = layout;
    broken.Add("blk.3.attn_sinks.weight", {72}, core::GgmlType::kF32);
    expect_fail(broken, "not part of the Laguna layout", "extra tensor");
    expect_fail(broken, "blk.3.attn_sinks.weight", "extra tensor name");
    Layout nextn = layout;
    nextn.Add("blk.48.attn_norm.weight", {3072}, core::GgmlType::kF32);
    expect_fail(nextn, "blk.48.attn_norm.weight", "tensor past the last layer");
  }
  {  // Dense tensors on a MoE layer are leftovers too.
    Layout broken = layout;
    broken.Add("blk.5.ffn_gate.weight", {3072, 12288}, core::GgmlType::kQ8_0);
    expect_fail(broken, "blk.5.ffn_gate.weight", "dense tensor on MoE layer");
  }
  {  // Duplicate name.
    Layout broken = layout;
    broken.tensors.push_back(broken.tensors[IndexOf(broken, "output.weight")]);
    expect_fail(broken, "duplicate tensor output.weight", "duplicate");
  }
  {  // Wrong shapes.
    Layout broken = layout;
    broken.tensors[IndexOf(broken, "blk.1.attn_q.weight")].dimensions = {
        3072, 6144};  // sliding layer needs 9216 rows
    expect_fail(broken, "blk.1.attn_q.weight has shape", "full-width q on SWA");
    broken = layout;
    broken.tensors[IndexOf(broken, "blk.2.attn_gate.weight")].dimensions = {
        3072, 72 * 128};  // per-element gate
    expect_fail(broken, "blk.2.attn_gate.weight has shape", "per-element gate");
    broken = layout;
    broken.tensors[IndexOf(broken, "blk.9.ffn_up_exps.weight")].dimensions = {
        3072, 1024, 128};
    expect_fail(broken, "blk.9.ffn_up_exps.weight has shape", "128 experts");
    broken = layout;
    broken.tensors[IndexOf(broken, "blk.9.ffn_down_exps.weight")].dimensions = {
        3072, 1024, 256};  // transposed
    expect_fail(broken, "blk.9.ffn_down_exps.weight has shape", "transposed");
    broken = layout;
    broken.tensors[IndexOf(broken, "blk.9.exp_probs_b.bias")].dimensions = {256,
                                                                            1};
    expect_fail(broken, "blk.9.exp_probs_b.bias has shape", "2D bias");
    broken = layout;
    broken.tensors[IndexOf(broken, "token_embd.weight")].dimensions = {3072,
                                                                       100351};
    expect_fail(broken, "token_embd.weight has shape", "short vocabulary");
  }
  {  // Wrong types.
    Layout broken = layout;
    auto& norm = broken.tensors[IndexOf(broken, "blk.4.attn_q_norm.weight")];
    norm.type = core::GgmlType::kF16;
    norm.size_bytes = 128 * 2;
    expect_fail(broken, "unsupported format F16", "F16 norm");
    broken = layout;
    auto& gate = broken.tensors[IndexOf(broken, "blk.4.ffn_gate_inp.weight")];
    gate.type = core::GgmlType::kQ8_0;
    gate.size_bytes =
        gufo::quant::EncodedSizeBytes(gate.type, gate.ElementCount());
    expect_fail(broken, "unsupported format Q8_0", "quantized router");
    broken = layout;
    auto& exps = broken.tensors[IndexOf(broken, "blk.4.ffn_up_exps.weight")];
    exps.type = core::GgmlType::kQ2_K;
    exps.size_bytes = 1;
    expect_fail(broken, "unsupported format Q2_K", "unsupported expert type");
    broken = layout;
    broken.tensors[IndexOf(broken, "blk.4.attn_k.weight")].size_bytes += 1;
    expect_fail(broken, "blk.4.attn_k.weight records", "wrong recorded size");
  }
}

void ExpectSplit(std::string_view text, std::vector<std::string> want) {
  const auto got = laguna::Tokenizer::PreTokenize(text);
  std::string joined;
  for (const auto piece : got) {
    joined += piece;
  }
  Require(joined == text, "pieces do not concatenate to the input");
  const bool same = got.size() == want.size() &&
                    std::equal(got.begin(), got.end(), want.begin());
  if (!same) {
    std::string message = "split of '" + std::string(text) + "' gave:";
    for (const auto piece : got) {
      message += " [" + std::string(piece) + "]";
    }
    throw std::runtime_error(message);
  }
}

// Expected splits come from the Laguna regexes run through the Python `regex`
// module (stage one `[^\n]+|[\n]+`, stage two the Qwen2-style expression).
void CheckPreTokenizer() {
  ExpectSplit("Hello, world!", {"Hello", ",", " world", "!"});
  ExpectSplit("don't I'm WE'LL", {"don", "'t", " I", "'m", " WE", "'LL"});
  ExpectSplit("x'S y'Re z've", {"x", "'S", " y", "'Re", " z", "'ve"});
  ExpectSplit("2026", {"2", "0", "2", "6"});
  ExpectSplit("a  b", {"a", " ", " b"});
  ExpectSplit("x\n\ny", {"x", "\n\n", "y"});
  ExpectSplit("foo   \nbar", {"foo", "   ", "\n", "bar"});
  ExpectSplit("  \n  x", {"  ", "\n", " ", " x"});
  ExpectSplit("tab\there", {"tab", "\there"});
  ExpectSplit("a\r\nb", {"a", "\r", "\n", "b"});
  ExpectSplit("!!! ???", {"!!!", " ???"});
  ExpectSplit("x=1+2;", {"x", "=", "1", "+", "2", ";"});
  ExpectSplit("a 1", {"a", " ", "1"});
  ExpectSplit("hello ", {"hello", " "});
  ExpectSplit(" ", {" "});
  ExpectSplit("", {});
  ExpectSplit("'s", {"'s"});
  ExpectSplit("'", {"'"});
  ExpectSplit("a'", {"a", "'"});
  ExpectSplit("\xF0\x9F\x98\x80 hi", {"\xF0\x9F\x98\x80", " hi"});
  ExpectSplit("\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E",
              {"\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E"});
  ExpectSplit("caf\xC3\xA9", {"caf\xC3\xA9"});
  // A combining mark is not \p{L}: it splits off.
  ExpectSplit("e\xCC\x81", {"e", "\xCC\x81"});
  // Invalid UTF-8 still partitions the input exactly.
  ExpectSplit(
      "a\xFF"
      "b\xC3",
      {"a",
       "\xFF"
       "b",
       "\xC3"});
}

}  // namespace

int main() {
  try {
    CheckConfig();
    CheckWeights();
    CheckPreTokenizer();
    std::cout << "PASS laguna.contract\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL laguna.contract: " << e.what() << '\n';
    return 1;
  }
}
