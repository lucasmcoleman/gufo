// Config against the real Laguna-S-2.1 GGUF header. Only the header is read
// (the 74 GB file is memory-mapped). Without the file the test ends with the
// CTest skip status; nothing is verified in that case.
#include "src/models/laguna/config.hpp"

#include <cmath>
#include <iostream>
#include <string>

#include "tests/models/laguna/support.hpp"

namespace laguna = gufo::models::laguna;
using gufo::testing::laguna::Require;

int main() {
  try {
    const auto reader = gufo::testing::laguna::OpenModelOrSkip("laguna.config");
    std::string error;
    const auto parsed = laguna::Config::FromGguf(*reader, &error);
    Require(parsed.has_value(), "Config::FromGguf failed: " + error);
    const laguna::Config& c = *parsed;

    // Expected values written out independently of the parser.
    Require(c.num_layers == 48, "block_count");
    Require(c.hidden_size == 3072, "embedding_length");
    Require(c.ffn_size == 12288, "feed_forward_length");
    Require(c.vocab_size == 100352, "vocab_size");
    Require(c.context_length == 262144, "context_length");
    Require(std::abs(c.rms_eps - 1e-6F) < 1e-12F, "rms eps");
    Require(c.num_kv_heads == 8, "head_count_kv");
    Require(c.head_dim == 128, "key/value length");
    Require(c.sliding_window == 512, "sliding_window");
    Require(c.rope_theta_full == 500000.0F, "freq_base");
    Require(c.rope_theta_swa == 10000.0F, "freq_base_swa");
    Require(c.rotary_dim_full == 64, "rope dimension_count");
    Require(c.rotary_dim_swa == 128, "rope dimension_count_swa");
    Require(c.yarn.factor == 32.0F, "yarn factor");
    Require(c.yarn.original_context == 8192, "yarn original context");
    Require(c.yarn.beta_fast == 32.0F && c.yarn.beta_slow == 1.0F,
            "yarn betas");
    Require(c.yarn.yarn_attn_factor == 1.0F, "yarn_attn_factor");
    Require(c.rope_attn_factor == 1.0F, "rope attn_factor default");
    Require(c.num_experts == 256 && c.num_experts_used == 10, "expert counts");
    Require(c.expert_ff == 1024 && c.shared_expert_ff == 1024, "expert ff");
    Require(c.shared_expert_count == 1, "shared expert count");
    Require(c.expert_weights_norm, "expert_weights_norm");
    Require(c.expert_weights_scale == 2.5F, "expert_weights_scale");
    Require(c.leading_dense_layers == 1, "leading_dense_block_count");

    // Per-layer helpers across all 48 layers.
    std::uint32_t full = 0;
    for (std::uint32_t il = 0; il < 48; ++il) {
      const bool is_full = il % 4 == 0;
      Require(c.is_full_attention(il) == is_full, "layer pattern");
      Require(c.n_head(il) == (is_full ? 48U : 72U), "n_head");
      Require(c.gqa_group(il) == (is_full ? 6U : 9U), "gqa group");
      Require(c.rotary_dim(il) == (is_full ? 64U : 128U), "rotary_dim");
      Require(c.rope_theta(il) == (is_full ? 500000.0F : 10000.0F),
              "rope_theta");
      Require(c.uses_yarn(il) == is_full, "yarn only on full layers");
      Require(c.is_moe(il) == (il >= 1), "dense lead");
      full += is_full ? 1 : 0;
    }
    Require(full == 12 && c.full_layer_count() == 12, "12 full layers");
    Require(c.n_head(48 - 1) == 72, "last layer is sliding");

    // The same fields read straight from the file, to catch a parser that
    // agrees with its own constants but not with the artifact.
    Require(reader->GetMetadataString("tokenizer.ggml.pre") == "laguna",
            "tokenizer.ggml.pre");
    Require(reader->GetMetadataUint32("laguna.attention.head_count_kv") == 8U,
            "raw head_count_kv");
    Require(std::abs(c.YarnMscale() - (1.0F + 0.1F * std::log(32.0F))) < 1e-6F,
            "YarnMscale");

    // The GGUF vocabulary travels with the metadata and must match.
    Require(reader->GetMetadataStringArray("tokenizer.ggml.tokens").size() ==
                c.vocab_size,
            "vocabulary length");

    std::cout << "PASS laguna.config: " << c.num_layers << " layers, " << full
              << " full attention\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL laguna.config: " << e.what() << '\n';
    return 1;
  }
}
