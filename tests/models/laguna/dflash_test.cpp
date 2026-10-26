// Laguna DFlash drafter against the real laguna-s-2.1-DFlash-Q4_K.gguf header:
// the config, all 76 tensors bound with none left over, and the pairing with
// the real target's config. Only headers are read. Without the drafter file the
// test ends with the CTest skip status; nothing is verified in that case.
#include "src/models/laguna/dflash.hpp"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <map>
#include <set>
#include <string>

#include "tests/models/laguna/support.hpp"

namespace laguna = gufo::models::laguna;
namespace core = gufo::core;
using gufo::testing::laguna::Require;

namespace {

std::filesystem::path DrafterPath() {
  if (const char* env = std::getenv("GUFO_LAGUNA_DFLASH_GGUF");
      env != nullptr && *env != '\0') {
    return env;
  }
  return "/server/ai/models/Myric/Laguna-S-2.1-APEX-GGUF/"
         "laguna-s-2.1-DFlash-Q4_K.gguf";
}

}  // namespace

int main() {
  try {
    const auto path = DrafterPath();
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
      std::cout << "SKIP: laguna.dflash needs the Laguna-S-2.1 DFlash GGUF at "
                << path.string()
                << " (set GUFO_LAGUNA_DFLASH_GGUF); nothing was verified\n";
      return gufo::testing::laguna::kSkipExitCode;
    }
    std::string error;
    const auto reader = core::GgufReader::OpenFile(path, &error);
    Require(reader != nullptr, "drafter exists but does not open: " + error);
    Require(reader->GetTensorCount() == 76, "the drafter has 76 tensors");

    const auto parsed = laguna::DFlashConfig::FromGguf(*reader, &error);
    Require(parsed.has_value(), "DFlashConfig::FromGguf failed: " + error);
    const laguna::DFlashConfig& c = *parsed;

    // Expected values written out independently of the parser, from the
    // header manifest and the Hugging Face config (num_hidden_layers 6,
    // target_layer_ids [1, 10, 19, 29, 38, 47],
    // eagle_aux_hidden_state_layer_ids [2, 11, 20, 30, 39, 48], mask_token_id
    // 12, block_size 16).
    Require(c.num_layers == 6 && c.hidden_size == 3072 && c.ffn_size == 12288,
            "dimensions");
    Require(c.vocab_size == 100352, "vocabulary length");
    Require(c.context_length == 1048576, "context_length");
    Require(std::abs(c.rms_eps - 1e-6F) < 1e-12F, "rms eps");
    Require(c.num_heads == 72 && c.num_kv_heads == 8 && c.head_dim == 128,
            "attention geometry");
    Require(c.sliding_window == 512, "sliding_window");
    Require(c.rope_theta == 500000.0F && c.rotary_dim == 128, "rope");
    Require(c.block_size == 16 && c.max_draft_tokens() == 15, "block size");
    Require(c.mask_token_id == 12, "mask token");
    const std::vector<std::uint32_t> expected_gguf{2, 11, 20, 30, 39, 48};
    const std::vector<std::uint32_t> expected_outputs{1, 10, 19, 29, 38, 47};
    Require(c.target_layers == expected_gguf, "target_layers as written");
    for (std::size_t i = 0; i < expected_outputs.size(); ++i) {
      Require(c.tap_layer(i) == expected_outputs[i], "tap layer output index");
    }
    Require(c.fused_input_width() == 18432, "encoder input width");

    // Raw reads, to catch a parser that agrees with its own constants only.
    Require(reader->GetMetadataString("general.architecture") == "dflash",
            "raw architecture");
    Require(reader->GetMetadataString("dflash.decoder_arch") == "laguna",
            "raw decoder_arch");
    Require(reader->GetMetadataUint32("dflash.block_size") == 16U,
            "raw block_size");
    Require(reader->GetMetadataUint32("tokenizer.ggml.mask_token_id") == 12U,
            "raw mask token");
    const auto tokens = reader->GetMetadataStringArray("tokenizer.ggml.tokens");
    Require(tokens.size() == c.vocab_size, "vocabulary length");
    Require(tokens.at(12) == "\xE3\x80\x88|MASK|\xE3\x80\x89",
            "token 12 is the MASK control token");

    const auto weights = laguna::DFlashWeights::Bind(*reader, &error);
    Require(weights.has_value(), "DFlashWeights::Bind failed: " + error);
    Require(weights->layers.size() == 6, "6 layers bound");
    Require(weights->tensor_count == 76, "all 76 tensors bound");
    Require(laguna::DFlashWeights::ExpectedTensorCount(c) == 76,
            "layout formula gives 76");

    // Account for each tensor in the file exactly once, independently of
    // Bind's own bookkeeping.
    std::set<std::string_view> seen;
    std::map<core::GgmlType, std::size_t> census;
    std::uint64_t bytes = 0;
    const auto count = [&](const laguna::TensorRef& t) {
      Require(!t.empty(), "an expected tensor is unbound");
      Require(seen.insert(t.name).second,
              "tensor bound twice: " + std::string(t.name));
      const auto* info = reader->FindTensor(t.name);
      Require(info != nullptr, "bound tensor missing from file");
      Require(info->data == t.data && info->type == t.type,
              "binding disagrees with file: " + std::string(t.name));
      Require(t.SizeBytes() == info->size_bytes,
              "size: " + std::string(t.name));
      ++census[t.type];
      bytes += t.SizeBytes();
    };
    count(weights->aux_norm);
    count(weights->enc_output_norm);
    count(weights->fc);
    count(weights->output_norm);
    for (const auto& l : weights->layers) {
      for (const auto* t :
           {&l.attn_norm, &l.attn_q, &l.attn_k, &l.attn_v, &l.attn_output,
            &l.attn_q_norm, &l.attn_k_norm, &l.attn_gate, &l.ffn_norm,
            &l.ffn_gate, &l.ffn_up, &l.ffn_down}) {
        count(*t);
      }
    }
    Require(seen.size() == reader->GetTensorCount(),
            "every tensor in the file is accounted for");
    for (const auto& info : reader->GetTensors()) {
      Require(seen.contains(info.name),
              "unbound tensor in file: " + std::string(info.name));
    }
    Require(bytes == weights->total_bytes, "total bytes");
    // Header manifest census: 27 F32, 45 Q4_K, 4 Q6_K.
    Require(census[core::GgmlType::kF32] == 27 &&
                census[core::GgmlType::kQ4_K] == 45 &&
                census[core::GgmlType::kQ6_K] == 4 && census.size() == 3,
            "type census F32 27 / Q4_K 45 / Q6_K 4");
    // Spot checks of shapes the model card and the HF config fix.
    Require(weights->aux_norm.cols == 3072 && weights->aux_norm.rows == 6,
            "aux_norm [3072, 6]");
    Require(weights->fc.cols == 18432 && weights->fc.rows == 3072,
            "fc [18432, 3072]");
    Require(weights->layers[0].attn_gate.rows == 72, "per-head gate rows");
    Require(weights->layers[2].attn_v.type == core::GgmlType::kQ6_K &&
                weights->layers[2].ffn_down.type == core::GgmlType::kQ6_K &&
                weights->layers[5].attn_v.type == core::GgmlType::kQ6_K &&
                weights->layers[5].ffn_down.type == core::GgmlType::kQ6_K,
            "Q6_K value and down projections in layers 2 and 5");

    // The pairing with the real target, read from its header only.
    const auto target_reader = gufo::testing::laguna::OpenModelOrSkip(
        "laguna.dflash (target pairing)");
    const auto target = laguna::Config::FromGguf(*target_reader, &error);
    Require(target.has_value(), "target config: " + error);
    Require(c.ValidateAgainstTarget(*target, &error),
            "drafter does not pair with the target: " + error);
    Require(c.target_layers.back() == target->num_layers,
            "last tap equals the target layer count");

    std::cout << "PASS laguna.dflash: " << c.num_layers << " layers, "
              << c.tap_count() << " taps, block " << c.block_size << ", "
              << weights->tensor_count << " tensors\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL laguna.dflash: " << e.what() << '\n';
    return 1;
  }
}
