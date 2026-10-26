// Weight index against the real Laguna-S-2.1 GGUF header: all 814 tensors must
// bind, by name, shape and ggml type, with none left over. Header only; the
// payload is never read. Skips (CTest status 77) when the file is absent.
#include "src/models/laguna/weights.hpp"

#include <iostream>
#include <map>
#include <set>
#include <string>

#include "tests/models/laguna/support.hpp"

namespace laguna = gufo::models::laguna;
namespace core = gufo::core;
using gufo::testing::laguna::Require;

int main() {
  try {
    const auto reader =
        gufo::testing::laguna::OpenModelOrSkip("laguna.weights");
    Require(reader->GetTensorCount() == 814, "the artifact has 814 tensors");

    std::string error;
    const auto weights = laguna::ModelWeights::Bind(*reader, &error);
    Require(weights.has_value(), "ModelWeights::Bind failed: " + error);
    const laguna::Config& c = weights->config;
    Require(weights->layers.size() == 48, "48 layers bound");
    Require(weights->tensor_count == 814, "all 814 tensors bound");
    Require(laguna::ModelWeights::ExpectedTensorCount(c) == 814,
            "layout formula gives 814");

    // Walk every bound reference and account for each tensor in the file
    // exactly once, independently of Bind's own bookkeeping.
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
    count(weights->token_embd);
    count(weights->output_norm);
    count(weights->output);
    for (std::uint32_t il = 0; il < 48; ++il) {
      const auto& l = weights->layers[il];
      Require(l.full_attention == (il % 4 == 0), "layer kind");
      Require(l.moe == (il != 0), "layer moe flag");
      for (const auto* t :
           {&l.attn_norm, &l.attn_q, &l.attn_k, &l.attn_v, &l.attn_output,
            &l.attn_q_norm, &l.attn_k_norm, &l.attn_gate, &l.ffn_norm}) {
        count(*t);
      }
      if (l.moe) {
        for (const auto* t :
             {&l.ffn_gate_inp, &l.ffn_exp_probs_b, &l.ffn_gate_exps,
              &l.ffn_up_exps, &l.ffn_down_exps, &l.ffn_gate_shexp,
              &l.ffn_up_shexp, &l.ffn_down_shexp}) {
          count(*t);
        }
        Require(l.ffn_gate.empty() && l.ffn_up.empty() && l.ffn_down.empty(),
                "MoE layer carries dense tensors");
      } else {
        for (const auto* t : {&l.ffn_gate, &l.ffn_up, &l.ffn_down}) {
          count(*t);
        }
        Require(l.ffn_gate_inp.empty() && l.ffn_gate_exps.empty(),
                "dense layer carries MoE tensors");
      }
    }
    Require(seen.size() == 814, "814 distinct tensors accounted for");
    for (const auto& info : reader->GetTensors()) {
      Require(seen.contains(info.name),
              "unaccounted tensor: " + std::string(info.name));
    }
    Require(bytes == weights->total_bytes, "total bytes");

    // Type census recorded from the header manifest of this artifact.
    Require(census[core::GgmlType::kF32] == 335, "F32 count");
    Require(census[core::GgmlType::kQ6_K] == 221, "Q6_K count");
    Require(census[core::GgmlType::kQ8_0] == 144, "Q8_0 count");
    Require(census[core::GgmlType::kQ5_K] == 30, "Q5_K count");
    Require(census[core::GgmlType::kIQ4_XS] == 84, "IQ4_XS count");

    // Routed experts per layer range: the quantization the GPU phase must
    // dispatch (Q6_K L1-4/43-47, Q5_K L5-9/38-42, IQ4_XS L10-37).
    for (std::uint32_t il = 1; il < 48; ++il) {
      const auto want = (il <= 4 || il >= 43)   ? core::GgmlType::kQ6_K
                        : (il <= 9 || il >= 38) ? core::GgmlType::kQ5_K
                                                : core::GgmlType::kIQ4_XS;
      const auto& l = weights->layers[il];
      for (const auto* t :
           {&l.ffn_gate_exps, &l.ffn_up_exps, &l.ffn_down_exps}) {
        Require(t->type == want,
                "routed expert type at layer " + std::to_string(il));
      }
    }
    // Shapes spot-checked against the manifest: GQA widths and gate per head.
    const auto& full0 = weights->layers[0];
    const auto& swa1 = weights->layers[1];
    Require(full0.attn_q.rows == 6144 && swa1.attn_q.rows == 9216, "q width");
    Require(full0.attn_output.cols == 6144 && swa1.attn_output.cols == 9216,
            "o width");
    Require(full0.attn_gate.rows == 48 && swa1.attn_gate.rows == 72,
            "per-head gate rows");
    Require(weights->layers[10].ffn_down_exps.cols == 1024 &&
                weights->layers[10].ffn_down_exps.rows == 3072 &&
                weights->layers[10].ffn_down_exps.experts == 256,
            "expert down shape");

    std::cout << "PASS laguna.weights: 814 tensors bound, " << bytes
              << " payload bytes\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL laguna.weights: " << e.what() << '\n';
    return 1;
  }
}
