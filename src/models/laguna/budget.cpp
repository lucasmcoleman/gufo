#include "src/models/laguna/budget.hpp"

#include <algorithm>
#include <cstdio>

namespace gufo::models::laguna {
namespace {

constexpr std::size_t kRowPadding = 512;  // ggml MATRIX_ROW_PADDING
constexpr std::size_t kF = sizeof(float);
constexpr std::size_t kArenaStart = std::size_t{256} << 20;  // qfn_stack_pool
constexpr std::size_t kDispatchReserve = std::size_t{4} << 30;
constexpr std::uint32_t kVectorMaxRows = 32;
constexpr std::size_t kRuntimeAllowance = std::size_t{32} << 20;

/// qfn_mmq_q8_1_bytes(N, K): N rows of K padded to 512, 36 bytes per 32.
std::size_t Q8Bytes(std::size_t rows, std::size_t k) {
  const std::size_t padded = (k + kRowPadding - 1) / kRowPadding * kRowPadding;
  return rows * padded * 36 / 32;
}

std::string Gib(std::size_t bytes) {
  char text[64];
  std::snprintf(text, sizeof(text), "%8.2f GiB (%zu B)",
                static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0), bytes);
  return text;
}

}  // namespace

std::size_t DeviceTensorBytes(const TensorRef& t) noexcept {
  if (t.empty()) {
    return 0;
  }
  const std::size_t tail =
      t.type == core::GgmlType::kF32 ? 0 : t.RowBytes() * kRowPadding / t.cols;
  return t.SizeBytes() + tail;
}

std::size_t WeightDeviceBytes(const ModelWeights& w) {
  std::size_t total =
      DeviceTensorBytes(w.output_norm) + DeviceTensorBytes(w.output);
  for (const LayerWeights& l : w.layers) {
    for (const TensorRef* t :
         {&l.attn_norm,    &l.attn_q,          &l.attn_k,
          &l.attn_v,       &l.attn_output,     &l.attn_q_norm,
          &l.attn_k_norm,  &l.attn_gate,       &l.ffn_norm,
          &l.ffn_gate,     &l.ffn_up,          &l.ffn_down,
          &l.ffn_gate_inp, &l.ffn_exp_probs_b, &l.ffn_gate_exps,
          &l.ffn_up_exps,  &l.ffn_down_exps,   &l.ffn_gate_shexp,
          &l.ffn_up_shexp, &l.ffn_down_shexp}) {
      total += DeviceTensorBytes(*t);
    }
  }
  return total;
}

MemoryBudget ComputeBudget(const ModelWeights& weights,
                           const BudgetRequest& request) {
  const Config& c = weights.config;
  MemoryBudget b;
  b.request = request;
  b.reserve = kDispatchReserve;
  b.weights = WeightDeviceBytes(weights);

  const std::size_t kv_row = std::size_t{c.num_kv_heads} * c.head_dim * 2;
  std::uint32_t max_heads = 0;
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    max_heads = std::max(max_heads, c.n_head(il));
    const std::size_t positions =
        c.is_full_attention(il)
            ? request.context
            : std::size_t{c.sliding_window} + request.max_chunk;
    (c.is_full_attention(il) ? b.kv_full : b.kv_swa) += 2 * positions * kv_row;
  }

  const std::size_t C = request.max_chunk;
  const std::size_t H = c.hidden_size;
  const std::size_t V = c.vocab_size;
  // h, normed, ffn_inp, branch output; a zero id row per token.
  b.activations = 4 * C * H * kF + C * sizeof(std::int32_t);
  b.logits = (request.all_row_logits ? C : 1) * V * kF;
  b.host_staging = C * H * kF + (request.all_row_logits ? C : 1) * V * kF;

  const std::size_t q_dim = std::size_t{max_heads} * c.head_dim;
  const std::size_t kv_dim = c.kv_dim();
  b.attention_scratch = 2 * C * q_dim * kF + 2 * C * kv_dim * kF +
                        C * max_heads * kF + C * sizeof(std::int32_t);

  const std::size_t used = c.num_experts_used;
  const std::size_t rows_ff = std::max<std::size_t>(
      {C * c.ffn_size, C * used * c.expert_ff, C * c.shared_expert_ff});
  std::size_t max_k =
      std::max<std::size_t>({H, c.ffn_size, c.expert_ff, c.shared_expert_ff,
                             std::size_t{c.n_head(0)} * c.head_dim, q_dim});
  b.ffn_scratch = C * c.num_experts * kF + 2 * C * used * sizeof(float) +
                  3 * rows_ff * kF + C * used * H * kF + C * H * kF +
                  Q8Bytes(std::min<std::size_t>(C, kVectorMaxRows), max_k);

  // The stack arena starts at 256 MiB and doubles; its peak is a few
  // activation stagings (MMQ layout of the widest input, plus gathered rows
  // for routed experts).
  const std::size_t need = 2 * Q8Bytes(std::max<std::size_t>(C, 1), max_k) +
                           Q8Bytes(std::max<std::size_t>(C * used, 1),
                                   std::max<std::size_t>(c.expert_ff, H));
  std::size_t arena = kArenaStart;
  while (arena < need) {
    arena *= 2;
  }
  b.mmq_arena = arena;
  b.runtime = kRuntimeAllowance;
  return b;
}

std::string MemoryBudget::Format() const {
  std::string text;
  text += "laguna memory budget (context " + std::to_string(request.context) +
          ", chunk " + std::to_string(request.max_chunk) + ", " +
          (request.all_row_logits ? "all-row" : "last-row") + " logits)\n";
  const auto line = [&](const char* name, std::size_t bytes) {
    text += std::string("  ") + name + Gib(bytes) + "\n";
  };
  line("weights (device)      ", weights);
  line("kv full layers        ", kv_full);
  line("kv sliding rings      ", kv_swa);
  line("activations           ", activations);
  line("attention scratch     ", attention_scratch);
  line("ffn scratch           ", ffn_scratch);
  line("logits (device)       ", logits);
  line("mmq stream arena      ", mmq_arena);
  line("runtime allowance     ", runtime);
  line("device total          ", device_total());
  line("driver reserve        ", reserve);
  line("host staging          ", host_staging);
  return text;
}

}  // namespace gufo::models::laguna
