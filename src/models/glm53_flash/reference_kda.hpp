#ifndef GUFO_MODELS_GLM53_FLASH_REFERENCE_KDA_HPP_
#define GUFO_MODELS_GLM53_FLASH_REFERENCE_KDA_HPP_

// Scalar oracle for the GLM-5.3-Flash Kimi Delta Attention (KDA) mixer,
// excluding the quantized projections (the executor's GEMMs). F32 storage,
// F64 accumulation. Mirrors HF transformers 09281b8 modeling_glm5_next.py:
//   causal_conv1d_update / causal_conv1d_fn (HF:409-448)   -> KdaConv
//   Glm5NextTextForgetGate.forward          (HF:353-369)   -> KdaGate
//   l2norm, q scale, sigmoid(b_proj)  (HF:451-459, 478-487, 733) -> KdaPrep
//   recurrent_kimi_delta_attention loop     (HF:497-511)   -> KdaRecurrent
//   Glm5NextTextRMSNormGated.forward        (HF:381-393)   -> KdaOutput
// The module wiring is Glm5NextTextLinearAttention.forward (HF:664-769).
//
// Layouts (row-major):
//   q|k|v, conv in/out  [T][channels]   channels = 3 * H * D, q|k|v
//   concatenated conv weight         [channels][kernel]  (GGUF ne {kernel, 1,
//   dim}; tap
//                       kernel-1 multiplies the current token)
//   conv history        [kernel-1][channels], row 0 the oldest token
//   g, f                [T][H][Dk]   natural-log per-key decay
//   beta                [T][H]
//   state               [H][Dk][Dv]  HF layout: key rows, value columns. The
//                       decay scales key ROWS: S <- diag(exp g) S (HF:506).
//   o, gate             [T][H][Dv]

#include <cstdint>

namespace gufo::models::glm53_flash::reference {

/// Causal depthwise conv over [history ; x] followed by SiLU. `history`
/// ([kernel-1][channels]) is read as the tokens preceding x[0] and
/// replaced by the last kernel-1 inputs of [history ; x]. A zero history is
/// HF's fresh prefill (left zero padding, HF:438-448).
void KdaConv(const float* x, const float* weight, float* history, float* out,
             std::uint32_t n_tokens, std::uint32_t channels,
             std::uint32_t kernel);

/// Per-channel log-decay from the forget-gate low-rank output f
/// (= f_b(f_a(x)), [T][H*Dk]), dt_bias [H*Dk] and a [H] (= -exp(A_log), the
/// GGUF ssm_a). With a finite lower bound lb:
///   g = lb * sigmoid(exp(A_log) * (f + dt_bias)) = lb * sigmoid(-a * (f+dt))
/// otherwise (lb = -inf) g = a * softplus(f + dt_bias) (HF:363-369).
void KdaGate(const float* f, const float* dt_bias, const float* a,
             float lower_bound, float* g, std::uint32_t n_tokens,
             std::uint32_t heads, std::uint32_t key_dim);

/// In place: q <- q / sqrt(sum q^2 + eps) / sqrt(Dk), k <- k / sqrt(sum k^2
/// + eps) per (token, head) row ([T][H][Dk]); beta <- sigmoid(beta)
/// ([T][H]). eps is 1e-6 in the model (HF:458, independent of rms_eps).
/// `beta` may be null.
void KdaPrep(float* q, float* k, float* beta, std::uint32_t n_tokens,
             std::uint32_t heads, std::uint32_t key_dim, float eps);

/// Sequential KDA recurrence over T tokens on prepared q, k (KdaPrep), raw
/// v, log-decay g and beta = sigmoid(..). Per token and head:
///   S <- diag(exp g) S;  delta = beta (v - S^T k);  S <- S + k delta^T;
///   o = S^T q.
/// `state` [H][Dk][Dv] is the initial state on entry, the final on exit.
/// Splitting T across calls is exact (prefill then decode).
void KdaRecurrent(const float* q, const float* k, const float* v,
                  const float* g, const float* beta, float* state, float* o,
                  std::uint32_t n_tokens, std::uint32_t heads,
                  std::uint32_t key_dim, std::uint32_t value_dim);

/// out = RMSNorm_head(o) * weight * sigmoid(gate), per (token, head) row of
/// Dv values; eps is rms_norm_eps (1e-5, HF:637,658). weight is [Dv],
/// shared by the heads. out may alias o.
void KdaOutput(const float* o, const float* gate, const float* weight,
               float eps, float* out, std::uint32_t n_tokens,
               std::uint32_t heads, std::uint32_t value_dim);

}  // namespace gufo::models::glm53_flash::reference

#endif  // GUFO_MODELS_GLM53_FLASH_REFERENCE_KDA_HPP_
