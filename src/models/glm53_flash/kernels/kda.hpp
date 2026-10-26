#ifndef GUFO_MODELS_GLM53_FLASH_KERNELS_KDA_HPP_
#define GUFO_MODELS_GLM53_FLASH_KERNELS_KDA_HPP_

#include <hip/hip_runtime_api.h>

#include <cstdint>

/// GLM-5.3-Flash Kimi Delta Attention (KDA) kernels, forked from the Qwen3.8
/// Flash-Next gated delta net (qwen38_flash_next/kernels/rocm/kernels.hip.cpp:
/// SsmConv4Kernel, GdnPrepKernel, GdnPrepAbKernel, GdnKernel,
/// GdnEpilogueKernel). The projections (q/k/v, f_a/f_b, beta, g_a/g_b, o) are
/// the caller's GEMMs; these kernels cover everything between them. The
/// scalar contract is reference_kda.hpp. Head dim 128 (key = value) only;
/// any head count. Differences from QFN: a per-key-channel decay vector,
/// the lb*sigmoid forget gate, no GQA, the l2 eps passed separately.
///
/// Layouts, all F32 row-major and device-resident (H heads, D = 128):
///   qkv       [T][qkv_stride]   q|k|v projections, channels = 3*H*D
///   conv_w    [channels][4]     the three GGUF convs concatenated q|k|v
///   conv_state [3][channels]    last three inputs, row 0 oldest
///   conv_out  [T][channels]     SiLU(conv) of q|k|v
///   f         [T][f_stride]     f_b(f_a(x)), H*D used
///   g, decay  [T][H][D]         log decay and exp(log decay)
///   beta_in   [T][beta_stride]  b_proj(x) logits, H used; beta [T][H]
///   qn, kn    [T][H][D]         l2-normed q (scaled by 1/sqrt(D)) and k
///   state     [H][D v][D k]     TRANSPOSED vs HF's [k][v]: row j is value j,
///                               its columns are keys, so the per-key decay
///                               scales columns. state[h][j][i] =
///                               S_hf[h][i][j].
///   o         [T][H][D]         raw recurrence output
///   gate      [T][gate_stride]  g_b(g_a(x)), H*D used
///   out       [T][H][D]         RMSNorm(o)*w*sigmoid(gate), the o_proj input
/// Every launch is asynchronous on `stream`. Launchers return true without
/// launching for n_tokens == 0, false (without launching) on a null pointer,
/// an unsupported shape or a misaligned float4 operand, and false when the
/// launch fails.
namespace gufo::models::glm53_flash::rocm {

constexpr std::uint32_t kKdaHeadDim = 128;
constexpr std::uint32_t kKdaConvTaps = 4;

/// Causal depthwise 4-tap conv over [conv_state ; qkv] then SiLU, and the
/// conv_state update to the last three inputs. conv_w must be 16-byte
/// aligned. conv_out may not alias qkv. With `checkpoint`
/// [n_tokens - 1][3][channels], also writes the history after each token
/// but the last (speculative verify; at most 8 tokens).
bool KdaConv(const float* qkv, std::uint32_t qkv_stride, const float* conv_w,
             float* conv_state, float* conv_out, std::uint32_t n_tokens,
             std::uint32_t channels, hipStream_t stream,
             float* checkpoint = nullptr);

/// Forget gate: z = f + dt_bias, a = -exp(A_log) per head (GGUF ssm_a);
/// g = lower_bound * sigmoid(-a z) for a finite lower_bound, else
/// a * softplus(z). Writes decay = exp(g) and, when non-null, g.
bool KdaGate(const float* f, std::uint32_t f_stride, const float* dt_bias,
             const float* a, float lower_bound, float* g, float* decay,
             std::uint32_t n_tokens, std::uint32_t heads, hipStream_t stream);

/// qn = q / sqrt(sum q^2 + eps) / sqrt(D), kn = k / sqrt(sum k^2 + eps)
/// from conv_out (q at channel 0, k at channel H*D, row stride conv_stride),
/// and beta = sigmoid(beta_in). eps is 1e-6 in the model.
bool KdaPrep(const float* conv_out, std::uint32_t conv_stride,
             const float* beta_in, std::uint32_t beta_stride, float* qn,
             float* kn, float* beta, std::uint32_t n_tokens,
             std::uint32_t heads, float eps, hipStream_t stream);

/// Sequential recurrence over n_tokens (1 for decode), carrying `state` in
/// place: per token and head, S <- diag(decay) S; delta = beta (v - S^T k);
/// S += k delta^T; o = S^T qn (HF layout notation; see `state` above).
/// v is read at v[t * v_stride + h * D + j] (conv_out + 2*H*D with the conv
/// stride).
/// With `checkpoint` [n_tokens - 1][heads][D][D], also writes the state
/// after each token but the last, bit-identical to stopping there.
bool KdaRecurrent(const float* qn, const float* kn, const float* v,
                  std::uint32_t v_stride, const float* decay, const float* beta,
                  float* state, float* o, std::uint32_t n_tokens,
                  std::uint32_t heads, hipStream_t stream,
                  float* checkpoint = nullptr);

/// out = RMSNorm(o row of D) * norm_w * sigmoid(gate), eps = rms_norm_eps.
/// out may alias o.
bool KdaOutput(const float* o, const float* gate, std::uint32_t gate_stride,
               const float* norm_w, float eps, float* out,
               std::uint32_t n_tokens, std::uint32_t heads, hipStream_t stream);

/// Prefill low-rank up-projection Y[t][r] = sum_k W[r][k] X[t][k] for a
/// Q8_0 W of `rows` x 128 (f_b / g_b: D -> H*D), X [n_tokens][128] and Y
/// [n_tokens][rows], F32. The 128-column rows fall outside the Q8_1 WMMA
/// prefill GEMM (cols % 256) and gufo's batched quant GEMM is slow on them.
/// A block dequantizes 64 rows into LDS (d * q, exact) and dots them with 32
/// staged tokens; every output sums its 128 products in k order, in F32
/// (exact activations, not Q8-rounded). Not bit-identical to the GEMM it
/// replaces (summation order). Idea from the token-batched KDA projections of
/// mikealanni/gufo 3ba034e2 / a9ea8e67 (MIT, Mike Alani); GLM-local kernel.
/// False (nothing launched) unless cols == 128, w is 2-byte aligned and the
/// pointers are non-null. Switch: GUFO_GLM53_KDA_LOWRANK=1 (executor.cpp).
bool KdaLowRankQ8(const void* w, std::uint32_t rows, std::uint32_t cols,
                  const float* x, float* y, std::uint32_t n_tokens,
                  hipStream_t stream);

}  // namespace gufo::models::glm53_flash::rocm

#endif  // GUFO_MODELS_GLM53_FLASH_KERNELS_KDA_HPP_
