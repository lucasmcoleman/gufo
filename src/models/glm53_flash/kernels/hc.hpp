#ifndef GUFO_MODELS_GLM53_FLASH_KERNELS_HC_HPP_
#define GUFO_MODELS_GLM53_FLASH_KERNELS_HC_HPP_

#include <hip/hip_runtime_api.h>

#include <cstdint>

/// GLM-5.3-Flash hyper-connection (mHC) kernels, forked from DeepSeek V4's
/// ds4_rocm_hc.hip.hpp (hc4_split_one, the fused norm/mix/split/collapse and
/// the expand4 kernels) with F32 mix weights and Config-supplied epsilons.
/// The scalar contract is reference_hc.hpp. Four streams only.
///
/// Layouts, all F32 row-major and device-resident:
///   H        [tokens][4][hidden]   residual streams
///   fn       [24][4*hidden]        hc_{attn,ffn}_fn (GGUF ne = {4*hidden, 24})
///   base     [24], scale [3]
///   split    [tokens][24]          pre[0:4], post[4:8], comb[8:24]
///                                  (comb is [src][dst])
///   x, a     [tokens][hidden]
/// Every launch is asynchronous on `stream`. Launchers return false, without
/// launching, on an unsupported shape or misaligned pointer, and false when
/// the launch itself fails.
namespace gufo::models::glm53_flash::rocm {

constexpr std::uint32_t kHcStreams = 4;
constexpr std::uint32_t kHcMixDim = 24;  ///< 2*4 + 4*4

/// Unweighted RMSNorm over the flattened streams (eps `rms_eps`), the 24-logit
/// projection by `fn`, the sigmoid/Sinkhorn split (eps `hc_eps`,
/// `sinkhorn_iters` total column normalisations), and the collapse
/// x = sum_s pre[s] * H[s]. Writes `split`; `logits` (raw projection, [T][24])
/// may be null. Hidden 4096 takes the fused register-resident path (for at
/// most 8 tokens, the decode case, as two launches that spread the 24 mix
/// rows over many blocks; bitwise identical to the single-launch kernel);
/// other sizes a generic one-block-per-token path. The two-launch path keeps
/// its partial sums in a module-level device buffer, so two HcPre calls must
/// not run concurrently on different streams.
bool HcPre(const float* H, const float* fn, const float* base,
           const float* scale, float* x, float* split, float* logits,
           std::uint32_t n_tokens, std::uint32_t hidden,
           std::uint32_t sinkhorn_iters, float hc_eps, float rms_eps,
           hipStream_t stream);

/// H_new[t][dst] = post[t][dst] * a[t] + sum_src comb[t][src][dst] *
/// H_old[t][src]. H_new may equal H_old (in-place update); any other overlap
/// is not allowed.
bool HcPost(const float* a, const float* H_old, const float* split,
            float* H_new, std::uint32_t n_tokens, std::uint32_t hidden,
            hipStream_t stream);

/// h[t] = mean_s H[t][s] (plain mean; GLM has no output_hc weights).
bool HcHeadMean(const float* H, float* h, std::uint32_t n_tokens,
                std::uint32_t hidden, hipStream_t stream);

}  // namespace gufo::models::glm53_flash::rocm

#endif  // GUFO_MODELS_GLM53_FLASH_KERNELS_HC_HPP_
