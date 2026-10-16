#ifndef GUFO_MODELS_GLM53_FLASH_REFERENCE_HC_HPP_
#define GUFO_MODELS_GLM53_FLASH_REFERENCE_HC_HPP_

// Scalar oracle for GLM-5.3-Flash manifold-constrained hyper-connections
// (mHC). One token at a time, F32 storage with F64 accumulation. Mirrors HF
// transformers 09281b8 modeling_glm5_next.py:
//   Glm5NextTextUnweightedRMSNorm   (HF:209-215)
//   Glm5NextTextHyperConnection     (HF:271-329)  -> HcPre
//   Glm5NextTextDecoderLayer expand (HF:1336-1338, 1344-1346) -> HcPost
//   Glm5NextTextHyperHead           (HF:332-336)  -> HcHeadMean
//   stream init: 4 copies of the token embedding (HF:1493)
//
// Layouts: H is [hc_count][hidden] (stream-major, the flatten order HF uses);
// fn is the GGUF tensor [cols = hc_count*hidden, rows = 2*hc + hc^2], row j
// contiguous; comb is [src][dst], so H_new[dst] sums comb[src][dst]*H[src].

#include <cstdint>
#include <vector>

#include "src/models/glm53_flash/config.hpp"

namespace gufo::models::glm53_flash::reference {

/// Per-token mixing weights of one hyper-connection.
struct HcMix {
  std::uint32_t hc_count{0};
  std::uint32_t hidden{0};
  std::vector<float> logits;  ///< [2*hc + hc^2], fn . RMSNorm(flat(H))
  std::vector<float> pre;     ///< [hc] sigmoid(..) + hc_eps
  std::vector<float> post;    ///< [hc] 2 * sigmoid(..)
  std::vector<float> comb;    ///< [hc][hc] doubly stochastic, [src][dst]
};

/// Computes the mixing weights from H and collapses H into the sub-layer
/// input x_out = sum_s pre[s] * H[s] ([hidden]). `fn`, `base`, `scale` are
/// one mixer's hc_{attn,ffn}_{fn,base,scale}. `mix_out` may be null.
void HcPre(const Config& config, const float* fn, const float* base,
           const float* scale, const float* H, float* x_out, HcMix* mix_out);

/// H_new[dst] = post[dst] * a + sum_src comb[src][dst] * H_old[src].
/// `a` is the sub-layer output [hidden]; H_new may not alias H_old.
void HcPost(const HcMix& mix, const float* a, const float* H_old, float* H_new);

/// Plain mean over the streams: h_out[d] = mean_s H[s][d].
void HcHeadMean(const Config& config, const float* H, float* h_out);

/// H[s] = embedding for every stream.
void HcInit(const Config& config, const float* embedding, float* H);

}  // namespace gufo::models::glm53_flash::reference

#endif  // GUFO_MODELS_GLM53_FLASH_REFERENCE_HC_HPP_
