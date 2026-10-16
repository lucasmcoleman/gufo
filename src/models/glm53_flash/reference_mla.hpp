#ifndef GUFO_MODELS_GLM53_FLASH_REFERENCE_MLA_HPP_
#define GUFO_MODELS_GLM53_FLASH_REFERENCE_MLA_HPP_

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gufo::models::glm53_flash::reference {

// All storage is F32 row-major, reduction in F64. These are mathematical
// oracles; production cache rounding is a separate device-parity contract.
void LayerNorm(const float* x, const float* weight, const float* bias,
               std::size_t width, float eps, float* y);
// One complete pool: keys/gates/ape all [pool_size][dim]. Normalization
// is across tokens independently for each channel; keys are pre-normalized.
void PoolKeys(const float* keys, const float* gates, const float* ape,
              std::size_t pool_size, std::size_t dim, float* pooled);
// iq [heads][dim], pooled [pools][dim], raw projection weights [heads].
// Applies 1/sqrt(heads*dim), matching llama.cpp's placement of the scale.
void IndexerScores(const float* iq, const float* pooled, const float* weights,
                   std::size_t heads, std::size_t dim, std::size_t pools,
                   float* scores);
struct Selection {
  std::vector<std::int32_t> pools;
  std::vector<std::int32_t> tokens;
};
// Single sequence. first_valid permits a left-padded reference fixture.
// Scores start at the pool beginning at first_valid. Only complete visible
// pools participate; tail is appended in raw-token order. Ties use pool ID.
Selection SelectPools(const float* scores, std::uint32_t visible_tokens,
                      std::uint32_t first_valid, std::uint32_t pool_size,
                      std::uint32_t pool_budget, bool select_tail = true);
// q [heads][qk_dim], kb [heads][latent_dim][qk_dim] in GGUF row layout.
void AbsorbQuery(const float* q, const float* kb, std::size_t heads,
                 std::size_t qk_dim, std::size_t latent_dim, float* absorbed);
// K=V=latent [rows][latent_dim]; q [heads][latent_dim]. Caller supplies
// the causal selection (unique nonnegative indices). Empty selection -> 0.
// Scale is 1/sqrt(qk_dim), NOT 1/sqrt(latent_dim).
void LatentAttention(const float* q, const float* latent, std::size_t rows,
                     std::size_t heads, std::size_t latent_dim,
                     std::size_t qk_dim,
                     const std::vector<std::int32_t>& selected, float* context);
// vb [heads][value_dim][latent_dim], ctx [heads][latent_dim].
void ProjectValues(const float* context, const float* vb, std::size_t heads,
                   std::size_t latent_dim, std::size_t value_dim, float* y);

}  // namespace gufo::models::glm53_flash::reference
#endif
