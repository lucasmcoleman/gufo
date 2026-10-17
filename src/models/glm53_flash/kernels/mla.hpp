#ifndef GUFO_MODELS_GLM53_FLASH_KERNELS_MLA_HPP_
#define GUFO_MODELS_GLM53_FLASH_KERNELS_MLA_HPP_

// Internal interface between kernels/mla.hip.cpp (absorbed MLA, the
// MlaForward entry point, cache and scratch layout) and kernels/indexer.hip.cpp
// (the pooled DSA indexer). The scalar contract is ReferenceModel::Mla
// (reference.cpp) on top of reference_mla.hpp.
//
// MlaCache layout (one MLA layer, `capacity` positions, F32, each region
// 256-byte aligned):
//   latent   [capacity][kv_lora]      RMSNorm(kv_a x), the whole K = V cache
//   pooled   [capacity / kpool][idim] pooled indexer keys, one per complete
//                                     pool
//   ring_key [kpool][idim]            LayerNorm indexer keys and gate logits
//   ring_gate[kpool][idim]            of the incomplete pool, slot pos % kpool
//   scores   [kSelectTokens][capacity / kpool]
//                                     pool scores of the query tokens being
//                                     selected (workspace; sized by capacity,
//                                     which the scratch cannot know)
// Every value is a function of the positions written so far, so the state
// carries across calls with any chunking; nothing needs zeroing.
//
// MlaForward taps (llama.cpp cb() names, ggml order): q_resid [q_lora, T],
// kv_cmpr [kv_lora, 1, T] (the cache rows), indexer_q [idim, ih, T],
// indexer_k [idim, T], indexer_gate [idim, T], indexer_weights [ih, T]
// (scaled), indexer_pool_k [idim, 1, pools], indexer_top_k [budget, T] I32
// (descending score, -1 past the token's pool count), kqv_out
// [heads * v_dim, T], attn_out [hidden, T]. A call that needs several
// scratch passes taps once per pass with that pass's T. Every TapView is a
// borrowed view of scratch or cache that later work on ctx.stream reuses:
// the consumer reads it or enqueues its copy on ctx.stream before the tap
// returns and never keeps it as a snapshot. Taps add no sync or allocation.

#include <cstddef>
#include <cstdint>

#include "src/models/glm53_flash/config.hpp"
#include "src/models/glm53_flash/layer_ops.hpp"

namespace gufo::models::glm53_flash::mla {

/// Query tokens whose pool scores are held at once (cache `scores` rows).
inline constexpr std::uint32_t kSelectTokens = 16;

struct CacheLayout {
  float* latent{nullptr};
  float* pooled{nullptr};
  float* ring_key{nullptr};
  float* ring_gate{nullptr};
  float* scores{nullptr};
  std::uint32_t pool_capacity{0};  ///< capacity / kpool, the scores stride
};

/// Sizes (and, with `base`, lays out) the cache of `capacity` positions.
[[nodiscard]] std::size_t LayoutCache(const Config& config,
                                      std::uint32_t capacity, void* base,
                                      CacheLayout* out) noexcept;

/// Indexer buffers of one pass of `n` tokens (scratch, caller laid out).
struct IndexerBuffers {
  const float* x{nullptr};        ///< [n][hidden] layer input
  const float* q_resid{nullptr};  ///< [n][q_lora]
  float* iq{nullptr};             ///< [n][ih][idim]
  float* ik{nullptr};             ///< [n][idim] LayerNorm keys
  float* ig{nullptr};             ///< [n][idim] pool gate logits
  float* iw{nullptr};             ///< [n][ih] head weights * 1/sqrt(ih*idim)
  std::int32_t* sel{nullptr};     ///< [n][budget] selected pools, ascending
  std::int32_t* top_k{nullptr};   ///< [n][budget] tap order, or nullptr
};

/// Y[t][r] = sum_c W[expert][r][c] X[t][c] for `tokens` rows of X (row
/// stride w.cols) into Y (row stride w.rows). F32 weights use a
/// warp-per-output kernel (any cols); quantized and BF16 weights use gufo's
/// GEMMs. Every route computes a row independently of the batch.
[[nodiscard]] bool Project(const DeviceMatrix& w, std::uint64_t expert,
                           const float* x, float* y, std::uint32_t tokens,
                           Stream stream);

/// True if `w` (any expert) has a Project route.
[[nodiscard]] bool CanProject(const DeviceMatrix& w) noexcept;

/// Row-wise RMSNorm (weight, no bias) or LayerNorm (weight + bias) of
/// `rows` rows of `width`, src stride `width`, dst stride `width`. In place
/// is allowed.
[[nodiscard]] bool NormRows(const float* src, const float* weight,
                            const float* bias, float* dst, std::uint32_t rows,
                            std::uint32_t width, float eps, Stream stream);

/// The DSA indexer for one pass: projections, key LayerNorm, pooling of
/// every pool completed by [pos0, pos0 + n), ring update, pool scores and
/// the per-token top-budget selection (ties to the lower pool ID). Token t
/// sees the (pos0 + t + 1) / kpool complete pools; with that count at or
/// below the budget it selects them all. sel[t] lists min(pools, budget)
/// pool IDs in ascending order; the raw tail is implied. With `b.top_k` set
/// it also writes the selected pools ordered by descending score (ties by
/// ID), -1 past the count, as llama.cpp's `indexer_top_k`.
[[nodiscard]] bool IndexerForward(const Config& config, const MlaLayerDevice& w,
                                  const CacheLayout& cache,
                                  const IndexerBuffers& b, std::uint32_t n,
                                  std::uint32_t pos0, Stream stream);

/// True if the indexer kernels support `config` (dims, LDS budget).
[[nodiscard]] bool IndexerSupported(const Config& config) noexcept;

}  // namespace gufo::models::glm53_flash::mla

#endif  // GUFO_MODELS_GLM53_FLASH_KERNELS_MLA_HPP_
