#include "src/models/glm53_flash/reference_mla.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace gufo::models::glm53_flash::reference {
void LayerNorm(const float* x, const float* weight, const float* bias,
               std::size_t width, float eps, float* y) {
  if (width == 0 || !(eps > 0))
    throw std::invalid_argument("invalid norm shape/eps");
  double mean = 0, var = 0;
  for (std::size_t d = 0; d < width; ++d)
    mean += x[d];
  mean /= width;
  for (std::size_t d = 0; d < width; ++d) {
    const double delta = x[d] - mean;
    var += delta * delta;
  }
  const double inv = 1.0 / std::sqrt(var / width + eps);
  for (std::size_t d = 0; d < width; ++d)
    y[d] = static_cast<float>((x[d] - mean) * inv * weight[d] + bias[d]);
}
void PoolKeys(const float* keys, const float* gates, const float* ape,
              std::size_t pool_size, std::size_t dim, float* pooled) {
  if (!pool_size || !dim)
    throw std::invalid_argument("empty pool");
  for (std::size_t d = 0; d < dim; ++d) {
    float maximum = -std::numeric_limits<float>::infinity();
    for (std::size_t t = 0; t < pool_size; ++t)
      maximum = std::max(maximum, gates[t * dim + d] + ape[t * dim + d]);
    double sum = 0, value = 0;
    for (std::size_t t = 0; t < pool_size; ++t) {
      const double e =
          std::exp(double(gates[t * dim + d] + ape[t * dim + d] - maximum));
      sum += e;
      value += e * keys[t * dim + d];
    }
    pooled[d] = static_cast<float>(value / sum);
  }
}
void IndexerScores(const float* iq, const float* pooled, const float* weights,
                   std::size_t heads, std::size_t dim, std::size_t pools,
                   float* scores) {
  if (!heads || !dim)
    throw std::invalid_argument("empty indexer shape");
  const double scale = 1.0 / std::sqrt(double(heads) * dim);
  for (std::size_t p = 0; p < pools; ++p) {
    double sum = 0;
    for (std::size_t h = 0; h < heads; ++h) {
      double dot = 0;
      for (std::size_t d = 0; d < dim; ++d)
        dot += double(iq[h * dim + d]) * pooled[p * dim + d];
      sum += weights[h] * scale * std::max(0.0, dot);
    }
    scores[p] = static_cast<float>(sum);
  }
}
Selection SelectPools(const float* scores, std::uint32_t visible_tokens,
                      std::uint32_t first_valid, std::uint32_t pool_size,
                      std::uint32_t pool_budget, bool select_tail) {
  if (!pool_size)
    throw std::invalid_argument("zero pool size");
  Selection result;
  if (visible_tokens <= first_valid)
    return result;
  const auto count = (visible_tokens - first_valid) / pool_size;
  result.pools.resize(count);
  std::iota(result.pools.begin(), result.pools.end(), 0);
  for (std::uint32_t p = 0; p < count; ++p)
    if (!std::isfinite(scores[p]))
      throw std::invalid_argument("nonfinite pool score");
  const auto keep = std::min(count, pool_budget);
  std::partial_sort(result.pools.begin(), result.pools.begin() + keep,
                    result.pools.end(), [&](auto a, auto b) {
                      return scores[a] == scores[b] ? a < b
                                                    : scores[a] > scores[b];
                    });
  result.pools.resize(keep);
  for (auto p : result.pools)
    for (std::uint32_t i = 0; i < pool_size; ++i)
      result.tokens.push_back(first_valid + p * pool_size + i);
  if (select_tail)
    for (auto t = first_valid + count * pool_size; t < visible_tokens; ++t)
      result.tokens.push_back(t);
  return result;
}
void AbsorbQuery(const float* q, const float* kb, std::size_t heads,
                 std::size_t qk_dim, std::size_t latent_dim, float* absorbed) {
  for (std::size_t h = 0; h < heads; ++h)
    for (std::size_t l = 0; l < latent_dim; ++l) {
      double dot = 0;
      for (std::size_t d = 0; d < qk_dim; ++d)
        dot +=
            double(q[h * qk_dim + d]) * kb[(h * latent_dim + l) * qk_dim + d];
      absorbed[h * latent_dim + l] = static_cast<float>(dot);
    }
}
void LatentAttention(const float* q, const float* latent, std::size_t rows,
                     std::size_t heads, std::size_t latent_dim,
                     std::size_t qk_dim,
                     const std::vector<std::int32_t>& selected,
                     float* context) {
  if (!qk_dim)
    throw std::invalid_argument("zero attention head dimension");
  for (auto row : selected)
    if (row < 0 || static_cast<std::size_t>(row) >= rows)
      throw std::out_of_range("attention selection outside cache");
  std::fill(context, context + heads * latent_dim, 0.0F);
  if (selected.empty())
    return;
  const double scale = 1.0 / std::sqrt(double(qk_dim));
  std::vector<double> probs(selected.size());
  for (std::size_t h = 0; h < heads; ++h) {
    double maximum = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < selected.size(); ++i) {
      double dot = 0;
      for (std::size_t d = 0; d < latent_dim; ++d)
        dot += double(q[h * latent_dim + d]) *
               latent[selected[i] * latent_dim + d];
      probs[i] = dot * scale;
      maximum = std::max(maximum, probs[i]);
    }
    double sum = 0;
    for (auto& p : probs) {
      p = std::exp(p - maximum);
      sum += p;
    }
    for (std::size_t d = 0; d < latent_dim; ++d) {
      double value = 0;
      for (std::size_t i = 0; i < selected.size(); ++i)
        value += probs[i] * latent[selected[i] * latent_dim + d];
      context[h * latent_dim + d] = static_cast<float>(value / sum);
    }
  }
}
void ProjectValues(const float* context, const float* vb, std::size_t heads,
                   std::size_t latent_dim, std::size_t value_dim, float* y) {
  for (std::size_t h = 0; h < heads; ++h)
    for (std::size_t v = 0; v < value_dim; ++v) {
      double dot = 0;
      for (std::size_t l = 0; l < latent_dim; ++l)
        dot += double(context[h * latent_dim + l]) *
               vb[(h * value_dim + v) * latent_dim + l];
      y[h * value_dim + v] = static_cast<float>(dot);
    }
}
}  // namespace gufo::models::glm53_flash::reference
