// Test-only C ABI for independent NPZ fixture comparisons.
#include "src/models/glm53_flash/reference_moe.hpp"
using namespace gufo::models::glm53_flash::reference;
extern "C" {
void glm_route(const float* logits, const float* bias, int experts, int used,
               int hf, int* ids, float* weights) {
  auto r = Route(
      logits, bias, experts, used, 2.5F, true,
      hf ? RouterDenominator::kHfEpsilon : RouterDenominator::kLlamaClamp);
  for (int i = 0; i < used; ++i) {
    ids[i] = static_cast<int>(r.ids[i]);
    weights[i] = r.weights[i];
  }
}
void glm_dense(const float* x, const float* gate, const float* up,
               const float* down, int hidden, int intermediate, float* y) {
  DenseFfn(x, gate, up, down, hidden, intermediate, 10.0F, y);
}
void glm_swiglu(const float* gu, int intermediate, float* y) {
  for (int i = 0; i < intermediate; ++i)
    y[i] = ClampedSwiGlu(gu[i], gu[i + intermediate], 10.0F);
}
void glm_routed(const float* x, const float* gu, const float* down, int hidden,
                int intermediate, const int* ids, const float* weights,
                int used, float* y) {
  Routing r;
  for (int i = 0; i < used; ++i) {
    r.ids.push_back(ids[i]);
    r.weights.push_back(weights[i]);
  }
  RoutedFfn(x, gu, down, hidden, intermediate, r, 10.0F, y);
}
}

#include "src/models/glm53_flash/reference_mla.hpp"
extern "C" {
void glm_norm(const float* x, const float* w, const float* bias, int d,
              float* y) {
  LayerNorm(x, w, bias, d, 1e-6F, y);
}
void glm_pool(const float* k, const float* g, const float* ape, int pool, int d,
              float* y) {
  PoolKeys(k, g, ape, pool, d, y);
}
void glm_scores(const float* iq, const float* pk, const float* w, int h, int d,
                int pools, float* y) {
  IndexerScores(iq, pk, w, h, d, pools, y);
}
int glm_select(const float* scores, int visible, int first, int pool,
               int budget, int tail, int* ids) {
  auto s = SelectPools(scores, visible, first, pool, budget, tail != 0);
  for (std::size_t i = 0; i < s.tokens.size(); ++i)
    ids[i] = s.tokens[i];
  return static_cast<int>(s.tokens.size());
}
}
extern "C" {
void glm_absorb(const float* q, const float* kb, int heads, int qk, int latent,
                float* y) {
  AbsorbQuery(q, kb, heads, qk, latent, y);
}
void glm_attention(const float* q, const float* latent, int rows, int heads,
                   int dim, int qk, const int* ids, int count, float* y) {
  std::vector<std::int32_t> selected(ids, ids + count);
  LatentAttention(q, latent, rows, heads, dim, qk, selected, y);
}
void glm_values(const float* ctx, const float* vb, int heads, int latent,
                int value, float* y) {
  ProjectValues(ctx, vb, heads, latent, value, y);
}
}
