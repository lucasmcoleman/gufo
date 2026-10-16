#include "src/models/glm53_flash/reference_moe.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace gufo::models::glm53_flash::reference {
namespace {
float Sigmoid(float x) {
  if (x >= 0.0F)
    return 1.0F / (1.0F + std::exp(-x));
  const float e = std::exp(x);
  return e / (1.0F + e);
}
float Dot(const float* a, const float* b, std::size_t n) {
  double sum = 0;
  for (std::size_t i = 0; i < n; ++i)
    sum += double(a[i]) * b[i];
  return static_cast<float>(sum);
}
}  // namespace

Routing Route(const float* logits, const float* bias, std::uint32_t experts,
              std::uint32_t used, float scale, bool normalize,
              RouterDenominator denominator) {
  if (experts == 0 || used == 0 || used > experts)
    throw std::invalid_argument("invalid router expert count");
  std::vector<float> scores(experts), choice(experts);
  std::vector<std::uint32_t> ids(experts);
  std::iota(ids.begin(), ids.end(), 0U);
  for (std::uint32_t e = 0; e < experts; ++e) {
    scores[e] = Sigmoid(logits[e]);
    choice[e] = scores[e] + (bias ? bias[e] : 0.0F);
    if (!std::isfinite(choice[e]))
      throw std::invalid_argument("nonfinite router score");
  }
  std::partial_sort(
      ids.begin(), ids.begin() + used, ids.end(), [&](auto a, auto b) {
        return choice[a] == choice[b] ? a < b : choice[a] > choice[b];
      });
  ids.resize(used);
  float sum = 0;
  for (auto e : ids)
    sum += scores[e];
  float denom = 1.0F;
  if (normalize) {
    denom = denominator == RouterDenominator::kLlamaClamp
                ? std::max(sum, 0x1p-14F)
                : sum + 1e-20F;
  }
  Routing result{ids, std::vector<float>(used)};
  for (std::uint32_t i = 0; i < used; ++i)
    result.weights[i] = (scores[ids[i]] / denom) * scale;
  return result;
}

float ClampedSwiGlu(float gate, float up, float clamp) {
  if (clamp > 0) {
    gate = std::min(gate, clamp);
    up = std::clamp(up, -clamp, clamp);
  }
  return gate * Sigmoid(gate) * up;
}

void DenseFfn(const float* x, const float* gate, const float* up,
              const float* down, std::size_t hidden, std::size_t intermediate,
              float clamp, float* y) {
  std::vector<float> mid(intermediate);
  for (std::size_t i = 0; i < intermediate; ++i)
    mid[i] = ClampedSwiGlu(Dot(gate + i * hidden, x, hidden),
                           Dot(up + i * hidden, x, hidden), clamp);
  for (std::size_t i = 0; i < hidden; ++i)
    y[i] = Dot(down + i * intermediate, mid.data(), intermediate);
}

void RoutedFfn(const float* x, const float* gate_up, const float* down,
               std::size_t hidden, std::size_t intermediate,
               const Routing& routing, float clamp, float* y) {
  if (routing.ids.size() != routing.weights.size())
    throw std::invalid_argument("routing weights/ids size mismatch");
  std::vector<float> expert_y(hidden);
  std::fill(y, y + hidden, 0.0F);
  for (std::size_t i = 0; i < routing.ids.size(); ++i) {
    const auto e = routing.ids[i];
    const float* gate = gate_up + e * 2 * intermediate * hidden;
    DenseFfn(x, gate, gate + intermediate * hidden,
             down + e * hidden * intermediate, hidden, intermediate, clamp,
             expert_y.data());
    for (std::size_t d = 0; d < hidden; ++d)
      y[d] += routing.weights[i] * expert_y[d];
  }
}
}  // namespace gufo::models::glm53_flash::reference
