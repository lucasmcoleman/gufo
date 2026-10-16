#include "src/models/glm53_flash/reference_hc.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace gufo::models::glm53_flash::reference {
namespace {

double Sigmoid(double z) {
  return 1.0 / (1.0 + std::exp(-z));
}

}  // namespace

void HcPre(const Config& config, const float* fn, const float* base,
           const float* scale, const float* H, float* x_out, HcMix* mix_out) {
  const std::size_t hc = config.hc_count;
  const std::size_t hidden = config.hidden_size;
  const std::size_t hc_dim = hc * hidden;
  const std::size_t mix_dim = (2 * hc) + (hc * hc);
  if (hc == 0 || hidden == 0 || config.hc_sinkhorn_iters == 0) {
    throw std::invalid_argument("HcPre: empty hyper-connection config");
  }
  const double eps = config.hc_eps;

  // Unweighted RMSNorm over the flattened streams (HF:215), then fn (HF:303).
  double sumsq = 0.0;
  for (std::size_t k = 0; k < hc_dim; ++k) {
    sumsq += static_cast<double>(H[k]) * H[k];
  }
  const double inv_rms =
      1.0 / std::sqrt((sumsq / static_cast<double>(hc_dim)) + config.rms_eps);
  std::vector<double> m(mix_dim);
  for (std::size_t j = 0; j < mix_dim; ++j) {
    const float* row = fn + (j * hc_dim);
    double acc = 0.0;
    for (std::size_t k = 0; k < hc_dim; ++k) {
      acc += static_cast<double>(row[k]) * H[k];
    }
    m[j] = acc * inv_rms;
  }

  // HF:313-314.
  std::vector<double> pre(hc);
  std::vector<double> post(hc);
  for (std::size_t i = 0; i < hc; ++i) {
    pre[i] = Sigmoid((m[i] * scale[0]) + base[i]) + eps;
    post[i] = 2.0 * Sigmoid((m[hc + i] * scale[1]) + base[hc + i]);
  }

  // HF:315: softmax over the last (dst) axis of the [src][dst] view, + eps.
  std::vector<double> c(hc * hc);
  for (std::size_t r = 0; r < hc; ++r) {
    double mx = -INFINITY;
    for (std::size_t col = 0; col < hc; ++col) {
      const std::size_t idx = (r * hc) + col;
      c[idx] = (m[(2 * hc) + idx] * scale[2]) + base[(2 * hc) + idx];
      mx = std::max(mx, c[idx]);
    }
    double s = 0.0;
    for (std::size_t col = 0; col < hc; ++col) {
      c[(r * hc) + col] = std::exp(c[(r * hc) + col] - mx);
      s += c[(r * hc) + col];
    }
    for (std::size_t col = 0; col < hc; ++col) {
      c[(r * hc) + col] = (c[(r * hc) + col] / s) + eps;
    }
  }
  // HF:321-324: column norm (sum over dim -2, the src axis), then
  // iters-1 x {row norm, column norm}; eps is added to every denominator.
  auto norm_cols = [&] {
    for (std::size_t col = 0; col < hc; ++col) {
      double s = 0.0;
      for (std::size_t r = 0; r < hc; ++r)
        s += c[(r * hc) + col];
      for (std::size_t r = 0; r < hc; ++r)
        c[(r * hc) + col] /= s + eps;
    }
  };
  auto norm_rows = [&] {
    for (std::size_t r = 0; r < hc; ++r) {
      double s = 0.0;
      for (std::size_t col = 0; col < hc; ++col)
        s += c[(r * hc) + col];
      for (std::size_t col = 0; col < hc; ++col)
        c[(r * hc) + col] /= s + eps;
    }
  };
  norm_cols();
  for (std::uint32_t it = 1; it < config.hc_sinkhorn_iters; ++it) {
    norm_rows();
    norm_cols();
  }

  // HF:328: collapse with pre.
  for (std::size_t d = 0; d < hidden; ++d) {
    double acc = 0.0;
    for (std::size_t s = 0; s < hc; ++s)
      acc += pre[s] * H[(s * hidden) + d];
    x_out[d] = static_cast<float>(acc);
  }

  if (mix_out != nullptr) {
    mix_out->hc_count = static_cast<std::uint32_t>(hc);
    mix_out->hidden = static_cast<std::uint32_t>(hidden);
    mix_out->logits.assign(m.begin(), m.end());
    mix_out->pre.assign(pre.begin(), pre.end());
    mix_out->post.assign(post.begin(), post.end());
    mix_out->comb.assign(c.begin(), c.end());
  }
}

void HcPost(const HcMix& mix, const float* a, const float* H_old,
            float* H_new) {
  const std::size_t hc = mix.hc_count;
  const std::size_t hidden = mix.hidden;
  // HF:1336-1338: post[dst] * a + (comb^T @ residual)[dst].
  for (std::size_t dst = 0; dst < hc; ++dst) {
    for (std::size_t d = 0; d < hidden; ++d) {
      double acc = static_cast<double>(mix.post[dst]) * a[d];
      for (std::size_t src = 0; src < hc; ++src) {
        acc += static_cast<double>(mix.comb[(src * hc) + dst]) *
               H_old[(src * hidden) + d];
      }
      H_new[(dst * hidden) + d] = static_cast<float>(acc);
    }
  }
}

void HcHeadMean(const Config& config, const float* H, float* h_out) {
  const std::size_t hc = config.hc_count;
  const std::size_t hidden = config.hidden_size;
  for (std::size_t d = 0; d < hidden; ++d) {
    double acc = 0.0;
    for (std::size_t s = 0; s < hc; ++s)
      acc += H[(s * hidden) + d];
    h_out[d] = static_cast<float>(acc / static_cast<double>(hc));
  }
}

void HcInit(const Config& config, const float* embedding, float* H) {
  const std::size_t hidden = config.hidden_size;
  for (std::size_t s = 0; s < config.hc_count; ++s) {
    std::copy(embedding, embedding + hidden, H + (s * hidden));
  }
}

}  // namespace gufo::models::glm53_flash::reference
