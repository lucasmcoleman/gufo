#include "src/models/glm53_flash/reference_kda.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace gufo::models::glm53_flash::reference {
namespace {

double Sigmoid(double z) {
  return 1.0 / (1.0 + std::exp(-z));
}

// HF's softplus with its overflow guard (HF:367).
double Softplus(double z) {
  return z > 20.0 ? z : std::log1p(std::exp(z));
}

}  // namespace

void KdaConv(const float* x, const float* weight, float* history, float* out,
             std::uint32_t n_tokens, std::uint32_t channels,
             std::uint32_t kernel) {
  if (kernel == 0 || channels == 0) {
    throw std::invalid_argument("KdaConv: empty conv");
  }
  const std::size_t hist = kernel - 1;
  const std::size_t C = channels;
  // Input at position p of [history ; x] (p < hist is history).
  const auto input = [&](std::size_t p, std::size_t c) {
    return p < hist ? history[(p * C) + c] : x[((p - hist) * C) + c];
  };
  for (std::size_t t = 0; t < n_tokens; ++t) {
    for (std::size_t c = 0; c < C; ++c) {
      double acc = 0.0;
      for (std::size_t j = 0; j < kernel; ++j) {
        // Output t covers [history ; x] positions t .. t + kernel - 1.
        acc += static_cast<double>(weight[(c * kernel) + j]) * input(t + j, c);
      }
      out[(t * C) + c] = static_cast<float>(acc * Sigmoid(acc));
    }
  }
  // New history: the last `hist` positions of [history ; x], ascending
  // (each read is ahead of the row being written).
  const std::size_t total = hist + n_tokens;
  for (std::size_t r = 0; r < hist; ++r) {
    for (std::size_t c = 0; c < C; ++c) {
      history[(r * C) + c] = input(total - hist + r, c);
    }
  }
}

void KdaGate(const float* f, const float* dt_bias, const float* a,
             float lower_bound, float* g, std::uint32_t n_tokens,
             std::uint32_t heads, std::uint32_t key_dim) {
  const std::size_t width = std::size_t{heads} * key_dim;
  const bool bounded = std::isfinite(lower_bound);
  for (std::size_t t = 0; t < n_tokens; ++t) {
    for (std::size_t i = 0; i < width; ++i) {
      const double a_h = a[i / key_dim];  // -exp(A_log)
      const double z = static_cast<double>(f[(t * width) + i]) + dt_bias[i];
      const double v =
          bounded ? lower_bound * Sigmoid(-a_h * z) : a_h * Softplus(z);
      g[(t * width) + i] = static_cast<float>(v);
    }
  }
}

void KdaPrep(float* q, float* k, float* beta, std::uint32_t n_tokens,
             std::uint32_t heads, std::uint32_t key_dim, float eps) {
  const std::size_t rows = std::size_t{n_tokens} * heads;
  const double scale = 1.0 / std::sqrt(static_cast<double>(key_dim));
  for (std::size_t r = 0; r < rows; ++r) {
    float* qr = q + (r * key_dim);
    float* kr = k + (r * key_dim);
    double qs = 0.0;
    double ks = 0.0;
    for (std::size_t i = 0; i < key_dim; ++i) {
      qs += static_cast<double>(qr[i]) * qr[i];
      ks += static_cast<double>(kr[i]) * kr[i];
    }
    const double qn = scale / std::sqrt(qs + eps);
    const double kn = 1.0 / std::sqrt(ks + eps);
    for (std::size_t i = 0; i < key_dim; ++i) {
      qr[i] = static_cast<float>(qr[i] * qn);
      kr[i] = static_cast<float>(kr[i] * kn);
    }
    if (beta != nullptr) {
      beta[r] = static_cast<float>(Sigmoid(beta[r]));
    }
  }
}

void KdaRecurrent(const float* q, const float* k, const float* v,
                  const float* g, const float* beta, float* state, float* o,
                  std::uint32_t n_tokens, std::uint32_t heads,
                  std::uint32_t key_dim, std::uint32_t value_dim) {
  const std::size_t Dk = key_dim;
  const std::size_t Dv = value_dim;
  const std::size_t H = heads;
  std::vector<double> u(Dv);
  std::vector<double> decay(Dk);
  for (std::size_t t = 0; t < n_tokens; ++t) {
    for (std::size_t h = 0; h < H; ++h) {
      const std::size_t row = (t * H) + h;
      const float* qh = q + (row * Dk);
      const float* kh = k + (row * Dk);
      const float* vh = v + (row * Dv);
      const float* gh = g + (row * Dk);
      const double b = beta[row];
      float* S = state + (h * Dk * Dv);  // S[i][j], i key, j value
      // S <- diag(exp g) S: key row i scales by exp(g[i]) (HF:504-506).
      for (std::size_t i = 0; i < Dk; ++i) {
        decay[i] = std::exp(static_cast<double>(gh[i]));
      }
      for (std::size_t i = 0; i < Dk; ++i) {
        for (std::size_t j = 0; j < Dv; ++j) {
          S[(i * Dv) + j] = static_cast<float>(S[(i * Dv) + j] * decay[i]);
        }
      }
      // u = S^T k (HF:507).
      for (std::size_t j = 0; j < Dv; ++j) {
        u[j] = 0.0;
      }
      for (std::size_t i = 0; i < Dk; ++i) {
        for (std::size_t j = 0; j < Dv; ++j) {
          u[j] += static_cast<double>(S[(i * Dv) + j]) * kh[i];
        }
      }
      // delta = beta (v - u); S += k delta^T (HF:508-510).
      for (std::size_t j = 0; j < Dv; ++j) {
        u[j] = (vh[j] - u[j]) * b;
      }
      for (std::size_t i = 0; i < Dk; ++i) {
        for (std::size_t j = 0; j < Dv; ++j) {
          S[(i * Dv) + j] =
              static_cast<float>(S[(i * Dv) + j] + (kh[i] * u[j]));
        }
      }
      // o = S^T q (HF:511).
      float* oh = o + (row * Dv);
      for (std::size_t j = 0; j < Dv; ++j) {
        double acc = 0.0;
        for (std::size_t i = 0; i < Dk; ++i) {
          acc += static_cast<double>(S[(i * Dv) + j]) * qh[i];
        }
        oh[j] = static_cast<float>(acc);
      }
    }
  }
}

void KdaOutput(const float* o, const float* gate, const float* weight,
               float eps, float* out, std::uint32_t n_tokens,
               std::uint32_t heads, std::uint32_t value_dim) {
  const std::size_t rows = std::size_t{n_tokens} * heads;
  const std::size_t Dv = value_dim;
  for (std::size_t r = 0; r < rows; ++r) {
    const float* src = o + (r * Dv);
    double ss = 0.0;
    for (std::size_t j = 0; j < Dv; ++j) {
      ss += static_cast<double>(src[j]) * src[j];
    }
    const double inv = 1.0 / std::sqrt((ss / static_cast<double>(Dv)) + eps);
    for (std::size_t j = 0; j < Dv; ++j) {
      const double gated = static_cast<double>(src[j]) * inv * weight[j] *
                           Sigmoid(gate[(r * Dv) + j]);
      out[(r * Dv) + j] = static_cast<float>(gated);
    }
  }
}

}  // namespace gufo::models::glm53_flash::reference
