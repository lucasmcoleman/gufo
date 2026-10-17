// GPU MLA + DSA indexer operator (layer_ops.hpp MlaForward) against the CPU
// oracle, which is ReferenceModel::Mla's math (reference.cpp) on
// reference_mla.hpp, and against the independent HF fixture.
//
//   mla_ops_test                    fixture + synthetic checks (ctest)
//   mla_ops_test real-cpu MODEL DIR ReferenceModel on layers 0..7 of a real
//                                   GGUF; dumps the MLA layers' input and
//                                   output (layers 3 and 7) to DIR
//   mla_ops_test real-gpu MODEL DIR GPU MlaForward on those dumps
//
// Checks: per-token rel-L2 <= 1e-5 against the oracle / HF, identical
// selected token sets, and bit-identical outputs for any chunking of the same
// sequence (one prefill, uneven prefills, decode steps, scratch-limited
// passes). The synthetic long case has more than top_k raw tokens visible so
// the indexer drops pools. The fixture (mla_attention.npz, HF 09281b8) comes
// from GLM53_LANE_B_FIXTURE_DIR; without it that part is reported skipped.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <random>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/glm53_flash/config.hpp"
#include "src/models/glm53_flash/executor.hpp"
#include "src/models/glm53_flash/layer_ops.hpp"
#include "src/models/glm53_flash/reference.hpp"
#include "src/models/glm53_flash/reference_mla.hpp"
#include "src/models/glm53_flash/weights.hpp"
#include "tests/models/glm53_flash/host_ops.hpp"

namespace glm = gufo::models::glm53_flash;
namespace ref = gufo::models::glm53_flash::reference;

namespace {

constexpr double kTol = 1e-5;

void Hip(hipError_t e, const char* what) {
  if (e != hipSuccess) {
    throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
  }
}

double RelL2(const float* got, const float* want, std::size_t n) {
  double num = 0.0;
  double den = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double d = static_cast<double>(got[i]) - want[i];
    num += d * d;
    den += static_cast<double>(want[i]) * want[i];
  }
  return den == 0.0 ? std::sqrt(num) : std::sqrt(num / den);
}

// ---- NPZ (stored zip of .npy) ----

struct Array {
  std::vector<std::size_t> shape;
  std::vector<float> data;  ///< f4 as is; b1 / i4 / i8 converted
};

std::uint32_t Le(const std::vector<char>& b, std::size_t at, int bytes) {
  std::uint32_t v = 0;
  for (int i = bytes - 1; i >= 0; --i) {
    v = (v << 8) | static_cast<unsigned char>(b.at(at + i));
  }
  return v;
}

std::map<std::string, Array> LoadNpz(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("cannot open " + path);
  }
  const std::vector<char> b((std::istreambuf_iterator<char>(in)),
                            std::istreambuf_iterator<char>());
  std::size_t eocd = b.size() < 22 ? 0 : b.size() - 22;
  while (eocd > 0 && Le(b, eocd, 4) != 0x06054b50U) {
    --eocd;
  }
  if (Le(b, eocd, 4) != 0x06054b50U) {
    throw std::runtime_error("no zip directory in " + path);
  }
  const std::uint32_t entries = Le(b, eocd + 10, 2);
  std::size_t at = Le(b, eocd + 16, 4);
  std::map<std::string, Array> out;
  for (std::uint32_t e = 0; e < entries; ++e) {
    if (Le(b, at, 4) != 0x02014b50U || Le(b, at + 10, 2) != 0) {
      throw std::runtime_error("unsupported zip entry in " + path);
    }
    const std::uint32_t size = Le(b, at + 20, 4);
    const std::uint32_t nlen = Le(b, at + 28, 2);
    const std::uint32_t elen = Le(b, at + 30, 2);
    const std::uint32_t clen = Le(b, at + 32, 2);
    const std::uint32_t local = Le(b, at + 42, 4);
    std::string name(b.data() + at + 46, nlen);
    at += 46 + nlen + elen + clen;
    const std::size_t data =
        local + 30 + Le(b, local + 26, 2) + Le(b, local + 28, 2);
    if (std::memcmp(b.data() + data, "\x93NUMPY", 6) != 0) {
      throw std::runtime_error("not an npy: " + name);
    }
    const int major = static_cast<unsigned char>(b[data + 6]);
    const std::size_t hlen =
        major == 1 ? Le(b, data + 8, 2) : Le(b, data + 8, 4);
    const std::size_t hstart = data + (major == 1 ? 10 : 12);
    const std::string header(b.data() + hstart, hlen);
    const std::size_t payload = hstart + hlen;
    const auto descr_at = header.find("'descr': '") + 10;
    const std::string descr = header.substr(descr_at, 3);
    if (header.find("'fortran_order': False") == std::string::npos) {
      throw std::runtime_error("fortran order npy: " + name);
    }
    Array a;
    const auto open = header.find('(', header.find("'shape'"));
    const auto close = header.find(')', open);
    std::string dims = header.substr(open + 1, close - open - 1);
    std::size_t count = 1;
    for (std::size_t p = 0; p < dims.size();) {
      const auto comma = dims.find(',', p);
      const std::string tok = dims.substr(p, comma - p);
      if (tok.find_first_of("0123456789") != std::string::npos) {
        a.shape.push_back(std::stoul(tok));
        count *= a.shape.back();
      }
      if (comma == std::string::npos) {
        break;
      }
      p = comma + 1;
    }
    a.data.resize(count);
    const char* src = b.data() + payload;
    for (std::size_t i = 0; i < count; ++i) {
      if (descr == "<f4") {
        std::memcpy(&a.data[i], src + (4 * i), 4);
      } else if (descr == "|b1") {
        a.data[i] = src[i] != 0 ? 1.0F : 0.0F;
      } else if (descr == "<i4") {
        std::int32_t v = 0;
        std::memcpy(&v, src + (4 * i), 4);
        a.data[i] = static_cast<float>(v);
      } else if (descr == "<i8") {
        std::int64_t v = 0;
        std::memcpy(&v, src + (8 * i), 8);
        a.data[i] = static_cast<float>(v);
      } else {
        throw std::runtime_error("unsupported dtype " + descr + " in " + name);
      }
    }
    (void)size;
    if (name.size() > 4 && name.ends_with(".npy")) {
      name.resize(name.size() - 4);
    }
    out.emplace(std::move(name), std::move(a));
  }
  return out;
}

// ---- Host weights and the CPU oracle ----

struct HostLayer {
  std::vector<float> q_a, q_b, kv_a, k_b, v_b, out;
  std::vector<float> idx_q_b, idx_k, idx_proj, idx_gate;
  std::vector<float> q_a_norm, kv_a_norm, k_norm, k_norm_bias, ape;
};

/// ReferenceModel::Mla (reference.cpp) with the state kept here; also
/// records each token's selected raw positions.
class CpuMla {
public:
  CpuMla(const glm::Config& c, const HostLayer& w) : c_(c), w_(w) {}

  void Run(const float* x, std::uint32_t n, float* y) {
    namespace so = glm::stand_in;
    const std::size_t hidden = c_.hidden_size;
    const std::size_t heads = c_.num_heads;
    const std::size_t q_lora = c_.q_lora_rank;
    const std::size_t kv_lora = c_.kv_lora_rank;
    const std::size_t nope = c_.mla_head_dim;
    const std::size_t v_dim = c_.mla_value_dim;
    const std::size_t ih = c_.indexer_heads;
    const std::size_t idim = c_.indexer_head_dim;
    const std::size_t kpool = c_.indexer_kpool;
    std::vector<float> tmp(std::max({q_lora, kv_lora, idim}));
    std::vector<float> q_resid(q_lora), q(heads * nope), q_abs(heads * kv_lora),
        ctx(heads * kv_lora), attn(heads * v_dim), iq(ih * idim), iw(ih),
        key(idim), pooled(idim), scores;
    for (std::uint32_t t = 0; t < n; ++t) {
      const float* xt = x + (t * hidden);
      const std::uint32_t pos = pos_ + t;
      so::MatVec(w_.q_a, hidden, xt, tmp.data());
      so::RmsNorm(tmp.data(), w_.q_a_norm.data(), q_lora, c_.rms_eps,
                  q_resid.data());
      so::MatVec(w_.q_b, q_lora, q_resid.data(), q.data());
      so::MatVec(w_.kv_a, hidden, xt, tmp.data());
      latent_.resize((std::size_t{pos} + 1) * kv_lora);
      so::RmsNorm(tmp.data(), w_.kv_a_norm.data(), kv_lora, c_.rms_eps,
                  latent_.data() + (std::size_t{pos} * kv_lora));
      so::MatVec(w_.idx_k, hidden, xt, tmp.data());
      ref::LayerNorm(tmp.data(), w_.k_norm.data(), w_.k_norm_bias.data(), idim,
                     c_.norm_eps, key.data());
      tail_keys_.insert(tail_keys_.end(), key.begin(), key.end());
      so::MatVec(w_.idx_gate, hidden, xt, tmp.data());
      tail_gates_.insert(tail_gates_.end(), tmp.begin(),
                         tmp.begin() + static_cast<std::ptrdiff_t>(idim));
      if (tail_keys_.size() == kpool * idim) {
        ref::PoolKeys(tail_keys_.data(), tail_gates_.data(), w_.ape.data(),
                      kpool, idim, pooled.data());
        pooled_.insert(pooled_.end(), pooled.begin(), pooled.end());
        tail_keys_.clear();
        tail_gates_.clear();
      }
      so::MatVec(w_.idx_q_b, q_lora, q_resid.data(), iq.data());
      so::MatVec(w_.idx_proj, hidden, xt, iw.data());
      const std::size_t pools = pooled_.size() / idim;
      scores.assign(pools, 0.0F);
      ref::IndexerScores(iq.data(), pooled_.data(), iw.data(), ih, idim, pools,
                         scores.data());
      const ref::Selection sel =
          ref::SelectPools(scores.data(), pos + 1, 0, c_.indexer_kpool,
                           c_.IndexerPoolBudget(), /*select_tail=*/true);
      selections.push_back(sel.tokens);
      ref::AbsorbQuery(q.data(), w_.k_b.data(), heads, nope, kv_lora,
                       q_abs.data());
      ref::LatentAttention(q_abs.data(), latent_.data(), std::size_t{pos} + 1,
                           heads, kv_lora, nope, sel.tokens, ctx.data());
      ref::ProjectValues(ctx.data(), w_.v_b.data(), heads, kv_lora, v_dim,
                         attn.data());
      so::MatVec(w_.out, heads * v_dim, attn.data(), y + (t * hidden));
    }
    pos_ += n;
  }

  std::vector<std::vector<std::int32_t>> selections;

private:
  const glm::Config& c_;
  const HostLayer& w_;
  std::uint32_t pos_{0};
  std::vector<float> latent_, pooled_, tail_keys_, tail_gates_;
};

// ---- Device side ----

glm::DeviceMatrix UploadF32(glm::DeviceArena& arena,
                            const std::vector<float>& v, std::uint64_t cols,
                            std::uint64_t rows, std::uint64_t experts = 1) {
  if (v.size() != cols * rows * experts) {
    throw std::runtime_error("UploadF32: size mismatch");
  }
  void* p = arena.Allocate(v.size() * sizeof(float));
  if (p == nullptr) {
    throw std::runtime_error("device allocation failed");
  }
  Hip(hipMemcpy(p, v.data(), v.size() * sizeof(float), hipMemcpyHostToDevice),
      "upload");
  return glm::DeviceMatrix{.data = p,
                           .type = gufo::core::GgmlType::kF32,
                           .cols = cols,
                           .rows = rows,
                           .experts = experts,
                           .row_bytes = cols * sizeof(float)};
}

const float* UploadVec(glm::DeviceArena& arena, const std::vector<float>& v) {
  return static_cast<const float*>(UploadF32(arena, v, v.size(), 1).data);
}

glm::MlaLayerDevice Upload(glm::DeviceArena& a, const glm::Config& c,
                           const HostLayer& h) {
  const std::uint64_t hidden = c.hidden_size;
  const std::uint64_t heads = c.num_heads;
  glm::MlaLayerDevice w;
  w.q_a = UploadF32(a, h.q_a, hidden, c.q_lora_rank);
  w.q_b = UploadF32(a, h.q_b, c.q_lora_rank, heads * c.mla_head_dim);
  w.kv_a = UploadF32(a, h.kv_a, hidden, c.kv_lora_rank);
  w.k_b = UploadF32(a, h.k_b, c.mla_head_dim, c.kv_lora_rank, heads);
  w.v_b = UploadF32(a, h.v_b, c.kv_lora_rank, c.mla_value_dim, heads);
  w.out = UploadF32(a, h.out, heads * c.mla_value_dim, hidden);
  w.idx_q_b = UploadF32(a, h.idx_q_b, c.q_lora_rank,
                        std::uint64_t{c.indexer_heads} * c.indexer_head_dim);
  w.idx_k = UploadF32(a, h.idx_k, hidden, c.indexer_head_dim);
  w.idx_proj = UploadF32(a, h.idx_proj, hidden, c.indexer_heads);
  w.idx_gate = UploadF32(a, h.idx_gate, hidden, c.indexer_head_dim);
  w.q_a_norm = UploadVec(a, h.q_a_norm);
  w.kv_a_norm = UploadVec(a, h.kv_a_norm);
  w.idx_k_norm = UploadVec(a, h.k_norm);
  w.idx_k_norm_bias = UploadVec(a, h.k_norm_bias);
  w.idx_ape = UploadVec(a, h.ape);
  return w;
}

/// One device sequence: cache, scratch, x/y buffers, and the run helper.
class GpuSeq {
public:
  GpuSeq(const glm::Config& c, const glm::MlaLayerDevice& w,
         std::uint32_t layer, std::uint32_t capacity,
         std::uint32_t scratch_tokens)
      : c_(c), w_(w), layer_(layer) {
    Hip(hipStreamCreate(&stream_), "stream");
    cache_.capacity = capacity;
    cache_.bytes = glm::MlaCacheBytes(c, capacity);
    cache_.data = arena_.Allocate(cache_.bytes);
    scratch_bytes_ = glm::MlaScratchBytes(c, scratch_tokens);
    scratch_ = arena_.Allocate(scratch_bytes_);
    const std::size_t io = std::size_t{capacity} * c.hidden_size;
    x_ = static_cast<float*>(arena_.Allocate(io * sizeof(float)));
    y_ = static_cast<float*>(arena_.Allocate(io * sizeof(float)));
    if (cache_.data == nullptr || scratch_ == nullptr || x_ == nullptr ||
        y_ == nullptr) {
      throw std::runtime_error("device allocation failed");
    }
    // Garbage, so nothing relies on a zeroed cache.
    Hip(hipMemset(cache_.data, 0x7F, cache_.bytes), "memset cache");
  }
  ~GpuSeq() { (void)hipStreamDestroy(stream_); }
  GpuSeq(const GpuSeq&) = delete;
  GpuSeq& operator=(const GpuSeq&) = delete;

  /// Runs x[0..T) in calls of `chunks` (the rest as decode steps); returns
  /// y [T][hidden]. With `top_k` non-null collects the indexer_top_k tap
  /// rows per token.
  std::vector<float> Run(
      const std::vector<float>& x, std::uint32_t T,
      const std::vector<std::uint32_t>& chunks,
      std::vector<std::vector<std::int32_t>>* top_k = nullptr) {
    const std::size_t hidden = c_.hidden_size;
    Hip(hipMemcpy(x_, x.data(), std::size_t{T} * hidden * sizeof(float),
                  hipMemcpyHostToDevice),
        "x");
    Hip(hipMemset(y_, 0xFF, std::size_t{T} * hidden * sizeof(float)), "y");
    const std::uint32_t budget = c_.IndexerPoolBudget();
    std::uint32_t pass_pos = 0;
    const glm::TapFn tap = [&](std::string_view name, std::uint32_t,
                               const glm::TapView& v) {
      if (name != "indexer_top_k" || top_k == nullptr) {
        return;
      }
      std::vector<std::int32_t> rows(v.count());
      Hip(hipMemcpyAsync(rows.data(), v.device, rows.size() * 4,
                         hipMemcpyDeviceToHost, stream_),
          "tap copy");
      Hip(hipStreamSynchronize(stream_), "tap sync");
      for (std::uint64_t t = 0; t < v.ne1; ++t) {
        std::vector<std::int32_t> r(rows.begin() + (t * budget),
                                    rows.begin() + ((t + 1) * budget));
        (*top_k)[pass_pos + t] = std::move(r);
      }
      pass_pos += static_cast<std::uint32_t>(v.ne1);
    };
    if (top_k != nullptr) {
      top_k->assign(T, {});
    }
    const glm::OpContext ctx{.stream = stream_,
                             .scratch = scratch_,
                             .scratch_bytes = scratch_bytes_,
                             .tap = top_k != nullptr ? &tap : nullptr};
    std::uint32_t pos = 0;
    std::size_t ci = 0;
    while (pos < T) {
      const std::uint32_t n =
          ci < chunks.size() ? std::min(chunks[ci++], T - pos) : 1;
      if (!glm::MlaForward(ctx, c_, layer_, w_, cache_, x_ + (pos * hidden), n,
                           pos, y_ + (pos * hidden))) {
        throw std::runtime_error("MlaForward returned false at pos " +
                                 std::to_string(pos));
      }
      pos += n;
    }
    Hip(hipStreamSynchronize(stream_), "sync");
    std::vector<float> y(std::size_t{T} * hidden);
    Hip(hipMemcpy(y.data(), y_, y.size() * sizeof(float),
                  hipMemcpyDeviceToHost),
        "y");
    return y;
  }

  [[nodiscard]] std::size_t scratch_bytes() const { return scratch_bytes_; }

private:
  const glm::Config& c_;
  const glm::MlaLayerDevice& w_;
  std::uint32_t layer_;
  glm::DeviceArena arena_;
  hipStream_t stream_{};
  glm::MlaCache cache_;
  void* scratch_{nullptr};
  std::size_t scratch_bytes_{0};
  float* x_{nullptr};
  float* y_{nullptr};
};

/// Raw positions selected for token `pos` from its indexer_top_k row.
std::vector<std::int32_t> TokensOf(const glm::Config& c,
                                   const std::vector<std::int32_t>& row,
                                   std::uint32_t pos) {
  const std::uint32_t kpool = c.indexer_kpool;
  const std::uint32_t pools = (pos + 1) / kpool;
  std::vector<std::int32_t> out;
  for (const std::int32_t p : row) {
    if (p < 0) {
      continue;
    }
    for (std::uint32_t i = 0; i < kpool; ++i) {
      out.push_back((p * static_cast<std::int32_t>(kpool)) +
                    static_cast<std::int32_t>(i));
    }
  }
  for (std::uint32_t t = pools * kpool; t <= pos; ++t) {
    out.push_back(static_cast<std::int32_t>(t));
  }
  std::sort(out.begin(), out.end());
  return out;
}

struct Verdict {
  bool pass{true};
  void Check(bool ok, const std::string& what) {
    std::cout << (ok ? "ok   " : "FAIL ") << what << '\n';
    pass = pass && ok;
  }
};

/// Per-token max rel-L2 of got vs want (and the token it occurs at).
double MaxTokenRel(const std::vector<float>& got,
                   const std::vector<float>& want, std::size_t width,
                   std::uint32_t T, std::uint32_t* at) {
  double worst = 0.0;
  for (std::uint32_t t = 0; t < T; ++t) {
    const double r =
        RelL2(got.data() + (t * width), want.data() + (t * width), width);
    if (!(r <= worst)) {
      worst = r;
      if (at != nullptr) {
        *at = t;
      }
    }
  }
  return worst;
}

std::string Sci(double v) {
  char b[32];
  std::snprintf(b, sizeof b, "%.3e", v);
  return b;
}

bool Same(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() &&
         std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

glm::Config TinyConfig(std::uint32_t hidden, std::uint32_t heads,
                       std::uint32_t q_lora, std::uint32_t kv_lora,
                       std::uint32_t nope, std::uint32_t v_dim,
                       std::uint32_t ih, std::uint32_t idim,
                       std::uint32_t top_k, std::uint32_t kpool) {
  glm::Config c;
  c.num_layers = 1;
  c.num_layers_all = 1;
  c.hidden_size = hidden;
  c.rms_eps = 1e-5F;
  c.norm_eps = 1e-6F;
  c.mixers = {glm::Config::Mixer::kMla};
  c.num_heads = heads;
  c.q_lora_rank = q_lora;
  c.kv_lora_rank = kv_lora;
  c.mla_head_dim = nope;
  c.mla_value_dim = v_dim;
  c.indexer_heads = ih;
  c.indexer_head_dim = idim;
  c.indexer_top_k = top_k;
  c.indexer_kpool = kpool;
  return c;
}

// ---- Fixture ----

bool FixtureCheck(Verdict& v) {
  std::string dir;
  if (const char* env = std::getenv("GLM53_LANE_B_FIXTURE_DIR");
      env != nullptr && *env != '\0') {
    dir = env;
  }
#ifdef GLM53_LANE_B_FIXTURE_DIR
  if (dir.empty()) {
    dir = GLM53_LANE_B_FIXTURE_DIR;
  }
#endif
  const std::string path = dir + "/mla_attention.npz";
  if (dir.empty() || !std::ifstream(path)) {
    std::cout << "SKIP fixture: no mla_attention.npz (GLM53_LANE_B_FIXTURE_DIR)"
              << '\n';
    return false;
  }
  const auto z = LoadNpz(path);
  const auto get = [&](const std::string& k) -> const Array& {
    const auto it = z.find(k);
    if (it == z.end()) {
      throw std::runtime_error("fixture lacks " + k);
    }
    return it->second;
  };
  // fixtures.mla_attention.config in the manifest.
  const glm::Config c = TinyConfig(32, 3, 20, 12, 10, 6, 4, 8, 8, 4);
  HostLayer h;
  h.q_a = get("w.q_a_proj").data;
  h.q_a_norm = get("w.q_a_layernorm").data;
  h.q_b = get("w.q_b_proj").data;
  h.kv_a = get("w.kv_a_proj_with_mqa").data;
  h.kv_a_norm = get("w.kv_a_layernorm").data;
  // torch k_b [H][nope][kv_lora] -> GGUF attn_k_b [H][kv_lora][nope].
  const Array& kb = get("w.k_b");
  h.k_b.resize(kb.data.size());
  for (std::size_t hh = 0; hh < 3; ++hh)
    for (std::size_t d = 0; d < 10; ++d)
      for (std::size_t l = 0; l < 12; ++l)
        h.k_b[(hh * 12 + l) * 10 + d] = kb.data[(hh * 10 + d) * 12 + l];
  h.v_b = get("w.v_b").data;
  h.out = get("w.o_proj").data;
  h.idx_q_b = get("w.indexer.wq_b.weight").data;
  h.idx_k = get("w.indexer.wk.weight").data;
  h.idx_proj = get("w.indexer.weights_proj.weight").data;
  h.idx_gate = get("w.indexer.index_kpool_compress_gate").data;
  h.k_norm = get("w.indexer.k_norm.weight").data;
  h.k_norm_bias = get("w.indexer.k_norm.bias").data;
  h.ape = get("w.indexer.index_kpool_compress_ape").data;
  glm::DeviceArena arena;
  const glm::MlaLayerDevice w = Upload(arena, c, h);
  for (const std::string kase : {"dense", "sparse"}) {
    const Array& x = get(kase + ".x");
    const Array& want = get(kase + ".y");
    const Array& topk = get(kase + ".topk_indices");
    const auto T = static_cast<std::uint32_t>(x.shape[1]);
    CpuMla cpu(c, h);
    std::vector<float> y_cpu(want.data.size());
    cpu.Run(x.data.data(), T, y_cpu.data());
    GpuSeq seq(c, w, 0, 32, T);
    std::vector<std::vector<std::int32_t>> top;
    const auto y = seq.Run(x.data, T, {T}, &top);
    std::uint32_t at = 0;
    const double r_hf = MaxTokenRel(y, want.data, 32, T, &at);
    const double r_cpu = MaxTokenRel(y, y_cpu, 32, T, nullptr);
    const double r_cpu_hf = MaxTokenRel(y_cpu, want.data, 32, T, nullptr);
    char line[256];
    std::snprintf(line, sizeof line,
                  "fixture %s T=%u: GPU vs HF max token rel %.3e (token %u), "
                  "GPU vs CPU %.3e, CPU vs HF %.3e",
                  kase.c_str(), T, r_hf, at, r_cpu, r_cpu_hf);
    v.Check(r_hf <= kTol && r_cpu <= kTol, line);
    // Selected token sets vs HF topk_indices (-1 padded) and the oracle.
    bool sets = true;
    std::uint32_t dropped = 0;
    const std::size_t width = topk.shape[2];
    for (std::uint32_t t = 0; t < T; ++t) {
      std::vector<std::int32_t> hf;
      for (std::size_t i = 0; i < width; ++i) {
        const auto id = static_cast<std::int32_t>(topk.data[t * width + i]);
        if (id >= 0) {
          hf.push_back(id);
        }
      }
      std::sort(hf.begin(), hf.end());
      const auto gpu = TokensOf(c, top[t], t);
      auto cpu_sel = cpu.selections[t];
      std::sort(cpu_sel.begin(), cpu_sel.end());
      sets = sets && gpu == hf && gpu == cpu_sel;
      dropped += t + 1 - static_cast<std::uint32_t>(gpu.size());
    }
    v.Check(sets, "fixture " + kase + ": selected token sets == HF == CPU (" +
                      std::to_string(dropped) + " causal positions dropped)");
    if (kase == "sparse") {
      const auto y_split = seq.Run(x.data, T, {5, 10});
      const auto y_dec = seq.Run(x.data, T, {});
      const double r_split = MaxTokenRel(
          y_split, get("sparse.y_split_5_plus_10").data, 32, T, nullptr);
      v.Check(Same(y_split, y) && Same(y_dec, y) && r_split <= kTol,
              "fixture sparse: prefill 5+10 and 15 decode steps bit-identical "
              "to prefill 15; vs HF split rel " +
                  Sci(r_split));
    }
  }
  return true;
}

// ---- Synthetic ----

HostLayer RandomLayer(const glm::Config& c, std::uint32_t seed) {
  std::mt19937 rng(seed);
  const auto fill = [&](std::size_t n, float scale, float offset = 0.0F) {
    std::normal_distribution<float> d(0.0F, 1.0F);
    std::vector<float> v(n);
    for (auto& x : v) {
      x = offset + (scale * d(rng));
    }
    return v;
  };
  const std::size_t hidden = c.hidden_size;
  const std::size_t heads = c.num_heads;
  const auto inv = [](std::size_t n) {
    return 1.0F / std::sqrt(static_cast<float>(n));
  };
  HostLayer h;
  h.q_a = fill(c.q_lora_rank * hidden, inv(hidden));
  h.q_b = fill(heads * c.mla_head_dim * c.q_lora_rank, inv(c.q_lora_rank));
  h.kv_a = fill(c.kv_lora_rank * hidden, inv(hidden));
  h.k_b =
      fill(heads * c.kv_lora_rank * c.mla_head_dim, 2.0F * inv(c.mla_head_dim));
  h.v_b = fill(heads * c.mla_value_dim * c.kv_lora_rank, inv(c.kv_lora_rank));
  h.out = fill(hidden * heads * c.mla_value_dim, inv(heads * c.mla_value_dim));
  const std::size_t ih = c.indexer_heads;
  const std::size_t idim = c.indexer_head_dim;
  h.idx_q_b = fill(ih * idim * c.q_lora_rank, inv(c.q_lora_rank));
  h.idx_k = fill(idim * hidden, inv(hidden));
  h.idx_proj = fill(ih * hidden, inv(hidden));
  h.idx_gate = fill(idim * hidden, inv(hidden));
  h.q_a_norm = fill(c.q_lora_rank, 0.2F, 1.0F);
  h.kv_a_norm = fill(c.kv_lora_rank, 0.2F, 1.0F);
  h.k_norm = fill(idim, 0.2F, 1.0F);
  h.k_norm_bias = fill(idim, 0.1F);
  h.ape = fill(c.indexer_kpool * idim, 0.5F);
  return h;
}

std::vector<float> RandomX(std::size_t n, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> d(0.0F, 1.0F);
  std::vector<float> x(n);
  for (auto& v : x) {
    v = d(rng);
  }
  return x;
}

void SyntheticCheck(Verdict& v, const std::string& label, const glm::Config& c,
                    std::uint32_t T,
                    const std::vector<std::vector<std::uint32_t>>& splits,
                    std::uint32_t limited_scratch) {
  const HostLayer h = RandomLayer(c, 1234 + T);
  const auto x = RandomX(std::size_t{T} * c.hidden_size, 99 + T);
  const auto t0 = std::chrono::steady_clock::now();
  CpuMla cpu(c, h);
  std::vector<float> y_cpu(x.size());
  cpu.Run(x.data(), T, y_cpu.data());
  const double cpu_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count();
  glm::DeviceArena arena;
  const glm::MlaLayerDevice w = Upload(arena, c, h);
  GpuSeq seq(c, w, 0, T + 3, T);
  std::vector<std::vector<std::int32_t>> top;
  const auto t1 = std::chrono::steady_clock::now();
  const auto y = seq.Run(x, T, {T}, &top);
  const double gpu_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t1)
          .count();
  std::uint32_t at = 0;
  const double r = MaxTokenRel(y, y_cpu, c.hidden_size, T, &at);
  std::uint32_t mismatched = 0;
  std::uint32_t first_bad = T;
  std::uint64_t dropped = 0;
  std::uint32_t sparse_tokens = 0;
  for (std::uint32_t t = 0; t < T; ++t) {
    auto want = cpu.selections[t];
    std::sort(want.begin(), want.end());
    const auto got = TokensOf(c, top[t], t);
    if (got != want) {
      ++mismatched;
      first_bad = std::min(first_bad, t);
    }
    dropped += t + 1 - got.size();
    sparse_tokens += got.size() < t + 1 ? 1 : 0;
  }
  char line[320];
  std::snprintf(line, sizeof line,
                "%s T=%u heads=%u: GPU vs CPU max token rel %.3e (token %u); "
                "CPU %.1fs GPU %.3fs",
                label.c_str(), T, c.num_heads, r, at, cpu_s, gpu_s);
  v.Check(r <= kTol, line);
  std::snprintf(line, sizeof line,
                "%s: selected token sets identical for %u/%u tokens (first "
                "mismatch %d); %u tokens sparse, %llu causal positions dropped",
                label.c_str(), T - mismatched, T,
                mismatched ? static_cast<int>(first_bad) : -1, sparse_tokens,
                static_cast<unsigned long long>(dropped));
  v.Check(mismatched == 0, line);
  for (const auto& split : splits) {
    std::string name;
    for (const auto n : split) {
      name += std::to_string(n) + "+";
    }
    name += "decode";
    const auto ys = seq.Run(x, T, split);
    std::uint32_t bad = 0;
    const double rs = MaxTokenRel(ys, y, c.hidden_size, T, &bad);
    v.Check(Same(ys, y), label + ": " + name + " bit-identical to prefill " +
                             std::to_string(T) + " (max rel " + Sci(rs) + ")");
  }
  if (limited_scratch > 0) {
    GpuSeq small(c, w, 0, T + 3, limited_scratch);
    const auto yl = small.Run(x, T, {T});
    v.Check(Same(yl, y), label + ": one call with scratch for " +
                             std::to_string(limited_scratch) +
                             " tokens (internal passes) bit-identical");
  }
}

int RunDefault() {
  Verdict v;
  const bool fixture = FixtureCheck(v);
  // G = 2 attention path, short context (no pool dropped), odd splits.
  SyntheticCheck(v, "synthetic-short",
                 TinyConfig(64, 2, 32, 48, 16, 24, 8, 32, 32, 4), 77,
                 {{1, 2, 3, 30}, {40, 37}}, 5);
  // Real top_k / kpool / indexer head dim, 4 heads (G = 4 path) and more
  // visible tokens than top_k raw tokens: pools are dropped from token 2051.
  SyntheticCheck(v, "synthetic-long",
                 TinyConfig(64, 4, 32, 64, 16, 16, 8, 128, 2048, 4), 2600,
                 {{1000, 1, 1, 777, 300, 500}, {2049, 3}}, 33);
  std::cout << (v.pass ? "PASS" : "FAIL")
            << (fixture ? "" : " (fixture part skipped)") << '\n';
  return v.pass ? 0 : 1;
}

// ---- Real GGUF ----

const std::vector<std::uint32_t> kRealLayers = {3, 7};
constexpr std::uint32_t kPrefill = 40;
constexpr std::uint32_t kDecode = 3;

std::vector<std::int32_t> RealIds() {
  // "The capital of France is" then a deterministic spread of ids.
  std::vector<std::int32_t> ids = {785, 6722, 315, 9621, 374};
  for (auto i = static_cast<std::uint32_t>(ids.size()); i < kPrefill + kDecode;
       ++i) {
    ids.push_back(static_cast<std::int32_t>(1000 + ((i * 7919) % 90000)));
  }
  return ids;
}

void WriteF32(const std::string& path, const std::vector<float>& v) {
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(v.data()),
            static_cast<std::streamsize>(v.size() * sizeof(float)));
  if (!out) {
    throw std::runtime_error("cannot write " + path);
  }
}

std::vector<float> ReadF32(const std::string& path, std::size_t count) {
  std::ifstream in(path, std::ios::binary);
  std::vector<float> v(count);
  in.read(reinterpret_cast<char*>(v.data()),
          static_cast<std::streamsize>(count * sizeof(float)));
  if (!in) {
    throw std::runtime_error("cannot read " + path);
  }
  return v;
}

int RunRealCpu(const glm::ModelWeights& weights, const std::string& dir) {
  // ReferenceModel truncated after the last probed layer (a smaller copy of
  // the config; every layer it runs is the real one).
  glm::Config c = weights.config;
  c.num_layers = kRealLayers.back() + 1;
  ref::GgufWeightSource source(weights);
  ref::ReferenceModel model(c, source);
  const auto ids = RealIds();
  std::map<std::string, std::vector<float>> dumps;
  const ref::TapFn tap = [&](std::string_view name, int layer,
                             const float* data, std::size_t width,
                             std::uint32_t n) {
    if (std::find(kRealLayers.begin(), kRealLayers.end(),
                  static_cast<std::uint32_t>(layer)) == kRealLayers.end()) {
      return;
    }
    const char* tag = name == "input_layernorm.out" ? "x"
                      : name == "self_attn.out"     ? "y"
                                                    : nullptr;
    if (tag == nullptr) {
      return;
    }
    auto& v = dumps[std::string(tag) + "-" + std::to_string(layer)];
    v.insert(v.end(), data, data + (width * n));
  };
  const auto t0 = std::chrono::steady_clock::now();
  (void)model.Forward(std::span(ids).first(kPrefill), false, tap);
  for (std::uint32_t i = 0; i < kDecode; ++i) {
    (void)model.Forward(std::span(ids).subspan(kPrefill + i, 1), false, tap);
  }
  std::cout << "ReferenceModel layers 0.." << kRealLayers.back() << ", "
            << kPrefill << " prefill + " << kDecode << " decode in "
            << std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                             t0)
                   .count()
            << " s\n";
  for (const auto& [name, v] : dumps) {
    WriteF32(dir + "/" + name + ".f32", v);
  }
  return 0;
}

int RunRealGpu(const glm::ModelWeights& weights, const std::string& dir) {
  const glm::Config& c = weights.config;
  const std::uint32_t T = kPrefill + kDecode;
  const std::size_t hidden = c.hidden_size;
  Verdict v;
  for (const std::uint32_t il : kRealLayers) {
    if (!c.IsMlaLayer(il)) {
      throw std::runtime_error("layer " + std::to_string(il) + " is not MLA");
    }
    const std::string l = std::to_string(il);
    const auto x = ReadF32(dir + "/x-" + l + ".f32", T * hidden);
    const auto want = ReadF32(dir + "/y-" + l + ".f32", T * hidden);
    glm::DeviceArena arena;
    glm::MlaLayerDevice w;
    std::string error;
    if (!glm::UploadMlaLayer(arena, c, weights.layers[il], il, &w, &error)) {
      throw std::runtime_error("upload: " + error);
    }
    const auto& lw = weights.layers[il];
    std::cout << "layer " << il << " types q_a " << int(lw.mla_q_a.type)
              << " q_b " << int(lw.mla_q_b.type) << " kv_a "
              << int(lw.mla_kv_a.type) << " k_b " << int(lw.mla_k_b.type)
              << " v_b " << int(lw.mla_v_b.type) << " out "
              << int(lw.attn_out.type) << " idx_q_b " << int(lw.idx_q_b.type)
              << " idx_k " << int(lw.idx_k.type) << '\n';
    GpuSeq seq(c, w, il, 256, T);
    std::vector<std::vector<std::int32_t>> top;
    const auto y = seq.Run(x, T, {T}, &top);
    const auto y_pd = seq.Run(x, T, {kPrefill});
    const auto y_odd = seq.Run(x, T, {7, 1, 20});
    std::uint32_t at = 0;
    const double r = MaxTokenRel(y, want, hidden, T, &at);
    const double r_pd = MaxTokenRel(y_pd, want, hidden, T, nullptr);
    double r_dec = 0.0;
    for (std::uint32_t t = kPrefill; t < T; ++t) {
      r_dec = std::max(r_dec, RelL2(y_pd.data() + (t * hidden),
                                    want.data() + (t * hidden), hidden));
    }
    char line[320];
    std::snprintf(line, sizeof line,
                  "real layer %u: prefill %u max token rel %.3e (token %u); "
                  "prefill %u + %u decode %.3e (decode rows %.3e)",
                  il, T, r, at, kPrefill, kDecode, r_pd, r_dec);
    v.Check(r <= kTol && r_pd <= kTol, line);
    v.Check(Same(y, y_pd) && Same(y, y_odd),
            "real layer " + l +
                ": prefill 43, 40+3 decode, 7+1+20+decode "
                "bit-identical");
    bool full = true;
    for (std::uint32_t t = 0; t < T; ++t) {
      std::vector<std::int32_t> all(t + 1);
      for (std::uint32_t i = 0; i <= t; ++i) {
        all[i] = static_cast<std::int32_t>(i);
      }
      full = full && TokensOf(c, top[t], t) == all;
    }
    v.Check(full, "real layer " + l +
                      ": every token selects all causal positions (T < "
                      "top_k, the reference's selection)");
  }
  std::cout << (v.pass ? "PASS" : "FAIL") << '\n';
  return v.pass ? 0 : 1;
}

int RunReal(const std::string& mode, const std::string& model,
            const std::string& dir) {
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(model, &error);
  if (!reader) {
    throw std::runtime_error("open failed: " + error);
  }
  auto weights = glm::ModelWeights::Bind(*reader, /*bind_nextn=*/false, &error);
  if (!weights) {
    throw std::runtime_error("bind failed: " + error);
  }
  return mode == "real-cpu" ? RunRealCpu(*weights, dir)
                            : RunRealGpu(*weights, dir);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 1) {
      return RunDefault();
    }
    const std::string mode = argv[1];
    if (argc == 4 && (mode == "real-cpu" || mode == "real-gpu")) {
      return RunReal(mode, argv[2], argv[3]);
    }
    std::cerr << "usage: " << argv[0]
              << " [real-cpu|real-gpu MODEL.gguf DIR]\n";
    return 2;
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << '\n';
    return 1;
  }
}
