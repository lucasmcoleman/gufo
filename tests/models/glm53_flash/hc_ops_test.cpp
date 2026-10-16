// GPU hyper-connection kernels (kernels/hc.hpp) against the CPU oracle
// (reference_hc.hpp): random inputs at hidden 4096 (fused path) for
// T in {1, 7, 64}, and the HF fixture at hidden 32 (generic path). Every
// output (logits, split, x, H_new in place, head mean) at rel-L2 <= 1e-5.
// Allocates under 10 MB of device memory.

#include <hip/hip_runtime.h>

#include <cstdint>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/glm53_flash/config.hpp"
#include "src/models/glm53_flash/kernels/hc.hpp"
#include "src/models/glm53_flash/reference_hc.hpp"
#include "tests/models/glm53_flash/hc_fixture.hpp"

namespace glm = gufo::models::glm53_flash;
namespace ref = gufo::models::glm53_flash::reference;
namespace rocm = gufo::models::glm53_flash::rocm;

namespace {

int failures = 0;

void CheckHip(hipError_t error, const char* what) {
  if (error != hipSuccess) {
    throw std::runtime_error(std::string(what) + ": " +
                             hipGetErrorString(error));
  }
}

class DeviceBuffer {
public:
  explicit DeviceBuffer(const std::vector<float>& host) : n_(host.size()) {
    CheckHip(hipMalloc(&ptr_, (n_ == 0 ? 1 : n_) * sizeof(float)), "hipMalloc");
    if (n_ != 0) {
      CheckHip(hipMemcpy(ptr_, host.data(), n_ * sizeof(float),
                         hipMemcpyHostToDevice),
               "hipMemcpy H2D");
    }
  }
  explicit DeviceBuffer(std::size_t n) : DeviceBuffer(std::vector<float>(n)) {}
  ~DeviceBuffer() { (void)hipFree(ptr_); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&&) = delete;
  DeviceBuffer& operator=(DeviceBuffer&&) = delete;

  [[nodiscard]] float* get() const noexcept {
    return static_cast<float*>(ptr_);
  }
  [[nodiscard]] std::vector<float> Download() const {
    std::vector<float> host(n_);
    CheckHip(
        hipMemcpy(host.data(), ptr_, n_ * sizeof(float), hipMemcpyDeviceToHost),
        "hipMemcpy D2H");
    return host;
  }

private:
  void* ptr_{nullptr};
  std::size_t n_{0};
};

void Check(const std::string& what, const std::vector<float>& got,
           const std::vector<float>& want) {
  const double rel =
      glm53_hc_fixture::RelL2(got.data(), want.data(), want.size());
  const bool ok = got.size() == want.size() && rel <= 1e-5;
  std::cout << (ok ? "ok   " : "FAIL ") << what << " rel_l2 " << rel << "\n";
  if (!ok)
    ++failures;
}

struct Case {
  std::string name;
  std::uint32_t tokens;
  std::uint32_t hidden;
  std::vector<float> H, fn, base, scale, a;
  // Optional HF expectations (fixture only).
  const std::map<std::string, glm53_hc_fixture::Array>* hf{nullptr};
};

void Run(const Case& k) {
  glm::Config c;
  c.hidden_size = k.hidden;
  c.hc_count = 4;
  c.hc_sinkhorn_iters = 20;
  c.hc_eps = 1e-6F;
  c.rms_eps = 1e-5F;
  const std::uint32_t T = k.tokens;
  const std::uint32_t D = k.hidden;
  const std::uint32_t hc_dim = 4 * D;

  // CPU oracle.
  std::vector<float> logits(T * 24), split(T * 24), x(T * D), h_new(T * hc_dim),
      head(T * D);
  for (std::uint32_t t = 0; t < T; ++t) {
    ref::HcMix m;
    const float* h = k.H.data() + (t * hc_dim);
    ref::HcPre(c, k.fn.data(), k.base.data(), k.scale.data(), h,
               x.data() + (t * D), &m);
    for (std::uint32_t j = 0; j < 24; ++j)
      logits[(t * 24) + j] = m.logits[j];
    for (std::uint32_t j = 0; j < 4; ++j)
      split[(t * 24) + j] = m.pre[j];
    for (std::uint32_t j = 0; j < 4; ++j)
      split[(t * 24) + 4 + j] = m.post[j];
    for (std::uint32_t j = 0; j < 16; ++j)
      split[(t * 24) + 8 + j] = m.comb[j];
    ref::HcPost(m, k.a.data() + (t * D), h, h_new.data() + (t * hc_dim));
    ref::HcHeadMean(c, h_new.data() + (t * hc_dim), head.data() + (t * D));
  }

  // GPU.
  hipStream_t stream = nullptr;
  CheckHip(hipStreamCreate(&stream), "hipStreamCreate");
  DeviceBuffer d_h(k.H), d_fn(k.fn), d_base(k.base), d_scale(k.scale), d_a(k.a),
      d_x(T * D), d_split(T * 24), d_logits(T * 24), d_head(T * D);
  bool ok = rocm::HcPre(d_h.get(), d_fn.get(), d_base.get(), d_scale.get(),
                        d_x.get(), d_split.get(), d_logits.get(), T, D,
                        c.hc_sinkhorn_iters, c.hc_eps, c.rms_eps, stream);
  // In place, as the executor will use it.
  ok = ok && rocm::HcPost(d_a.get(), d_h.get(), d_split.get(), d_h.get(), T, D,
                          stream);
  ok = ok && rocm::HcHeadMean(d_h.get(), d_head.get(), T, D, stream);
  CheckHip(hipStreamSynchronize(stream), "hipStreamSynchronize");
  CheckHip(hipStreamDestroy(stream), "hipStreamDestroy");
  if (!ok) {
    std::cout << "FAIL " << k.name << " launcher returned false\n";
    ++failures;
    return;
  }
  const auto g_logits = d_logits.Download();
  const auto g_split = d_split.Download();
  const auto g_x = d_x.Download();
  const auto g_h = d_h.Download();
  const auto g_head = d_head.Download();
  Check(k.name + " logits", g_logits, logits);
  Check(k.name + " split", g_split, split);
  Check(k.name + " x", g_x, x);
  Check(k.name + " H_new", g_h, h_new);
  Check(k.name + " head", g_head, head);

  if (k.hf != nullptr) {
    std::vector<float> pre(T * 4), post(T * 4), comb(T * 16);
    for (std::uint32_t t = 0; t < T; ++t) {
      for (std::uint32_t j = 0; j < 4; ++j) {
        pre[(t * 4) + j] = g_split[(t * 24) + j];
        post[(t * 4) + j] = g_split[(t * 24) + 4 + j];
      }
      for (std::uint32_t j = 0; j < 16; ++j) {
        comb[(t * 16) + j] = g_split[(t * 24) + 8 + j];
      }
    }
    Check(k.name + " vs HF mix", g_logits, k.hf->at("mix").data);
    Check(k.name + " vs HF pre", pre, k.hf->at("pre").data);
    Check(k.name + " vs HF post", post, k.hf->at("post").data);
    Check(k.name + " vs HF comb", comb, k.hf->at("comb").data);
    Check(k.name + " vs HF collapsed", g_x, k.hf->at("collapsed").data);
    Check(k.name + " vs HF hc_out", g_h, k.hf->at("hc_out").data);
    Check(k.name + " vs HF mean_head", g_head, k.hf->at("mean_head").data);
  }
}

Case RandomCase(std::uint32_t tokens, std::uint32_t hidden, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> n01(0.0F, 1.0F);
  Case k;
  k.name = "random T=" + std::to_string(tokens) +
           " hidden=" + std::to_string(hidden);
  k.tokens = tokens;
  k.hidden = hidden;
  const std::uint32_t hc_dim = 4 * hidden;
  k.H.resize(std::size_t{tokens} * hc_dim);
  k.fn.resize(std::size_t{24} * hc_dim);
  k.base.resize(24);
  k.a.resize(std::size_t{tokens} * hidden);
  for (auto& v : k.H)
    v = n01(rng);
  for (auto& v : k.fn)
    v = n01(rng) * 0.01F;  // logits of order 1
  for (auto& v : k.base)
    v = n01(rng) * 0.5F;
  for (auto& v : k.a)
    v = n01(rng);
  k.scale = {1.3F, 0.7F, 1.1F};
  return k;
}

}  // namespace

int main() {
  try {
    int devices = 0;
    if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
      std::cerr << "no HIP device\n";
      return 77;
    }
    for (std::uint32_t t : {1U, 7U, 64U})
      Run(RandomCase(t, 4096, 100 + t));
    Run(RandomCase(5, 4100, 7));  // generic path, scalar HcPost

    const auto fx = glm53_hc_fixture::Load(GLM53_HC_FIXTURE);
    Case k;
    k.name = "hf fixture";
    k.hidden = fx.at("H").shape[3];
    k.tokens = fx.at("H").shape[0] * fx.at("H").shape[1];
    k.H = fx.at("H").data;
    k.fn = fx.at("w.fn").data;
    k.base = fx.at("w.base").data;
    k.scale = fx.at("w.scale").data;
    k.a = fx.at("block_out").data;
    k.hf = &fx;
    Run(k);
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << "\n";
    return 1;
  }
  if (failures != 0) {
    std::cerr << failures << " check(s) failed\n";
    return 1;
  }
  std::cout << "glm53_flash hc_ops: PASS\n";
  return 0;
}
