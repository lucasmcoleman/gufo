// GPU KDA kernels (kernels/kda.hpp) against the CPU oracle
// (reference_kda.hpp) on random data at head dim 128: (H, T) in
// {(64, 1), (64, 5), (16, 64)} with a nonzero conv history and recurrent
// state and padded row strides. Every stage (conv out + history, g, decay,
// qn, kn, beta, recurrence output, final state, gated output) at rel-L2 <=
// 1e-5; a prefill-then-decode GPU run must match the full GPU prefill and
// the oracle; a recurrence that decays the value axis must miss the GPU
// output. Allocates under 10 MB of device memory.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/glm53_flash/kernels/kda.hpp"
#include "src/models/glm53_flash/reference_kda.hpp"

namespace ref = gufo::models::glm53_flash::reference;
namespace rocm = gufo::models::glm53_flash::rocm;

namespace {

constexpr std::uint32_t kD = rocm::kKdaHeadDim;
constexpr std::uint32_t kTaps = rocm::kKdaConvTaps;
constexpr float kLowerBound = -5.0F;
constexpr float kL2Eps = 1e-6F;
constexpr float kRmsEps = 1e-5F;
constexpr double kTol = 1e-5;
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
    Upload(host);
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
  void Upload(const std::vector<float>& host) {
    if (host.size() != n_)
      throw std::runtime_error("Upload size mismatch");
    if (n_ != 0) {
      CheckHip(hipMemcpy(ptr_, host.data(), n_ * sizeof(float),
                         hipMemcpyHostToDevice),
               "hipMemcpy H2D");
    }
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

double RelL2(const std::vector<float>& got, const std::vector<float>& want) {
  double num = 0.0;
  double den = 0.0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    const double d = static_cast<double>(got[i]) - want[i];
    num += d * d;
    den += static_cast<double>(want[i]) * want[i];
  }
  return den == 0.0 ? std::sqrt(num) : std::sqrt(num / den);
}

void Check(const std::string& what, const std::vector<float>& got,
           const std::vector<float>& want) {
  const double rel = RelL2(got, want);
  const bool ok =
      got.size() == want.size() && std::isfinite(rel) && rel <= kTol;
  std::cout << (ok ? "ok   " : "FAIL ") << what << " rel_l2 " << rel << "\n";
  if (!ok)
    ++failures;
}

void CheckMutantFails(const std::string& what, const std::vector<float>& got,
                      const std::vector<float>& want) {
  const double rel = RelL2(got, want);
  const bool ok = rel > kTol;
  std::cout << (ok ? "ok   " : "FAIL ") << what << " mutant rel_l2 " << rel
            << " (must exceed " << kTol << ")\n";
  if (!ok)
    ++failures;
}

/// Rows [0, n) of a [rows][stride] matrix, width columns each, packed.
std::vector<float> Pack(const std::vector<float>& m, std::size_t rows,
                        std::size_t stride, std::size_t width,
                        std::size_t col0 = 0) {
  std::vector<float> out(rows * width);
  for (std::size_t r = 0; r < rows; ++r)
    std::copy_n(m.begin() + static_cast<std::ptrdiff_t>((r * stride) + col0),
                width, out.begin() + static_cast<std::ptrdiff_t>(r * width));
  return out;
}

/// [H][a][b] <-> [H][b][a].
std::vector<float> TransposeHeads(const std::vector<float>& s,
                                  std::uint32_t heads) {
  std::vector<float> out(s.size());
  for (std::size_t h = 0; h < heads; ++h)
    for (std::size_t a = 0; a < kD; ++a)
      for (std::size_t b = 0; b < kD; ++b)
        out[(h * kD * kD) + (b * kD) + a] = s[(h * kD * kD) + (a * kD) + b];
  return out;
}

/// Decay along the value axis (S[i][j] *= exp(g[j])): the bug the HF
/// orientation case pins. HF state layout.
void MutantRecurrent(const float* q, const float* k, const float* v,
                     const float* g, const float* beta, float* state, float* o,
                     std::uint32_t n_tokens, std::uint32_t heads) {
  std::vector<double> u(kD);
  for (std::size_t t = 0; t < n_tokens; ++t) {
    for (std::size_t h = 0; h < heads; ++h) {
      const std::size_t row = (t * heads) + h;
      float* S = state + (h * kD * kD);
      for (std::size_t i = 0; i < kD; ++i)
        for (std::size_t j = 0; j < kD; ++j)
          S[(i * kD) + j] *= std::exp(g[(row * kD) + j]);
      std::fill(u.begin(), u.end(), 0.0);
      for (std::size_t i = 0; i < kD; ++i)
        for (std::size_t j = 0; j < kD; ++j)
          u[j] += static_cast<double>(S[(i * kD) + j]) * k[(row * kD) + i];
      for (std::size_t j = 0; j < kD; ++j)
        u[j] = (v[(row * kD) + j] - u[j]) * beta[row];
      for (std::size_t i = 0; i < kD; ++i)
        for (std::size_t j = 0; j < kD; ++j)
          S[(i * kD) + j] += static_cast<float>(k[(row * kD) + i] * u[j]);
      for (std::size_t j = 0; j < kD; ++j) {
        double acc = 0.0;
        for (std::size_t i = 0; i < kD; ++i)
          acc += static_cast<double>(S[(i * kD) + j]) * q[(row * kD) + i];
        o[(row * kD) + j] = static_cast<float>(acc);
      }
    }
  }
}

struct Inputs {
  std::uint32_t T{0}, H{0}, width{0}, channels{0};
  std::uint32_t qkv_stride{0}, f_stride{0}, beta_stride{0}, gate_stride{0};
  std::vector<float> qkv, conv_w, hist0, f, dt, a, beta_in, gate, norm_w;
  std::vector<float> state0;  // HF layout [H][k][v]
};

Inputs MakeInputs(std::uint32_t T, std::uint32_t H, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> n01(0.0F, 1.0F);
  Inputs in;
  in.T = T;
  in.H = H;
  in.width = H * kD;
  in.channels = 3 * in.width;
  in.qkv_stride = in.channels + 64;
  in.f_stride = in.width + 32;
  in.beta_stride = H + 3;
  in.gate_stride = in.width + 128;
  const auto fill = [&](std::vector<float>& v, std::size_t n, float mean,
                        float sd) {
    v.resize(n);
    for (auto& x : v)
      x = mean + (sd * n01(rng));
  };
  fill(in.qkv, std::size_t{T} * in.qkv_stride, 0.0F, 1.0F);
  fill(in.conv_w, std::size_t{in.channels} * kTaps, 0.0F, 0.5F);
  fill(in.hist0, std::size_t{kTaps - 1} * in.channels, 0.0F, 1.0F);
  fill(in.f, std::size_t{T} * in.f_stride, 0.0F, 1.5F);
  fill(in.dt, in.width, 0.0F, 0.5F);
  fill(in.a, H, 0.0F, 0.5F);
  for (auto& x : in.a)
    x = -std::exp(x);  // -exp(A_log)
  fill(in.beta_in, std::size_t{T} * in.beta_stride, 0.0F, 1.0F);
  fill(in.gate, std::size_t{T} * in.gate_stride, 0.0F, 1.0F);
  fill(in.norm_w, kD, 1.0F, 0.2F);
  fill(in.state0, std::size_t{H} * kD * kD, 0.0F, 0.3F);
  return in;
}

struct Outputs {
  std::vector<float> conv, hist, g, decay, qn, kn, beta, o, state, out;
};

/// Oracle chain over all T tokens. `mutant_o` receives the value-axis decay
/// recurrence output.
Outputs RunOracle(const Inputs& in, std::vector<float>* mutant_o) {
  const std::uint32_t T = in.T;
  const std::uint32_t H = in.H;
  Outputs r;
  const auto qkv = Pack(in.qkv, T, in.qkv_stride, in.channels);
  r.hist = in.hist0;
  r.conv.resize(qkv.size());
  ref::KdaConv(qkv.data(), in.conv_w.data(), r.hist.data(), r.conv.data(), T,
               in.channels, kTaps);
  const auto f = Pack(in.f, T, in.f_stride, in.width);
  r.g.resize(f.size());
  ref::KdaGate(f.data(), in.dt.data(), in.a.data(), kLowerBound, r.g.data(), T,
               H, kD);
  r.decay.resize(r.g.size());
  for (std::size_t i = 0; i < r.g.size(); ++i)
    r.decay[i] = static_cast<float>(std::exp(static_cast<double>(r.g[i])));
  r.qn = Pack(r.conv, T, in.channels, in.width, 0);
  r.kn = Pack(r.conv, T, in.channels, in.width, in.width);
  const auto v = Pack(r.conv, T, in.channels, in.width, 2 * in.width);
  r.beta = Pack(in.beta_in, T, in.beta_stride, H);
  ref::KdaPrep(r.qn.data(), r.kn.data(), r.beta.data(), T, H, kD, kL2Eps);
  r.state = in.state0;
  r.o.resize(std::size_t{T} * in.width);
  ref::KdaRecurrent(r.qn.data(), r.kn.data(), v.data(), r.g.data(),
                    r.beta.data(), r.state.data(), r.o.data(), T, H, kD, kD);
  if (mutant_o != nullptr) {
    auto s = in.state0;
    mutant_o->resize(r.o.size());
    MutantRecurrent(r.qn.data(), r.kn.data(), v.data(), r.g.data(),
                    r.beta.data(), s.data(), mutant_o->data(), T, H);
  }
  const auto gate = Pack(in.gate, T, in.gate_stride, in.width);
  r.out.resize(r.o.size());
  ref::KdaOutput(r.o.data(), gate.data(), in.norm_w.data(), kRmsEps,
                 r.out.data(), T, H, kD);
  return r;
}

struct Device {
  explicit Device(const Inputs& in)
      : qkv(in.qkv),
        conv_w(in.conv_w),
        hist(in.hist0),
        f(in.f),
        dt(in.dt),
        a(in.a),
        beta_in(in.beta_in),
        gate(in.gate),
        norm_w(in.norm_w),
        state(TransposeHeads(in.state0, in.H)),
        conv(std::size_t{in.T} * in.channels),
        g(std::size_t{in.T} * in.width),
        decay(g_size(in)),
        qn(g_size(in)),
        kn(g_size(in)),
        beta(std::size_t{in.T} * in.H),
        o(g_size(in)),
        out(g_size(in)) {}
  static std::size_t g_size(const Inputs& in) {
    return std::size_t{in.T} * in.width;
  }
  DeviceBuffer qkv, conv_w, hist, f, dt, a, beta_in, gate, norm_w, state, conv,
      g, decay, qn, kn, beta, o, out;
};

/// Runs the GPU pipeline on tokens [t0, t0 + n), carrying conv history and
/// state in the device buffers.
bool RunGpu(const Inputs& in, Device& d, std::uint32_t t0, std::uint32_t n,
            hipStream_t stream, float* conv_ckpt = nullptr,
            float* state_ckpt = nullptr) {
  const std::uint32_t H = in.H;
  const std::size_t tw = std::size_t{t0} * in.width;
  float* conv = d.conv.get() + (std::size_t{t0} * in.channels);
  bool ok = rocm::KdaConv(d.qkv.get() + (std::size_t{t0} * in.qkv_stride),
                          in.qkv_stride, d.conv_w.get(), d.hist.get(), conv, n,
                          in.channels, stream, conv_ckpt);
  ok = ok && rocm::KdaGate(d.f.get() + (std::size_t{t0} * in.f_stride),
                           in.f_stride, d.dt.get(), d.a.get(), kLowerBound,
                           d.g.get() + tw, d.decay.get() + tw, n, H, stream);
  ok = ok && rocm::KdaPrep(conv, in.channels,
                           d.beta_in.get() + (std::size_t{t0} * in.beta_stride),
                           in.beta_stride, d.qn.get() + tw, d.kn.get() + tw,
                           d.beta.get() + (std::size_t{t0} * H), n, H, kL2Eps,
                           stream);
  ok = ok && rocm::KdaRecurrent(d.qn.get() + tw, d.kn.get() + tw,
                                conv + (2 * in.width), in.channels,
                                d.decay.get() + tw,
                                d.beta.get() + (std::size_t{t0} * H),
                                d.state.get(), d.o.get() + tw, n, H, stream,
                                state_ckpt);
  ok = ok && rocm::KdaOutput(d.o.get() + tw,
                             d.gate.get() + (std::size_t{t0} * in.gate_stride),
                             in.gate_stride, d.norm_w.get(), kRmsEps,
                             d.out.get() + tw, n, H, stream);
  return ok;
}

void RunCase(std::uint32_t T, std::uint32_t H, std::uint32_t prefill,
             unsigned seed) {
  const std::string name = "H=" + std::to_string(H) + " T=" + std::to_string(T);
  const Inputs in = MakeInputs(T, H, seed);
  std::vector<float> mutant_o;
  const Outputs want = RunOracle(in, &mutant_o);

  hipStream_t stream = nullptr;
  CheckHip(hipStreamCreate(&stream), "hipStreamCreate");
  Outputs full;
  Outputs split;
  bool ok = true;
  {
    Device d(in);
    ok = RunGpu(in, d, 0, T, stream);
    CheckHip(hipStreamSynchronize(stream), "hipStreamSynchronize");
    full.conv = d.conv.Download();
    full.hist = d.hist.Download();
    full.g = d.g.Download();
    full.decay = d.decay.Download();
    full.qn = d.qn.Download();
    full.kn = d.kn.Download();
    full.beta = d.beta.Download();
    full.o = d.o.Download();
    full.state = TransposeHeads(d.state.Download(), H);
    full.out = d.out.Download();
  }
  if (prefill < T) {
    Device d(in);
    ok = ok && RunGpu(in, d, 0, prefill, stream);
    for (std::uint32_t t = prefill; t < T; ++t)
      ok = ok && RunGpu(in, d, t, 1, stream);
    CheckHip(hipStreamSynchronize(stream), "hipStreamSynchronize");
    split.conv = d.conv.Download();
    split.hist = d.hist.Download();
    split.o = d.o.Download();
    split.state = TransposeHeads(d.state.Download(), H);
    split.out = d.out.Download();
  }
  CheckHip(hipStreamDestroy(stream), "hipStreamDestroy");
  if (!ok) {
    std::cout << "FAIL " << name << " launcher returned false\n";
    ++failures;
    return;
  }
  Check(name + " conv_out", full.conv, want.conv);
  Check(name + " conv_state", full.hist, want.hist);
  Check(name + " g", full.g, want.g);
  Check(name + " decay", full.decay, want.decay);
  Check(name + " qn", full.qn, want.qn);
  Check(name + " kn", full.kn, want.kn);
  Check(name + " beta", full.beta, want.beta);
  Check(name + " o", full.o, want.o);
  Check(name + " state", full.state, want.state);
  Check(name + " out", full.out, want.out);
  CheckMutantFails(name + " value-axis decay o", full.o, mutant_o);
  if (prefill < T) {
    const std::string sp = name + " prefill(" + std::to_string(prefill) +
                           ")+decode(" + std::to_string(T - prefill) + ")";
    Check(sp + " conv_out vs full", split.conv, full.conv);
    Check(sp + " conv_state vs full", split.hist, full.hist);
    Check(sp + " o vs full", split.o, full.o);
    Check(sp + " state vs full", split.state, full.state);
    Check(sp + " out vs full", split.out, full.out);
    Check(sp + " out vs oracle", split.out, want.out);
    Check(sp + " state vs oracle", split.state, want.state);
  }
}

/// Speculative verify checkpoints: a pass of n tokens after `prefill`
/// records the conv history and state after each of its first n - 1
/// tokens; each must equal, bit for bit, a pass that stopped there.
void RunCheckpointCase(std::uint32_t T, std::uint32_t H, std::uint32_t prefill,
                       unsigned seed) {
  const Inputs in = MakeInputs(T, H, seed);
  const std::uint32_t n = T - prefill;
  const std::size_t hist = in.hist0.size();
  const std::size_t state = in.state0.size();
  hipStream_t stream = nullptr;
  CheckHip(hipStreamCreate(&stream), "hipStreamCreate");
  DeviceBuffer conv_ckpt((n - 1) * hist);
  DeviceBuffer state_ckpt((n - 1) * state);
  bool ok = true;
  {
    Device d(in);
    ok = RunGpu(in, d, 0, prefill, stream) &&
         RunGpu(in, d, prefill, n, stream, conv_ckpt.get(), state_ckpt.get());
  }
  CheckHip(hipStreamSynchronize(stream), "hipStreamSynchronize");
  const auto ck_conv = conv_ckpt.Download();
  const auto ck_state = state_ckpt.Download();
  for (std::uint32_t keep = 1; ok && keep < n; ++keep) {
    Device d(in);
    ok = RunGpu(in, d, 0, prefill, stream) &&
         RunGpu(in, d, prefill, keep, stream);
    CheckHip(hipStreamSynchronize(stream), "hipStreamSynchronize");
    const auto h = d.hist.Download();
    const auto st = d.state.Download();
    const bool same =
        std::memcmp(h.data(), ck_conv.data() + ((keep - 1) * hist),
                    hist * sizeof(float)) == 0 &&
        std::memcmp(st.data(), ck_state.data() + ((keep - 1) * state),
                    state * sizeof(float)) == 0;
    const std::string what = "H=" + std::to_string(H) + " verify " +
                             std::to_string(n) + " after " +
                             std::to_string(prefill) + ": checkpoint " +
                             std::to_string(keep) +
                             " bit-identical to a pass of " +
                             std::to_string(keep);
    std::cout << (same ? "ok   " : "FAIL ") << what << "\n";
    if (!same)
      ++failures;
  }
  CheckHip(hipStreamDestroy(stream), "hipStreamDestroy");
  if (!ok) {
    std::cout << "FAIL checkpoint case launcher returned false\n";
    ++failures;
  }
}

}  // namespace

int main() {
  try {
    int devices = 0;
    if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
      std::cerr << "no HIP device\n";
      return 77;
    }
    RunCase(1, 64, 1, 11);    // decode
    RunCase(5, 64, 2, 12);    // short prefill; split reads history + new tokens
    RunCase(64, 16, 40, 13);  // multi-tile conv + history kernel
    RunCase(9, 3, 8, 14);     // partial final conv tile + partial norm group
    RunCheckpointCase(7, 64, 2, 15);  // verify 5 after a 2-token prefill
    RunCheckpointCase(2, 16, 0, 16);  // verify 2 from the initial state
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << "\n";
    return 1;
  }
  if (failures != 0) {
    std::cerr << failures << " check(s) failed\n";
    return 1;
  }
  std::cout << "glm53_flash kda_ops: PASS\n";
  return 0;
}
