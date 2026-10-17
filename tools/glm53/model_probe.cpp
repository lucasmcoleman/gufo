// V4 probe: the gufo GPU Executor over all layers of the real Maya-S GGUF,
// logits for "The capital of France is" = [785, 6722, 315, 9621, 374]
// compared with llama.cpp's logits and the CPU ReferenceModel's (V3 dumps).
//
// usage: glm53_flash_model_probe MODEL.gguf OUTDIR hostffn|laneb|ffntime
//   hostffn  GPU MlaForward (kernels/mla.hip.cpp + indexer.hip.cpp) with a
//            host FFN stand-in. The stand-in reads back only the routed
//            experts a token selects (tests/.../host_ops.hpp would dequantise
//            all 288 experts of every layer, ~29 GB F32 each, and cache them).
//   laneb    the executor's LaneBOps() binding; needs GLM53_LANE_B_OPS.
//   ffntime  no GPU: runs the host FFN of layer 3 on the V3 reference input
//            (post_attention_layernorm.out-3) from the mapped GGUF, checks it
//            against the V3 mlp.out-3 and prints the time for 5 and 1 tokens.
// Each GPU mode runs (a) a 5-token prefill and (b) after Reset a 4-token
// prefill plus a 1-token decode, writes OUTDIR/<mode>-<a|b>.f32
// ([5][154880] F32) and OUTDIR/compare.json, and prints the verdict.
// Memory: the executor uploads ~87 GiB with hipMalloc, which stalls at the KFD
// ~57 GiB system-memory cap on this APU (the UploadGuard SIGTERMs after 300 s
// without GTT growth). Run with LD_PRELOAD=/server/staging/glm53-gufo/v4/
// libhipmalloc_managed.so (hipMalloc -> hipMallocManaged; the GTT counter then
// stays flat, so the guard only matters for the plain-hipMalloc case; upload
// takes ~2 min), under gpu-lock-exec (v4/run.sh).
// Env: V3_DIR (default /server/staging/glm53-gufo/v3) holds llama/ and gufo/.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <iostream>
#include <map>
#include <numeric>
#include <csignal>
#include <thread>
#include <sys/mman.h>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/glm53_flash/executor.hpp"
#include "src/models/glm53_flash/reference_moe.hpp"
#include "src/models/glm53_flash/weights.hpp"
#include "tests/models/glm53_flash/host_ops.hpp"

namespace glm = gufo::models::glm53_flash;
namespace ref = gufo::models::glm53_flash::reference;
namespace so = gufo::models::glm53_flash::stand_in;

namespace {

const std::vector<std::int32_t> kIds = {785, 6722, 315, 9621, 374};
constexpr std::size_t kT = 5;
constexpr double kMaxKl = 1e-2;
constexpr double kMaxRelRef = 1e-3;
constexpr double kMaxRelRun = 1e-5;
constexpr double kTapRel = 1e-3;

double Now() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

std::vector<float> ReadF32(const std::string& path, std::size_t count) {
  std::ifstream in(path, std::ios::binary);
  std::vector<float> v(count);
  in.read(reinterpret_cast<char*>(v.data()),
          static_cast<std::streamsize>(count * sizeof(float)));
  if (!in)
    throw std::runtime_error("cannot read " + std::to_string(count) +
                             " floats from " + path);
  return v;
}

void WriteF32(const std::string& path, const std::vector<float>& v) {
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(v.data()),
            static_cast<std::streamsize>(v.size() * sizeof(float)));
  if (!out)
    throw std::runtime_error("cannot write " + path);
}

// ---- Host FFN stand-in that fetches only what it uses ----

/// A stacked weight in host or device memory, in its GGUF encoding.
struct Mat {
  const void* data{nullptr};
  gufo::core::GgmlType type{gufo::core::GgmlType::kF32};
  std::uint64_t cols{0}, rows{1}, experts{1};
  std::size_t row_bytes{0};
  bool device{false};
};

Mat FromDevice(const glm::DeviceMatrix& m) {
  return {m.data, m.type, m.cols, m.rows, m.experts, m.row_bytes, true};
}
Mat FromHost(const glm::TensorRef& t) {
  return {t.data, t.type, t.cols, t.rows, t.experts, t.RowBytes(), false};
}
Mat FloatsDevice(const float* p, std::uint64_t rows, std::uint64_t cols) {
  return {p, gufo::core::GgmlType::kF32, cols, rows, 1, cols * 4, true};
}

/// Expert `e` of `m` (rows x cols) as F32.
std::vector<float> Fetch(const Mat& m, std::uint64_t e) {
  if (m.data == nullptr)
    throw std::runtime_error("host ffn: empty matrix");
  const std::size_t bytes = m.rows * m.row_bytes;
  const auto* src = static_cast<const std::uint8_t*>(m.data) + (e * bytes);
  std::vector<std::uint8_t> raw(bytes);
  if (m.device) {
    if (hipMemcpy(raw.data(), src, bytes, hipMemcpyDeviceToHost) != hipSuccess)
      throw std::runtime_error("host ffn: matrix download failed");
  } else {
    std::memcpy(raw.data(), src, bytes);
  }
  std::vector<float> out(m.rows * m.cols);
  for (std::size_t r = 0; r < m.rows; ++r)
    so::DequantizeRow(m.type, raw.data() + (r * m.row_bytes), m.cols,
                      out.data() + (r * m.cols));
  return out;
}

struct FfnMats {
  bool dense{false};
  float clamp_exp{0}, clamp_shexp{0};
  Mat gate, up, down;  ///< dense FFN, or the shared expert
  Mat router, bias;    ///< F32 [experts][hidden], [1][experts]
  Mat gate_exps, up_exps, down_exps;
};

/// Runs f(0..n-1) on up to 8 threads; rethrows the first exception.
template<typename F>
void ParallelFor(std::size_t n, F&& f) {
  std::atomic<std::size_t> next{0};
  std::vector<std::future<void>> jobs;
  for (std::size_t k = 0; k < std::min<std::size_t>(8, n); ++k)
    jobs.push_back(std::async(std::launch::async, [&] {
      for (std::size_t i = next++; i < n; i = next++)
        f(i);
    }));
  for (auto& j : jobs)
    j.get();
}

/// ReferenceModel::Ffn's math on n tokens: x [n][hidden] -> y.
void HostFfn(const glm::Config& c, const FfnMats& w, const float* x,
             std::uint32_t n, float* y) {
  const std::size_t hidden = c.hidden_size;
  const auto gate = Fetch(w.gate, 0);
  const auto up = Fetch(w.up, 0);
  const auto down = Fetch(w.down, 0);
  const std::size_t ff = w.gate.rows;
  if (w.dense) {
    for (std::uint32_t t = 0; t < n; ++t)
      ref::DenseFfn(x + (t * hidden), gate.data(), up.data(), down.data(),
                    hidden, ff, w.clamp_exp, y + (t * hidden));
    return;
  }
  const std::size_t ne = c.num_experts;
  const auto router = Fetch(w.router, 0);
  const auto bias = Fetch(w.bias, 0);
  std::vector<ref::Routing> routing(n);
  std::vector<std::uint32_t> experts;
  for (std::uint32_t t = 0; t < n; ++t) {
    std::vector<float> logits(ne);
    for (std::size_t e = 0; e < ne; ++e) {
      double acc = 0.0;
      for (std::size_t d = 0; d < hidden; ++d)
        acc += static_cast<double>(router[(e * hidden) + d]) * x[(t * hidden) + d];
      logits[e] = static_cast<float>(acc);
    }
    routing[t] = ref::Route(logits.data(), bias.data(),
                            static_cast<std::uint32_t>(ne),
                            c.num_experts_used, c.expert_weights_scale,
                            c.expert_weights_norm,
                            ref::RouterDenominator::kLlamaClamp);
    for (auto id : routing[t].ids)
      if (std::find(experts.begin(), experts.end(), id) == experts.end())
        experts.push_back(id);
  }
  // Compact stacks over the selected experts; routing ids remapped.
  const std::size_t m = c.expert_ff * hidden;
  std::vector<float> gate_up(experts.size() * 2 * m);
  std::vector<float> down_exps(experts.size() * m);
  ParallelFor(experts.size(), [&](std::size_t i) {
    const auto g = Fetch(w.gate_exps, experts[i]);
    const auto u = Fetch(w.up_exps, experts[i]);
    const auto d = Fetch(w.down_exps, experts[i]);
    std::memcpy(gate_up.data() + (i * 2 * m), g.data(), m * sizeof(float));
    std::memcpy(gate_up.data() + (i * 2 * m) + m, u.data(), m * sizeof(float));
    std::memcpy(down_exps.data() + (i * m), d.data(), m * sizeof(float));
  });
  ParallelFor(n, [&](std::size_t t) {
    std::vector<float> routed(hidden);
    std::vector<float> shared(hidden);
    ref::Routing r = routing[t];
    for (auto& id : r.ids)
      id = static_cast<std::uint32_t>(
          std::find(experts.begin(), experts.end(), id) - experts.begin());
    ref::RoutedFfn(x + (t * hidden), gate_up.data(), down_exps.data(), hidden,
                   c.expert_ff, r, w.clamp_exp, routed.data());
    ref::DenseFfn(x + (t * hidden), gate.data(), up.data(), down.data(),
                  hidden, ff, w.clamp_shexp, shared.data());
    for (std::size_t d = 0; d < hidden; ++d)
      y[(t * hidden) + d] = routed[d] + shared[d];
  });
}

FfnMats FromDeviceFfn(const glm::Config& c, const glm::FfnLayerDevice& w) {
  FfnMats m;
  m.dense = w.dense;
  m.clamp_exp = w.clamp_exp;
  m.clamp_shexp = w.clamp_shexp;
  if (w.dense) {
    m.gate = FromDevice(w.gate);
    m.up = FromDevice(w.up);
    m.down = FromDevice(w.down);
  } else {
    m.gate = FromDevice(w.shexp_gate);
    m.up = FromDevice(w.shexp_up);
    m.down = FromDevice(w.shexp_down);
    m.router = FloatsDevice(w.router, c.num_experts, c.hidden_size);
    m.bias = FloatsDevice(w.router_bias, 1, c.num_experts);
    m.gate_exps = FromDevice(w.gate_exps);
    m.up_exps = FromDevice(w.up_exps);
    m.down_exps = FromDevice(w.down_exps);
  }
  return m;
}

class HostFfnOps {
public:
  [[nodiscard]] glm::LayerOps Bind() {
    glm::LayerOps ops;
    ops.mla_cache_bytes = [](const glm::Config& c, std::uint32_t cap) {
      return glm::MlaCacheBytes(c, cap);
    };
    ops.op_scratch_bytes = [](const glm::Config& c, std::uint32_t max_tokens) {
      return glm::MlaScratchBytes(c, max_tokens);
    };
    ops.mla = [](const glm::OpContext& ctx, const glm::Config& c,
                 std::uint32_t il, const glm::MlaLayerDevice& w,
                 glm::MlaCache& cache, const float* x, std::uint32_t n,
                 std::uint32_t pos0, float* y) {
      return glm::MlaForward(ctx, c, il, w, cache, x, n, pos0, y);
    };
    ops.ffn = [this](const glm::OpContext& ctx, const glm::Config& c,
                     std::uint32_t il, const glm::FfnLayerDevice& w,
                     const float* x, std::uint32_t n, float* y) {
      const std::size_t count = std::size_t{n} * c.hidden_size;
      std::vector<float> hx(count);
      std::vector<float> hy(count);
      try {
        if (hipMemcpyAsync(hx.data(), x, count * sizeof(float),
                           hipMemcpyDeviceToHost, ctx.stream) != hipSuccess ||
            hipStreamSynchronize(ctx.stream) != hipSuccess)
          throw std::runtime_error("activation download failed");
        const double t0 = Now();
        HostFfn(c, FromDeviceFfn(c, w), hx.data(), n, hy.data());
        seconds_ += Now() - t0;
        ++calls_;
        if (hipMemcpyAsync(y, hy.data(), count * sizeof(float),
                           hipMemcpyHostToDevice, ctx.stream) != hipSuccess ||
            hipStreamSynchronize(ctx.stream) != hipSuccess)
          throw std::runtime_error("activation upload failed");
      } catch (const std::exception& e) {
        last_error_ = "layer " + std::to_string(il) + ": " + e.what();
        return false;
      }
      return true;
    };
    return ops;
  }
  double seconds_{0};
  int calls_{0};
  std::string last_error_;
};

// ---- Metrics ----

std::vector<double> LogSoftmax(const float* x, std::size_t n) {
  double m = x[0];
  for (std::size_t i = 1; i < n; ++i)
    m = std::max<double>(m, x[i]);
  double s = 0.0;
  for (std::size_t i = 0; i < n; ++i)
    s += std::exp(x[i] - m);
  const double l = std::log(s);
  std::vector<double> r(n);
  for (std::size_t i = 0; i < n; ++i)
    r[i] = x[i] - m - l;
  return r;
}

std::vector<std::uint32_t> TopK(const float* x, std::size_t n, std::size_t k) {
  std::vector<std::uint32_t> ids(n);
  std::iota(ids.begin(), ids.end(), 0U);
  std::partial_sort(ids.begin(), ids.begin() + k, ids.end(),
                    [&](auto a, auto b) {
                      return x[a] == x[b] ? a < b : x[a] > x[b];
                    });
  ids.resize(k);
  return ids;
}

double RelL2(const float* got, const float* want, std::size_t n) {
  double dd = 0.0, ww = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double d = static_cast<double>(got[i]) - want[i];
    dd += d * d;
    ww += static_cast<double>(want[i]) * want[i];
  }
  return std::sqrt(dd / ww);
}

using TapStore = std::map<std::string, std::vector<float>>;

glm::TapFn Capture(TapStore& store) {
  return [&store](std::string_view name, std::uint32_t layer,
                  const glm::TapView& v) {
    if (layer >= 45 || (name != "out0" && name != "self_attn.out" &&
                        name != "mlp.out"))
      return;
    std::vector<float> host(v.count());
    if (hipDeviceSynchronize() != hipSuccess ||
        hipMemcpy(host.data(), v.device, host.size() * sizeof(float),
                  hipMemcpyDeviceToHost) != hipSuccess)
      throw std::runtime_error("tap download failed");
    auto& dst = store[std::string(name) + "-" + std::to_string(layer)];
    dst.insert(dst.end(), host.begin(), host.end());
  };
}

struct RunResult {
  std::vector<float> logits;  ///< [5][V]
  TapStore taps;
  double seconds{0};
};

bool RunOnce(glm::Executor& exec, int variant, std::size_t vocab,
             RunResult* r, std::string* error) {
  const double t0 = Now();
  const glm::TapFn tap = Capture(r->taps);
  glm::ForwardResult out;
  if (!exec.Reset()) {
    *error = "reset failed";
    return false;
  }
  const std::span<const std::int32_t> ids(kIds);
  auto append = [&](const glm::ForwardResult& o) {
    r->logits.insert(r->logits.end(), o.logits.begin(), o.logits.end());
  };
  if (variant == 0) {
    if (!exec.Forward(ids, true, &out, &tap, error))
      return false;
    append(out);
  } else {
    if (!exec.Forward(ids.first(kT - 1), true, &out, &tap, error))
      return false;
    append(out);
    if (!exec.Forward(ids.subspan(kT - 1), true, &out, &tap, error))
      return false;
    append(out);
  }
  r->seconds = Now() - t0;
  if (r->logits.size() != kT * vocab) {
    *error = "unexpected logits size " + std::to_string(r->logits.size());
    return false;
  }
  return true;
}

/// Bytes of GTT in use (the unified-memory pool the weights are uploaded to).
std::uint64_t GttUsed() {
  std::ifstream f("/sys/class/drm/card0/device/mem_info_gtt_used");
  std::uint64_t v = 0;
  f >> v;
  return v;
}

/// The upload copies pageable GGUF pages with hipMemcpy. On this box the page
/// cache of the mapped shards plus the GTT allocation exhaust RAM (96 GB of
/// weights in 121 GB) and the upload stalls in reclaim. While `stop` is
/// clear this thread drops the clean mapped shard pages every few seconds
/// (MADV_DONTNEED on read-only file maps: pages simply fault back in) and logs
/// progress to stderr; if GTT does not grow for `stall_s` seconds it sends
/// SIGTERM to this process (a clean exit that frees the GTT).
void UploadGuard(const std::atomic<bool>& stop, double stall_s) {
  std::uint64_t best = 0;
  double best_at = Now();
  const double t0 = Now();
  while (!stop.load()) {
    std::ifstream maps("/proc/self/maps");
    std::string line;
    std::size_t dropped = 0;
    while (std::getline(maps, line)) {
      if (line.find(".gguf") == std::string::npos ||
          line.find(" r--") == std::string::npos)
        continue;
      unsigned long lo = 0, hi = 0;
      if (std::sscanf(line.c_str(), "%lx-%lx", &lo, &hi) == 2 &&
          madvise(reinterpret_cast<void*>(lo), hi - lo, MADV_DONTNEED) == 0)
        dropped += hi - lo;
    }
    const std::uint64_t g = GttUsed();
    if (g > best + (64ULL << 20)) {
      best = g;
      best_at = Now();
    }
    std::fprintf(stderr, "[upload %.0f s] gtt %.1f GiB, madvise %.0f GiB mapped\n",
                 Now() - t0, static_cast<double>(g) / (1ULL << 30),
                 static_cast<double>(dropped) / (1ULL << 30));
    if (Now() - best_at > stall_s) {
      std::fprintf(stderr, "[upload] GTT stalled %.0f s, sending SIGTERM\n",
                   Now() - best_at);
      std::raise(SIGTERM);
      return;
    }
    for (int i = 0; i < 50 && !stop.load(); ++i)
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

int FfnTime(const glm::ModelWeights& weights, const std::string& v3) {
  const glm::Config& c = weights.config;
  const std::uint32_t il = 3;
  const glm::LayerWeights& l = weights.layers[il];
  FfnMats m;
  m.dense = false;
  m.clamp_exp = il < c.swiglu_clamp_exp.size() ? c.swiglu_clamp_exp[il] : 0.0F;
  m.clamp_shexp =
      il < c.swiglu_clamp_shexp.size() ? c.swiglu_clamp_shexp[il] : 0.0F;
  m.gate = FromHost(l.shexp_gate);
  m.up = FromHost(l.shexp_up);
  m.down = FromHost(l.shexp_down);
  m.router = FromHost(l.router);
  m.bias = FromHost(l.router_bias);
  m.gate_exps = FromHost(l.ffn_gate_exps);
  m.up_exps = FromHost(l.ffn_up_exps);
  m.down_exps = FromHost(l.ffn_down_exps);
  const std::size_t H = c.hidden_size;
  const auto x = ReadF32(v3 + "/gufo/post_attention_layernorm.out-3.f32", kT * H);
  const auto want = ReadF32(v3 + "/gufo/mlp.out-3.f32", kT * H);
  for (std::uint32_t n : {5U, 1U, 5U}) {
    std::vector<float> y(n * H);
    const double t0 = Now();
    HostFfn(c, m, x.data(), n, y.data());
    std::printf("ffntime layer 3, %u tokens: %.2f s, rel-L2 vs V3 mlp.out-3 "
                "%.3e\n",
                n, Now() - t0, RelL2(y.data(), want.data(), n * H));
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: " << argv[0]
              << " MODEL.gguf OUTDIR hostffn|laneb|ffntime\n";
    return 2;
  }
  const std::string out_dir = argv[2];
  const std::string mode = argv[3];
  const char* env = std::getenv("V3_DIR");
  const std::string v3 = env ? env : "/server/staging/glm53-gufo/v3";
  if (mode != "hostffn" && mode != "laneb" && mode != "ffntime") {
    std::cerr << "mode must be hostffn, laneb or ffntime\n";
    return 2;
  }
#if !defined(GLM53_LANE_B_OPS)
  if (mode == "laneb") {
    std::cerr << "laneb: built without GLM53_LANE_B_OPS\n";
    return 3;
  }
#endif
  try {
    std::string error;
    auto reader = gufo::core::GgufReader::OpenFile(argv[1], &error);
    if (!reader) {
      std::cerr << "open failed: " << error << "\n";
      return 1;
    }
    auto weights =
        glm::ModelWeights::Bind(*reader, /*bind_nextn=*/false, &error);
    if (!weights) {
      std::cerr << "bind failed: " << error << "\n";
      return 1;
    }
    if (mode == "ffntime")
      return FfnTime(*weights, v3);

    const glm::Config& c = weights->config;
    const std::size_t V = c.vocab_size;
    const auto llama = ReadF32(v3 + "/llama/result_output.f32", kT * V);
    const auto cpu = ReadF32(v3 + "/gufo/logits.f32", kT * V);

    HostFfnOps host;
    glm::LayerOps ops;
    if (mode == "hostffn") {
      ops = host.Bind();
    } else {
#if defined(GLM53_LANE_B_OPS)
      ops = glm::LaneBOps();
#endif
    }
    glm::ExecutorOptions options;
    options.first_layer = 0;
    options.last_layer = 0;  // every trunk layer
    options.embed = true;
    options.head = true;
    options.max_tokens = 8;
    options.capacity = 16;
    std::size_t free0 = 0, total = 0;
    (void)hipMemGetInfo(&free0, &total);
    std::cout << "device free " << free0 / (1 << 20) << " MiB of "
              << total / (1 << 20) << " MiB\n";
    const double up0 = Now();
    std::atomic<bool> upload_done{false};
    std::thread guard(UploadGuard, std::cref(upload_done), 300.0);
    auto exec = glm::Executor::Create(*weights, options, ops, &error);
    upload_done = true;
    guard.join();
    if (!exec) {
      std::cout << "FAIL create: " << error << "\n";
      return 1;
    }
    std::cout << "uploaded " << exec->DeviceBytes() / (1 << 20) << " MiB in "
              << Now() - up0 << " s\n"
              << std::flush;

    RunResult runs[2];
    for (int v = 0; v < 2; ++v) {
      if (!RunOnce(*exec, v, V, &runs[v], &error)) {
        std::cout << "FAIL run " << "ab"[v] << ": " << error << " ("
                  << host.last_error_ << ")\n";
        return 1;
      }
      std::cout << "run " << "ab"[v] << " " << runs[v].seconds
                << " s (host ffn total " << host.seconds_ << " s, "
                << host.calls_ << " calls)\n"
                << std::flush;
      WriteF32(out_dir + "/" + mode + "-" + "ab"[v] + ".f32", runs[v].logits);
    }

    // ---- Verdict ----
    bool top1_all = true, top5_all = true;
    double max_kl = 0.0, max_rel_ref = 0.0;
    std::ostringstream json;
    json << "{\"mode\":\"" << mode << "\",\"ids\":[785,6722,315,9621,374],"
         << "\"positions\":[";
    std::printf("%-3s %-8s %-8s %-5s %-10s %-10s %-10s %-10s\n", "pos",
                "top1_g", "top1_l", "top5", "KL", "rel_ref", "rel_llama",
                "a_vs_b");
    for (std::size_t t = 0; t < kT; ++t) {
      const float* g = runs[0].logits.data() + (t * V);
      const float* l = llama.data() + (t * V);
      const float* r = cpu.data() + (t * V);
      const auto tg = TopK(g, V, 5);
      const auto tl = TopK(l, V, 5);
      std::size_t overlap = 0;
      for (auto id : tg)
        overlap += std::find(tl.begin(), tl.end(), id) != tl.end();
      const auto lg = LogSoftmax(g, V);
      const auto ll = LogSoftmax(l, V);
      double kl = 0.0;
      for (std::size_t i = 0; i < V; ++i)
        kl += std::exp(ll[i]) * (ll[i] - lg[i]);
      const double rel_ref = RelL2(g, r, V);
      const double rel_llama = RelL2(g, l, V);
      const double rel_ab = RelL2(runs[1].logits.data() + (t * V), g, V);
      top1_all = top1_all && tg[0] == tl[0];
      top5_all = top5_all && overlap == 5;
      max_kl = std::max(max_kl, kl);
      max_rel_ref = std::max(max_rel_ref, rel_ref);
      std::printf("%-3zu %-8u %-8u %zu/5   %-10.3e %-10.3e %-10.3e %-10.3e\n",
                  t, tg[0], tl[0], overlap, kl, rel_ref, rel_llama, rel_ab);
      json << (t ? "," : "") << "{\"pos\":" << t << ",\"top1_gufo\":" << tg[0]
           << ",\"top1_llama\":" << tl[0] << ",\"top5_overlap\":" << overlap
           << ",\"kl\":" << kl << ",\"rel_ref\":" << rel_ref
           << ",\"rel_llama\":" << rel_llama << ",\"rel_a_vs_b\":" << rel_ab
           << "}";
    }
    const double rel_ab_all =
        RelL2(runs[1].logits.data(), runs[0].logits.data(), kT * V);
    // First diverging tap vs the V3 ReferenceModel taps (run a, forward order).
    std::string first = "none";
    json << "],\"taps\":[";
    bool first_json = true;
    bool diverged = false;
    for (std::uint32_t il = 0; il < c.num_layers; ++il) {
      for (const char* name : {"self_attn.out", "mlp.out", "out0"}) {
        const std::string key = std::string(name) + "-" + std::to_string(il);
        const auto it = runs[0].taps.find(key);
        if (it == runs[0].taps.end())
          continue;
        const auto want = ReadF32(v3 + "/gufo/" + key + ".f32", it->second.size());
        const double rel = RelL2(it->second.data(), want.data(), want.size());
        json << (first_json ? "" : ",") << "{\"tap\":\"" << key
             << "\",\"rel\":" << rel << "}";
        first_json = false;
        if (!diverged && !(rel <= kTapRel)) {
          diverged = true;
          first = key + " rel " + std::to_string(rel);
        }
      }
    }
    const bool p1 = top1_all, p2 = top5_all, p3 = max_kl <= kMaxKl,
               p4 = max_rel_ref <= kMaxRelRef, p5 = rel_ab_all <= kMaxRelRun;
    std::printf("prefill vs prefill+decode rel-L2 (all positions): %.3e\n",
                rel_ab_all);
    std::printf("top-1 5/5: %s\ntop-5 sets identical 5/5: %s\n"
                "max KL <= 1e-2: %s (%.3e)\nrel-L2 vs ReferenceModel <= 1e-3: "
                "%s (max %.3e)\nprefill vs decode <= 1e-5: %s\n",
                p1 ? "PASS" : "FAIL", p2 ? "PASS" : "FAIL", p3 ? "PASS" : "FAIL",
                max_kl, p4 ? "PASS" : "FAIL", max_rel_ref, p5 ? "PASS" : "FAIL");
    std::printf("first tap with rel > 1e-3 vs V3 ReferenceModel taps (run a): "
                "%s\n",
                first.c_str());
    const bool pass = p1 && p2 && p3 && p4 && p5;
    json << "],\"first_diverging_tap\":\"" << first << "\",\"max_kl\":" << max_kl
         << ",\"max_rel_ref\":" << max_rel_ref
         << ",\"rel_a_vs_b\":" << rel_ab_all << ",\"criteria\":{\"top1\":"
         << (p1 ? "true" : "false") << ",\"top5\":" << (p2 ? "true" : "false")
         << ",\"kl\":" << (p3 ? "true" : "false")
         << ",\"rel_ref\":" << (p4 ? "true" : "false")
         << ",\"prefill_vs_decode\":" << (p5 ? "true" : "false")
         << "},\"pass\":" << (pass ? "true" : "false") << "}\n";
    std::ofstream(out_dir + "/compare.json") << json.str();
    std::cout << (pass ? "PASS" : "FAIL") << "\n";
    return pass ? 0 : 1;
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << "\n";
    return 1;
  }
}
