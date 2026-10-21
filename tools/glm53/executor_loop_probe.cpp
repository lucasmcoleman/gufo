// A3 layer-loop probe: the GPU Executor (executor.hpp) on layers 0-2 plus the
// embedding of a real GGUF against the CPU ReferenceModel, for the prompt
// "The capital of France is" = [785, 6722, 315, 9621, 374]. Layers 0-2 are
// KDA with the dense FFN (Q6_K), so this covers the Q6_K embedding gather,
// HcPre / HcPost, the RMSNorms and the KDA mixer end to end; the FFN is the
// host stand-in of tests/models/glm53_flash/host_ops.hpp (weights read back
// from the uploaded FfnLayerDevice).
//
// usage: glm53_flash_executor_loop_probe MODEL.gguf OUTDIR cpu|gpu|all
//   cpu  runs the ReferenceModel through layer 2 (a tap stores every layer
//        0-2 tap as OUTDIR/<layer>-<name>.f32, then stops the forward after
//        layer 2's out0).
//   gpu  reads those, uploads layers [0,3) and the embedding, and runs the
//        loop twice: a 5-token prefill, then (after Reset) a 4-token prefill
//        and a 1-token decode. Prints rel-L2 and cosine per layer and tap,
//        per token for `in` / `out0`, and writes OUTDIR/parity.json.
//   all  both, in one process.
// The split keeps the slow CPU phase outside the GPU admission lock.
// Exit 0 when every `in` / `out0` token has rel-L2 <= 1e-4 and cosine >=
// 0.99999.

#include <hip/hip_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/glm53_flash/executor.hpp"
#include "src/models/glm53_flash/reference.hpp"
#include "src/models/glm53_flash/weights.hpp"
#include "tests/models/glm53_flash/host_ops.hpp"

namespace glm = gufo::models::glm53_flash;
namespace ref = gufo::models::glm53_flash::reference;

namespace {

const std::vector<std::int32_t> kIds = {785, 6722, 315, 9621, 374};
constexpr std::uint32_t kLayers = 3;
constexpr double kMaxRel = 1e-4;
constexpr double kMinCos = 0.99999;
// ReferenceModel's layer taps, in forward order, with their per-token width.
enum class Width { kHc, kComb, kHidden, kStreams };
const std::vector<std::pair<std::string, Width>> kTaps = {
    {"in", Width::kStreams},
    {"attn_hc.out0", Width::kHc},
    {"attn_hc.out1", Width::kComb},
    {"attn_hc.out2", Width::kHidden},
    {"input_layernorm.out", Width::kHidden},
    {"self_attn.out", Width::kHidden},
    {"ffn_hc.in", Width::kStreams},
    {"ffn_hc.out0", Width::kHc},
    {"ffn_hc.out1", Width::kComb},
    {"ffn_hc.out2", Width::kHidden},
    {"post_attention_layernorm.out", Width::kHidden},
    {"mlp.out", Width::kHidden},
    {"out0", Width::kStreams}};

struct StopForward {};

std::size_t TapWidth(const glm::Config& c, Width w) {
  switch (w) {
    case Width::kHc:
      return c.hc_count;
    case Width::kComb:
      return std::size_t{c.hc_count} * c.hc_count;
    case Width::kStreams:
      return c.HcDim();
    case Width::kHidden:
      break;
  }
  return c.hidden_size;
}

void CheckHip(hipError_t error, const char* what) {
  if (error != hipSuccess) {
    throw std::runtime_error(std::string(what) + ": " +
                             hipGetErrorString(error));
  }
}

void WriteF32(const std::string& path, const float* v, std::size_t n) {
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(v),
            static_cast<std::streamsize>(n * sizeof(float)));
  if (!out)
    throw std::runtime_error("cannot write " + path);
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

std::string TapPath(const std::string& dir, std::uint32_t layer,
                    const std::string& name) {
  return dir + "/" + std::to_string(layer) + "-" + name + ".f32";
}

struct Stats {
  double cos{0.0};
  double rel{0.0};
};

Stats Compare(const float* got, const float* want, std::size_t n) {
  double dot = 0.0;
  double gg = 0.0;
  double ww = 0.0;
  double dd = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double g = got[i];
    const double w = want[i];
    dot += g * w;
    gg += g * g;
    ww += w * w;
    dd += (g - w) * (g - w);
  }
  return {dot / std::sqrt(gg * ww), std::sqrt(dd / ww)};
}

void RunCpu(const glm::ModelWeights& weights, const std::string& dir) {
  ref::GgufWeightSource source(weights);
  ref::ReferenceModel model(weights.config, source);
  const auto t0 = std::chrono::steady_clock::now();
  const ref::TapFn tap = [&](std::string_view name, int layer,
                             const float* data, std::size_t width,
                             std::uint32_t n) {
    if (layer < 0 || layer >= static_cast<int>(kLayers))
      return;
    WriteF32(TapPath(dir, static_cast<std::uint32_t>(layer), std::string(name)),
             data, width * n);
    if (name == "out0") {
      std::cerr << "cpu layer " << layer << " done at "
                << std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - t0)
                       .count()
                << " s\n";
      if (layer + 1 == static_cast<int>(kLayers))
        throw StopForward{};
    }
  };
  try {
    (void)model.Forward(kIds, false, tap);
    throw std::runtime_error("ReferenceModel did not reach layer 2");
  } catch (const StopForward&) {
  }
  model.Reset();
}

using TapStore = std::map<std::string, std::vector<float>>;

/// Appends each tap's rows, so taps of consecutive calls concatenate in
/// token order.
glm::TapFn Capture(TapStore& store) {
  return [&store](std::string_view name, std::uint32_t layer,
                  const glm::TapView& v) {
    if (layer >= kLayers)
      return;
    std::vector<float> host(v.count());
    CheckHip(hipDeviceSynchronize(), "tap sync");
    CheckHip(hipMemcpy(host.data(), v.device, host.size() * sizeof(float),
                       hipMemcpyDeviceToHost),
             "tap download");
    auto& dst = store[std::to_string(layer) + "-" + std::string(name)];
    dst.insert(dst.end(), host.begin(), host.end());
  };
}

bool RunGpu(const glm::ModelWeights& weights, const std::string& dir) {
  const glm::Config& c = weights.config;
  const auto T = static_cast<std::uint32_t>(kIds.size());
  std::size_t free0 = 0;
  std::size_t total = 0;
  CheckHip(hipMemGetInfo(&free0, &total), "hipMemGetInfo");
  std::cout << "device free " << free0 / (1 << 20) << " MiB of "
            << total / (1 << 20) << " MiB\n";
  std::cout << "token_embd type " << static_cast<int>(weights.token_embd.type)
            << ", layer 0 ffn_gate type "
            << static_cast<int>(weights.layers[0].ffn_gate.type) << "\n";

  glm::stand_in::HostOps ops(c);
  glm::ExecutorOptions options;
  options.first_layer = 0;
  options.last_layer = kLayers;
  options.embed = true;
  options.head = false;
  options.max_tokens = 8;
  options.capacity = 16;
  std::string error;
  const auto up0 = std::chrono::steady_clock::now();
  auto exec = glm::Executor::Create(weights, options, ops.Bind(), &error);
  if (!exec) {
    std::cout << "FAIL create: " << error << "\n";
    return false;
  }
  std::cout << "uploaded " << exec->DeviceBytes() / (1 << 20) << " MiB in "
            << std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                             up0)
                   .count()
            << " s\n";

  bool pass = true;
  std::ostringstream json;
  json << "{\"ids\":[785,6722,315,9621,374],\"modes\":[";
  for (int mode = 0; mode < 2; ++mode) {
    const char* label = mode == 0 ? "prefill5" : "prefill4+decode1";
    const auto run0 = std::chrono::steady_clock::now();
    TapStore taps;
    const glm::TapFn tap = Capture(taps);
    glm::ForwardResult out;
    bool ok = exec->Reset();
    if (mode == 0) {
      ok = ok && exec->Forward(kIds, false, &out, &tap, &error);
    } else {
      const std::span<const std::int32_t> ids(kIds);
      ok = ok && exec->Forward(ids.first(T - 1), false, &out, &tap, &error) &&
           exec->Forward(ids.subspan(T - 1), false, &out, &tap, &error);
    }
    if (!ok) {
      std::cout << "FAIL " << label << ": " << error << " (" << ops.last_error()
                << ")\n";
      return false;
    }
    std::cout << label << " ran in "
              << std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - run0)
                     .count()
              << " s\n";
    json << (mode ? "," : "") << "{\"mode\":\"" << label << "\",\"layers\":[";
    for (std::uint32_t il = 0; il < kLayers; ++il) {
      json << (il ? "," : "") << "{\"layer\":" << il << ",\"taps\":{";
      bool first_tap = true;
      for (const auto& [name, kind] : kTaps) {
        const std::size_t width = TapWidth(c, kind);
        const auto want = ReadF32(TapPath(dir, il, name), T * width);
        const auto it = taps.find(std::to_string(il) + "-" + name);
        if (it == taps.end() || it->second.size() != want.size()) {
          std::cout << "FAIL " << label << " layer " << il << " tap " << name
                    << " missing or wrong size\n";
          pass = false;
          continue;
        }
        const std::vector<float>& got = it->second;
        const Stats all = Compare(got.data(), want.data(), want.size());
        const bool gated = name == "in" || name == "out0";
        char line[200];
        std::snprintf(
            line, sizeof line, "%s %-16s layer %u %-28s cos %.9f rel_l2 %.3e\n",
            gated ? "gate" : "info", label, il, name.c_str(), all.cos, all.rel);
        std::cout << line;
        json << (first_tap ? "" : ",") << "\"" << name
             << "\":{\"cos\":" << all.cos << ",\"rel\":" << all.rel;
        first_tap = false;
        if (gated) {
          json << ",\"tokens\":[";
          for (std::uint32_t t = 0; t < T; ++t) {
            const Stats s = Compare(got.data() + (t * width),
                                    want.data() + (t * width), width);
            const bool tok_ok =
                std::isfinite(s.rel) && s.rel <= kMaxRel && s.cos >= kMinCos;
            pass = pass && tok_ok;
            std::snprintf(
                line, sizeof line,
                "%s %-16s layer %u %-5s token %u%s cos %.9f rel_l2 %.3e\n",
                tok_ok ? "ok  " : "FAIL", label, il, name.c_str(), t,
                mode == 1 && t + 1 == T ? " (decode)" : "         ", s.cos,
                s.rel);
            std::cout << line;
            json << (t ? "," : "") << "{\"token\":" << t << ",\"cos\":" << s.cos
                 << ",\"rel\":" << s.rel << "}";
          }
          json << "]";
        }
        json << "}";
      }
      json << "}}";
    }
    json << "]}";
  }
  std::size_t free1 = 0;
  CheckHip(hipMemGetInfo(&free1, &total), "hipMemGetInfo");
  std::cout << "device free after " << free1 / (1 << 20) << " MiB; ffn calls "
            << ops.ffn_calls() << "\n";
  json << "],\"max_rel\":" << kMaxRel << ",\"min_cos\":" << kMinCos
       << ",\"pass\":" << (pass ? "true" : "false") << "}\n";
  std::ofstream(dir + "/parity.json") << json.str();
  return pass;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: " << argv[0] << " MODEL.gguf OUTDIR cpu|gpu|all\n";
    return 2;
  }
  const std::string dir = argv[2];
  const std::string mode = argv[3];
  if (mode != "cpu" && mode != "gpu" && mode != "all") {
    std::cerr << "mode must be cpu, gpu or all\n";
    return 2;
  }
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
    for (std::uint32_t il = 0; il < kLayers; ++il) {
      if (!weights->config.IsKdaLayer(il) ||
          !weights->config.IsDenseFfnLayer(il)) {
        std::cerr << "layer " << il << " is not a KDA + dense FFN layer\n";
        return 1;
      }
    }
    if (mode != "gpu")
      RunCpu(*weights, dir);
    if (mode != "cpu") {
      const bool pass = RunGpu(*weights, dir);
      std::cout << (pass ? "PASS" : "FAIL") << "\n";
      return pass ? 0 : 1;
    }
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
