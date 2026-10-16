// A3 probe: GPU KdaMixerForward (executor.hpp) against the CPU ReferenceModel
// on the KDA layers 0-2 of a real GGUF, for the prompt
// "The capital of France is" = [785, 6722, 315, 9621, 374].
//
// usage: glm53_flash_kda_executor_probe MODEL.gguf OUTDIR cpu|gpu|all
//   cpu  runs the ReferenceModel through layer 2's mixer (a tap captures
//        layer 0-2 `input_layernorm.out` and `self_attn.out`, then stops the
//        forward) and writes OUTDIR/x-<L>.f32, OUTDIR/y-<L>.f32 [T][hidden].
//   gpu  reads those, uploads one layer at a time, and runs the GPU mixer
//        twice per layer: a 5-token prefill, and a 4-token prefill followed
//        by a 1-token decode on the same state. Prints cosine and rel-L2 per
//        token against y and writes OUTDIR/parity.json.
//   all  both, in one process.
// The split keeps the slow CPU phase outside the GPU admission lock.
// Exit 0 when every token has rel-L2 <= 1e-4 and cosine >= 0.99999.

#include <hip/hip_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/glm53_flash/executor.hpp"
#include "src/models/glm53_flash/reference.hpp"
#include "src/models/glm53_flash/weights.hpp"

namespace glm = gufo::models::glm53_flash;
namespace ref = gufo::models::glm53_flash::reference;

namespace {

const std::vector<std::int32_t> kIds = {785, 6722, 315, 9621, 374};
constexpr std::uint32_t kLayers = 3;
constexpr double kMaxRel = 1e-4;
constexpr double kMinCos = 0.99999;

struct StopForward {};

void CheckHip(hipError_t error, const char* what) {
  if (error != hipSuccess) {
    throw std::runtime_error(std::string(what) + ": " +
                             hipGetErrorString(error));
  }
}

void WriteF32(const std::string& path, const std::vector<float>& v) {
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(v.data()),
            static_cast<std::streamsize>(v.size() * sizeof(float)));
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
    const std::vector<float> v(data, data + (width * n));
    const std::string l = std::to_string(layer);
    if (name == "input_layernorm.out")
      WriteF32(dir + "/x-" + l + ".f32", v);
    if (name == "self_attn.out") {
      WriteF32(dir + "/y-" + l + ".f32", v);
      std::cerr << "cpu layer " << layer << " mixer done at "
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

bool RunGpu(const glm::ModelWeights& weights, const std::string& dir) {
  const glm::Config& c = weights.config;
  const auto T = static_cast<std::uint32_t>(kIds.size());
  const std::size_t hidden = c.hidden_size;
  bool pass = true;
  std::ostringstream json;
  json << "{\"ids\":[785,6722,315,9621,374],\"layers\":[";
  hipStream_t stream = nullptr;
  CheckHip(hipStreamCreate(&stream), "hipStreamCreate");
  std::size_t free0 = 0;
  std::size_t total = 0;
  CheckHip(hipMemGetInfo(&free0, &total), "hipMemGetInfo");
  std::cout << "device free " << free0 / (1 << 20) << " MiB of "
            << total / (1 << 20) << " MiB\n";
  for (std::uint32_t il = 0; il < kLayers; ++il) {
    const std::string l = std::to_string(il);
    const auto x = ReadF32(dir + "/x-" + l + ".f32", T * hidden);
    const auto want = ReadF32(dir + "/y-" + l + ".f32", T * hidden);
    const auto& lw = weights.layers[il];
    std::cout << "layer " << il << " types: qkv "
              << static_cast<int>(lw.kda_q.type) << " f_a "
              << static_cast<int>(lw.kda_f_a.type) << " f_b "
              << static_cast<int>(lw.kda_f_b.type) << " beta "
              << static_cast<int>(lw.kda_beta.type) << " out "
              << static_cast<int>(lw.attn_out.type) << "\n";

    glm::DeviceArena arena;
    glm::KdaLayerDevice w;
    std::string error;
    const auto up0 = std::chrono::steady_clock::now();
    if (!glm::UploadKdaLayer(arena, c, lw, il, &w, &error)) {
      std::cout << "FAIL layer " << il << " upload: " << error << "\n";
      return false;
    }
    glm::KdaState state;
    const std::size_t scratch_bytes = glm::KdaScratchBytes(c, T);
    void* scratch = arena.Allocate(scratch_bytes);
    auto* dx = static_cast<float*>(arena.Allocate(x.size() * sizeof(float)));
    auto* dy = static_cast<float*>(arena.Allocate(x.size() * sizeof(float)));
    if (!glm::AllocateKdaState(arena, c, &state) || scratch == nullptr ||
        dx == nullptr || dy == nullptr) {
      std::cout << "FAIL layer " << il << " allocation\n";
      return false;
    }
    CheckHip(hipMemcpy(dx, x.data(), x.size() * sizeof(float),
                       hipMemcpyHostToDevice),
             "upload x");
    std::cout << "layer " << il << " uploaded " << arena.bytes() / (1 << 20)
              << " MiB in "
              << std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - up0)
                     .count()
              << " s\n";
    const glm::OpContext ctx{.stream = stream,
                             .scratch = scratch,
                             .scratch_bytes = scratch_bytes,
                             .tap = nullptr};

    json << (il ? "," : "") << "{\"layer\":" << il;
    for (int mode = 0; mode < 2; ++mode) {
      const char* label = mode == 0 ? "prefill5" : "prefill4+decode1";
      bool ok = glm::ResetKdaState(c, state, stream);
      CheckHip(hipMemsetAsync(dy, 0, x.size() * sizeof(float), stream),
               "memset");
      if (mode == 0) {
        ok = ok && glm::KdaMixerForward(ctx, c, il, w, state, dx, T, dy);
      } else {
        ok = ok && glm::KdaMixerForward(ctx, c, il, w, state, dx, T - 1, dy);
        ok = ok &&
             glm::KdaMixerForward(ctx, c, il, w, state, dx + ((T - 1) * hidden),
                                  1, dy + ((T - 1) * hidden));
      }
      CheckHip(hipStreamSynchronize(stream), "sync");
      if (!ok) {
        std::cout << "FAIL layer " << il << " " << label << " launch\n";
        return false;
      }
      std::vector<float> got(x.size());
      CheckHip(hipMemcpy(got.data(), dy, got.size() * sizeof(float),
                         hipMemcpyDeviceToHost),
               "download y");
      json << ",\"" << label << "\":[";
      for (std::uint32_t t = 0; t < T; ++t) {
        const Stats s = Compare(got.data() + (t * hidden),
                                want.data() + (t * hidden), hidden);
        const bool tok_ok =
            std::isfinite(s.rel) && s.rel <= kMaxRel && s.cos >= kMinCos;
        pass = pass && tok_ok;
        char line[160];
        std::snprintf(line, sizeof line,
                      "%s layer %u %-16s token %u%s cos %.9f rel_l2 %.3e\n",
                      tok_ok ? "ok  " : "FAIL", il, label, t,
                      mode == 1 && t + 1 == T ? " (decode)" : "         ",
                      s.cos, s.rel);
        std::cout << line;
        json << (t ? "," : "") << "{\"token\":" << t << ",\"cos\":" << s.cos
             << ",\"rel\":" << s.rel << "}";
      }
      json << "]";
    }
    json << "}";
  }
  std::size_t free1 = 0;
  CheckHip(hipMemGetInfo(&free1, &total), "hipMemGetInfo");
  std::cout << "device free after " << free1 / (1 << 20) << " MiB\n";
  CheckHip(hipStreamDestroy(stream), "hipStreamDestroy");
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
      if (!weights->config.IsKdaLayer(il)) {
        std::cerr << "layer " << il << " is not a KDA layer\n";
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
