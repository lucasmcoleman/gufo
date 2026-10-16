// V3 probe: runs the GLM-5.3-Flash CPU ReferenceModel on a real GGUF for a
// fixed token sequence and writes every tap as raw float32, so it can be
// compared with llama.cpp's cb_eval dump (see /server/staging/glm53-gufo/v3).
//
// usage: glm53_flash_reference_probe MODEL.gguf OUTDIR id0 id1 ...
// Writes OUTDIR/<tap>-<layer>.f32 (layer taps), OUTDIR/<tap>.f32 (global
// taps), OUTDIR/logits.f32 [T][vocab], OUTDIR/hidden.f32 [T][hidden], and
// OUTDIR/index.json (name, layer, width, n_tokens, seconds since start).

#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/glm53_flash/reference.hpp"
#include "src/models/glm53_flash/weights.hpp"

namespace glm = gufo::models::glm53_flash;
namespace ref = gufo::models::glm53_flash::reference;

namespace {

void WriteF32(const std::string& path, const float* data, std::size_t count) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (f == nullptr) {
    throw std::runtime_error("cannot write " + path);
  }
  std::fwrite(data, sizeof(float), count, f);
  std::fclose(f);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "usage: " << argv[0] << " MODEL.gguf OUTDIR ids...\n";
    return 2;
  }
  const std::string dir = argv[2];
  std::vector<std::int32_t> ids;
  for (int i = 3; i < argc; ++i) {
    ids.push_back(std::atoi(argv[i]));
  }
  const auto t0 = std::chrono::steady_clock::now();
  const auto secs = [&] {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
        .count();
  };
  try {
    std::string error;
    auto reader = gufo::core::GgufReader::OpenFile(argv[1], &error);
    if (!reader) {
      std::cerr << "open failed: " << error << "\n";
      return 1;
    }
    auto weights = glm::ModelWeights::Bind(*reader, /*bind_nextn=*/false, &error);
    if (!weights) {
      std::cerr << "bind failed: " << error << "\n";
      return 1;
    }
    ref::GgufWeightSource source(*weights);
    ref::ReferenceModel model(weights->config, source);
    std::cerr << "bound in " << secs() << " s\n";

    std::vector<std::string> index;
    const ref::TapFn tap = [&](std::string_view name, int layer,
                               const float* data, std::size_t width,
                               std::uint32_t n) {
      std::string file(name);
      if (layer >= 0) {
        file += "-" + std::to_string(layer);
      }
      file += ".f32";
      WriteF32(dir + "/" + file, data, width * n);
      char line[512];
      std::snprintf(line, sizeof line,
                    "{\"name\":\"%.*s\",\"layer\":%d,\"width\":%zu,"
                    "\"n_tokens\":%u,\"t\":%.2f,\"file\":\"%s\"}",
                    static_cast<int>(name.size()), name.data(), layer, width, n,
                    secs(), file.c_str());
      index.emplace_back(line);
      if (name == "out0") {
        std::cerr << "layer " << layer << " done at " << secs() << " s\n";
      }
    };
    const auto out = model.Forward(ids, /*want_logits=*/true, tap);
    WriteF32(dir + "/hidden.f32", out.hidden.data(), out.hidden.size());
    WriteF32(dir + "/logits.f32", out.logits.data(), out.logits.size());
    FILE* f = std::fopen((dir + "/index.json").c_str(), "w");
    std::fprintf(f, "{\"ids\":[");
    for (std::size_t i = 0; i < ids.size(); ++i) {
      std::fprintf(f, "%s%d", i ? "," : "", ids[i]);
    }
    std::fprintf(f, "],\"vocab\":%u,\"hidden\":%u,\"total_s\":%.2f,\"taps\":[\n",
                 weights->config.vocab_size, weights->config.hidden_size,
                 secs());
    for (std::size_t i = 0; i < index.size(); ++i) {
      std::fprintf(f, "%s%s\n", index[i].c_str(),
                   i + 1 < index.size() ? "," : "");
    }
    std::fprintf(f, "]}\n");
    std::fclose(f);
    std::cerr << "done in " << secs() << " s, logits " << out.logits.size()
              << "\n";
  } catch (const std::exception& e) {
    std::cerr << "FAILED at " << secs() << " s: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
