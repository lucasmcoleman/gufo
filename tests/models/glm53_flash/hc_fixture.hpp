#ifndef GUFO_TESTS_MODELS_GLM53_FLASH_HC_FIXTURE_HPP_
#define GUFO_TESTS_MODELS_GLM53_FLASH_HC_FIXTURE_HPP_

// Loader for fixtures/hc_fixture.bin, the hyper-connection subset of the HF
// transformers 09281b8 fixture set (hyper_connection.npz; see
// fixtures/make_hc_fixture.py for provenance and format). hidden 32, 4
// streams, 2 x 5 tokens, hc_eps 1e-6, rms eps 1e-5, 20 Sinkhorn iterations.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace glm53_hc_fixture {

struct Array {
  std::vector<std::uint32_t> shape;
  std::vector<float> data;
};

inline std::map<std::string, Array> Load(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw std::runtime_error("cannot open fixture " + path);
  const std::vector<char> bytes((std::istreambuf_iterator<char>(in)),
                                std::istreambuf_iterator<char>());
  std::size_t pos = 0;
  auto take = [&](void* dst, std::size_t n) {
    if (pos + n > bytes.size())
      throw std::runtime_error("truncated fixture");
    std::memcpy(dst, bytes.data() + pos, n);
    pos += n;
  };
  auto u32 = [&] {
    std::uint32_t v = 0;
    take(&v, sizeof(v));
    return v;
  };
  char magic[8];
  take(magic, sizeof(magic));
  if (std::memcmp(magic, "GLMHCFX1", 8) != 0) {
    throw std::runtime_error("bad fixture magic in " + path);
  }
  std::map<std::string, Array> out;
  const std::uint32_t count = u32();
  for (std::uint32_t i = 0; i < count; ++i) {
    std::string name(u32(), '\0');
    take(name.data(), name.size());
    Array a;
    a.shape.resize(u32());
    std::size_t n = 1;
    for (auto& d : a.shape) {
      d = u32();
      n *= d;
    }
    a.data.resize(n);
    take(a.data.data(), n * sizeof(float));
    out.emplace(std::move(name), std::move(a));
  }
  return out;
}

/// ||got - want|| / ||want||, accumulated in double.
inline double RelL2(const float* got, const float* want, std::size_t n) {
  double num = 0.0;
  double den = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double d = static_cast<double>(got[i]) - want[i];
    num += d * d;
    den += static_cast<double>(want[i]) * want[i];
  }
  return den == 0.0 ? std::sqrt(num) : std::sqrt(num / den);
}

}  // namespace glm53_hc_fixture

#endif  // GUFO_TESTS_MODELS_GLM53_FLASH_HC_FIXTURE_HPP_
