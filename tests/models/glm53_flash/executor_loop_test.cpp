// GPU layer loop (executor.hpp Executor) against the CPU ReferenceModel on
// the e2e fixture model (fixtures/e2e_fixture.bin: hidden 32, 5 layers =
// KDA+dense x3, MLA/DSA+MoE, KDA+MoE). The GPU KDA kernels need head dim 128
// and the fixture's is 8, so its four KDA layers are redrawn (seeded) at
// 4 heads x 128; every other tensor (embedding, hyper-connections, norms,
// MLA + indexer, dense FFN, MoE) is the fixture's, plus a synthetic
// output.weight for the logits path. Q8_0 (power-of-two scales, exact):
// token_embd, output, layer 0 q|k|v and attn_output, layer 2 ffn_gate /
// ffn_up, layer 4 ffn_gate_exps. Both sides read the same TensorRefs.
// MLA and FFN run through the host stand-ins of host_ops.hpp, which read
// their weights back from the uploaded MlaLayerDevice / FfnLayerDevice.
// Checks, at rel-L2 <= kTol:
//   (a) prefill(12) + 2 single-token decodes: every ReferenceModel tap per
//       layer, the final hidden state and the logits of each call;
//   (b) the 14 tokens in one call with max_tokens 5 (passes 5+5+4);
//   (c) Reset() then the 12-token prefill again reproduces (a);
//   (d) a [0,3) range with embedding and no head, then [3,5) with the head
//       from its streams (ForwardStreams), vs the full 14-token reference;
//   (e) refusals: an MLA range without an MLA operator, the embedding with a
//       range past layer 0, and a failing operator leaving the executor
//       Failed() until Reset().

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/glm53_flash/config.hpp"
#include "src/models/glm53_flash/executor.hpp"
#include "src/models/glm53_flash/reference.hpp"
#include "src/models/glm53_flash/weights.hpp"
#include "tests/models/glm53_flash/host_ops.hpp"
#include "tests/models/glm53_flash/loop_model.hpp"

namespace glm = gufo::models::glm53_flash;
namespace ref = gufo::models::glm53_flash::reference;
using glm::loop_model::BuildModel;
using glm::loop_model::Fixture;
using glm::loop_model::Load;
using glm::loop_model::Model;

namespace {

constexpr double kTol = 1e-5;
constexpr std::uint32_t kPrefill = 12;
int failures = 0;

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

void Check(const std::string& what, bool ok) {
  std::cout << (ok ? "ok   " : "FAIL ") << what << "\n";
  if (!ok)
    ++failures;
}

void CheckRel(const std::string& what, const std::vector<float>& got,
              const std::vector<float>& want, double tol = kTol) {
  const double rel = got.size() == want.size()
                         ? RelL2(got.data(), want.data(), want.size())
                         : INFINITY;
  const bool ok = !want.empty() && std::isfinite(rel) && rel <= tol;
  std::cout << (ok ? "ok   " : "FAIL ") << what << " rel_l2 " << rel << "\n";
  if (!ok)
    ++failures;
}

/// Taps keyed "L<layer>.<name>" (or the bare name outside a layer), with
/// the order they arrived in.
struct Taps {
  std::map<std::string, std::vector<float>> data;
  std::vector<std::string> order;
  void Put(const std::string& key, const float* p, std::size_t n) {
    if (!data.contains(key))
      order.push_back(key);
    data[key].assign(p, p + n);
  }
};

std::string Key(std::string_view name, long long layer) {
  return layer < 0 ? std::string(name)
                   : "L" + std::to_string(layer) + "." + std::string(name);
}

ref::TapFn RefTap(Taps& taps) {
  return [&taps](std::string_view name, int layer, const float* data,
                 std::size_t width, std::uint32_t n) {
    taps.Put(Key(name, layer), data, width * n);
  };
}

glm::TapFn GpuTap(Taps& taps) {
  return [&taps](std::string_view name, std::uint32_t layer,
                 const glm::TapView& v) {
    if (name == "kda_out")
      return;  // operator tap, no ReferenceModel counterpart
    std::vector<float> host(v.count());
    if (hipDeviceSynchronize() != hipSuccess ||
        hipMemcpy(host.data(), v.device, host.size() * sizeof(float),
                  hipMemcpyDeviceToHost) != hipSuccess) {
      throw std::runtime_error("tap download failed");
    }
    taps.Put(Key(name, layer == glm::kNoLayer ? -1LL : layer), host.data(),
             host.size());
  };
}

/// Every reference tap against the GPU's, with the worst rel-L2 per layer.
void CompareTaps(const std::string& label, const Taps& gpu, const Taps& cpu,
                 std::uint32_t layers) {
  std::map<std::string, std::pair<double, std::string>> worst;
  std::size_t checked = 0;
  for (const std::string& key : cpu.order) {
    const auto it = gpu.data.find(key);
    const std::vector<float>& want = cpu.data.at(key);
    const std::string group =
        key[0] == 'L' ? key.substr(0, key.find('.')) : "head";
    if (it == gpu.data.end() || it->second.size() != want.size()) {
      std::cout << "FAIL " << label << " tap " << key
                << (it == gpu.data.end() ? " missing" : " size mismatch")
                << "\n";
      ++failures;
      continue;
    }
    const double r = RelL2(it->second.data(), want.data(), want.size());
    ++checked;
    auto& w = worst[group];
    if (!(r <= w.first)) {
      w = {r, group == "head" ? key : key.substr(key.find('.') + 1)};
    }
    if (!(r <= kTol)) {
      std::cout << "FAIL " << label << " tap " << key << " rel_l2 " << r
                << "\n";
      ++failures;
    }
  }
  for (std::uint32_t il = 0; il < layers; ++il) {
    const std::string l = "L" + std::to_string(il);
    const auto out0 = [&](const char* name) {
      const std::string key = l + "." + name;
      return gpu.data.contains(key) && cpu.data.contains(key)
                 ? RelL2(gpu.data.at(key).data(), cpu.data.at(key).data(),
                         cpu.data.at(key).size())
                 : INFINITY;
    };
    std::cout << "  " << label << " layer " << il << ": worst "
              << worst[l].first << " (" << worst[l].second << "), out0 "
              << out0("out0") << ", self_attn " << out0("self_attn.out")
              << ", mlp " << out0("mlp.out") << "\n";
  }
  if (worst.contains("head")) {
    std::cout << "  " << label << " head: worst " << worst["head"].first << " ("
              << worst["head"].second << ")\n";
  }
  Check(label + ": " + std::to_string(checked) + " of " +
            std::to_string(cpu.order.size()) + " taps matched",
        checked == cpu.order.size() && checked > 0);
}

glm::ExecutorOptions Options(std::uint32_t max_tokens) {
  glm::ExecutorOptions o;
  o.max_tokens = max_tokens;
  o.capacity = 64;
  return o;
}

// (a) + (c).
void CheckIncremental(const Model& m) {
  const glm::Config& c = m.config;
  const ref::GgufWeightSource source(m.weights);
  ref::ReferenceModel model(c, source);
  glm::stand_in::HostOps ops(c);
  std::string error;
  auto exec = glm::Executor::Create(m.weights, Options(16), ops.Bind(), &error);
  Check("create full executor " + error, exec != nullptr);
  if (!exec)
    return;
  std::cout << "device bytes " << exec->DeviceBytes() << "\n";
  const std::span<const std::int32_t> ids(m.ids);
  std::vector<float> first_hidden;
  std::vector<float> first_logits;
  for (std::uint32_t step = 0; step < 3; ++step) {
    const auto span =
        step == 0 ? ids.first(kPrefill) : ids.subspan(kPrefill + step - 1, 1);
    const std::string label =
        step == 0 ? "prefill(12)"
                  : "decode token " + std::to_string(kPrefill + step - 1);
    Taps cpu_taps;
    Taps gpu_taps;
    const auto want = model.Forward(span, true, RefTap(cpu_taps));
    glm::ForwardResult got;
    const glm::TapFn tap = GpuTap(gpu_taps);
    const bool ok = exec->Forward(span, true, &got, &tap, &error);
    Check(label + " forward " + error, ok);
    if (!ok)
      return;
    CompareTaps(label, gpu_taps, cpu_taps, c.num_layers);
    CheckRel(label + " final hidden", got.hidden, want.hidden);
    CheckRel(label + " logits", got.logits, want.logits);
    if (step == 0) {
      first_hidden = got.hidden;
      first_logits = got.logits;
    }
  }
  Check("position after prefill + 2 decodes", exec->Position() == 14);
  Check("stand-ins ran (mla " + std::to_string(ops.mla_calls()) + ", ffn " +
            std::to_string(ops.ffn_calls()) + ")",
        ops.mla_calls() == 3 && ops.ffn_calls() == 15);

  // (c) Reset, then the same prefill without taps.
  Check("reset", exec->Reset() && exec->Position() == 0);
  glm::ForwardResult again;
  Check("prefill after reset " + error,
        exec->Forward(ids.first(kPrefill), true, &again, nullptr, &error));
  CheckRel("prefill after reset vs first prefill (hidden)", again.hidden,
           first_hidden, 1e-6);
  CheckRel("prefill after reset vs first prefill (logits)", again.logits,
           first_logits, 1e-6);
}

// (b) + (d).
void CheckPassesAndRanges(const Model& m) {
  const glm::Config& c = m.config;
  const ref::GgufWeightSource source(m.weights);
  ref::ReferenceModel model(c, source);
  Taps cpu_taps;
  const auto want = model.Forward(m.ids, true, RefTap(cpu_taps));
  std::string error;

  {
    glm::stand_in::HostOps ops(c);
    auto exec =
        glm::Executor::Create(m.weights, Options(5), ops.Bind(), &error);
    glm::ForwardResult got;
    const bool ok = exec && exec->Forward(m.ids, true, &got, nullptr, &error);
    Check("14 tokens in passes of 5 " + error, ok);
    CheckRel("passes of 5: final hidden", got.hidden, want.hidden);
    CheckRel("passes of 5: logits", got.logits, want.logits);
  }

  glm::stand_in::HostOps ops(c);
  glm::ExecutorOptions front = Options(16);
  front.last_layer = 3;
  front.head = false;
  auto a = glm::Executor::Create(m.weights, front, ops.Bind(), &error);
  Check("create [0,3) with embedding, no head " + error, a != nullptr);
  glm::ExecutorOptions back = Options(16);
  back.first_layer = 3;
  back.embed = false;
  auto b = glm::Executor::Create(m.weights, back, ops.Bind(), &error);
  Check("create [3,5) with head " + error, b != nullptr);
  if (!a || !b)
    return;
  std::cout << "device bytes [0,3) " << a->DeviceBytes() << ", [3,5) "
            << b->DeviceBytes() << "\n";
  glm::ForwardResult mid;
  Taps gpu_taps;
  const glm::TapFn tap = GpuTap(gpu_taps);
  Check("[0,3) forward " + error, a->Forward(m.ids, false, &mid, &tap, &error));
  CheckRel("[0,3) streams vs reference layer 2 out0", mid.streams,
           cpu_taps.data.at("L2.out0"));
  Check("[0,3) reports no head", mid.hidden.empty() && mid.logits.empty());
  Check("[0,3) taps only its layers",
        !gpu_taps.data.contains("L3.in") && gpu_taps.data.contains("L2.out0"));
  glm::ForwardResult end;
  Check("[3,5) forward from streams " + error,
        b->ForwardStreams(mid.streams.data(),
                          static_cast<std::uint32_t>(m.ids.size()), true, &end,
                          nullptr, &error));
  CheckRel("[3,5) final hidden", end.hidden, want.hidden);
  CheckRel("[3,5) logits", end.logits, want.logits);
}

// (e).
void CheckRefusals(const Model& m) {
  std::string error;
  glm::stand_in::HostOps ops(m.config);
  glm::LayerOps no_mla = ops.Bind();
  no_mla.mla = nullptr;
  const bool refused_mla =
      glm::Executor::Create(m.weights, Options(16), no_mla, &error) == nullptr;
  Check("MLA range without an MLA operator refused (" + error + ")",
        refused_mla);
  glm::ExecutorOptions late = Options(16);
  late.first_layer = 3;
  const bool refused_embed =
      glm::Executor::Create(m.weights, late, ops.Bind(), &error) == nullptr;
  Check("embedding with a range past layer 0 refused (" + error + ")",
        refused_embed);
  glm::ExecutorOptions kda_only = Options(16);
  kda_only.last_layer = 3;
  Check("KDA-only range needs no MLA operator",
        glm::Executor::Create(m.weights, kda_only, no_mla, &error) != nullptr);

  // An FFN failure at layer 4 marks the executor failed until Reset().
  glm::LayerOps failing = ops.Bind();
  const auto ffn = failing.ffn;
  bool fail = true;
  failing.ffn = [&](const glm::OpContext& ctx, const glm::Config& c,
                    std::uint32_t il, const glm::FfnLayerDevice& w,
                    const float* x, std::uint32_t n, float* y) {
    return !(fail && il == 4) && ffn(ctx, c, il, w, x, n, y);
  };
  auto exec = glm::Executor::Create(m.weights, Options(16), failing, &error);
  glm::ForwardResult out;
  const bool first = exec && exec->Forward(m.ids, true, &out, nullptr, &error);
  Check("failing operator reported (" + error + ")",
        exec && !first && exec->Failed());
  fail = false;
  Check("call after a failure refused",
        exec && !exec->Forward(m.ids, true, &out, nullptr, &error));
  const ref::GgufWeightSource source(m.weights);
  ref::ReferenceModel model(m.config, source);
  const auto want = model.Forward(m.ids, true);
  error.clear();
  const bool again = exec && exec->Reset() &&
                     exec->Forward(m.ids, true, &out, nullptr, &error);
  Check("forward after Reset " + error, again);
  CheckRel("forward after failure + Reset: logits", out.logits, want.logits);
}

}  // namespace

int main() {
  try {
    int devices = 0;
    if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
      std::cerr << "no HIP device\n";
      return 77;
    }
    std::cout << std::scientific << std::setprecision(2);
    const Fixture fx = Load(GLM53_E2E_FIXTURE);
    Model m;
    BuildModel(m, fx);
    CheckIncremental(m);
    CheckPassesAndRanges(m);
    CheckRefusals(m);
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << "\n";
    return 1;
  }
  if (failures != 0) {
    std::cerr << failures << " check(s) failed\n";
    return 1;
  }
  std::cout << "glm53_flash executor_loop: PASS\n";
  return 0;
}
