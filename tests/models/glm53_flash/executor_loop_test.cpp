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
//   (f) ForwardLast (the session path): last-token logits only, in the final
//       pass of the call, with identical state afterwards; Forward still
//       returns hidden and logits for every token;
//   (e) refusals: an MLA range without an MLA operator, the embedding with a
//       range past layer 0, and a failing operator leaving the executor
//       Failed() until Reset().

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
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

// (f) Executor::ForwardLast vs the per-token Forward and the reference.
void CheckLastLogits(const Model& m) {
  const glm::Config& c = m.config;
  const std::size_t vocab = c.vocab_size;
  const std::size_t hidden = c.hidden_size;
  const ref::GgufWeightSource source(m.weights);
  ref::ReferenceModel model(c, source);
  const std::span<const std::int32_t> ids(m.ids);
  const auto prefill = model.Forward(ids.first(kPrefill), true);
  const auto step12 = model.Forward(ids.subspan(kPrefill, 1), true);
  const auto step13 = model.Forward(ids.subspan(kPrefill + 1, 1), true);
  const std::vector<float> want_last(prefill.logits.end() - vocab,
                                     prefill.logits.end());
  std::string error;

  // The per-token API is unchanged: a row of hidden and of logits per token.
  {
    glm::stand_in::HostOps ops(c);
    auto exec =
        glm::Executor::Create(m.weights, Options(5), ops.Bind(), &error);
    glm::ForwardResult got;
    const bool ok =
        exec && exec->Forward(ids.first(kPrefill), true, &got, nullptr, &error);
    Check("Forward(12) per-token API " + error, ok);
    if (!ok)
      return;
    Check("Forward still returns hidden and logits for every token",
          ok && got.hidden.size() == kPrefill * hidden &&
              got.logits.size() == kPrefill * vocab);
    CheckRel("Forward(12) logits vs reference", got.logits, prefill.logits);
    glm::ForwardResult none;
    Check("Forward(want_logits=false) still returns hidden only",
          exec->Forward(ids.subspan(kPrefill, 1), false, &none, nullptr,
                        &error) &&
              none.hidden.size() == hidden && none.logits.empty());
  }

  // ForwardLast over several pass sizes: the 12 prompt tokens take
  // ceil(12 / max_tokens) passes (one MLA call each), then single-token
  // calls on the same executor continue from identical state.
  for (const std::uint32_t max_tokens : {16U, 5U, 12U, 1U}) {
    const std::string label = "max_tokens " + std::to_string(max_tokens);
    glm::stand_in::HostOps ops(c);
    auto exec = glm::Executor::Create(m.weights, Options(max_tokens),
                                      ops.Bind(), &error);
    Check("create " + label + " " + error, exec != nullptr);
    if (!exec)
      return;
    glm::ForwardResult got;
    bool ok =
        exec->ForwardLast(ids.first(kPrefill), true, &got, nullptr, &error);
    Check(label + ": ForwardLast(12) " + error, ok);
    if (!ok)
      return;
    Check(label + ": one logits row, no hidden", got.logits.size() == vocab &&
                                                     got.hidden.empty() &&
                                                     got.streams.empty());
    CheckRel(label + ": last-token logits vs reference", got.logits, want_last);
    const std::uint32_t passes = (kPrefill + max_tokens - 1) / max_tokens;
    Check(label + ": " + std::to_string(passes) + " passes for 12 tokens (" +
              std::to_string(ops.mla_calls()) + " MLA calls)",
          ops.mla_calls() == passes && exec->Position() == kPrefill);

    glm::ForwardResult next;
    ok = exec->Forward(ids.subspan(kPrefill, 1), true, &next, nullptr, &error);
    Check(label + ": decode after ForwardLast " + error, ok);
    CheckRel(label + ": decode logits (state carried over)", next.logits,
             step12.logits);
    ok = exec->ForwardLast(ids.subspan(kPrefill + 1, 1), true, &next, nullptr,
                           &error);
    Check(label + ": ForwardLast(1) " + error, ok);
    CheckRel(label + ": ForwardLast(1) logits", next.logits, step13.logits);

    // No logits wanted: nothing is downloaded, state still advances.
    Check(label + ": reset", exec->Reset());
    glm::ForwardResult quiet;
    ok = exec->ForwardLast(ids.first(kPrefill), false, &quiet, nullptr, &error);
    Check(label + ": ForwardLast(want_logits=false) " + error,
          ok && quiet.logits.empty() && quiet.hidden.empty() &&
              exec->Position() == kPrefill);
    ok = exec->Forward(ids.subspan(kPrefill, 1), true, &next, nullptr, &error);
    CheckRel(label + ": decode after a logits-free ForwardLast", next.logits,
             step12.logits);

    // want_hidden (the MTP session's prefill): every row's hidden, still one
    // logits row and no extra pass.
    Check(label + ": reset", exec->Reset());
    const std::uint32_t calls = ops.mla_calls();
    glm::ForwardResult rows;
    ok = exec->ForwardLast(ids.first(kPrefill), true, &rows, nullptr, &error,
                           /*want_hidden=*/true);
    Check(label + ": ForwardLast(want_hidden) " + error,
          ok && rows.logits.size() == vocab &&
              rows.hidden.size() == kPrefill * hidden &&
              ops.mla_calls() - calls == passes);
    CheckRel(label + ": want_hidden rows vs reference", rows.hidden,
             prefill.hidden);
    CheckRel(label + ": want_hidden last logits", rows.logits, want_last);
  }
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

// (h) A pass longer than the 256-row logits buffer (Executor::HeadLogits
// projects the head in row blocks): per-token logits of one 300-token pass
// match 100-token passes. Both stay on the >= 64-token Q8_1 prefill GEMM
// route (PrefillQuantGemm); passes under 64 tokens take the exact F32 route
// and differ by the activation quantization (~1e-2 rel), not by blocking.
void CheckLongPassLogits(const Model& m) {
  glm::stand_in::HostOps long_ops(m.config);
  glm::stand_in::HostOps short_ops(m.config);
  std::string error;
  glm::ExecutorOptions lo = Options(300);
  lo.capacity = 512;
  glm::ExecutorOptions so = Options(100);
  so.capacity = 512;
  auto one = glm::Executor::Create(m.weights, lo, long_ops.Bind(), &error);
  auto many = glm::Executor::Create(m.weights, so, short_ops.Bind(), &error);
  Check("long-pass executors " + error, one && many);
  if (!one || !many)
    return;
  std::vector<std::int32_t> ids(300);
  for (std::size_t i = 0; i < ids.size(); ++i) {
    ids[i] = m.ids[i % m.ids.size()];
  }
  glm::ForwardResult a, b;
  Check("long pass " + error, one->Forward(ids, true, &a, nullptr, &error));
  Check("short passes " + error, many->Forward(ids, true, &b, nullptr, &error));
  Check("long pass logits rows",
        a.logits.size() == ids.size() * m.config.vocab_size);
  CheckRel("300-token pass vs 100-token passes: logits", a.logits, b.logits);
}

// (g) GUFO_GLM53_DECODE_GRAPH: replayed position-independent half-layer
// runs (here the KDA mixer halves; the stand-in FFN declares no capture
// safety and the stand-in MLA syncs, so both stay eager) give bitwise the
// eager executor's logits, over decode steps, an MTP-style verify of two
// tokens with Rollback, and a second cycle after Reset (graphs survive
// Reset; replay must not double-advance or reuse stale state). Run with
// GUFO_DISPATCH_TELEMETRY=1 to see the graph "hit" events.
void CheckDecodeGraph(const Model& m) {
  glm::stand_in::HostOps graph_ops(m.config);
  glm::stand_in::HostOps eager_ops(m.config);
  std::string error;
  glm::ExecutorOptions o = Options(16);
  o.max_verify_tokens = 2;
  ::setenv("GUFO_GLM53_DECODE_GRAPH", "1", 1);
  auto graph = glm::Executor::Create(m.weights, o, graph_ops.Bind(), &error);
  ::unsetenv("GUFO_GLM53_DECODE_GRAPH");
  auto eager = glm::Executor::Create(m.weights, o, eager_ops.Bind(), &error);
  Check("graph comparison executors " + error, graph && eager);
  if (!graph || !eager)
    return;
  const std::span<const std::int32_t> ids(m.ids);
  const auto token = [&](std::size_t i) {
    return std::vector<std::int32_t>{ids[(kPrefill + i) % ids.size()]};
  };
  const auto same = [](const std::vector<float>& a,
                       const std::vector<float>& b) {
    return !a.empty() && a.size() == b.size() &&
           std::all_of(a.begin(), a.end(),
                       [](float v) { return std::isfinite(v); }) &&
           std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
  };
  std::vector<float> first_cycle;
  for (int cycle = 0; cycle < 2; ++cycle) {
    const std::string tag = " cycle " + std::to_string(cycle);
    if (cycle != 0) {
      Check("graph reset", graph->Reset());
      Check("eager reset", eager->Reset());
    }
    glm::ForwardResult got, want;
    Check(
        "graph/eager prefill" + tag,
        graph->ForwardLast(ids.first(kPrefill), true, &got, nullptr, &error) &&
            eager->ForwardLast(ids.first(kPrefill), true, &want, nullptr,
                               &error));
    Check("prefill logits bitwise" + tag, same(got.logits, want.logits));
    std::vector<float> cycle_logits;
    // Six single-token steps: the first warms each run, the second records
    // it, the rest replay.
    for (std::size_t t = 0; t < 6; ++t) {
      const auto one = token(t);
      const bool ok = graph->ForwardLast(one, true, &got, nullptr, &error) &&
                      eager->ForwardLast(one, true, &want, nullptr, &error);
      Check("graph/eager decode " + std::to_string(t) + tag + " " + error, ok);
      Check("decode logits bitwise " + std::to_string(t) + tag,
            ok && same(got.logits, want.logits));
      cycle_logits.insert(cycle_logits.end(), got.logits.begin(),
                          got.logits.end());
    }
    // Three verify rounds of two tokens (recording KDA checkpoints: its own
    // graph key), keeping one, then a plain step after the rollback.
    for (std::size_t r = 0; r < 3; ++r) {
      std::vector<std::int32_t> two = token(6 + (2 * r));
      two.push_back(token(7 + (2 * r))[0]);
      const bool ok = graph->ForwardVerify(two, true, &got, &error) &&
                      eager->ForwardVerify(two, true, &want, &error);
      Check("graph/eager verify " + std::to_string(r) + tag + " " + error, ok);
      Check("verify logits bitwise " + std::to_string(r) + tag,
            ok && same(got.logits, want.logits));
      Check("rollback " + std::to_string(r) + tag,
            graph->Rollback(1) && eager->Rollback(1));
      const auto one = token(20 + r);
      const bool step = graph->ForwardLast(one, true, &got, nullptr, &error) &&
                        eager->ForwardLast(one, true, &want, nullptr, &error);
      Check("after rollback " + std::to_string(r) + tag + " " + error,
            step && same(got.logits, want.logits));
      cycle_logits.insert(cycle_logits.end(), got.logits.begin(),
                          got.logits.end());
    }
    if (cycle == 0) {
      first_cycle = cycle_logits;
    } else {
      Check("graph reset reproduces the first cycle",
            same(cycle_logits, first_cycle));
    }
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
    std::cout << std::scientific << std::setprecision(2);
    const Fixture fx = Load(GLM53_E2E_FIXTURE);
    Model m;
    BuildModel(m, fx);
    CheckIncremental(m);
    CheckPassesAndRanges(m);
    CheckLastLogits(m);
    CheckRefusals(m);
    CheckDecodeGraph(m);
    CheckLongPassLogits(m);
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
