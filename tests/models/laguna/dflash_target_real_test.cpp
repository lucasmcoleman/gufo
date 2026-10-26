// The DFlash target side (phase 6d) on the real Laguna-S-2.1 target (68.6 GiB
// on the device) and the real drafter: the six tap layers captured on the
// device by the executor and fed to the GPU drafter, against the CPU oracle's
// taps of the same tokens.
//
//  * fixture: the 9-token capital prompt of
//    /server/staging/laguna/oracle-20261025-capital-cpuq8 (tokens confirmed by
//    the recorded inp_embd). Captured taps (prefill, batched path; and decode
//    rows, vector path) against the oracle's l_out-1, 10, 19, 29, 38, 47;
//    taps one layer early (oracle l_out-0, 9, ... and an executor tapping
//    layers 0, 9, ...) must be far. Drafts from the GPU-captured taps against
//    drafts from the oracle's taps through the same GPU drafter: every
//    disagreement must be a near tie.
//  * long: 700 real tokens (BOS + the first 699 of the 6b-0 code text;
//    GUFO_LAGUNA_6D_TEXT overrides), prefilled 512 + 188, the last 511 rows
//    injected. Device-injected drafts equal drafts from the same taps
//    uploaded from the host bit for bit; with the CPU oracle's taps of the
//    same tokens (GUFO_LAGUNA_6D_ORACLE, default
//    /server/staging/laguna/dflash-6d/oracle-long700, written by laguna_oracle
//    --act-quant gpu --kv-f16 --taps l_out; skipped with a notice if absent)
//    the taps and the drafts are compared as for the fixture. Then four
//    speculative steps (propose 16, verify on the vector path, greedy
//    acceptance, commit as a rewind): the first verify equals the same 16
//    tokens decoded alone bit for bit at depth 700.
//
// Skips (77, nothing verified) without the target, the drafter or the
// fixture. Loads the full target: run it alone under the GPU admission lock.
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "src/core/gguf_reader.hpp"
#include "src/models/laguna/kernels/device_model.hpp"
#include "src/models/laguna/kernels/dflash_session.hpp"
#include "src/models/laguna/tokenizer.hpp"
#include "tests/models/laguna/dflash_gpu.hpp"

using namespace gufo::testing::laguna;
namespace core = gufo::core;
namespace fs = std::filesystem;

namespace {

std::string Env(const char* name, const char* fallback) {
  const char* v = std::getenv(name);
  return v != nullptr && *v != '\0' ? v : fallback;
}

std::vector<float> ReadF32(const fs::path& path, std::size_t count) {
  std::ifstream in(path, std::ios::binary);
  Require(static_cast<bool>(in), "cannot read " + path.string());
  std::vector<float> v(count);
  in.read(reinterpret_cast<char*>(v.data()),
          static_cast<std::streamsize>(count * sizeof(float)));
  Require(in.gcount() == static_cast<std::streamsize>(count * sizeof(float)),
          path.string() + " is shorter than expected");
  return v;
}

double Seconds(std::chrono::steady_clock::time_point since) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - since)
      .count();
}

/// Oracle features [rows][taps][H] from l_out-<layer> files of `n` rows,
/// rows [from, n).
std::vector<float> OracleFeatures(const fs::path& dir,
                                  const std::vector<std::uint32_t>& layers,
                                  std::size_t n, std::size_t from,
                                  std::size_t H) {
  const std::size_t taps = layers.size();
  std::vector<float> f((n - from) * taps * H);
  for (std::size_t k = 0; k < taps; ++k) {
    const auto all =
        ReadF32(dir / ("l_out-" + std::to_string(layers[k]) + ".f32"), n * H);
    for (std::size_t t = from; t < n; ++t) {
      std::copy_n(all.data() + t * H, H,
                  f.data() + ((t - from) * taps + k) * H);
    }
  }
  return f;
}

/// Relative rms of each tap column over all rows.
std::vector<double> PerTapRel(const std::vector<float>& got,
                              const std::vector<float>& want, std::size_t taps,
                              std::size_t H) {
  const std::size_t rows = got.size() / (taps * H);
  std::vector<double> out;
  for (std::size_t k = 0; k < taps; ++k) {
    std::vector<float> a, b;
    for (std::size_t t = 0; t < rows; ++t) {
      const std::size_t o = (t * taps + k) * H;
      a.insert(a.end(), got.begin() + o, got.begin() + o + H);
      b.insert(b.end(), want.begin() + o, want.begin() + o + H);
    }
    out.push_back(RelRms(a, b));
  }
  return out;
}

/// Median over rows of each tap's per-row relative rms (rows are normalized
/// one by one before the drafter reads them, so a few huge rows must not
/// dominate the measure).
std::vector<double> PerTapRowMedian(const std::vector<float>& got,
                                    const std::vector<float>& want,
                                    std::size_t taps, std::size_t H) {
  const std::size_t rows = got.size() / (taps * H);
  std::vector<double> out;
  for (std::size_t k = 0; k < taps; ++k) {
    std::vector<double> r;
    for (std::size_t t = 0; t < rows; ++t) {
      const std::size_t o = (t * taps + k) * H;
      double num = 0;
      double den = 0;
      for (std::size_t i = 0; i < H; ++i) {
        const double d = static_cast<double>(got[o + i]) - want[o + i];
        num += d * d;
        den += static_cast<double>(want[o + i]) * want[o + i];
      }
      r.push_back(std::sqrt(num / std::max(den, 1e-30)));
    }
    std::nth_element(r.begin(), r.begin() + r.size() / 2, r.end());
    out.push_back(r[r.size() / 2]);
  }
  return out;
}

std::string Join(const std::vector<double>& v) {
  std::ostringstream m;
  m << std::scientific << std::setprecision(2);
  for (std::size_t i = 0; i < v.size(); ++i) {
    m << (i ? " " : "") << v[i];
  }
  return m.str();
}

/// Correct taps must be close to the oracle and taps one layer early far:
/// every early column at least 3x its correct one. Returns the worst ratio.
double RequireTapsMatch(const std::string& what,
                        const std::vector<double>& right,
                        const std::vector<double>& early) {
  double worst = 1e30;
  for (std::size_t k = 0; k < right.size(); ++k) {
    Require(right[k] < 0.1, what + ": tap " + std::to_string(k) + " is " +
                                std::to_string(right[k]) + " from the oracle");
    Require(early[k] > 3.0 * right[k],
            what + ": tap " + std::to_string(k) +
                " one layer early is not told apart");
    worst = std::min(worst, early[k] / right[k]);
  }
  return worst;
}

struct DraftCompare {
  std::size_t agree{0};
  std::vector<std::string> notes;
};

/// GPU-tap drafts against oracle-tap drafts (both from the GPU drafter): a
/// disagreement must be a near tie, i.e. the oracle-tap logits' own gap
/// between their best and the GPU-tap draft is within twice the largest
/// logit difference between the two runs.
DraftCompare CompareTapDrafts(const std::vector<std::int32_t>& gd,
                              const std::vector<float>& gl,
                              const std::vector<std::int32_t>& od,
                              const std::vector<float>& ol, std::size_t V) {
  DraftCompare c;
  const double tie = 2.0 * MaxAbsDiff(gl, ol);
  for (std::size_t r = 0; r < gd.size(); ++r) {
    if (gd[r] == od[r]) {
      ++c.agree;
      continue;
    }
    const float* o = ol.data() + r * V;
    const double gap = static_cast<double>(o[od[r]]) - o[gd[r]];
    std::ostringstream m;
    m << "row " << r << ": gpu-tap " << gd[r] << " oracle-tap " << od[r]
      << " gap " << gap << " (tie bound " << tie << ")";
    c.notes.push_back(m.str());
    Require(gap <= tie, "draft disagreement beyond a near tie: " + m.str());
  }
  return c;
}

void Print(const char* what, const std::vector<std::int32_t>& d) {
  std::cout << "    " << what << ":";
  for (const auto t : d) {
    std::cout << " " << t;
  }
  std::cout << "\n";
}

std::int32_t Argmax(const float* row, std::size_t V) {
  return static_cast<std::int32_t>(
      ref::RowArgmax(std::span<const float>(row, V), V)[0]);
}

/// A host-fed reference drafter on its own stream.
class HostDrafter {
public:
  HostDrafter(const gpu::DrafterWeights& w, const gpu::DeviceModel& dev,
              std::int32_t mask, std::uint32_t floor) {
    CheckHip(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking),
             "stream");
    std::string error;
    d_ = gpu::DrafterSession::Create(w, dev.output(), dev.weights().token_embd,
                                     mask, 32, stream_, &error);
    Require(d_ != nullptr, "host drafter: " + error);
    d_->kv().Reset(floor);
  }
  ~HostDrafter() {
    d_.reset();
    qfn_mmq_release_stream(stream_);
    (void)hipStreamDestroy(stream_);
  }
  HostDrafter(const HostDrafter&) = delete;
  HostDrafter& operator=(const HostDrafter&) = delete;
  void Inject(const std::vector<float>& f, std::uint32_t start,
              std::size_t width) {
    const auto rows = static_cast<std::uint32_t>(f.size() / width);
    std::string error;
    for (std::uint32_t p = 0; p < rows; p += 32) {
      const std::uint32_t n = std::min(32U, rows - p);
      Require(
          d_->Inject(std::span<const float>(f).subspan(p * width, n * width),
                     start + p, gpu::Path::kVector, &error),
          "host inject: " + error);
    }
  }
  std::vector<std::int32_t> Propose(std::int32_t anchor,
                                    std::vector<float>* logits) {
    std::vector<std::int32_t> d;
    std::string error;
    Require(d_->Propose(anchor, 16, &d, logits, gpu::Path::kBatched, &error),
            "host propose: " + error);
    return d;
  }

private:
  hipStream_t stream_{};
  std::unique_ptr<gpu::DrafterSession> d_;
};

std::vector<float> Captured(gpu::DFlashSession& s, std::size_t rows) {
  CheckHip(hipStreamSynchronize(s.target().stream()), "sync");
  return ToHost(s.captured_features(), rows * s.target().feature_width());
}

}  // namespace

int main() {
  try {
    const fs::path drafter_path =
        Env("GUFO_LAGUNA_DFLASH_GGUF",
            "/server/ai/models/Myric/Laguna-S-2.1-APEX-GGUF/"
            "laguna-s-2.1-DFlash-Q4_K.gguf");
    const fs::path fixture =
        Env("GUFO_LAGUNA_DFLASH_TAPS",
            "/server/staging/laguna/oracle-20261025-capital-cpuq8");
    const fs::path text_path =
        Env("GUFO_LAGUNA_6D_TEXT",
            "/server/staging/laguna/dflash-6b0/verify-context-code.txt");
    const fs::path oracle_long =
        Env("GUFO_LAGUNA_6D_ORACLE",
            "/server/staging/laguna/dflash-6d/oracle-long700");
    std::error_code ec;
    if (!fs::exists(ModelPath(), ec) || !fs::exists(drafter_path, ec) ||
        !fs::exists(fixture / "l_out-1.f32", ec) ||
        !fs::exists(text_path, ec)) {
      std::cout << "SKIP: laguna.dflash_target_real needs the target and "
                   "drafter GGUFs, the tap fixture and the long text; nothing "
                   "was verified\n";
      return kSkipExitCode;
    }
    std::string error;
    const auto treader = core::GgufReader::OpenFile(ModelPath(), &error);
    Require(treader != nullptr, "target: " + error);
    const auto dreader = core::GgufReader::OpenFile(drafter_path, &error);
    Require(dreader != nullptr, "drafter: " + error);
    const auto target = ml::ModelWeights::Bind(*treader, &error);
    Require(target.has_value(), "target bind: " + error);
    const auto drafter = ml::DFlashWeights::Bind(*dreader, &error);
    Require(drafter.has_value(), "drafter bind: " + error);
    const ml::DFlashConfig& c = drafter->config;
    Require(c.ValidateAgainstTarget(target->config, &error),
            "pairing: " + error);
    const auto tok = ml::Tokenizer::FromGguf(*treader, &error);
    Require(tok != nullptr, "tokenizer: " + error);
    const std::size_t H = c.hidden_size;
    const std::size_t taps = c.tap_count();
    const std::size_t width = taps * H;
    const std::size_t V = target->output.rows;
    std::vector<std::uint32_t> layers, early;
    for (std::size_t k = 0; k < taps; ++k) {
      layers.push_back(c.tap_layer(k));
      early.push_back(c.tap_layer(k) - 1);
    }
    const auto mask = static_cast<std::int32_t>(c.mask_token_id);

    // Tokens. The fixture's are confirmed by its recorded embedding rows.
    std::vector<std::int32_t> fix{2};
    for (const auto id :
         tok->Encode("The capital of France is Paris, and", {false, false})) {
      fix.push_back(static_cast<std::int32_t>(id));
    }
    Require(fix.size() == 9, "fixture prompt is not 9 tokens");
    Require(BitIdentical(Embeddings(*target, fix),
                         ReadF32(fixture / "inp_embd.f32", 9 * H)),
            "fixture tokens do not reproduce the recorded embeddings");
    std::vector<std::int32_t> longt{2};
    {
      std::ifstream in(text_path, std::ios::binary);
      std::stringstream text;
      text << in.rdbuf();
      for (const auto id : tok->Encode(text.str(), {false, false})) {
        longt.push_back(static_cast<std::int32_t>(id));
      }
      Require(longt.size() >= 700, "long text too short");
      longt.resize(700);
    }
    const bool have_oracle = fs::exists(oracle_long / "l_out-47.f32", ec);
    if (have_oracle) {
      // The oracle ran on llama-tokenize's ids; they must be ours.
      std::ifstream in(oracle_long.parent_path() / "long700-tokens.txt");
      std::string line;
      std::getline(in, line);
      std::ostringstream ours;
      for (std::size_t i = 0; i < longt.size(); ++i) {
        ours << (i ? "," : "") << longt[i];
      }
      Require(line == ours.str(),
              "the long oracle's tokens differ from gufo's tokenization");
    }

    // Device.
    const auto t0 = std::chrono::steady_clock::now();
    auto dev = gpu::DeviceModel::Upload(*target, treader.get(), &error);
    Require(dev != nullptr, "target upload: " + error);
    DrafterCtx dctx(*drafter);
    std::cout << "upload (target + drafter): " << Seconds(t0) << " s\n"
              << std::flush;
    const auto make = [&](std::uint32_t context) {
      gpu::ExecutorOptions o;
      o.context = context;
      o.max_chunk = 512;
      o.path = gpu::Path::kAuto;
      auto ex = gpu::Executor::Create(*dev, o, &error);
      Require(ex != nullptr, "executor: " + error);
      return ex;
    };
    auto ex = make(1024);
    auto session =
        gpu::DFlashSession::Create(*ex, dctx.weights(), layers, mask, &error);
    Require(session != nullptr, "session: " + error);

    // ---- fixture -------------------------------------------------------
    std::cout << "fixture (9 tokens, anchor = oracle top-1)\n";
    const auto oracle_fix = OracleFeatures(fixture, layers, 9, 0, H);
    const auto oracle_fix_early = OracleFeatures(fixture, early, 9, 0, H);
    const auto oracle_last = ReadF32(fixture / "logits.f32", V);
    const std::int32_t anchor = Argmax(oracle_last.data(), V);
    std::vector<float> last(V);
    Require(session->Prefill(fix, gpu::LogitsRows::kLast, last.data(), &error),
            "fixture prefill: " + error);
    std::cout << "  gpu top-1 " << Argmax(last.data(), V) << ", oracle "
              << anchor << "\n";
    const auto gf = Captured(*session, 9);
    const auto right = PerTapRel(gf, oracle_fix, taps, H);
    const auto off = PerTapRel(gf, oracle_fix_early, taps, H);
    const double ratio = RequireTapsMatch("fixture prefill", right, off);
    std::cout << "  prefill taps vs oracle l_out-1..47: " << Join(right)
              << "\n  vs l_out one layer early:         " << Join(off)
              << " (worst ratio " << ratio << ")\n";
    {
      // An executor tapping one layer early, against the correct oracle taps.
      auto ex_early = make(1024);
      Require(ex_early->SetFeatureTaps(early, &error), error);
      float* buf = nullptr;
      CheckHip(hipMalloc(&buf, 9 * width * sizeof(float)), "alloc");
      gpu::ForwardArgs args;
      args.features = buf;
      Require(
          ex_early->Forward(fix, gpu::LogitsRows::kNone, nullptr, args, &error),
          "early forward: " + error);
      const auto ef = ToHost(buf, 9 * width);
      (void)hipFree(buf);
      const auto wrong = PerTapRel(ef, oracle_fix, taps, H);
      RequireTapsMatch("executor tapping one layer early", right, wrong);
      std::cout << "  executor tapped one layer early vs oracle: "
                << Join(wrong) << " (must fail: it does)\n";
    }
    std::vector<float> gl, ol;
    std::vector<std::int32_t> gd;
    Require(session->Propose(anchor, 16, &gd, &gl, &error),
            "fixture propose: " + error);
    {
      HostDrafter h(dctx.weights(), *dev, mask, 0);
      h.Inject(oracle_fix, 0, width);
      const auto od = h.Propose(anchor, &ol);
      Print("drafts, gpu taps   ", gd);
      Print("drafts, oracle taps", od);
      const auto cmp = CompareTapDrafts(gd, gl, od, ol, V);
      std::cout << "  top-1 agreement " << cmp.agree << "/15, logits "
                << RelRms(gl, ol) << " rel rms";
      for (const auto& n : cmp.notes) {
        std::cout << "; " << n;
      }
      std::cout << "\n";
    }
    // Decode rows (vector path), one token at a time.
    {
      Require(session->Reset(&error), "reset: " + error);
      std::vector<float> dfeat;
      std::vector<float> rowl(V);
      for (std::size_t t = 0; t < 9; ++t) {
        Require(
            session->Verify(std::span(fix).subspan(t, 1), rowl.data(), &error),
            "decode: " + error);
        const auto row = Captured(*session, 1);
        dfeat.insert(dfeat.end(), row.begin(), row.end());
      }
      const auto dright = PerTapRel(dfeat, oracle_fix, taps, H);
      const auto doff = PerTapRel(dfeat, oracle_fix_early, taps, H);
      RequireTapsMatch("fixture decode", dright, doff);
      std::vector<float> dl;
      std::vector<std::int32_t> dd;
      Require(session->Propose(anchor, 16, &dd, &dl, &error),
              "decode propose: " + error);
      std::size_t same = 0;
      for (std::size_t r = 0; r < 15; ++r) {
        same += dd[r] == gd[r] ? 1 : 0;
      }
      std::cout << "  decode-row taps vs oracle: " << Join(dright)
                << "\n  decode-row taps vs prefill taps: " << RelRms(dfeat, gf)
                << " rel rms; drafts equal to the "
                << "prefill-tap drafts at " << same << "/15\n";
    }

    // ---- long context --------------------------------------------------
    std::cout << "long context (700 tokens, prefill 512 + 188, last 511 rows "
                 "injected)\n";
    Require(session->Reset(&error), "reset: " + error);
    const auto t1 = std::chrono::steady_clock::now();
    Require(
        session->Prefill(longt, gpu::LogitsRows::kLast, last.data(), &error),
        "long prefill: " + error);
    std::cout << "  prefill " << Seconds(t1) << " s, floor "
              << session->drafter().kv().floor() << ", committed "
              << session->drafter().kv().committed() << "\n";
    Require(session->drafter().kv().floor() == 189 &&
                session->drafter().kv().committed() == 700,
            "the session did not inject exactly rows 189..699");
    const std::int32_t lanchor = Argmax(last.data(), V);
    // All 700 rows' taps on a second executor (same chunks, same path).
    auto twin = make(1024);
    Require(twin->SetFeatureTaps(layers, &error), error);
    std::vector<float> all;
    {
      float* buf = nullptr;
      CheckHip(hipMalloc(&buf, 512 * width * sizeof(float)), "alloc");
      std::vector<float> tl(V);
      for (std::uint32_t p = 0; p < 700; p += 512) {
        const std::uint32_t n = std::min(512U, 700 - p);
        gpu::ForwardArgs args;
        args.features = buf;
        Require(twin->Forward(std::span(longt).subspan(p, n),
                              gpu::LogitsRows::kLast, tl.data(), args, &error),
                "twin: " + error);
        const auto f = ToHost(buf, n * width);
        all.insert(all.end(), f.begin(), f.end());
      }
      (void)hipFree(buf);
      Require(BitIdentical(tl, last), "twin logits differ from the session's");
      Require(BitIdentical(
                  Captured(*session, 188),
                  std::vector<float>(all.begin() + 512 * width, all.end())),
              "the session's capture differs from the twin's");
    }
    const std::vector<float> gpu_rows(all.begin() + 189 * width, all.end());
    if (const char* dump = std::getenv("GUFO_LAGUNA_6D_DUMP_TAPS")) {
      std::ofstream out(dump, std::ios::binary);
      out.write(reinterpret_cast<const char*>(gpu_rows.data()),
                static_cast<std::streamsize>(gpu_rows.size() * sizeof(float)));
      std::cout << "  wrote rows 189..699 of the GPU taps to " << dump << "\n";
    }
    std::vector<float> sl;
    std::vector<std::int32_t> sd;
    Require(session->Propose(lanchor, 16, &sd, &sl, &error),
            "long propose: " + error);
    {
      HostDrafter h(dctx.weights(), *dev, mask, 189);
      h.Inject(gpu_rows, 189, width);
      std::vector<float> hl;
      const auto hd = h.Propose(lanchor, &hl);
      Require(hd == sd && BitIdentical(hl, sl),
              "device-injected drafts differ from the same taps uploaded");
      std::cout << "  anchor " << lanchor
                << "; device-injected drafts equal host-uploaded drafts bit "
                   "for bit\n";
    }
    if (have_oracle) {
      // Decode-path taps of the same 700 tokens (vector path: the rounding
      // of the oracle's --act-quant gpu mode), on a third executor.
      std::vector<float> vall;
      {
        auto vex = make(1024);
        Require(vex->SetFeatureTaps(layers, &error), error);
        float* buf = nullptr;
        CheckHip(hipMalloc(&buf, width * sizeof(float)), "alloc");
        gpu::ForwardArgs args;
        args.features = buf;
        for (std::size_t t = 0; t < longt.size(); ++t) {
          Require(vex->Forward(std::span(longt).subspan(t, 1),
                               gpu::LogitsRows::kNone, nullptr, args, &error),
                  "decode-path taps: " + error);
          const auto f = ToHost(buf, width);
          vall.insert(vall.end(), f.begin(), f.end());
        }
        (void)hipFree(buf);
      }
      const auto ofull = OracleFeatures(oracle_long, layers, 700, 189, H);
      const auto oearly = OracleFeatures(oracle_long, early, 700, 189, H);
      const std::vector<float> vrows(vall.begin() + 189 * width, vall.end());
      const auto vo = PerTapRowMedian(vrows, ofull, taps, H);
      const auto bo = PerTapRowMedian(gpu_rows, ofull, taps, H);
      const auto bv = PerTapRowMedian(gpu_rows, vrows, taps, H);
      const auto ve = PerTapRowMedian(vrows, oearly, taps, H);
      std::cout << "  per-row rel rms, median over rows 189..699, taps "
                   "1,10,19,29,38,47:\n"
                << "    decode-path GPU vs oracle (same rounding): " << Join(vo)
                << "\n    prefill GPU vs oracle:                    "
                << Join(bo)
                << "\n    prefill GPU vs decode-path GPU:           "
                << Join(bv)
                << "\n    decode-path GPU vs oracle one layer early: "
                << Join(ve) << "\n";
      // At 700 rows two roundings of the same GPU model (prefill on the
      // batched path, decode on the vector path) already differ by 1e-3 at
      // tap 1 up to ~1e-1 at tap 47 (repeated-token runs flip routing). The
      // oracle must sit no farther from the GPU than that, and a tap one
      // layer early must be far beyond it.
      for (std::size_t k = 0; k < taps; ++k) {
        Require(vo[k] <= 1.25 * bv[k] && bo[k] <= 1.25 * bv[k],
                "long: tap " + std::to_string(k) +
                    " is farther from the oracle than the GPU's own two "
                    "roundings are from each other");
        Require(ve[k] > 3.0 * vo[k], "long: tap " + std::to_string(k) +
                                         " one layer early is not told apart");
      }
      const auto olast = ReadF32(oracle_long / "logits.f32", V);
      std::cout << "  gpu top-1 " << lanchor << ", oracle "
                << Argmax(olast.data(), V) << "\n";
      std::vector<float> ol2, vl2;
      std::vector<std::int32_t> od, vd;
      {
        HostDrafter h(dctx.weights(), *dev, mask, 189);
        h.Inject(ofull, 189, width);
        od = h.Propose(lanchor, &ol2);
      }
      {
        HostDrafter h(dctx.weights(), *dev, mask, 189);
        h.Inject(vrows, 189, width);
        vd = h.Propose(lanchor, &vl2);
      }
      Print("drafts, prefill taps    ", sd);
      Print("drafts, decode-path taps", vd);
      Print("drafts, oracle taps     ", od);
      const auto agree = [](const std::vector<std::int32_t>& a,
                            const std::vector<std::int32_t>& b) {
        std::size_t n = 0;
        for (std::size_t r = 0; r < a.size(); ++r) {
          n += a[r] == b[r] ? 1 : 0;
        }
        return n;
      };
      std::cout << "  top-1 agreement: decode-path vs oracle " << agree(vd, od)
                << "/15 (logits " << RelRms(vl2, ol2) << "), prefill vs oracle "
                << agree(sd, od) << "/15 (logits " << RelRms(sl, ol2)
                << "), prefill vs decode-path " << agree(sd, vd)
                << "/15 (logits " << RelRms(sl, vl2) << ")\n";
    } else {
      std::cout << "  NOT RUN: long-context oracle comparison (no "
                << oracle_long.string() << ")\n";
    }

    // Speculative steps: verify on the vector path, greedy acceptance,
    // commit. The first verify equals solo decode at depth 700.
    std::int32_t a0 = lanchor;
    std::vector<std::int32_t> drafts = sd;
    std::size_t accepted_total = 0;
    for (int step = 0; step < 4; ++step) {
      std::vector<std::int32_t> block{a0};
      block.insert(block.end(), drafts.begin(), drafts.end());
      const std::uint32_t n = session->position();
      std::vector<float> vl(16 * V);
      const auto tv = std::chrono::steady_clock::now();
      Require(session->Verify(block, vl.data(), &error), "verify: " + error);
      const double verify_ms = Seconds(tv) * 1e3;
      if (step == 0) {
        std::vector<float> solo;
        std::vector<float> row(V);
        for (const auto t : block) {
          Require(twin->Forward(std::span<const std::int32_t>(&t, 1),
                                gpu::LogitsRows::kLast, row.data(), &error),
                  "solo: " + error);
          solo.insert(solo.end(), row.begin(), row.end());
        }
        Require(BitIdentical(vl, solo),
                "the 16-row verify differs from solo decode at depth 700");
        std::cout << "  verify of 16 rows at 700 equals solo decode bit for "
                     "bit\n";
      }
      std::size_t a = 0;
      while (a < 15 && Argmax(vl.data() + a * V, V) == drafts[a]) {
        ++a;
      }
      const std::int32_t next = Argmax(vl.data() + a * V, V);
      Require(session->Commit(n + static_cast<std::uint32_t>(a) + 1, &error),
              "commit: " + error);
      accepted_total += a;
      std::cout << "  step " << step << " at " << n << ": accepted " << a
                << "/15, verify " << std::fixed << std::setprecision(1)
                << verify_ms << std::defaultfloat << " ms\n";
      a0 = next;
      std::vector<float> pl;
      Require(session->Propose(a0, 16, &drafts, &pl, &error),
              "propose: " + error);
    }
    Require(session->drafter_active() && !ex->poisoned(),
            "the session lost its drafter or target");
    std::cout << "  greedy accepted " << accepted_total
              << " drafts in 4 steps\n";
    std::cout << "PASS: laguna.dflash_target_real\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << "\n";
    return 1;
  }
}
