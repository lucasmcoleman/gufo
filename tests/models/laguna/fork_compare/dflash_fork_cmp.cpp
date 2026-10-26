// CPU-only comparison of gufo's Laguna DFlash oracle against Poolside's
// llama.cpp fork (06f8cebd7), through its libllama. Not part of the gufo build.
//
// Input: a dump directory written by laguna.dflash_real_smoke when
// GUFO_LAGUNA_DFLASH_DUMP=DIR is set: features.f32 (n_ctx rows of taps*hidden
// floats, the oracle's taps), meta.txt ("n_ctx anchor block_size mask_id
// vocab"), logits.f32 (the oracle's draft logits, block_size-1 rows).
//
// It runs the fork's drafter exactly as common/speculative.cpp does
// (llama_encode of the features, K/V injection by llama_decode of the encoder
// output at positions 0..n_ctx-1, causal block decode at n_ctx) with the target
// loaded only as ctx_other (token embeddings and head; no target forward), all
// on the CPU, and prints top-1 agreement and the logit difference per row.
//
// Build: see tests/models/laguna/fork_compare/build.sh.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"
#include "llama-ext.h"
#include "llama.h"

static std::vector<float> ReadF32(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    std::fprintf(stderr, "cannot read %s\n", path.c_str());
    std::exit(2);
  }
  std::vector<float> v(static_cast<size_t>(f.tellg()) / sizeof(float));
  f.seekg(0);
  f.read(reinterpret_cast<char*>(v.data()),
         static_cast<std::streamsize>(v.size() * sizeof(float)));
  return v;
}

// Optional layer-by-layer diff: when argv[4] is a directory of oracle taps
// (GUFO_LAGUNA_DFLASH_TAPS), every graph tensor whose name has a <name>.f32
// file there is compared as it is computed.
struct TapDiff {
  std::string dir;
  std::vector<std::string> done;
};

static bool OnTensor(struct ggml_tensor* t, bool ask, void* user) {
  auto* d = static_cast<TapDiff*>(user);
  const std::string name = t->name;
  const std::string path = d->dir + "/" + name + ".f32";
  std::ifstream probe(path, std::ios::binary);
  if (!probe || t->type != GGML_TYPE_F32) {
    return ask ? false : true;
  }
  if (ask) {
    return true;
  }
  const std::vector<float> oracle = ReadF32(path);
  std::vector<float> got(ggml_nelements(t));
  ggml_backend_tensor_get(t, got.data(), 0, got.size() * sizeof(float));
  if (got.size() != oracle.size()) {
    std::printf("tap %-24s size mismatch fork %zu oracle %zu\n", name.c_str(),
                got.size(), oracle.size());
    return true;
  }
  double num = 0, den = 0, worst = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double e = double(got[i]) - oracle[i];
    num += e * e;
    den += double(oracle[i]) * oracle[i];
    worst = std::max(worst, std::abs(e));
  }
  std::printf("tap %-24s n=%7zu rel-rms %.3e max-abs %.3e\n", name.c_str(),
              got.size(), std::sqrt(num / std::max(den, 1e-30)), worst);
  if (std::getenv("TAP_ROWS") != nullptr && t->ne[1] > 1) {
    // Per token row, for locating a mask or position fault.
    const size_t w = static_cast<size_t>(t->ne[0]);
    for (size_t r = 0; r < static_cast<size_t>(t->ne[1]); ++r) {
      double rn = 0, rd = 0;
      for (size_t i = 0; i < w; ++i) {
        const double e = double(got[r * w + i]) - oracle[r * w + i];
        rn += e * e;
        rd += double(oracle[r * w + i]) * oracle[r * w + i];
      }
      std::printf("    row %2zu rel-rms %.3e\n", r,
                  std::sqrt(rn / std::max(rd, 1e-30)));
    }
  }
  return true;
}

int main(int argc, char** argv) {
  if (argc != 4 && argc != 5) {
    std::fprintf(
        stderr,
        "usage: %s DUMP_DIR TARGET.gguf DRAFTER.gguf [ORACLE_TAPS_DIR]\n",
        argv[0]);
    return 2;
  }
  TapDiff tap_diff;
  if (argc == 5) {
    tap_diff.dir = argv[4];
  }
  const std::string dir = argv[1];
  int n_ctx_rows = 0, anchor = 0, block = 0, mask = 0, vocab = 0;
  {
    std::ifstream m(dir + "/meta.txt");
    m >> n_ctx_rows >> anchor >> block >> mask >> vocab;
  }
  const std::vector<float> features = ReadF32(dir + "/features.f32");
  const std::vector<float> oracle = ReadF32(dir + "/logits.f32");

  llama_backend_init();
  llama_model_params mp = llama_model_default_params();
  mp.n_gpu_layers = 0;  // CPU only
  mp.use_mmap = true;
  llama_model* tgt = llama_model_load_from_file(argv[2], mp);
  llama_model* dft = llama_model_load_from_file(argv[3], mp);
  if (!tgt || !dft) {
    std::fprintf(stderr, "model load failed\n");
    return 1;
  }
  llama_context_params cp = llama_context_default_params();
  cp.n_ctx = 1024;
  cp.n_batch = 512;
  cp.n_ubatch = 512;
  cp.n_seq_max = 1;
  llama_context* ctx_tgt = llama_init_from_model(tgt, cp);
  if (!ctx_tgt) {
    std::fprintf(stderr, "target context failed\n");
    return 1;
  }
  llama_context_params dp = cp;
  dp.ctx_other = ctx_tgt;
  if (argc == 5) {
    dp.cb_eval = OnTensor;
    dp.cb_eval_user_data = &tap_diff;
  }
  llama_context* ctx = llama_init_from_model(dft, dp);
  if (!ctx) {
    std::fprintf(stderr, "drafter context failed\n");
    return 1;
  }
  llama_set_embeddings_nextn(ctx, true, /*masked=*/true);
  llama_set_causal_attn(ctx, true);  // decoder_arch laguna

  const int n_dec = llama_model_n_embd(dft);
  // Encoder: features -> fused rows.
  llama_batch enc = {};
  enc.n_tokens = n_ctx_rows;
  enc.embd = const_cast<float*>(features.data());
  if (llama_encode(ctx, enc) != 0) {
    std::fprintf(stderr, "llama_encode failed\n");
    return 1;
  }
  const float* fused = llama_get_embeddings_nextn(ctx);
  std::vector<float> fused_copy(
      fused, fused + static_cast<size_t>(n_ctx_rows) * n_dec);

  // Inject at positions 0..n_ctx-1.
  llama_batch inj = llama_batch_init(n_ctx_rows, n_dec, 1);
  inj.n_tokens = n_ctx_rows;
  std::copy(fused_copy.begin(), fused_copy.end(), inj.embd);
  for (int i = 0; i < n_ctx_rows; ++i) {
    inj.pos[i] = i;
    inj.n_seq_id[i] = 1;
    inj.seq_id[i][0] = 0;
    inj.logits[i] = false;
  }
  if (llama_decode(ctx, inj) != 0) {
    std::fprintf(stderr, "inject decode failed\n");
    return 1;
  }

  // Block at n_ctx: [anchor, MASK x (block-1)].
  llama_batch bb = llama_batch_init(block, 0, 1);
  bb.n_tokens = block;
  for (int i = 0; i < block; ++i) {
    bb.token[i] = i == 0 ? anchor : mask;
    bb.pos[i] = n_ctx_rows + i;
    bb.n_seq_id[i] = 1;
    bb.seq_id[i][0] = 0;
    bb.logits[i] = true;
  }
  if (llama_decode(ctx, bb) != 0) {
    std::fprintf(stderr, "block decode failed\n");
    return 1;
  }

  int agree = 0;
  double worst_rel = 0.0;
  for (int r = 1; r < block; ++r) {
    const float* fl = llama_get_logits_ith(ctx, r);
    const float* ol = oracle.data() + static_cast<size_t>(r - 1) * vocab;
    int fa = 0, oa = 0;
    double num = 0, den = 0;
    for (int v = 0; v < vocab; ++v) {
      if (fl[v] > fl[fa])
        fa = v;
      if (ol[v] > ol[oa])
        oa = v;
      num += (double(fl[v]) - ol[v]) * (double(fl[v]) - ol[v]);
      den += double(ol[v]) * ol[v];
    }
    const double rel = std::sqrt(num / std::max(den, 1e-30));
    worst_rel = std::max(worst_rel, rel);
    agree += fa == oa;
    std::printf("row %2d: fork top1 %6d oracle top1 %6d %s rel-rms %.3e\n", r,
                fa, oa, fa == oa ? "same" : "DIFF", rel);
  }
  std::printf("RESULT top-1 agreement %d/%d, worst row rel-rms %.3e\n", agree,
              block - 1, worst_rel);
  return 0;
}
