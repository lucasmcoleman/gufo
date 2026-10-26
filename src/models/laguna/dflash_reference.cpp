#include "src/models/laguna/dflash_reference.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

namespace gufo::models::laguna::reference {

DFlashModel::DFlashModel(const DFlashWeights& drafter,
                         const ModelWeights& target, Options options)
    : drafter_(drafter), target_(target), options_(options) {}

std::vector<float> DFlashModel::Forward(const DFlashInputs& in,
                                        const TapFn& tap) const {
  const DFlashConfig& c = drafter_.config;
  const std::size_t H = c.hidden_size;
  const std::size_t taps = c.tap_count();
  const std::size_t hd = c.head_dim;
  const std::size_t heads = c.num_heads;
  const std::size_t kvh = c.num_kv_heads;
  const std::size_t q_dim = c.q_dim();
  const std::size_t kv_dim = c.kv_dim();
  const std::size_t width = c.fused_input_width();
  const std::size_t N = in.context_positions.size();
  const std::size_t B = in.block_tokens.size();
  if (B < 2 || B > c.block_size) {
    throw std::invalid_argument("laguna dflash oracle: bad block length");
  }
  if (in.features.size() != N * width) {
    throw std::invalid_argument("laguna dflash oracle: feature size mismatch");
  }
  if (target_.config.hidden_size != H || target_.token_embd.cols != H ||
      target_.output.cols != H) {
    throw std::invalid_argument("laguna dflash oracle: target hidden mismatch");
  }
  for (const auto position : in.context_positions) {
    if (position < 0 || position >= in.block_position) {
      throw std::invalid_argument(
          "laguna dflash oracle: context position must precede the block");
    }
  }

  const auto emit_f32 = [&](const char* name, int layer,
                            std::array<std::uint64_t, 3> ne,
                            const std::vector<float>& data) {
    if (tap) {
      const std::string full =
          layer >= 0 ? std::string(name) + "-" + std::to_string(layer)
                     : std::string(name);
      tap(Tap{full, TapType::kF32, ne, data.data()});
    }
  };

  const RopeParams rope{.n_rot = c.rotary_dim, .freq_base = c.rope_theta};
  const float kq_scale = 1.0F / std::sqrt(static_cast<float>(hd));

  // ---- Encoder: per-tap RMSNorm, fc, RMSNorm (per context token). ----
  std::vector<float> fused(N * H);
  if (N > 0) {
    const std::vector<float> aux_w = DequantizeTensor(drafter_.aux_norm);
    std::vector<float> aux(N * width);
    for (std::size_t t = 0; t < N; ++t) {
      for (std::size_t k = 0; k < taps; ++k) {
        RmsNormVector(in.features.data() + t * width + k * H, &aux_w[k * H], H,
                      c.rms_eps, aux.data() + t * width + k * H);
      }
    }
    emit_f32("enc_aux_norm", -1, {width, N, 1}, aux);
    std::vector<float> u(N * H);
    MatMulRows(drafter_.fc, 0, aux.data(), N, u.data(), options_);
    emit_f32("fc_out", -1, {H, N, 1}, u);
    const std::vector<float> enc_w = DequantizeTensor(drafter_.enc_output_norm);
    for (std::size_t t = 0; t < N; ++t) {
      RmsNormVector(u.data() + t * H, enc_w.data(), H, c.rms_eps,
                    fused.data() + t * H);
    }
    emit_f32("enc_norm_out", -1, {H, N, 1}, fused);
  }

  // Key positions: injected context rows first, then the block's own rows.
  std::vector<std::int32_t> block_positions(B);
  for (std::size_t r = 0; r < B; ++r) {
    block_positions[r] = in.block_position + static_cast<std::int32_t>(r);
  }
  std::vector<std::int32_t> key_positions(in.context_positions.begin(),
                                          in.context_positions.end());
  key_positions.insert(key_positions.end(), block_positions.begin(),
                       block_positions.end());

  // ---- Block embeddings from the target's token_embd (no scale). ----
  std::vector<float> h(B * H);
  for (std::size_t r = 0; r < B; ++r) {
    const auto token = in.block_tokens[r];
    if (token < 0 ||
        static_cast<std::uint64_t>(token) >= target_.token_embd.rows) {
      throw std::out_of_range("laguna dflash oracle: token outside vocabulary");
    }
    DequantizeRow(
        target_.token_embd.type,
        static_cast<const std::uint8_t*>(target_.token_embd.data) +
            static_cast<std::size_t>(token) * target_.token_embd.RowBytes(),
        h.data() + r * H, H);
  }
  emit_f32("inp_noise_embd", -1, {H, B, 1}, h);

  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    const DFlashLayerWeights& lw = drafter_.layers[il];
    const int layer = static_cast<int>(il);
    const std::vector<float> attn_norm_w = DequantizeTensor(lw.attn_norm);
    const std::vector<float> q_norm_w = DequantizeTensor(lw.attn_q_norm);
    const std::vector<float> k_norm_w = DequantizeTensor(lw.attn_k_norm);

    // K/V rows for [context ; block], layer-local.
    std::vector<float> k_all((N + B) * kv_dim);
    std::vector<float> v_all((N + B) * kv_dim);

    // Context injection: K = rope(knorm(normed @ Wk)) at the target position,
    // V = normed @ Wv.
    if (N > 0) {
      std::vector<float> kv_inp(N * H);
      for (std::size_t t = 0; t < N; ++t) {
        RmsNormVector(fused.data() + t * H, attn_norm_w.data(), H, c.rms_eps,
                      kv_inp.data() + t * H);
      }
      emit_f32("kv_inp_normed", layer, {H, N, 1}, kv_inp);
      std::vector<float> ck(N * kv_dim);
      std::vector<float> cv(N * kv_dim);
      MatMulRows(lw.attn_k, 0, kv_inp.data(), N, ck.data(), options_);
      MatMulRows(lw.attn_v, 0, kv_inp.data(), N, cv.data(), options_);
      for (std::size_t i = 0; i < N * kvh; ++i) {
        RmsNormVector(ck.data() + i * hd, k_norm_w.data(), hd, c.rms_eps,
                      ck.data() + i * hd);
      }
      ApplyRope(ck.data(), in.context_positions,
                static_cast<std::uint32_t>(kvh), c.head_dim, rope);
      emit_f32("Kcur_injected", layer, {hd, kvh, N}, ck);
      emit_f32("Vcur_injected", layer, {hd, kvh, N}, cv);
      std::copy(ck.begin(), ck.end(), k_all.begin());
      std::copy(cv.begin(), cv.end(), v_all.begin());
    }

    // Block: input norm, q/k/v, per-head norms, RoPE.
    std::vector<float> noise(B * H);
    for (std::size_t r = 0; r < B; ++r) {
      RmsNormVector(h.data() + r * H, attn_norm_w.data(), H, c.rms_eps,
                    noise.data() + r * H);
    }
    emit_f32("noise_norm", layer, {H, B, 1}, noise);
    std::vector<float> q(B * q_dim);
    std::vector<float> bk(B * kv_dim);
    std::vector<float> bv(B * kv_dim);
    MatMulRows(lw.attn_q, 0, noise.data(), B, q.data(), options_);
    MatMulRows(lw.attn_k, 0, noise.data(), B, bk.data(), options_);
    MatMulRows(lw.attn_v, 0, noise.data(), B, bv.data(), options_);
    for (std::size_t i = 0; i < B * heads; ++i) {
      RmsNormVector(q.data() + i * hd, q_norm_w.data(), hd, c.rms_eps,
                    q.data() + i * hd);
    }
    for (std::size_t i = 0; i < B * kvh; ++i) {
      RmsNormVector(bk.data() + i * hd, k_norm_w.data(), hd, c.rms_eps,
                    bk.data() + i * hd);
    }
    ApplyRope(q.data(), block_positions, static_cast<std::uint32_t>(heads),
              c.head_dim, rope);
    ApplyRope(bk.data(), block_positions, static_cast<std::uint32_t>(kvh),
              c.head_dim, rope);
    emit_f32("Qcur", layer, {hd, heads, B}, q);
    emit_f32("Kcur", layer, {hd, kvh, B}, bk);
    emit_f32("Vcur", layer, {hd, kvh, B}, bv);
    std::copy(bk.begin(), bk.end(), k_all.begin() + N * kv_dim);
    std::copy(bv.begin(), bv.end(), v_all.begin() + N * kv_dim);

    if (options_.round_kv_f16) {
      // llama.cpp's CPU flash attention also converts Q to F16 (K is F16).
      for (float& value : q) {
        value = RoundToF16(value);
      }
      for (float& value : k_all) {
        value = RoundToF16(value);
      }
      for (float& value : v_all) {
        value = RoundToF16(value);
      }
    }

    // Causal sliding-window attention over [context ; block].
    std::vector<float> attn(B * q_dim);
    GqaAttentionAt(q.data(), block_positions, k_all.data(), v_all.data(),
                   key_positions, static_cast<std::uint32_t>(heads),
                   static_cast<std::uint32_t>(kvh), c.head_dim,
                   c.sliding_window, kq_scale, attn.data());

    // Per-head softplus gate from the same normed input, then output proj.
    std::vector<float> gate(B * heads);
    MatMulRows(lw.attn_gate, 0, noise.data(), B, gate.data(), options_);
    for (float& g : gate) {
      g = SoftplusF32(g);
    }
    emit_f32("attn_gate_softplus", layer, {heads, B, 1}, gate);
    for (std::size_t r = 0; r < B; ++r) {
      for (std::size_t head = 0; head < heads; ++head) {
        float* o = attn.data() + r * q_dim + head * hd;
        for (std::size_t d = 0; d < hd; ++d) {
          o[d] *= gate[r * heads + head];
        }
      }
    }
    emit_f32("attn_gated", layer, {q_dim, B, 1}, attn);
    std::vector<float> o(B * H);
    MatMulRows(lw.attn_output, 0, attn.data(), B, o.data(), options_);
    std::vector<float> ffn_inp(B * H);
    for (std::size_t i = 0; i < B * H; ++i) {
      ffn_inp[i] = o[i] + h[i];
    }
    emit_f32("ffn_inp", layer, {H, B, 1}, ffn_inp);

    // Dense SwiGLU through the target oracle's FfnForward.
    const std::vector<float> ffn_norm_w = DequantizeTensor(lw.ffn_norm);
    std::vector<float> fn(B * H);
    for (std::size_t r = 0; r < B; ++r) {
      RmsNormVector(ffn_inp.data() + r * H, ffn_norm_w.data(), H, c.rms_eps,
                    fn.data() + r * H);
    }
    emit_f32("ffn_norm", layer, {H, B, 1}, fn);
    LayerWeights dense;
    dense.moe = false;
    dense.ffn_gate = lw.ffn_gate;
    dense.ffn_up = lw.ffn_up;
    dense.ffn_down = lw.ffn_down;
    Config ffn_config;
    ffn_config.hidden_size = c.hidden_size;
    const std::vector<float> ffn_out =
        FfnForward(ffn_config, dense, fn.data(), B, options_, {}, -1);
    emit_f32("ffn_out", layer, {H, B, 1}, ffn_out);
    for (std::size_t i = 0; i < B * H; ++i) {
      h[i] = ffn_out[i] + ffn_inp[i];
    }
    emit_f32("l_out", layer, {H, B, 1}, h);
  }

  // ---- Final norm and the target's head, rows 1..B-1. ----
  const std::vector<float> out_w = DequantizeTensor(drafter_.output_norm);
  std::vector<float> normed(B * H);
  for (std::size_t r = 0; r < B; ++r) {
    RmsNormVector(h.data() + r * H, out_w.data(), H, c.rms_eps,
                  normed.data() + r * H);
  }
  emit_f32("result_norm", -1, {H, B, 1}, normed);
  std::vector<float> logits((B - 1) * target_.output.rows);
  MatMulRows(target_.output, 0, normed.data() + H, B - 1, logits.data(),
             options_);
  emit_f32("result_output", -1, {target_.output.rows, B - 1, 1}, logits);
  return logits;
}

std::vector<std::uint32_t> RowArgmax(std::span<const float> rows,
                                     std::size_t width) {
  std::vector<std::uint32_t> best;
  if (width == 0) {
    return best;
  }
  for (std::size_t r = 0; r + width <= rows.size(); r += width) {
    std::size_t arg = 0;
    for (std::size_t i = 1; i < width; ++i) {
      if (rows[r + i] > rows[r + arg]) {
        arg = i;
      }
    }
    best.push_back(static_cast<std::uint32_t>(arg));
  }
  return best;
}

}  // namespace gufo::models::laguna::reference
