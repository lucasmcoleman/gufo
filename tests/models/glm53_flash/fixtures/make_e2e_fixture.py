"""Convert the HF end-to-end fixture into GGUF-named F32 arrays.

Source: e2e_tiny.npz from the GLM-5.3-Flash fixture set (HF transformers
09281b8, Glm5NextTextModel, hidden 32, 5 layers: KDA x3 + dense FFN, MLA/DSA +
MoE, KDA + MoE; random weights seed 1234, input seed 5685; see that set's
README.md and manifest.json).

Weights are renamed and reshaped exactly as llama.cpp's Glm5NextModel
converter (conversion/glm.py @ 23f7aab) does, so the C++ test reads the same
names and layouts gufo binds from a real GGUF (weights.cpp):
  * the fused HF KDA conv1d [3*dim, 1, K] (q|k|v channels) splits into
    ssm_conv1d_{q,k,v}.weight [dim, 1, K];
  * A_log becomes ssm_a = -exp(A_log); dt_bias becomes ssm_dt.bias;
  * kv_b_proj [H*(nope+v), kv_lora] is viewed [H, nope+v, kv_lora] and split:
    attn_k_b = k_b.transpose(1, 2) [H, kv_lora, nope], attn_v_b [H, v, kv_lora];
  * experts.gate_up_proj [E, 2I, hidden] splits gate-first into
    ffn_{gate,up}_exps [E, I, hidden]; down_proj is ffn_down_exps [E, hidden, I];
  * the router bias e_score_correction_bias is exp_probs_b.bias.
Array shapes are numpy (C order); the last axis is the GGUF ne[0].

Taps and outputs keep their npz names with the batch axis dropped. The MLA
attention-probability tap (self_attn.out1) is not exported: the absorbed
reference never forms the unabsorbed probabilities.

Format (little endian): b"GLME2EF1", u32 array count, then per array:
u32 name length, name bytes, u32 ndim, u32 dims[ndim], f32 data (C order).

Usage: python3 make_e2e_fixture.py <e2e_tiny.npz> <e2e_fixture.bin>
"""

import hashlib
import re
import struct
import sys

import numpy as np

DIRECT = {
    "attn_hc.fn": "hc_attn_fn.weight",
    "attn_hc.base": "hc_attn_base.weight",
    "attn_hc.scale": "hc_attn_scale.weight",
    "ffn_hc.fn": "hc_ffn_fn.weight",
    "ffn_hc.base": "hc_ffn_base.weight",
    "ffn_hc.scale": "hc_ffn_scale.weight",
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    # KDA
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.forget_gate.dt_bias": "ssm_dt.bias",
    "self_attn.forget_gate.f_a_proj.weight": "ssm_f_a.weight",
    "self_attn.forget_gate.f_b_proj.weight": "ssm_f_b.weight",
    "self_attn.b_proj.weight": "ssm_beta.weight",
    "self_attn.g_a_proj.weight": "ssm_g_a.weight",
    "self_attn.g_b_proj.weight": "ssm_g_b.weight",
    "self_attn.o_norm.weight": "ssm_norm.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    # MLA
    "self_attn.q_a_proj.weight": "attn_q_a.weight",
    "self_attn.q_a_layernorm.weight": "attn_q_a_norm.weight",
    "self_attn.q_b_proj.weight": "attn_q_b.weight",
    "self_attn.kv_a_proj_with_mqa.weight": "attn_kv_a_mqa.weight",
    "self_attn.kv_a_layernorm.weight": "attn_kv_a_norm.weight",
    # DSA indexer
    "self_attn.indexer.wq_b.weight": "indexer.attn_q_b.weight",
    "self_attn.indexer.wk.weight": "indexer.attn_k.weight",
    "self_attn.indexer.k_norm.weight": "indexer.k_norm.weight",
    "self_attn.indexer.k_norm.bias": "indexer.k_norm.bias",
    "self_attn.indexer.weights_proj.weight": "indexer.proj.weight",
    "self_attn.indexer.index_kpool_compress_gate":
        "indexer_compressor_gate.weight",
    "self_attn.indexer.index_kpool_compress_ape":
        "indexer_compressor_ape.weight",
    # Dense FFN
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
    # MoE
    "mlp.gate.weight": "ffn_gate_inp.weight",
    "mlp.gate.e_score_correction_bias": "exp_probs_b.bias",
    "mlp.experts.down_proj": "ffn_down_exps.weight",
    "mlp.shared_experts.gate_proj.weight": "ffn_gate_shexp.weight",
    "mlp.shared_experts.up_proj.weight": "ffn_up_shexp.weight",
    "mlp.shared_experts.down_proj.weight": "ffn_down_shexp.weight",
}


def convert_weights(z, num_heads, nope, v_dim):
    out = {}
    for key in z.files:
        if not key.startswith("w."):
            continue
        name = key[2:]
        a = z[key].astype("<f4")
        if name == "embed_tokens.weight":
            out["token_embd.weight"] = a
        elif name == "norm.weight":
            out["output_norm.weight"] = a
        else:
            m = re.fullmatch(r"layers\.(\d+)\.(.+)", name)
            assert m, name
            p, rest = f"blk.{m.group(1)}.", m.group(2)
            if rest in DIRECT:
                out[p + DIRECT[rest]] = a
            elif rest == "self_attn.conv1d.weight":
                for part, w in zip("qkv", np.split(a, 3, axis=0)):
                    out[p + f"ssm_conv1d_{part}.weight"] = w
            elif rest == "self_attn.forget_gate.A_log":
                out[p + "ssm_a"] = -np.exp(a)
            elif rest == "self_attn.kv_b_proj.weight":
                kv_b = a.reshape(num_heads, nope + v_dim, a.shape[-1])
                out[p + "attn_k_b.weight"] = kv_b[:, :nope].transpose(0, 2, 1)
                out[p + "attn_v_b.weight"] = kv_b[:, nope:]
            elif rest == "mlp.experts.gate_up_proj":
                gate, up = np.split(a, 2, axis=1)
                out[p + "ffn_gate_exps.weight"] = gate
                out[p + "ffn_up_exps.weight"] = up
            else:
                raise KeyError(f"unmapped HF weight {name}")
    return out


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    z = np.load(src)
    # Tiny config (manifest global.config): 4 heads, nope 8, v 8.
    arrays = convert_weights(z, num_heads=4, nope=8, v_dim=8)
    for key in z.files:
        if key.startswith("w.") or key.endswith("self_attn.out1"):
            continue
        a = z[key]
        assert a.shape[0] == 1, key
        arrays[key] = a[0].astype("<f4")  # drop the batch axis
    out = bytearray(b"GLME2EF1")
    out += struct.pack("<I", len(arrays))
    for key in sorted(arrays):
        a = np.ascontiguousarray(arrays[key], dtype="<f4")
        name = key.encode()
        out += struct.pack("<I", len(name)) + name
        out += struct.pack("<I", a.ndim) + struct.pack(f"<{a.ndim}I", *a.shape)
        out += a.tobytes()
    open(dst, "wb").write(out)
    print(f"source sha256 {hashlib.sha256(open(src, 'rb').read()).hexdigest()}")
    print(f"wrote {dst}: {len(arrays)} arrays, {len(out)} bytes, "
          f"sha256 {hashlib.sha256(out).hexdigest()}")


if __name__ == "__main__":
    main()
