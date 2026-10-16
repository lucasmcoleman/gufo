"""Extract the KDA subset of the HF fixture set into kda_fixture.bin.

Sources (GLM-5.3-Flash fixture set, HF transformers 09281b8, random
weights; see that set's README.md and manifest.json):
  kda_recurrent_vs_chunk.npz  recurrent_kimi_delta_attention vs the chunk
      form, Dk 12 / Dv 8 / 3 heads / 2 x 20 tokens, unequal per-key decay,
      nonzero initial state, plus the 2x2 key-row decay orientation case.
      Stored with the prefix "rc.".
  kda_layer.npz  the layer-0 Glm5NextTextLinearAttention forward (hidden 32,
      4 heads of 8, 2 x 11 tokens) with every intermediate and weight.
      Stored with the prefix "layer.".

Format (little endian): b"GLMKDAF1", u32 array count, then per array:
u32 name length, name bytes, u32 ndim, u32 dims[ndim], f32 data (C order).

Usage: python3 make_kda_fixture.py <kda_recurrent_vs_chunk.npz>
           <kda_layer.npz> <kda_fixture.bin>
"""

import hashlib
import struct
import sys

import numpy as np


def main() -> None:
    rc_src, layer_src, dst = sys.argv[1], sys.argv[2], sys.argv[3]
    arrays = []
    for prefix, src in (("rc.", rc_src), ("layer.", layer_src)):
        z = np.load(src)
        for key in sorted(z.files):
            arrays.append((prefix + key, z[key]))
        print(f"{src} sha256 {hashlib.sha256(open(src, 'rb').read()).hexdigest()}")
    out = bytearray(b"GLMKDAF1")
    out += struct.pack("<I", len(arrays))
    for key, arr in arrays:
        a = np.ascontiguousarray(arr, dtype="<f4")
        name = key.encode()
        out += struct.pack("<I", len(name)) + name
        out += struct.pack("<I", a.ndim) + struct.pack(f"<{a.ndim}I", *a.shape)
        out += a.tobytes()
    open(dst, "wb").write(out)
    print(f"wrote {dst}: {len(arrays)} arrays, {len(out)} bytes, "
          f"sha256 {hashlib.sha256(out).hexdigest()}")


if __name__ == "__main__":
    main()
