"""Extract the hyper-connection subset of the HF fixture into hc_fixture.bin.

Source: hyper_connection.npz from the GLM-5.3-Flash fixture set (HF
transformers 09281b8, Glm5NextTextHyperConnection layer-0 attn_hc, random
weights, seed 5681; see that set's README.md and manifest.json).

Format (little endian): b"GLMHCFX1", u32 array count, then per array:
u32 name length, name bytes, u32 ndim, u32 dims[ndim], f32 data (C order).

Usage: python3 make_hc_fixture.py <hyper_connection.npz> <hc_fixture.bin>
"""

import hashlib
import struct
import sys

import numpy as np

KEYS = ("H", "mix", "pre", "post", "comb", "collapsed", "block_out", "hc_out",
        "mean_head", "w.fn", "w.base", "w.scale")


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    z = np.load(src)
    out = bytearray(b"GLMHCFX1")
    out += struct.pack("<I", len(KEYS))
    for key in KEYS:
        a = np.ascontiguousarray(z[key], dtype="<f4")
        name = key.encode()
        out += struct.pack("<I", len(name)) + name
        out += struct.pack("<I", a.ndim) + struct.pack(f"<{a.ndim}I", *a.shape)
        out += a.tobytes()
    open(dst, "wb").write(out)
    print(f"source sha256 {hashlib.sha256(open(src, 'rb').read()).hexdigest()}")
    print(f"wrote {dst}: {len(out)} bytes, "
          f"sha256 {hashlib.sha256(out).hexdigest()}")


if __name__ == "__main__":
    main()
