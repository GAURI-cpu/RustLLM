# Makes engine/tests/data/golden_quants.bin: packed blocks + what the reference (gguf-py) unpacks them to.
import numpy as np, struct, sys
from gguf import quants, GGMLQuantizationType as T
rng = np.random.default_rng(42)
# type id: (block bytes, numbers per block, byte offsets of f16 scale fields)
types = {
    T.F16: (2, 1, None), T.BF16: (2, 1, None),
    T.Q4_0: (18, 32, [0]), T.Q4_1: (20, 32, [0, 2]), T.Q5_0: (22, 32, [0]), T.Q5_1: (24, 32, [0, 2]),
    T.Q8_0: (34, 32, [0]), T.Q2_K: (84, 256, [80, 82]), T.Q3_K: (110, 256, [108]),
    T.Q4_K: (144, 256, [0, 2]), T.Q5_K: (176, 256, [0, 2]), T.Q6_K: (210, 256, [208]),
}
out = open(sys.argv[1], 'wb')
for t, (bb, n, halves) in types.items():
    blocks = 3 if n > 1 else 64
    raw = bytearray(rng.integers(0, 256, size=bb * blocks, dtype=np.uint8).tobytes())
    if t == T.F16:
        raw = bytearray(rng.normal(0, 2, blocks).astype(np.float16).tobytes())
    elif t == T.BF16:
        raw = bytearray((rng.normal(0, 2, blocks).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16).tobytes())
    else:
        for b in range(blocks):
            for off in halves:
                raw[b * bb + off: b * bb + off + 2] = np.float16(rng.uniform(-0.1, 0.1)).tobytes()
    arr = np.frombuffer(bytes(raw), dtype=np.uint8)
    expect = quants.dequantize(arr, t).astype(np.float32).ravel()
    assert expect.size == blocks * n and np.isfinite(expect).all()
    out.write(struct.pack('<II', int(t), len(raw))); out.write(bytes(raw))
    out.write(struct.pack('<I', expect.size)); out.write(expect.tobytes())
    print(t.name, blocks, 'blocks', expect.size, 'numbers')