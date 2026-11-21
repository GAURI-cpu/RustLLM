# Makes a small random Qwen2-shaped GGUF whose weights use many storage types,
# to check the engine against Candle on every format.
#   python scripts/make_test_model.py out.gguf mix|q4km
import sys, numpy as np
from gguf import GGUFWriter, GGMLQuantizationType as T
out, style = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(7)
H, FF, HEADS, KV, LAYERS, VOCAB = 256, 512, 4, 2, 2, 512
HD = H // HEADS
# bytes per block, numbers per block, f16 fields: (offset, low, high)
BLOCK = {
    T.Q4_0: (18, 32, [(0, -.02, .02)]), T.Q4_1: (20, 32, [(0, .002, .006), (2, -.03, -.01)]),
    T.Q5_0: (22, 32, [(0, -.01, .01)]), T.Q5_1: (24, 32, [(0, .001, .003), (2, -.03, -.01)]),
    T.Q8_0: (34, 32, [(0, -.0004, .0004)]), T.Q2_K: (84, 256, [(80, .001, .002), (82, .0005, .001)]),
    T.Q3_K: (110, 256, [(108, .0002, .0004)]), T.Q4_K: (144, 256, [(0, .00005, .0001), (2, .00005, .0001)]),
    T.Q5_K: (176, 256, [(0, .00002, .00005), (2, .00005, .0001)]), T.Q6_K: (210, 256, [(208, -.0002, .0002)]),
}
def packed(rows, cols, t):
    if t == T.F32: return rng.normal(0, .05, (rows, cols)).astype(np.float32), None
    if t == T.F16: return rng.normal(0, .05, (rows, cols)).astype(np.float16), None
    bb, n, halves = BLOCK[t]
    blocks = rows * cols // n
    raw = rng.integers(0, 256, size=(blocks, bb), dtype=np.uint8)
    for off, lo, hi in halves:
        raw[:, off:off + 2] = rng.uniform(lo, hi, blocks).astype(np.float16).view(np.uint8).reshape(blocks, 2)
    return raw.reshape(rows, cols // n * bb), t

if style == 'mix':   # every type at least once
    plan = [{'attn_q': T.Q4_K, 'attn_k': T.Q5_0, 'attn_v': T.Q8_0, 'attn_output': T.Q5_K,
             'ffn_gate': T.Q4_1, 'ffn_up': T.Q5_1, 'ffn_down': T.Q6_K},
            {'attn_q': T.Q2_K, 'attn_k': T.Q3_K, 'attn_v': T.Q4_0, 'attn_output': T.F16,
             'ffn_gate': T.Q4_K, 'ffn_up': T.Q5_K, 'ffn_down': T.Q8_0}]
    embed, output = T.Q8_0, T.Q6_K
else:                # like llama.cpp's Q4_K_M: mostly Q4_K, some Q6_K / Q5_0
    plan = [{'attn_q': T.Q5_0, 'attn_k': T.Q5_0, 'attn_v': T.Q6_K, 'attn_output': T.Q5_0,
             'ffn_gate': T.Q4_K, 'ffn_up': T.Q4_K, 'ffn_down': T.Q6_K}] * LAYERS
    embed, output = T.Q8_0, None   # tied embeddings, like Qwen2.5-0.5B

w = GGUFWriter(out, 'qwen2')
w.add_context_length(512); w.add_embedding_length(H); w.add_block_count(LAYERS)
w.add_feed_forward_length(FF); w.add_head_count(HEADS); w.add_head_count_kv(KV)
w.add_layer_norm_rms_eps(1e-6); w.add_rope_freq_base(1e6)
def add(name, rows, cols, t):
    data, raw = packed(rows, cols, t)
    if raw is None: w.add_tensor(name, data)
    else: w.add_tensor(name, data, raw_shape=data.shape, raw_dtype=raw)   # raw_shape = the shape in bytes
add('token_embd.weight', VOCAB, H, embed)
if output: add('output.weight', VOCAB, H, output)
w.add_tensor('output_norm.weight', (1 + rng.normal(0, .1, H)).astype(np.float32))
shape = {'attn_q': (H, H), 'attn_k': (KV * HD, H), 'attn_v': (KV * HD, H), 'attn_output': (H, H),
         'ffn_gate': (FF, H), 'ffn_up': (FF, H), 'ffn_down': (H, FF)}
for i, types in enumerate(plan):
    w.add_tensor(f'blk.{i}.attn_norm.weight', (1 + rng.normal(0, .1, H)).astype(np.float32))
    w.add_tensor(f'blk.{i}.ffn_norm.weight', (1 + rng.normal(0, .1, H)).astype(np.float32))
    for b in ('attn_q', 'attn_k', 'attn_v'):
        w.add_tensor(f'blk.{i}.{b}.bias', rng.normal(0, .02, shape[b][0]).astype(np.float32))
    for name, t in types.items():
        add(f'blk.{i}.{name}.weight', *shape[name], t)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print('wrote', out, style)