#!/usr/bin/env python3
"""
Export LiquidAI LFM3-1.7B (BF16 safetensors + tokenizer.json) into the
.asmbin v3 layout for the Qwenputer-X86 kernel's LFM3 path.

LFM3 is NOT a Qwen3: 24 layers = 15 short-conv + 5 full-attention +
4 sliding-attention (window 256), sandwich RMSNorms, per-head sigmoid
attention gate, partial zero-tail RoPE on full layers (theta 1e6,
factor 0.5), full RoPE theta 1e4 on sliding layers, final norm =
model.embedding_norm.weight, tied embeddings, vocab 128000.

v3 format deltas vs v2:
  - header version = 3, flags |= FLAG_LFM3 (8)
  - header reserved Qword @112: offset of the arch table
  - header reserved u32s @120/@124: sliding_window, conv_kernel_size
  - arch table section: layer_types[24] (0=conv, 1=full, 2=sliding),
    f32 rope_theta_sliding @24, u32 im_start @28, u32 bos @32, then
    u64 layer_offsets[24] @40 (non-uniform layer stride: conv and
    attention layers differ in size)
  - per-layer layout (offsets from layer start):
        0      operator_norm      fp32 [2048]
        8192   post_operator_norm fp32 [2048]
        16384  ffn_norm           fp32 [2048]
        24576  post_ffn_norm      fp32 [2048]
        32768  q_norm             fp32 [128]   (attention layers only)
        33280  k_norm             fp32 [128]   (attention layers only)
        33792  SLOT_A  conv in_proj [6144,2048] Q8 | attn q_proj [2048,2048] Q8
        +A     SLOT_B  conv weight fp32 [2048,7]  | attn k_proj [512,2048] Q8
        +B     SLOT_C  conv out_proj [2048,2048] Q8 | attn v_proj [512,2048] Q8
        +C     SLOT_D  attn o_proj [2048,2048] Q8   (conv: empty)
        +D     SLOT_E  attn gate_proj [16,2048] Q8  (conv: empty)
        +E     w1 [7680,2048] Q8
               w3 [7680,2048] Q8
               w2 [2048,7680] Q8
    Slots are individually sized per layer type (offsets table makes
    the stride non-uniform); unused slots are omitted entirely.
"""

import argparse
import json
import os
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_weights import (HEADER_SIZE, MAGIC, TensorStore, align64,
                            build_tokenizer_blob, pad64,
                            quantize_matrix_q8_0_row_split)

VERSION_V3 = 3
FLAG_TIED_EMB = 1
FLAG_QK_NORM = 2
FLAG_FP16_SCALES = 4
FLAG_LFM3 = 8

LAYER_TYPES = {"conv": 0, "full_attention": 1, "sliding_attention": 2}

# Per-layer fixed offsets (bytes from layer start)
N_OFF_OP = 0
N_OFF_POST_OP = 8192
N_OFF_FFN = 16384
N_OFF_POST_FFN = 24576
N_OFF_QNORM = 32768
N_OFF_KNORM = 33280
OFF_SLOTS = 33792


def export_lfm3(model_dir: str, out_path: str, max_seq_len: int = 2048):
    t0 = time.time()
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    assert cfg["model_type"] == "lfm3", cfg["model_type"]

    dim = int(cfg["hidden_size"])
    hidden_dim = int(cfg["intermediate_size"])
    n_layers = int(cfg["num_hidden_layers"])
    n_heads = int(cfg["num_attention_heads"])
    n_kv_heads = int(cfg["num_key_value_heads"])
    head_dim = int(cfg["head_dim"])
    vocab_size = int(cfg["vocab_size"])
    eps = float(cfg["norm_eps"])
    layer_types = [LAYER_TYPES[t] for t in cfg["layer_types"]]
    rope_full = float(cfg["rope_parameters"]["full_attention"]["rope_theta"])
    rope_sliding = float(cfg["rope_parameters"]["sliding_attention"]["rope_theta"])
    sliding_window = int(cfg["sliding_window"])
    conv_kernel = int(cfg["conv_L_cache"])
    eos_1 = int(cfg["eos_token_id"])            # <|im_end|> 124900
    eos_2 = 124895                              # <|endoftext|>

    store = TensorStore(model_dir)
    tok_blob = build_tokenizer_blob(
        os.path.join(model_dir, "tokenizer.json"), vocab_size)

    def q8(name, shape):
        mat = store.read_f32(name)
        assert mat.shape == shape, f"{name}: got {mat.shape}, want {shape}"
        return quantize_matrix_q8_0_row_split(mat, scale_dtype="fp16")

    def f4(name, shape):
        v = store.read_f32(name)
        assert v.shape == shape, f"{name}: got {v.shape}, want {shape}"
        return v.astype("<f4").tobytes()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "wb") as out:
        out.write(b"\x00" * HEADER_SIZE)

        off_tokenizer = pad64(out)
        out.write(tok_blob)

        # Arch table placeholder (offsets known after layers are laid out)
        off_arch = pad64(out)
        arch_size = align64(40 + 24 * 8)
        out.write(b"\x00" * arch_size)

        off_embed = pad64(out)
        emb = store.read_f32("model.embed_tokens.weight")
        assert emb.shape == (vocab_size, dim)
        print(f"Quantizing embed_tokens {emb.shape} -> Q8_0 ...")
        out.write(quantize_matrix_q8_0_row_split(emb, scale_dtype="fp16"))
        del emb

        off_layers = pad64(out)
        layer_offsets = []
        for l in range(n_layers):
            layer_start = pad64(out)
            layer_offsets.append(layer_start - off_layers)
            p = f"model.layers.{l}"
            ltype = layer_types[l]

            norms = bytearray(OFF_SLOTS)
            for name, off in [
                (f"{p}.operator_norm.weight", N_OFF_OP),
                (f"{p}.post_operator_norm.weight", N_OFF_POST_OP),
                (f"{p}.ffn_norm.weight", N_OFF_FFN),
                (f"{p}.post_ffn_norm.weight", N_OFF_POST_FFN),
            ]:
                norms[off:off + dim * 4] = f4(name, (dim,))
            if ltype != 0:
                norms[N_OFF_QNORM:N_OFF_QNORM + head_dim * 4] = f4(
                    f"{p}.self_attn.q_norm.weight", (head_dim,))
                norms[N_OFF_KNORM:N_OFF_KNORM + head_dim * 4] = f4(
                    f"{p}.self_attn.k_norm.weight", (head_dim,))
            out.write(norms)

            if ltype == 0:  # conv layer
                out.write(q8(f"{p}.conv.in_proj.weight", (3 * dim, dim)))
                pad64(out)
                conv_w = store.read_f32(f"{p}.conv.conv.weight")
                assert conv_w.shape == (dim, 1, conv_kernel)
                out.write(conv_w.astype("<f4").tobytes())
                pad64(out)
                out.write(q8(f"{p}.conv.out_proj.weight", (dim, dim)))
                pad64(out)
            else:  # attention layer
                out.write(q8(f"{p}.self_attn.q_proj.weight",
                             (n_heads * head_dim, dim)))
                pad64(out)
                out.write(q8(f"{p}.self_attn.k_proj.weight",
                             (n_kv_heads * head_dim, dim)))
                pad64(out)
                out.write(q8(f"{p}.self_attn.v_proj.weight",
                             (n_kv_heads * head_dim, dim)))
                pad64(out)
                out.write(q8(f"{p}.self_attn.o_proj.weight",
                             (dim, n_heads * head_dim)))
                pad64(out)
                out.write(q8(f"{p}.self_attn.gate_proj.weight",
                             (n_heads, dim)))
                pad64(out)
            out.write(q8(f"{p}.feed_forward.w1.weight", (hidden_dim, dim)))
            pad64(out)
            out.write(q8(f"{p}.feed_forward.w3.weight", (hidden_dim, dim)))
            pad64(out)
            out.write(q8(f"{p}.feed_forward.w2.weight", (dim, hidden_dim)))
            pad64(out)
            if (l + 1) % 6 == 0 or l == n_layers - 1:
                print(f"  Exported layer {l + 1}/{n_layers} "
                      f"(type={cfg['layer_types'][l]})")

        off_final_norm = pad64(out)
        out.write(f4("model.embedding_norm.weight", (dim,)))
        file_size = pad64(out)

        # Arch table: types, sliding theta, im_start, bos, layer offsets
        arch = bytearray(arch_size)
        arch[0:24] = bytes(layer_types)
        struct.pack_into("<f", arch, 24, rope_sliding)
        struct.pack_into("<I", arch, 28, 124899)   # <|im_start|>
        struct.pack_into("<I", arch, 32, int(cfg["bos_token_id"]))
        for i, off in enumerate(layer_offsets):
            struct.pack_into("<Q", arch, 40 + i * 8, off)
        out.seek(off_arch)
        out.write(arch)

        flags = FLAG_TIED_EMB | FLAG_QK_NORM | FLAG_FP16_SCALES | FLAG_LFM3
        header = struct.pack(
            "<4sIIIIIIIIIffIIIIQQQQQQQII",
            MAGIC, VERSION_V3, dim, hidden_dim, n_layers, n_heads,
            n_kv_heads, head_dim, vocab_size, max_seq_len, eps, rope_full,
            eos_1, eos_2, 524287, flags,
            off_tokenizer, off_embed, off_layers, 0, off_final_norm,
            file_size, off_arch, sliding_window, conv_kernel,
        )
        assert len(header) == HEADER_SIZE, len(header)
        out.seek(0)
        out.write(header)

    dt = time.time() - t0
    print(f"\nExport complete: {out_path}")
    print(f"  Format    : .asmbin v3 (LFM3, flags=0x{flags:x})")
    print(f"  Total size: {file_size:,} bytes ({file_size / 1024**2:.1f} MiB)")
    print(f"  Layers    : {layer_types.count(0)} conv, "
          f"{layer_types.count(1)} full, {layer_types.count(2)} sliding")
    print(f"  Elapsed   : {dt:.1f} s")


def main():
    ap = argparse.ArgumentParser(description="Export LFM3-1.7B to .asmbin v3")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    args = ap.parse_args()
    export_lfm3(args.model_dir, args.out, args.max_seq_len)


if __name__ == "__main__":
    main()
