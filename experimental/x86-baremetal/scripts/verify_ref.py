#!/usr/bin/env python3
"""
Independent Python/NumPy reference implementation for .asmbin Qwen3-0.6B (Q8_0)
plus verification harness against:
  1. Original PyTorch BF16/FP32 Qwen3-0.6B (--compare-pt)
  2. Pure x86-64 assembly binary asm-llm (--compare-asm)
"""

import argparse
import math
import mmap
import os
import re
import struct
import subprocess
import tempfile
import time
import numpy as np

FIB_MULT = 0x9E3779B97F4A7C15
HT_BITS = 19


def align64(n: int) -> int:
    return (n + 63) & ~63


class AsmBinModel:
    def __init__(self, path: str):
        self.path = path
        self._f = open(path, "rb")
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
        self.buf = memoryview(self._mm)

        hdr = struct.unpack_from("<4sIIIIIIIIIffIIIIQQQQQQ", self.buf, 0)
        assert hdr[0] == b"ASML", f"Bad magic: {hdr[0]}"
        assert hdr[1] in (1, 2), f"Bad version: {hdr[1]}"
        (
            self.magic,
            self.version,
            self.dim,
            self.hidden_dim,
            self.n_layers,
            self.n_heads,
            self.n_kv_heads,
            self.head_dim,
            self.vocab_size,
            self.max_seq_len,
            self.rms_norm_eps,
            self.rope_theta,
            self.eos_1,
            self.eos_2,
            self.ht_mask,
            self.flags,
            self.off_tokenizer,
            self.off_embed,
            self.off_layers,
            self.layer_stride,
            self.off_final_norm,
            self.file_size,
        ) = hdr

        self.fp16_scales = bool(self.flags & 4) or (self.version >= 2)
        self.scale_bytes = 2 if self.fp16_scales else 4
        self.q_dim = self.n_heads * self.head_dim
        self.kv_dim = self.n_kv_heads * self.head_dim
        self._load_tokenizer()
        self._load_weights()

    def close(self):
        self.layers.clear()
        for attr in ("emb_q", "emb_s", "final_norm", "byte_to_tok", "tok_offsets", "ht", "ht_keys", "ht_merged", "ht_rank"):
            if hasattr(self, attr):
                delattr(self, attr)
        self.buf.release()
        self._mm.close()
        self._f.close()

    def _load_tokenizer(self):
        base = self.off_tokenizer
        off_b2t, off_offs, off_bytes, off_ht = struct.unpack_from("<QQQQ", self.buf, base)
        self.byte_to_tok = np.frombuffer(self.buf, dtype="<u4", count=256, offset=base + off_b2t)
        self.tok_offsets = np.frombuffer(
            self.buf, dtype="<u4", count=self.vocab_size + 1, offset=base + off_offs
        )
        bytes_len = int(self.tok_offsets[-1])
        self.tok_bytes = bytes(self.buf[base + off_bytes : base + off_bytes + bytes_len])
        ht_dtype = np.dtype([("key", "<u8"), ("merged_id", "<u4"), ("rank", "<u4")])
        self.ht = np.frombuffer(self.buf, dtype=ht_dtype, count=self.ht_mask + 1, offset=base + off_ht)
        self.ht_keys = self.ht["key"]
        self.ht_merged = self.ht["merged_id"]
        self.ht_rank = self.ht["rank"]

    def _view_q8_0(self, offset: int, rows: int, cols: int):
        nb = cols // 32
        row_stride = cols + nb * self.scale_bytes
        raw = np.frombuffer(self.buf, dtype=np.uint8, count=rows * row_stride, offset=offset).reshape(
            rows, row_stride
        )
        q = raw[:, :cols].view(np.int8)
        if self.fp16_scales:
            s = np.ascontiguousarray(raw[:, cols:]).view("<f2").reshape(rows, nb).astype(np.float32)
        else:
            s = np.ascontiguousarray(raw[:, cols:]).view("<f4").reshape(rows, nb)
        return q, s

    def _load_weights(self):
        self.emb_q, self.emb_s = self._view_q8_0(self.off_embed, self.vocab_size, self.dim)
        self.final_norm = np.frombuffer(self.buf, dtype="<f4", count=self.dim, offset=self.off_final_norm)
        self.layers = []
        for l in range(self.n_layers):
            pos = self.off_layers + l * self.layer_stride
            in_norm = np.frombuffer(self.buf, dtype="<f4", count=self.dim, offset=pos)
            pos += self.dim * 4
            q_norm = np.frombuffer(self.buf, dtype="<f4", count=self.head_dim, offset=pos)
            pos += self.head_dim * 4
            k_norm = np.frombuffer(self.buf, dtype="<f4", count=self.head_dim, offset=pos)
            pos += self.head_dim * 4
            post_norm = np.frombuffer(self.buf, dtype="<f4", count=self.dim, offset=pos)
            pos += self.dim * 4

            def next_mat(rows, cols):
                nonlocal pos
                pos = align64(pos)
                mat = self._view_q8_0(pos, rows, cols)
                pos += rows * (cols + (cols // 32) * self.scale_bytes)
                return mat

            self.layers.append(
                {
                    "in_norm": in_norm,
                    "q_norm": q_norm,
                    "k_norm": k_norm,
                    "post_norm": post_norm,
                    "q_proj": next_mat(self.q_dim, self.dim),
                    "k_proj": next_mat(self.kv_dim, self.dim),
                    "v_proj": next_mat(self.kv_dim, self.dim),
                    "o_proj": next_mat(self.dim, self.q_dim),
                    "gate_proj": next_mat(self.hidden_dim, self.dim),
                    "up_proj": next_mat(self.hidden_dim, self.dim),
                    "down_proj": next_mat(self.dim, self.hidden_dim),
                }
            )

    def ht_lookup(self, left_id: int, right_id: int):
        key = (left_id << 32) | right_id
        slot = ((key * FIB_MULT) & 0xFFFFFFFFFFFFFFFF) >> (64 - HT_BITS)
        while True:
            r = int(self.ht_rank[slot])
            if r == 0xFFFFFFFF:
                return None
            if int(self.ht_keys[slot]) == key:
                return int(self.ht_merged[slot]), r
            slot = (slot + 1) & self.ht_mask

    def encode_bpe(self, text: str) -> list[int]:
        raw = text.encode("utf-8")
        if not raw:
            return []
        toks = [int(self.byte_to_tok[b]) for b in raw]
        while len(toks) > 1:
            best_rank = 0xFFFFFFFF
            best_i = -1
            best_c = -1
            for i in range(len(toks) - 1):
                res = self.ht_lookup(toks[i], toks[i + 1])
                if res is not None and res[1] < best_rank:
                    best_c, best_rank = res
                    best_i = i
            if best_i < 0:
                break
            toks[best_i] = best_c
            del toks[best_i + 1]
        return toks

    def format_chat_prompt(self, user_prompt: str, think: bool = False) -> list[int]:
        # <|im_start|>user\n + BPE(user_prompt) + <|im_end|>\n<|im_start|>assistant\n [+ <think>\n\n</think>\n\n]
        ids = [151644, 872, 198] + self.encode_bpe(user_prompt) + [151645, 198, 151644, 77091, 198]
        if not think:
            ids += [151667, 271, 151668, 271]
        return ids

    def decode(self, ids: list[int]) -> str:
        out = bytearray()
        for tid in ids:
            if 0 <= tid < self.vocab_size:
                a = int(self.tok_offsets[tid])
                b = int(self.tok_offsets[tid + 1])
                out.extend(self.tok_bytes[a:b])
        return out.decode("utf-8", errors="replace")

    @staticmethod
    def matvec_q8_f32(wq: np.ndarray, ws: np.ndarray, x: np.ndarray) -> np.ndarray:
        M, K = wq.shape
        nb = K // 32
        wb = wq.reshape(M, nb, 32).astype(np.float32)
        xb = x.reshape(nb, 32).astype(np.float32)
        # Per-block FP32 dot products scaled by Q8_0 block scales (matches AVX2 kernel)
        dots = np.einsum("mnb,nb->mn", wb, xb, optimize=True).astype(np.float32)
        return np.sum(dots * ws, axis=1, dtype=np.float32)

    @staticmethod
    def rmsnorm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
        ss = np.mean(x * x, axis=-1, keepdims=True, dtype=np.float32)
        inv = np.float32(1.0) / np.sqrt(ss + np.float32(eps))
        return (x * inv * w).astype(np.float32)

    @staticmethod
    def silu_mul(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
        # Overflow-free SiLU matching AVX2 implementation
        ax = np.abs(gate)
        e = np.exp(-ax).astype(np.float32)
        num = np.where(gate < 0, e, np.float32(1.0))
        silu = (gate * num) / (np.float32(1.0) + e)
        return (silu * up).astype(np.float32)

    def forward_token(self, token_id: int, pos: int, k_cache: np.ndarray, v_cache: np.ndarray, compute_logits: bool = True):
        # Embedding dequantization
        x = (
            self.emb_q[token_id].reshape(-1, 32).astype(np.float32)
            * self.emb_s[token_id][:, None]
        ).reshape(-1)

        half = self.head_dim // 2
        freqs = 1.0 / (self.rope_theta ** (np.arange(0, self.head_dim, 2, dtype=np.float64) / self.head_dim))
        angles = pos * freqs
        cos_v = np.cos(angles).astype(np.float32)[None, :]
        sin_v = np.sin(angles).astype(np.float32)[None, :]
        scale = np.float32(1.0 / math.sqrt(self.head_dim))
        gqa_group = self.n_heads // self.n_kv_heads

        for l in range(self.n_layers):
            L = self.layers[l]
            h = self.rmsnorm(x, L["in_norm"], self.rms_norm_eps)
            q = self.matvec_q8_f32(*L["q_proj"], h).reshape(self.n_heads, self.head_dim)
            k = self.matvec_q8_f32(*L["k_proj"], h).reshape(self.n_kv_heads, self.head_dim)
            v = self.matvec_q8_f32(*L["v_proj"], h).reshape(self.n_kv_heads, self.head_dim)

            # QK-Norm per head
            q = self.rmsnorm(q, L["q_norm"], self.rms_norm_eps)
            k = self.rmsnorm(k, L["k_norm"], self.rms_norm_eps)

            # Split-half RoPE
            q1, q2 = q[:, :half], q[:, half:]
            q = np.concatenate([q1 * cos_v - q2 * sin_v, q2 * cos_v + q1 * sin_v], axis=-1)
            k1, k2 = k[:, :half], k[:, half:]
            k = np.concatenate([k1 * cos_v - k2 * sin_v, k2 * cos_v + k1 * sin_v], axis=-1)

            k_cache[l, pos] = k
            v_cache[l, pos] = v

            # GQA attention
            att_out = np.zeros((self.n_heads, self.head_dim), dtype=np.float32)
            for head in range(self.n_heads):
                kv_h = head // gqa_group
                scores = (k_cache[l, : pos + 1, kv_h] @ q[head]) * scale
                scores = scores - np.max(scores)
                probs = np.exp(scores).astype(np.float32)
                probs = probs / np.sum(probs, dtype=np.float32)
                att_out[head] = probs @ v_cache[l, : pos + 1, kv_h]

            x = x + self.matvec_q8_f32(*L["o_proj"], att_out.reshape(-1))

            # SwiGLU MLP
            h2 = self.rmsnorm(x, L["post_norm"], self.rms_norm_eps)
            gate = self.matvec_q8_f32(*L["gate_proj"], h2)
            up = self.matvec_q8_f32(*L["up_proj"], h2)
            act = self.silu_mul(gate, up)
            x = x + self.matvec_q8_f32(*L["down_proj"], act)

        if not compute_logits:
            return None
        x = self.rmsnorm(x, self.final_norm, self.rms_norm_eps)
        return self.matvec_q8_f32(self.emb_q, self.emb_s, x)

    def generate_greedy(self, prompt_ids: list[int], max_new_tokens: int = 64):
        max_len = len(prompt_ids) + max_new_tokens + 4
        k_cache = np.zeros((self.n_layers, max_len, self.n_kv_heads, self.head_dim), dtype=np.float32)
        v_cache = np.zeros((self.n_layers, max_len, self.n_kv_heads, self.head_dim), dtype=np.float32)

        prefill_logits = None
        for pos, tid in enumerate(prompt_ids):
            is_last = pos == len(prompt_ids) - 1
            lg = self.forward_token(tid, pos, k_cache, v_cache, compute_logits=is_last)
            if is_last:
                prefill_logits = lg

        gen = []
        next_tok = int(np.argmax(prefill_logits))
        for step in range(max_new_tokens):
            if next_tok in (self.eos_1, self.eos_2):
                break
            gen.append(next_tok)
            if step == max_new_tokens - 1:
                break
            lg = self.forward_token(next_tok, len(prompt_ids) + step, k_cache, v_cache, compute_logits=True)
            next_tok = int(np.argmax(lg))
        return prefill_logits, gen


def metrics(a: np.ndarray, b: np.ndarray):
    diff = np.abs(a - b)
    max_abs = float(np.max(diff))
    mean_abs = float(np.mean(diff))
    cos = float(np.dot(a.astype(np.float64), b.astype(np.float64)) / (
        np.linalg.norm(a.astype(np.float64)) * np.linalg.norm(b.astype(np.float64))
    ))
    return max_abs, mean_abs, cos


def main():
    ap = argparse.ArgumentParser(description="Verify .asmbin against PyTorch FP32 and asm-llm")
    ap.add_argument("--model", "--asmbin", dest="asmbin", default="models/qwen3-0.6b-q8_0.asmbin", help="Path to exported .asmbin model")
    ap.add_argument("--prompt", default="Explain how a CPU cache works in 3 concise sentences.")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--ref-dir", "--compare-pt", dest="compare_pt", default=None, help="Path to HuggingFace Qwen3-0.6B directory for PyTorch FP32 check")
    ap.add_argument("--compare-asm", default=None, help="Path to compiled build/asm-llm binary")
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()

    model = AsmBinModel(args.asmbin)
    prompt_ids = model.format_chat_prompt(args.prompt, think=args.think)
    print(f"Prompt ({len(prompt_ids)} tokens): {args.prompt!r}")
    print(f"Prompt IDs: {prompt_ids}")

    t0 = time.time()
    ref_logits, ref_gen = model.generate_greedy(prompt_ids, max_new_tokens=args.max_tokens)
    dt = time.time() - t0
    print(f"\n[Python .asmbin Q8_0 ref] ({dt:.1f}s, {len(ref_gen)} tokens):")
    print(f"  top5 ids   : {np.argsort(ref_logits)[-5:][::-1].tolist()}")
    print(f"  top5 logits: {[round(float(x), 4) for x in np.sort(ref_logits)[-5:][::-1]]}")
    print(f"  gen ids    : {ref_gen}")
    print(f"  decoded    : {model.decode(ref_gen)!r}")

    pt_logits = None
    pt_gen_clean = None
    if args.compare_pt:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.compare_pt)
        msgs = [{"role": "user", "content": args.prompt}]
        chat_str = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=args.think
        )
        hf_ids = tok.encode(chat_str, add_special_tokens=False)
        print(f"\n[Tokenizer check vs HF] exact_match={hf_ids == prompt_ids}")
        if hf_ids != prompt_ids:
            print(f"  HF ids : {hf_ids}")
            print(f"  ASM ids: {prompt_ids}")

        pt_model = AutoModelForCausalLM.from_pretrained(
            args.compare_pt, dtype=torch.float32, device_map="cpu"
        )
        pt_model.eval()
        with torch.no_grad():
            inp = torch.tensor([prompt_ids])
            pt_logits = pt_model(inp).logits[0, -1].numpy()
            pt_gen = pt_model.generate(
                inp, max_new_tokens=args.max_tokens, do_sample=False
            )[0, len(prompt_ids) :].tolist()
        # Strip EOS
        pt_gen_clean = []
        for t in pt_gen:
            if t in (model.eos_1, model.eos_2):
                break
            pt_gen_clean.append(t)

        max_d, mean_d, cos = metrics(ref_logits, pt_logits)
        prefix_match = 0
        for a, b in zip(ref_gen, pt_gen_clean):
            if a == b:
                prefix_match += 1
            else:
                break
        print(f"\n[Python .asmbin Q8_0 vs PyTorch FP32]")
        print(f"  max_abs_diff     : {max_d:.6f}")
        print(f"  mean_abs_diff    : {mean_d:.6f}")
        print(f"  cosine_similarity: {cos:.7f}")
        print(f"  top1_match       : {int(np.argmax(ref_logits)) == int(np.argmax(pt_logits))}")
        print(f"  greedy_prefix    : {prefix_match}/{min(len(ref_gen), len(pt_gen_clean))} tokens")
        print(f"  PT decoded       : {tok.decode(pt_gen_clean)!r}")

    if args.compare_asm:
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tf:
            dump_path = tf.name
        try:
            cmd = [
                args.compare_asm,
                "-m",
                args.asmbin,
                "-p",
                args.prompt,
                "-n",
                str(args.max_tokens),
                "-t",
                str(args.threads),
                "--dump-logits",
                dump_path,
                "--print-tokens",
            ]
            if args.think:
                cmd.append("--think")
            proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
            asm_logits = np.fromfile(dump_path, dtype="<f4")
            assert asm_logits.shape == (model.vocab_size,), f"Bad logits shape {asm_logits.shape}"

            # Parse prompt and gen token IDs from stderr
            asm_prompt_ids = None
            asm_gen_ids = None
            for line in proc.stderr.splitlines():
                if line.startswith("PROMPT_TOKENS:"):
                    asm_prompt_ids = [int(x) for x in line.split(":", 1)[1].strip().split(",") if x]
                elif line.startswith("GEN_TOKENS:"):
                    asm_gen_ids = [int(x) for x in line.split(":", 1)[1].strip().split(",") if x]

            max_d, mean_d, cos = metrics(asm_logits, ref_logits)
            top10_asm = np.argsort(asm_logits)[-10:][::-1].tolist()
            top10_ref = np.argsort(ref_logits)[-10:][::-1].tolist()
            exact_gen = asm_gen_ids == ref_gen
            match_cnt = sum(1 for a, b in zip(asm_gen_ids or [], ref_gen) if a == b)
            print(f"\n[Pure x86-64 asm-llm vs Python .asmbin Q8_0 ref]")
            print(f"  prompt_tok_match : {asm_prompt_ids == prompt_ids}")
            print(f"  max_abs_diff     : {max_d:.7f}")
            print(f"  mean_abs_diff    : {mean_d:.7f}")
            print(f"  cosine_similarity: {cos:.9f}")
            print(f"  top10_match      : {top10_asm == top10_ref} (asm={top10_asm[:5]}, ref={top10_ref[:5]})")
            print(f"  greedy_match     : {exact_gen} ({match_cnt}/{len(ref_gen)} tokens)")
            if pt_logits is not None and pt_gen_clean is not None:
                p_max_d, p_mean_d, p_cos = metrics(asm_logits, pt_logits)
                p_prefix = 0
                for a, b in zip(asm_gen_ids or [], pt_gen_clean):
                    if a == b:
                        p_prefix += 1
                    else:
                        break
                p_match = sum(1 for a, b in zip(asm_gen_ids or [], pt_gen_clean) if a == b)
                print(f"\n[Pure x86-64 asm-llm vs PyTorch FP32]")
                print(f"  max_abs_diff     : {p_max_d:.6f}")
                print(f"  mean_abs_diff    : {p_mean_d:.6f}")
                print(f"  cosine_similarity: {p_cos:.7f}")
                print(f"  top1_match       : {int(np.argmax(asm_logits)) == int(np.argmax(pt_logits))}")
                print(f"  greedy_prefix    : {p_prefix}/{min(len(asm_gen_ids or []), len(pt_gen_clean))} tokens (pos_match={p_match}/{len(pt_gen_clean)})")
            print(f"  asm stdout       : {proc.stdout!r}")
            print(f"  asm stderr       :\n{proc.stderr.strip()}")
            if not exact_gen:
                print(f"  ASM gen ids: {asm_gen_ids}")
                print(f"  REF gen ids: {ref_gen}")
        finally:
            if os.path.exists(dump_path):
                os.unlink(dump_path)

    model.close()


if __name__ == "__main__":
    main()
