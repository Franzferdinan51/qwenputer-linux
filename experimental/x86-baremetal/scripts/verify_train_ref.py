#!/usr/bin/env python3
"""
Phase 3 Verification Harness (strictly CPU-only, CUDA_VISIBLE_DEVICES="").

Verifies build/asm-gpt-train against an independent PyTorch FP32 CPU reference:
  1. Forward Cross-Entropy Loss parity (abs_diff < 1e-5)
  2. Per-tensor Analytical Backward Gradient Check across EVERY parameter tensor
     (tok_emb, all ln1/wq/wk/wv/wo/ln2/wgate/wup/wdown, final_ln, lm_head):
       - cosine_similarity > 0.99999
       - max_abs_diff      < 1e-4
  3. 10-step Consecutive AdamW + Warmup/Cosine LR + Global Grad-Clip Trajectory:
       - loss[0..9] abs_diff < 1e-4
       - final_weights max_abs_diff < 1e-4
  4. Multi-thread Determinism (-t 1 vs -t 8 vs -t 16):
       - losses, step0_grads, final_weights match within < 1e-5 (target: exact 0.0!)
"""

import os

# Enforce strict CPU-only execution before importing torch!
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import math
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * inv_rms * self.weight


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, T, H, hd], cos/sin: [1, T, 1, hd//2]
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, hidden_dim: int, eps: float):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.ln1 = RMSNorm(dim, eps)
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.ln2 = RMSNorm(dim, eps)
        self.wgate = nn.Linear(dim, hidden_dim, bias=False)
        self.wup = nn.Linear(dim, hidden_dim, bias=False)
        self.wdown = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        h = self.ln1(x)
        q = self.wq(h).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(h).view(B, T, self.n_heads, self.head_dim)
        v = self.wv(h).view(B, T, self.n_heads, self.head_dim)

        q = apply_rope(q, cos, sin).transpose(1, 2)  # [B, H, T, hd]
        k = apply_rope(k, cos, sin).transpose(1, 2)
        v = v.transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        probs = F.softmax(scores, dim=-1)
        att = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.wo(att)

        h2 = self.ln2(x)
        act = F.silu(self.wgate(h2)) * self.wup(h2)
        x = x + self.wdown(act)
        return x


class MiniGPT(nn.Module):
    def __init__(
        self,
        vocab: int,
        dim: int,
        n_layers: int,
        n_heads: int,
        hidden_dim: int,
        seq_len: int,
        eps: float,
        rope_theta: float,
    ):
        super().__init__()
        self.vocab = vocab
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len

        self.tok_emb = nn.Embedding(vocab, dim)
        self.layers = nn.ModuleList(
            [TransformerBlock(dim, n_heads, hidden_dim, eps) for _ in range(n_layers)]
        )
        self.final_ln = RMSNorm(dim, eps)
        self.lm_head = nn.Linear(dim, vocab, bias=False)

        # Exact FP64 RoPE frequencies -> FP32 cos/sin tables (matches asm init_rope_freqs)
        idx = torch.arange(0, self.head_dim, 2, dtype=torch.float64)
        inv_freq = torch.exp(-math.log(rope_theta) * idx / self.head_dim)
        pos = torch.arange(seq_len, dtype=torch.float64)[:, None]
        angles = pos * inv_freq[None, :]
        self.register_buffer("cos", torch.cos(angles).to(torch.float32)[None, :, None, :])
        self.register_buffer("sin", torch.sin(angles).to(torch.float32)[None, :, None, :])

    def named_flat_params(self):
        """Exact parameter order matching model.asm flat buffer layout."""
        items = [("tok_emb", self.tok_emb.weight)]
        for l, blk in enumerate(self.layers):
            items.extend(
                [
                    (f"L{l}.ln1", blk.ln1.weight),
                    (f"L{l}.wq", blk.wq.weight),
                    (f"L{l}.wk", blk.wk.weight),
                    (f"L{l}.wv", blk.wv.weight),
                    (f"L{l}.wo", blk.wo.weight),
                    (f"L{l}.ln2", blk.ln2.weight),
                    (f"L{l}.wgate", blk.wgate.weight),
                    (f"L{l}.wup", blk.wup.weight),
                    (f"L{l}.wdown", blk.wdown.weight),
                ]
            )
        items.append(("final_ln", self.final_ln.weight))
        items.append(("lm_head", self.lm_head.weight))
        return items

    def load_flat_weights(self, flat: np.ndarray):
        off = 0
        with torch.no_grad():
            for _, p in self.named_flat_params():
                n = p.numel()
                p.copy_(torch.from_numpy(flat[off : off + n].copy()).view_as(p))
                off += n
        assert off == flat.size, f"Param count mismatch: {off} vs {flat.size}"

    def dump_flat_weights(self) -> np.ndarray:
        return np.concatenate(
            [p.detach().cpu().numpy().reshape(-1) for _, p in self.named_flat_params()]
        )

    def forward(self, idx: torch.Tensor, targets: torch.Tensor):
        B, T = idx.shape
        x = self.tok_emb(idx)
        cos = self.cos[:, :T]
        sin = self.sin[:, :T]
        for blk in self.layers:
            x = blk(x, cos, sin)
        x = self.final_ln(x)
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, self.vocab), targets.view(-1))
        return logits, loss


def compute_lr_ref(step: int, total_steps: int, warmup_steps: int, max_lr: float) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return float(np.float32(max_lr * (step + 1) / warmup_steps))
    if total_steps <= warmup_steps or total_steps <= 1:
        return float(np.float32(max_lr))
    progress = min(1.0, (step - warmup_steps) / (total_steps - warmup_steps))
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(np.float32(max_lr * (0.1 + 0.9 * coeff)))


def tensor_metrics(a: np.ndarray, b: np.ndarray):
    a64 = a.astype(np.float64).reshape(-1)
    b64 = b.astype(np.float64).reshape(-1)
    diff = np.abs(a64 - b64)
    max_abs = float(np.max(diff))
    mean_abs = float(np.mean(diff))
    na = float(np.linalg.norm(a64))
    nb = float(np.linalg.norm(b64))
    if na == 0.0 and nb == 0.0:
        cos = 1.0
    elif na == 0.0 or nb == 0.0:
        cos = 0.0
    else:
        cos = float(np.dot(a64, b64) / (na * nb))
    return max_abs, mean_abs, cos


def run_asm_debug(binary: str, data: str, out_dir: Path, cfg: dict, threads: int):
    cmd = [
        binary,
        "--data", data,
        "--layers", str(cfg["layers"]),
        "--dim", str(cfg["dim"]),
        "--heads", str(cfg["heads"]),
        "--hidden", str(cfg["hidden"]),
        "--batch", str(cfg["batch"]),
        "--seq", str(cfg["seq"]),
        "--steps", str(cfg["steps"]),
        "--lr", str(cfg["lr"]),
        "--wd", str(cfg["wd"]),
        "--grad-clip", str(cfg["grad_clip"]),
        "--warmup", str(cfg["warmup"]),
        "--seed", str(cfg["seed"]),
        "-t", str(threads),
        "--eval-every", "0",
        "--sample-every", "0",
        "--plain",
        "--dump-debug", str(out_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return proc


def load_debug_dir(d: Path):
    meta_raw = (d / "meta.bin").read_bytes()
    assert len(meta_raw) == 64, f"Bad meta.bin size: {len(meta_raw)}"
    (
        vocab, dim, n_layers, n_heads, hidden_dim, seq_len, batch_size, steps,
        n_params, lr, wd, grad_clip, rms_eps, rope_theta, warmup_steps
    ) = struct.unpack("<IIIIIIIIQfffffI", meta_raw)
    meta = dict(
        vocab=vocab, dim=dim, n_layers=n_layers, n_heads=n_heads,
        hidden_dim=hidden_dim, seq_len=seq_len, batch_size=batch_size, steps=steps,
        n_params=n_params, lr=lr, wd=wd, grad_clip=grad_clip,
        rms_eps=rms_eps, rope_theta=rope_theta, warmup_steps=warmup_steps,
    )
    init_w = np.fromfile(d / "init_weights.bin", dtype="<f4")
    bx = np.fromfile(d / "batch_x.bin", dtype="<u4").reshape(batch_size, seq_len)
    by = np.fromfile(d / "batch_y.bin", dtype="<u4").reshape(batch_size, seq_len)
    g0 = np.fromfile(d / "step0_grads.bin", dtype="<f4")
    losses = np.fromfile(d / "losses.bin", dtype="<f4")
    lrs = np.fromfile(d / "lrs.bin", dtype="<f4")
    final_w = np.fromfile(d / "final_weights.bin", dtype="<f4")
    assert init_w.size == n_params
    assert g0.size == n_params
    assert final_w.size == n_params
    assert losses.size == steps
    assert lrs.size == steps
    return meta, init_w, bx, by, g0, losses, lrs, final_w


def main():
    ap = argparse.ArgumentParser(description="Verify Phase 3 pure-asm GPT training vs PyTorch CPU")
    ap.add_argument("--bin", default="build/asm-gpt-train")
    ap.add_argument("--data", default="data/tinyshakespeare.txt")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq", type=int, default=16)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    assert not torch.cuda.is_available(), "CUDA must be disabled (CPU-only verification!)"
    torch.set_num_threads(4)

    cfg = vars(args)
    tmp_root = Path(tempfile.mkdtemp(prefix="asm_gpt_verify_"))
    try:
        print("=" * 78)
        print("Phase 3 Pure x86-64 Mini-GPT Training Verification (PyTorch CPU Reference)")
        print(
            f"Config: L={args.layers} D={args.dim} H={args.heads} FF={args.hidden} "
            f"B={args.batch} T={args.seq} steps={args.steps} lr={args.lr} wd={args.wd} "
            f"clip={args.grad_clip} warmup={args.warmup}"
        )
        print("=" * 78)

        # Run asm binary at -t 1, -t 8, -t 16
        dumps = {}
        for thr in (1, 8, 16):
            out_dir = tmp_root / f"t{thr}"
            out_dir.mkdir(parents=True, exist_ok=True)
            run_asm_debug(args.bin, args.data, out_dir, cfg, threads=thr)
            dumps[thr] = load_debug_dir(out_dir)

        meta, init_w, bx, by, asm_g0, asm_losses, asm_lrs, asm_final_w = dumps[1]
        print(f"Loaded meta: V={meta['vocab']} n_params={meta['n_params']:,}")

        # Build PyTorch CPU reference with exact same initial weights & batch
        model = MiniGPT(
            vocab=meta["vocab"],
            dim=meta["dim"],
            n_layers=meta["n_layers"],
            n_heads=meta["n_heads"],
            hidden_dim=meta["hidden_dim"],
            seq_len=meta["seq_len"],
            eps=meta["rms_eps"],
            rope_theta=meta["rope_theta"],
        )
        model.load_flat_weights(init_w)

        decay_params = [p for _, p in model.named_flat_params() if p.ndim >= 2]
        nodecay_params = [p for _, p in model.named_flat_params() if p.ndim < 2]
        opt = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": meta["wd"]},
                {"params": nodecay_params, "weight_decay": 0.0},
            ],
            lr=meta["lr"],
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        x_t = torch.from_numpy(bx.astype(np.int64))
        y_t = torch.from_numpy(by.astype(np.int64))

        pt_losses = []
        pt_lrs = []
        pt_g0_tensors = None

        for step in range(meta["steps"]):
            lr_s = compute_lr_ref(step, meta["steps"], meta["warmup_steps"], meta["lr"])
            for pg in opt.param_groups:
                pg["lr"] = lr_s
            pt_lrs.append(lr_s)

            opt.zero_grad(set_to_none=True)
            _, loss = model(x_t, y_t)
            pt_losses.append(float(loss.item()))
            loss.backward()

            if step == 0:
                pt_g0_tensors = [
                    (name, p.grad.detach().cpu().numpy().copy())
                    for name, p in model.named_flat_params()
                ]

            if meta["grad_clip"] > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), meta["grad_clip"])
            opt.step()

        pt_losses = np.array(pt_losses, dtype=np.float32)
        pt_lrs = np.array(pt_lrs, dtype=np.float32)
        pt_final_w = model.dump_flat_weights()

        # --------------------------------------------------------------------
        # 1. Forward Loss Check (Step 0)
        # --------------------------------------------------------------------
        loss0_diff = abs(float(asm_losses[0]) - float(pt_losses[0]))
        print("\n[1] Forward Cross-Entropy Loss Check (Step 0):")
        print(f"    ASM loss0     : {asm_losses[0]:.8f}")
        print(f"    PyTorch loss0 : {pt_losses[0]:.8f}")
        print(f"    abs_diff      : {loss0_diff:.3e}  (gate < 1.0e-5)")
        ok_fwd = loss0_diff < 1e-5

        # --------------------------------------------------------------------
        # 2. Per-Tensor Analytical Gradient Check (Step 0, pre-clip)
        # --------------------------------------------------------------------
        print("\n[2] Per-Tensor Analytical Gradient Check (Step 0, pre-clip):")
        print(f"    {'Tensor':<14} {'Shape':<12} {'max_abs_diff':>13} {'mean_abs_diff':>14} {'cosine_sim':>14}  Status")
        print("    " + "-" * 72)

        off = 0
        worst_grad_abs = 0.0
        min_grad_cos = 1.0
        ok_grads = True
        for name, pt_g in pt_g0_tensors:
            n = pt_g.size
            asm_g = asm_g0[off : off + n].reshape(pt_g.shape)
            off += n
            max_d, mean_d, cos = tensor_metrics(asm_g, pt_g)
            worst_grad_abs = max(worst_grad_abs, max_d)
            min_grad_cos = min(min_grad_cos, cos)
            passed = (cos > 0.99999) and (max_d < 1e-4)
            ok_grads = ok_grads and passed
            st = "PASS" if passed else "FAIL"
            print(
                f"    {name:<14} {str(tuple(pt_g.shape)):<12} "
                f"{max_d:13.3e} {mean_d:14.3e} {cos:14.10f}  {st}"
            )
        print("    " + "-" * 72)
        print(
            f"    Summary: worst max_abs_diff={worst_grad_abs:.3e} (<1e-4), "
            f"min cosine_sim={min_grad_cos:.10f} (>0.99999) -> {'PASS' if ok_grads else 'FAIL'}"
        )

        # --------------------------------------------------------------------
        # 3. 10-Step Consecutive AdamW + Warmup/Cosine LR Trajectory Check
        # --------------------------------------------------------------------
        print(f"\n[3] {meta['steps']}-Step Consecutive AdamW Trajectory Check:")
        print(f"    {'Step':>4} {'LR (ASM)':>11} {'LR (PT)':>11} {'Loss (ASM)':>12} {'Loss (PT)':>12} {'abs_diff':>11}")
        max_loss_diff = 0.0
        max_lr_diff = 0.0
        for s in range(meta["steps"]):
            ld = abs(float(asm_losses[s]) - float(pt_losses[s]))
            rd = abs(float(asm_lrs[s]) - float(pt_lrs[s]))
            max_loss_diff = max(max_loss_diff, ld)
            max_lr_diff = max(max_lr_diff, rd)
            print(
                f"    {s:4d} {asm_lrs[s]:11.7f} {pt_lrs[s]:11.7f} "
                f"{asm_losses[s]:12.7f} {pt_losses[s]:12.7f} {ld:11.3e}"
            )

        w_max_d, w_mean_d, w_cos = tensor_metrics(asm_final_w, pt_final_w)
        ok_adam = (max_loss_diff < 1e-4) and (w_max_d < 1e-4)
        print(f"    Max LR diff            : {max_lr_diff:.3e}")
        print(f"    Max Loss diff [0..{meta['steps']-1}]   : {max_loss_diff:.3e}  (gate < 1.0e-4)")
        print(f"    Final Weights max_diff : {w_max_d:.3e}  (gate < 1.0e-4)")
        print(f"    Final Weights mean_diff: {w_mean_d:.3e}")
        print(f"    Final Weights cos_sim  : {w_cos:.10f}")
        print(f"    Trajectory status      : {'PASS' if ok_adam else 'FAIL'}")

        # --------------------------------------------------------------------
        # 4. Multi-Thread Determinism Check (-t 1 vs -t 8 vs -t 16)
        # --------------------------------------------------------------------
        print("\n[4] Multi-Thread Determinism Check (-t 1 vs -t 8 vs -t 16):")
        ok_det = True
        for thr in (8, 16):
            _, _, _, _, g_t, l_t, _, fw_t = dumps[thr]
            dl = float(np.max(np.abs(l_t - asm_losses)))
            dg = float(np.max(np.abs(g_t - asm_g0)))
            dw = float(np.max(np.abs(fw_t - asm_final_w)))
            exact = (dl == 0.0 and dg == 0.0 and dw == 0.0)
            passed = (dl < 1e-5 and dg < 1e-5 and dw < 1e-5)
            ok_det = ok_det and passed
            tag = "BIT-EXACT" if exact else ("PASS" if passed else "FAIL")
            print(
                f"    -t 1 vs -t {thr:2d}: loss_diff={dl:.3e}, grad_diff={dg:.3e}, "
                f"final_w_diff={dw:.3e}  [{tag}]"
            )

        all_ok = ok_fwd and ok_grads and ok_adam and ok_det
        print("\n" + "=" * 78)
        print(f"OVERALL VERIFICATION: {'ALL GATES PASSED' if all_ok else 'FAILED'}")
        print("=" * 78)
        sys.exit(0 if all_ok else 1)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
