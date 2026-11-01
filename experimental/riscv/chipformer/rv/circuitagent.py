"""Run the bare-metal agent harness ON THE COMPILED CIRCUIT.

Same guest (env/progs/agent.c), same mailbox contract, same backends as
bareagent.py - but the CPU executing the guest is the circuit itself:
the compiled graph (the layers that live inside the Qwenputer
checkpoint), stepped by the fastexec Executor. The host's only hands
are the ones the circuit gives it: seed the state (core.snapshot),
read/write the RAM tables' state rows (that IS the guest's memory -
a write here is the same row-scatter the circuit's own stores do),
and collect the BYTE tokens the guest prints.

    ~/ComfyUI/.venv/bin/python -m chipformer.rv.circuitagent agent.elf \
        [--backend echo|builtin|lmstudio] [--device cpu] [--log out.txt]

Verdict mirrors bareagent: PASS when the guest reaches its answer and
the AGENT_ANSWER line carries the stored word and the exact product.
Halt note: the reference machine stops at the exit ecall; the circuit
keeps generating, so the run stops when the answer is out and the
guest goes quiet (or a stall/cap trips) - the transcript is the proof.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from chipformer.q35.ir import A_BIT, A_REF, HD, REF_COLS
from chipformer.rv import core, elf, ref
from chipformer.rv import fastexec
from chipformer.rv import spec as S
from chipformer.rv.qwenbackend import make_backend

PRODUCT = str(123456 * 654321)


def row_of(word: int) -> np.ndarray:
    row = np.zeros(HD, dtype=np.float32)
    row[list(REF_COLS)] = A_REF
    for i in range(32):
        row[2 + i] = A_BIT * (word >> i & 1)
    return row


def word_of(row: np.ndarray) -> int:
    w = 0
    for i in range(32):
        if row[2 + i] > A_BIT / 2:
            w |= 1 << i
    return w


class CircuitRAM:
    """Read/write guest RAM in the executor's live state tensors."""

    def __init__(self, ex):
        self.ex = ex
        self.map = {}
        for si, st in enumerate(ex.steps):
            if not isinstance(st, dict):
                continue
            j = 0
            for n in st["heads"]:
                for x in range(len(n.vheads())):
                    if n.name.startswith("ram"):
                        self.map[(int(n.name[3:]), x)] = (si, j)
                    j += 1

    def _cell(self, addr: int):
        wi = addr >> 2
        h = (wi & 15) | ((wi >> 12) << 4)
        x = (wi >> 4) & 1
        key = (wi >> 5) & 127
        si, j = self.map[(h, x)]
        return self.ex.steps[si]["S"], j, key

    def read_word(self, addr: int) -> int:
        S_, j, key = self._cell(addr)
        return word_of(S_[0, j, key].numpy())

    def write_word(self, addr: int, word: int) -> None:
        S_, j, key = self._cell(addr)
        S_[0, j, key] = torch.from_numpy(row_of(word & 0xFFFFFFFF))

    def read_byte(self, addr: int) -> int:
        return (self.read_word(addr) >> (8 * (addr & 3))) & 0xFF

    def write_byte(self, addr: int, b: int) -> None:
        w = self.read_word(addr)
        sh = 8 * (addr & 3)
        self.write_word(addr, (w & ~(0xFF << sh)) | (b << sh))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("elf")
    ap.add_argument("--backend", choices=("echo", "builtin", "lmstudio"),
                    default="builtin")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--log", default=None)
    ap.add_argument("--max-tokens", type=int, default=600_000)
    args = ap.parse_args()

    syms = json.loads(Path(args.elf + ".sym.json").read_text())
    backend = make_backend(args.backend, None)
    model = getattr(backend, "model", args.backend)
    print(f"[circuitagent: backend={args.backend} model={model} "
          f"device={args.device}]", flush=True)

    c = core.build()
    prog = elf.program(args.elf)
    core.snapshot(c, ref.Machine.from_program(prog))
    ex = fastexec.Executor(c, args.device)
    ex.load_init()
    ram = CircuitRAM(ex)
    want = (S.RAM_WORDS >> 8) * 2
    assert len(ram.map) == want, f"ram map incomplete: {len(ram.map)}"
    print(f"[circuitagent: circuit live, ram map {len(ram.map)} cells]",
          flush=True)

    tok = ref.Machine.from_program(prog).run(2)[0]
    out = bytearray()
    logf = open(args.log, "wb") if args.log else None
    prompts = []
    t0 = time.time()
    toks = 0
    last_byte_at = 0
    same_tok_run = 0
    prev_tok = -1
    answered_at = None
    while toks < args.max_tokens:
        x = ex.embed([tok])
        resid = ex.step(x)[0].double().cpu().numpy()
        tok = core.predictions(c, resid[None, :])[0]
        toks += 1
        same_tok_run = same_tok_run + 1 if tok == prev_tok else 0
        prev_tok = tok
        fam, p = S.family(tok) if tok < S.VOCAB else ("?", 0)
        if fam == "BYTE" and p < 256:
            out.append(p)
            last_byte_at = toks
            if logf:
                logf.write(bytes([p]))
                logf.flush()
        if toks % 256 == 0 and ram.read_word(syms["mb_req"]):
            cap = int(syms.get("mb_size", 896)) - 16
            n = min(ram.read_word(syms["mb_outlen"]), cap)
            prompt = bytes(ram.read_byte(syms["mb_out"] + i)
                           for i in range(n)).decode("utf-8", "replace")
            prompts.append(prompt)
            print(f"\n[circuitagent: prompt {len(prompts)} ({n} chars) "
                  f"at token {toks}]", flush=True)
            answer = backend(prompt)
            data = " ".join(str(answer).split())[:cap].encode(
                "utf-8", "replace")
            for i, b in enumerate(data):
                ram.write_byte(syms["mb_in"] + i, b)
            ram.write_word(syms["mb_inlen"], len(data))
            ram.write_word(syms["mb_req"], 0)
            print(f"[circuitagent: answer {len(data)} chars written to "
                  f"the circuit's memory]", flush=True)
        if answered_at is None and b"AGENT_ANSWER:" in out:
            answered_at = toks
        if answered_at is not None and toks - last_byte_at > 2000:
            print("[circuitagent: answer out, guest quiet - stopping]",
                  flush=True)
            break
        if same_tok_run > 20000:
            print("[circuitagent: 20k identical tokens - stall, stopping]",
                  flush=True)
            break

    text = bytes(out).decode("utf-8", "replace")
    print(text)
    answers = [l for l in text.splitlines() if l.startswith("AGENT_ANSWER:")]
    final = answers[-1] if answers else ""
    ok = "banana" in final and PRODUCT in final
    print(f"=== tokens={toks:,} wall={time.time() - t0:.1f}s "
          f"backend={args.backend} prompts={len(prompts)} ===")
    print("CIRCUIT AGENT:", "PASS" if ok else
          f"FAIL (final={final[:120]!r})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
