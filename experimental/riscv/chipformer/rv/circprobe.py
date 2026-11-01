"""Probe: can the compiled circuit be driven with a live mailbox?

Builds the stock circuit, snapshots the bare-agent ELF into it, creates
the fastexec Executor on CPU, reverse-maps the RAM tables' state rows
empirically (seed rows -> state rows), then free-runs a few thousand
tokens and watches the mailbox flag. Run on the Mac:

    ~/ComfyUI/.venv/bin/python -m chipformer.rv.circprobe agent.elf
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

from chipformer.rv import core, elf, ref
from chipformer.rv import fastexec
from chipformer.rv import spec as S_
from chipformer.q35.ir import A_BIT, A_REF, HD, REF_COLS


def row_of(word: int) -> np.ndarray:
    row = np.zeros(HD)
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


def main() -> int:
    elf_path = sys.argv[1] if len(sys.argv) > 1 else "agent.elf"
    syms = json.loads(Path(elf_path + ".sym.json").read_text())
    t0 = time.time()
    c = core.build()
    print(f"[probe: graph built {time.time() - t0:.1f}s, {len(c.tables)} tables]", flush=True)
    prog = elf.program(elf_path)
    m = ref.Machine.from_program(prog)
    core.snapshot(c, m)
    t0 = time.time()
    ex = fastexec.Executor(c, "cpu")
    ex.load_init()
    print(f"[probe: executor ready {time.time() - t0:.1f}s]", flush=True)

    # reverse-map: for each ram table + sub, find (step_idx, j) whose
    # state rows equal the snapshot's init rows at the same keys
    S = [(i, st) for i, st in enumerate(ex.steps) if isinstance(st, dict)]
    ram_map = {}
    for h in range(32):
        init = c.tables[f"ram{h}"].init
        for x in range(2):
            rows = init[x] if x < len(init) else {}
            probe = next(((k, v) for k, v in rows.items()
                          if np.abs(v[2:34]).sum() > 0), None)
            found = None
            for si, st in S:
                buf = st["S"][0].numpy()
                for j in range(buf.shape[0]):
                    if probe is not None:
                        k, v = probe
                        if np.array_equal(buf[j, k], v.astype(np.float32)):
                            found = (si, j)
                            break
                    else:
                        found = (si, None)  # empty table: resolve lazily
                if found and found[1] is not None:
                    break
            ram_map[(h, x)] = found
    print(f"[probe: ram map sample: ram0={ram_map.get((0, 0))} "
          f"ram1={ram_map.get((1, 0))}]", flush=True)

    def read_word(addr: int) -> int:
        wi = addr >> 2
        h = (wi & 15) | ((wi >> 12) << 4)
        x = (wi >> 4) & 1
        key = (wi >> 5) & 127
        si, j = ram_map[(h, x)]
        if j is None:
            return 0
        row = ex.steps[si]["S"][0, j, key].numpy()
        return word_of(row)

    print(f"[probe: mb_req at start = {read_word(syms['mb_req'])}]", flush=True)

    m2 = ref.Machine.from_program(prog)
    tok = m2.run(2)[0]
    out = bytearray()
    t0 = time.time()
    n = 3000
    req_seen = None
    for i in range(n):
        x = ex.embed([tok])
        resid = ex.step(x)[0].double().cpu().numpy()
        tok = core.predictions(c, resid[None, :])[0]
        fam, p = S_.family(tok) if tok < S_.VOCAB else ("?", 0)
        if fam == "BYTE" and p < 256:
            out.append(p)
        if i % 250 == 0 or (req_seen is None and i > 100):
            if read_word(syms["mb_req"]) == 1:
                req_seen = i
                break
    el = time.time() - t0
    print(f"[probe: {i + 1} tokens in {el:.1f}s = {(i + 1) / el:.0f} tok/s; "
          f"mb_req seen at token {req_seen}]", flush=True)
    print(f"[probe: guest output so far: {bytes(out[:400])!r}]", flush=True)
    if req_seen is not None:
        ln = read_word(syms["mb_outlen"])
        data = bytearray()
        base = syms["mb_out"]
        for k in range(ln):
            w = read_word(base + 4 * (k // 4))
            data.append((w >> (8 * (k % 4))) & 0xFF)
        print(f"[probe: mb_out ({ln} B): {bytes(data[:200])!r}]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
