"""Diagnose the circuit free-run: teacher-forced check + divergence hunt.

    ~/ComfyUI/.venv/bin/python -m chipformer.rv.circdiag agent.elf
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from chipformer.rv import core, elf, ref  # noqa: E402
from chipformer.rv import fastexec  # noqa: E402
from chipformer.rv import spec as S  # noqa: E402


def main() -> int:
    elf_path = sys.argv[1] if len(sys.argv) > 1 else "agent.elf"
    c = core.build()
    prog = elf.program(elf_path)
    core.snapshot(c, ref.Machine.from_program(prog))
    m = ref.Machine.from_program(prog)
    ref_toks = m.run(601)
    print(f"[diag: ref trace {len(ref_toks)} tokens; first 8: "
          f"{[S.name(t) for t in ref_toks[:8]]}]", flush=True)

    t0 = time.time()
    try:
        n = fastexec.check(c, ref_toks[:300], "cpu")
        print(f"[diag: fastexec.check PASS {n} tokens "
              f"({time.time() - t0:.1f}s)]", flush=True)
    except AssertionError as e:
        print(f"[diag: fastexec.check FAIL: {e}]", flush=True)

    ex = fastexec.Executor(c, "cpu")
    ex.load_init()
    tok = ref_toks[0]
    div = None
    for i in range(600):
        x = ex.embed([tok])
        resid = ex.step(x)[0].double().cpu().numpy()
        nxt = core.predictions(c, resid[None, :])[0]
        if nxt != ref_toks[i + 1]:
            div = (i + 1, ref_toks[i + 1], nxt)
            break
        tok = nxt
    if div is None:
        print("[diag: free-run matches ref for 600 tokens]", flush=True)
    else:
        i, want, got = div
        print(f"[diag: free-run DIVERGES at token {i}: ref "
              f"{S.name(want)} vs gen {S.name(got)}]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
