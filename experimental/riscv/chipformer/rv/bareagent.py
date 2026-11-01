"""Run the bare-metal agent harness (env/progs/agent.c) on the reference
machine and sit in the model's seat: service the RAM mailbox.

    python3 -m chipformer.rv.bareagent agent.elf [--backend echo|lmstudio|builtin]
        [--log bare-lm.out]

The guest writes a prompt into mb_out and sets mb_req; this runner reads
it, asks the backend, writes the (flattened, one-line) answer into
mb_in, and clears mb_req. In the endgame the model's own language
layers hold this seat inside one checkpoint; here an LM Studio model
holds it through the same callable the /dev/qwen backends use, so the
guest binary does not change when the seat does.

Verdict: PASS when the guest halts and its AGENT_ANSWER line carries
both the stored word and the exact product (computed here, in Python -
not taken from the model or the guest).
"""

import argparse
import json
import time
from pathlib import Path

from chipformer.rv import elf, ref
from chipformer.rv.qwenbackend import make_backend

PRODUCT = str(123456 * 654321)


def rd(m, addr):
    return (m.ram[addr >> 2] >> (8 * (addr & 3))) & 0xFF


def wr(m, addr, b):
    i, sh = addr >> 2, 8 * (addr & 3)
    m.ram[i] = (m.ram[i] & ~(0xFF << sh) | (b << sh)) & 0xFFFFFFFF


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("elf")
    ap.add_argument("--backend", choices=("echo", "lmstudio", "builtin"),
                    default="lmstudio")
    ap.add_argument("--log", default=None)
    ap.add_argument("--max-tokens", type=int, default=3_000_000)
    args = ap.parse_args()

    syms = json.loads(Path(args.elf + ".sym.json").read_text())
    backend = make_backend(args.backend, None)
    m = ref.Machine.from_program(elf.program(args.elf))
    logf = open(args.log, "wb") if args.log else None
    printed = 0
    toks = 0
    prompts = []
    t0 = time.time()
    while not m.halted and toks < args.max_tokens:
        toks += len(m.run(4000))
        if m.ram[syms["mb_req"] >> 2]:
            cap = int(syms.get("mb_size", 896)) - 16
            n = min(m.ram[syms["mb_outlen"] >> 2], cap)
            prompt = bytes(rd(m, syms["mb_out"] + i) for i in range(n)).decode(
                "utf-8", "replace")
            prompts.append(prompt)
            answer = backend(prompt)
            data = " ".join(str(answer).split())[:cap].encode("utf-8", "replace")
            for i, b in enumerate(data):
                wr(m, syms["mb_in"] + i, b)
            m.ram[syms["mb_inlen"] >> 2] = len(data)
            m.ram[syms["mb_req"] >> 2] = 0
        if len(m.out) > printed:
            chunk = bytes(m.out[printed:])
            printed = len(m.out)
            if logf:
                logf.write(chunk)
                logf.flush()
            print(chunk.decode("utf-8", "replace"), end="", flush=True)

    out = bytes(m.out).decode("utf-8", "replace")
    answers = [l for l in out.splitlines() if l.startswith("AGENT_ANSWER:")]
    final = answers[-1] if answers else ""
    ok = bool(m.halted and "banana" in final and PRODUCT in final)
    print(f"\n=== tokens={toks:,} wall={time.time() - t0:.1f}s "
          f"backend={args.backend} prompts={len(prompts)} ===")
    print("BARE AGENT:", "PASS" if ok else
          f"FAIL (halted={m.halted} final={final[:120]!r})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
