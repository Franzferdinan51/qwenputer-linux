"""Machine snapshots for the Linux machine: boot once, start warm.

Booting to a logged-in, tools-installed shell costs ~60M steps on the
reference machine — most of every test's wall time, and all of the
live terminal's startup wait. A snapshot freezes the whole machine
(RAM, registers, CSRs, device state) right after the install, at the
ready shell prompt; restoring it starts there in milliseconds.

    python3 -m chipformer.rv.snapshot Image shell.snap   # prime + save
    python3 -m chipformer.rv.agenttest Image --snapshot shell.snap

The primed shell has /tmp/ask, /tmp/agent and /tmp/hermes-agent
installed and has printed ASK_READY. The console output buffer is
cleared in the snapshot so drivers see a fresh stream; the input
queue is empty; the step counter is kept (the guest's clocks stay
monotonic).
"""

import argparse
import pickle
import time
from pathlib import Path

from chipformer.rv.linux import LinuxMachine

HERE = Path(__file__).parent
ASK = (HERE / "env" / "linux" / "ask.sh").read_text()
AGENT = (HERE / "env" / "linux" / "agent.sh").read_text()

_FIELDS = ("ram", "x", "pc", "priv", "inp", "halted", "exit_reason",
           "steps", "csr", "mtime", "mtimecmp", "msip", "reservation",
           "uart_regs", "qwen_prompt", "qwen_response", "qwen_line",
           "csr_fallback", "ram_size", "ticks_per_step")


def save(m: LinuxMachine, path) -> None:
    state = {f: getattr(m, f) for f in _FIELDS}
    state["ram"] = bytes(m.ram)
    state["out_cleared"] = True
    with open(path, "wb") as fh:
        pickle.dump(state, fh, protocol=4)


def load(path) -> LinuxMachine:
    with open(path, "rb") as fh:
        state = pickle.load(fh)
    m = LinuxMachine(ram_size=state["ram_size"],
                     ticks_per_step=state["ticks_per_step"])
    for f in _FIELDS:
        setattr(m, f, state[f])
    m.ram = bytearray(state["ram"])
    m.out = bytearray()  # fresh stream for the restoring driver
    return m


def prime(image: bytes, dtb: bytes, log=print) -> LinuxMachine:
    """Boot, log in, install ask + hermes-agent; return at ASK_READY."""
    m = LinuxMachine()
    m.load_image(image)
    m.boot(dtb)
    script = (
        "cat > /tmp/ask <<'QWEOF'\n" + ASK + "QWEOF\n"
        "cat > /tmp/agent <<'QWEOF'\n" + AGENT + "QWEOF\n"
        "cp /tmp/agent /tmp/hermes-agent\n"
        "chmod +x /tmp/ask /tmp/agent /tmp/hermes-agent\n"
        "echo ASK_READY try ask or hermes-agent\n"
    ).encode()
    chunks = [script[i:i + 256] for i in range(0, len(script), 256)]
    phase, phase_slice, slices = 0, 0, 0
    fed = set()
    t0 = time.time()
    while not m.halted:
        m.run(250_000)
        slices += 1
        out = bytes(m.out)
        if phase == 0:
            if out.rstrip().endswith(b"buildroot login:"):
                m.feed_input(b"root\n")
                phase, phase_slice = 1, slices
        elif phase == 1:
            rel = slices - phase_slice
            if rel >= 2 and "login2" not in fed:
                m.feed_input(b"root\n")
                fed.add("login2")
            if rel >= 6 and "canary1" not in fed:
                m.feed_input(b"echo CANARY_$((40+2))\n")
                fed.add("canary1")
            if rel >= 16 and "canary2" not in fed:
                m.feed_input(b"echo CANARY_$((40+2))\n")
                fed.add("canary2")
            if b"CANARY_42" in out:
                phase = 2
        elif phase == 2:
            if chunks:
                m.feed_input(chunks.pop(0))
            elif b"ASK_READY" in out:
                log(f"[snapshot: primed at {m.steps:,} steps, "
                    f"{time.time() - t0:.1f}s]")
                return m
        if slices > 1200:
            raise RuntimeError("prime did not reach ASK_READY")
    raise RuntimeError(f"machine halted while priming: {m.exit_reason}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("out")
    ap.add_argument("--dtb", default=str(HERE / "env" / "linux" / "sixtyfourmb.dtb"))
    args = ap.parse_args()
    m = prime(Path(args.image).read_bytes(), Path(args.dtb).read_bytes())
    save(m, args.out)
    print(f"[snapshot: saved {args.out} ({Path(args.out).stat().st_size:,} B)]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
