"""Machine snapshots for the Linux machine: boot once, start warm.

Booting to a logged-in, tools-installed shell costs ~60M steps on the
reference machine — most of every test's wall time, and all of the
live terminal's startup wait. A snapshot freezes the whole machine
(RAM, registers, CSRs, device state) right after the install, at the
ready shell prompt; restoring it starts there in milliseconds.

    python3 -m chipformer.rv.snapshot Image shell.snap   # prime + save
    python3 -m chipformer.rv.agenttest Image --snapshot shell.snap

The primed shell has the full current toolset installed - /tmp/ask,
/tmp/agent, /tmp/hermes-agent and /tmp/agentshell, byte-verified by
md5sum in the guest - with /root/SOUL.md seeded, and sits at the
agentshell prompt (the same state the live terminal boots into).
The console output buffer is cleared in the snapshot so drivers see
a fresh stream; the input queue is empty; the step counter is kept
(the guest's clocks stay monotonic).
"""

import argparse
import hashlib
import pickle
import time
from pathlib import Path

from chipformer.rv.linux import LinuxMachine

HERE = Path(__file__).parent
ASK = (HERE / "env" / "linux" / "ask.sh").read_text()
AGENT = (HERE / "env" / "linux" / "agent.sh").read_text()
ASHELL = (HERE / "env" / "linux" / "agentshell.sh").read_text()
# The guest's soul lives in env/linux/SOUL.md (single source):
# the Hermes Agent SOUL.md (Nous Research, nousresearch/
# hermes-agent, hermes_cli/default_soul.py) re-homed - its
# behavioral text kept, the identity pointed at this machine.
# Its first ~300 chars carry identity + directness on purpose:
# agent.sh folds only head -c 300 of SOUL.md into wake-up prompts.
SOUL = (HERE / "env" / "linux" / "SOUL.md").read_text()
if not SOUL.endswith("\n"):
    SOUL += "\n"
INSTALL = (
    "cat > /tmp/ask <<'QWEOF'\n" + ASK + "QWEOF\n"
    "cat > /tmp/agent <<'QWEOF'\n" + AGENT + "QWEOF\n"
    "cat > /tmp/agentshell <<'QWEOF'\n" + ASHELL + "QWEOF\n"
    "cp /tmp/agent /tmp/hermes-agent\n"
    "cp /tmp/agentshell /root/agentshell\n"
    "chmod +x /tmp/ask /tmp/agent /tmp/hermes-agent /tmp/agentshell\n"
    "mkdir -p /root/nb /root/skills /root/apps\n"
    "[ -f /root/SOUL.md ] || cat > /root/SOUL.md <<'QWEOF'\n" + SOUL + "QWEOF\n"
)
VERIFY = b"md5sum /tmp/ask /tmp/agent /tmp/agentshell; echo MD5\"CHK\"\n"
HASHES = [hashlib.md5(x.encode()).hexdigest().encode()
          for x in (ASK, AGENT, ASHELL)]


def units_of(text: str):
    """Line-paced feed units (see agenttest.units_of): one unit in
    flight, released by its own echo - blind chunk feeding tears
    heredocs on this guest tty."""
    units = []
    for line in text.splitlines(keepends=True):
        b = line.encode()
        units += [b[i:i + 256] for i in range(0, len(b), 256)] or [b""]
    return units

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
    m = LinuxMachine(ram_size=state.get("ram_size", 64 * 1024 * 1024),
                     ticks_per_step=state.get("ticks_per_step", 1))
    for f in _FIELDS:
        if f in state:  # snapshots from older builds may lack new fields
            setattr(m, f, state[f])
    m.ram = bytearray(state["ram"])
    m.out = bytearray()  # fresh stream for the restoring driver
    return m


def prime(image: bytes, dtb: bytes, log=print) -> LinuxMachine:
    """Boot, log in, install the toolset (md5-verified, retrying a
    torn install whole), seed SOUL, land in the agentshell; return
    with the supervisor's prompt up."""
    m = LinuxMachine()
    m.load_image(image)
    m.boot(dtb)
    chunks = units_of(INSTALL)
    phase, phase_slice, slices = 0, 0, 0
    last_feed_slice = 0
    unit = None
    unit_mark = 0
    verify_mark = 0
    retries = 0
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
            # line-paced feeding, one unit in flight (see agenttest)
            if unit is not None:
                if len(out) - unit_mark >= len(unit) - 4 \
                        or slices - last_feed_slice >= 200:
                    unit = None
            elif chunks:
                unit = chunks.pop(0)
                m.feed_input(unit)
                unit_mark = len(out)
                last_feed_slice = slices
            else:
                verify_mark = len(out)
                m.feed_input(VERIFY)
                phase = 25
        elif phase == 25:
            # the guest's own md5sums decide; a torn install retries
            tail = out[verify_mark:]
            if b"MD5CHK" in tail:
                if all(h in tail for h in HASHES):
                    log("[snapshot: install verified byte-exact "
                        "in the guest]")
                    m.feed_input(b"echo ASK_READY the agentshell "
                                 b"takes the console now\n"
                                 b"/tmp/agentshell\n")
                    phase = 3
                elif retries < 3:
                    retries += 1
                    log(f"[snapshot: install torn (retry {retries})]")
                    chunks = units_of(INSTALL)
                    phase = 2
                else:
                    raise RuntimeError("prime: install torn 3 times")
        elif phase == 3:
            if b"agent> " in out[verify_mark:]:
                log(f"[snapshot: primed at {m.steps:,} steps, "
                    f"{time.time() - t0:.1f}s]")
                return m
        if slices > 6000:
            raise RuntimeError("prime did not reach the agentshell")
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
