"""Automated agent-harness test: boot, log in, install ask + agent
(verified byte-exact by md5sum in the guest - the UART feed can tear
lines out of a big heredoc, so the install is retried until the
guest's own hashes match), give the agent a two-step task, and verify
its work from the ground truth.

    python3 -m chipformer.rv.agenttest Image [--backend lmstudio]
        [--log agent-lm.out]

The task: "write the word banana into /tmp/hello.txt then read it back
and tell me what it says". PASS requires BOTH: an AGENT_ANSWER line from
the harness, and - checked afterwards with od, not the model's word -
/tmp/hello.txt really containing b a n a n a.
"""

import argparse
import hashlib
import time
from pathlib import Path

from chipformer.rv.linux import LinuxMachine
from chipformer.rv.qwenbackend import make_backend

HERE = Path(__file__).parent
ASK = (HERE / "env" / "linux" / "ask.sh").read_text()
AGENT = (HERE / "env" / "linux" / "agent.sh").read_text()
TASK = ("write the word banana into /tmp/hello.txt then read it back "
        "and tell me what it says")
INSTALL = (
    "cat > /tmp/ask <<'QWEOF'\n" + ASK + "QWEOF\n"
    "cat > /tmp/agent <<'QWEOF'\n" + AGENT + "QWEOF\n"
    "cp /tmp/agent /tmp/hermes-agent\n"
    "chmod +x /tmp/ask /tmp/agent /tmp/hermes-agent\n"
)
VERIFY = b"md5sum /tmp/ask /tmp/agent; echo MD5\"CHK\"\n"
HASHES = [hashlib.md5(ASK.encode()).hexdigest().encode(),
          hashlib.md5(AGENT.encode()).hexdigest().encode()]


def units_of(text: str):
    """Feed units for the install: one line each (long lines split at
    256 B). The guest tty echoes bytes on RECEIPT, not on shell
    consumption, and its canonical read buffer drops bytes past
    ~4 KB unread - so the drivers feed one unit, then wait for that
    unit's own echo before sending more (see phase 2)."""
    units = []
    for line in text.splitlines(keepends=True):
        b = line.encode()
        units += [b[i:i + 256] for i in range(0, len(b), 256)] or [b""]
    return units


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--dtb", default=str(HERE / "env" / "linux" / "sixtyfourmb.dtb"))
    ap.add_argument("--backend", choices=("echo", "lmstudio", "builtin"), default="builtin")
    ap.add_argument("--log", default=None)
    ap.add_argument("--max-steps", type=int, default=1_200_000_000)
    ap.add_argument("--slice", type=int, default=250_000)
    ap.add_argument("--snapshot", default=None,
                    help="start warm from a snapshot.py shell (skips boot+install)")
    ap.add_argument("--fast", action="store_true",
                    help="run on the fastlinux.c native core (full boot, no snapshot)")
    ap.add_argument("--fast-bin", default=None)
    args = ap.parse_args()

    backend = make_backend(args.backend, None)
    _warm = bool(args.snapshot)
    if args.snapshot:
        from chipformer.rv.snapshot import load as snap_load
        m = snap_load(args.snapshot)
        m.on_qwen_prompt = backend
        chunks = units_of(f"/tmp/hermes-agent {TASK}\n")
    else:
        if args.fast:
            from chipformer.rv.fastmachine import FastLinuxMachine
            m = FastLinuxMachine(args.image, args.dtb, backend=backend,
                                 binary=args.fast_bin)
        else:
            m = LinuxMachine()
            m.on_qwen_prompt = backend
            m.load_image(Path(args.image).read_bytes())
            m.boot(Path(args.dtb).read_bytes())
        chunks = units_of(INSTALL)
    logf = open(args.log, "wb") if args.log else None
    logged = 0
    phase = 2 if _warm else 0
    phase_slice = 0
    last_feed_slice = 0
    unit = None
    unit_mark = 0
    verify_mark = 0
    retries = 0
    install_failed = False
    fed = set()
    t0 = time.time()
    slices = 0
    while m.steps < args.max_steps and not m.halted:
        m.run(args.slice)
        slices += 1
        out = bytes(m.out)
        if logf and len(out) > logged:
            logf.write(out[logged:])
            logf.flush()
            logged = len(out)
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
            # line-paced feeding: one unit in flight, released by its
            # own echo (the tty read buffer drops bytes past ~4 KB
            # unread, which tore heredoc lines on the 15 KB agent.sh)
            if unit is not None:
                if len(out) - unit_mark >= len(unit) - 4 \
                        or slices - last_feed_slice >= 200:
                    unit = None
            elif chunks:
                unit = chunks.pop(0)
                m.feed_input(unit)
                unit_mark = len(out)
                last_feed_slice = slices
            elif _warm:
                phase = 3
            else:
                verify_mark = len(out)
                m.feed_input(VERIFY)
                phase = 25
        elif phase == 25:
            # the guest's own md5sums decide whether the install is
            # byte-exact; a torn heredoc retries whole, up to 3 times
            tail = out[verify_mark:]
            if b"MD5CHK" in tail:
                if all(h in tail for h in HASHES):
                    print("[agenttest] install verified byte-exact in the guest",
                          flush=True)
                    m.feed_input(f"echo STAGE_AGENT\n/tmp/hermes-agent {TASK}\n".encode())
                    phase = 3
                elif retries < 3:
                    retries += 1
                    print(f"[agenttest] install torn (retry {retries})", flush=True)
                    chunks = units_of(INSTALL)
                    phase = 2
                else:
                    install_failed = True
                    break
        elif phase == 3:
            # line-start anchor: the heredoc echo of agent.sh itself
            # contains "AGENT_ANSWER:" mid-line and must not trigger this
            if b"\nAGENT_ANSWER:" in out:
                m.feed_input(b"cat /tmp/hello.txt | od -c | head -1; echo CAT\"DONE\"\n")
                phase = 4
        elif phase == 4:
            if b"CATDONE" in out:
                break
        if slices % 20 == 0:
            print(f"[{m.steps:,} steps, phase={phase}, out={len(out)} B, "
                  f"{m.steps / max(time.time() - t0, 1e-9):,.0f} steps/s]", flush=True)

    out = bytes(m.out)
    if install_failed:
        print("AGENT ROUND TRIP: FAIL (install never verified byte-exact)")
        return 1
    answer_seen = b"\nAGENT_ANSWER:" in out
    file_ok = b"b   a   n   a   n   a" in out
    print(f"=== steps={m.steps:,} wall={time.time() - t0:.1f}s "
          f"backend={args.backend} ===")
    print("AGENT ROUND TRIP:", "PASS" if (answer_seen and file_ok)
          else f"FAIL (answer={answer_seen} file_content_verified={file_ok})")
    return 0 if (answer_seen and file_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
