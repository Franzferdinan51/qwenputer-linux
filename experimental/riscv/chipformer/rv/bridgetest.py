"""Automated delegation-bridge test: can the model in the seat USE the
guest's hermes-agent command?

    python3 -m chipformer.rv.bridgetest Image [--live] [--log bridge.out]

Default mode seats a deterministic scripted model behind
qwenbackend.agent_bridge_backend: the "model" answers the terminal's
question by delegating (AGENT: ...), serves the harness's tool loop
itself, then folds the HERMES REPORT into its final answer. That proves
the whole path with no model quality in the loop: bridge marker ->
ask.sh runs /tmp/hermes-agent in this same partition -> real file work
-> report -> final QWEN_ANSWER. Ground truth is checked with od.

--live swaps the script for the real LM Studio model behind the same
bridge: whether it chooses to delegate is then the model's call, and
the verdict lines say which parts happened.
"""

import argparse
import hashlib
import time
from pathlib import Path

from chipformer.rv.linux import LinuxMachine
from chipformer.rv.qwenbackend import AGENT_MARK, agent_bridge_backend, make_backend

HERE = Path(__file__).parent
ASK = (HERE / "env" / "linux" / "ask.sh").read_text()
AGENT = (HERE / "env" / "linux" / "agent.sh").read_text()
QUESTION = ('create a file named /tmp/deep.txt containing the word '
            'banana, then tell me what the file says')
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
    """Line-paced feed units (see agenttest.units_of): one unit in
    flight, released by its own echo - blind chunk feeding tears
    heredocs on this guest tty."""
    units = []
    for line in text.splitlines(keepends=True):
        b = line.encode()
        units += [b[i:i + 256] for i in range(0, len(b), 256)] or [b""]
    return units


class ScriptedSeat:
    """A model that always delegates, then works its own tool loop."""

    def __init__(self):
        self.harness_calls = 0

    def __call__(self, prompt: str) -> str:
        if "HERMES REPORT:" in prompt:
            rep = prompt.split("HERMES REPORT:", 1)[1].strip()
            return f"the hermes agent reports: {rep}"
        if AGENT_MARK in prompt:
            self.harness_calls += 1
            if self.harness_calls == 1:
                return "TOOL run echo banana > /tmp/deep.txt"
            if self.harness_calls == 2:
                return "TOOL run cat /tmp/deep.txt"
            return "SAY: the file says banana"
        return ("AGENT: write the word banana into /tmp/deep.txt then "
                "read it back and tell me what it says")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--dtb", default=str(HERE / "env" / "linux" / "sixtyfourmb.dtb"))
    ap.add_argument("--live", action="store_true",
                    help="seat the real LM Studio model instead of the script")
    ap.add_argument("--log", default=None)
    ap.add_argument("--max-steps", type=int, default=250_000_000)
    ap.add_argument("--slice", type=int, default=250_000)
    ap.add_argument("--snapshot", default=None,
                    help="start warm from a snapshot.py shell (skips boot+install)")
    ap.add_argument("--fast", action="store_true",
                    help="run on the fastlinux.c native core (full boot, no snapshot)")
    ap.add_argument("--fast-bin", default=None)
    args = ap.parse_args()

    if args.live:
        backend = agent_bridge_backend(make_backend("builtin", None))
    else:
        backend = agent_bridge_backend(ScriptedSeat())
    if args.snapshot:
        from chipformer.rv.snapshot import load as snap_load
        m = snap_load(args.snapshot)
        m.on_qwen_prompt = backend
        chunks = units_of(f'/tmp/ask "{QUESTION}"\n')
        _warm = True
    else:
        _warm = False
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
            elif _warm:
                phase = 3
            else:
                verify_mark = len(out)
                m.feed_input(VERIFY)
                phase = 25
        elif phase == 25:
            # the guest's own md5sums decide; a torn install retries
            tail = out[verify_mark:]
            if b"MD5CHK" in tail:
                if all(h in tail for h in HASHES):
                    print("[bridgetest] install verified byte-exact "
                          "in the guest", flush=True)
                    m.feed_input(b"echo STAGE_BRIDGE\n")
                    m.feed_input(f'/tmp/ask "{QUESTION}"\n'.encode())
                    phase = 3
                elif retries < 3:
                    retries += 1
                    print(f"[bridgetest] install torn (retry {retries})",
                          flush=True)
                    chunks = units_of(INSTALL)
                    phase = 2
                else:
                    install_failed = True
                    break
        elif phase == 3:
            if b"\nQWEN_ANSWER:" in out:
                m.feed_input(b"cat /tmp/deep.txt | od -c | head -1; echo CAT\"DONE\"\n")
                phase = 4
        elif phase == 4:
            if b"CATDONE" in out:
                break
        if slices % 20 == 0:
            print(f"[{m.steps:,} steps, phase={phase}, out={len(out)} B, "
                  f"{m.steps / max(time.time() - t0, 1e-9):,.0f} steps/s]", flush=True)

    out = bytes(m.out)
    # line-start anchor: the install heredoc's echo of ask.sh itself
    # contains "[ask] hermes-agent reported:" mid-line and must not
    # count as a real delegation
    delegated = b"\n[ask] hermes-agent reported:" in out
    qa = [l for l in out.split(b"\n") if l.startswith(b"QWEN_ANSWER:")]
    answer_ok = bool(qa) and b"banana" in qa[-1]
    file_ok = b"b   a   n   a   n   a" in out
    ok = delegated and answer_ok and file_ok
    print(f"=== steps={m.steps:,} wall={time.time() - t0:.1f}s "
          f"mode={'live' if args.live else 'scripted'} ===")
    print("BRIDGE:", "PASS" if ok else
          f"FAIL (delegated={delegated} answer_has_banana={answer_ok} "
          f"file_content_verified={file_ok} "
          f"install_failed={install_failed})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
