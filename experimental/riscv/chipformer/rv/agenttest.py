"""Automated agent-harness test: boot, log in, install ask + agent, give
the agent a two-step task, and verify its work from the ground truth.

    python3 -m chipformer.rv.agenttest Image [--backend lmstudio]
        [--log agent-lm.out]

The task: "write the word banana into /tmp/hello.txt then read it back
and tell me what it says". PASS requires BOTH: an AGENT_ANSWER line from
the harness, and - checked afterwards with od, not the model's word -
/tmp/hello.txt really containing b a n a n a.
"""

import argparse
import time
from pathlib import Path

from chipformer.rv.linux import LinuxMachine
from chipformer.rv.qwenbackend import make_backend

HERE = Path(__file__).parent
ASK = (HERE / "env" / "linux" / "ask.sh").read_text()
AGENT = (HERE / "env" / "linux" / "agent.sh").read_text()
TASK = ("write the word banana into /tmp/hello.txt then read it back "
        "and tell me what it says")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--dtb", default=str(HERE / "env" / "linux" / "sixtyfourmb.dtb"))
    ap.add_argument("--backend", choices=("echo", "lmstudio"), default="lmstudio")
    ap.add_argument("--log", default=None)
    ap.add_argument("--max-steps", type=int, default=250_000_000)
    ap.add_argument("--slice", type=int, default=250_000)
    args = ap.parse_args()

    backend = make_backend(args.backend, None)
    m = LinuxMachine()
    m.on_qwen_prompt = backend
    m.load_image(Path(args.image).read_bytes())
    m.boot(Path(args.dtb).read_bytes())

    script = (
        "cat > /tmp/ask <<'QWEOF'\n" + ASK + "QWEOF\n"
        "cat > /tmp/agent <<'QWEOF'\n" + AGENT + "QWEOF\n"
        "chmod +x /tmp/ask /tmp/agent\n"
        "echo STAGE_AGENT\n"
        f"/tmp/agent {TASK}\n"
    ).encode()
    chunks = [script[i:i + 256] for i in range(0, len(script), 256)]
    logf = open(args.log, "wb") if args.log else None
    logged = 0
    phase = 0
    phase_slice = 0
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
            if chunks:
                m.feed_input(chunks.pop(0))
            else:
                phase = 3
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
    answer_seen = b"\nAGENT_ANSWER:" in out
    file_ok = b"b   a   n   a   n   a" in out
    print(f"=== steps={m.steps:,} wall={time.time() - t0:.1f}s "
          f"backend={args.backend} ===")
    print("AGENT ROUND TRIP:", "PASS" if (answer_seen and file_ok)
          else f"FAIL (answer={answer_seen} file_content_verified={file_ok})")
    return 0 if (answer_seen and file_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
