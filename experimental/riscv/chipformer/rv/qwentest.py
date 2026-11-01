"""End-to-end proof for the Qwen coprocessor device.

Boots the Linux image on the reference machine, logs in as root, installs
the guest-side /tmp/ask script (env/linux/ask.sh) over the serial console,
runs it, and checks the answer made the full round trip:

    guest shell -> MMIO window -> host backend -> MMIO window -> guest shell

    python3 -m chipformer.rv.qwentest Image [--backend echo|lmstudio|builtin]
        [--prompt "hello qwen"] [--max-steps 300000000]
        [--log boot-logs/qwen1.out]

Exit code 0 = PASS (the answer text appeared on the guest console and the
host saw the exact prompt), 1 = FAIL (the transcript says why).
"""

import argparse
import shlex
import sys
import time
from pathlib import Path

from chipformer.rv.linux import LinuxMachine
from chipformer.rv.qwenbackend import make_backend

HERE = Path(__file__).parent
ASK_SH = (HERE / "env" / "linux" / "ask.sh").read_text()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--dtb", default=str(HERE / "env" / "linux" / "sixtyfourmb.dtb"))
    ap.add_argument("--backend", choices=("echo", "lmstudio", "builtin"),
                    default="echo")
    ap.add_argument("--lm-url", default=None)
    ap.add_argument("--prompt", default="hello qwen")
    ap.add_argument("--max-steps", type=int, default=300_000_000)
    ap.add_argument("--slice", type=int, default=250_000)
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    seen: list[str] = []
    base = make_backend(args.backend, args.lm_url)

    def backend(prompt: str) -> str:
        seen.append(prompt)
        return base(prompt)

    m = LinuxMachine()
    m.on_qwen_prompt = backend
    m.load_image(Path(args.image).read_bytes())
    m.boot(Path(args.dtb).read_bytes())

    logf = open(args.log, "wb") if args.log else None

    def emit(b: bytes) -> None:
        if logf:
            logf.write(b)
            logf.flush()

    m.on_output = emit

    script = (
        "echo STAGE_PROBE\n"
        "command -v devmem; command -v dd; command -v od; ls -l /dev/mem\n"
        "cat > /tmp/ask <<'QWEOF'\n" + ASK_SH + "QWEOF\n"
        "chmod +x /tmp/ask\n"
        "echo STAGE_ASK\n"
        f"/tmp/ask {shlex.quote(args.prompt)}\n"
    ).encode()
    # fed only after the answer is seen on the console: typing the next
    # command while ask is still waiting on its `read` races the injected
    # answer for the same tty input queue (observed 2026-10-31 - the shell
    # ate the answer line and tried to execute it)
    marker = b'echo ASKT"EST_DONE"\n'

    # phase 0: wait for the getty login prompt
    # phase 1: login sent; wait a few slices, send a canary, wait for it to run
    # phase 2: canary seen; feed the script in chunks, one per slice
    # phase 3: script fully fed; wait for ASKTEST_DONE
    phase = 0
    phase_steps = 0
    login_out_len = 0
    fed: set = set()
    pending = bytearray()
    t0 = time.time()
    while not m.halted and m.steps < args.max_steps:
        m.run(args.slice)
        out = bytes(m.out)
        if phase == 0:
            if b"buildroot login:" in out:
                m.feed_input(b"root\n")
                login_out_len = len(out)
                phase_steps = m.steps
                phase = 1
        elif phase == 1:
            if b"CANARY_42" in out:
                pending = bytearray(script)
                phase = 2
            else:
                # getty may flush input queued before it opened the tty, and
                # its read timing is not observable from here - so retry the
                # login and the canary on a fixed schedule, whatever the
                # console shows.
                due = m.steps - phase_steps
                for at, data in ((2, b"root\n"), (6, b"echo CANARY_$((40+2))\n"),
                                 (16, b"echo CANARY_$((40+2))\n"),
                                 (32, b"echo CANARY_$((40+2))\n")):
                    key = (at, data)
                    if due >= at * args.slice and key not in fed:
                        fed.add(key)
                        m.feed_input(data)
        elif phase == 2:
            if pending:
                m.feed_input(bytes(pending[:256]))
                del pending[:256]
            else:
                phase = 3
        elif phase == 3:
            # line-start anchor: the heredoc echo of ask.sh itself contains
            # "QWEN_ANSWER:" mid-line (in the printf source) and must not
            # trigger this - only ask's real output starts a line with it
            if b"\nQWEN_ANSWER:" in out:
                m.feed_input(marker)
                phase = 4
        elif phase == 4:
            if b"ASKTEST_DONE" in out:
                break
        if m.steps % (20 * args.slice) < args.slice:
            print(f"[{m.steps:,} steps, phase={phase}, pc={m.pc:#010x}, "
                  f"inp={len(m.inp)}, out={len(out)} B, "
                  f"{m.steps / max(time.time() - t0, 1e-9):,.0f} steps/s]", file=sys.stderr)

    out = bytes(m.out)
    if logf:
        logf.close()
    print(f"=== steps={m.steps:,} wall={time.time() - t0:.1f}s "
          f"backend={args.backend} prompts_seen={seen!r} ===", file=sys.stderr)
    if args.backend == "echo":
        ok = seen == [args.prompt] and f"QWEN_ANSWER: qwen-echo: {args.prompt}".encode() in out
    else:
        ok = seen == [args.prompt] and b"QWEN_ANSWER:" in out and b"ASKTEST_DONE" in out
    print("QWEN ROUND TRIP: " + ("PASS" if ok else "FAIL"), file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
