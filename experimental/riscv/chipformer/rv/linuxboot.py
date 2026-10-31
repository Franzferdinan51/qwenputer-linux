"""Boot the proven nommu Linux image on the Linux-capable reference machine.

    python3 -m chipformer.rv.linuxboot Image [--dtb sixtyfourmb.dtb]
        [--max-steps 80000000] [--type "uname -a;ls /"]

The image is the prebuilt kernel+initramfs from cnlohr/mini-rv32ima-images
(MIT; see docs/linux.md). The UART streams to stdout; --type commands are fed
to the console one per slice after /init starts, like typing at the shell.
"""

import argparse
import sys
import time
from pathlib import Path

from chipformer.rv.linux import LinuxMachine

HERE = Path(__file__).parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--dtb", default=str(HERE / "env" / "linux" / "sixtyfourmb.dtb"))
    ap.add_argument("--max-steps", type=int, default=90_000_000)
    ap.add_argument("--slice", type=int, default=250_000)
    ap.add_argument("--type", default="", help=";-separated console commands, typed after boot")
    ap.add_argument("--quiet", action="store_true", help="no progress lines on stderr")
    ap.add_argument("--qwen", choices=("off", "echo", "lmstudio"), default="echo",
                    help="backend behind the Qwen coprocessor device (guest /tmp/ask)")
    ap.add_argument("--lm-url", default=None,
                    help="LM Studio root for --qwen lmstudio (default: env QWEN_LM_URL or http://127.0.0.1:1234)")
    args = ap.parse_args()

    m = LinuxMachine()
    if args.qwen != "off":
        from chipformer.rv.qwenbackend import make_backend
        m.on_qwen_prompt = make_backend(args.qwen, args.lm_url)
        model = getattr(m.on_qwen_prompt, "model", None)
        print(f"[qwen device: backend={args.qwen}" + (f" model={model}]" if model else "]"),
              file=sys.stderr)
    m.load_image(Path(args.image).read_bytes())
    m.boot(Path(args.dtb).read_bytes())

    out = sys.stdout.buffer

    def emit(b: bytes) -> None:
        out.write(b)
        out.flush()

    m.on_output = emit
    cmds = [c for c in args.type.split(";") if c.strip()]
    typed = 0
    booted = False
    t0 = time.time()
    while not m.halted and m.steps < args.max_steps:
        m.run(args.slice)
        if not args.quiet:
            print(f"[{m.steps:,} steps, pc={m.pc:#010x}, out={len(m.out)} B, "
                  f"{m.steps / max(time.time() - t0, 1e-9):,.0f} steps/s]", file=sys.stderr)
        if not booted and b"Run /init as init process" in m.out:
            booted = True
            m.feed_input(b"\n")
        elif booted and typed < len(cmds):
            m.feed_input(cmds[typed].encode() + b"\n")
            typed += 1
    print(f"\n=== halted={m.halted} reason={m.exit_reason} steps={m.steps:,} "
          f"wall={time.time() - t0:.1f}s ===", file=sys.stderr)
    if m.csr_fallback:
        top = sorted(m.csr_fallback.items(), key=lambda kv: -kv[1])[:12]
        print("unknown-CSR touches (trap-and-emulate seam): "
              + ", ".join(f"{c:#x}x{n}" for c, n in top), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
