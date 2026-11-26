#!/usr/bin/env python3
"""Boot the Qwenputer-X86 bare-metal OS in QEMU and test its REPL commands.

Boots with NO batch prompt, so no inference runs: the kernel binds the model,
enters the REPL, and every Qwenputer command (/calc, /put, /get, /list, /del,
/info) is exercised over the serial console. Chat inference parity is covered
separately by upstream's scripts/verify_os.py.

Usage:
    python3 qwenputer/qp_serial_test.py \
        --kernel build/asm-llm-os.elf --model models/qwen3-0.6b-q8_0.asmbin \
        [--smp 4] [--timeout 120] [--log boot-logs/qp-serial-test.out]

Exit code 0 iff every check passes. Requires qemu-system-x86_64; KVM is used
when /dev/kvm is available, otherwise TCG (fine here: no inference runs).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qpcalc import calc  # noqa: E402

CHECKS = [
    # (command sent, expected substring in reply)
    ("/info", "[qwenputer] Qwenputer-X86 v1.0"),
    ("/calc 123456789*987654321", "[calc] 121932631112635269"),
    ("/calc 2+3*4", "[calc] 14"),
    ("/calc (2+3)*4", "[calc] 20"),
    ("/calc 100/7", "[calc] 14"),
    ("/calc 100%7", "[calc] 2"),
    ("/calc -5*3+1", "[calc] -14"),
    ("/calc 10/0", "[calc] error"),
    ("/calc 9223372036854775807+1", "[calc] -9223372036854775808"),
    ("/put hello.txt the quick brown fox", "[fs] stored 'hello.txt' (19 B)"),
    ("/get hello.txt", "the quick brown fox"),
    ("/put note remember the milk", "[fs] stored 'note' (17 B)"),
    ("/list", "hello.txt"),
    ("/del hello.txt", "[fs] deleted 'hello.txt'"),
    ("/get hello.txt", "[fs] not found"),
    ("/qhelp", "/calc <expr>"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", default="build/asm-llm-os.elf")
    ap.add_argument("--model", required=True)
    ap.add_argument("--smp", type=int, default=4)
    ap.add_argument("--mem", default="2G")
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--log", default="")
    args = ap.parse_args()

    # oracle sanity: the python mirror must agree with the hardcoded checks
    for cmd, expect in CHECKS:
        if cmd.startswith("/calc ") and expect.startswith("[calc] ") and "error" not in expect:
            assert calc(cmd[6:]) == int(expect[7:]), (cmd, expect)

    accel = "kvm" if os.access("/dev/kvm", os.W_OK) else "tcg"
    cpu = "host" if accel == "kvm" else "max"
    cmd = [
        "qemu-system-x86_64",
        "-machine", f"accel={accel}",
        "-cpu", cpu,
        "-m", args.mem,
        "-smp", str(args.smp),
        "-kernel", args.kernel,
        "-initrd", args.model,
        "-nographic",
        "-serial", "stdio",
        "-monitor", "none",
        "-display", "none",
    ]
    print(f"[test] {' '.join(cmd)}  (accel={accel})", flush=True)
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, bufsize=0,
    )
    transcript = bytearray()
    deadline = time.time() + args.timeout

    def read_until(marker: bytes, start: int = 0) -> bool:
        while marker not in transcript[start:]:
            if time.time() > deadline:
                return False
            chunk = proc.stdout.read(1) if proc.stdout else b""
            if not chunk:
                if proc.poll() is not None:
                    return False
                time.sleep(0.02)
                continue
            transcript.extend(chunk)
        return True

    def send(line: str):
        assert proc.stdin is not None
        proc.stdin.write(line.encode() + b"\r\n")
        proc.stdin.flush()

    failures = []
    try:
        if not read_until(b"qwenputer>"):
            print("[test] FAIL: REPL prompt never appeared")
            failures.append("boot")
        else:
            print("[test] booted to qwenputer> prompt", flush=True)
            for command, expect in CHECKS:
                before = len(transcript)
                send(command)
                ok = read_until(b"qwenputer>", before)
                reply = bytes(transcript[before:]).decode("utf-8", "replace")
                if ok and expect in reply:
                    print(f"[test] PASS {command!r} -> {expect!r}")
                else:
                    print(f"[test] FAIL {command!r}: expected {expect!r}, got:\n{reply}")
                    failures.append(command)
            send("/halt")
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
    finally:
        if proc.poll() is None:
            proc.kill()
        if args.log:
            os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
            with open(args.log, "wb") as f:
                f.write(bytes(transcript))

    if failures:
        print(f"[test] {len(failures)} FAILURES: {failures}")
        return 1
    print(f"[test] ALL {len(CHECKS)} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
