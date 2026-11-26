#!/usr/bin/env python3
"""Qwenputer-X86 host bridge: serial session, scheduler, and agent loop.

The bare-metal OS is the computer; this script is its window on the host —
the x86 sibling of the RISC-V track's liveterm.py / agentshell.sh:

  * QemuOS        — spawn QEMU, talk to the `qwenputer>` REPL over serial,
                    return each reply's text between prompts.
  * Scheduler     — agentshell ported: a jobs file (JSON lines) whose due
                    lines are sent to the OS on a tick —
                        {"delay_s": 60, "line": "/calc 6*7"}
                        {"delay_s": 3600, "line": "Summarise today's notes."}
                    One-shot: a job fires once, then is removed from the file.
  * Agent loop    — the model is the brain, the kernel is the hands. The
                    prompt teaches the model to emit tool lines:
                        CALC: <expr>        -> answered by kernel /calc (exact)
                        PUT: <name> <text>  -> kernel /put
                        GET: <name>         -> kernel /get
                    Tool results are fed back as the next turn, up to
                    --max-steps times. Same gadget pattern as the RISC-V
                    track's hermes-agent, now against a real x86 CPU.

Usage:
    python3 qwenputer/qp_agent.py --model models/qwen3-0.6b-q8_0.asmbin \
        [--jobs qwenputer/jobs.json] [--smp 4]
    # then type chat, /calc ..., or /halt to power off.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

PROMPT = b"qwenputer>"
TOOL_RE = re.compile(r"^(CALC|PUT|GET):\s*(.+)$", re.MULTILINE)


def parse_tool_lines(text: str):
    """Extract (tool, args) pairs from a model reply."""
    out = []
    for tool, args in TOOL_RE.findall(text):
        out.append((tool, args.strip()))
    return out


def tool_to_kernel_line(tool: str, args: str) -> str:
    if tool == "CALC":
        return f"/calc {args}"
    if tool == "PUT":
        return f"/put {args}"
    if tool == "GET":
        return f"/get {args.split()[0]}"
    raise ValueError(tool)


class QemuOS:
    def __init__(self, kernel: str, model: str, smp: int = 4, timeout: int = 600):
        accel = "kvm" if os.access("/dev/kvm", os.W_OK) else "tcg"
        cpu = "host" if accel == "kvm" else "max"
        self.timeout = timeout
        self.proc = subprocess.Popen(
            [
                "qemu-system-x86_64", "-machine", f"accel={accel}",
                "-cpu", cpu, "-m", "2G", "-smp", str(smp),
                "-kernel", kernel, "-initrd", model,
                "-nographic", "-serial", "stdio", "-monitor", "none",
                "-display", "none",
            ],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, bufsize=0,
        )
        self.buf = bytearray()
        self._read_until_prompt()  # boot banner + first prompt

    def _read_until_prompt(self) -> str:
        start = len(self.buf)
        deadline = time.time() + self.timeout
        while PROMPT not in self.buf[start:]:
            if time.time() > deadline:
                raise TimeoutError("OS did not reach the qwenputer> prompt")
            chunk = self.proc.stdout.read(1) if self.proc.stdout else b""
            if not chunk:
                if self.proc.poll() is not None:
                    raise RuntimeError("QEMU exited early")
                time.sleep(0.02)
                continue
            self.buf += chunk
        text = bytes(self.buf[start:]).decode("utf-8", "replace")
        return text.replace("qwenputer>", "").strip()

    def send(self, line: str) -> str:
        """Send one REPL line; return the reply text (prompt stripped)."""
        assert self.proc.stdin is not None
        self.proc.stdin.write(line.encode() + b"\r\n")
        self.proc.stdin.flush()
        return self._read_until_prompt()

    def halt(self):
        try:
            assert self.proc.stdin is not None
            self.proc.stdin.write(b"/halt\r\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=20)
        except Exception:
            self.proc.kill()


class Scheduler:
    """agentshell-style due-job scanner over a JSON-lines jobs file."""

    def __init__(self, path: str, os_: QemuOS, log=print):
        self.path = path
        self.os = os_
        self.log = log
        self.started = time.time()

    def tick(self):
        if not os.path.exists(self.path):
            return
        with open(self.path) as f:
            jobs = [json.loads(l) for l in f if l.strip()]
        now = time.time() - self.started
        due = [j for j in jobs if float(j.get("delay_s", 0)) <= now]
        if not due:
            return
        keep = [j for j in jobs if float(j.get("delay_s", 0)) > now]
        for job in due:
            self.log(f"[sched] firing: {job['line']}")
            reply = self.os.send(job["line"])
            self.log(f"[sched] reply: {reply[:400]}")
        with open(self.path, "w") as f:
            for job in keep:
                f.write(json.dumps(job) + "\n")


AGENT_INSTRUCTIONS = (
    "You have exact tools. To use one, reply with a single line:\n"
    "CALC: <integer expression>  (exact arithmetic)\n"
    "PUT: <name> <text>  (save a note)\n"
    "GET: <name>  (read a note)\n"
    "When you know the final answer, reply starting with SAY: .\n"
    "Task: "
)


def agent_run(os_: QemuOS, task: str, max_steps: int = 4, log=print) -> str:
    reply = os_.send(AGENT_INSTRUCTIONS + task)
    for step in range(max_steps):
        if "SAY:" in reply:
            return reply.split("SAY:", 1)[1].strip()
        tools = parse_tool_lines(reply)
        if not tools:
            return reply
        results = []
        for tool, args in tools:
            line = tool_to_kernel_line(tool, args)
            log(f"[agent] step {step}: {line}")
            results.append(f"{tool} result: {os_.send(line)}")
        reply = os_.send("\n".join(results) + "\nContinue, or answer with SAY:")
    return reply


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", default="build/asm-llm-os.elf")
    ap.add_argument("--model", required=True)
    ap.add_argument("--smp", type=int, default=4)
    ap.add_argument("--jobs", default="")
    ap.add_argument("--agent", default="", help="run one agent task and exit")
    args = ap.parse_args()

    os_ = QemuOS(args.kernel, args.model, smp=args.smp)
    print("[qp] Qwenputer-X86 up. Chat, /calc, /put, /get, /list, /halt.")
    try:
        if args.agent:
            print(agent_run(os_, args.agent))
            return 0
        sched = Scheduler(args.jobs, os_) if args.jobs else None
        while True:
            if sched:
                sched.tick()
            try:
                line = input("you> ")
            except EOFError:
                break
            if not line.strip():
                continue
            print(os_.send(line))
            if line.strip() in ("/halt", "/quit", "exit"):
                break
    finally:
        os_.halt()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
