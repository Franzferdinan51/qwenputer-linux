#!/usr/bin/env python3
"""Qwenputer-X86 agent framework (v2) — the Hermes-style host bridge.

The bare-metal OS is the computer; this framework is the agent shell around
it — the x86 sibling of the RISC-V track's liveterm.py / agentshell.sh,
grown toward the Hermes pattern now that x86 has room for a full framework:

  * QemuOS        — spawn QEMU, talk to the `qwenputer>` REPL over serial,
                    return each reply's text between prompts.
  * Tool registry — every kernel command is a named tool the model can
                    call with one line: CALC / PUT / GET / DEL / LIST /
                    INFO. Tool lines are parsed from the model reply,
                    executed in the kernel, results fed back next turn.
  * Agent loop    — multi-step (default 8), SAY: ends the task; tool
                    errors are surfaced to the model, not hidden.
  * Memory        — Hermes-style persistent notes: a JSON memory file on
                    the host is synced INTO the kernel RAM FS at boot
                    (one /put per note) and pulled back out before halt,
                    so the session-only FS gains cross-session memory.
                    Kernel limits apply: 32 notes, names <= 31 chars,
                    text <= 255 bytes (longer text is truncated + marked).
  * Scheduler     — agentshell ported: JSON-lines jobs file; a job is a
                    raw REPL line ({"line": ...}), an agent task
                    ({"agent": "..."}), one-shot ({"delay_s": N}) or
                    repeating ({"every_s": N}).
  * Transcripts   — every session is logged (timestamped) to a sessions
                    directory, like Hermes session history.

Usage:
    python3 qwenputer/qp_agent.py --model models/qwen3-1.7b-q8_0.asmbin \
        --memory qwenputer/memory.json --jobs qwenputer/jobs.json \
        --sessions qwenputer/sessions --agent "task ..." --smp 8 --mem 4G
    # or no --agent for an interactive shell (type /halt to power off).
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
TOOL_RE = re.compile(r"^(CALC|PUT|GET|DEL|LIST|INFO):?\s*(.*)$", re.MULTILINE)
FS_NAME_MAX = 31
FS_TEXT_MAX = 255
FS_SLOTS = 32


def parse_tool_lines(text: str):
    """Extract (tool, args) pairs from a model reply."""
    return [(t, a.strip()) for t, a in TOOL_RE.findall(text)]


def tool_to_kernel_line(tool: str, args: str) -> str:
    if tool == "CALC":
        return f"/calc {args}"
    if tool == "PUT":
        return f"/put {args}"
    if tool == "GET":
        return f"/get {args.split()[0]}"
    if tool == "DEL":
        return f"/del {args.split()[0]}"
    if tool == "LIST":
        return "/list"
    if tool == "INFO":
        return "/info"
    raise ValueError(tool)


class QemuOS:
    def __init__(self, kernel: str, model: str, smp: int = 4,
                 mem: str = "2G", timeout: int = 900):
        accel = "kvm" if os.access("/dev/kvm", os.W_OK) else "tcg"
        cpu = "host" if accel == "kvm" else "max"
        self.timeout = timeout
        self.proc = subprocess.Popen(
            [
                "qemu-system-x86_64", "-machine", f"accel={accel}",
                "-cpu", cpu, "-m", mem, "-smp", str(smp),
                "-kernel", kernel, "-initrd", model,
                "-nographic", "-serial", "stdio", "-monitor", "none",
                "-display", "none",
            ],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, bufsize=0,
        )
        self.buf = bytearray()
        self.boot_text = self._read_until_prompt()

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
        """Send one REPL line; return the reply text with the typed-line
        echo and the kernel's [asm-llm-os] stats lines stripped — what
        remains is the machine's actual output. (v2 lesson: the echo
        contains the instruction text itself, which otherwise poisons
        tool-line parsing — it literally contains 'SAY:' and 'CALC:'.)"""
        assert "\n" not in line, "REPL input must be a single line"
        assert self.proc.stdin is not None
        self.proc.stdin.write(line.encode() + b"\r\n")
        self.proc.stdin.flush()
        text = self._read_until_prompt().replace("\r\n", "\n")
        if text.startswith(line):
            text = text[len(line):].lstrip("\n")
        lines = [l for l in text.split("\n")
                 if not l.startswith("[asm-llm-os]")]
        return "\n".join(lines).strip()

    def halt(self):
        try:
            assert self.proc.stdin is not None
            self.proc.stdin.write(b"/halt\r\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=20)
        except Exception:
            self.proc.kill()


class Memory:
    """Host-side persistent notes, synced through the kernel RAM FS."""

    def __init__(self, path: str, log=print):
        self.path = path
        self.log = log
        self.notes: dict[str, str] = {}
        if path and os.path.exists(path):
            with open(path) as f:
                self.notes = json.load(f)

    @staticmethod
    def _fit(name: str, text: str) -> tuple[str, str]:
        name = name.replace("\n", " ")[:FS_NAME_MAX]
        text = text.replace("\n", " ")
        if len(text) > FS_TEXT_MAX:
            text = text[: FS_TEXT_MAX - 4] + " ..."
        return name, text

    def sync_in(self, os_: QemuOS):
        for name, text in list(self.notes.items())[:FS_SLOTS]:
            n, t = self._fit(name, text)
            os_.send(f"/put {n} {t}")
        if self.notes:
            self.log(f"[memory] synced {min(len(self.notes), FS_SLOTS)} "
                     f"note(s) into the kernel FS")

    def sync_out(self, os_: QemuOS):
        listing = os_.send("/list")
        # kernel /list prints one "  <name>  (<n> B)" line per file
        names = re.findall(r"^\s*(\S+)\s+\(\d+ B\)", listing, re.MULTILINE)
        fresh: dict[str, str] = {}
        for name in names:
            body = os_.send(f"/get {name}")
            m = re.search(r":\s*(.*)", body, re.DOTALL)
            fresh[name] = (m.group(1).strip() if m else body.strip())
        if fresh != self.notes:
            self.notes = fresh
            if self.path:
                with open(self.path, "w") as f:
                    json.dump(self.notes, f, indent=1)
            self.log(f"[memory] saved {len(fresh)} note(s) from the kernel FS")

    def close(self, os_: QemuOS):
        try:
            self.sync_out(os_)
        except Exception as e:  # never lose the session over memory
            self.log(f"[memory] sync-out failed: {e}")


class Transcript:
    def __init__(self, sessions_dir: str, log=print):
        self.f = None
        if sessions_dir:
            os.makedirs(sessions_dir, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            self.path = os.path.join(sessions_dir, f"session-{stamp}.log")
            self.f = open(self.path, "a", buffering=1)
            log(f"[qp] transcript: {self.path}")

    def write(self, who: str, text: str):
        if self.f:
            self.f.write(f"[{time.strftime('%H:%M:%S')}] {who}: {text}\n")


class Scheduler:
    """agentshell-style due-job scanner over a JSON-lines jobs file.

    Job shapes:
        {"delay_s": 60, "line": "/calc 6*7"}          one-shot raw line
        {"delay_s": 60, "agent": "task text"}          one-shot agent task
        {"every_s": 3600, "line": "/info"}             repeating
    """

    def __init__(self, path: str, os_: QemuOS, agent_fn, log=print):
        self.path = path
        self.os = os_
        self.agent_fn = agent_fn
        self.log = log
        self.started = time.time()
        self.last_fired: dict[int, float] = {}

    def tick(self):
        if not self.path or not os.path.exists(self.path):
            return
        with open(self.path) as f:
            jobs = [json.loads(l) for l in f if l.strip()]
        now = time.time() - self.started
        keep, changed = [], False
        for idx, job in enumerate(jobs):
            every = job.get("every_s")
            if every:
                if now - self.last_fired.get(idx, -1e18) >= float(every):
                    self.last_fired[idx] = now
                    self._fire(job)
                keep.append(job)
            elif float(job.get("delay_s", 0)) <= now:
                self._fire(job)
                changed = True
            else:
                keep.append(job)
        if changed:
            with open(self.path, "w") as f:
                for job in keep:
                    f.write(json.dumps(job) + "\n")

    def _fire(self, job):
        if "agent" in job:
            self.log(f"[sched] agent task: {job['agent']}")
            self.log(f"[sched] answer: {self.agent_fn(job['agent'])}")
        else:
            self.log(f"[sched] firing: {job['line']}")
            self.log(f"[sched] reply: {self.os.send(job['line'])[:400]}")


# Single line, well under the kernel's 1024-byte line buffer, task included.
AGENT_INSTRUCTIONS = (
    "Agent mode. Tools - reply with exactly ONE line: "
    "CALC: <expr> (exact math) | PUT: <name> <text> (save note) | "
    "GET: <name> (read note) | DEL: <name> | LIST: (note names) | "
    "INFO: (status). The tool result returns as the next message. "
    "Do every part of the task in order, one tool per reply; never "
    "invent a tool result. Example: CALC: 123*456. Finish by replying: "
    "SAY: <final answer>. Task: "
)


def agent_run(os_: QemuOS, task: str, max_steps: int = 8, log=print,
              transcript: Transcript | None = None) -> str:
    def say(who, text):
        log(text if who == "[agent]" else f"{who} {text}")
        if transcript:
            transcript.write(who, text)

    reply = os_.send(AGENT_INSTRUCTIONS + task)
    say("[task]", task)
    for step in range(max_steps):
        say("[model]", reply[:600])
        # A small model often writes the WHOLE plan in one reply, SAY and
        # hallucinated results included. Discipline: if a tool line comes
        # before the first SAY:, run the tool — its real result replaces
        # whatever the model imagined. SAY: only finishes when no tool
        # line precedes it.
        first_tool = TOOL_RE.search(reply)
        say_at = reply.find("SAY:")
        if first_tool and (say_at == -1 or first_tool.start() < say_at):
            tool, args = first_tool.group(1), first_tool.group(2).strip()
            line = tool_to_kernel_line(tool, args)
            say("[agent]", f"step {step}: {line}")
            result = os_.send(line)
            say("[tool]", f"{tool} result: {result[:300]}")
            reply = os_.send(f"{tool} result: {result}. "
                             "Continue the task in order: next tool line, "
                             "or SAY: <final answer> only when every part "
                             "of the task is done.")
            continue
        if say_at >= 0:
            return reply[say_at + 4:].split("\n")[0].strip()
        return reply
    return reply


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", default="build/asm-llm-os.elf")
    ap.add_argument("--model", required=True)
    ap.add_argument("--smp", type=int, default=4)
    ap.add_argument("--mem", default="2G")
    ap.add_argument("--jobs", default="")
    ap.add_argument("--memory", default="")
    ap.add_argument("--sessions", default="")
    ap.add_argument("--agent", default="", help="run one agent task and exit")
    ap.add_argument("--max-steps", type=int, default=8)
    args = ap.parse_args()

    transcript = Transcript(args.sessions)
    os_ = QemuOS(args.kernel, args.model, smp=args.smp, mem=args.mem)
    print("[qp] Qwenputer-X86 agent framework v2 up. "
          "Chat, /calc, /put, /get, /list, /info, /halt.")
    transcript.write("[boot]", os_.boot_text[:500])
    memory = Memory(args.memory)
    memory.sync_in(os_)

    def run_task(task: str) -> str:
        return agent_run(os_, task, max_steps=args.max_steps,
                         transcript=transcript)

    try:
        if args.agent:
            answer = run_task(args.agent)
            print(f"[answer] {answer}")
            return 0
        sched = Scheduler(args.jobs, os_, run_task) if args.jobs else None
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
        memory.close(os_)
        os_.halt()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
