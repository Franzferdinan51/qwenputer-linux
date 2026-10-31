"""FastLinuxMachine: the fastlinux.c native core behind the same
interface the Python drivers use for LinuxMachine (out / feed_input /
run / steps / halted / exit_reason / on_qwen_prompt).

The core speaks the fd 3/4 host protocol from fastlinux.c: console on
its stdout (drained by a reader thread into .out), commands on fd 3,
events on fd 4. QWEN:/QWENM prompts arriving on fd 4 are answered by
the on_qwen_prompt backend, synchronously, in the event thread — the
core blocks for the ANS frame exactly like the device blocks the CPU.
With no backend the core is spawned with FASTLINUX_HOST=0 and never
asks.
"""

import os
import subprocess
import threading

DEFAULT_BIN = os.environ.get("FASTLINUX_BIN", "fastlinux")


class FastLinuxMachine:
    def __init__(self, image_path, dtb_path, backend=None, binary=None):
        self.out = bytearray()
        self.steps = 0
        self.halted = False
        self.exit_reason = None
        self.on_qwen_prompt = backend
        self.on_output = None
        self._wlock = threading.Lock()
        self._done = threading.Event()
        self._hash_ev = threading.Event()
        self._done_info = None
        self._hash_val = None

        cmd_r, cmd_w = os.pipe()
        ev_r, ev_w = os.pipe()
        env = dict(os.environ)
        env["FASTLINUX_CTL_IN"] = str(cmd_r)
        env["FASTLINUX_CTL_OUT"] = str(ev_w)
        if backend is None:
            env["FASTLINUX_HOST"] = "0"

        self._p = subprocess.Popen(
            [binary or DEFAULT_BIN, str(image_path), str(dtb_path)],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            pass_fds=(cmd_r, ev_w), env=env)
        os.close(cmd_r)
        os.close(ev_w)
        self._cmd = os.fdopen(cmd_w, "wb", buffering=0)
        self._ev = os.fdopen(ev_r, "rb", buffering=0)
        self._ready = threading.Event()
        threading.Thread(target=self._out_loop, daemon=True).start()
        threading.Thread(target=self._ev_loop, daemon=True).start()
        if not self._ready.wait(timeout=30):
            raise RuntimeError("fastlinux did not signal READY")

    # ---- plumbing -----------------------------------------------------

    def _out_loop(self):
        while True:
            chunk = self._p.stdout.read1(65536) if hasattr(self._p.stdout, "read1") \
                else self._p.stdout.read(65536)
            if not chunk:
                return
            self.out += chunk
            if self.on_output:
                self.on_output(chunk)

    def _ev_loop(self):
        buf = self._ev
        while True:
            line = buf.readline()
            if not line:
                return
            parts = line.split()
            if not parts:
                continue
            tag = parts[0]
            if tag == b"READY":
                self._ready.set()
            elif tag == b"DONE":
                self.steps = int(parts[1])
                self.halted = parts[2] == b"1"
                reason = parts[3].decode() if len(parts) > 3 else "none"
                self.exit_reason = None if reason == "none" else reason
                self._done_info = True
                self._done.set()
            elif tag == b"HASH":
                self._hash_val = parts[1].decode()
                self._hash_ev.set()
            elif tag in (b"QWEN", b"QWENM"):
                n = int(parts[1])
                payload = b""
                while len(payload) < n:
                    chunk = buf.read(n - len(payload))
                    if not chunk:
                        return
                    payload += chunk
                self._answer_qwen(payload)

    def _answer_qwen(self, payload: bytes):
        prompt = payload.decode("utf-8", "replace")
        if self.on_qwen_prompt is None:
            answer = "[qwen: no model attached]"
        else:
            try:
                answer = str(self.on_qwen_prompt(prompt))
            except Exception as e:  # mirror the oracle: never hang the guest
                answer = f"[qwen backend error: {e}]"
        data = answer.encode("utf-8", "replace")
        with self._wlock:
            self._cmd.write(b"ANS %d\n" % len(data) + data)

    def _send(self, data: bytes):
        with self._wlock:
            self._cmd.write(data)

    # ---- the LinuxMachine-shaped interface -----------------------------

    def feed_input(self, data: bytes) -> None:
        if data:
            self._send(b"FEED %d\n" % len(data) + data)

    def run(self, max_steps: int) -> int:
        before = self.steps
        self._done.clear()
        self._send(b"RUN %d\n" % max_steps)
        if not self._done.wait(timeout=3600):
            raise RuntimeError("fastlinux RUN did not complete")
        return self.steps - before

    def hash(self) -> str:
        self._hash_ev.clear()
        self._send(b"HASH\n")
        if not self._hash_ev.wait(timeout=120):
            raise RuntimeError("fastlinux HASH did not complete")
        return self._hash_val

    def close(self) -> None:
        try:
            self._send(b"QUIT\n")
            self._p.wait(timeout=10)
        except Exception:
            self._p.kill()
