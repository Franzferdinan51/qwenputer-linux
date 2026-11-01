"""A live, interactive Qwen-Linux terminal in the browser.

Runs the Linux image on the reference machine (chipformer.rv.linux) and
bridges its serial console to a small web page: the boot log streams in,
you log in (root, no password), and the shell is yours. The Qwen device
is wired to a real backend (default: LM Studio), and once your shell is
up this server types the /tmp/ask installer for you - after that,

    ask "what is the capital of France"

works right in the guest.

    python3 -m chipformer.rv.liveterm Image [--port 8951]
        [--backend lmstudio|echo|off] [--lm-url URL]

Stdlib only. Binds 0.0.0.0 - open it via this machine's LAN
address (or its tailnet address, if one is up).
"""

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from chipformer.rv.linux import LinuxMachine
from chipformer.rv.qwenbackend import (agent_bridge_backend,
                                       fetch_backend_wrapper, make_backend)

HERE = Path(__file__).parent
ASK_SH = (HERE / "env" / "linux" / "ask.sh").read_text()
AGENT_SH = (HERE / "env" / "linux" / "agent.sh").read_text()
ASHELL_SH = (HERE / "env" / "linux" / "agentshell.sh").read_text()
NB_SH = (HERE / "env" / "linux" / "nb.sh").read_text()
CALC_SH = (HERE / "env" / "linux" / "calc.sh").read_text()
# The guest's soul: env/linux/SOUL.md - the Hermes Agent SOUL.md
# (Nous Research) re-homed for this machine; seeded into the
# guest at /root/SOUL.md and folded into wake-up prompts by
# agent.sh (first ~300 chars carry identity + directness).
SOUL_MD = (HERE / "env" / "linux" / "SOUL.md").read_text()
# Seed skills (phase 3): the agent's starter library. Installed with
# [ -f ] guards like SOUL.md - the agent may fixskill them, and a
# reinstall must never clobber what it learned.
SKILL_UPKEEP = (HERE / "env" / "linux" / "skills" / "upkeep.md").read_text()
SKILL_PLAN = (HERE / "env" / "linux" / "skills" / "plan-long-task.md").read_text()
# Every installed file feeds the guest through a <<'QWEOF' heredoc, so
# each one must end with a newline - otherwise its last line glues
# onto the QWEOF terminator, the heredoc never ends, and the rest of
# the install (including the agentshell launch) is swallowed into it.
for _name in ("ASK_SH", "AGENT_SH", "ASHELL_SH", "NB_SH", "CALC_SH", "SOUL_MD",
              "SKILL_UPKEEP", "SKILL_PLAN"):
    if not globals()[_name].endswith("\n"):
        globals()[_name] += "\n"

PAGE = """<!doctype html>
<meta charset="utf-8">
<title>Qwen-Linux terminal</title>
<style>
  body { background:#0b0e11; color:#d6deeb; font-family:ui-monospace,Menlo,monospace;
         margin:0; padding:12px; }
  h1 { font-size:15px; font-weight:600; margin:2px 0 8px; }
  h1 span { color:#7f8c98; font-weight:400; }
  #term { background:#000; border:1px solid #26313a; border-radius:8px; padding:10px;
          height:70vh; overflow-y:auto; white-space:pre-wrap; word-break:break-all;
          font-size:13px; line-height:1.35; outline:none; }
  #row { display:flex; gap:8px; margin-top:8px; }
  #box { flex:1; background:#10161c; color:#d6deeb; border:1px solid #26313a;
         border-radius:6px; padding:8px; font:inherit; font-size:13px; }
  button { background:#1b2836; color:#d6deeb; border:1px solid #2e4356; border-radius:6px;
           padding:8px 14px; font:inherit; cursor:pointer; }
  p.hint { color:#7f8c98; font-size:12px; }
</style>
<h1>Qwen-Linux <span>Linux 6.1.14 on the Qwenputer reference CPU - login: root (no password), then try: ask "what is the capital of France" or hermes-agent "your task in quotes"</span></h1>
<div id="term" tabindex="0"></div>
<div id="row">
  <input id="box" placeholder="or type here and Send (Enter sends too) - good for phones">
  <button id="send">Send</button>
</div>
<p class="hint">Click the black terminal and type directly, keys go to the guest as you press them. The machine is patient: ~0.5M instructions/sec, so give commands a few seconds.</p>
<script>
const term = document.getElementById('term');
const box = document.getElementById('box');
let since = 0, text = '', raw = '';
// Long QWEN: prompt lines (the full system prompt, re-sent every agent
// round) fold to one short line once complete - the terminal shows the
// head plus a char count instead of ~2KB of repeated prompt per round.
function fold(line) {
  if (line.startsWith('QWEN:') && line.length > 160)
    return 'QWEN: ' + line.slice(5, 125) + ' ... [' + line.length + ' chars]';
  return line;
}
async function poll() {
  try {
    const r = await fetch('/api/out?since=' + since);
    const j = await r.json();
    if (j.data) {
      for (const ch of j.data) {
        if (ch === '\\b' || ch === '\\x7f') raw = raw.slice(0, -1);
        else raw += ch;
      }
      since = j.until;
      const parts = raw.split('\\n');
      raw = parts.pop();
      for (const ln of parts) text += fold(ln) + '\\n';
      // sticky scroll: follow the bottom only if the user was already
      // there - reading scrollback no longer gets yanked away.
      const stick = term.scrollHeight - term.scrollTop - term.clientHeight < 60;
      term.textContent = text + raw;
      if (stick) term.scrollTop = term.scrollHeight;
    }
  } catch (e) {}
  setTimeout(poll, 250);
}
async function send(s) {
  await fetch('/api/in', {method:'POST', headers:{'Content-Type':'application/json'},
                          body: JSON.stringify({data: s})});
}
term.addEventListener('keydown', e => {
  let s = null;
  if (e.key === 'Enter') s = '\\n';
  else if (e.key === 'Backspace') s = '\\x7f';
  else if (e.key === 'Tab') s = '\\t';
  else if (e.key === 'Escape') s = '\\x1b';
  else if (e.key.length === 1) s = e.ctrlKey ? String.fromCharCode(e.key.toUpperCase().charCodeAt(0) - 64) : e.key;
  if (s !== null) { e.preventDefault(); send(s); }
});
document.getElementById('send').onclick = () => { send(box.value + '\\n'); box.value=''; term.focus(); };
box.addEventListener('keydown', e => { if (e.key === 'Enter') document.getElementById('send').click(); });
term.focus();
poll();
</script>
"""


class MachineThread(threading.Thread):
    """Runs the guest in slices; auto-installs /tmp/ask once a shell is up."""

    def __init__(self, image: bytes, dtb: bytes, backend, machine=None, warm=False):
        super().__init__(daemon=True)
        if machine is not None:
            self.m = machine
            if warm:
                self.phase = 2  # snapshot shells are already installed
        else:
            self.m = LinuxMachine()
            self.m.load_image(image)
            self.m.boot(dtb)
        if backend is not None:
            self.m.on_qwen_prompt = backend
        self.lock = threading.Lock()
        if machine is None or not warm:
            self.phase = 0  # 0 boot, 1 canary sent, 2 ask installed
        self.phase_steps = 0
        self.seen_len = 0

    def feed(self, data: bytes) -> None:
        with self.lock:
            self.m.feed_input(data)

    def out_since(self, n: int) -> tuple[str, int]:
        with self.lock:
            chunk = bytes(self.m.out[n:])
            total = len(self.m.out)
        return chunk.decode("utf-8", "replace"), total

    def run(self) -> None:
        install = (
            "cat > /tmp/ask <<'QWEOF'\n" + ASK_SH + "QWEOF\n"
            "cat > /tmp/agent <<'QWEOF'\n" + AGENT_SH + "QWEOF\n"
            "cp /tmp/agent /tmp/hermes-agent\n"
            "cat > /tmp/agentshell <<'QWEOF'\n" + ASHELL_SH + "QWEOF\n"
            "cp /tmp/agentshell /root/agentshell\n"
            "cat > /tmp/nb <<'QWEOF'\n" + NB_SH + "QWEOF\n"
            "cat > /tmp/calc <<'QWEOF'\n" + CALC_SH + "QWEOF\n"
            "chmod +x /tmp/ask /tmp/agent /tmp/hermes-agent /tmp/agentshell "
            "/root/agentshell /tmp/nb /tmp/calc\n"
            "mkdir -p /root/nb /root/skills /root/apps\n"
            "[ -f /root/SOUL.md ] || cat > /root/SOUL.md <<'QWEOF'\n" + SOUL_MD + "QWEOF\n"
            "[ -f /root/skills/upkeep.md ] || cat > /root/skills/upkeep.md <<'QWEOF'\n" + SKILL_UPKEEP + "QWEOF\n"
            "[ -f /root/skills/plan-long-task.md ] || cat > /root/skills/plan-long-task.md <<'QWEOF'\n" + SKILL_PLAN + "QWEOF\n"
            "echo ASK_READY the agentshell takes the console now: shell commands and hermes-agent tasks run there, scheduled jobs fire on their own, exit returns here\n"
            "/tmp/agentshell\n"
        ).encode()
        pending = bytearray()
        fed_mark = 0
        fed_stall = 0
        while not self.m.halted:
            with self.lock:
                self.m.run(20_000)
                out = bytes(self.m.out)
                steps = self.m.steps
            if self.phase == 0 and b"buildroot login:" in out:
                self.phase = 1  # the user logs in; watch for their shell
                self.phase_steps = steps
            elif self.phase == 1:
                if b"CANARY_42" in out:
                    pending = bytearray(install)
                    self.phase = 2
                elif (b"login on" in out or b"~ #" in out) \
                        and steps - self.phase_steps >= 500_000:
                    # evidence of a completed login (the login banner or a
                    # shell prompt). Only NOW probe for the shell - a canary
                    # typed at the login prompt would be eaten as a username
                    # and wedge the session at a Password: prompt.
                    self.phase_steps = steps
                    self.feed(b"echo CANARY_$((40+2))\n")
            elif self.phase == 2 and pending:
                # drip-feed on echo growth: blind 256B chunks overflow
                # the guest UART queue on big installs and silently
                # tear lines out of the heredoc (seen 2026-10-31 with
                # the 21 KB framework install). Feed more only when
                # the guest has echoed (consumed) what it has - or
                # after a long stall, in case echo paused.
                if len(out) > fed_mark or fed_stall >= 300:
                    self.feed(bytes(pending[:256]))
                    del pending[:256]
                    fed_mark = len(out)
                    fed_stall = 0
                else:
                    fed_stall += 1
            time.sleep(0.01)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", default=None)
    ap.add_argument("--dtb", default=str(HERE / "env" / "linux" / "sixtyfourmb.dtb"))
    ap.add_argument("--port", type=int, default=8951)
    ap.add_argument("--backend", choices=("builtin", "lmstudio", "echo", "off"), default="builtin")
    ap.add_argument("--lm-url", default=None)
    ap.add_argument("--bridge", action="store_true",
                    help="let the model delegate questions to the guest's "
                         "hermes-agent (qwenbackend.agent_bridge_backend)")
    ap.add_argument("--snapshot", default=None,
                    help="start warm from a snapshot.py shell (image not needed)")
    ap.add_argument("--fast", action="store_true",
                    help="run the guest on the fastlinux.c native core")
    ap.add_argument("--fast-bin", default=None)
    args = ap.parse_args()
    if not args.snapshot and not args.image:
        ap.error("the Image path is required unless --snapshot is given")

    backend = None
    if args.backend != "off":
        backend = make_backend(args.backend, args.lm_url)
        if args.bridge:
            backend = agent_bridge_backend(backend)
        # the guest has no network: FETCHURL prompts are fetched by the
        # host (TOOL fetch), everything else goes to the model/bridge
        backend = fetch_backend_wrapper(backend)
    machine = None
    warm = False
    image = dtb = b""
    if args.snapshot:
        from chipformer.rv.snapshot import load as snap_load
        machine = snap_load(args.snapshot)
        warm = True
    elif args.fast:
        from chipformer.rv.fastmachine import FastLinuxMachine
        machine = FastLinuxMachine(args.image, args.dtb, backend=backend,
                                   binary=args.fast_bin)
    else:
        image = Path(args.image).read_bytes()
        dtb = Path(args.dtb).read_bytes()
    mt = MachineThread(image, dtb, backend, machine=machine, warm=warm)
    mt.start()
    model = getattr(backend, "model", None)
    print(f"[liveterm: backend={args.backend}" + (f" model={model}" if model else "")
          + f" port={args.port}]", flush=True)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == "/":
                body = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path.startswith("/api/out"):
                try:
                    since = int(self.path.split("since=")[-1]) \
                        if "since=" in self.path else 0
                except ValueError:
                    self.send_response(400)
                    self.end_headers()
                    return
                data, until = mt.out_since(max(since, 0))
                body = json.dumps({"data": data, "until": until}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/stats":
                # engine counters for the dashboard's System tab: guest
                # CPU steps retired + console bytes out (dashboard
                # samples these into steps/s and B/s rates)
                with mt.lock:
                    steps = int(getattr(mt.m, "steps", 0) or 0)
                    outb = len(getattr(mt.m, "out", b"") or b"")
                body = json.dumps(
                    {"steps": steps, "out_bytes": outb}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            if self.path == "/api/in":
                try:
                    n = int(self.headers.get("Content-Length", 0))
                    payload = json.loads(self.rfile.read(max(n, 0)) or b"{}")
                    data = str(payload.get("data", ""))
                except (ValueError, AttributeError):
                    self.send_response(400)
                    self.end_headers()
                    return
                mt.feed(data.encode("utf-8", "replace"))
                self.send_response(200)
                self.end_headers()
            else:
                self.send_response(404)
                self.end_headers()

    ThreadingHTTPServer(("0.0.0.0", args.port), H).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
