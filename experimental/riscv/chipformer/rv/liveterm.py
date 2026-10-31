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

Stdlib only. Binds 0.0.0.0 - on Duckets' Mac that means the tailnet
address is the one to open (http://100.68.208.113:8951).
"""

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from chipformer.rv.linux import LinuxMachine
from chipformer.rv.qwenbackend import make_backend

HERE = Path(__file__).parent
ASK_SH = (HERE / "env" / "linux" / "ask.sh").read_text()
AGENT_SH = (HERE / "env" / "linux" / "agent.sh").read_text()

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
<h1>Qwen-Linux <span>Linux 6.1.14 on the Qwenputer reference CPU - login: root (no password), then try: ask "what is the capital of France"</span></h1>
<div id="term" tabindex="0"></div>
<div id="row">
  <input id="box" placeholder="or type here and Send (Enter sends too) - good for phones">
  <button id="send">Send</button>
</div>
<p class="hint">Click the black terminal and type directly, keys go to the guest as you press them. The machine is patient: ~0.5M instructions/sec, so give commands a few seconds.</p>
<script>
const term = document.getElementById('term');
const box = document.getElementById('box');
let since = 0, text = '';
async function poll() {
  try {
    const r = await fetch('/api/out?since=' + since);
    const j = await r.json();
    if (j.data) {
      for (const ch of j.data) {
        if (ch === '\\b' || ch === '\\x7f') text = text.slice(0, -1);
        else text += ch;
      }
      since = j.until;
      term.textContent = text;
      term.scrollTop = term.scrollHeight;
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

    def __init__(self, image: bytes, dtb: bytes, backend):
        super().__init__(daemon=True)
        self.m = LinuxMachine()
        if backend is not None:
            self.m.on_qwen_prompt = backend
        self.m.load_image(image)
        self.m.boot(dtb)
        self.lock = threading.Lock()
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
            "chmod +x /tmp/ask /tmp/agent\n"
            "echo ASK_READY ask-a-question or: agent <task for the model>\n"
        ).encode()
        pending = bytearray()
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
                self.feed(bytes(pending[:256]))
                del pending[:256]
            time.sleep(0.01)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--dtb", default=str(HERE / "env" / "linux" / "sixtyfourmb.dtb"))
    ap.add_argument("--port", type=int, default=8951)
    ap.add_argument("--backend", choices=("lmstudio", "echo", "off"), default="lmstudio")
    ap.add_argument("--lm-url", default=None)
    args = ap.parse_args()

    backend = None
    if args.backend != "off":
        backend = make_backend(args.backend, args.lm_url)
    mt = MachineThread(Path(args.image).read_bytes(), Path(args.dtb).read_bytes(), backend)
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
                since = int(self.path.split("since=")[-1]) if "since=" in self.path else 0
                data, until = mt.out_since(since)
                body = json.dumps({"data": data, "until": until}).encode()
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
                n = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(n) or b"{}")
                mt.feed(str(payload.get("data", "")).encode("utf-8", "replace"))
                self.send_response(200)
                self.end_headers()
            else:
                self.send_response(404)
                self.end_headers()

    ThreadingHTTPServer(("0.0.0.0", args.port), H).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
