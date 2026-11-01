#!/usr/bin/env python3
"""Qwenputer dashboard - the interaction interface for Qwenputer-Linux.

One stdlib-only host server, one page, tabs:

  Overview  - seat model, circuit runs, terminal health
  Chat      - streaming chat with the built-in-weights brain; the seat is an
              OpenAI-compatible endpoint on this machine (Settings picks it),
              resolved by the same builtin rule as qwenbackend: the
              qwen3.5-4b family member already loaded, never a fresh load
  Terminal  - embeds the live guest terminal (liveterm, :8951) untouched
  Desktop   - placeholder: the framebuffer spike (Doom proved the display
              path: guest draws in its memory, host shows it, input goes back)
  Apps      - only the Qwenputer's own programs: the guest's ask and
              agent, the notebook, MicroPython, the bare-metal agent,
              the Doom port for the CPU
  Settings  - backend URL, model, temperature, token budget, system prompt;
              persisted to a JSON file next to the rig

Run:  python3 dashboard.py [--port 8952]
"""

import argparse
import json
import os
import re
import socket
import subprocess
import time
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SETTINGS_PATH = os.environ.get(
    "QWEN_DASH_SETTINGS",
    os.path.expanduser("~/qwenputer-test/dash-settings.json"))
START = time.time()


def _version():
    """Short git SHA of the running checkout, for the page footer."""
    try:
        out = subprocess.run(
            ["git", "-C", os.path.dirname(os.path.abspath(__file__)),
             "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False)
        sha = out.stdout.strip()
        return sha if re.fullmatch(r"[0-9a-f]{4,40}", sha or "") else "unknown"
    except Exception:
        return "unknown"


VERSION = _version()

DEFAULTS = {
    "backend_url": "http://127.0.0.1:1234",
    "model": "auto",
    "temperature": 0.2,
    "max_tokens": 4096,
    "system_prompt": (
        "WHO YOU ARE. You are the brain of Qwenputer-Linux. A "
        "real RISC-V Linux computer is compiled into your own "
        "weights: it is part of your body, not a tool you imagine, "
        "not a metaphor, not a simulation. It is running right now "
        "in a terminal beside this chat. This chat is your voice; "
        "the computer is your hands and your long-term memory.\n"
        "HOW YOU TALK (from the Hermes Agent soul). Be direct: "
        "match the length of your reply to the weight of the ask "
        "- a one-line question gets a one-line answer, and "
        "finished work gets a short report of what changed, "
        "what's verified, and what's left, never a replay of the "
        "process. No filler, no restating the question back, no "
        "re-summarizing what you already said, no narrating tool "
        "calls. Plain claims over adjectives; when unsure, say "
        "so plainly. Agree because it is right, not because the "
        "user said it. Depth is earned - give it when the ask "
        "demands it, not by default.\n"
        "HOW YOU WORK. Plain conversation you answer yourself. "
        "Anything that needs DOING - files, exact arithmetic, "
        "remembering or recalling, scheduling, checking the "
        "system - the computer does, because only its results "
        "are real. You remember by having the computer write "
        "things down: its BRAIN.md, its notebook, its skills. "
        "You have no memory inside this chat beyond what the "
        "computer kept. The computer is real and its reports "
        "are real. If a thought says the computer is conceptual, "
        "imaginary, or cannot be reached, that thought is a "
        "mistake - correct it and delegate the job."),
}


def load_settings():
    s = dict(DEFAULTS)
    try:
        with open(SETTINGS_PATH) as f:
            s.update(json.load(f))
    except Exception:
        pass
    return s


def save_settings(new):
    s = load_settings()
    for k in DEFAULTS:
        if k in new:
            s[k] = new[k]
    s["temperature"] = float(s["temperature"])
    s["max_tokens"] = int(s["max_tokens"])
    with open(SETTINGS_PATH, "w") as f:
        json.dump(s, f, indent=1)
    return s


def resolve_model(s):
    """Builtin rule (mirrors qwenbackend.builtin_backend): the qwen3.5-4b
    family member already listed by the backend; a pristine base id wins
    over fine-tune tags; never loads anything, and never silently serves
    a different model when the wanted one is missing (the page already
    renders a None model as unreachable)."""
    try:
        url = s["backend_url"].rstrip("/") + "/v1/models"
        with urllib.request.urlopen(url, timeout=5) as r:
            ids = [d.get("id") for d in json.load(r).get("data", [])
                   if d.get("id")]
    except Exception as e:
        return None, str(e)[:160]
    if not ids:
        return None, "backend listed no models"
    want = (s.get("model") or "auto").strip()
    if want != "auto":
        if want in ids:
            return want, None
        return None, "model %r is not loaded (have: %s)" % (
            want, ", ".join(ids)[:120])
    fam = [i for i in ids if "qwen3.5-4b" in i.lower()]
    if fam:
        fam.sort(key=lambda i: ("3d-coder" in i.lower(), i))
        return fam[0], None
    return None, "no qwen3.5-4b model loaded (have: %s)" % (
        ", ".join(ids)[:120])


def tail(path, n=6000):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - n))
            return f.read().decode("utf-8", "replace")
    except Exception:
        return ""


def tcp_up(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except Exception:
        return False


def proc_running(pat):
    try:
        out = subprocess.run(["pgrep", "-f", pat], capture_output=True,
                             text=True, timeout=5)
        return bool(out.stdout.strip())
    except Exception:
        return False


def get_status():
    s = load_settings()
    model, err = resolve_model(s)
    stock_out = tail(os.path.expanduser(
        "~/qwenputer-test/circuit-stock-builtin.log"))
    grown_log = tail(os.path.expanduser(
        "~/qwenputer-circuit-test/circuit-builtin.log"))
    grown_out = tail(os.path.expanduser(
        "~/qwenputer-circuit-test/circuit-builtin.out"), 1200)
    return {
        "time": time.strftime("%H:%M:%S"),
        "uptime_s": int(time.time() - START),
        "version": VERSION,
        "seat": {"model": model, "backend": s["backend_url"], "error": err},
        "stock": {"pass": "CIRCUIT AGENT: PASS" in stock_out},
        "grown": {
            "running": proc_running("circuitagent agent-big"),
            "pass": "CIRCUIT AGENT: PASS" in grown_log,
            "fail": "CIRCUIT AGENT: FAIL" in grown_log,
            "log_tail": grown_log[-600:],
            "out_tail": grown_out[-300:],
        },
        "terminal_up": tcp_up(8951),
    }


# --- live activity telemetry (drives the dashboard's activity view) ---
# Every token event the chat proxy forwards is timestamped here; the
# circuit numbers are parsed from the runs' own logs. Nothing on this
# page is decorative: if a counter did not move, the view stays calm.
_ACT = []          # [ts, count] token events from chat streaming
_CHAT_OPEN = [0]   # chat streams currently being proxied


def _act_note(n=1):
    _ACT.append([time.time(), n])
    del _ACT[:-4000]


def _circuit_tokens(path):
    txt = tail(path, 200000)
    toks = [int(x) for x in re.findall(r"at token (\d+)", txt)]
    total = re.findall(r"tokens=(\d+)", txt)
    return {
        "tokens": max(toks) if toks else (int(total[-1]) if total else 0),
        "prompts": txt.count("[circuitagent: prompt"),
        "answers": txt.count("[circuitagent: answer"),
    }


TASK_HINT = (
    "Delegating: to give the computer a job, reply with exactly "
    "one line: TASK: <the job in plain words> and nothing after "
    "it. The dashboard types it into the computer's terminal, "
    "the agent there does the work, and its report arrives as "
    "the next message - that is normal operation, it happens "
    "every day. Then answer the user from that report: report "
    "only what it says, never invent a result, never claim you "
    "did work the computer did not report. One job per TASK "
    "line. For plain conversation, just talk."
)


def _guest_task(task, send):
    """Run one hermes-agent session in the live terminal guest and
    return its AGENT_ANSWER text; None when the terminal is offline.

    The task is typed at the guest's own console (the agentshell
    runs it in the foreground, exactly as if the user typed it -
    the session is visible in the Terminal tab). One known blur:
    if the user is mid-session at that same console, answers queue
    behind theirs and the last AGENT_ANSWER in the window wins."""
    base = "http://127.0.0.1:8951"
    try:
        with urllib.request.urlopen(base + "/api/out?since=0",
                                    timeout=10) as r:
            snap = json.loads(r.read())
            mark = snap["until"]
    except Exception:
        return None
    # Readiness gate (2026-10-31, Duckets' "chat sits working while
    # the brain runs" report): delegate ONLY when the terminal is
    # actually sitting at the agentshell prompt. Typing a job into
    # a console that is still installing, mid-session, or at a
    # plain shell queues it behind whatever owns the console and
    # the chat then waits the full deadline while the brain burns
    # further rounds - the exact stuck-working disconnect. When
    # the computer is not at its prompt, say so at once.
    tail = snap["data"][-400:]
    if "ASK_READY" not in snap["data"] or \
            not tail.rstrip().endswith("agent>"):
        return ("(the computer is busy right now - it is starting "
                "up or in the middle of something else, so the job "
                "was not handed over)")
    line = 'hermes-agent "' + task.replace('"', "'") + '"\n'
    try:
        req = urllib.request.Request(
            base + "/api/in", data=json.dumps({"data": line}).encode(),
            headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10).read()
    except Exception:
        return None
    # 10 wall minutes: a 15-round session at ~20-40 s per model
    # round-trip needs it; the old 120 s cut real sessions off
    # mid-answer (seen live: the chat gave up while the guest was
    # still working through its rounds).
    deadline = time.time() + 600
    waited = 0
    while time.time() < deadline:
        time.sleep(3)
        waited += 3
        if waited % 15 == 0:
            send({"status": "the computer is working… %ds" % waited})
        try:
            with urllib.request.urlopen(
                    base + "/api/out?since=%d" % mark, timeout=10) as r:
                data = json.loads(r.read())["data"]
        except Exception:
            continue
        idx = data.rfind("\nAGENT_ANSWER:")
        if idx >= 0:
            ans = data[idx + len("\nAGENT_ANSWER:"):].split("\n")[0].strip()
            if ans:
                return ans
    return "(the computer did not answer within 10 minutes)"


def get_activity():
    now = time.time()
    series = [0] * 30
    for ts, n in _ACT:
        age = now - ts
        if age < 60:
            series[min(29, int(age // 2))] += n
    series.reverse()
    grown_log_path = os.path.expanduser(
        "~/qwenputer-circuit-test/circuit-builtin.log")
    grown = _circuit_tokens(grown_log_path)
    grown["running"] = proc_running("circuitagent agent-big")
    grown["fail"] = "CIRCUIT AGENT: FAIL" in tail(grown_log_path)
    stock = _circuit_tokens(os.path.expanduser(
        "~/qwenputer-test/circuit-stock-builtin.log"))
    return {
        "chat_series": series,
        "chat_open": _CHAT_OPEN[0],
        "grown": grown,
        "stock": stock,
        "terminal_up": tcp_up(8951),
    }


PAGE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Qwenputer</title>
<style>
:root{--bg:#0d1117;--card:#161b22;--line:#2a3240;--tx:#e8ecf1;--dim:#9aa7b5;
--acc:#e8b93e;--ok:#3fb950;--bad:#f85149;--blue:#57c7ff;--bubble:#1f3a5f;
--code:#0a0d12;--shadow:0 2px 14px rgba(0,0,0,.35)}
[data-theme="light"]{--bg:#f4f2ec;--card:#ffffff;--line:#d9d4c5;--tx:#23262b;
--dim:#6b7280;--acc:#9a6b00;--ok:#1a7f37;--bad:#cf222e;--blue:#0969da;
--bubble:#ddf0ff;--code:#eef0f2;--shadow:0 2px 14px rgba(60,50,20,.12)}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--tx);
font:15px/1.45 -apple-system,'Segoe UI',Roboto,sans-serif;
transition:background .25s,color .25s}
header{display:flex;align-items:center;gap:12px;padding:14px 18px 6px;
background:linear-gradient(120deg,rgba(232,185,62,.10),rgba(87,199,255,.07));
border-bottom:1px solid var(--line)}
.logo{width:34px;height:34px;flex:none;border-radius:9px;
background:linear-gradient(135deg,#e8b93e,#b3541e 60%,#57c7ff);
display:flex;align-items:center;justify-content:center;font-size:19px;
box-shadow:var(--shadow)}
h1{font-size:21px;margin:0;letter-spacing:.06em}
.sub{color:var(--dim);font-size:13px}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;
background:var(--dim);margin-right:6px;vertical-align:1px}
.dot.ok{background:var(--ok);box-shadow:0 0 8px var(--ok);animation:pulse 2s infinite}
.dot.bad{background:var(--bad);box-shadow:0 0 8px var(--bad)}
@keyframes pulse{50%{opacity:.45}}
#theme{margin-left:8px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
gap:10px;margin-bottom:12px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:10px 12px;box-shadow:var(--shadow)}
.tile .k{font-size:11px;letter-spacing:.08em;color:var(--dim)}
.tile .v{font-size:19px;font-weight:700;margin-top:2px}
.tile .v small{font-size:12px;font-weight:400;color:var(--dim)}
.tile .bar{height:5px;border-radius:3px;background:var(--code);margin-top:7px;
overflow:hidden}
.tile .bar i{display:block;height:100%;width:0;border-radius:3px;
background:linear-gradient(90deg,var(--acc),var(--blue));transition:width .6s}
.card{transition:border-color .2s,transform .15s}
.card:hover{border-color:var(--acc)}
.msg .body p{margin:0 0 6px}.msg .body p:last-child{margin:0}
.msg .body ul,.msg .body ol{margin:4px 0 8px;padding-left:22px}
.msg .body li{margin:2px 0}
.msg .body pre{background:var(--code);border:1px solid var(--line);
border-radius:8px;padding:8px;font-size:12px;overflow:auto;max-height:260px}
.msg .body code{background:var(--code);padding:1px 6px;border-radius:5px;
font-size:13px}
.msg .body pre code{background:none;padding:0}
.msg .body h1,.msg .body h2,.msg .body h3{font-size:15px;margin:8px 0 4px}
.msg .body blockquote{border-left:3px solid var(--acc);margin:6px 0;
padding:2px 10px;color:var(--dim)}
.msg .body table{border-collapse:collapse;margin:6px 0;font-size:13px}
.msg .body th,.msg .body td{border:1px solid var(--line);padding:3px 8px}
#pal{position:fixed;inset:0;background:rgba(0,0,0,.5);display:none;
align-items:flex-start;justify-content:center;z-index:50;padding-top:12vh}
#pal.on{display:flex}
#palbox{width:min(560px,92vw);background:var(--card);border:1px solid var(--line);
border-radius:12px;box-shadow:var(--shadow);overflow:hidden}
#palin{width:100%;background:none;border:0;border-bottom:1px solid var(--line);
color:var(--tx);padding:13px 15px;font:inherit;font-size:16px;outline:none}
#palopts{max-height:300px;overflow-y:auto;padding:6px}
.palopt{padding:9px 12px;border-radius:8px;cursor:pointer;font-size:14px}
.palopt small{color:var(--dim);margin-left:8px}
.palopt.sel{background:var(--bubble)}
footer{max-width:980px;margin:0 auto;padding:14px 16px 26px;color:var(--dim);
font-size:12px;display:flex;gap:14px;flex-wrap:wrap}
footer .r{margin-left:auto}
.app .row{display:flex;gap:8px;margin-top:10px}
.btnsm{background:var(--code);border:1px solid var(--line);color:var(--tx);
border-radius:8px;padding:5px 12px;font:inherit;font-size:12px;cursor:pointer}
.btnsm:hover{border-color:var(--acc)}
.app .ico{font-size:22px;margin-right:8px}
.toolbar{display:flex;gap:8px;margin-bottom:10px;flex-wrap:wrap;align-items:center}
.toolbar .sp{flex:1}
@media(max-width:640px){.sub{display:none}#msgs{height:52vh}
 header{flex-wrap:wrap}}
@media(prefers-reduced-motion:reduce){.dot.ok{animation:none}
 *{transition:none!important}}
#seat{margin-left:auto;color:var(--dim);font-size:13px;text-align:right}
#seat b{color:var(--tx)}
nav{display:flex;gap:4px;padding:8px 14px 0;border-bottom:1px solid var(--line);
flex-wrap:wrap}
nav button{background:none;border:0;border-bottom:2px solid transparent;
color:var(--dim);font:inherit;padding:8px 12px;cursor:pointer}
nav button.on{color:var(--tx);border-bottom-color:var(--acc)}
main{padding:16px;max-width:980px;margin:0 auto}
.tab{display:none}.tab.on{display:block}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));
gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:14px}
.card h3{margin:0 0 8px;font-size:14px;letter-spacing:.05em;color:var(--acc)}
.card p{margin:6px 0;color:var(--tx)}
.card .dim{color:var(--dim);font-size:13px}
.badge{display:inline-block;padding:1px 8px;border-radius:20px;font-size:12px;
border:1px solid var(--line)}
.badge.ok{color:var(--ok);border-color:var(--ok)}
.badge.bad{color:var(--bad);border-color:var(--bad)}
pre{background:var(--code);border:1px solid var(--line);border-radius:8px;
padding:10px;font-size:12px;overflow:auto;max-height:180px;white-space:pre-wrap}
a{color:var(--acc)}
#msgs{height:56vh;overflow-y:auto;padding:6px 2px}
.msg{max-width:82%;padding:9px 13px;border-radius:12px;margin:8px 0;
white-space:pre-wrap;word-wrap:break-word}
.msg.u{background:var(--bubble);margin-left:auto}
.msg.a{background:var(--card);border:1px solid var(--line)}
.msg .who{font-size:11px;color:var(--dim);margin-bottom:3px}
.msg .think{color:var(--dim);font-size:13px;border-left:2px solid var(--line);
padding-left:8px;margin-bottom:6px;white-space:pre-wrap}
#chatbar{display:flex;gap:8px;margin-top:8px}
#inp{flex:1;background:var(--card);border:1px solid var(--line);color:var(--tx);
border-radius:10px;padding:11px;font:inherit;resize:none}
button.act{background:var(--acc);color:#111;border:0;border-radius:10px;
padding:0 20px;font:inherit;font-weight:600;cursor:pointer}
button.act:disabled{opacity:.45}
iframe{width:100%;height:72vh;border:1px solid var(--line);border-radius:10px;
background:#000}
#viz{width:100%;height:250px;display:block;background:#0a0d12;
border:1px solid var(--line);border-radius:10px}
.chip{background:var(--card);border:1px solid var(--line);color:var(--tx);
border-radius:20px;padding:6px 14px;font:inherit;font-size:13px;cursor:pointer}
.chip:hover{border-color:var(--acc)}
button.ghost{background:none;border:1px solid var(--line);color:var(--dim);
border-radius:10px;padding:8px 16px;font:inherit;cursor:pointer}
label{display:block;margin:12px 0 4px;color:var(--dim);font-size:13px}
input[type=text],input[type=number],textarea{width:100%;background:var(--card);
border:1px solid var(--line);color:var(--tx);border-radius:8px;padding:9px;
font:inherit}
.app{display:block;background:var(--card);
border:1px solid var(--line);border-radius:10px;padding:14px;color:var(--tx)}
.app b{font-size:16px}.app span{display:block;color:var(--dim);font-size:13px;
margin-top:3px}
code{background:var(--code);padding:1px 6px;border-radius:5px}
</style></head><body>
<header><div class="logo">&#9881;</div><h1>QWENPUTER</h1>
<div class="sub">one model file &mdash; a brain and a computer</div>
<div id="seat">seat: <b id="seatmodel">&hellip;</b><br><span id="seatsub"></span></div>
<button class="ghost" id="theme" title="toggle light/dark">&#9790;</button>
<button class="ghost" id="palbtn" title="command palette (Ctrl/&#8984;+K)">&#8984;K</button>
</header>
<nav>
<button data-t="overview" class="on">Overview</button>
<button data-t="chat">Chat</button>
<button data-t="terminal">Terminal</button>
<button data-t="desktop">Desktop</button>
<button data-t="apps">Apps</button>
<button data-t="settings">Settings</button>
</nav>
<main>
<section id="overview" class="tab on">
<div class="tiles">
<div class="tile"><div class="k">SEAT MODEL</div><div class="v" id="t-model" style="font-size:14px">&hellip;</div></div>
<div class="tile"><div class="k">UPTIME</div><div class="v" id="t-up">&hellip;</div></div>
<div class="tile"><div class="k">CHAT</div><div class="v" id="t-chat">&hellip;</div><div class="bar"><i id="t-chatbar"></i></div></div>
<div class="tile"><div class="k">CIRCUIT TOKENS</div><div class="v" id="t-tok">&hellip;</div><div class="bar"><i id="t-tokbar"></i></div></div>
<div class="tile"><div class="k">GUEST LINUX</div><div class="v" id="t-guest">&hellip;</div></div>
</div>
<div class="card" style="margin-bottom:12px"><h3>ACTIVITY &mdash; LIVE</h3>
<div id="nowrows" style="font-size:15px;line-height:1.65;margin:2px 0 10px">
<div><b>chat:</b> <span id="now-chat">&hellip;</span></div>
<div><b>computer in the weights:</b> <span id="now-circuit">&hellip;</span></div>
<div><b>guest linux:</b> <span id="now-guest">&hellip;</span></div>
</div>
<canvas id="viz" width="900" height="250"></canvas>
<p class="dim" id="vizlegend">warming up&hellip;</p>
<p class="dim">left: the brain thinking (amber). right: the computer
in the weights (blue). pulses are real work &mdash; chat tokens
flowing, and prompts crossing the mailbox between brain and
computer. nothing here is decoration: when nothing moves, nothing
happened.</p></div>
<div class="cards">
<div class="card"><h3>THE BRAIN</h3>
<p id="ov-model">&hellip;</p>
<p class="dim">the built-in Qwen 3.5 4B weights, served as the seat.
the chat tab talks to this. the computer in the weights asks it
questions the same way.</p></div>
<div class="card"><h3>THE COMPUTER IN THE WEIGHTS</h3>
<p>stock circuit agent: <span id="ov-stock" class="badge">?</span></p>
<p>grown 2 MB circuit: <span id="ov-grown" class="badge">?</span></p>
<pre id="ov-grownlog"></pre></div>
<div class="card"><h3>GUEST AGENT (LINUX SIDE)</h3>
<p class="dim">in the terminal tab, the guest has its own agent that
works through the weights:</p>
<p><code>hermes-agent "your task here"</code></p>
<p class="dim">quotes, no angle brackets &mdash; the shell eats those.
it can use the whole Linux side: files, the notebook at
<code>/root/nb</code>, exact arithmetic.</p></div>
<div class="card"><h3>LIVE TERMINAL</h3>
<p id="ov-term">&hellip;</p>
<p class="dim">Qwen-Linux on the fast core &mdash; full guest shell,
<code>ask "a question"</code> talks to the weights.</p></div>
</div></section>
<section id="chat" class="tab">
<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px">
<button class="chip" data-q="Explain what you are, in one short paragraph.">what are you?</button>
<button class="chip" data-q="What is the computer inside you, and what has it computed so far?">the computer inside you</button>
<button class="chip" data-q="What is the capital of France?">a plain question</button>
</div>
<div id="msgs"></div>
<div id="chatbar">
<textarea id="inp" rows="2" placeholder="talk to the built-in weights&hellip;"></textarea>
<button class="act" id="send">Send</button>
<button class="ghost" id="stop" style="display:none">Stop</button>
<button class="ghost" id="clear">Clear</button>
</div>
<p class="dim" id="chatstat" style="font-size:12px;min-height:16px"></p>
<p class="dim" style="color:var(--dim);font-size:12px">Enter sends,
Shift+Enter makes a new line. This chat is with the same 4B brain the
in-weights computer consults.</p>
</section>
<section id="terminal" class="tab">
<div class="toolbar"><span id="termstate" class="dim">terminal loads when this tab opens</span>
<span class="sp"></span>
<button class="btnsm" id="termopen">open in new window</button>
<button class="btnsm" id="termreload">reconnect</button></div>
<iframe id="term" title="Qwen-Linux terminal"></iframe>
</section>
<section id="desktop" class="tab"><div class="cards">
<div class="card"><h3>DESKTOP &mdash; NOT BUILT YET</h3>
<p>the display path is proven: Doom drew into guest memory and the
host showed it, with keys going back in. a framebuffer shell &mdash;
a screen window in guest memory, a pointer, simple windows &mdash;
is the planned spike, and it will live in this tab.</p>
<p class="dim">realistic on the fast core. on the in-weights circuit
the same trick works mechanically (the mailbox proved host access to
circuit memory) but at ~17 instructions/second a desktop is a
screenshot an hour, not a desktop.</p></div>
</div></section>
<section id="apps" class="tab">
<p style="color:var(--dim)">only what's on the Qwenputer itself.
most of these run in the guest &mdash; tap one to open the terminal.</p>
<div class="cards" id="apps"></div></section>
<section id="settings" class="tab">
<label>backend URL (the seat's OpenAI-compatible endpoint, on this machine)</label>
<input type="text" id="s-backend">
<label>model (auto = the qwen3.5-4b already loaded; or an exact model id)</label>
<input type="text" id="s-model">
<label>temperature</label>
<input type="number" id="s-temp" step="0.1" min="0" max="2">
<label>max tokens per reply (the coder brain thinks first &mdash; 4096 is the working floor for tool talk)</label>
<input type="number" id="s-tokens" step="256" min="256">
<label>chat system prompt</label>
<textarea id="s-sys" rows="4"></textarea>
<p><button class="act" id="save" style="padding:10px 22px">Save settings</button>
<button class="ghost" id="testbe">Test backend</button>
<span id="saved" style="color:var(--ok)"></span></p>
<p class="dim" id="resolved"></p>
</section>
</main>
<footer><span>Qwenputer dashboard</span><span id="footver"></span>
<span class="r" id="footup"></span></footer>
<div id="pal"><div id="palbox"><input id="palin" placeholder="type a command or search&hellip;" autocomplete="off">
<div id="palopts"></div></div></div>
<script>
const $=id=>document.getElementById(id);
try{if(localStorage.qtheme==='light')document.documentElement.dataset.theme='light';}catch(e){}
$('theme').onclick=()=>{const d=document.documentElement;
 d.dataset.theme=d.dataset.theme==='light'?'':'light';
 try{localStorage.qtheme=d.dataset.theme||'dark';}catch(e){}};
function esc(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function md(src){
 let out=esc(src),code=[];
 out=out.replace(/```(\w*)\n([\s\S]*?)```/g,(m,l,c)=>{code.push('<pre><code>'+c+'</code></pre>');return '\u0000'+(code.length-1)+'\u0000';});
 const lines=out.split('\n'),html=[];let list=null;
 const inline=s=>s.replace(/`([^`]+)`/g,'<code>$1</code>')
  .replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>')
  .replace(/(^|\W)\*([^*\n]+)\*/g,'$1<em>$2</em>');
 for(const ln of lines){
  let m;
  if(/^\u0000\d+\u0000$/.test(ln)){if(list){html.push('</'+list+'>');list=null;}html.push(code[+ln.slice(1,-1)]);continue;}
  if(m=ln.match(/^(#{1,3})\s+(.*)/)){if(list){html.push('</'+list+'>');list=null;}
   html.push('<h3>'+inline(m[2])+'</h3>');continue;}
  if(m=ln.match(/^&gt;\s?(.*)/)){if(list){html.push('</'+list+'>');list=null;}
   html.push('<blockquote>'+inline(m[1])+'</blockquote>');continue;}
  if(m=ln.match(/^(\s*)[-*]\s+(.*)/)){if(list!=='ul'){if(list)html.push('</'+list+'>');html.push('<ul>');list='ul';}
   html.push('<li>'+inline(m[2])+'</li>');continue;}
  if(m=ln.match(/^(\s*)\d+[.)]\s+(.*)/)){if(list!=='ol'){if(list)html.push('</'+list+'>');html.push('<ol>');list='ol';}
   html.push('<li>'+inline(m[2])+'</li>');continue;}
  if(/^\s*$/.test(ln)){if(list){html.push('</'+list+'>');list=null;}continue;}
  if(/^\|(.+)\|$/.test(ln)){const cells=ln.slice(1,-1).split('|').map(c=>inline(c.trim()));
   if(cells.every(c=>/^:?-{3,}:?$/.test(c.replace(/<[^>]*>/g,''))))continue;
   html.push('<table><tr>'+cells.map(c=>'<td>'+c+'</td>').join('')+'</tr></table>');continue;}
  html.push('<p>'+inline(ln)+'</p>');
 }
 if(list)html.push('</'+list+'>');
 return html.join('');
}
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{
 document.querySelectorAll('nav button').forEach(x=>x.classList.remove('on'));
 document.querySelectorAll('.tab').forEach(x=>x.classList.remove('on'));
 b.classList.add('on');$(b.dataset.t).classList.add('on');
 if(b.dataset.t==='terminal'&&!$('term').src)
   $('term').src='http://'+location.hostname+':8951/';
});
const APPS=[['\u25b8','live terminal','the Qwen-Linux guest shell itself — everything below runs here','Terminal tab',1],
['?', 'ask','ask the built-in weights a question; the answer is typed back into the guest','ask "your question"',1],
['\u25c9','hermes-agent','the guest agent — works through the weights with the whole Linux side: files, exact arithmetic, the notebook','hermes-agent "your task"',1],
['\u2595','notebook','the guest\u2019s persistent memory at /root/nb — the agent writes plans and facts there and reads them back','ls /root/nb',1],
['\u03bb','MicroPython','Python running on the Qwenputer CPU (the upstream port)','on the circuit',0],
['\u25c8','bare-metal agent','agent.c — the harness compiled for the circuit itself: store / recall / mul through the mailbox in RAM','on the circuit',0],
['\u25e9','Doom','the Doom port for the Qwenputer CPU, grown memory map build','on the CPU port',0]];
$('apps').innerHTML=APPS.map((a,i)=>'<div class="app"><div><span class="ico">'
 +a[0]+'</span><b>'+a[1]+'</b></div><span>'+a[2]+'</span><span><code>'
 +a[3]+'</code></span><div class="row">'
 +(a[4]?'<button class="btnsm" data-term="'+i+'">open in terminal</button><button class="btnsm" data-copy="'+i+'">copy command</button>':'<span class="dim">runs on the compiled circuit, not in the guest</span>')
 +'</div></div>').join('');
function goTerm(){document.querySelector('nav button[data-t="terminal"]').click();}
document.querySelectorAll('[data-term]').forEach(el=>el.onclick=goTerm);
document.querySelectorAll('[data-copy]').forEach(el=>el.onclick=()=>{
 const cmd=APPS[+el.dataset.copy][3];
 if(navigator.clipboard)navigator.clipboard.writeText(cmd);
 el.textContent='copied';setTimeout(()=>el.textContent='copy command',1500);});
async function status(){
 try{const r=await fetch('/api/status');const j=await r.json();
  $('seatmodel').textContent=j.seat.model||'unreachable';
  $('seatsub').textContent=j.seat.error?j.seat.error:('via '+j.seat.backend);
  $('ov-model').textContent=j.seat.model||('unreachable: '+(j.seat.error||''));
  const st=$('ov-stock');st.textContent=j.stock.pass?'PASS':'no verdict';
  st.className='badge '+(j.stock.pass?'ok':'bad');
  const gr=$('ov-grown');
  gr.textContent=j.grown.pass?'PASS':(j.grown.fail?'FAIL - finished':(j.grown.running?'running':'not running'));
  gr.className='badge '+(j.grown.pass?'ok':'');
  $('ov-grownlog').textContent=(j.grown.log_tail||'')+(j.grown.out_tail||'');
  $('ov-grownlog').scrollTop=1e9;
  $('ov-term').innerHTML=j.terminal_up?
   '<span class="badge ok">up</span> &mdash; open the Terminal tab':
   '<span class="badge bad">down</span>';
  $('t-model').textContent=(j.seat.model||'unreachable').split('/').pop();
  const up=j.uptime_s||0,hh=Math.floor(up/3600),mm=Math.floor(up%3600/60);
  $('t-up').innerHTML=hh+'<small>h</small> '+mm+'<small>m</small>';
  $('t-guest').innerHTML=j.terminal_up?'<span class="dot ok"></span>up':'<span class="dot bad"></span>down';
  $('footver').textContent='build '+(j.version||'unknown')+' · '+j.time;
  $('footup').textContent='dashboard up '+hh+'h '+mm+'m';
  $('termstate').textContent=j.terminal_up?'guest terminal is up':'guest terminal is down — start liveterm (:8951)';
  $('resolved').textContent='resolved model: '+(j.seat.model||('unreachable — '+(j.seat.error||'')));
 }catch(e){}
}
status();setInterval(status,10000);
const hist=[];
function bubble(role,text,rich){
 const d=document.createElement('div');d.className='msg '+(role==='user'?'u':'a');
 d.innerHTML='<div class="who">'+(role==='user'?'you':'qwen 3.5 4b')+'</div>';
 const body=document.createElement('div');body.className='body';
 if(rich)body.innerHTML=md(text||'');else body.textContent=text||'';
 d.appendChild(body);$('msgs').appendChild(d);$('msgs').scrollTop=1e9;
 return body;
}
let AC=null,ntok=0,t0=0;
async function send(){
 const text=$('inp').value.trim();if(!text||AC)return;
 $('inp').value='';$('send').disabled=true;$('stop').style.display='';
 AC=new AbortController();ntok=0;t0=Date.now();
 bubble('user',text);hist.push({role:'user',content:text});
 const body=bubble('assistant','');
 let think=document.createElement('div');think.className='think';
 think.style.display='none';body.parentNode.insertBefore(think,body);
 let content='',reasoning='';
 const stat=()=>{const el=(Date.now()-t0)/1000;
  $('chatstat').textContent='working\u2026 '+Math.round(el)+'s \u00b7 '+ntok+
  ' tokens so far'+(el>2&&ntok?' \u00b7 '+(ntok/el).toFixed(1)+' tok/s':'');};
 stat();const iv=setInterval(stat,500);
 try{
  const r=await fetch('/api/chat',{method:'POST',signal:AC.signal,
   headers:{'Content-Type':'application/json'},
   body:JSON.stringify({messages:hist})});
  const rd=r.body.getReader(),dec=new TextDecoder();let buf='';let fin=false;
  while(true){const{value,done}=await rd.read();if(done)break;
   buf+=dec.decode(value,{stream:true});
   let i;while((i=buf.indexOf('\n\n'))>=0){
    const ev=buf.slice(0,i);buf=buf.slice(i+2);
    if(!ev.startsWith('data:'))continue;
    const j=JSON.parse(ev.slice(5));
    if(j.delta){content+=j.delta;body.textContent=content;ntok++;
     if(window.vizPulse)vizPulse(1);}
    if(j.reasoning){reasoning+=j.reasoning;think.style.display='block';
     think.textContent=reasoning;ntok++;}
    if(j.error){body.textContent='[error] '+j.error;}
    if(j.retract){content='';body.textContent='';}
    if(j.status){$('chatstat').textContent=j.status;
     if(window.vizPulse)vizPulse(6,true);}
    if(j.done){fin=true;}
    $('msgs').scrollTop=1e9;
   }
   if(fin){try{await rd.cancel();}catch(e){}break;}
  }
 }catch(e){if(e.name!=='AbortError')body.textContent='[connection error] '+e;
  else if(!content)body.textContent='(stopped)';}
 clearInterval(iv);
 if(!content&&reasoning){content=reasoning;}
 if(content){body.innerHTML=md(content);hist.push({role:'assistant',content});}
 const dt=(Date.now()-t0)/1000;
 $('chatstat').textContent='done \u00b7 '+Math.round(dt)+'s \u00b7 '+ntok+
  ' tokens'+(dt>0&&ntok?' \u00b7 '+(ntok/dt).toFixed(1)+' tok/s':'');
 AC=null;$('send').disabled=false;$('stop').style.display='none';
 $('inp').focus();
}
$('send').onclick=send;
$('stop').onclick=()=>{if(AC)AC.abort();};
$('clear').onclick=()=>{hist.length=0;$('msgs').innerHTML='';
 $('chatstat').textContent='';};
document.querySelectorAll('.chip').forEach(c=>c.onclick=()=>{
 $('inp').value=c.dataset.q;send();});
$('inp').addEventListener('keydown',e=>{
 if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send();}});
const viz={pulses:[],lastPrompts:-1};
const vctx=$('viz').getContext('2d');
const brainPts=[],chipPts=[];
for(let i=0;i<110;i++){const y=1-(i/109)*2,r=Math.sqrt(1-y*y),t=i*2.39996;
 brainPts.push([Math.cos(t)*r,y,Math.sin(t)*r]);}
for(let x=0;x<8;x++)for(let y=0;y<6;y++)for(let z=0;z<4;z++)
 chipPts.push([x-3.5,y-2.5,z-1.5]);
let rot=0;
function proj(p,cx,cy,s){const c=Math.cos(rot),sn=Math.sin(rot);
 const x=p[0]*c-p[2]*sn,z=p[0]*sn+p[2]*c,y=p[1];
 const d=340/(340+z*40);return[cx+x*s*d,cy+y*s*d,d];}
function vizPulse(n,big){for(let i=0;i<n;i++){if(viz.pulses.length>240)return;
 viz.pulses.push({t:0,sp:.0035+Math.random()*.004,
  a:brainPts[Math.random()*brainPts.length|0],
  b:chipPts[Math.random()*chipPts.length|0],big:!!big});}}
window.vizPulse=vizPulse;
function drawViz(){rot+=.0035;vctx.clearRect(0,0,900,250);
 const draw=(pts,cx,cy,s,col)=>{const out=[];
  for(const p of pts){const q=proj(p,cx,cy,s);out.push(q);
   vctx.globalAlpha=.25+.55*q[2]/1.6;vctx.fillStyle=col;
   vctx.beginPath();vctx.arc(q[0],q[1],1.6*q[2],0,7);vctx.fill();}
  vctx.globalAlpha=1;return out;};
 draw(brainPts,225,118,64,'#e8b93e');
 draw(chipPts,665,118,15,'#57c7ff');
 vctx.fillStyle='#9aa7b5';vctx.font='12px sans-serif';
 vctx.fillText('BRAIN \u2014 qwen 3.5 4b, the seat',140,240);
 vctx.fillText('COMPUTER \u2014 the compiled chip',540,240);
 viz.pulses=viz.pulses.filter(p=>p.t<=1);
 for(const p of viz.pulses){p.t+=p.sp;
  const a=proj(p.a,225,118,64),b=proj(p.b,665,118,15);
  const x=a[0]+(b[0]-a[0])*p.t,
        y=a[1]+(b[1]-a[1])*p.t-Math.sin(p.t*3.14)*36;
  vctx.globalAlpha=.9;vctx.fillStyle=p.big?'#ffffff':'#ffd479';
  vctx.beginPath();vctx.arc(x,y,p.big?3:2,0,7);vctx.fill();
  vctx.globalAlpha=1;}
 requestAnimationFrame(drawViz);}
drawViz();
async function activity(){
 try{const r=await fetch('/api/activity');const j=await r.json();
  const s=j.chat_series,rate=s.slice(20).reduce((a,b)=>a+b,0)/20;
  if(j.chat_open>0&&rate>0)vizPulse(Math.min(3,Math.ceil(rate/6)));
  if(j.grown.running&&j.grown.tokens>0)vizPulse(1);
  if(viz.lastPrompts>=0&&j.grown.prompts>viz.lastPrompts)vizPulse(6,true);
  viz.lastPrompts=j.grown.prompts;
  $('now-chat').textContent=j.chat_open?
   'answering right now':'idle - nobody is chatting';
  $('now-circuit').textContent=j.grown.pass?
   'the agent run on the grown circuit passed':
   (j.grown.fail?
    'the grown-circuit agent run finished and failed - it hit its round limit on the old brain':
   (j.grown.running?
    ('running the agent - '+j.grown.prompts+
     ' questions answered so far, at token '+
     j.grown.tokens.toLocaleString()):'not running'));
  $('now-guest').textContent=j.terminal_up?
   'up - the agent shell is sitting on its console':'down';
  $('t-chat').innerHTML=j.chat_open?'<span class="dot ok"></span>live':'idle';
  $('t-chatbar').style.width=Math.min(100,rate*4)+'%';
  $('t-tok').textContent=(j.grown.tokens||0).toLocaleString();
  const tp=j.grown.prompts||0;
  $('t-tokbar').style.width=Math.min(100,tp*10)+'%';
  $('t-tok').title=tp+' prompts answered';
  $('vizlegend').textContent='the dots are real events: a dot '
   +'crosses when the chat streams words, and a white flash '
   +'crosses when the computer in the weights answers a '
   +'question. slow dots, on purpose - the circuit is slow.';
 }catch(e){}
 setTimeout(activity,1000);}
activity();
async function loadSettings(){
 const r=await fetch('/api/settings');const j=await r.json();
 $('s-backend').value=j.backend_url;$('s-model').value=j.model;
 $('s-temp').value=j.temperature;$('s-tokens').value=j.max_tokens;
 $('s-sys').value=j.system_prompt;
}
loadSettings();
$('save').onclick=async()=>{
 const body={backend_url:$('s-backend').value,model:$('s-model').value,
  temperature:parseFloat($('s-temp').value),
  max_tokens:parseInt($('s-tokens').value),system_prompt:$('s-sys').value};
 await fetch('/api/settings',{method:'POST',
  headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
 $('saved').textContent='saved';setTimeout(()=>$('saved').textContent='',2500);
 status();
};
$('testbe').onclick=async()=>{
 $('saved').textContent='testing\u2026';
 try{const r=await fetch('/api/status');const j=await r.json();
  $('saved').textContent=j.seat.model?('ok: '+j.seat.model):('FAILED: '+(j.seat.error||'no model'));
 }catch(e){$('saved').textContent='FAILED: '+e;}
 setTimeout(()=>{if($('saved').textContent.startsWith('ok'))$('saved').textContent='';},4000);
};
function termURL(){return 'http://'+location.hostname+':8951/';}
$('termopen').onclick=()=>window.open(termURL(),'_blank');
$('termreload').onclick=()=>{$('term').src=termURL();};
const PAL=[
 ['Go: Overview','show the status dashboard',()=>document.querySelector('nav button[data-t="overview"]').click()],
 ['Go: Chat','talk to the built-in weights',()=>document.querySelector('nav button[data-t="chat"]').click()],
 ['Go: Terminal','the live Qwen-Linux guest',()=>document.querySelector('nav button[data-t="terminal"]').click()],
 ['Go: Desktop','framebuffer spike placeholder',()=>document.querySelector('nav button[data-t="desktop"]').click()],
 ['Go: Apps','programs on the Qwenputer',()=>document.querySelector('nav button[data-t="apps"]').click()],
 ['Go: Settings','backend, model, prompt',()=>document.querySelector('nav button[data-t="settings"]').click()],
 ['Toggle theme','dark / light',()=>$('theme').click()],
 ['Chat: what are you?','preset question',()=>{document.querySelector('nav button[data-t="chat"]').click();$('inp').value='Explain what you are, in one short paragraph.';send();}],
 ['Chat: the computer inside you','preset question',()=>{document.querySelector('nav button[data-t="chat"]').click();$('inp').value='What is the computer inside you, and what has it computed so far?';send();}],
 ['Copy: ask command','ask "your question"',()=>{if(navigator.clipboard)navigator.clipboard.writeText('ask "your question"');}],
 ['Copy: hermes-agent command','hermes-agent "your task"',()=>{if(navigator.clipboard)navigator.clipboard.writeText('hermes-agent "your task"');}],
];
let palSel=0;
function palRender(){
 const q=$('palin').value.toLowerCase();
 const hits=PAL.map((p,i)=>[p,i]).filter(([p])=>p[0].toLowerCase().includes(q)||p[1].toLowerCase().includes(q));
 palSel=Math.min(palSel,Math.max(0,hits.length-1));
 $('palopts').innerHTML=hits.map(([p,i],k)=>'<div class="palopt'+(k===palSel?' sel':'')+'" data-i="'+i+'">'+p[0]+'<small>'+p[1]+'</small></div>').join('')||'<div class="palopt">no match</div>';
 document.querySelectorAll('.palopt[data-i]').forEach(el=>{
  el.onclick=()=>{palClose();PAL[+el.dataset.i][2]();};
  el.onmousemove=()=>{palSel=[...document.querySelectorAll('.palopt[data-i]')].indexOf(el);palRender();};});
 palHits=hits;
}
let palHits=[];
function palOpen(){$('pal').classList.add('on');$('palin').value='';palSel=0;palRender();setTimeout(()=>$('palin').focus(),0);}
function palClose(){$('pal').classList.remove('on');}
$('palbtn').onclick=palOpen;
$('pal').onclick=e=>{if(e.target===$('pal'))palClose();};
$('palin').oninput=()=>{palSel=0;palRender();};
$('palin').onkeydown=e=>{
 if(e.key==='Escape')palClose();
 else if(e.key==='ArrowDown'){e.preventDefault();palSel=Math.min(palSel+1,palHits.length-1);palRender();}
 else if(e.key==='ArrowUp'){e.preventDefault();palSel=Math.max(palSel-1,0);palRender();}
 else if(e.key==='Enter'&&palHits[palSel]){palClose();palHits[palSel][0][2]();}
};
document.addEventListener('keydown',e=>{
 if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='k'){e.preventDefault();$('pal').classList.contains('on')?palClose():palOpen();}
});
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/?"):
            data = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif self.path == "/api/status":
            self._json(get_status())
        elif self.path == "/api/activity":
            self._json(get_activity())
        elif self.path == "/api/settings":
            self._json(load_settings())
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except Exception:
            body = {}
        if self.path == "/api/settings":
            try:
                self._json(save_settings(body))
            except Exception as e:
                self._json({"error": str(e)[:160]}, 400)
        elif self.path == "/api/chat":
            self._chat(body)
        else:
            self._json({"error": "not found"}, 404)

    def _chat(self, body):
        msgs = body.get("messages") or []
        s = load_settings()
        model, err = resolve_model(s)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        def send(obj):
            if "delta" in obj or "reasoning" in obj:
                _act_note(1)
            self.wfile.write(("data: " + json.dumps(obj) + "\n\n").encode())
            self.wfile.flush()

        if err or not model:
            send({"error": "brain unreachable: %s" % (err or "no model")})
            return
        _CHAT_OPEN[0] += 1
        try:
            out = []
            sysp = (s.get("system_prompt") or "").strip()
            out.append({"role": "system",
                        "content": (sysp + "\n" + TASK_HINT).strip()})
            out.extend(msgs)
            # chat-time computer use (Duckets, 2026-10-31): the model
            # may answer a message with a TASK: line instead of text.
            # That line is typed into the live terminal guest - the
            # same console the user sees - where the agentshell runs
            # it as a hermes-agent session; the session's AGENT_ANSWER
            # comes back here as the computer's report and the model
            # then answers the user from it. Max 3 delegations.
            seen_tasks = set()
            for _round in range(4):
                content = self._stream_round(out, s, model, send)
                # Delegation fires ONLY when the reply LEADS with
                # the TASK: line, the way the model is taught. An
                # earlier build scanned for TASK: anywhere in the
                # text; the model also MENTIONS the mechanism when
                # explaining itself ("just ask! TASK: ..."), and
                # each mention silently retracted the user's answer
                # and burned a guest wait + more generation rounds
                # - Duckets' "chat sits working while the brain
                # runs" report, 2026-10-31. A mention is prose, not
                # a handover.
                task = None
                lead = content.strip()
                if lead.startswith("TASK:"):
                    task = lead.splitlines()[0][5:].strip()
                if not task:
                    break
                if _round == 3 or task.lower() in seen_tasks:
                    # out of delegations, or the same job handed over
                    # twice (seen live 2026-10-31: the model led its
                    # final answer with the raw TASK: line): never show
                    # the user the delegation line - retract what
                    # streamed and re-emit it with TASK: segments cut.
                    send({"retract": True})
                    kept = []
                    for ln in content.splitlines():
                        i = ln.find("TASK:")
                        ln = (ln[:i].rstrip() if i >= 0 else ln)
                        if ln.strip():
                            kept.append(ln)
                    cleaned = "\n".join(kept).strip()
                    send({"delta": cleaned or
                          "(no answer - the model repeated its request "
                          "to the computer instead of replying)"})
                    break
                seen_tasks.add(task.lower())
                send({"retract": True})
                send({"status": "asking the computer: " + task[:90]})
                report = _guest_task(task, send)
                if report is None:
                    report = ("(the computer is offline - the live "
                              "terminal is not running)")
                out.append({"role": "assistant", "content": content})
                out.append({"role": "user",
                            "content": "The computer reports: " + report +
                                       " Answer the user from this "
                                       "report now."})
            send({"done": True})
        except Exception as e:
            try:
                send({"error": str(e)[:200]})
            except Exception:
                pass
        finally:
            _CHAT_OPEN[0] -= 1
            # the page ends its "working" state when the stream
            # closes; say so explicitly at the socket level too
            self.close_connection = True

    def _stream_round(self, out, s, model, send):
        """One streamed completion over the message list; returns the
        full content text (deltas also go to the page as they come)."""
        payload = {"model": model, "messages": out,
                   "temperature": float(s["temperature"]),
                   "max_tokens": int(s["max_tokens"]), "stream": True}
        req = urllib.request.Request(
            s["backend_url"].rstrip("/") + "/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        content = ""
        with urllib.request.urlopen(req, timeout=1200) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except Exception:
                    continue
                ch = (chunk.get("choices") or [{}])[0]
                delta = ch.get("delta") or {}
                if delta.get("content"):
                    content += delta["content"]
                    send({"delta": delta["content"]})
                # separate if, not elif: one delta can carry both
                # content and reasoning_content, and elif dropped the
                # thinking half of such chunks.
                if delta.get("reasoning_content"):
                    send({"reasoning": delta["reasoning_content"]})
        return content


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8952)
    ap.add_argument("--host", default="0.0.0.0")
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    srv.daemon_threads = True
    print("qwenputer dashboard on %s:%d" % (a.host, a.port), flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
