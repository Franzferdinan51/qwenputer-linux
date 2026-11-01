#!/usr/bin/env python3
"""Qwenputer dashboard - the interaction interface for Qwenputer-Linux.

One stdlib-only host server, one page, tabs:

  Overview  - seat model, circuit runs, terminal health
  Chat      - streaming chat with the built-in-weights brain; the seat is an
              OpenAI-compatible endpoint on this machine (Settings picks it),
              resolved by the same builtin rule as qwenbackend: the
              qwen3.5-4b family member already loaded, never a fresh load
  Terminal  - embeds the live guest terminal (liveterm, :8951) untouched
  Desktop   - honest placeholder: no framebuffer build exists in this
              tree (the Doom proof lives on the circuit-grown-map
              branch / upstream); what a spike needs is listed here
  Apps      - the truth about programs: apps the agent actually built
              (live from the mind mirror), its skills and jobs, the
              guest entry points, and what runs upstream instead
  System    - the weights-machine's own CPU/RAM (guest /proc, not the
              host PC), engine step rate, circuit token rates, queue
  Settings  - backend URL, model, temperature, token budget, system
              prompt, guest delegation switch + budget; mind actions
              (snapshot, trace export, upkeep pause); persisted next
              to the rig

Run:  python3 dashboard.py [--port 8952]
"""

import argparse
import collections
import json
import os
import re
import socket
import subprocess
import threading
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
    "guest_enabled": True,
    "guest_budget": 600,
    "decider_enabled": False,
    "decider_url": "http://127.0.0.1:1234",
    "decider_model": "",
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
    s["guest_enabled"] = bool(s.get("guest_enabled", True))
    s["guest_budget"] = max(60, min(3600, int(s.get("guest_budget", 600))))
    s["decider_enabled"] = bool(s.get("decider_enabled", False))
    with open(SETTINGS_PATH, "w") as f:
        json.dump(s, f, indent=1)
    return s


def resolve_model(s):
    """Builtin rule (mirrors qwenbackend.builtin_backend): the qwen3.5-4b
    family member already listed by the backend (QWEN_FAMILY overrides
    the family substring for fused/GGUF builds under a new name); a
    pristine base id wins over fine-tune tags; never loads anything,
    and never silently serves a different model when the wanted one is
    missing (the page already renders a None model as unreachable)."""
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
    family = (os.environ.get("QWEN_FAMILY") or "qwen3.5-4b").lower()
    fam = [i for i in ids if family in i.lower()]
    if fam:
        fam.sort(key=lambda i: ("3d-coder" in i.lower(), i))
        return fam[0], None
    return None, "no %s model loaded (have: %s)" % (
        family, ", ".join(ids)[:120])


def resolve_decider(s):
    """Opt-in fast decision model (a sidecar seat, e.g. a d1 build
    served by LM Studio). Off by default so fused/GGUF single-model
    setups never depend on it; when off, no probe runs at all. When
    on, the id must be exact and loaded - never a silent fallback."""
    if not s.get("decider_enabled"):
        return None, "off"
    want = (s.get("decider_model") or "").strip()
    if not want:
        return None, "enabled but no model id set"
    try:
        url = s["decider_url"].rstrip("/") + "/v1/models"
        with urllib.request.urlopen(url, timeout=5) as r:
            ids = [d.get("id") for d in json.load(r).get("data", [])
                   if d.get("id")]
    except Exception as e:
        return None, str(e)[:160]
    if want in ids:
        return want, None
    return None, "model %r is not loaded (have: %s)" % (
        want, ", ".join(ids)[:120])


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
    dmodel, derr = resolve_decider(s)
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
        "decider": {"enabled": bool(s.get("decider_enabled")),
                    "model": dmodel, "backend": s["decider_url"],
                    "error": derr},
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


# --- guest delegation queue (2026-11-01: the computer said it was
# busy, so requests queue instead of being refused) ---
# The computer is one console and it is slow; several chat turns (or
# a chat turn plus the user typing in the Terminal tab) contend for
# it. Delegations therefore line up in a strict FIFO instead of
# being refused: a waiter sends "queued #n" status until its turn,
# then waits for the agentshell prompt, hands over, and reads the
# report back. _GUEST_HANDS/_GUEST_DONES are monotonic counters for
# the activity view (handover pulses brain -> computer, report
# pulses computer -> brain).
_GUEST_COND = threading.Condition()
_GUEST_QUEUE = collections.deque()  # FIFO of opaque tickets
_GUEST_BUSY = [False]               # a delegation owns the console
_GUEST_HANDS = [0]                  # jobs handed over, all time
_GUEST_DONES = [0]                  # reports back, all time
_GUEST_TASK = [""]                  # task text currently in flight


def _guest_acquire(send, budget=600):
    """Take a FIFO ticket and block until this delegation owns the
    computer. True when the slot is ours (caller must _guest_release
    it); False when the budget ran out (ticket dropped)."""
    ticket = object()
    start = time.time()
    with _GUEST_COND:
        _GUEST_QUEUE.append(ticket)
        try:
            while True:
                try:
                    pos = _GUEST_QUEUE.index(ticket)
                except ValueError:
                    return False
                if pos == 0 and not _GUEST_BUSY[0]:
                    _GUEST_QUEUE.popleft()
                    _GUEST_BUSY[0] = True
                    return True
                remaining = budget - (time.time() - start)
                if remaining <= 0:
                    _GUEST_QUEUE.remove(ticket)
                    _GUEST_COND.notify_all()
                    return False
                if pos > 0:
                    send({"status":
                          "the computer is busy - queued #%d" % (pos + 1)})
                else:
                    send({"status": "next in line - the computer is "
                                    "finishing its current job"})
                _GUEST_COND.wait(timeout=min(15, remaining))
        except Exception:
            try:
                _GUEST_QUEUE.remove(ticket)
            except ValueError:
                pass
            _GUEST_COND.notify_all()
            raise


def _guest_release():
    with _GUEST_COND:
        _GUEST_BUSY[0] = False
        _GUEST_COND.notify_all()


def _guest_task(task, send):
    """Run one hermes-agent session in the live terminal guest and
    return its AGENT_ANSWER text; None when the terminal is offline.

    The task is typed at the guest's own console (the agentshell
    runs it in the foreground, exactly as if the user typed it -
    the session is visible in the Terminal tab). When the computer
    is busy the job waits its turn in the FIFO queue instead of
    being refused; only an offline terminal fails fast."""
    s = load_settings()
    if not s.get("guest_enabled", True):
        return ("(guest delegation is switched off in Settings - the "
                "job was NOT handed over. Answer from chat alone.)")
    try:
        budget = max(60, min(3600, int(s.get("guest_budget", 600))))
    except Exception:
        budget = 600
    base = "http://127.0.0.1:8951"
    try:
        with urllib.request.urlopen(base + "/api/out?since=0",
                                    timeout=10) as r:
            snap = json.loads(r.read())
            mark = snap["until"]
    except Exception:
        return None
    if not _guest_acquire(send, budget=budget):
        return ("(the computer stayed busy for %d minutes - the job "
                "was NOT done. Tell the user the computer is still "
                "busy and to try again shortly.)" % (budget // 60))
    try:
        # Readiness wait (was an instant gate 2026-10-31; a live
        # report showed chat sitting idle while the brain ran): hand
        # over ONLY when the terminal is actually sitting at the
        # agentshell prompt. Typing a job into a console that is
        # still installing, mid-session, or at a plain shell queues
        # it behind whatever owns the console and the chat then
        # waits while the brain burns further rounds. Now the wait
        # is explicit and statused instead of a refusal.
        send({"status": "our turn - waiting for the computer's prompt…"})
        waited_q = 0
        errors = 0
        while True:
            try:
                with urllib.request.urlopen(base + "/api/out?since=0",
                                            timeout=10) as r:
                    snap = json.loads(r.read())
                errors = 0
            except Exception:
                errors += 1
                if errors >= 3:
                    return None
                time.sleep(5)
                continue
            tail = snap["data"][-400:]
            if "ASK_READY" in snap["data"] and \
                    tail.rstrip().endswith("agent>"):
                mark = snap["until"]
                break
            time.sleep(5)
            waited_q += 5
            if waited_q % 15 == 0:
                send({"status": "waiting for the computer's prompt… %ds"
                               % waited_q})
            if waited_q >= budget:
                return ("(the computer stayed busy for %d minutes - the job "
                        "was NOT done. Tell the user the computer is still "
                        "busy and to try again shortly.)" % (budget // 60))
        line = 'hermes-agent "' + task.replace('"', "'") + '"\n'
        try:
            req = urllib.request.Request(
                base + "/api/in", data=json.dumps({"data": line}).encode(),
                headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=10).read()
        except Exception:
            return None
        _GUEST_HANDS[0] += 1
        _GUEST_TASK[0] = task[:90]
        # budget wall seconds (default 600): a 15-round session at
        # ~20-40 s per model round-trip needs it; the old 120 s cut
        # real sessions off mid-answer (seen live: the chat gave up
        # while the guest was still working through its rounds).
        deadline = time.time() + budget
        waited = 0
        try:
            while time.time() < deadline:
                time.sleep(3)
                waited += 3
                if waited % 15 == 0:
                    send({"status": "the computer is working… %ds" % waited})
                try:
                    with urllib.request.urlopen(
                            base + "/api/out?since=%d" % mark,
                            timeout=10) as r:
                        data = json.loads(r.read())["data"]
                except Exception:
                    continue
                idx = data.rfind("\nAGENT_ANSWER:")
                if idx >= 0:
                    ans = data[idx + len("\nAGENT_ANSWER:"):].split("\n")[0]
                    ans = ans.strip()
                    if ans:
                        return ans
            return ("(the computer did not answer within %d minutes)"
                    % (budget // 60))
        finally:
            _GUEST_DONES[0] += 1
    finally:
        _GUEST_TASK[0] = ""
        _guest_release()


_RATE = {}  # name -> [tokens, ts] for circuit tok/min rates


def _with_rate(name, d):
    now = time.time()
    prev = _RATE.get(name)
    d["tok_min"] = None
    if prev and now > prev[1] and d["tokens"] >= prev[0]:
        dt = (now - prev[1]) / 60.0
        if dt > 0:
            d["tok_min"] = round((d["tokens"] - prev[0]) / dt, 1)
    _RATE[name] = [d["tokens"], now]
    return d


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
    grown = _with_rate("grown", _circuit_tokens(grown_log_path))
    grown["running"] = proc_running("circuitagent agent-big")
    grown["fail"] = "CIRCUIT AGENT: FAIL" in tail(grown_log_path)
    stock = _with_rate("stock", _circuit_tokens(os.path.expanduser(
        "~/qwenputer-test/circuit-stock-builtin.log")))
    with _GUEST_COND:
        qlen = len(_GUEST_QUEUE)
    return {
        "chat_series": series,
        "chat_open": _CHAT_OPEN[0],
        "grown": grown,
        "stock": stock,
        "terminal_up": tcp_up(8951),
        "guest_hands": _GUEST_HANDS[0],
        "guest_dones": _GUEST_DONES[0],
        "guest_active": _GUEST_BUSY[0],
        "guest_queue": qlen,
        "guest_task": _GUEST_TASK[0],
    }


# --- weights-machine telemetry (the System tab) ---
# CPU/RAM of the machine running on the weights - never the host PC.
# The live guest has /proc (verified 2026-11-01), so a background
# sampler asks the guest itself every 60 s, but ONLY while the console
# sits idle at its prompt - never interleaving a session. CPU% comes
# from /proc/stat deltas, RAM from meminfo, load from loadavg. Engine
# steps/s comes from liveterm's /api/stats when it exists (older
# liveterms 404 and the tab says so honestly).
_SYS_CMD = ("echo QWSYSB; cat /proc/meminfo; echo QWSYSSTAT; "
            "cat /proc/stat; echo QWSYSLOAD; cat /proc/loadavg; echo QWSYSE")
_SYS = {"at": 0, "mem_total_kb": 0, "mem_avail_kb": 0, "cpu_pct": None,
        "load1": None, "console_bps": 0, "steps": None,
        "steps_per_s": None, "note": "no sample yet",
        "last_try": "never"}
_SYS_LOCK = threading.Lock()
_SYS_PREV = {"cpu": None, "at": 0, "until": None, "steps": None}


def _parse_sysstat(text):
    """Parse a QWSYSB..QWSYSE slice into mem/stat/load pieces (pure)."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    try:
        b = next(i for i, l in enumerate(lines) if l.strip() == "QWSYSB")
        e = next(i for i, l in enumerate(lines)
                 if l.strip() == "QWSYSE" and i > b)
    except StopIteration:
        raise ValueError("sys markers missing")
    out = {"mem_total_kb": 0, "mem_avail_kb": 0, "cpu": None,
           "load1": None}
    for l in lines[b + 1:e]:
        if l.startswith("MemTotal:"):
            out["mem_total_kb"] = int(l.split()[1])
        elif l.startswith("MemAvailable:"):
            out["mem_avail_kb"] = int(l.split()[1])
        elif l.startswith("cpu  "):
            nums = [int(x) for x in l.split()[1:9]]
            out["cpu"] = (sum(nums), nums[3])
        elif l and l[0].isdigit() and "." in (l.split() or [""])[0]:
            try:
                out["load1"] = float(l.split()[0])
            except ValueError:
                pass
    return out


def _harvest_last(text, note):
    """Fill mem/load from the newest QWSYS reply already in TEXT.

    Stale-but-real beats "no sample yet": after a restart, or while a
    long session holds the console, the tab still shows the last known
    numbers, honestly labeled. CPU% never comes from here (it needs a
    fresh delta pair, not one old line).
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    begins = [i for i, l in enumerate(lines) if l.strip() == "QWSYSB"]
    ends = [i for i, l in enumerate(lines) if l.strip() == "QWSYSE"]
    pairs = [(b, e) for b in begins for e in ends if e > b]
    if not pairs:
        return False
    b, e = max(pairs)
    try:
        p = _parse_sysstat("\n".join(lines[b:e + 1]))
    except ValueError:
        return False
    with _SYS_LOCK:
        if p["mem_total_kb"]:
            _SYS["mem_total_kb"] = p["mem_total_kb"]
            _SYS["mem_avail_kb"] = p["mem_avail_kb"]
        _SYS["load1"] = p["load1"]
        if not _SYS["at"]:
            _SYS["at"] = time.time()
        _SYS["note"] = note
    return True


def _sys_sample_once(base="http://127.0.0.1:8951"):
    def ret(s):
        with _SYS_LOCK:
            _SYS["last_try"] = s
        return s

    try:
        with urllib.request.urlopen(base + "/api/out?since=0",
                                    timeout=10) as r:
            snap = json.loads(r.read())
    except Exception:
        return ret("terminal offline")
    if "ASK_READY" not in snap["data"] or \
            not snap["data"].rstrip().endswith("agent>"):
        _harvest_last(snap["data"], "stale - console busy")
        return ret("console busy - kept last sample")
    mark = snap["until"]
    try:
        req = urllib.request.Request(
            base + "/api/in",
            data=json.dumps({"data": _SYS_CMD + "\n"}).encode(),
            headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10).read()
    except Exception:
        return ret("send failed")
    text = ""
    deadline = time.time() + 40
    while time.time() < deadline:
        time.sleep(3)
        try:
            d = json.loads(urllib.request.urlopen(
                base + "/api/out?since=%d" % mark, timeout=10).read())
        except Exception:
            continue
        mark = d["until"]
        text += d["data"].replace("\r\n", "\n")
        if any(l.strip() == "QWSYSE" for l in text.split("\n")):
            break
    else:
        return ret("guest never answered")
    try:
        p = _parse_sysstat(text)
    except ValueError:
        return ret("unparseable reply")
    now = time.time()
    with _SYS_LOCK:
        if p["mem_total_kb"]:
            _SYS["mem_total_kb"] = p["mem_total_kb"]
            _SYS["mem_avail_kb"] = p["mem_avail_kb"]
        _SYS["load1"] = p["load1"]
        prev = _SYS_PREV["cpu"]
        if prev and p["cpu"] and now > _SYS_PREV["at"]:
            dt = p["cpu"][0] - prev[0]
            dc = p["cpu"][1] - prev[1]
            if dt > 0:
                _SYS["cpu_pct"] = round(100.0 * (dt - dc) / dt, 1)
        if p["cpu"]:
            _SYS_PREV["cpu"] = p["cpu"]
            _SYS_PREV["at"] = now
        pu = _SYS_PREV["until"]
        if pu and now > pu[1]:
            _SYS["console_bps"] = round((mark - pu[0]) / (now - pu[1]), 1)
        _SYS_PREV["until"] = (mark, now)
        _SYS["at"] = now
        _SYS["note"] = "live from the guest"
    return ret("ok")


def _stats_sample_once(base="http://127.0.0.1:8951"):
    try:
        with urllib.request.urlopen(base + "/api/stats", timeout=5) as r:
            st = json.loads(r.read())
    except Exception:
        return
    now = time.time()
    with _SYS_LOCK:
        prev = _SYS_PREV["steps"]
        if prev and now > prev[1] and st["steps"] >= prev[0]:
            dt = now - prev[1]
            if dt > 0:
                _SYS["steps_per_s"] = round((st["steps"] - prev[0]) / dt)
        _SYS_PREV["steps"] = (st["steps"], now)
        _SYS["steps"] = st["steps"]


def _sys_loop():
    try:
        with urllib.request.urlopen("http://127.0.0.1:8951/api/out?since=0",
                                    timeout=15) as r:
            _harvest_last(json.loads(r.read())["data"],
                          "last console reply")
    except Exception:
        pass
    while True:
        try:
            _sys_sample_once()
        except Exception:
            pass
        try:
            _stats_sample_once()
        except Exception:
            pass
        time.sleep(60)


def get_sys():
    with _SYS_LOCK:
        sys = dict(_SYS)
    sys["age_s"] = int(time.time() - sys["at"]) if sys["at"] else None
    grown = _with_rate("grown", _circuit_tokens(os.path.expanduser(
        "~/qwenputer-circuit-test/circuit-builtin.log")))
    stock = _with_rate("stock", _circuit_tokens(os.path.expanduser(
        "~/qwenputer-test/circuit-stock-builtin.log")))
    return {"guest": sys, "grown": grown, "stock": stock,
            "terminal_up": tcp_up(8951)}


# --- the mind, read-only (the Apps tab) ---
MIND_DIR = os.path.expanduser("~/qwenputer-test/mind")


def _read_mind(minddir=MIND_DIR):
    m = {"present": False, "taken": None, "jobs": [],
         "jobs_log_tail": "", "brain_lines": 0, "failures_lines": 0,
         "nb_days": 0, "skills": [], "apps": [], "plan_pending": 0,
         "plan_total": 0, "upkeep_paused": False, "episodes": 0}
    root = os.path.join(minddir, "root")
    if not os.path.isdir(root):
        return m
    m["present"] = True

    def rd(*p):
        try:
            with open(os.path.join(root, *p)) as f:
                return f.read()
        except Exception:
            return ""

    try:
        with open(os.path.join(minddir, "SNAPSHOT.txt")) as f:
            snap = f.read()
    except Exception:
        snap = ""
    for l in snap.splitlines():
        if l.startswith("taken:"):
            m["taken"] = l[6:].strip()
    for line in rd("jobs").splitlines():
        f = line.split("|")
        if len(f) >= 5:
            try:
                due = int(f[0])
            except ValueError:
                due = 0
            m["jobs"].append({"due_in_s": due - int(time.time()),
                              "every_min": f[1], "fails": f[2],
                              "last": f[3][:90], "task": f[4][:120]})
    m["jobs_log_tail"] = rd("jobs.log")[-600:]
    m["brain_lines"] = len(
        [l for l in rd("BRAIN.md").splitlines() if l.strip()])
    m["failures_lines"] = len(
        [l for l in rd("FAILURES.md").splitlines() if l.strip()])
    try:
        m["nb_days"] = len([n for n in os.listdir(os.path.join(root, "nb"))
                            if n.endswith(".md")])
    except Exception:
        pass
    for dname, key, pline in (("skills", "skills", 2),
                              ("apps", "apps", 2)):
        d = os.path.join(root, dname)
        try:
            names = sorted(os.listdir(d))
        except Exception:
            continue
        for n in names[:40]:
            try:
                with open(os.path.join(d, n)) as f:
                    lines = f.read().splitlines()
            except Exception:
                continue
            prev = lines[pline].strip()[:100] \
                if len(lines) > pline else ""
            m[key].append({"name": n[:-3] if n.endswith(".md") else n,
                           "preview": prev})
    plan = rd("PLAN.md")
    steps = [l for l in plan.splitlines() if l.startswith("- [")]
    m["plan_total"] = len(steps)
    m["plan_pending"] = len([l for l in steps if l.startswith("- [ ]")])
    m["upkeep_paused"] = os.path.exists(os.path.join(root, "agentd.off"))
    try:
        with open(os.path.expanduser(
                "~/qwenputer-test/episodes.jsonl")) as f:
            m["episodes"] = sum(1 for _ in f)
    except Exception:
        pass
    return m


_MIRROR_STATE = ["idle"]


def _mirror_now():
    if _MIRROR_STATE[0] == "running":
        return {"ok": False, "msg": "a snapshot is already running"}
    try:
        from chipformer.rv.mindmirror import snapshot as _snap
    except Exception as e:
        return {"ok": False, "msg": "mirror tool missing: %s" % e}

    def run():
        _MIRROR_STATE[0] = "running"
        try:
            _MIRROR_STATE[0] = _snap("http://127.0.0.1:8951",
                                     os.path.expanduser(
                                         "~/qwenputer-test/mind"))
        except Exception as e:
            _MIRROR_STATE[0] = "fail: %s" % e

    threading.Thread(target=run, daemon=True).start()
    return {"ok": True, "msg": "snapshot started"}


def _traces_now():
    try:
        from chipformer.rv.traceexport import export as _exp
    except Exception as e:
        return {"ok": False, "msg": "trace tool missing: %s" % e}
    try:
        msg = _exp("http://127.0.0.1:8951",
                   os.path.expanduser("~/qwenputer-test/episodes.jsonl"))
        return {"ok": True, "msg": msg}
    except Exception as e:
        return {"ok": False, "msg": str(e)[:160]}


def _set_upkeep(paused):
    base = "http://127.0.0.1:8951"
    try:
        with urllib.request.urlopen(base + "/api/out?since=0",
                                    timeout=10) as r:
            snap = json.loads(r.read())
    except Exception:
        return {"ok": False, "msg": "terminal offline"}
    if "ASK_READY" not in snap["data"] or \
            not snap["data"].rstrip().endswith("agent>"):
        return {"ok": False,
                "msg": "console busy - try again in a moment"}
    cmd = ("touch /root/agentd.off; echo UPKEEP_OFF"
           if paused else "rm -f /root/agentd.off; echo UPKEEP_ON")
    try:
        req = urllib.request.Request(
            base + "/api/in", data=json.dumps({"data": cmd + "\n"}).encode(),
            headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10).read()
    except Exception as e:
        return {"ok": False, "msg": str(e)[:120]}
    return {"ok": True,
            "msg": "upkeep paused" if paused else "upkeep resumed"}


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
border:1px solid var(--line);border-radius:10px;cursor:pointer}
#vizbig{width:100%;height:420px;display:block;background:#0a0d12;
border:1px solid var(--line);border-radius:10px;cursor:grab}
#viztip{position:fixed;display:none;background:var(--card);color:var(--tx);
border:1px solid var(--acc);border-radius:8px;padding:6px 10px;font-size:12px;
z-index:60;pointer-events:none;max-width:260px;box-shadow:var(--shadow)}
#vizpop{position:fixed;display:none;background:var(--card);color:var(--tx);
border:1px solid var(--acc);border-radius:10px;padding:12px 14px;font-size:13px;
z-index:61;max-width:340px;box-shadow:var(--shadow)}
#vizpop h4{margin:0 0 6px;font-size:13px;letter-spacing:.05em;color:var(--acc)}
#vizpop .prow{display:flex;justify-content:space-between;gap:12px;padding:3px 0;border-top:1px solid var(--line)}
#vizpop .prow span{color:var(--dim);white-space:nowrap}
#vizpop .prow b{font-weight:600;text-align:right}
#vizpop .btnsm{margin-top:8px;margin-right:6px}
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
<button data-t="activity">Activity</button>
<button data-t="chat">Chat</button>
<button data-t="terminal">Terminal</button>
<button data-t="desktop">Desktop</button>
<button data-t="apps">Apps</button>
<button data-t="system">System</button>
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
<div class="tile"><div class="k">GUEST CPU</div><div class="v" id="t-cpu">&hellip;</div><div class="bar"><i id="t-cpubar"></i></div></div>
<div class="tile"><div class="k">GUEST RAM</div><div class="v" id="t-ram">&hellip;</div><div class="bar"><i id="t-rambar"></i></div></div>
</div>
<div class="card" style="margin-bottom:12px"><h3>ACTIVITY &mdash; LIVE</h3>
<div id="nowrows" style="font-size:15px;line-height:1.65;margin:2px 0 10px">
<div><b>chat:</b> <span id="now-chat">&hellip;</span></div>
<div><b>computer:</b> <span id="now-computer">&hellip;</span></div>
<div><b>computer, all time:</b> <span id="now-guest">&hellip;</span></div>
<div><b>circuit experiment:</b> <span id="now-circuit">&hellip;</span></div>
</div>
<canvas id="viz" width="900" height="250"></canvas>
<div id="viztip"></div>
<div id="vizpop" style="display:none"></div>
<p class="dim" id="vizlegend">warming up&hellip;</p>
<p class="dim">left: the brain thinking (amber). right: the live
computer &mdash; guest linux on the fast native core, borrowing
the brain through the seat (blue). pulses are real work: amber flows
brain&nbsp;&rarr;&nbsp;computer when a job is handed over, blue
flows back when the computer reports. nothing here is decoration:
when nothing moves, nothing happened.</p></div>
<div class="cards">
<div class="card"><h3>THE BRAIN</h3>
<p id="ov-model">&hellip;</p>
<p class="dim">the built-in Qwen 3.5 4B weights, served as the seat.
the chat tab talks to this. the guest computer asks it
questions the same way.</p></div>
<div class="card"><h3>COMPILED CIRCUIT (EXPERIMENT)</h3>
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
<section id="activity" class="tab">
<div class="card"><h3>ACTIVITY &mdash; INTERACTIVE</h3>
<p class="dim" id="vizstats">&hellip;</p>
<canvas id="vizbig" width="900" height="420"></canvas>
<div class="toolbar" style="margin-top:10px;margin-bottom:0">
<button class="btnsm" id="vizpause">pause</button>
<button class="btnsm" id="vizreset">reset view</button>
<span class="sp"></span>
<span class="dim">drag to spin &middot; scroll to zoom &middot; hover nodes for numbers &middot; click a node to inspect it</span>
</div>
<p class="dim">the same live pulses as Overview: amber hands a job
brain&nbsp;&rarr;&nbsp;computer, blue reports back, white flashes for
the circuit experiment. amber ticks under the computer orb are queued
jobs.</p></div>
</section>
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
<p>no framebuffer build exists in this tree. the display path was
proven by the Doom port (<code>github.com/Franzferdinan51
/qwenputer-doom</code>: guest draws in its memory, host shows it,
keys go back in), and a framebuffer shell &mdash; a screen window in
guest memory, a pointer, simple windows &mdash; is the planned spike.
it will live in this tab when it lands.</p>
<p class="dim">realistic on the fast core. on the in-weights circuit
the same trick works mechanically (the mailbox proved host access to
circuit memory) but at ~17 instructions/second a desktop is a
screenshot an hour, not a desktop.</p></div>
</div></section>
<section id="apps" class="tab">
<p style="color:var(--dim)">the truth about programs, live from the
guest's own mind <span id="mindage"></span> &mdash;
<button class="btnsm" id="mindrefresh">refresh</button></p>
<h3 style="font-size:14px;letter-spacing:.05em;color:var(--acc)">BUILT BY THE AGENT</h3>
<div class="cards" id="builtapps"></div>
<h3 style="font-size:14px;letter-spacing:.05em;color:var(--acc)">SKILLS IT SAVED</h3>
<div class="cards" id="skills"></div>
<h3 style="font-size:14px;letter-spacing:.05em;color:var(--acc)">SCHEDULED JOBS</h3>
<div class="cards" id="jobs"></div>
<h3 style="font-size:14px;letter-spacing:.05em;color:var(--acc)">GUEST ENTRY POINTS</h3>
<p style="color:var(--dim)">these run in the guest &mdash; tap one to
open the terminal.</p>
<div class="cards" id="entryapps"></div>
<h3 style="font-size:14px;letter-spacing:.05em;color:var(--acc)">ELSEWHERE, NOT ON THIS MACHINE</h3>
<div class="cards" id="upstream"></div></section>
<section id="system" class="tab">
<p style="color:var(--dim)">the weights-machine's own load &mdash;
guest CPU/RAM from its <code>/proc</code> &mdash; never the host.
<span id="sysage"></span></p>
<div class="cards">
<div class="card"><h3>GUEST CPU</h3>
<canvas id="gauge-cpu" width="180" height="120" style="width:100%;max-width:220px"></canvas>
<p class="dim" id="sys-load"></p></div>
<div class="card"><h3>GUEST RAM</h3>
<canvas id="gauge-ram" width="180" height="120" style="width:100%;max-width:220px"></canvas>
<p class="dim" id="sys-mem"></p></div>
<div class="card"><h3>ENGINE</h3>
<p id="sys-engine">&hellip;</p>
<p class="dim" id="sys-console"></p></div>
<div class="card"><h3>CIRCUIT RATE</h3>
<p id="sys-circuit">&hellip;</p>
<p class="dim">tokens per minute from the runs' own logs.</p></div>
</div>
<div class="card" style="margin-top:12px"><h3>CPU HISTORY &mdash; 5 MINUTES</h3>
<canvas id="spark" width="900" height="90" style="width:100%;height:90px"></canvas></div>
</section>
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
<label style="margin-top:16px"><input type="checkbox" id="s-guest" style="width:auto;margin-right:8px">let chat hand jobs to the guest computer</label>
<label>guest wait budget, seconds (60&ndash;3600 &mdash; how long one delegation may queue + work)</label>
<input type="number" id="s-budget" step="60" min="60" max="3600">
<h3 style="font-size:14px;letter-spacing:.05em;color:var(--acc);margin-top:20px">DECISION MODEL (OPTIONAL)</h3>
<label><input type="checkbox" id="s-decider" style="width:auto;margin-right:8px">enable a fast decision-model sidecar</label>
<label>decision model URL (LM Studio serves here by default &mdash; same address as the seat is fine)</label>
<input type="text" id="s-deciderurl">
<label>decision model id (the exact id loaded there, e.g. liquidai/d1-omni-600m)</label>
<input type="text" id="s-decidermodel">
<p class="dim">leave this off for fused/GGUF single-model setups: with no
sidecar running, enabling it only produces errors. nothing uses the
decider yet &mdash; this setting just verifies it is reachable.</p>
<p><button class="act" id="save" style="padding:10px 22px">Save settings</button>
<button class="ghost" id="testbe">Test backend</button>
<button class="ghost" id="testdecider">Test decider</button>
<span id="saved" style="color:var(--ok)"></span></p>
<p class="dim" id="resolved"></p>
<p class="dim" id="deciderstat"></p>
<h3 style="font-size:14px;letter-spacing:.05em;color:var(--acc);margin-top:20px">MIND &amp; DATA</h3>
<p><button class="ghost" id="actsnap">Snapshot mind now</button>
<button class="ghost" id="acttrace">Export traces now</button>
<button class="ghost" id="actupkeep">Pause upkeep</button>
<span id="actmsg" style="font-size:13px"></span></p>
<p class="dim" id="mindstat"></p>
<p class="dim">capability battery (10 ground-truth tasks, needs the
guest image, several minutes):<br>
<code>python3 -m chipformer.rv.battery Image --fast --fast-bin ./fastlinux --report battery.json</code></p>
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
 const inline=s=>{const ic=[];
  s=s.replace(/`([^`]+)`/g,(m,c)=>{ic.push('<code>'+c+'</code>');return '\u0001'+(ic.length-1)+'\u0001';});
  s=s.replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>')
   .replace(/(^|\W)\*([^*\n]+)\*/g,'$1<em>$2</em>')
   .replace(/\[([^\]]+)\]\((https?:[^)\s"']+)\)/g,'<a href="$2" target="_blank" rel="noopener">$1</a>');
  return s.replace(/\u0001(\d+)\u0001/g,(m,i)=>ic[+i]);};
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
const APPS=[['\u25b8','live terminal','the Qwen-Linux guest shell itself — everything below runs here','Terminal tab',null],
['?', 'ask','ask the built-in weights a question; the answer is typed back into the guest','ask "your question"','ask "your question"'],
['\u25c9','hermes-agent','the guest agent — works through the weights with the whole Linux side: files, exact arithmetic, the notebook','hermes-agent "your task"','hermes-agent "your task"'],
['\u2595','notebook','the guest\u2019s dated notes at /root/nb — the agent writes plans and facts there and reads them back','nb list','nb list'],
['\u2206','calc','exact integer arithmetic for the agent (no bc/awk/python in the guest)','calc \u2019123456789 * 987654321\u2019','calc 123456789*987654321']];
const UPSTREAM=[['\u03bb','MicroPython on the CPU','Python running on the Qwenputer CPU — the upstream port, proven on the reference model, not installed here.','huggingface.co/divagr1925/Qwen3.5-4B-Qwenputer'],
['\u25e9','Doom on Qwenputer','The Doom port (doomgeneric) for the CPU inside the weights — proven on the reference machine: title screen, menu, E1M1 rendering. Ships a native fast runner and a playable live view; the in-weights circuit run is next.','github.com/Franzferdinan51/qwenputer-doom'],
['\u25c8','bare-metal agent','agent.c — the harness compiled for the circuit itself: store / recall / mul through the mailbox in RAM. source ships in this tree (env/progs/agent.c), runs via bareagent.py on the circuit.','env/progs/agent.c']];
function escH(s){return String(s==null?'':s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
$('entryapps').innerHTML=APPS.map((a,i)=>'<div class="app"><div><span class="ico">'
 +a[0]+'</span><b>'+escH(a[1])+'</b></div><span>'+escH(a[2])+'</span><span><code>'
 +escH(a[3])+'</code></span><div class="row">'
 +'<button class="btnsm" data-term="'+i+'">open in terminal</button>'
 +(a[4]?'<button class="btnsm" data-copy="'+i+'">copy command</button>':'')
 +'</div></div>').join('');
$('upstream').innerHTML=UPSTREAM.map(a=>'<div class="app"><div><span class="ico">'
 +a[0]+'</span><b>'+escH(a[1])+'</b></div><span>'+escH(a[2])+'</span><span><code>'
 +escH(a[3])+'</code></span></div>').join('');
function goTerm(){document.querySelector('nav button[data-t="terminal"]').click();}
document.querySelectorAll('[data-term]').forEach(el=>el.onclick=goTerm);
document.querySelectorAll('[data-copy]').forEach(el=>el.onclick=()=>{
 const cmd=APPS[+el.dataset.copy][4];
 if(navigator.clipboard)navigator.clipboard.writeText(cmd);
 el.textContent='copied';setTimeout(()=>el.textContent='copy command',1500);});
let lastMind=null;
async function loadMind(){
 try{const r=await fetch('/api/mind');const j=await r.json();lastMind=j;
  $('mindage').textContent=j.present?('as of '+(j.taken||'unknown')):'(no snapshot yet — the mirror runs every 15 min)';
  const ago=$('actupkeep');if(ago)ago.textContent=j.upkeep_paused?'Resume upkeep':'Pause upkeep';
  $('mindstat').textContent=j.present?('mirror: '+(j.taken||'?')+' · '+j.skills.length+' skills · '+j.apps.length+' apps · '+j.jobs.length+' jobs · '+j.episodes+' tune episodes'):'mirror has not run yet';
  const card=(t,b)=>'<div class="app"><b>'+escH(t)+'</b><span>'+escH(b)+'</span></div>';
  $('builtapps').innerHTML=j.apps.length?j.apps.map(a=>'<div class="app"><b>'+escH(a.name)+'</b><span>'
   +escH(a.preview||'(no preview)')+'</span><span><code>TOOL app '+escH(a.name)+'</code></span></div>').join('')
   :card('nothing built yet','when the agent hits a gap it builds itself a tool here — check back after it works a while.');
  $('skills').innerHTML=j.skills.length?j.skills.map(a=>'<div class="app"><b>'+escH(a.name)+'</b><span>'
   +escH(a.preview||'(no preview)')+'</span><span><code>TOOL skill '+escH(a.name)+'</code></span></div>').join('')
   :card('no skills saved','the agent saves reusable procedures here after hard tasks.');
  $('jobs').innerHTML=j.jobs.length?j.jobs.map(b=>{
   const mins=Math.round(b.due_in_s/60);
   const due=b.due_in_s>0?('due in '+(mins>2880?Math.round(mins/1440)+'d':(mins>120?Math.round(mins/60)+'h':Math.max(1,mins)+' min'))):('due '+Math.max(1,-mins)+' min ago — will fire on the next tick');
   return '<div class="app"><b>'+escH(b.task)+'</b><span>'+escH(due)+(b.every_min&&+b.every_min>0?(' · repeats every '+b.every_min+' min'):' · one-shot')+'</span><span>last: '+escH(b.last||'never ran')+'</span></div>';}).join('')
   :card('no jobs scheduled','the agent schedules its own cron jobs here — memory upkeep reseeds itself daily.');
 }catch(e){}
}
loadMind();$('mindrefresh').onclick=loadMind;
async function status(){
 try{const r=await fetch('/api/status');const j=await r.json();
  j.grown=j.grown||{};j.stock=j.stock||{};j.seat=j.seat||{};viz.lastSeat=j.seat;
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
  const dz=j.decider||{};
  $('deciderstat').textContent=!dz.enabled?'decision model: off':('decision model: '+(dz.model||('unreachable — '+(dz.error||''))));
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
 if(content){body.innerHTML=md(content);hist.push({role:'assistant',content});
  if(hist.length>40)hist.splice(0,hist.length-40);}
 const dt=(Date.now()-t0)/1000;
 $('chatstat').textContent='done \u00b7 '+Math.round(dt)+'s \u00b7 '+ntok+
  ' tokens'+(dt>0&&ntok?' \u00b7 '+(ntok/dt).toFixed(1)+' tok/s':'');
 AC=null;$('send').disabled=false;$('stop').style.display='none';
 $('inp').focus();
}
$('send').onclick=send;
$('inp').onkeydown=e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send();}};
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
function vizPulse(n,big,back){for(let i=0;i<n;i++){if(viz.pulses.length>240)return;
 const a=brainPts[Math.random()*brainPts.length|0],
       b=chipPts[Math.random()*chipPts.length|0];
 viz.pulses.push({t:0,sp:.0035+Math.random()*.004,
  a:back?b:a,b:back?a:b,big:!!big,back:!!back});}}
window.vizPulse=vizPulse;
function drawViz(){rot+=.0035;vctx.clearRect(0,0,900,250);
 const draw=(pts,cx,cy,s,col)=>{const out=[];
  for(const p of pts){const q=proj(p,cx,cy,s);out.push(q);
   vctx.globalAlpha=.25+.55*q[2]/1.6;vctx.fillStyle=col;
   vctx.beginPath();vctx.arc(q[0],q[1],1.6*q[2],0,7);vctx.fill();}
  vctx.globalAlpha=1;return out;};
 draw(brainPts,225,118,64,'#e8b93e');
 draw(chipPts,665,118,15,'#57c7ff');
 const qn=Math.min(12,(viz.lastAct&&viz.lastAct.guest_queue)||0);
 vctx.fillStyle='#e8b93e';
 for(let i=0;i<qn;i++)vctx.fillRect(610+i*10,208,7,7);
 vctx.fillStyle='#9aa7b5';vctx.font='12px sans-serif';
 vctx.fillText('BRAIN \u2014 qwen 3.5 4b, the seat',140,240);
 vctx.fillText('COMPUTER \u2014 linux, fast core + Qwen brain',520,240);
 viz.pulses=viz.pulses.filter(p=>p.t<=1);
 for(const p of viz.pulses){p.t+=p.sp;
  const a=proj(p.a,225,118,64),b=proj(p.b,665,118,15);
  const x=a[0]+(b[0]-a[0])*p.t,
        y=a[1]+(b[1]-a[1])*p.t-Math.sin(p.t*3.14)*36;
  vctx.globalAlpha=.9;vctx.fillStyle=p.big?'#ffffff':(p.back?'#57c7ff':'#ffd479');
  vctx.beginPath();vctx.arc(x,y,p.big?3:2,0,7);vctx.fill();
  vctx.globalAlpha=1;}
 requestAnimationFrame(drawViz);}
drawViz();
function vizOrbAt(cx,cy){const dx=cx-225,dy=cy-118;
 if(dx*dx+dy*dy<95*95)return 'brain';
 const ex=cx-665,ey=cy-118;
 if(ex*ex+ey*ey<80*80)return 'computer';
 return null;}
function orbTip(o){
 if(o==='brain'){const m=($('seatmodel')||{}).textContent||'?';
  return 'BRAIN — '+(m||'?')+' · chat '+(viz.lastAct&&viz.lastAct.chat_open?'live now':'idle');}
 const a=viz.lastAct||{},g=(viz.lastSys&&viz.lastSys.guest)||{};
 if(o==='queue')return 'QUEUE — '+(a.guest_queue||0)+' waiting'+((a.guest_hands||a.guest_dones)?(' · '+(a.guest_hands||0)+' run · '+(a.guest_dones||0)+' back'):'');
 return 'COMPUTER — '+(a.guest_active?('working'+(a.guest_task?': '+a.guest_task:'')):'idle')
  +(a.guest_queue?(' · '+a.guest_queue+' queued'):'')
  +(g.cpu_pct==null?'':(' · CPU '+g.cpu_pct+'%'))
  +(g.mem_total_kb?(' · RAM '+Math.round((g.mem_total_kb-(g.mem_avail_kb||0))/1024)+'/'+Math.round(g.mem_total_kb/1024)+'M'):'');}
function showTip(e,t,verb){const tip=$('viztip');
 tip.textContent=t+' ('+(verb||'click to open')+')';
 tip.style.display='block';tip.style.left=(e.clientX+14)+'px';tip.style.top=(e.clientY+10)+'px';}
$('viz').addEventListener('mousemove',e=>{
 const r=$('viz').getBoundingClientRect();
 const o=vizOrbAt((e.clientX-r.left)*900/r.width,(e.clientY-r.top)*250/r.height);
 if(!o){$('viztip').style.display='none';return;}
 showTip(e,orbTip(o));});
$('viz').addEventListener('mouseleave',()=>{$('viztip').style.display='none';});
$('viz').addEventListener('click',e=>{
 const r=$('viz').getBoundingClientRect();
 const o=vizOrbAt((e.clientX-r.left)*900/r.width,(e.clientY-r.top)*250/r.height);
 if(o==='brain')document.querySelector('nav button[data-t="chat"]').click();
 if(o==='computer')document.querySelector('nav button[data-t="terminal"]').click();});
const vb={rot:0,zoom:1,paused:false,drag:null,moved:false};
const vbctx=$('vizbig').getContext('2d');
function projB(p,cx,cy,s){const c=Math.cos(vb.rot),sn=Math.sin(vb.rot);
 const x=p[0]*c-p[2]*sn,z=p[0]*sn+p[2]*c,y=p[1];
 const d=340/(340+z*40);return[cx+x*s*d*vb.zoom,cy+y*s*d*vb.zoom,d];}
function drawBig(){
 requestAnimationFrame(drawBig);
 if(vb.paused||!$('activity').classList.contains('on'))return;
 if(!vb.drag)vb.rot+=.0035;
 vbctx.clearRect(0,0,900,420);
 const draw=(pts,cx,cy,s,col)=>{for(const p of pts){const q=projB(p,cx,cy,s);
  vbctx.globalAlpha=.25+.55*q[2]/1.6;vbctx.fillStyle=col;
  vbctx.beginPath();vbctx.arc(q[0],q[1],1.8*q[2],0,7);vbctx.fill();}};
 draw(brainPts,225,185,64,'#e8b93e');
 draw(chipPts,665,185,15,'#57c7ff');
 vbctx.globalAlpha=1;vbctx.fillStyle='#9aa7b5';vbctx.font='13px sans-serif';
 vbctx.fillText('BRAIN — qwen 3.5 4b, the seat',140,405);
 vbctx.fillText('COMPUTER — linux, fast core + Qwen brain',500,405);
 const qn=Math.min(12,(viz.lastAct&&viz.lastAct.guest_queue)||0);
 vbctx.fillStyle='#e8b93e';
 for(let i=0;i<qn;i++)vbctx.fillRect(610+i*10,350,7,7);
 for(const p of viz.pulses){
  const a=projB(p.a,225,185,64),b=projB(p.b,665,185,15);
  const x=a[0]+(b[0]-a[0])*p.t,
        y=a[1]+(b[1]-a[1])*p.t-Math.sin(p.t*3.14)*52;
  vbctx.globalAlpha=.9;vbctx.fillStyle=p.big?'#ffffff':(p.back?'#57c7ff':'#ffd479');
  vbctx.beginPath();vbctx.arc(x,y,p.big?3.5:2.5,0,7);vbctx.fill();}
 vbctx.globalAlpha=1;}
drawBig();
function vbPos(e){const r=$('vizbig').getBoundingClientRect();
 return [(e.clientX-r.left)*900/r.width,(e.clientY-r.top)*420/r.height];}
function vbOrb(x,y){
 if(x>598&&x<742&&y>338&&y<366)return 'queue';
 const dx=x-225,dy=y-185;
 if(dx*dx+dy*dy<100*100)return 'brain';
 const ex=x-665,ey=y-185;
 if(ex*ex+ey*ey<85*85)return 'computer';
 return null;}
$('vizbig').addEventListener('mousedown',e=>{vb.drag={x:e.clientX,rot:vb.rot};vb.moved=false;});
$('vizbig').addEventListener('mouseup',()=>{vb.drag=null;});
$('vizbig').addEventListener('mouseleave',()=>{vb.drag=null;$('viztip').style.display='none';});
$('vizbig').addEventListener('mousemove',e=>{
 if(vb.drag){const dx=e.clientX-vb.drag.x;
  if(Math.abs(dx)>3)vb.moved=true;
  vb.rot=vb.drag.rot+dx*.006;return;}
 const xy=vbPos(e),o=vbOrb(xy[0],xy[1]);
 if(!o){$('viztip').style.display='none';return;}
 showTip(e,orbTip(o),'click to inspect');});
$('vizbig').addEventListener('wheel',e=>{e.preventDefault();
 vb.zoom=Math.max(.6,Math.min(2.5,vb.zoom*(e.deltaY>0?.92:1.08)));},{passive:false});
$('vizbig').addEventListener('click',e=>{
 if(vb.moved)return;
 const xy=vbPos(e),o=vbOrb(xy[0],xy[1]);
 if(!o){closeVizPop();return;}
 openVizPop(o,e);});
let vizpopFor=null;
function vizpopHTML(o){
 const a=viz.lastAct||{},g=(viz.lastSys&&viz.lastSys.guest)||{},
  m=lastMind||{},seat=viz.lastSeat||{};
 const row=(k,v)=>'<div class="prow"><span>'+k+'</span><b>'+v+'</b></div>';
 if(o==='brain'){
  const mem=m.present?((m.skills||[]).length+' skills · '+(m.apps||[]).length+' apps · '+(m.jobs||[]).length+' jobs · '+(m.episodes||0)+' tune episodes'):'no snapshot yet';
  return '<h4>BRAIN — the qwen weights</h4>'
  +row('model',escH(seat.model||($('seatmodel')||{}).textContent||'?'))
  +row('seat',escH(seat.error?('error: '+seat.error):('via '+(seat.backend||'?'))))
  +row('chat',escH(a.chat_open?'live now':'idle'))
  +row('memory',escH(mem))
  +row('in flight',viz.pulses.length+' pulses · '+(a.guest_queue||0)+' queued')
  +'<p class="dim">the brain thinks, the computer runs. every amber dot is a job crossing over.</p>'
  +'<button class="btnsm" data-go="chat">open chat</button><button class="btnsm" data-x="1">close</button>';
 }
 if(o==='queue'){
  return '<h4>QUEUE — waiting jobs</h4>'
  +row('queued',a.guest_queue||0)
  +row('run so far',(a.guest_hands||0)+' handed · '+(a.guest_dones||0)+' reported')
  +'<p class="dim">jobs wait here, first in first out, while the computer is busy.</p>'
  +'<button class="btnsm" data-go="terminal">watch terminal</button><button class="btnsm" data-x="1">close</button>';
 }
 const cpu=g.cpu_pct==null?'…':(g.cpu_pct+'%');
 const ram=g.mem_total_kb?(Math.round((g.mem_total_kb-(g.mem_avail_kb||0))/1024)+'/'+Math.round(g.mem_total_kb/1024)+'M'):'…';
 const eng=g.steps==null?'older engine — live after its restart':(Number(g.steps).toLocaleString()+' steps'+(g.steps_per_s==null?'':(' · '+Number(g.steps_per_s).toLocaleString()+'/s')));
 const state=!a.terminal_up?'down':(a.guest_active?('working'+(a.guest_task?(': '+a.guest_task):'')):'idle at its prompt');
 return '<h4>COMPUTER — qwen-linux guest</h4>'
 +row('state',escH(state))
 +row('CPU',escH(cpu+(g.load1==null?'':(' · load '+g.load1))))
 +row('RAM',escH(ram))
 +row('engine',escH(eng))
 +row('sample',escH(g.age_s==null?('no fresh sample — '+(g.last_try||'warming up')):(g.age_s+'s ago · '+(g.note||''))))
 +'<p class="dim">the linux PC inside the weights. its own numbers, sampled live.</p>'
 +'<button class="btnsm" data-go="terminal">open terminal</button><button class="btnsm" data-x="1">close</button>';
}
function openVizPop(o,e){
 e=e||{clientX:400,clientY:200};vizpopFor=o;const p=$('vizpop');
 p.innerHTML=vizpopHTML(o);
 const w=window.innerWidth||1024;let x=e.clientX+14;if(x>w-370)x=e.clientX-360;
 p.style.left=Math.max(8,x)+'px';p.style.top=(e.clientY+10)+'px';
 p.style.display='block';
}
function closeVizPop(){vizpopFor=null;$('vizpop').style.display='none';}
$('vizpop').onclick=e=>{const t=e.target||{},d=t.dataset||{};
 if(d.x)closeVizPop();
 else if(d.go)document.querySelector('nav button[data-t="'+d.go+'"]').click();};
$('vizpause').onclick=()=>{vb.paused=!vb.paused;
 $('vizpause').textContent=vb.paused?'resume':'pause';};
$('vizreset').onclick=()=>{vb.rot=0;vb.zoom=1;};
let sysTickN=0,cpuHist=[];
function gauge(cv,pct){
 const c=cv.getContext('2d');c.clearRect(0,0,180,120);
 c.lineWidth=13;c.lineCap='round';
 c.strokeStyle='#2a3240';c.beginPath();c.arc(90,95,62,Math.PI,2*Math.PI);c.stroke();
 if(pct!=null){const f=Math.max(0,Math.min(100,pct))/100;
  c.strokeStyle=pct>85?'#f85149':(pct>60?'#e8b93e':'#3fb950');
  c.beginPath();c.arc(90,95,62,Math.PI,Math.PI+f*Math.PI);c.stroke();}
 c.fillStyle='#e8ecf1';c.font='bold 26px sans-serif';c.textAlign='center';
 c.fillText(pct==null?'\u2026':(Math.round(pct)+'%'),90,88);}
function spark(){
 const cv=$('spark'),c=cv.getContext('2d');c.clearRect(0,0,900,90);
 c.strokeStyle='#2a3240';c.lineWidth=1;c.beginPath();c.moveTo(0,45);c.lineTo(900,45);c.stroke();
 if(cpuHist.length<2)return;
 c.strokeStyle='#e8b93e';c.lineWidth=2;c.beginPath();
 cpuHist.forEach((v,i)=>{const x=i/(60-1)*900,y=85-Math.max(0,Math.min(100,v))/100*80;
  i?c.lineTo(x,y):c.moveTo(x,y);});
 c.stroke();
 const l=cpuHist[cpuHist.length-1];
 c.fillStyle='#e8b93e';c.beginPath();
 c.arc((cpuHist.length-1)/59*900,85-Math.max(0,Math.min(100,l))/100*80,3,0,7);c.fill();}
async function sysTick(){
 try{const r=await fetch('/api/sys');const j=await r.json();viz.lastSys=j;
  const g=j.guest||{};
  if(g.cpu_pct==null){$('t-cpu').textContent='\u2026';$('t-cpubar').style.width='0';}
  else{$('t-cpu').textContent=g.cpu_pct+'%';$('t-cpubar').style.width=Math.min(100,g.cpu_pct)+'%';}
  if(g.mem_total_kb){const used=g.mem_total_kb-(g.mem_avail_kb||0);
   $('t-ram').textContent=Math.round(used/1024)+'/'+Math.round(g.mem_total_kb/1024)+'M';
   $('t-rambar').style.width=Math.round(used/g.mem_total_kb*100)+'%';}
  else{$('t-ram').textContent='\u2026';}
  gauge($('gauge-cpu'),g.cpu_pct);
  gauge($('gauge-ram'),g.mem_total_kb?Math.round((g.mem_total_kb-(g.mem_avail_kb||0))/g.mem_total_kb*100):null);
  $('sys-load').textContent=g.load1==null?('waiting \u2014 '+(g.last_try||'first sample soon')):('load 1m: '+g.load1);
  $('sys-mem').textContent=g.mem_total_kb?Math.round((g.mem_total_kb-(g.mem_avail_kb||0))/1024)+' of '+Math.round(g.mem_total_kb/1024)+' MB used · guest RAM':'waiting for first sample\u2026';
  $('sys-engine').textContent=(g.steps==null?'engine steps: this liveterm predates /api/stats — live after its next restart':(Number(g.steps).toLocaleString()+' steps retired'+(g.steps_per_s==null?'':' · '+Number(g.steps_per_s).toLocaleString()+'/s')));
  $('sys-console').textContent='console traffic '+(g.console_bps||0)+' B/s · terminal '+(j.terminal_up?'up':'down');
  const gr=j.grown||{},st=j.stock||{};
  $('sys-circuit').textContent='grown: '+(gr.tok_min==null?'\u2026':gr.tok_min+' tok/min')+' · stock: '+(st.tok_min==null?'\u2026':st.tok_min+' tok/min');
  $('sysage').textContent=g.age_s==null?('(no fresh sample — '+(g.last_try||'warming up')+')'):('(sampled '+g.age_s+'s ago · '+(g.note||'')+')');
  if(g.cpu_pct!=null){cpuHist.push(g.cpu_pct);if(cpuHist.length>60)cpuHist.shift();}
  spark();
 }catch(e){}
}
sysTick();
async function activity(){
 try{const r=await fetch('/api/activity');const j=await r.json();viz.lastAct=j;
  j.grown=j.grown||{};j.stock=j.stock||{};
  if(++sysTickN%5===0)sysTick();
  const s=j.chat_series,rate=s.slice(20).reduce((a,b)=>a+b,0)/20;
  if(j.chat_open>0&&rate>0)vizPulse(Math.min(3,Math.ceil(rate/6)));
  if(viz.lastHands===undefined){viz.lastHands=j.guest_hands||0;viz.lastDones=j.guest_dones||0;}
  if((j.guest_hands||0)>viz.lastHands){
   vizPulse(Math.min(6,((j.guest_hands||0)-viz.lastHands)*2));viz.lastHands=j.guest_hands||0;}
  if((j.guest_dones||0)>viz.lastDones){
   vizPulse(Math.min(6,((j.guest_dones||0)-viz.lastDones)*2),false,true);viz.lastDones=j.guest_dones||0;}
  if(j.grown.running&&(j.grown.tokens||0)>0)vizPulse(1);
  if(viz.lastPrompts>=0&&j.grown.prompts>viz.lastPrompts)vizPulse(6,true);
  viz.lastPrompts=j.grown.prompts;
  $('now-chat').textContent=j.chat_open?
   'answering right now':'idle - nobody is chatting';
  $('now-computer').textContent=!j.terminal_up?
   'down - start liveterm (:8951)':
   (j.guest_active?
    ('working'+(j.guest_task?': '+j.guest_task:'')+(j.guest_queue?' \u00b7 '+j.guest_queue+' queued':'')):
    (j.guest_queue?j.guest_queue+' queued, waiting for the prompt':'idle at its prompt'));
  $('now-guest').textContent=(j.guest_hands||0)+' jobs run \u00b7 '+(j.guest_dones||0)+' reports back';
  $('now-circuit').textContent=j.grown.pass?
   'the agent run on the grown circuit passed':
   (j.grown.fail?
    'the grown-circuit agent run finished and failed - it hit its round limit on the old brain':
   (j.grown.running?
    ('running the agent - '+j.grown.prompts+
     ' questions answered so far, at token '+
     j.grown.tokens.toLocaleString()):'not running'));
  $('t-chat').innerHTML=j.chat_open?'<span class="dot ok"></span>live':'idle';
  $('t-chatbar').style.width=Math.min(100,rate*4)+'%';
  $('t-tok').textContent=(j.grown.tokens||0).toLocaleString();
  const tp=j.grown.prompts||0;
  $('t-tokbar').style.width=Math.min(100,tp*10)+'%';
  $('t-tok').title=tp+' prompts answered';
  $('vizlegend').textContent='the dots are real events: amber '
   +'crosses brain \u2192 computer when a job is handed over, blue '
   +'crosses back when the computer reports, and a white flash '
   +'crosses when the circuit experiment answers. amber ticks under '
   +'the computer orb are queued jobs. hover an orb for live numbers, '
   +'click one to open its tab. slow dots, on purpose - the computer '
   +'is slow.';
  $('vizstats').textContent=(j.guest_hands||0)+' jobs handed over \u00b7 '
   +(j.guest_dones||0)+' reports back \u00b7 '+(j.guest_queue||0)
   +' queued \u00b7 chat '+(j.chat_open?'live':'idle')+' \u00b7 circuit '
   +(j.grown.prompts||0)+' prompts';
  if(vizpopFor)$('vizpop').innerHTML=vizpopHTML(vizpopFor);
 }catch(e){}
 setTimeout(activity,1000);}
activity();
async function loadSettings(){
 const r=await fetch('/api/settings');const j=await r.json();
 $('s-backend').value=j.backend_url;$('s-model').value=j.model;
 $('s-temp').value=j.temperature;$('s-tokens').value=j.max_tokens;
 $('s-sys').value=j.system_prompt;
 $('s-guest').checked=j.guest_enabled!==false;
 $('s-budget').value=j.guest_budget==null?600:j.guest_budget;
 $('s-decider').checked=j.decider_enabled===true;
 $('s-deciderurl').value=j.decider_url||'';
 $('s-decidermodel').value=j.decider_model||'';
}
loadSettings();
$('save').onclick=async()=>{
 const body={backend_url:$('s-backend').value,model:$('s-model').value,
  temperature:parseFloat($('s-temp').value),
  max_tokens:parseInt($('s-tokens').value),system_prompt:$('s-sys').value,
  guest_enabled:$('s-guest').checked,
  guest_budget:parseInt($('s-budget').value),
  decider_enabled:$('s-decider').checked,
  decider_url:$('s-deciderurl').value,decider_model:$('s-decidermodel').value};
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
$('testdecider').onclick=async()=>{
 $('saved').textContent='testing\u2026';
 try{const r=await fetch('/api/status');const j=await r.json();const d=j.decider||{};
  $('saved').textContent=d.model?('ok: '+d.model):('decider: '+(!d.enabled?'off':('FAILED: '+(d.error||'no model'))));
 }catch(e){$('saved').textContent='FAILED: '+e;}
 setTimeout(()=>{if($('saved').textContent.startsWith('ok'))$('saved').textContent='';},4000);
};
function termURL(){return 'http://'+location.hostname+':8951/';}
$('termopen').onclick=()=>window.open(termURL(),'_blank');
$('termreload').onclick=()=>{$('term').src=termURL();};
async function mindAct(path,body,after){
 $('actmsg').textContent='working\u2026';
 try{const r=await fetch(path,{method:'POST',
  headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})});
  const j=await r.json();$('actmsg').textContent=j.msg||(j.ok?'done':'failed');
 }catch(e){$('actmsg').textContent='failed: '+e;}
 if(after)setTimeout(after,2500);
}
$('actsnap').onclick=()=>mindAct('/api/mirror',{},loadMind);
$('acttrace').onclick=()=>mindAct('/api/traces',{},loadMind);
$('actupkeep').onclick=()=>mindAct('/api/upkeep',
 {paused:!(lastMind&&lastMind.upkeep_paused)},
 ()=>{loadMind();setTimeout(loadMind,30000);});
const PAL=[
 ['Go: Overview','show the status dashboard',()=>document.querySelector('nav button[data-t="overview"]').click()],
 ['Go: Activity','interactive 3D activity',()=>document.querySelector('nav button[data-t="activity"]').click()],
 ['Go: Chat','talk to the built-in weights',()=>document.querySelector('nav button[data-t="chat"]').click()],
 ['Go: Terminal','the live Qwen-Linux guest',()=>document.querySelector('nav button[data-t="terminal"]').click()],
 ['Go: Desktop','framebuffer spike placeholder',()=>document.querySelector('nav button[data-t="desktop"]').click()],
 ['Go: Apps','programs on the Qwenputer',()=>document.querySelector('nav button[data-t="apps"]').click()],
 ['Go: System','guest CPU/RAM, engine, circuit rates',()=>document.querySelector('nav button[data-t="system"]').click()],
 ['Go: Settings','backend, model, prompt',()=>document.querySelector('nav button[data-t="settings"]').click()],
 ['Mind: snapshot now','mirror the guest mind to the host',()=>mindAct('/api/mirror',{},loadMind)],
 ['Mind: export traces','console sessions to tune episodes',()=>mindAct('/api/traces',{},loadMind)],
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
 if(e.key==='Escape'&&vizpopFor)closeVizPop();
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
        elif self.path == "/api/sys":
            self._json(get_sys())
        elif self.path == "/api/mind":
            m = _read_mind()
            m["mirror_state"] = _MIRROR_STATE[0]
            self._json(m)
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
        elif self.path == "/api/mirror":
            self._json(_mirror_now())
        elif self.path == "/api/traces":
            self._json(_traces_now())
        elif self.path == "/api/upkeep":
            self._json(_set_upkeep(bool(body.get("paused", True))))
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
            # chat-time computer use (2026-10-31): the model
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
                # - the 2026-10-31 live report ("chat sits working
                # while the brain runs"). A mention is prose, not
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
    threading.Thread(target=_sys_loop, daemon=True).start()
    print("qwenputer dashboard on %s:%d" % (a.host, a.port), flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
