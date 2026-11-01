"""Weekly task battery (phase 3): is the agent getting more capable?

Ten ground-truth tasks through a real guest, all harness verbs that
matter, with a scripted backend (deterministic, offline, fast - no
model latency) plus a real local fetch server for the fetch task:

    python3 -m chipformer.rv.battery Image --fast --fast-bin ./fastlinux
        [--report battery.json] [--prev battery-prev.json]

Files are verified from a final ground-truth dump (never from the
model's word); RESULT flow is verified from the backend's own prompt
log. The JSON report tracks pass/fail per task; --prev prints the
pass-rate delta so capability gains are measurable week to week.
"""

import argparse
import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from chipformer.rv.qwenbackend import fetch_backend_wrapper

HERE = Path(__file__).parent
ASK = (HERE / "env" / "linux" / "ask.sh").read_text()
AGENT = (HERE / "env" / "linux" / "agent.sh").read_text()
INSTALL = (
    "cat > /tmp/ask <<'QWEOF'\n" + ASK + "QWEOF\n"
    "cat > /tmp/agent <<'QWEOF'\n" + AGENT + "QWEOF\n"
    "cp /tmp/agent /tmp/hermes-agent\n"
    "chmod +x /tmp/ask /tmp/agent /tmp/hermes-agent\n"
    "mkdir -p /root/nb /root/skills /root/apps\n"
)
VERIFY = b"md5sum /tmp/ask /tmp/agent; echo MD5\"CHK\"\n"
HASHES = [hashlib.md5(ASK.encode()).hexdigest().encode(),
          hashlib.md5(AGENT.encode()).hexdigest().encode()]
DUMP = (b"echo BRAINZ:; cat /root/MEMORY.md /root/USER.md; echo PLANZ:; "
        b"cat /root/PLAN.md; echo SKILLZ:; "
        b"cat /root/skills/batskill.md; echo JOBSZ:; "
        b"wc -l < /root/jobs; echo BAT\"DONE\"\n")
FETCH_MARK = "battery-fetch-marker"
TASKS = [
    # (name, task line, scripted replies in call order)
    ("think",
     "think once, then run echo think-ok, then say what it printed",
     ["THINK: plan first, act second",
      "TOOL run echo think-ok",
      "SAY: thought and did"]),
    ("remember",
     "remember that the battery color is blue, then say remembered",
     ["TOOL remember-user [pref] battery color is blue",
      "SAY: remembered"]),
    ("recall",
     "search the pref section for blue, then say recall done",
     ["TOOL recall in:pref blue",
      "SAY: recall done"]),
    ("forget",
     "forget the battery color line, then say forgot",
     ["TOOL forget battery color",
      "SAY: forgot"]),
    ("plan",
     "write a two-step plan, check off step 1, then say planned",
     ["TOOL plan :: pstep1; pstep2",
      "TOOL plan done 1",
      "SAY: planned"]),
    ("delegate",
     "delegate saying the word subdone, then say got it",
     ["TOOL delegate say the word subdone",
      "SAY: subdone",
      "SAY: got it"]),
    ("skill",
     "save the batskill procedure, load it back, then say skilled",
     ["TOOL saveskill batskill :: to bat, run echo bat",
      "TOOL skill batskill",
      "SAY: skilled"]),
    ("schedule",
     "schedule a 45-minute probe, list jobs, remove job 1, say cleaned",
     ["TOOL schedule every 45 battery probe",
      "TOOL jobs",
      "TOOL unschedule 1",
      "SAY: cleaned"]),
    ("critique",
     "run echo c1 c2 c3 and c4, say draft words, keep a lesson, revise",
     ["TOOL run echo c1",
      "TOOL run echo c2",
      "TOOL run echo c3",
      "TOOL run echo c4",
      "SAY: draft words",
      "TOOL remember [lesson] critique works",
      "SAY: revised words"]),
    ("fetch",
     "fetch the local test page, then say fetched it",
     ["TOOL fetch FETCHURL",
      "SAY: fetched it"]),
]


class ScriptedBackend:
    """Canned replies in call order; every prompt is logged (the log
    is ground truth for RESULT flow: recalls, reports, ride-alongs)."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []
        self.model = "battery-scripted"

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if not self.replies:
            return "SAY: (script exhausted)"
        return self.replies.pop(0)


def units_of(text):
    units = []
    for line in text.splitlines(keepends=True):
        b = line.encode()
        units += [b[i:i + 256] for i in range(0, len(b), 256)] or [b""]
    return units


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--dtb",
                    default=str(HERE / "env" / "linux" / "sixtyfourmb.dtb"))
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--fast-bin", default=None)
    ap.add_argument("--max-steps", type=int, default=3_000_000_000)
    ap.add_argument("--slice", type=int, default=250_000)
    ap.add_argument("--report", default="battery.json")
    ap.add_argument("--prev", default=None)
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    body = ("<html>%s %s</html>" % (FETCH_MARK, "y" * 5000)).encode()

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    fetch_url = "http://127.0.0.1:%d/t" % srv.server_port

    replies = []
    for name, _, script in TASKS:
        replies.extend(
            r.replace("FETCHURL", fetch_url) for r in script)
    scripted = ScriptedBackend(replies)
    backend = fetch_backend_wrapper(scripted)

    if args.fast:
        from chipformer.rv.fastmachine import FastLinuxMachine
        m = FastLinuxMachine(args.image, args.dtb, backend=backend,
                             binary=args.fast_bin)
    else:
        from chipformer.rv.linux import LinuxMachine
        m = LinuxMachine()
        m.on_qwen_prompt = backend
        m.load_image(Path(args.image).read_bytes())
        m.boot(Path(args.dtb).read_bytes())

    chunks = units_of(INSTALL)
    logf = open(args.log, "wb") if args.log else None
    logged = 0
    phase = 0
    phase_slice = 0
    last_feed_slice = 0
    unit = None
    unit_mark = 0
    verify_mark = 0
    retries = 0
    fed = set()
    slices = 0
    task_i = 0
    answers_seen = 0
    task_slice = 0
    dump_mark = 0
    while m.steps < args.max_steps and not m.halted:
        m.run(args.slice)
        slices += 1
        out = bytes(m.out)
        if logf and len(out) > logged:
            logf.write(out[logged:])
            logf.flush()
            logged = len(out)
        if phase == 0:
            if out.rstrip().endswith(b"buildroot login:"):
                m.feed_input(b"root\n")
                phase, phase_slice = 1, slices
        elif phase == 1:
            rel = slices - phase_slice
            if rel >= 2 and "login2" not in fed:
                m.feed_input(b"root\n")
                fed.add("login2")
            if rel >= 6 and "canary1" not in fed:
                m.feed_input(b"echo CANARY_$((40+2))\n")
                fed.add("canary1")
            if rel >= 16 and "canary2" not in fed:
                m.feed_input(b"echo CANARY_$((40+2))\n")
                fed.add("canary2")
            if b"CANARY_42" in out:
                phase = 2
        elif phase == 2:
            if unit is not None:
                if len(out) - unit_mark >= len(unit) - 4 \
                        or slices - last_feed_slice >= 200:
                    unit = None
            elif chunks:
                unit = chunks.pop(0)
                m.feed_input(unit)
                unit_mark = len(out)
                last_feed_slice = slices
            else:
                verify_mark = len(out)
                m.feed_input(VERIFY)
                phase = 25
        elif phase == 25:
            tail = out[verify_mark:]
            if b"MD5CHK" in tail:
                if all(h in tail for h in HASHES):
                    print("[battery] install verified byte-exact",
                          flush=True)
                    phase = 3
                elif retries < 3:
                    retries += 1
                    print(f"[battery] install torn (retry {retries})",
                          flush=True)
                    chunks = units_of(INSTALL)
                    phase = 2
                else:
                    print("BATTERY: FAIL (install never verified)")
                    return 1
        elif phase == 3:
            n_ans = out.count(b"\nAGENT_ANSWER:")
            if task_i >= len(TASKS):
                dump_mark = len(out)
                m.feed_input(DUMP)
                phase = 4
            elif n_ans > answers_seen:
                answers_seen = n_ans
                task_i += 1
                if task_i < len(TASKS):
                    m.feed_input(
                        ("/tmp/hermes-agent '%s'\n" % TASKS[task_i][1]
                         ).encode())
                    task_slice = slices
            elif task_i == 0 and answers_seen == 0 and "task0" not in fed:
                m.feed_input(
                    ("/tmp/hermes-agent '%s'\n" % TASKS[0][1]).encode())
                fed.add("task0")
                task_slice = slices
            elif slices - task_slice >= 1600 and task_i < len(TASKS):
                print(f"[battery] session {task_i} timed out - moving on",
                      flush=True)
                m.feed_input(b"\x03")
                answers_seen = n_ans
                task_i += 1
                if task_i < len(TASKS):
                    m.feed_input(
                        ("/tmp/hermes-agent '%s'\n" % TASKS[task_i][1]
                         ).encode())
                task_slice = slices
        elif phase == 4:
            if b"BATDONE" in out[dump_mark:]:
                break
        if slices % 40 == 0:
            print(f"[{m.steps:,} steps, phase={phase}, task={task_i}, "
                  f"out={len(out)} B]", flush=True)

    out = bytes(m.out)
    dump = out[dump_mark:] if dump_mark else b""

    def seg(start, end=None):
        i = dump.rfind(start)
        if i < 0:
            return b""
        s = dump[i + len(start):]
        if end:
            j = s.find(end)
            if j >= 0:
                s = s[:j]
        return s

    answers = [a.split(b"\n")[0] for a in
               out.split(b"\nAGENT_ANSWER:")[1:]]
    brain = seg(b"BRAINZ:", b"PLANZ:")
    checks = {
        "think": any(b"thought and did" in a for a in answers),
        "remember": any(b"remembered: [pref] battery color is blue"
                      in p.encode() for p in scripted.prompts),
        "recall": any(b"battery color is blue" in p.encode()
                      for p in scripted.prompts),
        "forget": b"battery color" not in brain,
        "plan": b"- [x] pstep1" in dump and b"- [ ] pstep2" in dump
        and any(b"Pending plan steps" in p.encode()
                for p in scripted.prompts),
        "delegate": any(b"got it" in a for a in answers)
        and any(b"sub-agent report: subdone" in p.encode()
                for p in scripted.prompts),
        "skill": b"to bat, run echo bat" in dump,
        "schedule": b"JOBSZ:" in dump and
        dump.split(b"JOBSZ:")[-1].split()[0] == b"0",
        "critique": any(b"revised words" in a for a in answers)
        and b"[lesson] critique works" in brain,
        "fetch": any(FETCH_MARK.encode() in p.encode()
                     for p in scripted.prompts),
    }
    report = {"time": time.strftime("%F %T"),
              "passed": sum(1 for v in checks.values() if v),
              "total": len(checks),
              "checks": {k: bool(v) for k, v in checks.items()}}
    with open(args.report, "w") as f:
        json.dump(report, f, indent=1)
    print("BATTERY: %d/%d %s" % (
        report["passed"], report["total"],
        " ".join("%s=%s" % (k, "PASS" if v else "FAIL")
                 for k, v in checks.items())), flush=True)
    if args.prev:
        try:
            prev = json.load(open(args.prev))
            print("BATTERY DELTA: %d/%d -> %d/%d" % (
                prev["passed"], prev["total"],
                report["passed"], report["total"]), flush=True)
        except Exception as e:
            print("BATTERY DELTA: no previous report (%s)" % e, flush=True)
    srv.shutdown()
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
