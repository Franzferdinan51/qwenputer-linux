"""Automated scheduler test for the debloated-Hermes guest framework:
boot, log in, install ask + agent + agentshell (verified byte-exact
by md5sum in the guest, retried if torn - see agenttest.py), start
the foreground supervisor, have the agent schedule a job 1 minute
out, then WAIT - typing nothing - and verify from ground truth, not
the model's word, that the job fired on its own:

    python3 -m chipformer.rv.schedtest Image [--backend builtin]
        [--fast --fast-bin ./fastlinux] [--log sched.out]

Why a foreground supervisor: a background process cannot read this
guest's console (its reads return empty lines instantly - measured),
so the old background agentd could never hold a QWEN: session. The
agentshell owns the console in the foreground; the test drives it
exactly the way a user at the terminal does.

PASS requires ALL of:
  1. the scheduling session ends with an AGENT_ANSWER,
  2. /tmp/melon.txt really contains m e l o n (checked with od in the
     final dump) - written by the supervisor's run of the job, with
     nobody asking again (phase 4 types nothing),
  3. /root/jobs.log carries the fired job's outcome line (continuity),
  4. the fired one-shot job removed itself from /root/jobs (the brain
     may schedule paraphrase variants too - the criteria follow the
     job that actually fired, which is the scheduler's real contract).
"""

import argparse
import hashlib
import re
import time
from pathlib import Path

from chipformer.rv.linux import LinuxMachine
from chipformer.rv.qwenbackend import make_backend

HERE = Path(__file__).parent
ASK = (HERE / "env" / "linux" / "ask.sh").read_text()
AGENT = (HERE / "env" / "linux" / "agent.sh").read_text()
ASHELL = (HERE / "env" / "linux" / "agentshell.sh").read_text()
SCHED_TASK = ("do NOT do this yourself now. Use TOOL schedule 1 to "
              "schedule a job for 1 minute from now. The job is: write "
              "the word melon into /tmp/melon.txt then read it back "
              "and say what it says")
INSTALL = (
    "cat > /tmp/ask <<'QWEOF'\n" + ASK + "QWEOF\n"
    "cat > /tmp/agent <<'QWEOF'\n" + AGENT + "QWEOF\n"
    "cp /tmp/agent /tmp/hermes-agent\n"
    "cat > /tmp/agentshell <<'QWEOF'\n" + ASHELL + "QWEOF\n"
    "chmod +x /tmp/ask /tmp/agent /tmp/hermes-agent /tmp/agentshell\n"
    "mkdir -p /root/nb /root/skills /root/apps\n"
)
VERIFY = b"md5sum /tmp/ask /tmp/agent /tmp/agentshell; echo MD5\"CHK\"\n"
HASHES = [hashlib.md5(x.encode()).hexdigest().encode()
          for x in (ASK, AGENT, ASHELL)]


def units_of(text: str):
    """Feed units: one line each (long lines split at 256 B), fed one
    at a time and released by their own echo - the guest tty read
    buffer drops bytes past ~4 KB unread (see agenttest.py)."""
    units = []
    for line in text.splitlines(keepends=True):
        b = line.encode()
        units += [b[i:i + 256] for i in range(0, len(b), 256)] or [b""]
    return units


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--dtb", default=str(HERE / "env" / "linux" / "sixtyfourmb.dtb"))
    ap.add_argument("--backend", choices=("echo", "lmstudio", "builtin"), default="builtin")
    ap.add_argument("--log", default=None)
    ap.add_argument("--max-steps", type=int, default=1_500_000_000)
    ap.add_argument("--slice", type=int, default=250_000)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--fast-bin", default=None)
    args = ap.parse_args()

    backend = make_backend(args.backend, None)
    if args.fast:
        from chipformer.rv.fastmachine import FastLinuxMachine
        m = FastLinuxMachine(args.image, args.dtb, backend=backend,
                             binary=args.fast_bin)
    else:
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
    install_failed = False
    fed = set()
    t0 = time.time()
    slices = 0
    tail_start = 0
    sup_mark = 0
    phase4_slice = 0
    dump_mark = 0
    dump_tries = 0
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
            # line-paced feeding: one unit in flight, released by its
            # own echo (the tty read buffer drops bytes past ~4 KB
            # unread, which tore heredoc lines on the 15 KB agent.sh)
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
                    print("[schedtest] install verified byte-exact in the guest",
                          flush=True)
                    sup_mark = len(out)
                    m.feed_input(b"/tmp/agentshell\n")
                    phase = 3
                elif retries < 3:
                    retries += 1
                    print(f"[schedtest] install torn (retry {retries})", flush=True)
                    chunks = units_of(INSTALL)
                    phase = 2
                else:
                    install_failed = True
                    break
        elif phase == 3:
            # the supervisor is up when its prompt appears after the
            # launch mark (the heredoc echo also contains the string,
            # hence the mark); then type the scheduling task at it -
            # the supervisor evals it in the foreground, like a user
            if "taskfed" not in fed and b"agent> " in out[sup_mark:]:
                m.feed_input(f"/tmp/hermes-agent {SCHED_TASK}\n".encode())
                fed.add("taskfed")
            elif "taskfed" in fed and b"\nAGENT_ANSWER:" in out:
                phase = 4
                phase4_slice = slices
                print("[schedtest] job scheduled; waiting, hands off, "
                      "for the supervisor to fire it", flush=True)
        elif phase == 4:
            # HANDS OFF: typing now would queue commands ahead of the
            # fired session's answers. The fired session announces
            # itself in the console stream; its AGENT_ANSWER is the
            # second one overall. Only dump once the supervisor is
            # provably back at its prompt (an "agent> " AFTER that
            # answer) - feeding mid-bookkeeping risks the line being
            # eaten by the wrong reader. Deadline: dump anyway.
            answers = out.count(b"\nAGENT_ANSWER:")
            idx = out.rfind(b"AGENT_ANSWER:")
            at_prompt = idx > 0 and b"agent> " in out[idx:]
            if (answers >= 2 and at_prompt and slices - phase4_slice >= 40) \
                    or slices - phase4_slice >= 2400:
                tail_start = len(out)
                dump_mark = slices
                dump_tries = 1
                m.feed_input(b"od -c /tmp/melon.txt; echo JOBSFILE:; "
                             b"cat /root/jobs; echo JOBSLOG:; "
                             b"cat /root/jobs.log; echo ALL\"DONE\"\n")
                phase = 5
        elif phase == 5:
            if b"ALLDONE" in out[tail_start:]:
                break
            # the dump line can be lost if it landed while the
            # supervisor was mid-bookkeeping - re-feed it a few times.
            # tail_start stays at the FIRST feed: the verdict window
            # must cover every dump output (resetting it on a re-feed
            # once wiped a good dump from the window and failed a
            # passing run)
            if slices - dump_mark >= 200 and dump_tries < 4:
                dump_mark = slices
                dump_tries += 1
                m.feed_input(b"od -c /tmp/melon.txt; echo JOBSFILE:; "
                             b"cat /root/jobs; echo JOBSLOG:; "
                             b"cat /root/jobs.log; echo ALL\"DONE\"\n")
            elif slices - dump_mark >= 800:
                break
        if slices % 40 == 0:
            print(f"[{m.steps:,} steps, phase={phase}, out={len(out)} B, "
                  f"{m.steps / max(time.time() - t0, 1e-9):,.0f} steps/s]", flush=True)

    out = bytes(m.out)
    if install_failed:
        print("SCHEDULER ROUND TRIP: FAIL (install never verified byte-exact)")
        return 1
    tail = out[tail_start:] if tail_start else b""
    answer_seen = b"\nAGENT_ANSWER:" in out
    # line-start anchor: the install heredoc's echo of agentshell.sh
    # itself contains "running scheduled job" mid-line and must not
    # count as a job actually firing
    fired_seen = b"\n[agentshell] running scheduled job" in out
    file_ok = b"m   e   l   o   n" in tail
    # the brain may schedule paraphrase variants of the job, so the
    # honest criteria are about the job that FIRED: it must be logged
    # in jobs.log with its outcome (continuity) and it must be gone
    # from /root/jobs afterwards (one-shot cleanup)
    # read the FINAL dump's sections: the markers also appear inside
    # the typed dump command's own echo (and in earlier re-dumps), so
    # split-based parsing reads the wrong slice - anchor on the LAST
    # occurrence of each marker instead
    def section(start: bytes, end: bytes = None) -> bytes:
        i = tail.rfind(start)
        if i < 0:
            return b""
        seg = tail[i + len(start):]
        if end:
            j = seg.find(end)
            if j >= 0:
                seg = seg[:j]
        return seg

    jobs_part = section(b"JOBSFILE:", b"JOBSLOG:")
    log_part = section(b"JOBSLOG:")
    fired = re.findall(rb"\| job: (.*?) =>", log_part)
    log_ok = bool(fired) and b"melon" in log_part
    # compare per job line, not substring: a leftover paraphrase that
    # merely CONTAINS a fired job's text must not fail the removal check
    remaining = [ln.split(b"|", 4)[-1].strip()
                 for ln in jobs_part.split(b"\n") if ln.strip()]
    jobs_ok = bool(fired) and all(t.strip() not in remaining for t in fired)
    print(f"=== steps={m.steps:,} wall={time.time() - t0:.1f}s "
          f"backend={args.backend} ===")
    ok = answer_seen and fired_seen and file_ok and log_ok and jobs_ok
    print("SCHEDULER ROUND TRIP:", "PASS" if ok else
          f"FAIL (answer={answer_seen} fired={fired_seen} file={file_ok} "
          f"log={log_ok} fired_job_removed={jobs_ok})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
