"""Verb-coverage test for the guest agent framework: boot, log in,
install ask + agent (md5-verified, see agenttest.py), then run SIX
foreground sessions, one per framework verb family, and verify each
from ground truth in a final dump - never from the model's word:

    python3 -m chipformer.rv.verbtest Image [--backend builtin]
        [--fast --fast-bin ./fastlinux] [--log verb.out]

  1. apps      saveapp doubler + app doubler 21  -> /root/apps/doubler
               exists and a session answer reports 42
  2. skills    saveskill greeting                -> /root/skills/
               greeting.md on disk with the procedure in it
  3. remember  remember ... Zorptown             -> /root/BRAIN.md
  4. recall    recall Zorptown (a NEW session)   -> its answer
               repeats what the memory says
  5. schedule  schedule every 45 ... then unschedule 1 in the same
               session                           -> /root/jobs empty
  6. reflect   a 6-tool-call task                -> the harness
               spends its reflection turn ("[agent] reflection:")
"""

import argparse
import hashlib
import time
from pathlib import Path

from chipformer.rv.linux import LinuxMachine
from chipformer.rv.qwenbackend import make_backend

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
TASKS = [
    # stepwise phrasing on purpose: outcome-phrased tasks ("build an
    # app", "keep this in memory") made the seat CLAIM the outcome
    # without invoking the verb (verb1.out, 2026-10-31) - the same
    # simulate-ahead disease as the circuit agent. Leading with the
    # exact verb line is what gets verbs invoked at this brain size.
    "first reply with exactly this line: TOOL saveapp doubler :: "
    "echo $(( $1 * 2 )) - after its result, reply: TOOL app "
    "doubler 21 - after that result, SAY: what it printed",
    "save a procedure as a skill: TOOL saveskill greeting :: to "
    "greet someone, run echo hello there and then say hello there. "
    "then reply SAY: skill saved",
    "reply with exactly this line: TOOL remember the capital of "
    "the test country is Zorptown. after the result, SAY: remembered",
    "search your memory for Zorptown: TOOL recall Zorptown then "
    "SAY: what the memory says",
    "first: TOOL schedule every 45 check the disk space. after "
    "the result: TOOL jobs. after that result: TOOL unschedule "
    "1. after that, SAY: cleaned up",
    "write the word one into /tmp/r1.txt, write two into "
    "/tmp/r2.txt, write three into /tmp/r3.txt, then read all "
    "three back, then SAY: the three words",
]
DUMP = (b"echo APPS:; ls /root/apps; echo SKILLFILE:; "
        b"cat /root/skills/greeting.md; echo BRAINZ:; "
        b"grep Zorptown /root/BRAIN.md; echo JOBSC:; "
        b"wc -l < /root/jobs; echo VERB\"DONE\"\n")


def units_of(text: str):
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
    ap.add_argument("--max-steps", type=int, default=3_000_000_000)
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
                    print("[verbtest] install verified byte-exact in the guest",
                          flush=True)
                    phase = 3
                elif retries < 3:
                    retries += 1
                    print(f"[verbtest] install torn (retry {retries})", flush=True)
                    chunks = units_of(INSTALL)
                    phase = 2
                else:
                    install_failed = True
                    break
        elif phase == 3:
            # run the sessions one by one; each ends with exactly one
            # AGENT_ANSWER line (success or honest failure). A session
            # that produces none within the slice budget is abandoned
            # (Ctrl-C) and counted as a miss for its verb.
            n_ans = out.count(b"\nAGENT_ANSWER:")
            if task_i >= len(TASKS):
                dump_mark = len(out)
                m.feed_input(DUMP)
                phase = 4
            elif n_ans > answers_seen:
                answers_seen = n_ans
                task_i += 1
                if task_i < len(TASKS):
                    m.feed_input(f"/tmp/hermes-agent '{TASKS[task_i]}'\n".encode())
                    task_slice = slices
            elif task_i == 0 and answers_seen == 0 and "task0" not in fed:
                m.feed_input(f"/tmp/hermes-agent '{TASKS[0]}'\n".encode())
                fed.add("task0")
                task_slice = slices
            elif slices - task_slice >= 1600 and task_i < len(TASKS):
                print(f"[verbtest] session {task_i} timed out - moving on",
                      flush=True)
                m.feed_input(b"\x03")
                answers_seen = n_ans
                task_i += 1
                # the abandoned session produced no answer, so the
                # advance branch above will never fire for it - feed the
                # next task here or every session after a timeout is
                # silently skipped too
                if task_i < len(TASKS):
                    m.feed_input(f"/tmp/hermes-agent '{TASKS[task_i]}'\n".encode())
                task_slice = slices
        elif phase == 4:
            if b"VERBDONE" in out[dump_mark:]:
                break
        if slices % 40 == 0:
            print(f"[{m.steps:,} steps, phase={phase}, task={task_i}, "
                  f"out={len(out)} B]", flush=True)

    out = bytes(m.out)
    if install_failed:
        print("VERB COVERAGE: FAIL (install never verified byte-exact)")
        return 1
    dump = out[dump_mark:] if dump_mark else b""

    def seg(start: bytes, end: bytes = None) -> bytes:
        i = dump.rfind(start)
        if i < 0:
            return b""
        s = dump[i + len(start):]
        if end:
            j = s.find(end)
            if j >= 0:
                s = s[:j]
        return s

    apps_sec = seg(b"APPS:", b"SKILLFILE:")
    skill_sec = seg(b"SKILLFILE:", b"BRAINZ:")
    brain_sec = seg(b"BRAINZ:", b"JOBSC:")
    jobs_sec = seg(b"JOBSC:", b"VERBDONE")
    answers = [l for l in out.split(b"\nAGENT_ANSWER:")]
    # 42 must be ON an answer line: a bare b"42" in out matches the
    # CANARY_42 login probe and would pass this check for free
    apps_ok = b"doubler" in apps_sec and any(
        b"42" in a.split(b"\n", 1)[0] for a in answers[1:])
    skill_ok = b"hello" in skill_sec
    recall_ok = b"Zorptown" in brain_sec and any(
        b"Zorptown" in a for a in answers[1:])
    unsched_ok = b"\n0" in jobs_sec.replace(b"\r", b"") or jobs_sec.strip().startswith(b"0")
    # line-start anchor: the install echo of agent.sh contains
    # "[agent] reflection:" mid-line and must not count as a real
    # reflection turn
    reflect_ok = b"\n[agent] reflection:" in out
    print(f"=== steps={m.steps:,} wall={time.time() - t0:.1f}s "
          f"backend={args.backend} ===")
    ok = apps_ok and skill_ok and recall_ok and unsched_ok and reflect_ok
    print("VERB COVERAGE:", "PASS" if ok else
          f"FAIL (apps={apps_ok} skill={skill_ok} recall={recall_ok} "
          f"unschedule={unsched_ok} reflection={reflect_ok})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
