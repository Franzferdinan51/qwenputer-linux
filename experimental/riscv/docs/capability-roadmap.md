# Capability roadmap — making this a highly capable model

Duckets' order (2026-11-01): figure out what we can do with this
system — recursive thought, memories, and the rest — to make it
highly capable. This doc is the verified answer: what exists, what
is confirmed missing, and what to build in which order.

## Verified findings (2026-11-01, against this tree)

- **No memory-upkeep job exists.** `build_agent.sh` and the guest
  scripts schedule nothing that curates BRAIN.md; "curated" appears
  only in comments and the wake-up fold. Memory grows by append
  (`TOOL remember` >> BRAIN.md) and is never consolidated.
- **The guest has no network.** No NIC in `fastlinux.c` / `linux.py` /
  `fastmachine.py`, no wget/curl/dhcp in the installer. The only
  route to the internet is host-side (a bridge/fetch tool).
- **The brain cannot think without acting.** Every model reply must
  be a TOOL verb or SAY: — the `*)` catch-all (`agent.sh` ~line 499)
  delivers anything else as the final answer and exits. A `THINK:`
  scratchpad needs a new harness branch before the catch-all: small.
- **Delegation is blocked by the one-agent lock** (`mkdir
  /tmp/agent.lockdir`, `AGENTD=1` defers with AGENT_BUSY). A child
  session needs lock depth or a fenced `TOOL delegate` runner.
- **Dead tasks leave nothing.** A session killed at the round cap
  writes one history line; no plan file, no resume.

## What exists today (the foundation)

15-round agent loop with echo/repeat/circle/loop guards; layered
memory (BRAIN.md + /root/nb notebook + /root/history); skills
(save/skill/fix); self-built gap apps; scheduled jobs + heartbeat
checks with continuity; one-turn reflection after ≥4 tool calls;
seat models (Qwen 3.5 4B, QWEN_MODEL/QWEN_FAMILY seatable, GGUF-safe);
dashboard chat with FIFO delegation queue; in-weights circuit
experiment (stock pass, grown 2 MB run).

## Candidate capabilities

Recursive thought:
1. `THINK:` scratchpad turns (think→act→observe→think). Small.
2. Draft→critique→revise: extend the 1-turn reflection. Small.
3. `TOOL delegate <subtask>`: fenced child sessions, lock depth,
   split round budget. Medium.
4. PLAN.md with steps + status; resume after cap/interrupt. Small.

Memory:
5. Nightly upkeep job: compress history→facts, drop stale, hold
   BRAIN.md ~2k chars (the framework doc's own design). Small.
6. Structured BRAIN.md (facts/preferences/lessons/projects, dated).
   Small.
7. FAILURES.md consulted before acting. Small.
8. Episodic memory: one file per finished task. Small-medium.
9. Host-side mind mirror (jobs+memory+skills survive reboot). Medium.

Compounding skills/apps:
10. Tested skills: self-test before the harness trusts one. Small.
11. Seed skill library + gaps/traces → seat tune data export. Medium.

Senses and hands:
12. Host fetch tool (URL→text via the bridge). Small-medium. The
    biggest capability-per-line item: no other internet route exists.
13. Wall-clock seeding at boot (`date -s` once). Small.
14. Dashboard Jobs / Built-apps / Memory views (P3 remainder). Medium.

Proving "highly capable":
15. Task battery: 15–20 ground-truth tasks, weekly pass rate
    (`verbtest.py` is the seed). Medium.

Endgame (context, not scheduled):
16. Fused/GGUF brain seat (supported today via QWEN_MODEL).
17. Persistent-kernel / in-weights Linux snapshot.

## Suggested build order

1. **Mind upgrade** (1, 2, 5, 6, 7): better thinking + memory first;
   it multiplies everything after.
2. **Reach upgrade** (3, 4, 12): split work, finish long tasks, learn
   from the internet.
3. **Immortality upgrade** (9, 15, 11): survive reboots, measurable
   gains, tune the seat on its own traces.

## Status (2026-11-01): all three phases built

- Phase 1+2 guest verbs live in `env/linux/agent.sh` (`THINK:`,
  critique/revise, `[section]` memory + `in:` recall, `forget`,
  `delegate`, `plan`, `fetch`, `FAILURES.md` + PITFALLS); the upkeep
  job self-seeds from `agentshell.sh`; host fetching rides the QWEN:
  protocol via `qwenbackend.fetch_backend_wrapper` (no C change).
- Phase 3 host tools: `mindmirror.py` (15-min LaunchAgent snapshots
  to `~/qwenputer-test/mind`, restore via chunked printf),
  `battery.py` (10 ground-truth tasks, JSON report + `--prev`
  delta), `traceexport.py` (transcript -> tune episodes),
  installer seed skills (`upkeep`, `plan-long-task`).
- Proven: sims 41/41 (sh + dash), battery 10/10 on a scratch guest,
  trace export 10 eps / 32 turns from the battery log, live snapshot
  8 files, pytest 42/42.
- Live note: the running guest keeps its boot-time scripts until the
  next restart; the fetch wrapper and seed skills activate then. The
  mirror timer runs now and needs nothing from the guest.
