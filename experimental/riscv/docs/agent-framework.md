# A full agent architecture for the Qwenputer brain — small enough to work

Duckets' order (2026-10-31): the Qwenputer should be able to make its
own brain cron jobs and schedule jobs — a full agent architecture,
researched against how Hermes Agent, OpenClaw and the minimal-loop
frameworks do it, but sized for THIS brain and THIS guest: a 4B model
(today; the tuned Ornith 9B tomorrow), busybox shell, no Python in
the guest, 64 MB of RAM, and a machine whose every instruction is
precious.

## What the research says (patterns worth stealing)

- **Hermes Agent** (NousResearch): the agent loop itself is the core;
  around it sit a cron scheduler, layered memory, and skills. Three
  details matter here: (1) cron jobs are files with a `deliver`
  target and natural-language tasks — the model creates them by
  describing them; (2) `continuity` — a job's last output is folded
  into its next run's prompt, so recurring work remembers itself;
  (3) *monitor jobs skip the model entirely when nothing changed* —
  the scheduler is code, the model only wakes when there is work.
  Memory = two small curated files (facts, preferences) injected at
  session start + searchable history. Skills = saved procedures the
  agent writes from experience.
- **OpenClaw**: a gateway owns everything; scheduling is TWO
  mechanisms — **cron** (exact times, isolated sessions, persisted
  jobs) and **heartbeat** (a periodic main-session turn that reads a
  tiny checklist file, batches small checks, and replies
  `HEARTBEAT_OK` when nothing needs attention). Standing orders are
  persistent instructions injected into every session.
- **Minimal loops** (smolagents, simple-agent-loop ~250 lines,
  7/24 Office ~3,500 lines zero-framework): an agent is a loop —
  prompt, one action, observe, repeat — with a hard iteration cap,
  session compaction, and tools as plain functions. Nothing else is
  load-bearing. 7/24 Office additionally proves runtime tool
  creation: the agent writes and keeps new tools as it works.

## The architecture (all shell, a few hundred lines total)

Everything lives in the guest and talks to the brain only through
the existing QWEN: contract. The brain never sees machinery — it
sees a slightly bigger verb set.

1. **The loop (exists)** — `agent.sh`: one `TOOL` per reply,
   `RESULT:` back, `SAY:` to finish, guards against gluing,
   simulation and repeats. This is the smolagents lesson kept.
2. **New verbs, handled by the harness itself** (not shell commands):
   - `TOOL schedule every <dur> <task>` / `TOOL schedule at <when> <task>`
     — appends a job to `/root/jobs`. The brain creates its own
     cron jobs by *saying so*, the Hermes way.
   - `TOOL jobs` — lists its jobs with last results.
   - `TOOL unschedule <id>` — removes one.
   - `TOOL remember <fact>` / `TOOL recall` — the curated brain file
     (below), distinct from raw shell file access.
3. **The supervisor — `agentshell.sh` (new)**: the console's ONE
   foreground process, part shell, part scheduler. **Why not a
   background ticker (measured, 2026-10-31): a background process
   cannot read this guest's console — its `read` returns an empty
   line instantly, even on a silent console — so a background
   daemon can never hold a QWEN: session; its jobs fired and died
   on empty answers while the model answers sat unread in the
   console queue.** The appliance fix: the supervisor owns the
   console in the foreground. Typed lines are shell commands (or
   `hermes-agent "task"`), run in the foreground; an idle 60 s
   (`read -t`) is the tick: heartbeat checks first, then due jobs
   from `/root/jobs` — **if nothing is due, no model call happens**
   (the Hermes monitor lesson — on a slow machine, model calls
   are the expensive thing). A due job runs through `hermes-agent`
   as an isolated foreground session; its outcome is appended to
   `/root/jobs.log` and its one-line summary written back into
   its jobs-file entry (**continuity**: the next run's prompt
   opens with "last time you found: ..."). One job at a time,
   always — and exactly one console reader ever, so the QWEN:
   protocol needs no arbitration. `exit` returns to the plain
   shell; while the supervisor is not running, jobs wait.
4. **Memory, layered the Hermes way**:
   - `/root/nb` — the notebook (exists): free-form working memory.
   - `/root/BRAIN.md` — a small curated file (facts, standing
     orders, lessons) that the harness folds into every wake-up
     prompt. This is recursive memory with an editor: the agent
     maintains the file that maintains its context.
   - `/root/jobs.log` — what its scheduled self actually did.
5. **Skills (phase 2)**: `/root/skills/<name>` — shell procedures
   the agent writes for itself after solving something once, then
   invokes by name (`TOOL skill <name> <args>`). Hermes' procedural
   memory, shell-sized.
6. **Self-built apps — gap prosthetics, not user apps** (Duckets,
   2026-10-31): the agent builds small programs FOR ITSELF, to cover
   its own gaps, the way the original Qwenputer used a calculator.
   Can't multiply reliably → it builds a calc app once and calls it
   forever. Can't remember → a notebook app. Can't wait → a timer.
   - `TOOL saveapp <name> :: <one-line shell program>` — the harness
     writes it to `/root/apps/<name>` and makes it executable. (The
     protocol is one line; a one-liner with `;` separators is a
     whole program in shell.)
   - `TOOL app <name> <args>` — run one of its own apps.
   - `TOOL apps` — list what it has built, with the gap each covers.
   These are the agent's exclusively — never presented as user
   apps. The dashboard may show them read-only, for transparency
   (what has it built, and why). The SYS prompt teaches the habit:
   *hit a gap → build the prosthetic once → use it every time*, and
   log the gap in the notebook's `gaps` page — that log doubles as
   training data for the seat tunes.
7. **Guards** (a self-scheduling agent needs brakes):
   - at most 16 jobs; a job run may create at most 2 new jobs;
   - per-run round cap inherited from the harness; a job that
     fails 3 runs in a row disables itself and logs why;
   - kill switch: the file `/root/agentd.off` stops job firing;
   - logs trim themselves; the ticker never runs two jobs at once.

## The debloat map — Hermes in, Hermes out

Duckets (2026-10-31): take as much of Hermes Agent as possible —
its agent style and system, self-learning included — debloated:
redundant and useless parts removed, everything else shrunk to
shell. Source of truth for the patterns: the Hermes source on the
Mac (`~/.hermes/hermes-agent/`: `cron/`, `tools/cronjob_tools.py`,
`tools/memory_tool.py`, `tools/skill_manager_tool.py`, `SOUL.md`).

**KEEP (each maps to a piece of this framework):**
- the agent loop itself + hard iteration cap → `agent.sh`, exists
- `SOUL.md` identity → `/root/SOUL.md`, folded into every wake-up
  prompt; the guest agent may edit its own within bounds
- cron with natural-language jobs + delivery → the `agentshell`
  supervisor + the jobs file; delivery = the jobs log + the
  notebook (no platforms here)
- `continuity` → last output folded into the next run's prompt
- heartbeat checklist → `/root/HEARTBEAT.md`: the supervisor
  runs the shell checks first; the model wakes ONLY if a check
  flags something (also Hermes' monitor rule: no change, no
  model call)
- layered memory → BRAIN.md (curated) + notebook + `/root/history`
  (every finished task appended; recall = grep, no database)
- approval safety → a harness denylist: destructive commands
  (`rm -rf /`, mkfs, dd to a device, fork bombs) are refused and
  logged, whoever asked
- **the self-learning loop** (the definite want). Mechanics checked
  against the upstream GitHub repo (main @ c419cf7, 2026-10-31 —
  Duckets directed: the repo, not the installed Mac copy):
  upstream triggers skill creation after a task with **5+ tool
  calls**, keeps curated memory deliberately tiny (its MEMORY.md
  cap ≈ 2,200 chars — curation over bulk), nudges the agent
  periodically to persist what a session taught it, refines skills
  *during* use, and searches past sessions (FTS5 + LLM summary
  there; grep here). Ours, shrunk to a 9-round agent:
  1. *skill birth* — when a task finishes after ≥4 tool calls, or
     after a failure→success recovery, the harness spends ONE
     reflection turn before delivering the answer: "save how you
     did that: `TOOL saveskill <name> :: <steps>` — or a fact worth
     keeping: `TOOL remember <fact>`"
  2. *skill use* — SYS teaches: check `TOOL skills` before
     inventing an approach; `TOOL skill <name>` loads the procedure.
     Skill files follow the agentskills.io shape (name, when to
     use, steps, pitfalls), as upstream's do
  3. *skill repair* — a failed run that used a skill earns the
     nudge "`TOOL fixskill <name> :: <what to do differently>`",
     appended to the skill as a dated CORRECTION section
  4. *memory upkeep* — a standing daily scheduled job: merge
     duplicates in BRAIN.md, drop stale facts, keep BRAIN.md small
     (≈2,000 chars, upstream's discipline); the agent curates the
     file that curates its context
  5. *gap prosthetics* — the self-built apps of section 6

**CUT (redundant or useless at this scale):** the gateway and all
messaging platforms; voice, vision and browser tooling; the six
terminal backends (there is one guest); subagent swarms (one
level: a job IS a session); SQLite/FTS session search (grep wins
at megabytes); Honcho user modeling (BRAIN.md is the user model);
the RL/Atropos training plumbing (tunes here are separate and
human-ordered); skills hub/sync (no network in the guest); the
plugin/MCP machinery; the TUI (liveterm + the dashboard are the
faces this agent gets).

## Honest limits

- **Guest time is virtual.** The machine's clock ticks with retired
  instructions, seeded at boot. Relative schedules ("every 30m",
  "in 2h") are exact; wall-clock schedules ("09:00") need the host
  to seed real time when the terminal starts (phase 2 — liveterm
  types `date -s` once at boot).
- **Persistence is the machine's persistence.** Jobs and memory
  live in the guest filesystem: they survive while the machine
  stays up and across snapshot restores. A cold reboot without a
  snapshot loses them — snapshot discipline is the answer today;
  a host-side mirror of `/root/jobs` is a later nicety.
- **The circuit form comes later.** The same shape shrinks onto
  the bare-metal harness (schedule/remember as mailbox verbs next
  to store/recall/mul) once the Linux form proves the design. At
  circuit speeds a ticker tick is geological; there, scheduling
  is a design proof, not a utility.

## Build order

- **P1**: `agentshell.sh` foreground supervisor + jobs file +
  `schedule`/`jobs`/`unschedule`/`remember`/`recall` verbs in
  `agent.sh`, SOUL.md + BRAIN.md folded into wake-up prompts,
  `/root/history` logging, the destructive-command denylist, plus
  a `schedtest.py` driver (create a job due in 1 minute through
  the supervisor, type nothing, watch it fire it, verify the file
  + the log + the cleanup from ground truth).
- **P2**: the self-learning loop — skill birth/use/repair verbs,
  the gap-prosthetic app verbs, the HEARTBEAT.md checklist in
  the supervisor, the memory-upkeep standing job, host time
  seeding, job-failure self-disable.
- **P3**: dashboard Jobs + Built-apps views (read the guest's jobs,
  log, and apps read-only), chat wired to the same verb set, then
  the circuit port of the verbs.

## Status (2026-10-31)

P1 is built and proven end to end: `schedtest.py` **PASS** (a job
scheduled through the supervisor fired hands-off, did its work,
logged the outcome with continuity, and removed itself) and
`agenttest.py` **PASS**. P2's verbs are all built (skills, apps,
heartbeat checks, reflection, loop guards). `verbtest.py` runs
six ground-truth sessions per verb; across four runs it scores
**4–5 of 6**, and the failures move around between runs — every
verb has passed cleanly in some run, and every failure traced to
the seat model's session discipline, not the machinery: it
re-plans its own job text mid-session (so a later unschedule
points at a different list than it thinks), claims a verb it
never sent, or parrots its whole plan line instead of the next
step. The harness answers each with a mechanical guard (one call
per reply, a consecutive-call block, dedupe with a receipt,
claiming-is-not-doing in the prompt) — what remains is a
brain-quality problem, and it is precisely what a small tune on
this harness's own traces is for. P3's chat wiring is built (the
dashboard's chat delegates into the live guest); the dashboard
Jobs/apps views and the circuit port remain.

## The one-sentence version

Hermes' cron-and-memory shape, OpenClaw's heartbeat discipline,
a minimal loop's body — rebuilt in busybox shell small enough for
a 4B brain on a computer made of weights to actually run.
