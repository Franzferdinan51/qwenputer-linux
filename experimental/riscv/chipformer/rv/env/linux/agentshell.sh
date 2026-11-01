#!/bin/sh
# agentshell - the console's foreground supervisor for Qwenputer-Linux.
#
# WHY THIS EXISTS (measured, 2026-10-31): a background process cannot
# read the console in this guest - its reads return empty lines
# instantly (probe: BG_GOT:[] with a silent console). So a background
# cron daemon (the old agentd) can never hold a QWEN: session. The fix
# is the appliance pattern: ONE foreground process owns the console.
# It is part shell, part scheduler:
#   - type a shell command -> it runs, foreground
#   - type: hermes-agent "a task" -> an agent session, foreground
#   - type: exit -> back to the plain shell (/tmp/agentshell to return)
#   - idle 60 guest seconds (~12 wall seconds: the emulated clock
#     runs ~5x fast) -> the tick: heartbeat checks + due jobs run,
#     foreground, and the read below simply waits again - no new
#     prompt, so an idle console stays to one agent> line.
# Everything is serialized through this one process, so exactly one
# reader ever waits on the console and the QWEN: protocol just works.
#
# State (all in /root, survives as long as the machine is up):
#   /root/jobs          due_epoch|every_min|fails|last_summary|task
#   /root/jobs.log      outcome lines + heartbeat outcomes (self-trim)
#   /root/HEARTBEAT.md  "CHECK: <shell>" lines; output = wake the brain
#   /root/agentd.off    kill switch: while it exists, no jobs fire

JOBS=/root/jobs
JLOG=/root/jobs.log
HB=/root/HEARTBEAT.md
AGENT=/tmp/hermes-agent
[ -f "$JOBS" ] || : > "$JOBS"
[ -f "$JLOG" ] || : > "$JLOG"

now_s() { date +%s; }

# run_session <task-for-the-agent> -> echoes the session output
run_session() {
  AGENTD=1 "$AGENT" "$1" 2>&1
}

# summary_of <output> -> the session's final answer, one short line
summary_of() {
  s=$(printf '%s' "$1" | grep 'AGENT_ANSWER:' | tail -1 | sed 's/.*AGENT_ANSWER: *//')
  [ -z "$s" ] && s="(no answer)"
  printf '%s' "$s" | head -c 120 | tr '|' '/'
}

log_line() {
  echo "$(date) | $1" >> "$JLOG"
  n=$(wc -l < "$JLOG")
  if [ "$n" -gt 200 ]; then
    tail -50 "$JLOG" > "$JLOG.tmp" && mv "$JLOG.tmp" "$JLOG"
  fi
}

# fire_job <lineno> <due> <every> <fails> <last> <task>
fire_job() {
  lineno=$1; every=$3; fails=$4; last=$5; task=$6
  if [ -e /tmp/agent.lockdir ]; then
    echo "[agentshell] job deferred (agent busy): $task"
    return 0
  fi
  echo "[agentshell] running scheduled job: $task"
  out=$(run_session "SCHEDULED JOB: $task LAST TIME YOU FOUND: $last")
  rc=$?
  printf '%s\n' "$out"
  if [ $rc -eq 3 ]; then
    echo "[agentshell] job deferred (agent busy)"
    return 0
  fi
  sum=$(summary_of "$out")
  if [ $rc -eq 0 ]; then nf=0; else nf=$((fails + 1)); fi
  tmp=$JOBS.tmp; : > "$tmp"; i=0
  while IFS='|' read -r f1 f2 f3 f4 f5; do
    i=$((i + 1))
    if [ "$i" -ne "$lineno" ]; then
      echo "$f1|$f2|$f3|$f4|$f5" >> "$tmp"; continue
    fi
    if [ "$nf" -ge 3 ]; then
      log_line "job DISABLED after 3 failed runs: $task"
      continue
    fi
    if [ "$every" -gt 0 ] 2>/dev/null; then
      nd=$(( $(now_s) + every * 60 ))
      echo "$nd|$every|$nf|$sum|$task" >> "$tmp"
    fi
    # one-shot jobs (every=0) are done: the line is dropped
  done < "$JOBS"
  mv "$tmp" "$JOBS"
  log_line "job: $task => $sum"
}

# scan for due jobs and fire them, oldest first, until none are due
fire_due_jobs() {
  [ -e /root/agentd.off ] && return 0
  [ -x "$AGENT" ] || return 0
  rounds=0
  while [ $rounds -lt 8 ]; do
    rounds=$((rounds + 1))
    now=$(now_s); pick=""
    i=0
    while IFS='|' read -r f1 f2 f3 f4 f5; do
      i=$((i + 1))
      case $f1 in ''|*[!0-9]*) continue ;; esac
      if [ "$f1" -le "$now" ]; then
        pick="$i|$f1|$f2|$f3|$f4|$f5"
        break
      fi
    done < "$JOBS"
    [ -z "$pick" ] && break
    # split the picked line back into fields
    ln=$(printf '%s' "$pick" | cut -d'|' -f1)
    pd=$(printf '%s' "$pick" | cut -d'|' -f2)
    pe=$(printf '%s' "$pick" | cut -d'|' -f3)
    pf=$(printf '%s' "$pick" | cut -d'|' -f4)
    pl=$(printf '%s' "$pick" | cut -d'|' -f5)
    pt=$(printf '%s' "$pick" | cut -d'|' -f6-)
    fire_job "$ln" "$pd" "$pe" "$pf" "$pl" "$pt"
  done
}

# heartbeat: CHECK: lines are plain shell. Any output at all means
# "something changed" and the brain gets one session to look at it.
heartbeat() {
  [ -e /root/agentd.off ] && return 0
  [ -f "$HB" ] || return 0
  [ -x "$AGENT" ] || return 0
  flags=""
  while IFS= read -r line; do
    case $line in
      CHECK:*)
        cmd=$(printf '%s' "$line" | sed 's/^CHECK: *//')
        o=$(eval "$cmd" 2>/dev/null | tr '\n' ' ')
        [ -n "$o" ] && flags="$flags [$cmd -> $o]"
        ;;
    esac
  done < "$HB"
  [ -z "$flags" ] && return 0
  if [ -e /tmp/agent.lockdir ]; then
    echo "[agentshell] heartbeat flagged but agent is busy; will retry"
    return 0
  fi
  echo "[agentshell] heartbeat flagged:$flags"
  out=$(run_session "HEARTBEAT: something changed:$flags")
  rc=$?
  printf '%s\n' "$out"
  [ $rc -eq 3 ] && return 0
  log_line "heartbeat => $(summary_of "$out")"
}

echo "agentshell: this console belongs to the agent now."
echo "  type a shell command, ask \"a question\", or: hermes-agent \"your task in quotes\""
echo "  scheduled jobs and heartbeat checks fire on their own."
echo "  type: exit (plain shell)  sh (interactive shell, exit returns)"
echo $$ > /tmp/agentshell.pid
trap 'rm -f /tmp/agentshell.pid' EXIT

prompt_now=1
while true; do
  heartbeat
  fire_due_jobs
  # the prompt prints once per command, never once per tick: the read
  # below times out every 60 GUEST seconds (~12 wall seconds - the
  # emulated clock runs ~5x fast), and reprinting each time filled
  # the console with hundreds of agent> prompts on one line.
  if [ $prompt_now -eq 1 ]; then printf 'agent> '; prompt_now=0; fi
  line=""
  # IFS= and -r: the line is a command, read it verbatim (the bare
  # read ate backslashes and edge whitespace out of typed commands)
  t0=$(date +%s)
  if IFS= read -r -t 60 line; then
    prompt_now=1
    # dispatch on the line with leading blanks stripped, so a typed
    # "  exit" still leaves; the line itself runs verbatim
    tline=$(printf '%s' "$line" | sed 's/^[[:space:]]*//')
    case $tline in
      '') continue ;;
      exit|quit)
        echo "agentshell: leaving - run '/tmp/agentshell' to come back"
        break ;;
      sh) /bin/sh ;;
      hermes-agent|hermes-agent\ *) eval "/tmp/hermes-agent ${tline#hermes-agent}" ;;
      ask|ask\ *) eval "/tmp/ask ${tline#ask}" ;;
      nb|nb\ *) eval "/tmp/nb ${tline#nb}" ;;
      calc|calc\ *) eval "/tmp/calc ${tline#calc}" ;;
      *) eval "$line" ;;
    esac
  else
    # a read that fails with no input in under a guest second is a
    # wedged console, not a timeout - wait before retrying instead
    # of spinning the tick loop.
    t1=$(date +%s)
    if [ $((t1 - t0)) -lt 1 ]; then sleep 2; fi
  fi
  # read timed out (60 guest seconds) or a command finished: loop
  # back to the tick. Timeouts re-read silently - the prompt above
  # is already showing.
done
