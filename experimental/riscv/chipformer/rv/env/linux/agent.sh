#!/bin/sh
# hermes-agent - the guest agent harness: Hermes Agent's style and
# system, debloated to busybox shell (see docs/agent-framework.md).
# The model is the brain, this shell is the hands. Installed in the
# guest as /tmp/hermes-agent and /tmp/agent. The loop talks to the
# model over the QWEN: console protocol (see ask) - one line each way,
# because the device flattens every answer to a single line.
#
# What the brain can say (the harness intercepts everything except
# run - none of these cost a shell eval):
#   TOOL run <command>            run a busybox command, get RESULT
#   TOOL remember <fact>          keep a note in /root/MEMORY.md
#   TOOL remember-user <fact>     keep a user fact in /root/USER.md
#   TOOL recall <words>           search MEMORY.md, USER.md, notebook, history
#   TOOL schedule <min> <task>    run a task once, in <min> minutes
#   TOOL schedule every <min> <task>
#   TOOL jobs                     list scheduled jobs
#   TOOL unschedule <n>           cancel job n
#   TOOL saveskill <name> :: <procedure>
#   TOOL addskill <name> :: <one more line of the procedure>
#   TOOL fixskill <name> :: <correction, appended dated>
#   TOOL skill <name>             load a skill's procedure as RESULT
#   TOOL skills                   list skills
#   TOOL saveapp <name> :: <one-line shell program>   (gap prosthetic)
#   TOOL app <name> <args>        run one of its own apps
#   TOOL apps                     list its apps
#   TOOL forget <words>           drop memory lines matching <words>
#   TOOL delegate <subtask>       run one fenced sub-agent, get its report
#   TOOL plan :: <s1; s2; s3>     write a step plan (pending steps persist)
#   TOOL plan done <n>            mark plan step n done (show/clear too)
#   TOOL fetch <http(s) URL>      fetch a page through the host bridge
#   THINK: <thought>              private scratchpad - thinking, not acting
#   SAY: <answer>                 finish
#
# Self-learning (Hermes' closed loop, shrunk): a task that took real
# work (>=4 tool calls, or a failure then a recovery) earns a critique
# turn and a revise turn before the answer is delivered - what is
# unverified gets fixed - plus the usual what-is-worth-keeping (a
# skill or a remembered fact).
#
# Usage: hermes-agent "your task in quotes"
# Example: hermes-agent "write hello into /tmp/hi.txt"
#
# Console output is compact on purpose: one start line, one line per
# model reply tagged [agent round/max], one line per TOOL run result,
# then AGENT_ANSWER. Long replies and outputs are capped at 200 chars
# for display only - the model still gets full results in its prompts.

# NOTE: SYS is built in two assignments because the guest tty drops
# input past ~1021 chars on a single line (measured 2026-10-31: a
# 1300-char line arrived as exactly 1021). No line fed to the guest -
# in this file or any install - may approach that limit.
SYS='Answer with exactly one line, no explanation. Verbs: TOOL run <busybox command> - run a command, you get its output as RESULT. TOOL remember <fact> - keep a note in MEMORY.md. TOOL remember-user <fact> - keep a user fact in USER.md. TOOL recall <words> - search memory, notebook and history. TOOL schedule <minutes> <task> - do a task later; TOOL schedule every <minutes> <task> repeats it. TOOL jobs / TOOL unschedule <n>. TOOL saveskill <name> :: <procedure> - save how to do something, so future you knows; TOOL skill <name> loads one back; check TOOL skills before inventing an approach. TOOL saveapp <name> :: <one-line shell program> - build yourself a small app when you hit a gap (bad at something? make a tool for it once, use it forever); TOOL app <name> <args> runs it; TOOL apps lists them.'
SYS="$SYS"' SAY: <answer> - finish, and SAY only reports what a RESULT showed you. Claiming is not doing: remembering, saving and building happen ONLY through their TOOL verbs - if you did not send the verb, it did not happen. You have a whole linux system: files persist, agentshell runs your scheduled jobs, your notes are /root/MEMORY.md, the user profile is /root/USER.md, notebook /root/nb. Examples - task "multiply 123456789 by 987654321 exactly": TOOL run echo $((123456789 * 987654321)). Task "check the disk every 30 minutes": TOOL schedule every 30 check disk space with df and remember the result. Task "remember my favorite fruit is kiwi": TOOL remember-user the favorite fruit is kiwi. Task "make /tmp/hi.txt containing hello, then read it back": TOOL run echo hello > /tmp/hi.txt and after the result TOOL run cat /tmp/hi.txt then SAY: what the file says.'
SYS="$SYS"' Helpers: TOOL run nb add <note> keeps a dated notebook note, TOOL run nb search <words> finds notes back; TOOL run calc <integer expression> does exact arithmetic.'
SYS="$SYS"' Memory: MEMORY.md notes tagged [fact] [lesson] or [project:name] (untagged saves as [fact]); USER.md takes user facts; TOOL recall in:lesson <words> or in:pref searches one place; TOOL forget <words> drops matching lines from both.'
SYS="$SYS"' THINK: <thought> thinks out loud, briefly (max 6 a session, not an action). Keep all thinking short - a few lines - then act; long thinking wastes your reply budget. TOOL delegate <subtask> runs one fenced sub-agent and returns its report (one level only). TOOL plan :: s1; s2; s3 writes steps, TOOL plan done <n> checks one off, pending steps ride along until done. TOOL fetch <url> fetches an http(s) page through the host.'

qwen_ask() {
  # the prompt must go to the CONSOLE, not stdout: this function is called
  # inside $(...), where stdout is the capture pipe
  printf 'QWEN:%s\n' "$1" > /dev/console
  # the host marks its typed answers QWENANS: - anything else on the
  # console line is a human typing mid-session; never mistake it for
  # the model's reply (that ended a live session on 2026-10-31).
  _tries=0
  while [ $_tries -lt 6 ]; do
    IFS= read -r _ans || { printf ''; return; }
    case "$_ans" in
      QWENANS:*) printf '%s' "${_ans#QWENANS:}"; return ;;
      *) _tries=$((_tries + 1))
         printf '[agent] ignored a console line that was not a model reply\n' > /dev/console ;;
    esac
  done
  printf ''
}

deny() {
  # the safety denylist: whoever asked, these never run. The rm rule
  # fires only when a recursive rm AND a standalone / or /* token are
  # both present (blanks and ; folded, string padded), so rm -rf
  # /tmp/x keeps working while any recursive rm aimed at / or /*
  # never does. (The old substring match blocked every absolute rm -rf
  # and missed rm -fr entirely.)
  s=" $(printf '%s' "$1" | tr -s ' \t;' '   ') "
  case "$s" in
    *" rm -"[fr]*" "*|*" rm "*"--recursive"*)
      case "$s" in
        *" / "*|*" /* "*) return 0 ;;
      esac ;;
  esac
  case "$1" in
    *mkfs*|*"dd if="*|*"of=/dev/"*|\
    *shutdown*|*poweroff*|*"reboot"*|*":(){"*|*"chmod -R 777 /"*)
      return 0 ;;
  esac
  return 1
}

plan_pending() {
  # pending (unchecked) plan steps, one capped line - these ride along
  # in every prompt so a plan survives round caps and sessions; a new
  # session sees the old plan's leftovers and can resume them
  [ -s "$PLANF" ] || return 0
  grep -e '^- \[ \]' "$PLANF" 2>/dev/null | sed 's/^- \[ \] //' | tr '\n' ';' | head -c 200
}

result_prompt() {
  prev_out="$1"
  prompt="$SYS TASK: $task RESULT: $1 Now reply with EXACTLY ONE line: a TOOL verb or SAY: <final answer>"
  pp=$(plan_pending)
  [ -n "$pp" ] && prompt="$prompt Pending plan steps: $pp"
}

log_failure() {
  # failures are memory too: one dated line each, self-trimmed, folded
  # back into every wake-up as PITFALLS so sessions stop repeating them
  echo "$(date) | $task => $1" >> /root/FAILURES.md
  n=$(wc -l < /root/FAILURES.md)
  if [ "$n" -gt 50 ]; then tail -30 /root/FAILURES.md > /root/FAILURES.md.tmp && mv /root/FAILURES.md.tmp /root/FAILURES.md; fi
}

clean_name() {
  # skill/app names come from the model - keep them to one safe path
  # segment so a ../../.. name can never escape /root/skills or
  # /root/apps into the rest of the filesystem
  printf '%s' "$1" | tr -cd 'A-Za-z0-9_-' | head -c 48
}

short() {
  # short <text> -> the text capped for CONSOLE display (200 chars plus
  # a marker). The model still gets full results inside its prompts -
  # this only keeps the terminal readable when replies and tool output
  # run long, especially mid-loop.
  s=$1
  if [ "${#s}" -gt 200 ]; then
    printf '%s...' "$(printf '%s' "$s" | head -c 200)"
  else
    printf '%s' "$s"
  fi
}

task="$*"
[ -z "$task" ] && { echo 'usage: hermes-agent "your task in quotes"'; echo 'example: hermes-agent "write hello into /tmp/hi.txt"'; exit 1; }

# sub-agent depth (TOOL delegate): a child shares its parent's console
# lock instead of taking its own (nested console sessions already work
# - ask.sh runs the agent as a child the same way); it gets a shorter
# round budget and a private plan file, and it may not delegate further.
DEPTH=${AGENT_DEPTH:-0}
if [ "$DEPTH" -ge 1 ]; then PLANF=/tmp/PLAN.$$; else PLANF=/root/PLAN.md; fi

# one agent at a time: the console protocol is a single queue, two
# loops would interleave prompts and answers into mush. agentd sets
# AGENTD=1 and defers when busy; a human's run waits a little. The
# holder leaves its pid behind so a lock orphaned by kill -9 (no trap)
# can be taken over instead of wedging every later session.
takeover_stale() {
  [ -f /tmp/agent.lockdir/pid ] || return 1
  p=$(cat /tmp/agent.lockdir/pid 2>/dev/null)
  case "$p" in ''|*[!0-9]*) return 1 ;; esac
  kill -0 "$p" 2>/dev/null && return 1
  rm -rf /tmp/agent.lockdir
}
try_lock() {
  mkdir /tmp/agent.lockdir 2>/dev/null && return 0
  takeover_stale 2>/dev/null && mkdir /tmp/agent.lockdir 2>/dev/null
}
if [ "$DEPTH" -eq 0 ]; then
if ! try_lock; then
  if [ -n "$AGENTD" ]; then
    echo "AGENT_BUSY"
    exit 3
  fi
  w=0
  locked=0
  while [ $w -lt 30 ]; do
    sleep 2; w=$((w + 2))
    if try_lock; then locked=1; break; fi
  done
  if [ $locked -eq 0 ]; then
    echo "AGENT_ANSWER: (the agent is busy with another task - try again shortly)"
    exit 1
  fi
fi
echo $$ > /tmp/agent.lockdir/pid
fi
: > /tmp/agent.runs.$$
: > /tmp/agent.think.$$
if [ "$DEPTH" -eq 0 ]; then
trap 'rm -rf /tmp/agent.lockdir; rm -f /tmp/agent.runs.$$ /tmp/agent.think.$$' EXIT INT TERM
else
trap 'rm -f /tmp/agent.runs.$$ /tmp/agent.think.$$ /tmp/PLAN.$$' EXIT INT TERM
fi

mkdir -p /root/nb /root/skills /root/apps
# Hermes-name migration (once): BRAIN.md splits into MEMORY.md (agent
# notes) + USER.md (user prefs); the original stays as a backup
if [ -f /root/BRAIN.md ] && [ ! -f /root/MEMORY.md ]; then
  grep '^\[pref\]' /root/BRAIN.md 2>/dev/null > /root/USER.md
  grep -v '^\[pref\]' /root/BRAIN.md 2>/dev/null > /root/MEMORY.md
  mv /root/BRAIN.md /root/BRAIN.md.migrated
fi
# wake-up context: identity + curated memory fold into the first
# prompt only, size-capped (Hermes keeps memory small on purpose -
# curation over bulk)
soul=""; brain=""; userprof=""; pitfalls=""
[ -f /root/SOUL.md ] && soul=$(head -c 300 /root/SOUL.md | tr '\n' ' ')
if [ -f /root/MEMORY.md ]; then
  # structured memory: the newest lines per section, so one chatty
  # section can't crowd the others out of the wake-up prompt
  brain=$({ for s in fact lesson; do grep "^\[$s\]" /root/MEMORY.md 2>/dev/null | tail -2; done; grep '^\[project:' /root/MEMORY.md 2>/dev/null | tail -2; grep -v '^\[' /root/MEMORY.md 2>/dev/null | tail -2; } | tr '\n' ' ' | head -c 700)
  brain="$(wc -c < /root/MEMORY.md 2>/dev/null || echo 0)/2200 chars: $brain"
fi
if [ -f /root/USER.md ]; then
  userprof=$(head -c 500 /root/USER.md | tr '\n' ' ')
  userprof="$(wc -c < /root/USER.md 2>/dev/null || echo 0)/1375 chars: $userprof"
fi
[ -f /root/FAILURES.md ] && pitfalls=$(tail -5 /root/FAILURES.md | tr '\n' ' ' | head -c 300)
prompt="$SYS"
[ -n "$soul" ] && prompt="$prompt YOU: $soul"
[ -n "$brain" ] && prompt="$prompt MEMORY: $brain"
[ -n "$userprof" ] && prompt="$prompt USER: $userprof"
[ -n "$pitfalls" ] && prompt="$prompt PITFALLS (do not repeat these): $pitfalls"
pp0=$(plan_pending); [ -n "$pp0" ] && prompt="$prompt Pending plan steps: $pp0"
prompt="$prompt TASK: $task"

prev_cmd=""
prev_out=""
prev_ans=""
prev_ans2=""
circles=0
empties=0
toolcalls=0
failed=0
recovered=0
did_reflect=0
pending_final=""
pending_stage=0
thinks=0
maxthinks=6
round=0
maxround=15
[ "$DEPTH" -ge 1 ] && maxround=6
if [ "$DEPTH" -ge 1 ]; then echo "[agent] start (sub-agent): $(short "$task") ($maxround rounds max)"; else echo "[agent] start: $(short "$task") ($maxround rounds max)"; fi
while [ $round -lt $maxround ]; do
  round=$((round + 1))
  ans=$(qwen_ask "$prompt")
  raw="$ans"
  # ONE call per reply: the brain likes to simulate the whole
  # conversation in a single line (a glued "; SAY:" tail, a second
  # TOOL call, an invented RESULT). Cut everything from the first
  # simulated continuation on - only the first verb is real. (There is
  # deliberately no generic "then" cut: it mangled valid shell such as
  # TOOL run if [ -f f ]; then echo y; fi. "then TOOL/SAY" is covered.)
  ans=$(printf '%s' "$ans" | sed -e 's/; *SAY:.*//' -e 's/; *RESULT:.*//' -e 's/; *TOOL .*//' -e 's/\. *TOOL .*//' -e 's/\. *SAY:.*//' -e 's/ then SAY:.*//' -e 's/ then TOOL .*//' -e 's/ then RESULT:.*//' -e 's/ and after .*//' -e 's/ and cat .*//' -e 's/ TOOL .*//' -e 's/ RESULT:.*//' -e 's/ SAY:.*//')
  # a bare empty reply (the fast core delivers it unadorned) must take
  # the same nudge-and-escalate path as a device notice - left alone
  # it would fall into the catch-all and be delivered as an (empty)
  # final answer, which is how the first v3 regression test "passed"
  # an empty answer in 1 round
  [ -z "$(printf '%s' "$ans" | tr -d ' \t')" ] && ans="[qwen: empty answer]"
  echo "[agent $round/$maxround] $(short "$ans")"

  # THINK: private scratchpad. The brain may think out loud instead of
  # acting - the thought is logged and folded into the next prompt as
  # SCRATCH. Thinks don't consume tool rounds (they still cost model
  # calls, so they're capped separately); the raw reply feeds the log
  # because truncation would eat a thought that mentions TOOL or SAY.
  case "$ans" in
    "THINK:"*|"THINK "*)
      if [ $thinks -ge $maxthinks ]; then
        echo "[agent] think budget spent - must act now"
        prompt="$SYS TASK: $task RESULT: (no more THINK: turns - you have thought enough. Send a TOOL verb now, or SAY: <final answer>)"
        continue
      fi
      thinks=$((thinks + 1))
      thought=$(printf '%s' "$raw" | sed 's/^THINK: *//;s/^THINK *//')
      echo "$thought" >> /tmp/agent.think.$$
      scratch=$(tail -3 /tmp/agent.think.$$ | tr '\n' ' ' | head -c 400)
      echo "[agent] think $thinks/$maxthinks: $(short "$thought")"
      round=$((round - 1))
      prompt="$SYS TASK: $task SCRATCH (your own notes so far): $scratch RESULT: (noted - continue: next TOOL verb, or SAY: <final answer>)"
      continue ;;
  esac

  # task-echo guard: the plain base brain sometimes answers by
  # repeating the task back under its own "TASK:" label instead of
  # doing it (seen live 2026-10-31: reply was the task text itself,
  # and the catch-all delivered it as the final answer). An echo is
  # not an answer - nudge and redraw; the round cap still bounds it.
  case "$ans" in
    "TASK: "*)
      echo "[agent] task echo blocked - nudging"
      prompt="$SYS TASK: $task RESULT: (you repeated the task back - that is not doing it. Send the FIRST tool call for this task now, exactly one line: a TOOL verb, or SAY: <answer> if no tool is needed)"
      continue ;;
  esac

  # critique + revise: real work earns two turns before the answer
  # goes out. Stage one asks what is unverified (a saveskill or
  # remember counts as the keeping); stage two takes the corrected
  # answer. A SAY: at stage one means "nothing to fix" - deliver it.
  if [ -n "$pending_final" ]; then
    if [ "$pending_stage" -eq 1 ]; then
      kept=0
      case "$ans" in
        *"TOOL saveskill "*)
          spec=$(printf '%s' "$ans" | sed 's/.*TOOL saveskill //')
          name=$(clean_name "$(printf '%s' "$spec" | sed 's/ .*//;s/::.*//')")
          [ -z "$name" ] && name=unnamed
          body=$(printf '%s' "$spec" | sed 's/^[^:]*:: *//')
          { echo "# skill: $name"; echo "saved: $(date)"; echo "$body"; } > "/root/skills/$name.md"
          echo "[agent] skill saved: $name"; kept=1 ;;
        *"TOOL remember "*)
          fact=$(printf '%s' "$ans" | sed 's/.*TOOL remember //')
          grep -qxF "$fact" /root/MEMORY.md 2>/dev/null || echo "$fact" >> /root/MEMORY.md
          echo "[agent] remembered: $(short "$fact")"; kept=1 ;;
        *"TOOL remember-user "*)
          fact=$(printf '%s' "$ans" | sed 's/.*TOOL remember-user //')
          grep -qxF "$fact" /root/USER.md 2>/dev/null || echo "$fact" >> /root/USER.md
          echo "[agent] remembered (user): $(short "$fact")"; kept=1 ;;
      esac
      case "$ans" in
        *"SAY: "*)
          final2=$(printf '%s' "$ans" | sed 's/.*SAY: //')
          echo "$(date) | $task => $final2" >> /root/history
          echo "AGENT_ANSWER: $final2"
          exit 0 ;;
      esac
      if [ $kept -eq 1 ]; then crit="(keeping noted)"; else crit="$ans"; echo "[agent] critique: $(short "$ans")"; fi
      pending_stage=2
      prompt="$SYS TASK: revise. The task '$task' draft answer was: $pending_final. Critique: $crit. Reply with the CORRECTED final answer as one SAY: line now."
      continue
    fi
    case "$ans" in
      *"SAY: "*) final2=$(printf '%s' "$ans" | sed 's/.*SAY: //') ;;
      *) final2="$ans" ;;
    esac
    echo "$(date) | $task => $final2" >> /root/history
    echo "AGENT_ANSWER: $final2"
    exit 0
  fi

  # general consecutive-call guard: the brain's simulate-ahead habit
  # makes it re-send its whole plan line every round; truncation
  # reduces that to the same first call again and again (seen live:
  # schedule -> dedupe result -> same schedule line, three rounds,
  # never advancing to the next step). The run branch has its own
  # prev_cmd check; this one covers every verb: an identical TOOL
  # line twice in a row is not executed again - the model is told
  # the step is done and pushed to the NEXT step of its task.
  case "$ans" in
    *"TOOL "*)
      if [ "$ans" = "$prev_ans" ]; then
        echo "[agent] repeat call blocked: $(short "$ans")"
        prompt="$SYS TASK: $task RESULT: (you already sent that exact call last round and it was done - do NOT send it again. Send the NEXT step of your task as one line, or SAY: <final answer> if the task is finished)"
        continue
      fi
      # two-step circle (A-B-A): the repeat check above only sees
      # consecutive twins, so the brain ping-pongs two paraphrases
      # to the round cap instead (seen live: TOOL recall memory /
      # TOOL recall memory notebook history alternating, nothing
      # found every time). The first lap earns a nudge; the second
      # lap ends the session honestly instead of burning the rest.
      if [ -n "$prev_ans2" ] && [ "$ans" = "$prev_ans2" ]; then
        circles=$((circles + 1))
        echo "[agent] circle detected ($circles): $(short "$ans")"
        if [ $circles -ge 2 ]; then
          echo "$(date) | $task => (circle) $prev_out" >> /root/history
          echo "AGENT_ANSWER: (going in circles - repeated the same searches with no new results; last result: $prev_out)"
          echo "[agent] stopped: circle, no progress"
          exit 1
        fi
        prompt="$SYS TASK: $task RESULT: (you are going in circles - that call was already tried and changed nothing. Do a DIFFERENT step now, or SAY: <final answer> with what you actually found)"
        prev_ans2="$prev_ans"
        prev_ans="$ans"
        continue
      fi
      circles=0
      prev_ans2="$prev_ans"
      prev_ans="$ans" ;;
  esac

  case "$ans" in
    *"TOOL remember "*)
      empties=0
      fact=$(printf '%s' "$ans" | sed 's/.*TOOL remember //')
      # structured memory: facts carry a [section] tag; untagged facts
      # save as [fact] so every line is greppable by section (old
      # untagged lines keep working - recall still searches them)
      case "$fact" in
        "["*) : ;;
        *) fact="[fact] $fact" ;;
      esac
      grep -qxF "$fact" /root/MEMORY.md 2>/dev/null || echo "$fact" >> /root/MEMORY.md
      result_prompt "remembered: $fact (it is in your long-term memory now)"
      ;;
    *"TOOL remember-user "*)
      empties=0
      fact=$(printf '%s' "$ans" | sed 's/.*TOOL remember-user //')
      grep -qxF "$fact" /root/USER.md 2>/dev/null || echo "$fact" >> /root/USER.md
      result_prompt "remembered: $fact (it is in the user profile now)"
      ;;
    *"TOOL recall "*)
      empties=0
      words=$(printf '%s' "$ans" | sed 's/.*TOOL recall //')
      scope=""
      case "$words" in
        "in:"*)
          scope=$(clean_name "$(printf '%s' "$words" | sed 's/^in://;s/ .*//')")
          words=$(printf '%s' "$words" | sed 's/^in:[^ ]* *//') ;;
      esac
      pat=$(printf '%s' "$words" | sed 's/ /|/g')
      if [ -n "$scope" ]; then
        if [ "$scope" = "user" ]; then
          hits=$(grep -iE -e "$pat" /root/USER.md 2>/dev/null | head -c 300 | tr '\n' ' ')
        else
          memf=/root/MEMORY.md
          [ "$scope" = "pref" ] && memf=/root/USER.md
          hits=$(grep -iE -e "^\[$scope\].*($pat)" $memf 2>/dev/null | head -c 300 | tr '\n' ' ')
        fi
        [ -z "$hits" ] && hits="(nothing in [$scope] for: $words)"
      else
        hits=$(grep -iE -e "$pat" /root/MEMORY.md /root/USER.md /root/FAILURES.md /root/history /root/nb/* /root/skills/*.md 2>/dev/null | head -c 300 | tr '\n' ' ')
        [ -z "$hits" ] && hits="(nothing found for: $words)"
      fi
      result_prompt "$hits"
      ;;
    *"TOOL forget "*)
      empties=0
      words=$(printf '%s' "$ans" | sed 's/.*TOOL forget //')
      pat=$(printf '%s' "$words" | sed 's/ /|/g')
      before=0
      for memf in /root/MEMORY.md /root/USER.md; do
        [ -f $memf ] && before=$((before + $(wc -l < $memf)))
        # exit 0 (lines left) or 1 (none left) both mean "pattern was
        # valid, rewrite"; anything else (bad pattern) keeps the file
        grep -viE -e "$pat" $memf 2>/dev/null > $memf.tmp; st=$?
        if [ $st -eq 0 ] || [ $st -eq 1 ]; then mv $memf.tmp $memf; else rm -f $memf.tmp; fi
      done
      after=0
      for memf in /root/MEMORY.md /root/USER.md; do [ -f $memf ] && after=$((after + $(wc -l < $memf))); done
      result_prompt "forgot $((before - after)) line(s) matching: $words ($after left)"
      ;;
    *"TOOL schedule "*)
      empties=0
      spec=$(printf '%s' "$ans" | sed 's/.*TOOL schedule //')
      every=0
      case "$spec" in
        "every "*) spec=$(printf '%s' "$spec" | sed 's/^every //'); every=1 ;;
      esac
      mins=$(printf '%s' "$spec" | sed 's/ .*//')
      # normalize the job text: the brain decorates it with quotes
      # that are not part of the task, and un-normalized variants
      # defeat the duplicate check below
      jtask=$(printf '%s' "$spec" | sed 's/^[^ ]* //' | tr '|' '/' | tr -d '"' | sed 's/^ *//;s/ *$//')
      case "$mins" in ''|*[!0-9]*) mins=1 ;; esac
      njobs=0; [ -f /root/jobs ] && njobs=$(wc -l < /root/jobs)
      if cut -d'|' -f5- /root/jobs 2>/dev/null | grep -qxF "$jtask"; then
        # this brain re-schedules the same job on every RESULT if the
        # harness lets it (8 copies in one session, live 2026-10-31) -
        # a duplicate is a no-op with a clear receipt, not a new job
        result_prompt "(that exact job is already on the list - it is scheduled ONCE and will run. do NOT schedule it again. if your task was only to schedule it, reply SAY: done)"
      elif [ "$njobs" -ge 16 ]; then
        result_prompt "(job list is full - 16 jobs max. unschedule one first)"
      else
        due=$(( $(date +%s) + mins * 60 ))
        if [ $every -eq 1 ]; then em=$mins; else em=0; fi
        echo "$due|$em|0|never run|$jtask" >> /root/jobs
        result_prompt "scheduled: '$jtask' in $mins minutes (job $((njobs + 1))). agentd will run it and log the outcome to /root/jobs.log"
      fi
      ;;
    *"TOOL jobs"*)
      empties=0
      if [ -s /root/jobs ]; then
        # pure shell: this rootfs has no awk applet
        out=""; n=0
        while IFS='|' read -r f1 f2 f3 f4 f5; do
          n=$((n + 1))
          out="$out job $n: due $f1, every $f2 min, last: $f4 - $f5;"
        done < /root/jobs
        out=$(printf '%s' "$out" | head -c 300)
      else
        out="(no jobs scheduled)"
      fi
      result_prompt "$out"
      ;;
    *"TOOL unschedule "*)
      empties=0
      n=$(printf '%s' "$ans" | sed 's/.*TOOL unschedule //;s/[^0-9].*//')
      nj=0; [ -f /root/jobs ] && nj=$(wc -l < /root/jobs)
      if [ -n "$n" ] && [ "$n" -ge 1 ] 2>/dev/null && [ "$n" -le "$nj" ]; then
        sed -i "${n}d" /root/jobs
        result_prompt "job $n removed"
      elif [ -z "$n" ]; then
        result_prompt "(no job number given - see TOOL jobs)"
      else
        result_prompt "(there is no job $n - see TOOL jobs)"
      fi
      ;;
    *"TOOL plan"*)
      empties=0
      spec=$(printf '%s' "$ans" | sed 's/.*TOOL plan *//')
      case "$spec" in
        "done "*)
          n=$(printf '%s' "$spec" | sed 's/^done //;s/[^0-9].*//')
          if [ -n "$n" ] && grep -q -e '^- \[ \]' "$PLANF" 2>/dev/null; then
            # check off the nth pending step (pure shell: no awk here)
            tmp=$PLANF.tmp; : > "$tmp"; i=0; done_n=0
            while IFS= read -r pl; do
              case "$pl" in
                "- [ ] "*)
                  i=$((i + 1))
                  if [ "$i" -eq "$n" ]; then echo "- [x]${pl#- \[ \]}" >> "$tmp"; done_n=1; else echo "$pl" >> "$tmp"; fi ;;
                *) echo "$pl" >> "$tmp" ;;
              esac
            done < "$PLANF"
            mv "$tmp" "$PLANF"
            [ $done_n -eq 1 ] && result_prompt "plan step $n checked off" || result_prompt "(no pending plan step $n)"
          else
            result_prompt "(no plan yet - write one with TOOL plan :: step1; step2)"
          fi ;;
        "show"*|"status"*)
          if [ -s "$PLANF" ]; then result_prompt "$(tr '\n' ';' < "$PLANF" | head -c 300)"; else result_prompt "(no plan yet)"; fi ;;
        "clear"*)
          rm -f "$PLANF"; result_prompt "plan cleared" ;;
        ":: "*|":"*)
          steps=$(printf '%s' "$spec" | sed 's/^:: *//;s/^: *//')
          { echo "# plan for: $task"; echo "written: $(date)"; printf '%s' "$steps" | tr ';' '\n' | sed 's/^ *//;s/ *$//' | grep -v '^$' | sed 's/^/- [ ] /'; } > "$PLANF"
          n=$(grep -c -e '^- \[ \]' "$PLANF")
          result_prompt "plan written ($n steps) - pending steps ride along in every prompt until checked off with TOOL plan done <n>" ;;
        *)
          result_prompt "(usage: TOOL plan :: step1; step2 - or TOOL plan done <n> / show / clear)" ;;
      esac
      ;;
    *"TOOL saveskill "*)
      empties=0
      spec=$(printf '%s' "$ans" | sed 's/.*TOOL saveskill //')
      name=$(clean_name "$(printf '%s' "$spec" | sed 's/ .*//;s/::.*//')")
      [ -z "$name" ] && name=unnamed
      body=$(printf '%s' "$spec" | sed 's/^[^:]*:: *//')
      { echo "# skill: $name"; echo "saved: $(date)"; echo "$body"; } > "/root/skills/$name.md"
      result_prompt "skill '$name' saved - future you can load it with TOOL skill $name"
      ;;
    *"TOOL addskill "*)
      empties=0
      spec=$(printf '%s' "$ans" | sed 's/.*TOOL addskill //')
      name=$(clean_name "$(printf '%s' "$spec" | sed 's/ .*//;s/::.*//')")
      [ -z "$name" ] && name=unnamed
      body=$(printf '%s' "$spec" | sed 's/^[^:]*:: *//')
      echo "$body" >> "/root/skills/$name.md"
      result_prompt "line added to skill '$name'"
      ;;
    *"TOOL fixskill "*)
      empties=0
      spec=$(printf '%s' "$ans" | sed 's/.*TOOL fixskill //')
      name=$(clean_name "$(printf '%s' "$spec" | sed 's/ .*//;s/::.*//')")
      [ -z "$name" ] && name=unnamed
      body=$(printf '%s' "$spec" | sed 's/^[^:]*:: *//')
      echo "CORRECTION $(date): $body" >> "/root/skills/$name.md"
      result_prompt "correction added to skill '$name'"
      ;;
    *"TOOL skills"*)
      empties=0
      out=""
      for f in /root/skills/*.md; do
        [ -f "$f" ] || continue
        out="$out $(basename "$f" .md): $(sed -n '3p' "$f" | head -c 60);"
      done
      [ -z "$out" ] && out="(no skills yet - save one with TOOL saveskill <name> :: <procedure>)"
      result_prompt "$out"
      ;;
    *"TOOL skill "*)
      empties=0
      name=$(clean_name "$(printf '%s' "$ans" | sed 's/.*TOOL skill //;s/ .*//')")
      if [ -f "/root/skills/$name.md" ]; then
        out=$(cat "/root/skills/$name.md" | tr '\n' ' ' | head -c 300)
      else
        out="(no skill named $name - see TOOL skills)"
      fi
      result_prompt "$out"
      ;;
    *"TOOL saveapp "*)
      empties=0
      spec=$(printf '%s' "$ans" | sed 's/.*TOOL saveapp //')
      name=$(clean_name "$(printf '%s' "$spec" | sed 's/ .*//;s/::.*//')")
      [ -z "$name" ] && name=unnamed
      body=$(printf '%s' "$spec" | sed 's/^[^:]*:: *//')
      { echo '#!/bin/sh'; echo "# built by the agent for itself: $(date)"; echo "$body"; } > "/root/apps/$name"
      chmod +x "/root/apps/$name"
      result_prompt "app '$name' built - run it with TOOL app $name"
      ;;
    *"TOOL apps"*)
      empties=0
      out=""
      for f in /root/apps/*; do
        [ -f "$f" ] || continue
        out="$out $(basename "$f"): $(sed -n '3p' "$f" | head -c 60);"
      done
      [ -z "$out" ] && out="(no apps yet - build one with TOOL saveapp <name> :: <one-line program>)"
      result_prompt "$out"
      ;;
    *"TOOL app "*)
      empties=0; toolcalls=$((toolcalls + 1))
      spec=$(printf '%s' "$ans" | sed 's/.*TOOL app //')
      name=$(clean_name "$(printf '%s' "$spec" | sed 's/ .*//')")
      args=$(printf '%s' "$spec" | sed 's/^[^ ]* *//')
      if [ -x "/root/apps/$name" ]; then
        # direct exec, not eval: the args are data, and eval would
        # re-parse them (a $(...) in args would run as a command)
        raw=$("/root/apps/$name" $args 2>&1); st=$?
        [ $st -ne 0 ] && failed=1
        out=$(printf '%s' "$raw" | tr '\n' ' ' | head -c 300)
        [ -z "$out" ] && out="(app ran, no output, exit $st)"
      else
        out="(no app named $name - see TOOL apps)"
      fi
      result_prompt "$out"
      ;;
    *"TOOL delegate "*)
      empties=0; toolcalls=$((toolcalls + 1))
      # one level only: a sub-agent's own delegate is refused, or
      # sessions would nest until the dashboard deadline kills them
      if [ "$DEPTH" -ge 1 ]; then
        result_prompt "(sub-agents cannot delegate - one level only. Do the step yourself with another TOOL)"
      else
        sub=$(printf '%s' "$ans" | sed 's/.*TOOL delegate //' | head -c 200)
        echo "[agent] delegating: $(short "$sub")"
        out=$(AGENT_DEPTH=1 AGENTD=1 "$0" "$sub" 2>&1)
        rep=$(printf '%s' "$out" | grep 'AGENT_ANSWER:' | tail -1 | sed 's/.*AGENT_ANSWER: *//')
        [ -z "$rep" ] && rep="(the sub-agent produced no answer)"
        echo "[agent] sub-agent reported: $(short "$rep")"
        result_prompt "sub-agent report: $rep"
      fi
      ;;
    *"TOOL fetch "*)
      empties=0; toolcalls=$((toolcalls + 1))
      url=$(printf '%s' "$ans" | sed 's/.*TOOL fetch //' | sed 's/ .*//')
      case "$url" in
        "http://"*|"https://"*)
          echo "[agent] fetching: $(short "$url")"
          page=$(qwen_ask "FETCHURL $url" | head -c 600)
          [ -z "$page" ] && page="(fetch returned nothing)"
          result_prompt "fetched ($url): $page" ;;
        *)
          result_prompt "(fetch needs an http(s) URL - got: $url)" ;;
      esac
      ;;
    *"TOOL run "*)
      empties=0; toolcalls=$((toolcalls + 1))
      cmd=$(printf '%s' "$ans" | sed 's/.*TOOL run //')
      if deny "$cmd"; then
        echo "[agent] refused (denylist): $(short "$cmd")"
        log_failure "denylist refused: $cmd"
        result_prompt "(refused by the harness: that command is on the safety denylist - it can destroy the system. do something else)"
        continue
      fi
      if [ "$cmd" = "$prev_cmd" ]; then
        echo "[agent] repeat of the last command - not running it again"
        prompt="$SYS TASK: $task RESULT: you already ran that command, its result was: $prev_out Do the NEXT step, or reply SAY: <final answer>"
        continue
      fi
      # loop guard: the consecutive-repeat check above misses the
      # write/read/write/read thrash (seen live: the model re-wrote a
      # file it had already read back, until the round cap). A third
      # run of the exact same command in one session is a loop, not
      # a check - refuse it and push the model to finish.
      runs=$(grep -cFx "$cmd" /tmp/agent.runs.$$ 2>/dev/null)
      if [ "${runs:-0}" -ge 2 ]; then
        echo "[agent] loop guard: same command already run $runs times"
        log_failure "loop-guard refused a 3rd run of: $cmd"
        prompt="$SYS TASK: $task RESULT: you have already run that exact command $runs times in this session and it keeps giving the same result. Do a DIFFERENT next step, or reply SAY: <final answer> if you have what you need"
        continue
      fi
      echo "$cmd" >> /tmp/agent.runs.$$
      raw=$(eval "$cmd" 2>&1); st=$?
      [ $st -ne 0 ] && failed=1
      [ $st -eq 0 ] && [ $failed -eq 1 ] && recovered=1
      out=$(printf '%s' "$raw" | tr '\n' ' ' | head -c 300)
      # a silent success (a file write prints nothing) reads as "nothing
      # happened" to the model and it just retries - say what happened
      [ -z "$out" ] && out="(command ran, no output, exit $st)"
      prev_cmd="$cmd"
      prev_out="$out"
      echo "[agent $round/$maxround] run: $(short "$cmd") => $(short "$out")"
      result_prompt "$out"
      ;;
    *"SAY: "*)
      final=$(printf '%s' "$ans" | sed 's/.*SAY: //')
      # the closed loop: real work earns critique + revise turns before
      # the answer goes out (>=4 tool calls, or a recovery) - what is
      # unverified gets fixed, and what is worth keeping gets kept
      if [ $did_reflect -eq 0 ] && { [ $toolcalls -ge 4 ] || [ $recovered -eq 1 ]; }; then
        did_reflect=1
        pending_final="$final"
        pending_stage=1
        prompt="$SYS TASK: critique. The task '$task' has a DRAFT answer: $final. What in it is unverified, guessed, or contradicted by the RESULTS above? If a reusable procedure is worth keeping reply TOOL saveskill <name> :: <steps and pitfalls>; if a fact is worth keeping reply TOOL remember <fact>; if the draft is solid reply SAY: <the draft, unchanged>. ONE line."
        continue
      fi
      echo "$(date) | $task => $final" >> /root/history
      echo "AGENT_ANSWER: $final"
      exit 0
      ;;
    "[qwen:"*)
      # a device/backend notice (e.g. an empty model reply), not an
      # answer. Nudge, escalate, then give up HONESTLY: three silent
      # replies in a row means this brain is not answering - an endless
      # nudge loop helps no one.
      empties=$((empties + 1))
      echo "[agent] empty or device reply ($empties in a row)"
      if [ $empties -ge 3 ]; then
        log_failure "model silent 3x in a row - task not completed"
        echo "AGENT_ANSWER: (the model gave no answer 3 times in a row - task not completed)"
        echo "[agent] stopped: model not answering"
        exit 1
      fi
      if [ $empties -eq 2 ]; then
        prompt="$SYS TASK: $task RESULT: (your last TWO replies were empty) You MUST reply now with ONE line starting with a TOOL verb or SAY: - do not reply empty."
      else
        prompt="$SYS TASK: $task RESULT: (your last reply was empty) Reply with EXACTLY ONE line: a TOOL verb or SAY: <final answer>"
      fi
      ;;
    *)
      echo "$(date) | $task => $ans" >> /root/history
      echo "AGENT_ANSWER: $ans"
      exit 0
      ;;
  esac
done
log_failure "round cap ($maxround) - last result: $(printf '%s' "$prev_out" | head -c 120)"
echo "$(date) | $task => (round limit) $prev_out" >> /root/history
echo "AGENT_ANSWER: (round limit reached) last tool result: $prev_out"
echo "[agent] stopped: tool round limit ($maxround) reached"
exit 1
