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
#   TOOL remember <fact>          keep a fact in /root/BRAIN.md
#   TOOL recall <words>           search BRAIN.md, the notebook, history
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
#   SAY: <answer>                 finish
#
# Self-learning (Hermes' closed loop, shrunk): a task that took real
# work (>=4 tool calls, or a failure then a recovery) earns ONE
# reflection turn before the answer is delivered - the harness asks
# what is worth keeping, as a skill or a remembered fact.
#
# Usage: hermes-agent "your task in quotes"
# Example: hermes-agent "write hello into /tmp/hi.txt"

# NOTE: SYS is built in two assignments because the guest tty drops
# input past ~1021 chars on a single line (measured 2026-10-31: a
# 1300-char line arrived as exactly 1021). No line fed to the guest -
# in this file or any install - may approach that limit.
SYS='Answer with exactly one line, no explanation. Verbs: TOOL run <busybox command> - run a command, you get its output as RESULT. TOOL remember <fact> - keep a fact in your long-term memory. TOOL recall <words> - search your memory, notebook and history. TOOL schedule <minutes> <task> - do a task later; TOOL schedule every <minutes> <task> repeats it. TOOL jobs / TOOL unschedule <n>. TOOL saveskill <name> :: <procedure> - save how to do something, so future you knows; TOOL skill <name> loads one back; check TOOL skills before inventing an approach. TOOL saveapp <name> :: <one-line shell program> - build yourself a small app when you hit a gap (bad at something? make a tool for it once, use it forever); TOOL app <name> <args> runs it; TOOL apps lists them.'
SYS="$SYS"' SAY: <answer> - finish, and SAY only reports what a RESULT showed you. Claiming is not doing: remembering, saving and building happen ONLY through their TOOL verbs - if you did not send the verb, it did not happen. You have a whole linux system: files persist, agentshell runs your scheduled jobs, your memory is /root/BRAIN.md, notebook /root/nb. Examples - task "multiply 123456789 by 987654321 exactly": TOOL run echo $((123456789 * 987654321)). Task "check the disk every 30 minutes": TOOL schedule every 30 check disk space with df and remember the result. Task "remember my favorite fruit is kiwi": TOOL remember the favorite fruit is kiwi. Task "make /tmp/hi.txt containing hello, then read it back": TOOL run echo hello > /tmp/hi.txt and after the result TOOL run cat /tmp/hi.txt then SAY: what the file says.'

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
  # the safety denylist: whoever asked, these never run
  case "$1" in
    *"rm -rf /"*|*"rm -rf /*"*|*mkfs*|*"dd if="*|*"of=/dev/"*|\
    *shutdown*|*poweroff*|*"reboot"*|*":(){"*|*"chmod -R 777 /"*)
      return 0 ;;
  esac
  return 1
}

result_prompt() {
  prompt="$SYS TASK: $task RESULT: $1 Now reply with EXACTLY ONE line: a TOOL verb or SAY: <final answer>"
}

task="$*"
[ -z "$task" ] && { echo 'usage: hermes-agent "your task in quotes"'; echo 'example: hermes-agent "write hello into /tmp/hi.txt"'; exit 1; }

# one agent at a time: the console protocol is a single queue, two
# loops would interleave prompts and answers into mush. agentd sets
# AGENTD=1 and defers when busy; a human's run waits a little.
if ! mkdir /tmp/agent.lockdir 2>/dev/null; then
  if [ -n "$AGENTD" ]; then
    echo "AGENT_BUSY"
    exit 3
  fi
  w=0
  while [ $w -lt 30 ]; do
    sleep 2; w=$((w + 2))
    mkdir /tmp/agent.lockdir 2>/dev/null && break
  done
  mkdir /tmp/agent.lockdir 2>/dev/null || { echo "AGENT_ANSWER: (the agent is busy with another task - try again shortly)"; exit 1; }
fi
: > /tmp/agent.runs.$$
trap 'rm -rf /tmp/agent.lockdir; rm -f /tmp/agent.runs.$$' EXIT INT TERM

mkdir -p /root/nb /root/skills /root/apps
# wake-up context: identity + curated memory fold into the first
# prompt only, size-capped (Hermes keeps memory small on purpose -
# curation over bulk)
soul=""; brain=""
[ -f /root/SOUL.md ] && soul=$(head -c 300 /root/SOUL.md | tr '\n' ' ')
[ -f /root/BRAIN.md ] && brain=$(head -c 700 /root/BRAIN.md | tr '\n' ' ')
prompt="$SYS"
[ -n "$soul" ] && prompt="$prompt YOU: $soul"
[ -n "$brain" ] && prompt="$prompt MEMORY: $brain"
prompt="$prompt TASK: $task"

prev_cmd=""
prev_out=""
empties=0
toolcalls=0
failed=0
recovered=0
did_reflect=0
pending_final=""
round=0
while [ $round -lt 9 ]; do
  round=$((round + 1))
  ans=$(qwen_ask "$prompt")
  # ONE call per reply: the brain likes to simulate the whole
  # conversation in a single line (a glued "; SAY:" tail, a second
  # TOOL call, an invented RESULT). Cut everything from the first
  # simulated continuation on - only the first verb is real.
  ans=$(printf '%s' "$ans" | sed -e 's/; *SAY:.*//' -e 's/; *RESULT:.*//' -e 's/; *TOOL .*//' -e 's/\. *TOOL .*//' -e 's/\. *SAY:.*//' -e 's/ TOOL run .*//' -e 's/ SAY: .*//' -e 's/ then SAY:.*//' -e 's/ then TOOL .*//')
  # a bare empty reply (the fast core delivers it unadorned) must take
  # the same nudge-and-escalate path as a device notice - left alone
  # it would fall into the catch-all and be delivered as an (empty)
  # final answer, which is how the first v3 regression test "passed"
  # an empty answer in 1 round
  [ -z "$(printf '%s' "$ans" | tr -d ' \t')" ] && ans="[qwen: empty answer]"
  echo "[agent] model: $ans"

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

  # the reflection turn's reply is handled by the same verb machinery;
  # whatever it says, afterwards we deliver the stashed final answer
  if [ -n "$pending_final" ]; then
    case "$ans" in
      *"TOOL saveskill "*)
        spec=$(printf '%s' "$ans" | sed 's/.*TOOL saveskill //')
        name=$(printf '%s' "$spec" | sed 's/ .*//;s/::.*//')
        body=$(printf '%s' "$spec" | sed 's/^[^:]*:: *//')
        { echo "# skill: $name"; echo "saved: $(date)"; echo "$body"; } > "/root/skills/$name.md"
        echo "[agent] skill saved: $name" ;;
      *"TOOL remember "*)
        fact=$(printf '%s' "$ans" | sed 's/.*TOOL remember //')
        grep -qxF "$fact" /root/BRAIN.md 2>/dev/null || echo "$fact" >> /root/BRAIN.md
        echo "[agent] remembered: $fact" ;;
      *) echo "[agent] reflection: nothing kept" ;;
    esac
    echo "$(date) | $task => $pending_final" >> /root/history
    echo "AGENT_ANSWER: $pending_final"
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
        echo "[agent] repeat call blocked: $ans"
        prompt="$SYS TASK: $task RESULT: (you already sent that exact call last round and it was done - do NOT send it again. Send the NEXT step of your task as one line, or SAY: <final answer> if the task is finished)"
        continue
      fi
      prev_ans="$ans" ;;
  esac

  case "$ans" in
    *"TOOL remember "*)
      empties=0
      fact=$(printf '%s' "$ans" | sed 's/.*TOOL remember //')
      grep -qxF "$fact" /root/BRAIN.md 2>/dev/null || echo "$fact" >> /root/BRAIN.md
      result_prompt "remembered: $fact (it is in your long-term memory now)"
      ;;
    *"TOOL recall "*)
      empties=0
      words=$(printf '%s' "$ans" | sed 's/.*TOOL recall //')
      pat=$(printf '%s' "$words" | sed 's/ /|/g')
      hits=$(grep -iE "$pat" /root/BRAIN.md /root/history /root/nb/* /root/skills/*.md 2>/dev/null | head -c 300 | tr '\n' ' ')
      [ -z "$hits" ] && hits="(nothing found for: $words)"
      result_prompt "$hits"
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
      if grep -qF "|$jtask" /root/jobs 2>/dev/null; then
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
      if [ -n "$n" ] && [ -f /root/jobs ]; then
        sed -i "${n}d" /root/jobs
        result_prompt "job $n removed"
      else
        result_prompt "(no job number given)"
      fi
      ;;
    *"TOOL saveskill "*)
      empties=0
      spec=$(printf '%s' "$ans" | sed 's/.*TOOL saveskill //')
      name=$(printf '%s' "$spec" | sed 's/ .*//;s/::.*//')
      body=$(printf '%s' "$spec" | sed 's/^[^:]*:: *//')
      { echo "# skill: $name"; echo "saved: $(date)"; echo "$body"; } > "/root/skills/$name.md"
      result_prompt "skill '$name' saved - future you can load it with TOOL skill $name"
      ;;
    *"TOOL addskill "*)
      empties=0
      spec=$(printf '%s' "$ans" | sed 's/.*TOOL addskill //')
      name=$(printf '%s' "$spec" | sed 's/ .*//;s/::.*//')
      body=$(printf '%s' "$spec" | sed 's/^[^:]*:: *//')
      echo "$body" >> "/root/skills/$name.md"
      result_prompt "line added to skill '$name'"
      ;;
    *"TOOL fixskill "*)
      empties=0
      spec=$(printf '%s' "$ans" | sed 's/.*TOOL fixskill //')
      name=$(printf '%s' "$spec" | sed 's/ .*//;s/::.*//')
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
      name=$(printf '%s' "$ans" | sed 's/.*TOOL skill //;s/ .*//')
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
      name=$(printf '%s' "$spec" | sed 's/ .*//;s/::.*//')
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
      name=$(printf '%s' "$spec" | sed 's/ .*//')
      args=$(printf '%s' "$spec" | sed 's/^[^ ]* *//')
      if [ -x "/root/apps/$name" ]; then
        raw=$(eval "\"/root/apps/$name\" $args" 2>&1); st=$?
        [ $st -ne 0 ] && failed=1
        out=$(printf '%s' "$raw" | tr '\n' ' ' | head -c 300)
        [ -z "$out" ] && out="(app ran, no output, exit $st)"
      else
        out="(no app named $name - see TOOL apps)"
      fi
      result_prompt "$out"
      ;;
    *"TOOL run "*)
      empties=0; toolcalls=$((toolcalls + 1))
      cmd=$(printf '%s' "$ans" | sed 's/.*TOOL run //')
      if deny "$cmd"; then
        echo "[agent] refused (denylist): $cmd"
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
      echo "[agent] ran: $cmd"
      echo "[agent] result: $out"
      result_prompt "$out"
      ;;
    *"SAY: "*)
      final=$(printf '%s' "$ans" | sed 's/.*SAY: //')
      # the closed loop: real work earns one reflection turn before
      # the answer goes out (>=4 tool calls, or a recovery) - what
      # should future-you keep from this?
      if [ $did_reflect -eq 0 ] && { [ $toolcalls -ge 4 ] || [ $recovered -eq 1 ]; }; then
        did_reflect=1
        pending_final="$final"
        prompt="$SYS TASK: reflection. The task '$task' is DONE (answer: $final). Before finishing: if a reusable procedure is worth keeping reply TOOL saveskill <name> :: <steps and pitfalls>; if a fact is worth keeping reply TOOL remember <fact>; otherwise reply SAY: done. ONE line."
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
echo "$(date) | $task => (round limit) $prev_out" >> /root/history
echo "AGENT_ANSWER: (round limit reached) last tool result: $prev_out"
echo "[agent] stopped: tool round limit (9) reached"
