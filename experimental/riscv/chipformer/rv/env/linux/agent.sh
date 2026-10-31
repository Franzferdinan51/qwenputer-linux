#!/bin/sh
# agent - a tiny agent harness in the Hermes-gadget shape: the model is
# the brain, this shell is the hands. A pre-configured system prompt and
# one tool (run) are baked in below; the loop talks to the model over the
# QWEN: console protocol (see ask) - one line each way, because the
# device flattens every answer to a single line.
#
# Protocol:
#   model -> "TOOL run <busybox shell command>" : the harness runs it and
#            sends the output back as the next prompt's RESULT
#   model -> "SAY: <text>"                      : the final answer
#   anything else                               : treated as the answer
#
# Usage: agent <task words...>
# Example: agent write the word banana into /tmp/hello.txt then read it back

SYS='Answer with exactly one line, no explanation. To run a busybox shell command reply: TOOL run <command> - you will get its output back as a RESULT. When the task is finished reply: SAY: <final answer>. Example - task "make a file /tmp/hi.txt containing hello": you reply TOOL run echo hello > /tmp/hi.txt and after the result you reply SAY: done'

qwen_ask() {
  # the prompt must go to the CONSOLE, not stdout: this function is called
  # inside $(...), where stdout is the capture pipe (ask.sh never noticed -
  # it is never called in a substitution)
  printf 'QWEN:%s\n' "$1" > /dev/console
  IFS= read -r _ans
  printf '%s' "$_ans"
}

task="$*"
[ -z "$task" ] && { echo "usage: agent <task>"; exit 1; }
prompt="$SYS TASK: $task"
prev_cmd=""
prev_out=""
round=0
while [ $round -lt 7 ]; do
  round=$((round + 1))
  ans=$(qwen_ask "$prompt")
  echo "[agent] model: $ans"
  case "$ans" in
    *"TOOL run "*)
      cmd=$(printf '%s' "$ans" | sed 's/.*TOOL run //')
      if [ "$cmd" = "$prev_cmd" ]; then
        echo "[agent] repeat of the last command - not running it again"
        prompt="$SYS TASK: $task RESULT: you already ran that command, its result was: $prev_out Do the NEXT step, or reply SAY: <final answer>"
        continue
      fi
      raw=$(eval "$cmd" 2>&1); st=$?
      out=$(printf '%s' "$raw" | tr '\n' ' ' | head -c 300)
      # a silent success (a file write prints nothing) reads as "nothing
      # happened" to the model and it just retries - say what happened
      [ -z "$out" ] && out="(command ran, no output, exit $st)"
      prev_cmd="$cmd"
      prev_out="$out"
      echo "[agent] ran: $cmd"
      echo "[agent] result: $out"
      prompt="$SYS TASK: $task RESULT: $out Now reply with EXACTLY ONE line: TOOL run <command> or SAY: <final answer>"
      ;;
    *"SAY: "*)
      final=$(printf '%s' "$ans" | sed 's/.*SAY: //')
      echo "AGENT_ANSWER: $final"
      exit 0
      ;;
    "[qwen:"*)
      # a device/backend notice (e.g. an empty model reply), not an
      # answer - nudge once and keep looping (the round cap bounds this)
      echo "[agent] empty or device reply - nudging the model"
      prompt="$SYS TASK: $task RESULT: (your last reply was empty) Reply with EXACTLY ONE line: TOOL run <command> or SAY: <final answer>"
      ;;
    *)
      echo "AGENT_ANSWER: $ans"
      exit 0
      ;;
  esac
done
echo "AGENT_ANSWER: (round limit reached) last tool result: $prev_out"
echo "[agent] stopped: tool round limit (7) reached"
