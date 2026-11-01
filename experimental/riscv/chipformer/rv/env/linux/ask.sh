#!/bin/sh
# ask - send a prompt to the Qwen coprocessor and print its answer.
#
# Two ways in, same backend (see docs/linux.md):
#   1. The MMIO window at 0x10010000 (DATA +0, STATUS +4, CTRL +8),
#      poked from userspace with devmem, or through /dev/mem with dd.
#   2. The console line protocol: print a line starting with "QWEN:"
#      and the answer is typed back on the console as one line, which
#      `read` collects. The prebuilt kernel has no /dev/mem and its
#      8250 driver is built for a single port, so on this rootfs the
#      console protocol is the working channel.
#
# Usage: ask "your question"

# Console answers arrive marked QWENANS: (the host prefixes the lines
# it types, so a human typing at the same console mid-answer is never
# mistaken for the model). Read until a marked line shows up.
read_ans() {
  ans=''
  _t=0
  while [ $_t -lt 6 ]; do
    IFS= read -r _l || return
    case "$_l" in
      QWENANS:*) ans="${_l#QWENANS:}"; return ;;
      *) _t=$((_t + 1)) ;;
    esac
  done
}

# the whole command line is the question: ask what is up must send
# all three words, not just $1 (an unquoted multiword ask used to
# silently send its first word only)
prompt="$*"
if command -v devmem >/dev/null 2>&1; then
  QW=0x10010000
  for b in $(printf '%s' "$prompt" | od -An -tu1); do
    devmem $QW 8 "$b" >/dev/null
  done
  devmem $((QW + 8)) 8 1 >/dev/null
  while [ $(( $(devmem $((QW + 4)) 8) & 1 )) -eq 0 ]; do :; done
  while :; do
    b=$(( $(devmem $QW 8) ))
    [ "$b" -eq 0 ] && break
    printf "\\$(printf '%03o' "$b")"
  done
  echo
elif [ -c /dev/mem ]; then
  D=268439552
  printf '%s' "$prompt" | dd of=/dev/mem bs=1 seek=$D 2>/dev/null
  printf '\001' | dd of=/dev/mem bs=1 seek=$((D + 8)) 2>/dev/null
  while :; do
    s=$(dd if=/dev/mem bs=1 skip=$((D + 4)) count=1 2>/dev/null | od -An -tu1 | tr -d ' \n')
    [ "$s" = "1" ] && break
  done
  while :; do
    b=$(dd if=/dev/mem bs=1 skip=$D count=1 2>/dev/null | od -An -tu1 | tr -d ' \n')
    [ -z "$b" ] || [ "$b" = "0" ] && break
    printf "\\$(printf '%03o' "$b")"
  done
  echo
else
  printf 'QWEN:%s\n' "$prompt"
  read_ans
  # Delegation: when the host bridge is on (qwenbackend.agent_bridge),
  # the model in the seat may answer with a [[RUN-AGENT: ...]] marker
  # instead of an answer - meaning "let the machine's own hermes-agent
  # do this". The agent then runs right here, in this same partition
  # (same files), and its report is folded into a second ask. One level
  # only: a second marker is printed as-is, never chased.
  case "$ans" in
    "[[RUN-AGENT: "*"]]")
      t=$(printf '%s' "$ans" | sed 's/^\[\[RUN-AGENT: //; s/\]\]$//')
      echo "[ask] the model delegated this to hermes-agent: $t"
      HA=/tmp/hermes-agent; [ -x "$HA" ] || HA=/tmp/agent
      rep=$("$HA" "$t" | sed -n 's/^AGENT_ANSWER: //p' | tail -1)
      [ -z "$rep" ] && rep="(the hermes agent produced no answer)"
      echo "[ask] hermes-agent reported: $rep"
      printf 'QWEN:%s HERMES REPORT: %s\n' "$prompt" "$rep"
      read_ans
      ;;
  esac
  printf 'QWEN_ANSWER: %s\n' "$ans"
fi
