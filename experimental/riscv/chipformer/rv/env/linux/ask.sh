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

if command -v devmem >/dev/null 2>&1; then
  QW=0x10010000
  for b in $(printf '%s' "$1" | od -An -tu1); do
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
  printf '%s' "$1" | dd of=/dev/mem bs=1 seek=$D 2>/dev/null
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
  printf 'QWEN:%s\n' "$1"
  IFS= read -r ans
  printf 'QWEN_ANSWER: %s\n' "$ans"
fi
