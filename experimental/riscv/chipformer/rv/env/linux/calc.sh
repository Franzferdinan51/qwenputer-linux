#!/bin/sh
# calc - exact integer arithmetic for the agent. This rootfs has no
# bc, awk or python, but the shell's $(( )) is exact for 64-bit
# integers - and the model must never do arithmetic in its head.
#
# Usage: calc '<expression>'   e.g. calc '123456789 * 987654321'
# Allows digits, whitespace and + - * / % ( ) only; anything else is
# refused rather than evaluated.
expr="$*"
[ -z "$expr" ] && { echo "usage: calc '<expression>'"; exit 1; }
# anything outside digits, blanks and +-*/%() is refused, never eval'd
bad=$(printf '%s' "$expr" | tr -d '0-9+ */%()-')
if [ -n "$bad" ]; then
  echo "(calc refuses that: digits and + - * / % ( ) only)"
  exit 1
fi
echo $(($expr))
