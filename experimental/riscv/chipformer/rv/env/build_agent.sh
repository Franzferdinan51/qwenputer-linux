#!/bin/bash
# Build the bare-metal agent harness (progs/agent.c) for the Qwenputer
# machine:   build_agent.sh [out.elf]
# Emits <out.elf>.sym.json with the mailbox symbol addresses for the
# runner (chipformer/rv/bareagent.py).
#
# Two toolchain routes, first found wins:
#   1. xPack riscv-none-elf-gcc (~/workspace/toolchains/xpack-*/ or
#      $RISCV_XPACK) + newlib, glued by progs/agent_newlib.c
#   2. Debian riscv64-unknown-elf-gcc + picolibc, using env/sys.c
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-$HERE/agent.elf}
MARCH=rv32im_zba
# Map + mailbox knobs: defaults are the stock 32 KB circuit build.
# Grown-map build:  AGENT_LD=$HERE/rom-big.ld AGENT_MB=8192 build_agent.sh
LD=${AGENT_LD:-$HERE/rom.ld}
AGENT_MB=${AGENT_MB:-896}

XPACK=${RISCV_XPACK:-$(ls -d ~/workspace/toolchains/xpack-riscv-none-elf-gcc-*/ 2>/dev/null | head -1)}
if [ -n "${XPACK:-}" ] && [ -x "$XPACK/bin/riscv-none-elf-gcc" ]; then
  GCC=$XPACK/bin/riscv-none-elf-gcc
  NM=$XPACK/bin/riscv-none-elf-nm
  LIBGCC=$($GCC -march=rv32im -mabi=ilp32 -print-libgcc-file-name)
  MULTI=$($GCC -march=$MARCH -mabi=ilp32 -print-multi-directory)
  LIBC=$XPACK/riscv-none-elf/lib/$MULTI/libc.a
  [ -f "$LIBC" ] || LIBC=$(find "$XPACK/riscv-none-elf/lib" -path "*rv32im*ilp32*" -name libc.a | head -1)
  echo "route: xPack newlib ($GCC, libc=$LIBC)"
  $GCC -march=$MARCH -mabi=ilp32 -O2 -I"$HERE" -DMB_SIZE=$AGENT_MB \
    -nostdlib -nostartfiles -static -T"$LD" -o "$OUT" \
    "$HERE/crt0.S" "$HERE/progs/agent.c" "$HERE/progs/agent_newlib.c" \
    -Wl,--start-group "$LIBC" "$LIBGCC" -Wl,--end-group
else
  PICO=/usr/lib/picolibc/riscv64-unknown-elf
  GCC=riscv64-unknown-elf-gcc
  NM=riscv64-unknown-elf-nm
  LIBGCC=$($GCC -march=rv32im -mabi=ilp32 -print-libgcc-file-name)
  echo "route: Debian picolibc ($GCC)"
  $GCC -march=$MARCH -mabi=ilp32 -O2 -isystem "$PICO/include" -I"$HERE" -DMB_SIZE=$AGENT_MB \
    -nostdlib -nostartfiles -static -T"$LD" -o "$OUT" \
    "$HERE/crt0.S" "$HERE/progs/agent.c" "$HERE/sys.c" \
    -Wl,--start-group "$PICO/lib/release/rv32im/ilp32/libc.a" "$LIBGCC" -Wl,--end-group
fi

$NM "$OUT" | AGENT_MB=$AGENT_MB python3 -c '
import sys, json, os
d = {p[2]: int(p[0], 16) for p in (l.split() for l in sys.stdin)
     if len(p) == 3 and p[2].startswith("mb_")}
assert len(d) == 5, d
d["mb_size"] = int(os.environ["AGENT_MB"])
json.dump(d, open(sys.argv[1], "w"), indent=1)
print("mailbox symbols:", d)' "$OUT.sym.json"
ls -la "$OUT"
