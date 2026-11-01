"""Parity check: fastlinux.c vs the Python LinuxMachine oracle.

Both machines boot the same image with no input and retire exactly the
same number of instructions; their full state digests (pc, privilege,
registers, step count, mtime, all 64 MiB of RAM — FNV-1a, byte order
fixed by fastlinux.c's fnv_state) and their console output must match
exactly. Deterministic time (1 tick per retired instruction) is what
makes this a fair fight.

    python3 -m chipformer.rv.paritycheck Image [--steps 8000000]
"""

import argparse
from pathlib import Path

from chipformer.rv.fastmachine import FastLinuxMachine
from chipformer.rv.linux import LinuxMachine

HERE = Path(__file__).parent


def py_hash(m: LinuxMachine) -> str:
    h = 1469598103934665603

    def mix(b):
        nonlocal h
        h ^= b
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF

    def mix32(v):
        for k in range(4):
            mix((v >> 8 * k) & 0xFF)

    def mix64(v):
        for k in range(8):
            mix((v >> 8 * k) & 0xFF)

    mix32(m.pc)
    mix32(m.priv)
    for r in m.x:
        mix32(r)
    mix64(m.steps)
    mix64(m.mtime)
    for b in bytes(m.ram):
        mix(b)
    return "%016x" % h


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--dtb", default=str(HERE / "env" / "linux" / "sixtyfourmb.dtb"))
    ap.add_argument("--steps", type=int, default=8_000_000)
    ap.add_argument("--bin", default=None)
    args = ap.parse_args()

    image = Path(args.image).read_bytes()
    dtb = Path(args.dtb).read_bytes()

    pm = LinuxMachine()
    pm.load_image(image)
    pm.boot(dtb)
    pm.run(args.steps)
    ph = py_hash(pm)
    print(f"[parity: python  steps={pm.steps:,} hash={ph} out={len(pm.out)} B]")

    fm = FastLinuxMachine(args.image, args.dtb, backend=None, binary=args.bin)
    try:
        fm.run(args.steps)
        fh = fm.hash()
        fout = bytes(fm.out)
    finally:
        fm.close()
    print(f"[parity: fast    steps={fm.steps:,} hash={fh} out={len(fout)} B]")

    ok_hash = ph == fh
    ok_out = bytes(pm.out) == fout
    print(f"[parity: hash {'MATCH' if ok_hash else 'MISMATCH'}; "
          f"console {'MATCH' if ok_out else 'MISMATCH'}]")
    if not ok_out:
        a, b = bytes(pm.out), fout
        for i, (u, v) in enumerate(zip(a, b)):
            if u != v:
                print(f"  first console diff at byte {i}: "
                      f"python ...{a[max(0, i - 40):i + 40]!r} "
                      f"fast ...{b[max(0, i - 40):i + 40]!r}")
                break
    return 0 if (ok_hash and ok_out) else 1


if __name__ == "__main__":
    raise SystemExit(main())
