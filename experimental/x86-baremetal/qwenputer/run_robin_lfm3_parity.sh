#!/bin/bash
# LFM3 parity: boot the OS with the LFM3 model, raw greedy completion,
# compare GEN_TOKENS against the transformers reference.
set -x
cd /work || exit 1
python3 - <<'EOF'
import json, re, subprocess, sys

ref = json.load(open("/refout/lfm3-reference.json"))
prompt = "The capital of France is"
want = ref[prompt]["gen_ids"]
print("reference ids:", want)

cmd = [
    "qemu-system-x86_64", "-machine", "accel=kvm", "-cpu", "host",
    "-m", "4G", "-smp", "8",
    "-kernel", "build/asm-llm-os.elf",
    "-initrd", "/models/lfm3-1.7b-q8_0.asmbin",
    "-append", f'prompt="{prompt}" raw=1 single_turn=1 max_tokens=32 '
               "print_tokens=1 shutdown=1 temp=0",
    "-nographic", "-serial", "stdio", "-monitor", "none", "-display", "none",
]
p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
out = p.stdout
open("/out/robin-lfm3-parity.log", "w").write(out)
m = re.search(r"GEN_TOKENS: ([0-9, ]+)", out)
if not m:
    print("NO GEN_TOKENS FOUND"); print(out[-2000:]); sys.exit(1)
got = [int(x) for x in m.group(1).split(",") if x.strip()]
print("kernel ids:  ", got)
n = min(len(got), len(want))
match = next((i for i in range(n) if got[i] != want[i]), n)
print(f"MATCH: {match}/{len(want)} tokens identical")
print("PARITY_PASS" if got[:len(want)] == want else "PARITY_DIFF")
EOF
echo "PARITY_EXIT=$?"
