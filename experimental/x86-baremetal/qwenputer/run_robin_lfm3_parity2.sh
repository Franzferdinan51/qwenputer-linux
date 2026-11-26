#!/bin/bash
# LFM3 parity #2 (raw, prompt 2) + chat-mode smoke, both vs reference v2.
set -x
cd /work || exit 1
python3 - <<'EOF'
import json, re, subprocess, sys

ref = json.load(open("/refout/lfm3-reference-v2.json"))

def boot(append, timeout=600):
    cmd = [
        "qemu-system-x86_64", "-machine", "accel=kvm", "-cpu", "host",
        "-m", "4G", "-smp", "8",
        "-kernel", "build/asm-llm-os.elf",
        "-initrd", "/models/lfm3-1.7b-q8_0.asmbin",
        "-append", append,
        "-nographic", "-serial", "stdio", "-monitor", "none",
        "-display", "none",
    ]
    return subprocess.run(cmd, capture_output=True, text=True,
                          timeout=timeout).stdout

# --- raw parity, prompt 2 ---
prompt = "Q: What is 2+2?\nA:"
want = ref[prompt]["gen_ids"]
out = boot(f'prompt="{prompt}" raw=1 single_turn=1 max_tokens=32 '
           "print_tokens=1 shutdown=1 temp=0")
open("/out/robin-lfm3-parity2.log", "w").write(out)
m = re.search(r"GEN_TOKENS:([0-9,]+)", out)
got = [int(x) for x in m.group(1).split(",")] if m else []
print("ref :", want)
print("ours:", got)
print("P2 PARITY_PASS" if got == want else "P2 PARITY_DIFF")

# --- chat mode smoke: answer should mention 4 ---
out = boot('prompt="What is 2+2? Answer with just the number." '
           "single_turn=1 max_tokens=64 print_tokens=1 shutdown=1 temp=0")
open("/out/robin-lfm3-chat.log", "w").write(out)
tail = out.split("qwenputer>")[-1] if "qwenputer>" in out else out[-1500:]
print("CHAT OUTPUT TAIL:", tail[-800:])
EOF
echo "EXIT=$?"
