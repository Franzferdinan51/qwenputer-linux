#!/bin/bash
# Runs INSIDE the qp-test container on Robin (KVM available via --device).
set -x
cd /work || exit 1

# 1. model download + export (skipped when already present in /models)
if [ ! -f /models/qwen3-0.6b-q8_0.asmbin ]; then
  python3 - <<'PYEOF' || exit 1
from huggingface_hub import snapshot_download
snapshot_download(
    "Qwen/Qwen3-0.6B",
    allow_patterns=["*.safetensors", "tokenizer.json", "config.json"],
    local_dir="/models/qwen3-0.6b-bf16",
)
print("download ok")
PYEOF
  python3 scripts/export_weights.py \
    --model-dir /models/qwen3-0.6b-bf16 \
    --out /models/qwen3-0.6b-q8_0.asmbin --scale-dtype fp16 || exit 1
fi
ls -la /models/qwen3-0.6b-q8_0.asmbin
sha256sum /models/qwen3-0.6b-q8_0.asmbin

# 2. build
make clean >/dev/null 2>&1
make os FAST_Q8=1 || exit 1
ls -la build/asm-llm-os.elf

# 3. oracle tests
python3 qwenputer/test_qpcalc.py || exit 1

# 4. boot test under KVM (16 checks, no inference)
QEMU_ACCEL=kvm QEMU_CPU=host python3 qwenputer/qp_serial_test.py \
  --model /models/qwen3-0.6b-q8_0.asmbin \
  --smp 8 --timeout 120 --log /out/robin-serial-test.out || exit 1

# 5. KVM chat bench: 64 tokens, decode speed printed by the OS itself
timeout 600 qemu-system-x86_64 -machine accel=kvm -cpu host -m 2G -smp 8 \
  -kernel build/asm-llm-os.elf -initrd /models/qwen3-0.6b-q8_0.asmbin \
  -append 'prompt="Write a one-paragraph story about a duck who learns to fly." max_tokens=64 think=0 shutdown=1' \
  -nographic -serial stdio -monitor none -display none > /out/robin-kvm-chat.log 2>&1
echo "KVM_CHAT_EXIT=$?"
grep -E "prefill|decode|boot" /out/robin-kvm-chat.log
echo "ROBIN_TEST_DONE"
