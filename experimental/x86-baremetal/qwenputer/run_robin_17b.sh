#!/bin/bash
# Runs INSIDE the qp-test container on Robin: Qwen3-1.7B end-to-end.
set -x
cd /work || exit 1

# 1. download + export the 1.7B (direct from HF on Robin)
if [ ! -f /models/qwen3-1.7b-q8_0.asmbin ]; then
  python3 - <<'PYEOF' || exit 1
from huggingface_hub import snapshot_download
snapshot_download(
    "Qwen/Qwen3-1.7B",
    allow_patterns=["*.safetensors", "tokenizer.json", "config.json"],
    local_dir="/models/qwen3-1.7b-bf16",
)
print("download ok")
PYEOF
  python3 scripts/export_weights.py \
    --model-dir /models/qwen3-1.7b-bf16 \
    --out /models/qwen3-1.7b-q8_0.asmbin --scale-dtype fp16 || exit 1
fi
ls -la /models/qwen3-1.7b-q8_0.asmbin
sha256sum /models/qwen3-1.7b-q8_0.asmbin

# 2. build (same kernel — geometry comes from the .asmbin header)
make clean >/dev/null 2>&1
make os FAST_Q8=1 || exit 1
ls -la build/asm-llm-os.elf

# 3. serial boot checks with the 1.7B bound (needs 4G: model is ~2 GB)
QEMU_ACCEL=kvm QEMU_CPU=host python3 qwenputer/qp_serial_test.py \
  --model /models/qwen3-1.7b-q8_0.asmbin --mem 4G \
  --smp 8 --timeout 180 --log /out/robin-17b-serial-test.out || exit 1

# 4. KVM chat bench, 64 tokens
timeout 600 qemu-system-x86_64 -machine accel=kvm -cpu host -m 4G -smp 8 \
  -kernel build/asm-llm-os.elf -initrd /models/qwen3-1.7b-q8_0.asmbin \
  -append 'prompt="Write a one-paragraph story about a duck who learns to fly." max_tokens=64 think=0 shutdown=1' \
  -nographic -serial stdio -monitor none -display none > /out/robin-17b-chat.log 2>&1
echo "KVM_CHAT_EXIT=$?"
grep -E "prefill|decode|boot" /out/robin-17b-chat.log

# 5. agent framework v2 end-to-end with the 1.7B
#    session 1: calc -> PUT note -> GET it back -> SAY
#    session 2 (fresh boot, same memory file): recall the note cold
rm -f /out/memory17.json
python3 qwenputer/qp_agent.py --model /models/qwen3-1.7b-q8_0.asmbin \
  --mem 4G --smp 8 --memory /out/memory17.json --sessions /out/sessions17 \
  --agent "Compute 123456 times 789 exactly. Then save the result as a note named product. Then read the note back to confirm it. Then answer with SAY: and the number." \
  > /out/robin-17b-agent1.log 2>&1
echo "AGENT1_EXIT=$?"
cat /out/memory17.json
python3 qwenputer/qp_agent.py --model /models/qwen3-1.7b-q8_0.asmbin \
  --mem 4G --smp 8 --memory /out/memory17.json --sessions /out/sessions17 \
  --agent "Look in your memory: read the note named product and answer with SAY: and its value." \
  > /out/robin-17b-agent2.log 2>&1
echo "AGENT2_EXIT=$?"
grep -h "answer" /out/robin-17b-agent1.log /out/robin-17b-agent2.log
echo "ROBIN_17B_DONE"
