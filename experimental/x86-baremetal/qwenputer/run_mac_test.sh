#!/bin/bash
# Qwenputer-X86 full Mac test: build on the Mac, export the model on the Mac,
# boot-test the OS on the Mac. Logs to ~/qwenputer-x86-test/mac-test.log.
set -x
export PATH=/opt/homebrew/bin:$PATH
cd ~/qwenputer-x86-test/src || exit 1

# 1. python env for the exporter
if [ ! -x ../venv/bin/python ]; then
  python3 -m venv ../venv || exit 1
fi
../venv/bin/pip install -q numpy safetensors huggingface_hub || exit 1

# 2. model download (direct from HF on the Mac)
if [ ! -f ../models/qwen3-0.6b-bf16/model.safetensors ]; then
  ../venv/bin/python - <<'PYEOF' || exit 1
from huggingface_hub import snapshot_download
snapshot_download(
    "Qwen/Qwen3-0.6B",
    allow_patterns=["*.safetensors", "tokenizer.json", "config.json"],
    local_dir="../models/qwen3-0.6b-bf16",
)
print("download ok")
PYEOF
fi

# 3. export .asmbin
if [ ! -f ../models/qwen3-0.6b-q8_0.asmbin ]; then
  ../venv/bin/python scripts/export_weights.py \
    --model-dir ../models/qwen3-0.6b-bf16 \
    --out ../models/qwen3-0.6b-q8_0.asmbin --scale-dtype fp16 || exit 1
fi
ls -la ../models/qwen3-0.6b-q8_0.asmbin

# 4. build the kernel natively on the Mac (nasm + x86_64-elf binutils)
make clean >/dev/null 2>&1
make os FAST_Q8=1 LD=x86_64-elf-ld OBJCOPY=x86_64-elf-objcopy || exit 1
ls -la build/asm-llm-os.elf

# 5. oracle tests
python3 qwenputer/test_qpcalc.py || exit 1

# 6. boot test: 16 serial checks in QEMU (TCG — ARM host, x86 guest)
python3 qwenputer/qp_serial_test.py \
  --model ../models/qwen3-0.6b-q8_0.asmbin \
  --smp 8 --timeout 300 --log ../mac-serial-test.out
echo "MAC_TEST_EXIT=$?"
