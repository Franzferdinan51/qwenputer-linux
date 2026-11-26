#!/bin/bash
# LFM3 regression pack: serial REPL checks on the LFM3 model, agent
# framework end-to-end on LFM3, and the Qwen3-0.6B regression (serial
# checks + raw chat parity) on the SAME new kernel binary.
set -x
cd /work || exit 1

echo "=== LFM3 serial REPL checks ==="
python3 qwenputer/qp_serial_test.py --kernel build/asm-llm-os.elf \
  --model /models/lfm3-1.7b-q8_0.asmbin --mem 4G --smp 8 \
  --log /out/qp-lfm3-serial-test.out
echo "LFM3_SERIAL_EXIT=$?"

echo "=== LFM3 agent framework ==="
rm -f /out/memorylfm3.json
python3 qwenputer/qp_agent.py --model /models/lfm3-1.7b-q8_0.asmbin \
  --mem 4G --smp 8 --memory /out/memorylfm3.json --sessions /out/sessionslfm3 \
  --agent "Compute 123456 times 789 exactly. Then save the result as a note named product. Then read the note back to confirm it. Then answer with SAY: and the number." \
  > /out/robin-lfm3-agent1.log 2>&1
echo "LFM3_AGENT1_EXIT=$?"
cat /out/memorylfm3.json
python3 qwenputer/qp_agent.py --model /models/lfm3-1.7b-q8_0.asmbin \
  --mem 4G --smp 8 --memory /out/memorylfm3.json --sessions /out/sessionslfm3 \
  --agent "Look in your memory: read the note named product and answer with SAY: and its value." \
  > /out/robin-lfm3-agent2.log 2>&1
echo "LFM3_AGENT2_EXIT=$?"
grep -h "answer" /out/robin-lfm3-agent1.log /out/robin-lfm3-agent2.log

echo "=== Qwen3-0.6B regression: serial checks ==="
python3 qwenputer/qp_serial_test.py --kernel build/asm-llm-os.elf \
  --model /models/qwen3-0.6b-q8_0.asmbin --mem 2G --smp 8 \
  --log /out/qp-qwen-regression-serial.out
echo "QWEN_SERIAL_EXIT=$?"
echo "CHECKS_DONE"
