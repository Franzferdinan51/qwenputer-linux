#!/bin/bash
# v3 rerun: skill (B) and todo+sub (C), after the multi-line result fix.
set -x
cd /work || exit 1
COMMON="--model /models/qwen3-1.7b-q8_0.asmbin --mem 4G --smp 8 --skills /work/qwenputer/skills --sessions /out/sessions-v3"

python3 qwenputer/qp_agent.py $COMMON --memory /out/memory-v3.json \
  --agent "Load the exact-arithmetic skill and follow it. What is 777*888? Answer with SAY: and the result." \
  > /out/robin-v3-b-skill.log 2>&1
echo "V3B_EXIT=$?"

python3 qwenputer/qp_agent.py $COMMON --memory /out/memory-v3.json \
  --max-steps 10 \
  --agent "First use TODO: add plan the work. Then delegate with SUB: compute 99*99 using CALC and SAY the result. When the subtask result comes back, answer with SAY: followed by that number." \
  > /out/robin-v3-c-sub.log 2>&1
echo "V3C_EXIT=$?"
grep -h "\[answer\]" /out/robin-v3-b-skill.log /out/robin-v3-c-sub.log
echo "V3B_DONE"
