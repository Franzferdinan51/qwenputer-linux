#!/bin/bash
# Agent framework v3 tests inside the qp-test container on Robin.
set -x
cd /work || exit 1
COMMON="--model /models/qwen3-1.7b-q8_0.asmbin --mem 4G --smp 8 --skills /work/qwenputer/skills --sessions /out/sessions-v3"

# A: frozen memory snapshot — product=97406784 already in memory17.json
python3 qwenputer/qp_agent.py $COMMON --memory /out/memory17.json \
  --agent "What is the value of the note named product? Answer with SAY: and the value." \
  > /out/robin-v3-a-snapshot.log 2>&1
echo "V3A_EXIT=$?"

# B: skill load + exact arithmetic (777*888 = 689976)
rm -f /out/memory-v3.json
python3 qwenputer/qp_agent.py $COMMON --memory /out/memory-v3.json \
  --agent "Load the exact-arithmetic skill and follow it. What is 777*888? Answer with SAY: and the result." \
  > /out/robin-v3-b-skill.log 2>&1
echo "V3B_EXIT=$?"

# C: todo plan + SUB delegation (99*99 = 9801)
python3 qwenputer/qp_agent.py $COMMON --memory /out/memory-v3.json \
  --max-steps 10 \
  --agent "First add a todo item: plan the work. Then delegate the math with SUB: compute 99*99 using CALC and SAY the result. Then SAY: the subtask answer." \
  > /out/robin-v3-c-sub.log 2>&1
echo "V3C_EXIT=$?"
grep -h "\[answer\]" /out/robin-v3-a-snapshot.log /out/robin-v3-b-skill.log /out/robin-v3-c-sub.log
echo "V3_DONE"
