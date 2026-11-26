#!/bin/bash
# Agent-framework-only rerun inside the qp-test container on Robin.
set -x
cd /work || exit 1
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
echo "AGENT_RERUN_DONE"
