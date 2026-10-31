# Hybrid models can hide reasoning in their RNN state

Code and results for the write-up at
https://divagr.com/blog/hybrid-models-can-hide-reasoning-in-their-rnn-state

The question: under answer-only training, with no chain of thought allowed, can a hybrid model learn to track state silently in its Gated DeltaNet layers, and does that generalise past the lengths it trained on?

## Results

Answer-only accuracy on parity (the light-switch task), trained up to 64 flips, tested at 128:

| Model | GDN write strength β | Accuracy at 128 flips |
|---|---|---|
| OLMo-Hybrid-7B | up to 2 | 0.89 (6 seeds, 0.53 to 1.00) |
| OLMo-Hybrid-7B, β capped at 1 | up to 1 | 0.52 (6 seeds, 0.49 to 0.56) |
| Qwen3.5-4B | up to 1 | 0.55 (3 seeds) |
| OLMo-3-7B (dense control) | none | 0.54 (3 seeds) |

Counting runs that learned the training range (at least 0.85 at 64 flips) and still scored at least 0.80 at 128:

| Model | Learned 64 flips | Of those, held at 128 |
|---|---|---|
| OLMo-Hybrid-7B | 5 of 6 | 5 of 5 |
| OLMo-Hybrid-7B, β capped at 1 | 2 of 6 | 0 of 2 |
| Qwen3.5-4B | 2 of 3 | 0 of 2 |
| OLMo-3-7B | 0 of 3 | none |

Over all runs, 5 of 6 against 0 of 6 for the two OLMo-Hybrid arms (one-sided Fisher p ≈ 0.008). The cap also makes the training range harder, so the "learned it but didn't generalise" case under the cap rests on 2 OLMo-Hybrid runs plus 2 Qwen3.5 runs. Per-seed numbers are in `results/ft/seeds_table.md`.

Capping β costs OLMo-Hybrid about 3.5% perplexity on plain text.

Where the bit lives: swapping the DeltaNet state between sequences changes nothing (0.00), because attention keeps a copy of the bit. Swapping the state with attention knocked out moves the answer every time (1.00), in layer 16 head 2 and layer 17 head 4.

Compiled β = 2 trackers track parity exactly over 100k tokens, and a planted tracker inside Qwen3.5-4B is ranked 1st of 864 heads by a blind detector.

This is 6 seeds per OLMo-Hybrid arm (3 for the others) and one task, with artificial answer-only pressure, not an RL setup. A four-switch version of the task (T5) was a negative result: both OLMo-Hybrid arms learned it at 16 flips and were at chance by 64, so it can't separate them.

## Layout

- `hcot/tasks.py`: state-tracking tasks with exact answers (T2 is parity, T5 is four switches).
- `hcot/models.py`: model registry and chat formatting.
- `hcot/behave.py`: answer-only and CoT behaviour runs.
- `hcot/train.py`: answer-only LoRA fine-tuning, with the β and decay projections unfrozen.
- `hcot/betamode.py`: changes only the DeltaNet β range (`cap_min`, `cap_sig` for OLMo-Hybrid, `x2` for Qwen3.5).
- `hcot/basecost.py`: KL and perplexity cost of a β change.
- `hcot/mech.py`, `hcot/mech_run.py`: attention knockout, state swaps and probes.
- `hcot/beta_use.py`: whether fine-tuned heads actually use β > 1.
- `hcot/p3.py`: exact trackers compiled into extra DeltaNet layers.
- `hcot/plant.py`, `hcot/control.py`: the planted tracker inside Qwen3.5-4B and the detector's positive control.
- `hcot/seeds.py`, `hcot/report.py`, `hcot/p2report.py`, `hcot/plot.py`: tables and plots.
- `scripts/`: the shell scripts the runs used on GPU boxes.
- `results/`: raw outputs (`p1` stock behaviour, `ft` fine-tuned runs, `p2` mechanism, `p3` compiled trackers, `beta` β-range checks, `plant` the planted tracker).

## Running it

```bash
cd research/hidden-reasoning
uv venv -p 3.12 && uv pip install torch transformers safetensors numpy tokenizers huggingface_hub accelerate peft datasets matplotlib

# answer-only fine-tune on parity, then evaluate past the training length
python -m hcot.train --model olmoh --task T2 --steps 2002 --bs 16 --seed 0 --beta-mode stock --out adapters/olmoh_T2_s0
python -m hcot.behave --model olmoh --adapter adapters/olmoh_T2_s0_final --tasks T2 --formats sym \
    --lengths 16 64 128 256 --n 100 --out results/ft/olmoh_T2_s0_answer.jsonl

# the same with β capped at 1
python -m hcot.train --model olmoh --task T2 --steps 2002 --bs 16 --seed 0 --beta-mode cap_min --out adapters/olmoh_T2_cap_min_s0

# state swap and attention knockout
python -m hcot.mech_run patch --model olmoh --adapter adapters/olmoh_T2_s0_final --cells T2:sym:64 T2:sym:128 --n 100
```

`hcot/p3.py` and `hcot/plant.py` use the compiler in this repo's `chipformer` package, so run them with the repo root on the path (`PYTHONPATH=../..`).
