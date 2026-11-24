# Experiment 002B — Synthetic-only relational learning proof of concept

**Status:** Executable data preparation, training launcher and evaluation harness. **No LLM training results have been produced in this repository yet.** This track requires no human reviewers or emotional annotation labels.

## What synthetic evidence can and cannot establish

We can test an exact, falsifiable claim: relative to its own frozen base checkpoint, an LLM trained on authored counterfactual conversation preferences learns to **change its preference between the same two responses when earlier context changes**, and this transfers to withheld scenario families. This is evidence of learned context-sensitive behavior on a synthetic distribution.

It does **not** establish felt emotion, love, hatred, empathy as subjective experience, a stable personality, or out-of-distribution human relational competence. Both training and evaluation examples reflect their author's assumptions. A different author, independent distribution or direct human observations would be needed to examine real-world transfer. That later work is optional for running this first controlled experiment.

## Two methods considered

1. **Write a custom neural relational module and loss:** introduces more variables, more maintenance and unclear attribution. Not selected for the first pilot.
2. **Fine-tune one existing language model with DPO/LoRA on synthetic counterfactual comparisons:** selected. The neural changes are in trainable model parameters, not a hidden mood prompt. Reuse the existing `schema.json`, `validate.py`, `export_dpo.py` and upstream TRL `DPOTrainer`.

Do **not** mistake preference optimization on explicit example labels for unprompted *emergence*. The interesting observation would be generalization to novel situations that were not directly taught.

## Scientific controls

- **Family holdout:** seed and challenge examples are split by entire scenario family, never by individual row. Public development/test are *held-out within this curated collection*, not contamination-resistant external benchmarks. Never train on them.
- **Counterfactual reversal:** within each family, the final user utterance and two possible replies are identical, but the preceding circumstance changes and the target preference reverses. Report *both-variant* success per family; one correct row is not enough.
- **Last-turn-only negative control:** train an otherwise identical copy using only the final user utterance. Each two-row family then presents exactly the same prompt with contradictory labels. No deterministic model using only that prompt can satisfy both variants. An apparently high score would indicate leakage or a broken evaluation.
- **Frozen base:** evaluate the original checkpoint *before* adaptation, using identical candidate responses and decoding/scoring rules. Don't compare one optimized checkpoint against a straw-man untrained language model.
- **Style control:** responses should be similarly plausible and fluent, not distinguishable only by emotional tone, length or the presence of context-exclusive nouns. Document easy cases and rewrite them.
- **Different seeds:** repeat successful pilot results with independently initialized LoRA adapters; record failures, not only the best run.
- **No generated response evaluation yet:** this first harness scores candidate continuations by per-token conditional log-likelihood. It is a preference-discrimination test, **not a measure of open-ended generation quality**. Add blinded generation-based tests after this mechanical gate passes.

## Run the experiment locally

From `experiments/relational-learning`:

```bash
# 1. Verify the immutable source examples and checks
python3 tools/generate_challenges.py --check
python3 tools/validate.py data/*.jsonl
python3 -m unittest discover -s tests -p 'test_*.py' -v

# 2. Export author-defined synthetic TRAIN labels explicitly (68 comparisons)
python3 tools/export_dpo.py --mode synthetic --prompt-mode full \
  --data data/train_seed.jsonl data/train_challenge.jsonl \
  --out runs/synthetic-full.jsonl --report runs/synthetic-full.report.json

# 3. Generate the deliberately handicapped last-turn control separately
python3 tools/export_dpo.py --mode synthetic --prompt-mode last_turn \
  --data data/train_seed.jsonl data/train_challenge.jsonl \
  --out runs/synthetic-last.jsonl --report runs/synthetic-last.report.json
```

`runs/` is ignored by Git. Outputs are immutable; choose a new filename for reruns. **Neither exporter pretends labels were human-reviewed.** The original default `export_dpo.py` still refuses synthetic labels unless `--mode synthetic` is requested explicitly.

Install model dependencies in an isolated environment using the compatible versions specified by your machine's PyTorch/CUDA installation and the upstream [TRL DPOTrainer documentation](https://huggingface.co/docs/trl/dpo_trainer): `torch`, `transformers`, `datasets`, `trl`, `peft`, `accelerate`. We cannot guarantee GPU memory requirements without knowing the device. Start with `Qwen/Qwen2.5-0.5B-Instruct`; choose a different supported checkpoint via `--model`. Model downloads and training run on your own machine, not on GitHub Pages.

```bash
# 4. Train full-context LoRA adapter using upstream DPOTrainer
python3 tools/train_dpo.py \
  --data runs/synthetic-full.jsonl --report runs/synthetic-full.report.json \
  --out runs/adapter-full-seed42 --model Qwen/Qwen2.5-0.5B-Instruct \
  --acknowledge-synthetic --seed 42

# Optional negative control: a fresh, separate adapter starting at the same checkpoint
python3 tools/train_dpo.py \
  --data runs/synthetic-last.jsonl --report runs/synthetic-last.report.json \
  --out runs/adapter-last-seed42 --model Qwen/Qwen2.5-0.5B-Instruct \
  --acknowledge-synthetic --seed 42

# 5. Score the unchanged base and full-context adapted model on DEV families
python3 tools/evaluate.py score \
  --data data/dev_seed.jsonl data/dev_challenge.jsonl \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter runs/adapter-full-seed42 --out runs/dev-full-scores.jsonl
python3 tools/evaluate.py summarize \
  --scores runs/dev-full-scores.jsonl --report runs/dev-full-report.json

# 6. Run the same evaluation with the last-turn adapter (separate output filename)
python3 tools/evaluate.py score \
  --data data/dev_seed.jsonl data/dev_challenge.jsonl \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter runs/adapter-last-seed42 --out runs/dev-last-scores.jsonl
python3 tools/evaluate.py summarize \
  --scores runs/dev-last-scores.jsonl --report runs/dev-last-report.json
```

Only after tuning decisions are frozen using dev, repeat scoring with `data/test_seed.jsonl data/test_challenge.jsonl`. The test sets remain public synthetic fixtures, so interpret the result conservatively.

## Interpretation of outputs

`evaluate.py` reports candidate-ranking **row accuracy** and stricter **family flip accuracy** for four modes: base/full-context, base/last-turn, trained/full-context and trained/last-turn. A last-turn-only context with identical choices cannot produce both required preference reversals. Report family counts, not just example counts. With only 8 dev families and 8 public test families, uncertainty is substantial; do not declare victory based on a few cases.

**Progress gate:** Reproducible improvement on withheld families against the frozen base, with correct context reversals and no comparable improvement for last-turn-only controls. This would support a limited learned-relational-discrimination claim. It would *not* prove emergent subjective emotion or scaling to massive LLMs.

## Scaling

The exported `prompt/chosen/rejected` schema works with many decoder-only LLM sizes supported by TRL. Each model needs its own compatible adapter training. A tiny-model improvement motivates, but does not predict, a larger-model improvement. The already retained `REVIEW.md` workflow is a separate optional pathway for external validity and is not a prerequisite for the synthetic track.
