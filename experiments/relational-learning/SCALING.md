# Scaling plan — same data contracts, different model sizes

## What is portable

The reviewed dataset contains **context and preferred/rejected completions**, not architecture-specific embeddings. Once validated by `review.py compile` and transformed by `tools/export_dpo.py`, it is compatible with conventional conversational DPO training. This applies to different decoder-only language-model sizes **provided the chosen model and trainer support the format**. The JSONL is an interface, not a guarantee of compute feasibility or learned outcomes.

An adapter trained for a small checkpoint is **not** portable to an unrelated large checkpoint. Each size requires its own baseline, fine-tuning run, tokenizer/chat template, adapter or full-weight update, and evaluation.

## Progression

1. **Local pilot, approximately 0.5–1.5B parameters.** Establish reproducible baseline metrics and ensure nonzero learning signal on *human-reviewed* pairs. Prefer LoRA on a pretrained chat model with verified license and training access.
2. **Intermediate comparison, approximately 3–8B.** Train on the same reviewed data and strict split protocol, record compute and variance. Inspect whether the large base already succeeds without training.
3. **Large model, 30B+ if justified.** Use suitable multi-GPU infrastructure or parameter-efficient / quantized approaches. Re-run evaluation with comparable decoding; do not extrapolate from small-model results.

These sizes are illustrative experimental tiers, not specific GPU memory promises. Activation length, optimizer states, precision, model architecture, reference-policy handling, and batch size materially change resource requirements.

## Data path

```text
Authored comparison candidates (never training truth)
       -> blind human annotations (two independent raters)
       -> compiled reviewed pairs (full families only)
       -> tools/export_dpo.py (TRL conversational format)
       -> choose model + tokenizer + chat template
       -> DPO / LoRA (standard library, not custom trainer)
       -> per-family held-out testing / blind generation review
```

The exported JSONL uses only `prompt`, `chosen`, `rejected`, each as standard chat-message lists. Analysis notes, preference metadata, and reviewer information stay outside model input. **No system prompt telling the model to be empathetic is added.**

## Export after review

```bash
# All paths shown are local; a prepared review batch and two independent
# rating files must exist before these commands can produce any approved data.
python3 tools/review.py compile \
  --data data/train_challenge.jsonl \
  --tasks reviews/challenge.tasks.jsonl \
  --ratings reviews/reviewer-a.jsonl reviews/reviewer-b.jsonl \
  --out reviews/challenge.approved.jsonl \
  --report reviews/challenge.review-report.json

python3 tools/export_dpo.py \
  --data reviews/challenge.approved.jsonl \
  --out reviews/challenge.dpo.jsonl \
  --report reviews/challenge.dpo-report.json
```

`export_dpo.py` refuses synthetic or disputed examples, any non-train split, incomplete families, source/output overwrite, and data without approved provenance. The *reviewer independence claim* is not verifiable by software; the lab must verify recruitment and review procedures separately.

## Evaluation design and failure criteria

- Use one frozen original checkpoint per size as the baseline. Compare unchanged checkpoint with adapted checkpoint using *identical prompts, decoding rules, and evaluation cases*.
- Freeze a genuinely private, human-authored evaluation corpus before the first training run. Public test fixtures are not independent evidence.
- Report both pairwise counterfactual preference and blind ratings of actual generated conversations. Good DPO loss alone is **not** a success criterion.
- Measure preference **flips** under changed prior context while the last user utterance and answer options stay identical. Report by family, not just by row.
- Add straightforward style baselines, anti-sycophancy tests, factuality regression, and qualitative failure review. Avoid judging only 'warmth'.
- Repeat with multiple seeds and across model sizes only after the first pilot shows a credible and externally reviewed signal.
- Stop if the gain disappears under genuinely new scenario families or is explained by length, sentiment, or generic positivity.

## Implementation discipline

Use an existing tested DPO trainer such as Hugging Face TRL rather than writing a parallel learning engine. LoRA / QLoRA are optimization choices, not new emotional architectures. Keep one versioned dataset schema and one validation/review/export chain. Freeze model revision, tokenizer revision, precision, beta, random seed, data hashes, chat template, and generation parameters for each run.

References:
- TRL DPO dataset formats: https://huggingface.co/docs/trl/dpo_trainer
- TRL PEFT/LoRA integration: https://huggingface.co/docs/trl/peft_integration
- Direct Preference Optimization: https://arxiv.org/abs/2305.18290

**Status:** This repository now prepares data for scale-comparable experiments; it does not yet contain human-reviewed challenge pairs or trained relational checkpoints.
