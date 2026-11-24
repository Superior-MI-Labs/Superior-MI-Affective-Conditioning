# Superior MI Labs — Relational Learning Experiment v0.3

[**Research website**](https://superior-mi-labs.github.io/Superior-MI-Affective-Conditioning/relational/) · [**Repository**](https://github.com/Superior-MI-Labs/Superior-MI-Affective-Conditioning)

**Scope:** One approach only: relational learning in the weights of an existing language model. No VAD vectors, emotional state controller, personality prompt, separate learned state, or new runtime framework.

## Main hypothesis

A language model trained on context-sensitive response comparisons will improve its ability to distinguish appropriate responses to **the same utterance under different circumstances**, including when an appropriate answer is disagreement, uncertainty, or respectful distance rather than automatic warmth.

This is a study of **learned relational behavior**. It is not evidence of subjective feelings, independently emerging emotions, or machine consciousness.

## Folder contents

- `OBJECTIVE.md`: operational hypothesis, learning objective, implementation decisions.
- [`REVIEW.md`](REVIEW.md): blinded two-rater annotation protocol, CLI, data provenance and approval gates.
- `tools/review.py`: offline task preparation and conservative reviewed-data export.
- `web/relational/review.html`: optional browser-only labeling interface; separate task packs generated for seed and challenge training candidates.
- `DATASET.md`: scenario protocol, annotation, leakage controls.
- `CHALLENGES.md`: 32 authored counterfactual challenge families, including 12 new v0.3 cases; all are unreviewed synthetic scenarios.
- `SCALING.md`: one scale-neutral dataset contract, multiple model sizes, reproducible evaluation.
- `source/challenges.json`: human-editable source of truth for the challenge batch.
- `tools/generate_challenges.py`: deterministic source-to-JSONL generator.
- `tools/export_dpo.py`: one conversational DPO exporter with a reviewed-only default and an explicit `--mode synthetic` research path.
- `tools/train_dpo.py`: optional upstream TRL/LoRA training launcher for a supported LLM checkpoint.
- `tools/evaluate.py`: frozen-base versus adapted-model context-reversal scorer; no empathy claims.
- [`SYNTHETIC_PROTOCOL.md`](SYNTHETIC_PROTOCOL.md): runnable commands, null controls, hypothesis and claim boundaries.
- `EVALUATION.md`: preregisterable evaluation criteria and failure conditions.
- `schema.json`: version 1.0 structural schema for one preference pair.
- `data/train_seed.jsonl`, `dev_seed.jsonl`, `test_seed.jsonl`: 36 **synthetic, unreviewed** seed comparisons across 18 families; `data/*_challenge.jsonl` adds 64 **unreviewed** comparisons across 32 new families. **Total: 100 comparisons, 50 families.** These can be used in explicitly labeled *synthetic proof-of-concept* training, but they are **not validated human preference data or a benchmark for human empathy**.
- `tools/validate.py`: standard-library-only dataset validator (schema shape, duplicate IDs, cross-split family leakage, counterfactual consistency and response reversals, source flags).
- `tests/test_dataset.py`: validator regression tests.
- `diagrams/training_path.mmd`: source of editable flow diagram.

## Validate

```bash
python3 tools/generate_challenges.py --check
python3 tools/validate.py data/*.jsonl
python3 -m unittest discover -s tests -v
# Run the actual LoRA pilot with the commands in SYNTHETIC_PROTOCOL.md
```

## Explicit experimental restrictions

1. Do not treat the seed or challenge pairs as human preferences. All 100 records are synthetic and unreviewed.
2. Do not train on the seed evaluation splits and claim scientific generalization.
3. Do not use `review_notes`, category names, preference labels, or other analysis metadata as input to the model.
4. Do not collect real conversations without participant permission, privacy safeguards, and appropriate use rights.
5. Do not publish claims of intrinsic emotions or empathy from preference-training alone.

**Next implementation milestone:** Run the **synthetic-only controlled DPO pilot** according to [SYNTHETIC_PROTOCOL.md](SYNTHETIC_PROTOCOL.md). It does not require human review. The separate [blind review pilot](REVIEW.md) remains optional and is for later external-validity checks. No LLM weights have been trained in this repository yet; measured gains must be reported with their synthetic provenance.
