# Superior MI Labs — Relational Learning Experiment v0.2

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
- `CHALLENGES.md`: 20 additional author-drafted harder counterfactual families, unreviewed.
- `SCALING.md`: one scale-neutral dataset contract, multiple model sizes, reproducible evaluation.
- `source/challenges.json`: human-editable source of truth for the challenge batch.
- `tools/generate_challenges.py`: deterministic source-to-JSONL generator.
- `tools/export_dpo.py`: reviewed-only export to conversational TRL DPO format; refuses all synthetic labels.
- `EVALUATION.md`: preregisterable evaluation criteria and failure conditions.
- `schema.json`: version 1.0 structural schema for one preference pair.
- `data/train_seed.jsonl`, `dev_seed.jsonl`, `test_seed.jsonl`: 36 **synthetic, unreviewed** seed comparisons across 18 families; `data/*_challenge.jsonl` adds 40 **unreviewed** comparisons across 20 new families. **Total: 76 comparisons, 38 families.** They are smoke-test fixtures, **not validated training or benchmark data**.
- `tools/validate.py`: standard-library-only dataset validator (schema shape, duplicate IDs, cross-split family leakage, counterfactual consistency and response reversals, source flags).
- `tests/test_dataset.py`: validator regression tests.
- `diagrams/training_path.mmd`: source of editable flow diagram.

## Validate

```bash
python3 tools/generate_challenges.py --check
python3 tools/validate.py data/*.jsonl
python3 -m unittest discover -s tests -v
```

## Explicit experimental restrictions

1. Do not treat the seed or challenge pairs as human preferences. All 76 records are synthetic and unreviewed.
2. Do not train on the seed evaluation splits and claim scientific generalization.
3. Do not use `review_notes`, category names, preference labels, or other analysis metadata as input to the model.
4. Do not collect real conversations without participant permission, privacy safeguards, and appropriate use rights.
5. Do not publish claims of intrinsic emotions or empathy from preference-training alone.

**Next implementation milestone:** Collect independent human judgments through the [blind review pilot](REVIEW.md). The model training stage remains blocked until comparisons have been reviewed and an uncontaminated evaluation set has been frozen. Then select a pretrained base checkpoint and run one controlled DPO pilot.
