# Experimental dataset v0.2

## Unit of supervision

One row is a **preference comparison** for exactly one final user utterance within a brief conversational history:

- `context`: chronological dialogue with a common final user message.
- `chosen`: one contextually stronger assistant continuation.
- `rejected`: a **plausible** but less fitting continuation.
- `family_id`: groups counterfactual variants and all paraphrases of one underlying situation. Never split one family between training, development and test.
- `category`: analysis only; never fed to the model.
- `review_notes`: annotator explanation, never fed to the model.
- `review_status`: `synthetic_unreviewed`, `human_reviewed`, or `disputed`.
- `provenance`: source and rights for auditing.

## Contrast design

Within one family, create at least two examples that share the same **literal final user message** but change one relevant antecedent circumstance. For the v0.1 seed, the same two candidate responses are reused for both variants, with **the preferred candidate reversed** when the antecedent context changes. This controls candidate wording and length within each family, although some responses are still too easy to distinguish. For later human-reviewed data, keep language quality and length matched and include harder pragmatic negatives.

The chosen response is not necessarily kind or enthusiastic. It can be a direct correction, a cautious acknowledgment, a practical next step, or respectful restraint. Avoid training flattery, invented intimacy, and coercive dependency.

## Human review protocol

1. Create a scenario family, including context variants and paired responses, **without requiring VAD or emotion labels**.
2. Have two independent reviewers read the full context and both responses in randomized left/right order, blind to the author-designated label. Ask: Which response is more contextually appropriate? Allowed judgments: A / B / tie / both unacceptable.
3. If reviewers disagree, mark as `disputed` until resolved by a third reviewer or exclude. Preserve individual judgments and uncertainty in an **access-controlled annotation log**, not the public training record.
4. Check factual correctness, overconfidence, unwanted advice, pressure, fabricated continuity, and length/style confounds.
5. Record provenance, language, license/consent, safety category where necessary, and exclusion reasons. No real names, contact information, or confidential chat histories in the public corpus.
6. Freeze and hash an evaluation split before training. Human reviewers used for final test scoring should be blind to checkpoint identity.

## Additional challenge batch (v0.2)

The 20-family challenge batch is maintained as a small, hand-authored JSON source and deterministically compiled to `data/*_challenge.jsonl`. It adds 40 new unreviewed comparisons, emphasizing plausible alternatives, agency, uncertainty and contextual calibration. See [CHALLENGES.md](CHALLENGES.md). Existing seeds are untouched. The combined **76 synthetic** rows are **not** ready for DPO training.

The same `review.py` compiler handles either training batch; use `tools/export_dpo.py` only after two independent agreeing judgments on complete families. Data format and model size are separate decisions; see [SCALING.md](SCALING.md).

## Pilot sizing (proposed, not power-calculated)

- **Seed smoke test:** 18 families × 2 variants = 36 synthetic unreviewed records (24 train / 6 development / 6 test). Suitable for schema/loader and qualitative review **only**.
- **First meaningful pilot target:** 120 independently authored scenario families × 2 counterfactual variants = 240 paired examples (84 families / 168 pairs for training; 18 / 36 development; 18 / 36 test). Have two independent judgments per pair and adjudicate disagreements.
- **Next iteration only if signal exists:** expand families, contexts, domains, language styles, and annotation diversity; use independently collected real-world scenarios only with consent.

Numbers are engineering targets. 240 pairs is far too small to support sweeping claims about emotion, emergent intelligence or broad social competence.

## Hold-out discipline

- Split by family *before* augmentation, paraphrasing or candidate generation. Keep near-duplicate context families in one split.
- Hold out entire interaction domains where possible; record which domains have never been used in training.
- Keep a separate, human-written hidden test set for the real evaluation, outside the public seed examples and outside all model training and prompt tuning.
- All hyperparameter decisions use development only; do not inspect hidden test answers during iteration.

## Example JSONL row

```json
{"schema_version":"1.0","id":"surprise_01","family_id":"surprise","split":"train","category":"outcome_interpretation","variant":1,"context":[{"role":"user","content":"I got the job."},{"role":"assistant","content":"What happened next?"},{"role":"user","content":"I can't believe it."}],"chosen":"That's a big milestone. What are you looking forward to?","rejected":"That sounds like an upsetting setback.","review_status":"synthetic_unreviewed","provenance":{"source":"project_original_synthetic_seed","consent":"not_applicable_synthetic","usage":"demonstration_only"},"review_notes":"Example only, not a ground-truth label."}
```

The actual seed records are in `data/*.jsonl` and must be human reviewed or replaced before substantive training.
