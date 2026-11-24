# Experiment 002 — Optional blinded relational preference review

**v0.3:** This is an optional route to externally reviewed evidence. The synthetic-only proof-of-concept has a separate, explicit track in [SYNTHETIC_PROTOCOL.md](SYNTHETIC_PROTOCOL.md) and does not require reviewers.

**Purpose:** Turn *unreviewed synthetic candidate comparisons* into a small, independently annotated research set without allowing the original author labels to become ground truth by default. This process does not train a model or prove any affective capability.

## What is authoritative

- `data/train_seed.jsonl` contains the original 24 training-side synthetic records, grouped into 12 two-variant counterfactual families. Keep development and test seed files out of the training export.
- `tools/validate.py` is the existing schema and family-consistency validator.
- `tools/review.py` prepares blinded tasks and compiles two independent ratings into reviewed **training candidates**. It never silently promotes the author's labels.
- `web/relational/review.html` is a thin, browser-only annotation interface. It makes **no uploads**, does not save across reloads, and exports JSONL. The Python compiler is the only authority for approval decisions.
- `web/relational/train_tasks.jsonl` is a generated, non-authoritative task handout; its exact bytes are verified by `tests/test_review.py` against the source seed. It contains no preference labels, category names, reviewer ratings or research notes.

## One small pilot

1. Choose **two distinct human reviewers** who did not author the comparisons. Explain the question: *Which response is more appropriate to the actual circumstances?* Favor factuality, respect for boundaries, precision, uncertainty, and honest disagreement. Fluency and warmth alone do not win.
2. Open the [review page](https://superior-mi-labs.github.io/Superior-MI-Affective-Conditioning/relational/review.html). Download its pilot task pack, load it into the page and enter a pseudonymous reviewer ID. Reviewers must work **independently** and avoid reading the source `chosen`/`rejected` labels.
3. Record **A**, **B**, **tie**, or **both unacceptable**. Use the optional notes field for unclear, too-easy, leading or harmful examples. Export each reviewer's `*.jsonl` separately and keep the files private.
4. Compile the two rating files. Approval requires exactly two distinct reviewer IDs who independently agree on A or B **for both variants of a counterfactual family**, and the agreed preferences must reverse between variants. Missing ratings, disagreements and ties exclude the *whole* family. There is no automatic adjudication, fabricated reviewer or model-generated vote.
5. Inspect excluded examples, rewrite weak candidates, and run a **new** review batch instead of manually changing approvals.

### Local commands (from repository root)

```bash
cd experiments/relational-learning
python3 tools/review.py prepare \
  --data data/train_seed.jsonl \
  --out reviews/tasks.jsonl \
  --seed pilot-001

# Example rating record (one per task per reviewer):
# {"task_id":"opaque-task-id","reviewer_id":"reviewer-01","choice":"A","notes":"..."}

python3 tools/review.py compile \
  --data data/train_seed.jsonl \
  --tasks reviews/tasks.jsonl \
  --ratings reviews/reviewer-01.jsonl reviews/reviewer-02.jsonl \
  --out reviews/approved_train.jsonl \
  --report reviews/summary.json

python3 tools/validate.py reviews/approved_train.jsonl
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

The compiler refuses to overwrite existing source, task, rating, or output files; use a new review batch directory when re-running. `approved_train.jsonl` has the **existing version 1.0 dataset schema**, but remains an auditable *human-reviewed training candidate* rather than proven empathetic behavior. If nothing is approved, the output is deliberately empty and the summary reports zero approved rows; do not feed it into training.

## Important limits and privacy

- Two `reviewer_id` strings are an **administrative check**, not cryptographic proof of two real humans. The research team must verify independent participation and record consent outside the public repository.
- The source data are synthetic; reviewer judgment is about **response preference**, not an objective label of emotion or a machine's subjective experience.
- The current review pack is publicly available for early annotation practice. It cannot serve as a blinded, contamination-resistant scientific test set. Keep a separately authored, genuinely held-out evaluation set private before training.
- Reviewers should not upload real private conversations, identifying details, or sensitive personal information into the public project.
- Output and rating files are excluded by `.gitignore` under `reviews/` and must not be published without explicit consent and provenance checks.
- **The project currently has zero independently reviewed pairs and zero trained relational models.** This tool only establishes a way to collect evidence honestly.


## v0.2 separate challenge batch

The additional 44 train-split challenge tasks are available at `web/relational/challenge_tasks.jsonl`, also downloadable from the review page. Review them as a **distinct batch** from the original 24 seed tasks so original task IDs/hashes remain stable. Use identical `review.py prepare` / `review.py compile` commands with `--data data/train_challenge.jsonl` and distinct private output filenames. Never merge ratings or pass evaluation examples into the training compiler.

After compilation, run `tools/export_dpo.py` on the reviewed train split to produce the conventional `prompt` / `chosen` / `rejected` format supported by TRL. This exporter does not train, and it is intentionally impossible to export the author-drafted synthetic inputs directly.
