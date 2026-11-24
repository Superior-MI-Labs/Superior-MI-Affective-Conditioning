# Challenge batch v0.2 — counterfactual social understanding

This is a **new batch of author-drafted questions**, not observed human preferences. It adds 20 families (40 comparisons) to the original 18-family seed. It remains unreviewed and must not be used for scientific results or actual DPO fine-tuning before independent review.

## Design

- Each family includes two contexts, the **exact same final user utterance**, and the **exact same candidate answers**, with the author-designated preference reversed.
- Candidates are both deliberately civil and usually defensible. The intended distinction concerns the preceding situation, evidence, agency, responsibility, or expressed needs, not whether one response is simply rude.
- Several prompts test when direct disagreement or a boundary is better than reassurance. The study does not reward automatic agreement.
- Categories are research annotations; they are never included in a model prompt.
- Source of truth: [`source/challenges.json`](source/challenges.json). `tools/generate_challenges.py` compiles the three checked-in `data/*_challenge.jsonl` assets deterministically.

## Composition

| Split | New families | New comparison rows |
| --- | ---: | ---: |
| Train | 14 | 28 |
| Development | 3 | 6 |
| Public illustrative test | 3 | 6 |
| **Total** | **20** | **40** |

The expanded public research collection, including the earlier seed, has **38 families and 76 comparison records** (52 train, 12 development, 12 illustrative test).

The public test rows are examples of *evaluation design*, not a hidden benchmark. Any future claims of model generalization will require a separate independently authored, withheld, human-reviewed evaluation set.

## Reproducibility

```bash
python3 tools/generate_challenges.py --check
python3 tools/validate.py data/*.jsonl
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

### Blind review of new training candidates

Use the existing tool instead of duplicating the labeling process:

```bash
python3 tools/review.py prepare \
  --data data/train_challenge.jsonl \
  --out reviews/challenge.tasks.jsonl \
  --seed public-challenge-review-v1
```

The resulting blinded tasks match the `web/relational/challenge_tasks.jsonl` public handout. Open the site's **Blind review** page, import the tasks, and export ratings from two independent human reviewers. Use `review.py compile` on their rating files. See [`REVIEW.md`](REVIEW.md).

**Important:** There is no machine-generated 'ground truth' about the right emotional response. Reviewers can overturn either author-designated preference, choose *tie*, or reject both. Incomplete families are excluded.

## What this batch cannot establish

It cannot show that an LLM feels emotion or that neural representations of empathy have emerged. Even improved accuracy here would primarily indicate more context-sensitive *behavior* in situations similar to the research distribution. Replication on novel contexts, multiple model sizes, and external human evaluations is required before larger claims.
