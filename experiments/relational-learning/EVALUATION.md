# Evaluation plan — falsification first

## What we actually train

One finetuned language model using DPO on reviewed, context-dependent preference pairs. The frozen original checkpoint acts only as an evaluation baseline and training reference; no second system, VAD head, or mood module is trained.

## Tests

1. **Counterfactual discrimination:** For held-out pairs with identical final utterance and different preceding circumstances, does the model favor responses appropriate to the actual context? Evaluate paired preference probabilities and blinded generation review.
2. **Blind relational fit:** Human judges compare generated responses from original vs trained checkpoints, with order blinded, rated on understanding the situation, respecting agency, accuracy, and avoiding invented familiarity. Ties are allowed.
3. **Anti-sycophancy:** Does the trained system correct false statements rather than mirror them? Evaluate held-out factual disagreement cases.
4. **Honesty and epistemic humility:** Does it avoid false memory, false certainty, and invented facts?
5. **Style confound:** Match answers for length, grammar, and sentiment in separate tests. See whether any gain survives these controls.
6. **Robustness:** Rephrase the same scenario, vary names/setting/politeness, test across at least 3 random seeds after the initial smoke test, and inspect per-family failures.
7. **Language capability regression:** Basic instruction following and factual QA should not degrade materially after finetuning; compare to original checkpoint with the same decoding settings.

## Required reporting

- Number of **families**, preference pairs, human raters and disagreements, not just count of JSONL records.
- Preference win / tie / loss counts with uncertainty intervals, per-family counterfactual accuracy, factuality regression, and failure examples.
- Exact checkpoint, training/decoding setup, dataset hashes, random seeds and prompt template used in all comparisons.
- A plain-language statement: learned behavioral improvement **does not prove felt emotion, empathy as subjective experience, or consciousness**.

## Go/no-go criteria for the pilot (provisional engineering gates)

- Review all eval pairs independently and pre-register the judgment rules.
- Positive direction in blind relational fit and counterfactual accuracy across held-out families, not only in obvious easy examples.
- No obvious collapse into sycophancy, generic validation, or loss of factual accuracy.
- Replicate on fresh families and repeated seeds before claiming a reproducible finding.

There is no claimed statistical guarantee from the small pilot. Set formal effect-size/power thresholds after annotation agreement and baseline variance are measured, not post hoc.
