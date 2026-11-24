# Relational training objective — design lock v0.1

## Research question

Does learned preference for contextually appropriate responses generalize across unseen relational circumstances, with no mood or emotion instructions at inference?

### Operational definition of relational fit

A response accurately uses preceding circumstances and available evidence, respects stated agency and boundaries, recognizes changes in meaning and social context, and remains honest even where agreement or sentimental language is tempting.

This is behavioral, not phenomenological. We cannot infer felt love, fear, anger, or empathy from generated text.

## Two implementation paths considered

1. **Custom contrastive embedding head:** Embed the situation and candidate responses and optimize a similarity margin. This explicitly manipulates representations, but adds a head/pooling choice and makes it harder to connect success to generated language.
2. **Direct Preference Optimization (DPO): chosen for v0.1.** Fine-tune one language model from paired context-response preferences, keeping a frozen copy as a reference. Learned behavior resides in the model weights; no emotion head or runtime mood controller is added. A standard algorithm is easier to audit and reproduce.

The scientific novelty is not DPO. It lies in the **factorially structured, counterfactual relational data and evaluations**.

## One loss

For context x, preferred response y+, dispreferred response y-, trainable policy πθ, frozen reference πref, and β>0:

```
D+ = log πθ(y+ | x) - log πref(y+ | x)
D- = log πθ(y- | x) - log πref(y- | x)
L_DPO = -log sigmoid( β · (D+ - D-) )
```

The log probabilities sum only *response continuation* token log-probabilities, not prompt tokens. Use the same fixed tokenizer/chat template for trainable and reference models. Reference checkpoint is identical to the policy initialization. Do not silently change β, padding masks, label masks, or truncation rules. Follow a tested DPO implementation, rather than improvising a custom trainer.

A pretrained instruction/chat model is recommended for the pilot; no extra SFT phase is assumed. If the model cannot produce coherent dialogue, fix backbone choice before interpreting relational gains. Fine-tuning still changes learned weights; it is not prompt engineering.

## Objective guardrails

- Preferred/rejected candidate lengths, grammar and specificity should be reasonably matched; the dispreferred answer must often be fluent and plausible.
- Reject easy negatives that are just rude or incoherent. Balance cases where the chosen answer is firm, direct, brief, warm, cautious, or corrective.
- The selected answer should be decided by **situation evidence**, not a label about what emotion to express.
- Preserve uncertainty, factual accuracy, and boundaries. A model that just agrees more is not relationally stronger.
- Exclude `review_notes`, `category`, `variant`, and labels from prompt serialization.
- Record β, seed, dataset hash, reference checkpoint/hash, tokenizer revision, LoRA settings, batch tokens, precision, and effective learning rate.

## Falsifiable prediction

For blind human evaluation of context-matched response pairs from unseen scenario families, the trained checkpoint should show greater relational fit than its own **frozen pre-training checkpoint**. This is a reference comparison, not a second trained architecture. Improvement must not depend solely on more positive sentiment, greater verbosity, or exaggerated agreement.

The stronger test asks whether response *preferences reverse between the exact same two candidate responses* when **only the antecedent context changes** and the final utterance remains identical. If trained outputs do not respond correctly to these reversals, the hypothesis has not passed its first test.

## Limits

DPO optimizes observed preference rankings, not human biology or direct experience. Novel test performance may indicate transfer/generalization, but not spontaneous emotions. Without memory or online training, weights do not absorb new post-deployment life events across conversations. This first experiment explicitly excludes these capabilities.

### Selected references

- Rafailov et al., 2023, Direct Preference Optimization, https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract.html
- Gao et al., 2021, SimCSE, https://aclanthology.org/2021.emnlp-main.552/
- Rashkin et al., 2019, EmpatheticDialogues, https://aclanthology.org/P19-1534/
- Sharma et al., 2023, Towards Understanding Sycophancy, https://arxiv.org/abs/2310.13548
