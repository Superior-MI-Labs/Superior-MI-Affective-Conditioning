# Superior MI Labs — Experimental AI Research

**[Open the Relational Learning research page](https://superior-mi-labs.github.io/Superior-MI-Affective-Conditioning/relational/)** · [Experiment files](https://github.com/Superior-MI-Labs/Superior-MI-Affective-Conditioning/tree/main/experiments/relational-learning) · [Blind review pilot](https://superior-mi-labs.github.io/Superior-MI-Affective-Conditioning/relational/review.html) · [Earlier VAD numerical workbench](https://superior-mi-labs.github.io/Superior-MI-Affective-Conditioning/)

## Active direction: Experiment 002 — Relational Learning

Our current hypothesis is that a language model can learn **context-sensitive relational behavior** through comparisons of situations, responses and consequences, without emotion prompts, a VAD controller or a scripted personality.

The starting design uses *counterfactual response preference pairs*: the same final message appears under different preceding circumstances, with the preferred response reversed. We propose evaluating a fine-tuned language model against the unchanged base checkpoint using novel situation families, factuality tests and blind human assessments.

**Research status:** Experiment design and 100 synthetic, **unreviewed** comparisons across 50 scenario families, including 24 new context-sensitive records. A synthetic-only DPO/LoRA training and base-versus-adapter evaluation path is now runnable without human annotation. **No relational language model has yet been trained or evaluated.** The seed files are smoke-test fixtures, not validated preference judgments. Improved social responses, if observed later, would not establish felt emotions, subjective empathy or consciousness.

### Reproduce the data checks

```bash
python3 experiments/relational-learning/tools/generate_challenges.py --check
python3 experiments/relational-learning/tools/validate.py experiments/relational-learning/data/*.jsonl
python3 -m unittest discover -s experiments/relational-learning/tests -p 'test_*.py' -v
```

Read the [synthetic-only experiment protocol](experiments/relational-learning/SYNTHETIC_PROTOCOL.md), [scaling plan](experiments/relational-learning/SCALING.md), [new challenge batch](experiments/relational-learning/CHALLENGES.md), [research hypothesis and DPO objective](experiments/relational-learning/OBJECTIVE.md), [dataset protocol](experiments/relational-learning/DATASET.md), [evaluation plan](experiments/relational-learning/EVALUATION.md), and [two-rater review guide](experiments/relational-learning/REVIEW.md). The browser review tool runs locally and does not upload ratings. A compiler requires two independent agreeing judgments per variant, rejects disputes and incomplete families, and forbids exporting the test split for training. No examples have been human reviewed yet.

The next milestone is to execute the synthetic-only model pilot, compare the frozen base and adapted checkpoint on held-out scenario families, and test the last-message-only control. Human review remains an optional external-validation pathway, not a prerequisite for this mechanical experiment. Improvements at small scale do not guarantee that they carry to larger LLMs. We deliberately are not building a separate mood-state system or custom AI framework for this experiment.

---

## Experiment 001 — VAD Numerical Workbench

An original, minimal **6 → 16 → 3** neural network trained from scratch to predict a next three-dimensional state from a current and an observed VAD vector.

**Research status:** Numerical proof of concept. The currently deployed model learned a deliberately artificial rule; it is **not** an empathy model, an emotion detector, or evidence of machine feelings. No pretrained language model, chatbot, selected dynamics mode, or prompt-based emotional conditioning is involved.

### The model

```text
Current VAD [v, a, d] ──┐
                       ├── 6 inputs ── 16 tanh neurons ── 3 tanh outputs ── Next VAD
Observed VAD [v, a, d] ┘                                        │
                                                                └── delta = next - current
```

- **V**alence: unpleasant (-1) to pleasant (+1)
- **A**rousal: low activation (-1) to high activation (+1)
- **D**ominance: low perceived control (-1) to high control (+1)

All coordinates are normalized to `[-1, 1]`. These are **input annotations**, not automatically inferred emotions. The network learns weights using mean squared error against explicitly supplied next-state targets.

### Try the web lab

The GitHub Pages website displays both input vectors, plots them with the model's prediction in a 3D VAD projection, and shows the calculated state delta. **It runs the actual weights exported from the PyTorch model**, using a small read-only JavaScript forward pass. It does not secretly run a second trainer or send prompts to another model.

Open the repository's Pages deployment after it's enabled. For local preview:

```bash
python3 -m http.server 8000 --directory web
# Open http://127.0.0.1:8000
```

### Train our network

Python 3 and PyTorch are the only model dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch

# Artificial 1,000-example dataset; validates that learning works.
PYTHONPATH=src python -m vad.model train --toy --web-out web/model.json

# Predict with the trained model.
PYTHONPATH=src python -m vad.model predict \
  --checkpoint runs/vad.pt \
  --current 0.2 0.4 -0.3 --observed 0.8 0.7 0.5
```

This is intentionally simple. The toy targets are calculated using `next = 0.65 × current + 0.35 × observed`; the training code does **not** implement that formula inside the neural network. It learns an approximation from examples. The current held-out test MSE of approximately `0.00195` measures reproduction of that rule, **not empathy or emotion**.

### Collect and train reviewed examples

The web lab includes a manual target annotation panel. Adjust the current and observed VAD vectors, move all three target controls, click **Add reviewed example**, then **Export JSONL** when you have collected enough examples. All annotations remain in browser memory until export; nothing is uploaded.

```bash
PYTHONPATH=src python -m vad.model train \
  --data vad-reviewed.jsonl \
  --out runs/reviewed.pt \
  --web-out web/model.json
```

The input format is one JSON object per line:

```json
{"source":"human_reviewed","current":[0.2,0.4,-0.3],"observed":[0.8,0.7,0.5],"next":[0.6,0.6,0.2]}
```

At least 10 examples are required for this initial numerical experiment. That minimum is a software requirement, **not** enough data to establish an empirical psychology model. For serious experiments, use independent situations, multiple annotators, held-out situation-level evaluation, and comparison baselines. The current random 80/20 split is for smoke testing only; correlated examples can leak across it.

### Tests

```bash
node --test tests/*.test.mjs
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

Model tests require PyTorch and run locally. GitHub Pages CI checks the exported model's numerical behavior and Python syntax without downloading PyTorch. Every deployment uses one Pages workflow.

### Repository layout

```text
src/vad/model.py          Single Python trainer, predictor, and weight exporter
web/model.json            Exported weights (currently synthetic toy model)
web/model.mjs             Read-only forward computation for the browser
web/app.mjs               VAD controls, 3D projection, example collection
web/index.html            Research workbench
web/styles.css            Workbench presentation
tests/                    Model contracts and numerical parity
docs/ARCHITECTURE.md      Hypothesis, data contract, next research milestones
```

Previous Qwen/affective-conditioning prototype code has been retired from the **current branch**. Git history preserves it if it is ever needed for reference.

Apache-2.0 licensed (see `LICENSE`).
