# First VAD neural experiment (not an empathy model)

A randomly initialized **6 → 16 → 3** neural network learns a numerical
transition from two VAD vectors to a third. The network does not receive
emotion names or hardcoded emotional response rules.

- `current`: internal VAD before a step (valence, arousal, dominance)
- `observed`: VAD assigned to the incoming experience
- `next`: reviewed VAD after that experience, used as the training target
- prediction delta: `next_predicted - current`
- training loss: mean squared error between predicted and reviewed `next`

Coordinates are finite numbers from -1 to +1. The output is bounded by `tanh`.
This is supervised transition learning, not proof of emotion or subjective experience.

## Run

From the repository root on Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch
export PYTHONPATH=src
python -m affective.vad_transition train --toy
python -m affective.vad_transition predict \
  --checkpoint runs/vad-transition.pt \
  --current 0.2 0.4 -0.3 --observed 0.8 0.7 0.5
```

`--toy` trains on an artificial known weighted-average rule to verify the
training machinery. Its held-out MSE does not establish emotional learning.

## Use reviewed data instead

Provide at least 10 JSON lines with `source=human_reviewed`, e.g. a schema example:

```json
{"source":"human_reviewed","current":[0.2,0.4,-0.3],"observed":[0.8,0.7,0.5],"next":[0.6,0.6,0.2]}
```

The example is illustrative, not a validated human annotation.

```bash
PYTHONPATH=src python -m affective.vad_transition train \
  --data path/to/reviewed.jsonl --out runs/vad-reviewed.pt
```

Current evaluation randomly splits rows. Real research must split by
independent people/situations to prevent leakage, and compare to baselines.
Existing Qwen and soft-prefix experiments are not part of this neural model.
