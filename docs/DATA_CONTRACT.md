# Versioned data contract: `schema_version: 1`

A training record has an explicit message, authored target response, observed affect and internal post-transition affect. It does not require retokenizing or replacing any language-model vocabulary entries.

```json
{
  "schema_version": 1,
  "mode": "vector",
  "user": "I finally finished the project!",
  "assistant": "That's great news. What was the hardest part?",
  "initial_ai_vad": [0.2, 0.1, 0.0],
  "observed_vad": [0.8, 0.8, 0.2],
  "state_after_vad": [0.4, 0.3, 0.1],
  "phase_before": 0.0,
  "phase_after": 0.0,
  "evolve": true,
  "approved": true,
  "provenance": "human-authored"
}
```

The values above are illustrative, not empirical labels. Browser export records preserve both the before and after states, but Python training only consumes `state_after_vad`, `observed_vad`, `phase_after`, `mode`, `user`, and `assistant`. The optional fields aid provenance, debugging and future longitudinal tests. A neutral feature does not imply a neutral human emotion.

## Invariants

- `user` and `assistant`: nonempty strings. `assistant` is an explicitly authored and approved target, **never** the automatic browser template.
- `state_after_vad` and `observed_vad`: exactly three finite numeric coordinates within `[-1,1]`.
- `mode`: `vector` or `phase`.
- `phase_after`: finite radians. Only used in phase mode; vector mode encodes `(sin,cos)` as `(0,0)`.
- `schema_version`: integer `1`. Unsupported versions must fail.
- `approved: true` is required for ordinary training. Explicit `--allow-synthetic` exists solely for development tests.
- The dataset includes no browser-origin encryption, signature, author authentication, or automated psychological ground truth; the researcher must review examples before using them.

The browser records one state transition at a time; Python does **not** independently recalculate it. For sequential experiments, keep a conversation/session identifier and preserve turn order when preparing the dataset. R0's minimal trainer optimizes individual examples, not full differentiable recurrent state trajectories.

## Training hypotheses

A training record with a new affect configuration but exactly the same target wording as all other configurations provides very little evidence for learning state-dependent behavior. Prefer paired, context-controlled examples with intentionally different and defensible target responses, plus held-out cases. Label provenance and source licenses must be reviewed before redistribution.
