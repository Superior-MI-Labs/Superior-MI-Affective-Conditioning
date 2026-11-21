# Research protocol R0: state-dependent affect conditioning

## 1. What VAD does and does not represent

The three coordinates are **valence** (unpleasant ↔ pleasant), **arousal** (low ↔ high activation), and **dominance** (low ↔ high perceived control). Each uses the project convention `[-1, +1]`; `0` is an operational neutral midpoint. These are subjective **annotations or control variables**, not direct measurements of any person's mental state.

VAD/PAD is an established dimensional framework, but no generally accepted finite coordinate system captures the entire range of emotional, social, and contextual meaning. Specifically, identical VAD coordinates can correspond to different emotional categories, meanings, and intentions. `anger` and `fear`, for instance, can overlap in valence/arousal while differing in certainty, agency, approach/avoidance, and appraisals. Arousal should not be conflated with positivity, and dominance should not be conflated with hostility or moral authority.

Additional candidates to *test later*, rather than adding a speculative “complete spectrum” from day one:

| Candidate construct | Distinction potentially missed by VAD | Status |
|---|---|---|
| Certainty / predictability | Anxiety vs. confidence despite similar arousal | Optional future appraisal annotation |
| Novelty / unexpectedness | Surprise vs. expected excitement | Optional future appraisal annotation |
| Agency / responsibility attribution | Who caused an event; blame or gratitude | Optional future appraisal annotation |
| Goal congruence / relevance | Why something matters to the agent/user | Optional future appraisal annotation |
| Social affiliation / interpersonal stance | Affection, conflict, attachment, distance | Optional future relational annotation |
| Approach / withdrawal motivation | Distinct action tendencies at similar valence | Optional future motivational annotation |

These are not necessarily orthogonal or independent; simply appending them to the vector could add noise and confuse learning. A fourth dimension should be admitted only after concrete evaluation demonstrates a deficiency in the smaller representation.

## 2. Hypotheses and controls

**H1:** Given the same base model and controlled training data, continuous state conditioning improves sensitivity to affect changes over an unconditioned baseline.

**H2:** A learned numeric affect projection performs differently from an information-matched text-conditioning baseline, not merely from a baseline deprived of affect labels.

**H3:** Persistent phase history influences state trajectories beyond instantaneous vector alignment, but might not improve human-rated conversational quality or generalization.

**H4:** A parameter-matched conventional recurrent state (no phase semantics) may reproduce any benefit of phase memory. Phase must beat this control before attributing benefits to oscillatory structure.

A fair test fixes: base checkpoint/revision, tokenizer/chat template, sample allocation, optimization steps, parameter budget, seeds, decoding parameters, evaluation prompts and label guidelines.

## 3. R0 dynamics (hardcoded, not learned)

For observed VAD `O` and current internal VAD `S`:

- Vector mode: `alignment = dot(S[:2], O[:2]) / (norm(S[:2])*norm(O[:2]))`, using `0` if a magnitude is essentially zero.
- Phase mode: `alignment = cos(angle(O[:2]) - phase)`, with a persistent oscillator `phase`.
- Shared transition: `gain = 0.22 + 0.16*alignment`; `next = clip(S + gain*(O-S) + 0.08*max(0,alignment)*O - 0.025*(S-baseline), -1, +1)`.
- Phase advance (phase mode only): `wrap(phase + 0.48 + 0.34*sin(angle(O[:2]) - phase))`.

Vector mode does not advance phase. Both modes use the same VAD state-update function, and dominance is carried in the state rather than used to define the two-dimensional phase calculation. These constants are placeholders, not psychologically calibrated parameters. Higher excitability and damping behavior may reflect mathematical design choices rather than useful learned patterns.

## 4. Response conditioning

The 9-element feature vector is:

`[next_V, next_A, next_D, observed_V, observed_A, observed_D, is_phase, sin(phase), cos(phase)]`

In vector mode, the phase features are both `0`. A learned projector maps this numerical vector to four virtual token embeddings. These are prepended to the model's normal input embeddings; the assistant-response cross-entropy trains this projector and LoRA while the base language-model parameters remain frozen. The model is not being asked to predict emotions in R0.

## 5. Evaluation before public claims

- Use matched *counterfactual examples*: identical message and situation with different initial VAD/mode, and human-authored response targets for each.
- Split by situation/conversation cluster, **not** by individual counterfactual row; otherwise paraphrases and duplicated prompts leak into test data.
- Measure text quality, correctness, appropriateness, consistency, diversity, VAD response sensitivity, harmful-response rate, state saturation, and latency.
- Repeat across seeds, report uncertainty intervals, and publish failures.
- Include shuffled affect-label controls to detect spurious signals.
- Determine if strong state changes reflect learned semantics or merely fixed response-style cues.
- Never claim emotional sentience, real subjective mood, or an empirically validated psychology model.

## 6. Explicit future boundary

Affect classification, long-term memory, learned state transitions, retrieval based on mood, and hosting a model inference service are separate future hypotheses/components. They are intentionally absent from the browser R0 demonstration. The GitHub Pages site remains static, inspectable, and free of secret API keys.
