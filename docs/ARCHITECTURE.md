# VAD neural architecture: experiment 001

## Research question

Can a small, randomly initialized neural network learn a repeatable, state-dependent transformation from a previous affective annotation and a new observed affective annotation to a next-state annotation?

This **does not** test whether a machine has feelings or empathy. It tests the first prerequisite: whether a trainable network can learn a numerical state-transition relation.

## One authoritative computation

1. `current = [V, A, D]` and `observed = [V, A, D]`, with each coordinate in `[-1, 1]`.
2. Concatenate into a six-component vector.
3. `hidden = tanh(W1 * input + b1)`, where W1 has shape `[16, 6]`.
4. `next = tanh(W2 * hidden + b2)`, where W2 has shape `[3, 16]`.
5. `delta = next - current` (derived, never stored as another source of truth).
6. On a reviewed training example, `loss = mean((next - target)^2)` and update weights by backpropagation.

The Python model is the only trainer and the source of exported model weights. Browser code evaluates the same two matrices on input vectors; the export includes a PyTorch reference probe validated at page load and in CI.

## Data contract

One JSONL row per independently reviewed transition:

```json
{"source":"human_reviewed","current":[0.0,0.0,0.0],"observed":[0.4,0.2,-0.1],"next":[0.1,0.1,0.0]}
```

These numbers are illustrative, not authoritative psychological judgments. Avoid converting model predictions directly into training targets. The website only collects manual labels, and data is exported explicitly by the user.

## Deliberate limits

- There is **no** learned emotion vocabulary, mood-script, language model, recurrent architecture beyond forwarding a previous prediction, automatic emotion detection, or consciousness mechanism.
- Advancing the prediction into the next input demonstrates iterative transitions, but the current network has **no separate learned hidden memory**. Calling it a recurrent empathic substrate would overstate it.
- Synthetic training uses a simple algebraic rule solely to test the pipeline. A low error on the toy rule is not evidence of real affective understanding.
- The 80/20 randomized holdout is adequate for the toy exercise only. Real evaluation must split by distinct situation/participant to avoid leakage.

## Next empirical milestone

Collect a small, diverse, reviewed set of VAD transition judgments, document the annotation instructions, test inter-annotator reliability, split by distinct scenarios, and compare the trained network against copy-current and simple linear baselines. Add temporal latent memory, language, or social interaction **only if an experiment shows why they're needed**.
