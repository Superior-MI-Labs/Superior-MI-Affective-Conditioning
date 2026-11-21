// The browser is the sole R0 authority for affective state transitions.
// Training consumes explicit, exported state records rather than duplicating
// this transition implementation in Python.
export const DEFAULTS = Object.freeze({
  ai: [0.5, 0.25, 0.1],
  message: [0.7, 0.7, 0.3],
  baseline: [0.2, 0.0, 0.0],
  phase: 0,
});

export function clamp(value, min = -1, max = 1) {
  if (!Number.isFinite(value)) throw new TypeError('Affect coordinates must be finite numbers');
  return Math.max(min, Math.min(max, value));
}

export function validateVAD(value) {
  if (!Array.isArray(value) || value.length !== 3) {
    throw new TypeError('VAD must contain exactly [valence, arousal, dominance]');
  }
  return value.map((entry) => {
    if (typeof entry !== 'number' || !Number.isFinite(entry) || entry < -1 || entry > 1) {
      throw new RangeError('VAD coordinates must be finite numbers between -1 and +1');
    }
    return entry;
  });
}

function angle(vad) {
  const [v, a] = vad;
  return Math.atan2(a, v);
}

function wrap(angleRadians) {
  return Math.atan2(Math.sin(angleRadians), Math.cos(angleRadians));
}

function vectorAlignment(ai, observed) {
  const normAI = Math.hypot(ai[0], ai[1]);
  const normObserved = Math.hypot(observed[0], observed[1]);
  if (normAI < 1e-9 || normObserved < 1e-9) return 0;
  return clamp((ai[0] * observed[0] + ai[1] * observed[1]) / (normAI * normObserved));
}

/** Returns a new state; never mutates the passed state. */
export function stepAffect(state, observedVAD, options = {}) {
  const ai = validateVAD(state.ai);
  const observed = validateVAD(observedVAD);
  const mode = options.mode ?? 'vector';
  if (mode !== 'vector' && mode !== 'phase') throw new RangeError('Unknown mode');
  const baseline = validateVAD(options.baseline ?? DEFAULTS.baseline);
  if (!Number.isFinite(state.phase)) throw new TypeError('Phase must be finite');
  const observedAngle = angle(observed);
  const alignment = mode === 'vector'
    ? vectorAlignment(ai, observed)
    : (Math.hypot(...observed.slice(0, 2)) < 1e-9 ? 0 : Math.cos(observedAngle - state.phase));
  const gain = 0.22 + 0.16 * alignment;
  const updated = ai.map((value, i) => clamp(
    value + gain * (observed[i] - value)
      + 0.08 * Math.max(0, alignment) * observed[i]
      - 0.025 * (value - baseline[i]),
  ));
  const phase = mode === 'phase'
    ? wrap(state.phase + 0.48 + 0.34 * Math.sin(observedAngle - state.phase))
    : state.phase;
  return {
    ai: updated,
    phase,
    alignment,
    gain,
    delta: observed.map((value, i) => value - ai[i]),
  };
}

export function initialState(ai = DEFAULTS.ai) {
  const value = validateVAD(ai);
  return {ai: [...value], phase: angle(value)};
}

export function affectName(vad) {
  const [v, a] = validateVAD(vad);
  if (v > 0.35 && a > 0.35) return 'Energized · Positive';
  if (v > 0.35) return 'Calm · Positive';
  if (v < -0.35 && a > 0.35) return 'Tense · Negative';
  if (v < -0.35) return 'Subdued · Negative';
  return a > 0.35 ? 'Activated · Mixed' : 'Neutral · Mixed';
}

// These strings are illustrative only: they are not model predictions.
export function templateResponse(message, updatedMood) {
  const mood = validateVAD(updatedMood);
  const text = message.trim();
  if (mood[0] > 0.65 && mood[1] > 0.45) return `That sounds interesting! Tell me a little more about what you mean by “${text.slice(0, 75)}”.`;
  if (mood[0] < -0.35) return 'I see what you are saying. Can you give me a concrete example to work from?';
  return 'I follow. What part would you like to explore first?';
}
