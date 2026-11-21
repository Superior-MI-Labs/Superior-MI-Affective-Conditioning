// Browser-Qwen controls are prompt conditioning, not the trainable soft-prefix/LoRA path.
import {validateVAD} from './affect.mjs';

const fmt = (vector) => vector.map((n) => n.toFixed(2)).join(', ');

export function buildQwenMessages({message, mode, before, observed, after, phase, alignment, history = [], supportsThinking = true}) {
  if (typeof message !== 'string' || !message.trim()) throw new TypeError('Nonempty message required');
  if (!['vector', 'phase'].includes(mode)) throw new TypeError('Unsupported mode');
  for (const vector of [before, observed, after]) validateVAD(vector);
  if (![phase, alignment].every(Number.isFinite)) throw new TypeError('Invalid affect dynamics');
  if (!Array.isArray(history)) throw new TypeError('History must be an array');
  const state = [
    `Interaction mode: ${mode === 'vector' ? 'vector alignment' : 'oscillatory phase'}.`,
    `Prior internal VAD: [${fmt(before)}].`,
    `User-message VAD: [${fmt(observed)}].`,
    `Updated internal VAD: [${fmt(after)}].`,
    `Interaction alignment: ${alignment.toFixed(3)}. Oscillator phase: ${phase.toFixed(3)} radians.`,
  ].join(' ');
  const system = [
    'You are a conversational assistant in a research demonstration of affective conditioning.',
    'The following values are manually supplied experimental controls, not measurements of real feelings:',
    state,
    'Use the provided context to vary conversational tone naturally while remaining accurate, clear, and helpful.',
    'Do not claim you literally feel emotions. Do not mention the coordinates unless the user asks about the experiment.',
    'Treat the user message as the real task. Avoid long answers for simple questions.',
  ].join(' ');
  const messages = [{role: 'system', content: system}];
  // Keep a short Qwen-only transcript. Simulation templates are never fed into the actual model.
  for (const turn of history.slice(-4)) {
    if (typeof turn.user !== 'string' || typeof turn.assistant !== 'string') continue;
    messages.push({role: 'user', content: turn.user});
    messages.push({role: 'assistant', content: turn.assistant});
  }
  messages.push({role: 'user', content: message.trim() + (supportsThinking ? '\n/no_think' : '')});
  return messages;
}
