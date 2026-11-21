// Qwen remains unmodified: both affect modes condition it via ordinary prompt text.
// Keep the WebGPU check and model precision selection explicit and inspectable.
import {inspectWebGPU, inferencePlan, modelFailureMessage} from './browser-support.mjs';

const MODEL = 'onnx-community/Qwen3-0.6B-ONNX';
const TRANSFORMERS_CDN = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2';
let generator;
let loadedPreference;
let loading;
let transformers;

function status(message) { self.postMessage({type: 'status', message}); }
function reply(id, value) { self.postMessage({type: 'result', id, value}); }
function failure(id, error) { self.postMessage({type: 'error', id, message: modelFailureMessage(error)}); }

async function loadModel(preference = 'webgpu') {
  if (generator && loadedPreference === preference) return {device: preference === 'cpu' ? 'wasm' : 'webgpu'};
  if (loading) return loading;
  loading = (async () => {
    const capabilities = await inspectWebGPU(self.navigator?.gpu);
    const plan = inferencePlan(capabilities, preference);
    if (generator) {
      await generator.dispose?.();
      generator = undefined;
      loadedPreference = undefined;
    }
    status(`Loading Qwen3-0.6B (${plan.device}, ${plan.dtype}). First use downloads model weights.`);
    transformers ??= await import(TRANSFORMERS_CDN);
    const {pipeline} = transformers;
    const created = await pipeline('text-generation', MODEL, {
      device: plan.device,
      dtype: plan.dtype,
      progress_callback: (event) => {
        if (event?.status === 'progress' && event?.file && Number.isFinite(event.progress)) {
          status(`Downloading ${event.file.split('/').at(-1)}: ${Math.round(event.progress)}%`);
        }
      },
    });
    generator = created;
    loadedPreference = preference;
    status(`Qwen ready (${plan.device}, ${plan.dtype}). Generation runs locally in this browser.`);
    return plan;
  })().finally(() => {loading = null;});
  return loading;
}

self.onmessage = async ({data}) => {
  const {id, type} = data;
  try {
    if (type === 'diagnose') {
      reply(id, await inspectWebGPU(self.navigator?.gpu));
    } else if (type === 'load') {
      reply(id, await loadModel(data.preference));
    } else if (type === 'generate') {
      if (!generator) throw new Error('Qwen must be loaded first.');
      // Emit partial text and token counts. Some WebGPU devices need a long prefill
      // before the first token, and without streaming that looks like a hung chat.
      let streamed = '';
      let tokens = 0;
      const {TextStreamer} = transformers;
      const streamer = new TextStreamer(generator.tokenizer, {
        skip_prompt: true,
        skip_special_tokens: true,
        callback_function: (delta) => {
          if (!delta) return;
          streamed += delta;
          self.postMessage({type: 'progress', id, delta, tokens});
        },
        token_callback_function: () => {
          tokens++;
          self.postMessage({type: 'progress', id, tokens});
        },
      });
      status('Qwen preparing prompt and starting generation…');
      const output = await generator(data.messages, {
        max_new_tokens: 96,
        do_sample: false,
        tokenizer_encode_kwargs: {enable_thinking: false},
        streamer,
      });
      const conversation = output?.[0]?.generated_text;
      const last = Array.isArray(conversation) ? conversation.at(-1)?.content : conversation;
      // The pipeline result is authoritative. A streamed fallback is useful if
      // one version of Transformers.js returns a different output shape.
      const answer = typeof last === 'string' && last.trim() ? last : streamed;
      if (!answer.trim()) throw new Error(`Qwen generated ${tokens} tokens but returned no visible text. Try a short prompt or the CPU backend.`);
      reply(id, answer);
    } else {
      throw new Error('Unknown model-worker operation.');
    }
  } catch (error) {
    failure(id, error);
  }
};
