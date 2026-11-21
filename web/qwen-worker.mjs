// Runs an upstream, browser-compatible Qwen3 checkpoint entirely on the visitor's WebGPU device.
// This model is NOT our trained affective LoRA/projector. Inputs are ordinary chat messages.
const MODEL = 'onnx-community/Qwen3-0.6B-ONNX';
const TRANSFORMERS_CDN = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2';
let generator;
let loading;

function status(message) {self.postMessage({type: 'status', message});}
function reply(id, value) {self.postMessage({type: 'result', id, value});}
function failure(id, error) {self.postMessage({type: 'error', id, message: String(error?.message || error)});}

async function loadModel() {
  if (generator) return;
  if (!self.navigator?.gpu) {
    throw new Error('WebGPU is unavailable in this browser. Use a WebGPU-capable browser and device, or switch to simulation.');
  }
  if (!loading) loading = (async () => {
    status('Loading Qwen3-0.6B runtime; first visit downloads model weights.');
    const {pipeline} = await import(TRANSFORMERS_CDN);
    const model = await pipeline('text-generation', MODEL, {
      device: 'webgpu',
      dtype: 'q4f16',
      progress_callback: (event) => {
        if (event?.status === 'progress' && event?.file && Number.isFinite(event.progress)) {
          status(`Downloading ${event.file.split('/').at(-1)}: ${Math.round(event.progress)}%`);
        }
      },
    });
    generator = model;
    status('Qwen3-0.6B ready. All message generation runs in this browser.');
  })().finally(() => {loading = null;});
  return loading;
}

self.onmessage = async ({data}) => {
  const {id, type} = data;
  try {
    if (type === 'load') {
      await loadModel();
      reply(id, true);
    } else if (type === 'generate') {
      if (!generator) throw new Error('Qwen must be loaded first.');
      const output = await generator(data.messages, {max_new_tokens: 180, do_sample: false});
      const conversation = output?.[0]?.generated_text;
      const last = Array.isArray(conversation) ? conversation.at(-1)?.content : conversation;
      const answer = typeof last === 'string' ? last : '';
      if (!answer.trim()) throw new Error('The model produced no response text.');
      reply(id, answer);
    } else {
      throw new Error('Unknown model-worker operation.');
    }
  } catch (error) {
    failure(id, error);
  }
};
