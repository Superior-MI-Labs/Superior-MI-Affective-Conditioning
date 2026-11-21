// Browser capability/diagnostic contract shared by the UI and model worker.
// GPU support is probed, not inferred from browser brand or user agent.
export function browserFamily(userAgent = '', hasVivaldiApi = false) {
  if (hasVivaldiApi || /Vivaldi/i.test(userAgent)) return 'vivaldi';
  if (/Edg\//i.test(userAgent)) return 'edge';
  if (/Firefox\//i.test(userAgent)) return 'firefox';
  if (/Chrom(e|ium)\//i.test(userAgent)) return 'chromium';
  if (/Safari\//i.test(userAgent) && !/Chrome\//i.test(userAgent)) return 'safari';
  return 'other';
}

export const BROWSER_NAMES = Object.freeze({
  vivaldi: 'Vivaldi', edge: 'Microsoft Edge', firefox: 'Firefox',
  chromium: 'Chrome / Chromium', safari: 'Safari', other: 'Other browser',
});

export function browserGuidance(family) {
  const common = 'After changing any setting, fully restart the browser, then use “Check again”. Flags cannot add shader-f16 support to unsupported GPU hardware or drivers.';
  const steps = {
    vivaldi: [
      'Open vivaldi://settings and enable Hardware Acceleration.',
      'Inspect vivaldi://gpu for WebGPU and hardware acceleration status.',
      'Advanced, at your own risk: vivaldi://flags/#enable-vulkan and vivaldi://flags/#enable-unsafe-webgpu may expose WebGPU on otherwise blocked configurations. These bypass browser safeguards and are not required on every device.',
    ],
    chromium: [
      'Update Chrome or Chromium; enable graphics/hardware acceleration in its Settings > System.',
      'Check chrome://gpu for WebGPU support and driver status.',
      'If WebGPU remains unavailable on Linux, check the browser GPU blocklist and Vulkan drivers. Unsafe WebGPU flags are an advanced, last-resort workaround.',
    ],
    edge: [
      'Update Edge; enable graphics acceleration in Settings > System and performance.',
      'Check edge://gpu for WebGPU status.',
      'Check your graphics driver and OS support if no adapter is available.',
    ],
    firefox: [
      'Use an updated Firefox build with WebGPU enabled for your OS and hardware.',
      'Check about:support for graphics capabilities, and verify current Firefox WebGPU support.',
      'If your configuration is not supported, try a recent Chromium-based browser or CPU mode.',
    ],
    safari: [
      'Update Safari and your operating system; WebGPU support varies by version and device.',
      'Check that graphics acceleration is available.',
      'Try CPU mode if WebGPU is not exposed to this page.',
    ],
    other: [
      'Update your browser and enable hardware acceleration.',
      'Check its GPU diagnostics page and OS graphics drivers.',
      'Use a compatible WebGPU browser, or explicitly choose slow CPU mode.',
    ],
  };
  return {steps: steps[family] ?? steps.other, common};
}

export async function inspectWebGPU(gpu) {
  if (!gpu || typeof gpu.requestAdapter !== 'function') return {status: 'api-unavailable', shaderF16: false};
  try {
    const adapter = await gpu.requestAdapter();
    if (!adapter) return {status: 'adapter-unavailable', shaderF16: false};
    return {status: 'ready', shaderF16: Boolean(adapter.features?.has?.('shader-f16'))};
  } catch (error) {
    return {status: 'probe-failed', shaderF16: false, reason: String(error?.message || error)};
  }
}

// The *graph*, not just the quantized weight file, determines shader requirements.
// WebGPU LLM execution on this family requires shader-f16 in practice; the
// no-f16 strategy is to switch execution providers, not only dtype.
// CPU execution still requires device-specific verification.
export function inferencePlan(capability, preference = 'auto') {
  if (!['auto', 'cpu'].includes(preference)) throw new TypeError('Invalid browser inference backend.');
  if (preference === 'cpu') return {
    device: 'wasm', dtype: 'q4', model: 'onnx-community/Qwen2.5-0.5B-Instruct',
    label: 'Qwen2.5-0.5B-Instruct', reason: 'CPU / WebAssembly explicitly selected; substantial browser RAM required',
  };
  if (capability?.status === 'ready' && capability.shaderF16 === true) return {
    device: 'webgpu', dtype: 'q4f16', model: 'onnx-community/Qwen3-0.6B-ONNX',
    label: 'Qwen3-0.6B', reason: 'WebGPU adapter reports shader-f16',
  };
  throw new Error('Browser WebGPU cannot execute this model without shader-f16. Choose Local GGUF server to use your existing Qwen models, or explicitly try the memory-intensive CPU option.');
}

export function modelFailureMessage(error) {
  const raw = String(error?.message || error);
  if (/requires f16|shader-f16|does not support fp16|device does not support f16/i.test(raw)) {
    return 'A WebGPU kernel requires shader-f16, which the selected adapter cannot execute. Use CPU / WebAssembly mode or a supported GPU/driver. Quantizing the weights does not remove all FP16 operations. Details: ' + raw;
  }
  if (/out of memory|allocation failed|memory limit|OOM|bad_alloc|can.t create a session/i.test(raw)) {
    return 'The browser model could not allocate its ONNX session. CPU/WebAssembly is not a reliable fallback on this device. Choose Local GGUF server to run the model outside the browser, or inspect browser/OS memory usage. Details: ' + raw;
  }
  return raw;
}
