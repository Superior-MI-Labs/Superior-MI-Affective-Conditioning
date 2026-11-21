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

export function inferencePlan(capability, preference = 'webgpu') {
  if (preference === 'cpu') return {device: 'wasm', dtype: 'q4', reason: 'CPU explicitly selected'};
  if (preference !== 'webgpu') throw new TypeError('Invalid inference backend.');
  if (capability?.status !== 'ready') {
    throw new Error('WebGPU adapter unavailable. Check browser/driver setup or select CPU (slow) before loading Qwen.');
  }
  return {
    device: 'webgpu',
    dtype: capability.shaderF16 ? 'q4f16' : 'q4',
    reason: capability.shaderF16 ? '16-bit shaders supported' : '16-bit shaders unavailable: use float32-compatible q4',
  };
}

export function modelFailureMessage(error) {
  const raw = String(error?.message || error);
  if (/requires f16|shader-f16|does not support fp16|device does not support f16/i.test(raw)) {
    return 'This browser/GPU cannot execute a shader-f16 operation required by the selected model. Try reloading with CPU (slow), or update your browser and GPU drivers. Details: ' + raw;
  }
  if (/out of memory|allocation failed|memory limit|OOM/i.test(raw)) {
    return 'Insufficient GPU/browser memory for this model. Close GPU-heavy tabs or try CPU mode (requires substantial RAM). Details: ' + raw;
  }
  return raw;
}
