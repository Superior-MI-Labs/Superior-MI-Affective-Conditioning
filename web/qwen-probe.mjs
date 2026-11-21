// A loaded ONNX session is not proof that its kernels can execute.
// Probe one generated token to catch unsupported shader operations before showing Ready.
export async function probeGeneration(generator) {
  if (typeof generator !== 'function') throw new TypeError('Invalid Qwen generator');
  await generator([{role:'user',content:'Hi'}], {max_new_tokens:1,do_sample:false});
}
