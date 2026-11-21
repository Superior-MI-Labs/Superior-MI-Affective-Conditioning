// A loaded ONNX session is not proof that its kernels can execute.
// Probe prefill AND cached decoding. A one-token test can miss the decode graph
// that fails only when the next token is requested.
export async function probeGeneration(generator) {
  if (typeof generator !== 'function') throw new TypeError('Invalid Qwen generator');
  await generator([{role:'user',content:'Please write at least three words in English.'}], {max_new_tokens:3,min_new_tokens:3,do_sample:false});
}
