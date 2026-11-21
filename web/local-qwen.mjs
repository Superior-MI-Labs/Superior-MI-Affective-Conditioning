// Local inference transport for an existing OpenAI-compatible llama.cpp server.
// The browser owns affect state; this client does not compute emotion or load weights.
// Fixed loopback origin: never accept an arbitrary network endpoint from page inputs.
const ORIGIN = 'http://127.0.0.1:8080';

function connectionError(error) {
  if (error?.name === 'AbortError') return 'Local generation stopped.';
  if (error?.name === 'TimeoutError') return 'Local model service did not respond in time.';
  if (error instanceof TypeError) return 'Cannot reach the local GGUF service at 127.0.0.1:8080. Start llama-server, check its CORS origin and allow any browser loopback-network permission. Check the browser console for blocked requests.';
  return String(error?.message || error);
}

function normalizeError(response, body) {
  if (response.status === 401 || response.status === 403) return 'Local model API key rejected. Enter the key printed by scripts/run-local-qwen.sh.';
  return `Local model service HTTP ${response.status}: ${String(body?.error?.message || body?.error || response.statusText).slice(0, 300)}`;
}

async function readResponseJSON(response) {
  let body;
  try { body = await response.json(); } catch { throw new Error(`Local model service returned an invalid response (HTTP ${response.status}).`); }
  if (!response.ok) throw new Error(normalizeError(response, body));
  return body;
}

export class LocalQwen {
  constructor({fetchImpl = (...args) => fetch(...args), apiKey = () => ''} = {}) {
    this.fetchImpl = fetchImpl;
    this.apiKey = apiKey;
    this.preference = null;
    this.plan = null;
    this.ready = false;
    this.controller = null;
  }

  headers(json = false) {
    const key = this.apiKey().trim();
    if (!key) throw new Error('Enter the local server API key before connecting.');
    return {...(json ? {'Content-Type': 'application/json'} : {}), Authorization: `Bearer ${key}`};
  }

  requestOptions(options = {}) {
    // Modern Chromium may ask the user to permit loopback-network access.
    // Older browsers ignore the targetAddressSpace hint.
    return {...options, targetAddressSpace: 'loopback', credentials: 'omit', redirect: 'error', cache: 'no-store'};
  }

  async load(preference = 'local') {
    if (preference !== 'local') throw new TypeError('LocalQwen only supports the local backend.');
    if (this.ready) return this.plan;
    this.controller = new AbortController();
    const signal = this.controller.signal;
    try {
      const response = await this.fetchImpl(`${ORIGIN}/v1/models`, this.requestOptions({
        method: 'GET', headers: this.headers(), signal,
      }));
      const body = await readResponseJSON(response);
      const model = body?.data?.[0]?.id;
      if (typeof model !== 'string' || !model.trim()) throw new Error('The local service did not report a loaded model.');
      this.plan = {model, label: model, device: 'local', dtype: 'gguf', reason: 'Connected to an authenticated loopback llama.cpp server'};
      this.preference = preference;
      this.ready = true;
      return this.plan;
    } catch (error) {
      this.ready = false;
      this.plan = null;
      throw new Error(connectionError(error));
    }
  }

  async generate(messages, onProgress = () => {}) {
    if (!this.ready) throw new Error('Connect to the local model server first.');
    this.controller = new AbortController();
    const signal = this.controller.signal;
    let answer = '';
    let chunks = 0;
    try {
      const response = await this.fetchImpl(`${ORIGIN}/v1/chat/completions`, this.requestOptions({
        method: 'POST', signal, headers: this.headers(true),
        body: JSON.stringify({model: this.plan.model, messages, stream: true, temperature: 0, max_tokens: 96}),
      }));
      if (!response.ok) {
        let body;
        try {body = await response.json();} catch {body = null;}
        throw new Error(normalizeError(response, body));
      }
      if (!response.body?.getReader) throw new Error('The local server did not return a readable response stream.');
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      const consumeEvent = (event) => {
        const data = event.split('\n').filter(line => line.startsWith('data:')).map(line => line.slice(5).trim()).join('\n');
        if (!data || data === '[DONE]') return;
        const item = JSON.parse(data);
        if (item.error) throw new Error(String(item.error.message || item.error));
        const delta = item.choices?.[0]?.delta?.content;
        if (typeof delta === 'string' && delta) {
          answer += delta;
          onProgress({delta, chunks: ++chunks});
        }
      };
      while (true) {
        const {done, value} = await reader.read();
        buffer += done ? decoder.decode() : decoder.decode(value, {stream: true});
        buffer = buffer.replace(/\r\n/g, '\n');
        let boundary;
        while ((boundary = buffer.indexOf('\n\n')) !== -1) {
          consumeEvent(buffer.slice(0, boundary));
          buffer = buffer.slice(boundary + 2);
        }
        if (done) break;
      }
      if (buffer.trim()) consumeEvent(buffer);
      if (!answer.trim()) throw new Error('The local model returned no visible response.');
      return answer.trim();
    } catch (error) {
      throw new Error(connectionError(error));
    }
  }

  dispose() {
    this.controller?.abort();
    this.ready = false;
    this.preference = null;
    this.plan = null;
  }
}
