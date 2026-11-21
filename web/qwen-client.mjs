// Browser-only transport. The worker owns the inference session; the page owns affect state.
export class BrowserQwen {
  constructor({onStatus = () => {}, workerFactory = () => new Worker(new URL('./qwen-worker.mjs', import.meta.url), {type: 'module'})} = {}) {
    this.onStatus = onStatus;
    this.worker = workerFactory();
    this.pending = new Map();
    this.sequence = 0;
    this.ready = false;
    this.preference = null;
    this.plan = null;
    this.loading = null;
    this.worker.addEventListener('message', (event) => this.receive(event.data));
    this.worker.addEventListener('error', () => this.failAll('Browser model worker failed; reload the page to retry.'));
  }

  receive(data) {
    if (data?.type === 'status') {
      this.onStatus(String(data.message || 'Loading model'));
      return;
    }
    const pending = this.pending.get(data?.id);
    if (!pending) return;
    if (data.type === 'progress') {
      pending.onProgress?.(data);
      return;
    }
    this.pending.delete(data.id);
    if (data.type === 'result') pending.resolve(data.value);
    else if (data.type === 'error') pending.reject(new Error(String(data.message || 'Unknown model error')));
  }

  failAll(message) {
    for (const entry of this.pending.values()) entry.reject(new Error(message));
    this.pending.clear();
    this.ready = false;
    this.loading = null;
    this.preference = null;
    this.plan = null;
    this.onStatus(message);
  }

  request(type, payload = {}, onProgress) {
    return new Promise((resolve, reject) => {
      const id = ++this.sequence;
      this.pending.set(id, {resolve, reject, onProgress});
      this.worker.postMessage({id, type, ...payload});
    });
  }

  diagnose() { return this.request('diagnose'); }

  async load(preference = 'webgpu') {
    if (this.ready && this.preference === preference) return this.plan;
    if (this.loading) return this.loading;
    this.loading = this.request('load', {preference}).then((plan) => {
      this.ready = true;
      this.preference = preference;
      this.plan = plan;
      return plan;
    }).catch((error) => {
      // Never keep a stale ready flag when switching backends fails.
      this.ready = false;
      this.preference = null;
      this.plan = null;
      throw error;
    }).finally(() => {this.loading = null;});
    return this.loading;
  }

  async generate(messages, onProgress) {
    if (!this.ready) throw new Error('Load Qwen before generating.');
    const answer = await this.request('generate', {messages}, onProgress);
    if (typeof answer !== 'string' || !answer.trim()) throw new Error('Qwen returned an empty response.');
    return answer.trim();
  }

  dispose() {
    this.failAll('Generation canceled; load Qwen again to retry.');
    this.worker.terminate?.();
  }
}
