// Browser-only model transport; the page owns affect state and the worker owns the model.
// No hosted endpoint, API key, or duplicate dynamics implementation.
export class BrowserQwen {
  constructor({onStatus = () => {}, workerFactory = () => new Worker(new URL('./qwen-worker.mjs', import.meta.url), {type: 'module'})} = {}) {
    this.onStatus = onStatus;
    this.worker = workerFactory();
    this.pending = new Map();
    this.sequence = 0;
    this.ready = false;
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
    this.pending.delete(data.id);
    if (data.type === 'result') pending.resolve(data.value);
    else if (data.type === 'error') pending.reject(new Error(String(data.message || 'Unknown model error')));
  }

  failAll(message) {
    for (const entry of this.pending.values()) entry.reject(new Error(message));
    this.pending.clear();
    this.ready = false;
    this.loading = null;
    this.onStatus(message);
  }

  request(type, payload = {}) {
    return new Promise((resolve, reject) => {
      const id = ++this.sequence;
      this.pending.set(id, {resolve, reject});
      this.worker.postMessage({id, type, ...payload});
    });
  }

  async load() {
    if (this.ready) return;
    if (this.loading) return this.loading;
    this.loading = this.request('load').then(() => {this.ready = true;}).finally(() => {this.loading = null;});
    return this.loading;
  }

  async generate(messages) {
    if (!this.ready) throw new Error('Load Qwen before generating.');
    const answer = await this.request('generate', {messages});
    if (typeof answer !== 'string' || !answer.trim()) throw new Error('Qwen returned an empty response.');
    return answer.trim();
  }
}
