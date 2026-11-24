'use strict';
// Thin, local-only review UI. Python review.py remains the authoritative compiler.
const $ = id => document.getElementById(id);
let tasks = [], index = 0, judgments = new Map();
const allowed = new Set(['A', 'B', 'tie', 'both_unacceptable']);

function render() {
  const active = tasks.length > 0;
  $('task-view').hidden = !active;
  if (!active) return;
  const task = tasks[index];
  const saved = judgments.get(task.task_id);
  $('counter').textContent = `TASK ${index + 1} OF ${tasks.length} · ${judgments.size} SAVED`;
  const history = $('history');
  history.replaceChildren();
  for (const turn of task.context) {
    const line = document.createElement('p');
    const speaker = document.createElement('strong');
    speaker.textContent = (turn.role === 'user' ? 'User: ' : 'Assistant: ');
    line.append(speaker, document.createTextNode(String(turn.content)));
    history.append(line);
  }
  $('response-a').textContent = task.response_a;
  $('response-b').textContent = task.response_b;
  for (const radio of document.querySelectorAll('[name="judgment"]')) radio.checked = saved?.choice === radio.value;
  $('notes').value = saved?.notes || '';
  $('previous').disabled = index === 0;
  $('save-next').textContent = index === tasks.length - 1 ? 'Save final judgment' : 'Save judgment & next';
}

$('task-file').addEventListener('change', async event => {
  tasks = []; judgments.clear(); index = 0;
  const file = event.target.files?.[0];
  if (!file) return render();
  try {
    const lines = (await file.text()).split(/\r?\n/).filter(line => line.trim());
    const parsed = lines.map(line => JSON.parse(line));
    const ids = new Set();
    if (!parsed.length || parsed.some(t => {
      if (typeof t.task_id !== 'string' || ids.has(t.task_id) || !Array.isArray(t.context) ||
          typeof t.response_a !== 'string' || typeof t.response_b !== 'string' ||
          typeof t.source_sha256 !== 'string') return true;
      ids.add(t.task_id);
      return t.context.some(turn => !['user', 'assistant'].includes(turn.role) || typeof turn.content !== 'string');
    })) throw new Error('Task file is empty or has invalid records');
    tasks = parsed;
    $('status').textContent = `${tasks.length} tasks loaded locally. No labels or data have been uploaded.`;
    render();
  } catch (error) {
    $('status').textContent = `Could not read task file: ${error.message}`;
    render();
  }
});

$('previous').addEventListener('click', () => { if (index > 0) { index--; render(); } });
$('save-next').addEventListener('click', () => {
  if (!tasks.length) return;
  const reviewer = $('reviewer-id').value.trim();
  if (!reviewer || reviewer.length > 64) { $('status').textContent = 'Enter a pseudonymous reviewer code first.'; return; }
  const choice = document.querySelector('[name="judgment"]:checked')?.value;
  if (!allowed.has(choice)) { $('status').textContent = 'Choose A, B, tie, or both unacceptable.'; return; }
  const task = tasks[index];
  judgments.set(task.task_id, {task_id: task.task_id, reviewer_id: reviewer, choice, notes: $('notes').value.trim()});
  $('status').textContent = `${judgments.size} of ${tasks.length} judgments recorded in memory. Export before leaving.`;
  if (index < tasks.length - 1) index++;
  render();
});
$('export').addEventListener('click', () => {
  if (!judgments.size) { $('status').textContent = 'Save at least one judgment before exporting.'; return; }
  const reviewer = $('reviewer-id').value.trim();
  if (!reviewer || [...judgments.values()].some(j => j.reviewer_id !== reviewer)) {
    $('status').textContent = 'Use the same reviewer code throughout this review; export before changing it.';
    return;
  }
  const content = [...judgments.values()].map(j => JSON.stringify(j)).join('\n') + '\n';
  const url = URL.createObjectURL(new Blob([content], {type: 'application/x-ndjson'}));
  const a = document.createElement('a');
  a.href = url;
  a.download = 'relational-review-' + reviewer.replace(/[^a-z0-9_-]/gi, '_') + '.jsonl';
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  $('status').textContent = `Exported ${judgments.size} judgments. Keep reviewer files private until reviewed and compiled.`;
});
