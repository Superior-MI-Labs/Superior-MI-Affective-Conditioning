"""Tests for blinded review, source immutability and conservative exports."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from review import compile_reviews, load_jsonl, prepare, seed_rows, task_key, make_tasks
from validate import validate


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.temp = Path(self.folder.name)
        original = load_jsonl(ROOT / 'data' / 'train_seed.jsonl')[:4]
        self.data = self.temp / 'train.jsonl'
        self.data.write_text(''.join(json.dumps(x) + '\n' for x in original), encoding='utf-8')
        self.tasks = self.temp / 'tasks.jsonl'
        prepare(self.data, self.tasks, 'fixed-seed')
        self.task_rows = load_jsonl(self.tasks)
        self.original_rows = seed_rows(self.data)

    def ratings(self, choice=lambda task, row: 'A'):
        original = {task_key(r): r for r in self.original_rows}
        for reviewer in ['r1', 'r2']:
            path = self.temp / f'{reviewer}.jsonl'
            rows = [{
                'task_id': t['task_id'], 'reviewer_id': reviewer,
                'choice': choice(t, original[t['task_id']]), 'notes': ''
            } for t in self.task_rows]
            path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
        return [self.temp / 'r1.jsonl', self.temp / 'r2.jsonl']

    def compile(self, ratings):
        return compile_reviews(self.data, self.tasks, ratings,
                               self.temp / 'approved.jsonl', self.temp / 'report.json')

    def test_published_task_handout_matches_source_of_truth(self):
        site = ROOT.parents[1] / 'web' / 'relational' / 'train_tasks.jsonl'
        prepared = make_tasks(seed_rows(ROOT / 'data' / 'train_seed.jsonl'), 'public-seed-review-v1')
        self.assertEqual(prepared, load_jsonl(site))
        self.assertEqual(len(prepared), 24)
        self.assertTrue(all('chosen' not in row and 'rejected' not in row for row in prepared))

    def test_source_and_report_must_never_be_overwritten(self):
        ratings = self.ratings()
        with self.assertRaisesRegex(ValueError, 'must not overwrite'):
            compile_reviews(self.data, self.tasks, ratings,
                            self.temp / 'approved.jsonl', self.data)

    def test_blind_tasks_have_no_preference_or_category_fields(self):
        for task in self.task_rows:
            self.assertEqual(set(task), {'task_id', 'source_sha256', 'context', 'response_a', 'response_b'})
            self.assertNotEqual(task['response_a'], task['response_b'])

    def test_prepare_deterministic_order_and_assignment(self):
        alternate = self.temp / 'tasks2.jsonl'
        prepare(self.data, alternate, 'fixed-seed')
        self.assertEqual(self.tasks.read_bytes(), alternate.read_bytes())

    def test_full_agreement_approves_complete_families(self):
        ratings = self.ratings(lambda task, row: 'A' if task['response_a'] == row['chosen'] else 'B')
        result = self.compile(ratings)
        self.assertEqual(result['approved_rows'], 4)
        self.assertEqual(result['approved_families'], 2)
        self.assertEqual(result['disputed_rows'], 0)
        self.assertEqual([], validate([str(self.temp / 'approved.jsonl')]))
        self.assertTrue(all(r['review_status'] == 'human_reviewed' for r in load_jsonl(self.temp / 'approved.jsonl')))

    def test_reviewers_can_reverse_original_author_labels(self):
        ratings = self.ratings(lambda task, row: 'B' if task['response_a'] == row['chosen'] else 'A')
        result = self.compile(ratings)
        self.assertEqual(result['label_reversals_from_author_draft'], 4)
        self.assertEqual([], validate([str(self.temp / 'approved.jsonl')]))

    def test_one_disagreement_blocks_entire_family(self):
        ratings = self.ratings(lambda task, row: 'A' if task['response_a'] == row['chosen'] else 'B')
        rows = load_jsonl(ratings[1]); rows[0]['choice'] = 'B' if rows[0]['choice'] == 'A' else 'A'
        ratings[1].write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
        result = self.compile(ratings)
        self.assertEqual(result['approved_rows'], 2)
        self.assertEqual(result['disputed_rows'], 1)

    def test_ties_and_unacceptable_excluded(self):
        ratings = self.ratings(lambda task, row: 'tie')
        result = self.compile(ratings)
        self.assertEqual(result['approved_rows'], 0)
        self.assertEqual(result['disputed_rows'], 4)

    def test_duplicate_reviewer_rejected(self):
        ratings = self.ratings()
        rows = load_jsonl(ratings[1])
        for row in rows: row['reviewer_id'] = 'r1'
        ratings[1].write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'duplicate reviewer'):
            self.compile(ratings)

    def test_unknown_task_rejected(self):
        ratings = self.ratings()
        rows = load_jsonl(ratings[0]); rows[0]['task_id'] = 'not_a_real_task'
        ratings[0].write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'unknown task'):
            self.compile(ratings)

    def test_modified_source_cannot_reuse_review_tasks(self):
        ratings = self.ratings()
        rows = load_jsonl(self.data)
        rows[0]['chosen'] += ' Extra.'
        rows[1]['rejected'] += ' Extra.'
        self.data.write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Source record changed'):
            self.compile(ratings)

    def test_non_train_split_rejected(self):
        rows = load_jsonl(self.data)
        for row in rows: row['split'] = 'test'
        self.data.write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Only train-split'):
            prepare(self.data, self.temp / 'other.jsonl', 'seed')

    def test_partial_review_excludes_unfinished_family(self):
        ratings = self.ratings(lambda task, row: 'A' if task['response_a'] == row['chosen'] else 'B')
        rows=load_jsonl(ratings[1]); rows=rows[:-1]
        ratings[1].write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
        result = self.compile(ratings)
        self.assertEqual(result['pending_rows'], 1)
        self.assertEqual(result['approved_rows'], 2)


if __name__ == '__main__':
    unittest.main()
