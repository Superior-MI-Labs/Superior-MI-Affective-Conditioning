"""Regression checks for challenge data, blind task pack, and scale-neutral DPO export."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from generate_challenges import generate, encode, write_files
from export_dpo import export
from review import load_jsonl, make_tasks, seed_rows
from validate import validate


class ChallengeDatasetTests(unittest.TestCase):
    def test_generator_exactly_matches_files(self):
        self.assertEqual({'train':44,'dev':10,'test':10}, write_files(ROOT / 'data', check=True))

    def test_challenges_are_counterfactual_and_isolated(self):
        paths=[str(p) for p in sorted((ROOT/'data').glob('*.jsonl'))]
        self.assertEqual([],validate(paths))
        families=generate()
        self.assertEqual(32,len({r['family_id'] for rows in families.values() for r in rows}))
        self.assertTrue(all(r['review_status']=='synthetic_unreviewed' for rows in families.values() for r in rows))

    def test_public_review_page_has_distinct_task_packs(self):
        page=(ROOT.parents[1]/'web'/'relational'/'review.html').read_text(encoding='utf-8')
        self.assertIn('href="./train_tasks.jsonl"',page)
        self.assertIn('href="./challenge_tasks.jsonl"',page)
        self.assertIn('Review each pack separately',page)

    def test_challenge_task_pack_matches_source_without_labels(self):
        source=ROOT/'data'/'train_challenge.jsonl'
        actual=load_jsonl(ROOT.parents[1]/'web'/'relational'/'challenge_tasks.jsonl')
        expected=make_tasks(seed_rows(source), 'public-challenge-review-v1')
        self.assertEqual(expected,actual)
        for row in actual:
            self.assertEqual({'task_id','source_sha256','context','response_a','response_b'},set(row))


class DPOExportTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name)
        self.rows=load_jsonl(ROOT/'data'/'train_challenge.jsonl')[:4]
        self.src=self.root/'source.jsonl'
        self.out=self.root/'dpo.jsonl'
        self.report=self.root/'dpo.report.json'

    def write(self,rows=None):
        self.src.write_text(''.join(json.dumps(r)+'\n' for r in (rows or self.rows)),encoding='utf-8')

    def test_synthetic_labels_never_train(self):
        self.write()
        with self.assertRaisesRegex(ValueError,'synthetic labels prohibited'):
            export(self.src,self.out,self.report)
        self.assertFalse(self.out.exists())

    def test_reviewed_records_convert_without_analysis_leak(self):
        rows=copy.deepcopy(self.rows)
        for r in rows:
            r['review_status']='human_reviewed'
            r['provenance']['usage']='reviewed_training_candidate'
            r['review_notes']='PRIVATE REVIEWER NOTE MUST NEVER APPEAR IN MODEL INPUT'
        self.write(rows)
        summary=export(self.src,self.out,self.report)
        actual=load_jsonl(self.out)
        self.assertEqual(4,summary['pairs'])
        self.assertEqual(2,summary['families'])
        self.assertEqual(set(actual[0]),{'prompt','chosen','rejected'})
        self.assertEqual(actual[0]['prompt'],rows[0]['context'])
        self.assertEqual(actual[0]['chosen'],[{'role':'assistant','content':rows[0]['chosen']}])
        self.assertEqual(actual[0]['rejected'],[{'role':'assistant','content':rows[0]['rejected']}])
        self.assertNotIn('PRIVATE REVIEWER NOTE', self.out.read_text())
        self.assertEqual('trl-conversational-dpo-v1',json.loads(self.report.read_text())['format'])

    def test_reviewed_status_alone_not_enough(self):
        rows=copy.deepcopy(self.rows)
        for r in rows:r['review_status']='human_reviewed'
        self.write(rows)
        with self.assertRaisesRegex(ValueError,'two-reviewer-approved'):
            export(self.src,self.out,self.report)

    def test_eval_split_forbidden_even_if_marked_reviewed(self):
        rows=copy.deepcopy(self.rows)
        for r in rows:
            r['split']='test';r['review_status']='human_reviewed'
            r['provenance']['usage']='reviewed_training_candidate'
        self.write(rows)
        with self.assertRaisesRegex(ValueError,'Only train-split'):
            export(self.src,self.out,self.report)

    def test_partial_family_rejected(self):
        rows=copy.deepcopy(self.rows[:3])
        for r in rows:
            r['review_status']='human_reviewed'
            r['provenance']['usage']='reviewed_training_candidate'
        self.write(rows)
        with self.assertRaisesRegex(ValueError,'expected exactly two'):
            export(self.src,self.out,self.report)

    def test_refuse_output_overwrites(self):
        self.write()
        with self.assertRaisesRegex(ValueError,'distinct'):
            export(self.src,self.src,self.report)
        self.out.write_text('keep me')
        with self.assertRaisesRegex(ValueError,'must not exist'):
            export(self.src,self.out,self.report)
        self.assertEqual('keep me',self.out.read_text())

if __name__=='__main__':unittest.main()
