"""Synthetic training is explicit, provenance-preserving, and separated from evaluation."""
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from export_dpo import export_many
from train_dpo import check_inputs
from evaluate import candidate_pair, load_evaluation, metrics, summarize
from review import load_jsonl


class SyntheticPOCTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name)
        self.train=[ROOT/'data'/'train_seed.jsonl',ROOT/'data'/'train_challenge.jsonl']
        self.dev=[ROOT/'data'/'dev_seed.jsonl',ROOT/'data'/'dev_challenge.jsonl']
        self.output=self.root/'dpo.jsonl'
        self.report=self.root/'export.json'

    def test_synthetic_opt_in_and_combined_train(self):
        report=export_many(self.train,self.output,self.report,mode='synthetic')
        self.assertEqual(68,report['pairs'])
        self.assertEqual(34,report['families'])
        self.assertEqual('synthetic_author_preference_unverified',report['review_status'])
        exported=load_jsonl(self.output)
        self.assertEqual({'prompt','chosen','rejected'},set(exported[0]))
        self.assertNotIn('review_notes',self.output.read_text())
        self.assertNotIn('category',self.output.read_text())
        self.assertEqual('synthetic',report['label_origin'])
        with self.assertRaisesRegex(ValueError,'acknowledge-synthetic'):
            check_inputs(self.output,self.report,self.root/'adapter',False)
        self.assertEqual(68,len(check_inputs(self.output,self.report,self.root/'adapter',True)['records']))

    def test_existing_reviewed_only_contract_does_not_change(self):
        with self.assertRaisesRegex(ValueError,'synthetic labels prohibited'):
            export_many(self.train,self.output,self.report)

    def test_data_tampering_is_caught_before_training(self):
        export_many(self.train,self.output,self.report,mode='synthetic')
        self.output.write_text(self.output.read_text()+'\n',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'SHA256'):
            check_inputs(self.output,self.report,self.root/'adapter',True)

    def test_synthetic_mode_will_not_launder_reviewed_or_disputed_provenance(self):
        rows=load_jsonl(self.train[0])[:2]
        rows[0]['review_status']='disputed'
        src=self.root/'tainted.jsonl'
        src.write_text(''.join(json.dumps(x)+'\n' for x in rows))
        with self.assertRaisesRegex(ValueError,'Synthetic mode requires'):
            export_many([src],self.output,self.report,mode='synthetic')

    def test_split_leak_or_eval_export_blocked(self):
        with self.assertRaisesRegex(ValueError,'Only train-split'):
            export_many(self.dev,self.output,self.report,mode='synthetic')
        with self.assertRaisesRegex(ValueError,'leaks across'):
            first=load_jsonl(self.train[0])[:2]
            first[1]['split']='test'
            path=self.root/'leak.jsonl'
            path.write_text(''.join(json.dumps(r)+'\n' for r in first))
            export_many([path],self.output,self.report,mode='synthetic')

    def test_last_turn_control_is_contradictory_by_design(self):
        report=export_many(self.train,self.output,self.report,mode='synthetic',prompt_mode='last_turn')
        self.assertEqual('last_turn',report['prompt_mode'])
        rows=load_jsonl(self.output)
        self.assertTrue(all(len(r['prompt'])==1 for r in rows))
        self.assertEqual(rows[0]['prompt'],rows[1]['prompt'])
        self.assertEqual(rows[0]['chosen'],rows[1]['rejected'])
        self.assertEqual(rows[0]['rejected'],rows[1]['chosen'])

    def test_evaluation_rejects_train_split(self):
        with self.assertRaisesRegex(ValueError,'Training-split'):
            load_evaluation(self.train)
        self.assertEqual(16,len(load_evaluation(self.dev)))

    def test_candidate_order_stable_across_counterfactual_contexts(self):
        a,b=load_jsonl(self.dev[0])[:2]
        self.assertEqual(candidate_pair(a),candidate_pair(b))

    def test_flip_metric_requires_both_contexts(self):
        rows=[]
        for i,margin in [(1,1.2),(2,-.8)]:
            rows.append({'id':str(i),'family_id':'family','split':'test','variant':i,
                         'base_full':margin,'base_last':1.0,
                         'trained_full':margin,'trained_last':1.0})
        result=metrics(rows)
        self.assertEqual(1,result['base_full']['family_flip_correct'])
        self.assertEqual(0,result['base_last']['family_flip_correct'])
        self.assertEqual(.5,result['base_last']['row_accuracy'])
        rows[1]['trained_full']=+.9
        self.assertEqual(0,metrics(rows)['trained_full']['family_flip_correct'])

    def test_malformed_and_leaking_scores_rejected(self):
        with self.assertRaisesRegex(ValueError,'Missing or duplicate'):
            metrics([{'family_id':'f','variant':1}])
        p=self.root/'scores.jsonl'
        p.write_text(json.dumps({'id':'fake','family_id':'f','variant':1,'split':'train',
                                  **{n:0 for n in ('base_full','base_last','trained_full','trained_last')}})+'\n')
        with self.assertRaisesRegex(ValueError,'Non-held-out'):
            summarize(p,self.root/'summary.json')

    def test_existing_seed_remains_unchanged(self):
        rows=load_jsonl(self.train[0])
        self.assertEqual(24,len(rows))
        self.assertTrue(all(row['review_status']=='synthetic_unreviewed' for row in rows))

if __name__=='__main__':unittest.main()
