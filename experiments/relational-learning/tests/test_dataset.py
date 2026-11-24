import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from validate import validate

class DatasetTests(unittest.TestCase):
    def test_seed_valid(self):
        self.assertEqual([],validate([str(ROOT/'data'/f'{p}_seed.jsonl') for p in ['train','dev','test']]))

    def test_cross_split_leak_rejected(self):
        src=(ROOT/'data'/'train_seed.jsonl').read_text(encoding='utf-8').splitlines()
        a=json.loads(src[0]); b=json.loads(src[1]); b['split']='test'
        with tempfile.TemporaryDirectory() as td:
            f=Path(td)/'broken.jsonl'
            f.write_text(json.dumps(a)+'\n'+json.dumps(b)+'\n',encoding='utf-8')
            self.assertTrue(any('leaks across' in e for e in validate([str(f)])))

    def test_identical_candidate_rejected(self):
        src=(ROOT/'data'/'train_seed.jsonl').read_text(encoding='utf-8').splitlines()
        a=json.loads(src[0]); b=json.loads(src[1]); a['rejected']=a['chosen']
        with tempfile.TemporaryDirectory() as td:
            f=Path(td)/'broken.jsonl'; f.write_text(json.dumps(a)+'\n'+json.dumps(b)+'\n',encoding='utf-8')
            self.assertTrue(any('identical' in e for e in validate([str(f)])))

    def test_reversed_candidate_pool(self):
        src=(ROOT/'data'/'train_seed.jsonl').read_text(encoding='utf-8').splitlines()
        a=json.loads(src[0]); b=json.loads(src[1]); b['rejected']='Something else.'
        with tempfile.TemporaryDirectory() as td:
            f=Path(td)/'broken.jsonl'; f.write_text(json.dumps(a)+'\n'+json.dumps(b)+'\n',encoding='utf-8')
            self.assertTrue(any('reversed preference' in e for e in validate([str(f)])))

    def test_malformed_records_report_errors_without_crashing(self):
        src=(ROOT/'data'/'train_seed.jsonl').read_text(encoding='utf-8').splitlines()
        a=json.loads(src[0]); b=json.loads(src[1]); a['context']=None
        with tempfile.TemporaryDirectory() as td:
            f=Path(td)/'broken.jsonl'; f.write_text(json.dumps(a)+'\n'+json.dumps(b)+'\n',encoding='utf-8')
            self.assertTrue(any('invalid context' in e for e in validate([str(f)])))

    def test_missing_file_reports_error(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertTrue(any('cannot read' in e for e in validate([str(Path(td)/'missing.jsonl')])))

    def test_last_message_counterfactual_consistency(self):
        src=(ROOT/'data'/'train_seed.jsonl').read_text(encoding='utf-8').splitlines()
        a=json.loads(src[0]); b=json.loads(src[1]); b['context'][-1]['content']='This is different.'
        with tempfile.TemporaryDirectory() as td:
            f=Path(td)/'broken.jsonl'; f.write_text(json.dumps(a)+'\n'+json.dumps(b)+'\n',encoding='utf-8')
            self.assertTrue(any('final user message' in e for e in validate([str(f)])))

if __name__=='__main__': unittest.main()
