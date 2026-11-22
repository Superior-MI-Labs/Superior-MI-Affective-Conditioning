"""Torch tests run locally; Pages builds validate exported browser weights with Node."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    import torch
    from vad.model import VADNetwork, check_vad, load_reviewed, synthetic_examples, train, web_artifact
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, "PyTorch optional for static Pages deployment")
class VADTests(unittest.TestCase):
    def test_validation(self):
        for invalid in ([0, 1], [0, 0, 2], [0, 0, float('nan')], [True, 0, 0]):
            with self.assertRaises(ValueError):
                check_vad(invalid, "value")

    def test_toy_inputs_and_targets_are_reproducible(self):
        x, y = synthetic_examples(100, 7)
        x2, y2 = synthetic_examples(100, 7)
        self.assertTrue(torch.equal(x, x2))
        self.assertTrue(torch.equal(y, y2))
        self.assertEqual(tuple(x.shape), (100, 6))

    def test_reviewed_records_require_explicit_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'examples.jsonl'
            example = {"source": "synthetic", "current": [0, 0, 0],
                       "observed": [1, 0, 0], "next": [.4, 0, 0]}
            path.write_text('\n'.join(json.dumps(example) for _ in range(10)))
            with self.assertRaisesRegex(ValueError, 'human_reviewed'):
                load_reviewed(path)
            example['source']='human_reviewed'
            path.write_text('\n'.join(json.dumps(example) for _ in range(10)))
            x, y = load_reviewed(path)
            self.assertEqual(tuple(x.shape), (10, 6))
            self.assertEqual(tuple(y.shape), (10, 3))

    def test_trained_network_learns_toy_mapping(self):
        x,y=synthetic_examples(400,23)
        model, metrics=train(x,y,epochs=120,seed=23)
        self.assertLess(metrics['test_mse'],metrics['copy_state_baseline_mse']*.15)
        self.assertTrue((model(x).abs()<=1).all().item())
        artifact=web_artifact(model,'synthetic_toy',metrics)
        self.assertEqual(artifact['architecture'],[6,16,3])
        self.assertEqual(len(artifact['layers'][0]['weights']),16)
        self.assertEqual(len(artifact['layers'][1]['weights']),3)


if __name__=='__main__':
    unittest.main()
