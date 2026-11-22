"""Optional torch-powered tests; keep the static Pages CI independent of torch."""
import json
import tempfile
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
    from affective.vad_transition import VADTransition, fit, load_examples, toy_examples, vad
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, "requires optional PyTorch")
class VADTransitionTests(unittest.TestCase):
    def test_validation(self):
        for bad in ([0, 1], [0, 0, 2], [0, 0, float("nan")], [True, 0, 0]):
            with self.assertRaises(ValueError):
                vad(bad, "input")

    def test_toy_data_deterministic_and_bounded(self):
        x, y = toy_examples(100, 5)
        x2, y2 = toy_examples(100, 5)
        self.assertTrue(torch.equal(x, x2))
        self.assertTrue(torch.equal(y, y2))
        self.assertEqual(tuple(x.shape), (100, 6))
        self.assertTrue((y.abs() <= 1).all().item())

    def test_reviewed_data_required(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "examples.jsonl"
            rows = [dict(source="synthetic", current=[0, 0, 0],
                         observed=[1, 0, 0], next=[0.5, 0, 0]) for _ in range(10)]
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "human_reviewed"):
                load_examples(path)
            for row in rows:
                row["source"] = "human_reviewed"
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            x, y = load_examples(path)
            self.assertEqual(tuple(x.shape), (10, 6))
            self.assertEqual(tuple(y.shape), (10, 3))

    def test_model_learns_toy_rule_and_outputs_bounded_vad(self):
        x, y = toy_examples(400, 23)
        model, metrics = fit(x, y, seed=23, epochs=120)
        self.assertLess(metrics["test_mse"], metrics["copy_current_baseline_mse"] * 0.15)
        self.assertTrue((model(x[:8, :3], x[:8, 3:]).abs() <= 1).all().item())
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 3), torch.zeros(1, 2))
        with torch.no_grad():
            current = torch.tensor([[0.2, 0.4, -0.3]])
            a = model(current, torch.tensor([[0.8, 0.7, 0.5]]))
            b = model(current, torch.tensor([[-0.8, -0.7, -0.5]]))
            self.assertGreater(torch.max(torch.abs(a - b)).item(), 0.2)


if __name__ == "__main__":
    unittest.main()
