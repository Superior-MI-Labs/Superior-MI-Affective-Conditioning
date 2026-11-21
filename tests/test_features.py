import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from affective.features import features, load_records, vad


def record(**updates):
    value = {"schema_version": 1, "mode": "vector", "state_after_vad": [0.1, 0.2, 0.3],
             "observed_vad": [-0.1, -0.2, 0.5], "phase_after": 0.2,
             "user": "Hello", "assistant": "Hi", "approved": True}
    value.update(updates)
    return value


class FeaturesTests(unittest.TestCase):
    def test_vector_has_neutral_phase_features(self):
        self.assertEqual(features(record()), [0.1, 0.2, 0.3, -0.1, -0.2, 0.5, 0., 0., 0.])

    def test_phase_encodes_angle(self):
        values = features(record(mode="phase", phase_after=0.0))
        self.assertEqual(values[-3:], [1., 0., 1.])

    def test_bad_values_fail(self):
        for bad in ([0, 2, 0], [0, float("nan"), 0], [False, 0, 0], [0, 0]):
            with self.assertRaises(ValueError):
                vad(bad)
        with self.assertRaises(ValueError):
            features(record(mode="unknown"))
        with self.assertRaises(ValueError):
            features(record(phase_after=float('inf')))

    def test_only_approved_examples_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'example.json'
            path.write_text(json.dumps(record(approved=False)))
            with self.assertRaises(ValueError):
                load_records(str(path))
            self.assertEqual(len(load_records(str(path), allow_synthetic=True)), 1)
            path.write_text(json.dumps(record()))
            self.assertEqual(len(load_records(str(path))), 1)


if __name__ == '__main__':
    unittest.main()
