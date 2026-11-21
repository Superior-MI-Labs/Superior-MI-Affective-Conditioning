"""Stable R0 conditioning contract. No emotion prediction is performed here."""
from __future__ import annotations

import json
import math
from pathlib import Path

FEATURE_DIM = 9


def vad(value: object) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("VAD must be a list of three numbers")
    if any(isinstance(x, bool) or not isinstance(x, (int, float))
           or not math.isfinite(x) or abs(x) > 1 for x in value):
        raise ValueError("VAD coordinates must be finite numbers between -1 and +1")
    return [float(x) for x in value]


def features(record: dict) -> list[float]:
    """Model features: updated VAD, observed VAD, mode bit, sin/cos(phase)."""
    if record.get("schema_version") != 1:
        raise ValueError("Unsupported example schema_version")
    internal = vad(record.get("state_after_vad"))
    observed = vad(record.get("observed_vad"))
    mode = record.get("mode")
    if mode not in ("vector", "phase"):
        raise ValueError("Unknown mode")
    phase = record.get("phase_after", 0.0)
    if isinstance(phase, bool) or not isinstance(phase, (float, int)) or not math.isfinite(phase):
        raise ValueError("Phase must be a finite number")
    phase_features = [math.sin(phase), math.cos(phase)] if mode == "phase" else [0., 0.]
    return internal + observed + [1. if mode == "phase" else 0.] + phase_features


def load_records(path: str, allow_synthetic: bool = False) -> list[dict]:
    """Accept exported .json files or an aggregated .jsonl file/directory."""
    source = Path(path)
    paths = sorted(p for p in source.iterdir() if p.suffix in (".json", ".jsonl")) if source.is_dir() else [source]
    if not paths:
        raise ValueError("No example files found")
    records = []
    for candidate in paths:
        if candidate.suffix == ".jsonl":
            with candidate.open(encoding="utf-8") as stream:
                parsed = [json.loads(line) for line in stream if line.strip()]
        elif candidate.suffix == ".json":
            parsed_file = json.loads(candidate.read_text(encoding="utf-8"))
            parsed = parsed_file if isinstance(parsed_file, list) else [parsed_file]
        else:
            raise ValueError("Expected .json or .jsonl")
        for item in parsed:
            if item.get("approved") is not True and not allow_synthetic:
                raise ValueError(f"Unapproved training example in {candidate}")
            for key in ("user", "assistant"):
                if not isinstance(item.get(key), str) or not item[key].strip():
                    raise ValueError(f"Example requires nonempty '{key}'")
            features(item)
            records.append(item)
    if not records:
        raise ValueError("Dataset contains no examples")
    return records
