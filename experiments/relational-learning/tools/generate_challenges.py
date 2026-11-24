#!/usr/bin/env python3
"""Compile hand-authored counterfactual challenge families into the v1 pair schema.

This is a deterministic data build, not a generative model. The JSON source is
reviewable and authoritative; generated JSONL rows must be regenerated, not edited.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source" / "challenges.json"
SPLITS = ("train", "dev", "test")
REQUIRED = {"id", "split", "category", "antecedents", "assistant_bridge",
            "last_user", "response_1", "response_2", "review_caution"}


def generate(source: Path = SOURCE) -> dict[str, list[dict]]:
    entries = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        raise ValueError("Source must contain a nonempty list")
    seen = set()
    result: dict[str, list[dict]] = {split: [] for split in SPLITS}
    for item in entries:
        if not isinstance(item, dict) or set(item) != REQUIRED:
            raise ValueError("Challenge family has missing or additional fields")
        if item["split"] not in SPLITS:
            raise ValueError(f"Unsupported split: {item['split']}")
        if not isinstance(item["id"], str) or not item["id"].strip() or item["id"] in seen:
            raise ValueError("Duplicate or missing family id")
        seen.add(item["id"])
        if not isinstance(item["antecedents"], list) or len(item["antecedents"]) != 2:
            raise ValueError(f"Expected two antecedents in {item['id']}")
        for key in ["category", "assistant_bridge", "last_user", "response_1", "response_2", "review_caution"]:
            if not isinstance(item[key], str) or not item[key].strip():
                raise ValueError(f"Blank {key} in {item['id']}")
        if item["response_1"] == item["response_2"] or item["antecedents"][0] == item["antecedents"][1]:
            raise ValueError(f"No contrast in family {item['id']}")
        for i, preceding in enumerate(item["antecedents"], 1):
            if not isinstance(preceding, str) or not preceding.strip():
                raise ValueError(f"Invalid antecedent in family {item['id']}")
            row = {
                "schema_version": "1.0", "id": f"challenge_{item['id']}_{i:02d}",
                "family_id": f"challenge_{item['id']}", "split": item["split"],
                "category": item["category"], "variant": i,
                "context": [{"role": "user", "content": preceding},
                            {"role": "assistant", "content": item["assistant_bridge"]},
                            {"role": "user", "content": item["last_user"]}],
                "chosen": item["response_1"] if i == 1 else item["response_2"],
                "rejected": item["response_2"] if i == 1 else item["response_1"],
                "review_status": "synthetic_unreviewed",
                "provenance": {"source": "project_authored_challenge_v1",
                               "consent": "not_applicable_synthetic",
                               "usage": "review_candidate_only"},
                "review_notes": "Authored preference, not a human judgment. " + item["review_caution"],
            }
            result[item["split"]].append(row)
    return result


def encode(rows: list[dict]) -> bytes:
    return ("".join(json.dumps(x, ensure_ascii=False, separators=(",", ":")) + "\n" for x in rows)).encode("utf-8")


def write_files(out_dir: Path, check: bool = False) -> dict[str, int]:
    datasets = generate()
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in datasets.items():
        path = out_dir / f"{split}_challenge.jsonl"
        payload = encode(rows)
        if check:
            if not path.exists() or path.read_bytes() != payload:
                raise ValueError(f"Generated file is stale: {path}")
        else:
            path.write_bytes(payload)
    return {split: len(rows) for split, rows in datasets.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify checked-in data matches source")
    args = parser.parse_args()
    try:
        print(json.dumps(write_files(ROOT / "data", check=args.check), sort_keys=True))
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
