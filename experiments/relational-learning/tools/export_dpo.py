#!/usr/bin/env python3
"""Export independently reviewed preference families to model-agnostic TRL DPO JSONL.

This is a data conversion gate, NOT a trainer or a claim of human review quality.
It deliberately refuses synthetic labels and non-training splits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from review import load_jsonl, save_jsonl
from validate import validate


def export(source: Path, output: Path, report: Path) -> dict:
    paths = {p.resolve() for p in [source, output, report]}
    if len(paths) != 3 or output.exists() or report.exists():
        raise ValueError("Source, output and report must be distinct; outputs must not exist")
    errors = validate([str(source)])
    if errors:
        raise ValueError("Invalid reviewed data: " + "; ".join(errors))
    rows = load_jsonl(source)
    if any(row["split"] != "train" for row in rows):
        raise ValueError("Only train-split data may be exported for DPO training")
    if any(row["review_status"] != "human_reviewed" or
           row["provenance"]["usage"] != "reviewed_training_candidate" for row in rows):
        raise ValueError("Only two-reviewer-approved records may be exported; synthetic labels prohibited")
    # Minimal TRL conversational preference schema. The context is exactly the
    # recorded conversation; analysis metadata and labels are never model input.
    output_rows = [{
        "prompt": row["context"],
        "chosen": [{"role": "assistant", "content": row["chosen"]}],
        "rejected": [{"role": "assistant", "content": row["rejected"]}],
    } for row in rows]
    save_jsonl(output, output_rows)
    summary = {
        "format": "trl-conversational-dpo-v1",
        "pairs": len(rows),
        "families": len({row["family_id"] for row in rows}),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "review_status": "two_reviewers_agreed_self_attested",
        "note": "Approval indicates recorded agreement, not independently verified reviewer identity or empirical generalization."
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Compiled reviewed training-only pairs")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(export(args.data, args.out, args.report), indent=2))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
