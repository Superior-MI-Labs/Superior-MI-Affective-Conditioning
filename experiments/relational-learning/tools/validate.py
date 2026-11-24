#!/usr/bin/env python3
"""Validate relational preference-pair data without third-party dependencies."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

FIELDS = {
    "schema_version", "id", "family_id", "split", "category", "variant",
    "context", "chosen", "rejected", "review_status", "provenance", "review_notes",
}
SPLITS = {"train", "dev", "test"}
STATUSES = {"synthetic_unreviewed", "human_reviewed", "disputed"}


def valid_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate(paths: list[str]) -> list[str]:
    errors: list[str] = []
    identifiers: set[str] = set()
    families: dict[str, list[dict]] = defaultdict(list)
    family_splits: dict[str, str] = {}

    for path in paths:
        try:
            lines = Path(path).read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError) as exc:
            errors.append(f"{path}: cannot read: {exc}")
            continue
        for line_num, line in enumerate(lines, 1):
            if not line.strip():
                continue
            where = f"{path}:{line_num}"
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{where}: invalid JSON: {exc}")
                continue
            if not isinstance(row, dict) or set(row) != FIELDS:
                errors.append(f"{where}: missing or unexpected fields")
                continue

            bad = []
            if row["schema_version"] != "1.0":
                bad.append("schema version")
            if not all(valid_string(row[k]) for k in ("id", "family_id", "category", "chosen", "rejected")):
                bad.append("required strings")
            if not isinstance(row["split"], str) or row["split"] not in SPLITS:
                bad.append("split")
            if not isinstance(row["review_status"], str) or row["review_status"] not in STATUSES:
                bad.append("review status")
            if type(row["variant"]) is not int or row["variant"] < 1:
                bad.append("variant")
            context = row["context"]
            if (
                not isinstance(context, list) or not context
                or any(
                    not isinstance(turn, dict) or set(turn) != {"role", "content"}
                    or turn["role"] not in {"user", "assistant"}
                    or not valid_string(turn["content"])
                    for turn in context
                )
                or (isinstance(context[-1], dict) and context[-1].get("role") != "user")
            ):
                bad.append("context")
            if valid_string(row["chosen"]) and valid_string(row["rejected"]) and row["chosen"].strip() == row["rejected"].strip():
                bad.append("identical chosen and rejected")
            if not isinstance(row["review_notes"], str):
                bad.append("review notes")
            source = row["provenance"]
            if not isinstance(source, dict) or set(source) != {"source", "consent", "usage"} or not all(valid_string(v) for v in source.values()):
                bad.append("provenance")
            if bad:
                errors.append(f"{where}: invalid {', '.join(bad)}")
                continue

            if row["id"] in identifiers:
                errors.append(f"{where}: duplicate id {row['id']}")
            identifiers.add(row["id"])
            old_split = family_splits.setdefault(row["family_id"], row["split"])
            if old_split != row["split"]:
                errors.append(f"{where}: family {row['family_id']} leaks across {old_split}/{row['split']}")
            families[row["family_id"]].append(row)

    for family, rows in families.items():
        if len(rows) != 2:
            errors.append(f"family {family}: expected exactly two counterfactual variants")
            continue
        first, second = rows
        if first["variant"] == second["variant"]:
            errors.append(f"family {family}: duplicate variant number")
        if first["chosen"] != second["rejected"] or first["rejected"] != second["chosen"]:
            errors.append(f"family {family}: candidate responses must be reused with reversed preference")
        if first["context"][-1]["content"] != second["context"][-1]["content"]:
            errors.append(f"family {family}: variants must share identical final user message")
        if first["context"][:-1] == second["context"][:-1]:
            errors.append(f"family {family}: contexts must differ before final user message")
    if not identifiers:
        errors.append("no valid records found")
    return errors


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python3 tools/validate.py data/*.jsonl", file=sys.stderr)
        return 2
    errors = validate(sys.argv[1:])
    for error in errors:
        print("ERROR:", error, file=sys.stderr)
    if errors:
        return 1
    print("PASS: preference records, family isolation, counterfactual pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
