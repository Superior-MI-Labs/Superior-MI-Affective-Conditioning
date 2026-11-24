#!/usr/bin/env python3
"""Blinded two-rater review and conservative training-candidate export.

No training, network calls, or hidden use of original preference labels by raters.
The initial synthetic judgments remain unverified until independent human review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

from validate import validate

CHOICES = {"A", "B", "tie", "both_unacceptable"}


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{n}: invalid JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{n}: expected object")
        rows.append(record)
    if not rows:
        raise ValueError(f"{path}: no records")
    return rows


def save_jsonl(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise ValueError(f"Output already exists: {path}; use a new review batch")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def seed_rows(source: Path) -> list[dict]:
    errors = validate([str(source)])
    if errors:
        raise ValueError("Invalid source dataset:\n" + "\n".join(errors))
    rows = load_jsonl(source)
    if any(row["split"] != "train" for row in rows):
        raise ValueError("Only train-split rows can enter this review/export path")
    if any(row["review_status"] != "synthetic_unreviewed" for row in rows):
        raise ValueError("Input must contain unreviewed synthetic candidates only")
    return rows


def task_key(row: dict) -> str:
    """Opaque stable ID; reviewers do not receive scenario-family names."""
    return hashlib.sha256(("relational-task:" + row["id"]).encode("utf-8")).hexdigest()[:24]


def make_tasks(rows: list[dict], seed: str) -> list[dict]:
    tasks = []
    for row in rows:
        # Stable, independent of row insertion order. No preference metadata is exported.
        flip = int(hashlib.sha256((seed + ":" + row["id"]).encode()).hexdigest(), 16) & 1
        a, b = (row["chosen"], row["rejected"]) if flip else (row["rejected"], row["chosen"])
        tasks.append({
            "task_id": task_key(row), "source_sha256": digest(row),
            "context": row["context"], "response_a": a, "response_b": b,
        })
    tasks.sort(key=lambda task: hashlib.sha256((seed + ":order:" + task["task_id"]).encode()).hexdigest())
    return tasks


def prepare(source: Path, output: Path, seed: str) -> dict:
    rows = seed_rows(source)
    tasks = make_tasks(rows, seed)
    save_jsonl(output, tasks)
    return {"tasks": len(tasks), "families": len({r['family_id'] for r in rows}),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "tasks_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}


def checked_tasks(rows: list[dict], tasks_path: Path) -> dict[str, dict]:
    original = {task_key(row): row for row in rows}
    tasks = load_jsonl(tasks_path)
    task_map = {}
    for task in tasks:
        if set(task) != {"task_id", "source_sha256", "context", "response_a", "response_b"}:
            raise ValueError("Task file has unexpected fields")
        key = task["task_id"]
        if not isinstance(key, str) or key in task_map or key not in original:
            raise ValueError("Task file contains duplicate or unknown task ID")
        row = original[key]
        if task["source_sha256"] != digest(row):
            raise ValueError(f"Source record changed after task preparation: {key}")
        if task["context"] != row["context"]:
            raise ValueError(f"Task context changed: {key}")
        if {task["response_a"], task["response_b"]} != {row["chosen"], row["rejected"]}:
            raise ValueError(f"Task candidates changed: {key}")
        task_map[key] = task
    if set(task_map) != set(original):
        raise ValueError("Task inventory does not match source dataset")
    return task_map


def compile_reviews(source: Path, tasks_path: Path, ratings_paths: list[Path], output: Path, report: Path) -> dict:
    existing_inputs = {x.resolve() for x in [source, tasks_path, *ratings_paths]}
    if output.resolve() == report.resolve() or {output.resolve(), report.resolve()} & existing_inputs:
        raise ValueError("Output or report must not overwrite source, tasks, or ratings")
    if output.exists() or report.exists():
        raise ValueError("Output or report already exists; use a new review batch")
    rows = seed_rows(source)
    task_map = checked_tasks(rows, tasks_path)
    votes: dict[str, dict[str, str]] = defaultdict(dict)
    for path in ratings_paths:
        for rating in load_jsonl(path):
            if set(rating) not in ({"task_id", "reviewer_id", "choice"}, {"task_id", "reviewer_id", "choice", "notes"}):
                raise ValueError(f"{path}: rating must contain task_id, reviewer_id, choice, and optional notes")
            task_id, reviewer, choice = rating["task_id"], rating["reviewer_id"], rating["choice"]
            if task_id not in task_map or not isinstance(reviewer, str) or not reviewer.strip() or choice not in CHOICES:
                raise ValueError(f"{path}: unknown task, invalid reviewer, or invalid choice")
            if "notes" in rating and not isinstance(rating["notes"], str):
                raise ValueError(f"{path}: notes must be text")
            if reviewer in votes[task_id] or len(votes[task_id]) >= 2:
                raise ValueError(f"{task_id}: duplicate reviewer or more than two ratings")
            votes[task_id][reviewer] = choice
    decisions = {}
    for task_id, task in task_map.items():
        judgments = list(votes[task_id].values())
        if len(judgments) != 2:
            decisions[task_id] = ("pending", None)
        elif judgments[0] == judgments[1] and judgments[0] in {"A", "B"}:
            decisions[task_id] = ("agreed", task["response_a"] if judgments[0] == "A" else task["response_b"])
        else:
            decisions[task_id] = ("disputed", None)
    by_family: dict[str, list[dict]] = defaultdict(list)
    originals_by_id = {row["id"]: row for row in rows}
    for row in rows:
        by_family[row["family_id"]].append(row)
    accepted = []
    accepted_families = set()
    for family, members in by_family.items():
        if not all(decisions[task_key(row)][0] == "agreed" for row in members):
            continue
        first, second = members
        first_choice, second_choice = (decisions[task_key(r)][1] for r in members)
        if first_choice == second_choice:
            # Same pair with reversed context must actually reverse preference.
            continue
        accepted_families.add(family)
        for row in members:
            choice = decisions[task_key(row)][1]
            other = row["rejected"] if choice == row["chosen"] else row["chosen"]
            approved = dict(row)
            approved["chosen"] = choice
            approved["rejected"] = other
            approved["review_status"] = "human_reviewed"
            approved["review_notes"] = "Two blinded ratings agreed; consult private review record."
            approved["provenance"] = {**row["provenance"], "usage": "reviewed_training_candidate"}
            accepted.append(approved)
    if accepted:
        # Reuse the project's existing authoritative structural and family validator.
        tmp = output.with_name(output.name + ".validation.tmp")
        try:
            tmp.write_text("".join(json.dumps(r) + "\n" for r in accepted), encoding="utf-8")
            errors = validate([str(tmp)])
        finally:
            tmp.unlink(missing_ok=True)
        if errors:
            raise ValueError("Approved output failed validation: " + "; ".join(errors))
    summary = {
        "schema": "relational-review-1.0", "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "tasks_sha256": hashlib.sha256(tasks_path.read_bytes()).hexdigest(),
        "ratings_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ratings_paths},
        "task_count": len(rows), "rated_count": sum(len(v) == 2 for v in votes.values()),
        "agreed_rows": sum(x[0] == "agreed" for x in decisions.values()),
        "pending_rows": sum(x[0] == "pending" for x in decisions.values()),
        "disputed_rows": sum(x[0] == "disputed" for x in decisions.values()),
        "approved_families": len(accepted_families), "approved_rows": len(accepted),
        "label_reversals_from_author_draft": sum(r["chosen"] != originals_by_id[r["id"]]["chosen"] for r in accepted),
        "warning": "Reviewer identities and independence are self-attested. This script cannot verify human participation.",
    }
    save_jsonl(output, accepted) if accepted else save_jsonl(output, [])
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Produce preference-blind review tasks from train candidates")
    prep.add_argument("--data", required=True, type=Path)
    prep.add_argument("--out", required=True, type=Path)
    prep.add_argument("--seed", default="relational-pilot-1")
    comp = sub.add_parser("compile", help="Accept only complete, two-rater-agreed counterfactual families")
    comp.add_argument("--data", required=True, type=Path)
    comp.add_argument("--tasks", required=True, type=Path)
    comp.add_argument("--ratings", nargs="+", required=True, type=Path)
    comp.add_argument("--out", required=True, type=Path)
    comp.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            result = prepare(args.data, args.out, args.seed)
        else:
            result = compile_reviews(args.data, args.tasks, args.ratings, args.out, args.report)
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
