#!/usr/bin/env python3
"""Convert validated relational comparisons to conversational TRL DPO JSONL.

Default: strictly two-rater-reviewed training rows. Synthetic research is an
explicit, separate mode using the SAME source schema and validation pipeline.
No conversion implies a behavioral or psychological finding.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from review import load_jsonl, save_jsonl
from validate import validate


def export_many(sources: list[Path], output: Path, report: Path, *, mode: str = 'reviewed',
                prompt_mode: str = 'full') -> dict:
    if mode not in {'reviewed', 'synthetic'}:
        raise ValueError('Invalid mode')
    if prompt_mode not in {'full', 'last_turn'}:
        raise ValueError('Invalid prompt mode')
    if not sources:
        raise ValueError('At least one source file is required')
    inputs = [p.resolve() for p in sources]
    paths = [*inputs, output.resolve(), report.resolve()]
    if len(set(paths)) != len(paths) or output.exists() or report.exists():
        raise ValueError('Source, output and report must be distinct; outputs must not exist')
    errors = validate([str(p) for p in sources])
    if errors:
        raise ValueError('Invalid preference data: ' + '; '.join(errors))
    rows = [row for p in sources for row in load_jsonl(p)]
    if any(row['split'] != 'train' for row in rows):
        raise ValueError('Only train-split data may be exported for DPO training')
    if mode == 'reviewed':
        if any(row['review_status'] != 'human_reviewed' or
               row['provenance']['usage'] != 'reviewed_training_candidate' for row in rows):
            raise ValueError('Only two-reviewer-approved records may be exported; synthetic labels prohibited')
    else:
        if any(row['review_status'] != 'synthetic_unreviewed' or
               row['provenance']['consent'] != 'not_applicable_synthetic' or
               row['provenance']['usage'] not in {'demonstration_only', 'review_candidate_only', 'candidate_review_only'}
               for row in rows):
            raise ValueError('Synthetic mode requires original unreviewed synthetic candidates, not human-reviewed/disputed records')
    data = [{
        'prompt': row['context'] if prompt_mode == 'full' else [row['context'][-1]],
        'chosen': [{'role': 'assistant', 'content': row['chosen']}],
        'rejected': [{'role': 'assistant', 'content': row['rejected']}]
    } for row in rows]
    save_jsonl(output, data)
    provenance = 'two_reviewers_agreed_self_attested' if mode == 'reviewed' else 'synthetic_author_preference_unverified'
    summary = {
        'format': 'trl-conversational-dpo-v1',
        'pairs': len(rows), 'families': len({r['family_id'] for r in rows}),
        'source_sha256': hashlib.sha256(b''.join(p.read_bytes() for p in sources)).hexdigest(),
        'source_files_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'output_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
        'review_status': provenance,
        'label_origin': mode, 'prompt_mode': prompt_mode,
        'note': ('Synthetic author-defined preferences; strictly a controlled learning experiment, '
                 'not empirical proof of empathy.' if mode == 'synthetic' else
                 'Recorded reviewer agreement is self-attested, not independently identity-verified.')
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    return summary


def export(source: Path, output: Path, report: Path) -> dict:
    """Preserve the original reviewed-only API used by existing callers/tests."""
    return export_many([source], output, report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', nargs='+', required=True, type=Path, help='Complete train-split family JSONL files')
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--mode', choices=['reviewed', 'synthetic'], default='reviewed')
    parser.add_argument('--prompt-mode', choices=['full', 'last_turn'], default='full')
    args = parser.parse_args()
    try:
        print(json.dumps(export_many(args.data, args.out, args.report,
                                     mode=args.mode, prompt_mode=args.prompt_mode), indent=2))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
