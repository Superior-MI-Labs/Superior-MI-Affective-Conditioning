#!/usr/bin/env python3
"""Optional LoRA-DPO training launcher; uses upstream TRL, not a second trainer.

No GPU/model dependencies are imported until after input/provenance checks pass.
Training is explicitly opt-in for synthetic data. Never runs in Pages CI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def check_inputs(data: Path, report: Path, output: Path, acknowledge_synthetic: bool) -> dict:
    if data.resolve() == report.resolve() or output.resolve() in {data.resolve(), report.resolve()}:
        raise ValueError('Training outputs must be distinct from source inputs')
    if output.exists():
        raise ValueError('Output directory already exists; use a fresh run directory')
    manifest = json.loads(report.read_text(encoding='utf-8'))
    if manifest.get('format') != 'trl-conversational-dpo-v1':
        raise ValueError('Not a supported DPO export manifest')
    if manifest.get('output_sha256') != hashlib.sha256(data.read_bytes()).hexdigest():
        raise ValueError('Training data does not match immutable export report SHA256')
    if manifest.get('prompt_mode') not in {'full', 'last_turn'}:
        raise ValueError('Missing prompt mode in export report')
    if manifest.get('label_origin') == 'synthetic':
        if not acknowledge_synthetic:
            raise ValueError('Synthetic labels require explicit --acknowledge-synthetic')
    elif manifest.get('label_origin') != 'reviewed':
        raise ValueError('Unknown preference label origin')
    lines = [json.loads(s) for s in data.read_text(encoding='utf-8').splitlines() if s.strip()]
    if len(lines) != manifest.get('pairs') or not lines:
        raise ValueError('DPO row count mismatch or empty data')
    for row in lines:
        if set(row) != {'prompt', 'chosen', 'rejected'} or not row['prompt'] or not row['chosen'] or not row['rejected']:
            raise ValueError('Malformed DPO training rows')
        if row['prompt'][-1].get('role') != 'user':
            raise ValueError('Last prompt turn must be from the user')
    return {'manifest': manifest, 'records': lines}


def train(args) -> dict:
    checked = check_inputs(args.data, args.report, args.out, args.acknowledge_synthetic)
    try:
        import torch
        import transformers
        import trl
        import peft
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoTokenizer
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:
        raise RuntimeError('Training dependencies missing: install torch transformers datasets peft accelerate trl') from exc
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = DPOConfig(
        output_dir=str(args.out),
        model_init_kwargs={'dtype': 'auto', **({'revision': args.revision} if args.revision else {})},
        learning_rate=args.learning_rate,
        beta=args.beta,
        max_length=args.max_length,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.accumulation,
        num_train_epochs=args.epochs,
        save_strategy='no', logging_steps=1,
        report_to='none', seed=args.seed,
        optim='adamw_torch',
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=False,
    )
    adapter = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                         task_type='CAUSAL_LM', target_modules='all-linear')
    trainer = DPOTrainer(model=args.model, args=config,
                         processing_class=tokenizer,
                         train_dataset=Dataset.from_list(checked['records']),
                         peft_config=adapter)
    resolved_revision = getattr(trainer.model.config, '_commit_hash', None)
    trainer.train()
    trainer.save_model(str(args.out))
    metadata = {
        'model_id': args.model, 'requested_revision': args.revision,
        'resolved_model_commit': resolved_revision,
        'training_type': 'DPO+LoRA',
        'dataset_sha256': checked['manifest']['output_sha256'],
        'label_origin': checked['manifest']['label_origin'],
        'prompt_mode': checked['manifest']['prompt_mode'],
        'epochs': args.epochs, 'seed': args.seed, 'beta': args.beta,
        'learning_rate': args.learning_rate, 'max_length': args.max_length,
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'trl': trl.__version__, 'peft': peft.__version__,
        'claim': 'Parameter update executed. Not evidence of empathy or subjective emotion.'
    }
    (args.out/'run.json').write_text(json.dumps(metadata, indent=2, sort_keys=True)+'\n',encoding='utf-8')
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model', default='Qwen/Qwen2.5-0.5B-Instruct')
    parser.add_argument('--revision', default=None, help='Optional immutable Hugging Face commit to pin the checkpoint')
    parser.add_argument('--acknowledge-synthetic', action='store_true')
    parser.add_argument('--epochs', type=float, default=2.0)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--beta', type=float, default=.1)
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--accumulation', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.epochs <= 0 or args.beta <= 0 or args.accumulation < 1 or args.max_length < 64:
        parser.error('Invalid numeric training configuration')
    try:
        print(json.dumps(train(args), indent=2))
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
