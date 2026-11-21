"""Minimal projector + LoRA trainer for local HF-format Qwen instruct checkpoints.

Not a full research training/evaluation pipeline. Never treats demo response
strings as a target unless separately authored and approved in a record.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .features import features, load_records


def format_example(tokenizer, record: dict, max_length: int) -> tuple[list[int], list[int]]:
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": "You are a conversational assistant."},
         {"role": "user", "content": record["user"]}],
        tokenize=True, add_generation_prompt=True,
    )
    completion = tokenizer.encode(record["assistant"], add_special_tokens=False)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must expose eos_token_id")
    completion.append(tokenizer.eos_token_id)
    ids = prompt + completion
    if len(ids) + 4 > max_length:
        raise ValueError(f"Example exceeds max_length={max_length}, refusing silent truncation")
    return ids, [-100] * len(prompt) + completion


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a continuous VAD soft prefix and LoRA adapter")
    parser.add_argument("--model", required=True, help="Local HF model directory or model ID, not a GGUF file")
    parser.add_argument("--data", required=True, help="Approved .json/.jsonl file or directory")
    parser.add_argument("--output", default="runs/affective-r0")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--allow-synthetic", action="store_true", help="Smoke testing only")
    args = parser.parse_args()
    if args.epochs < 1 or args.lr <= 0 or args.max_length < 32:
        parser.error("Invalid training hyperparameters")

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .model import AffectProjector, prefix_input

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    records = load_records(args.data, allow_synthetic=args.allow_synthetic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, trust_remote_code=False)
    base.config.use_cache = False
    lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                      target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    lm = get_peft_model(base, lora).to(device)
    lm.train()
    projector = AffectProjector(hidden_size=base.config.hidden_size).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in lm.parameters() if p.requires_grad] + list(projector.parameters()), lr=args.lr,
    )

    for epoch in range(args.epochs):
        random.shuffle(records)
        loss_total = 0.
        for record in records:
            ids, labels = format_example(tokenizer, record, args.max_length)
            token_ids = torch.tensor([ids], dtype=torch.long, device=device)
            targets = torch.tensor([labels], dtype=torch.long, device=device)
            cond = torch.tensor([features(record)], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            inputs = prefix_input(lm, projector, token_ids, cond, targets)
            outputs = lm(**inputs)
            if not torch.isfinite(outputs.loss):
                raise RuntimeError("Non-finite training loss; refusing to save checkpoint")
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in lm.parameters() if p.requires_grad] + list(projector.parameters()), 1.0,
            )
            optimizer.step()
            loss_total += outputs.loss.item()
        print(f"epoch {epoch+1}/{args.epochs}: mean_train_loss={loss_total/len(records):.4f}")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    lm.save_pretrained(output / "lora")
    torch.save({"config": {"hidden_size": projector.hidden_size,
                            "prefix_tokens": projector.prefix_tokens},
                "state_dict": projector.state_dict()}, output / "projector.pt")
    (output / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "base_model": args.model, "feature_dim": 9,
        "examples": len(records), "seed": args.seed, "epochs": args.epochs,
        "experimental": True,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Saved experimental adapter to {output}. Not evaluated for empathy or safety.")


if __name__ == "__main__":
    main()
