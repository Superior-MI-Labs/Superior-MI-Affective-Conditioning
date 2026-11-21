"""One-shot local inference from an already trained affective adapter.

Mood and phase are supplied explicitly; state changes remain owned by the
research controller/browser, not by a second transition implementation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .features import features


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a trained affective adapter on a Qwen checkpoint")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--model", required=True, help="Same HF-format base as the training run")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--mode", choices=["vector", "phase"], default="vector")
    parser.add_argument("--ai-vad", nargs=3, type=float, required=True, metavar=("V", "A", "D"),
                        help="Post-transition internal VAD")
    parser.add_argument("--message-vad", nargs=3, type=float, required=True, metavar=("V", "A", "D"))
    parser.add_argument("--phase", type=float, default=0.0, help="Post-transition phase (radians)")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    args = parser.parse_args()
    if not (1 <= args.max_new_tokens <= 2048):
        parser.error("max-new-tokens must be between 1 and 2048")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .model import AffectProjector, prefix_input

    adapter = Path(args.adapter)
    manifest = json.loads((adapter / "manifest.json").read_text(encoding="utf-8"))
    if args.model != manifest["base_model"]:
        parser.error("Model reference differs from the training manifest; use the same model")
    record = {"schema_version": 1, "mode": args.mode,
              "state_after_vad": args.ai_vad, "observed_vad": args.message_vad,
              "phase_after": args.phase}
    conditioning = features(record)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, trust_remote_code=False)
    lm = PeftModel.from_pretrained(base, adapter / "lora").to(device)
    lm.eval()
    saved = torch.load(adapter / "projector.pt", map_location="cpu", weights_only=True)
    projector = AffectProjector(**saved["config"]).to(device)
    projector.load_state_dict(saved["state_dict"])
    projector.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "system", "content": "You are a conversational assistant."},
         {"role": "user", "content": args.prompt}],
        tokenize=True, add_generation_prompt=True,
    )
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    features_tensor = torch.tensor([conditioning], dtype=torch.float32, device=device)
    with torch.no_grad():
        packed = prefix_input(lm, projector, ids, features_tensor)
        outputs = lm(**packed, use_cache=True)
        cache = outputs.past_key_values
        mask = packed["attention_mask"]
        logits = outputs.logits[:, -1, :]
        generated = []
        for _ in range(args.max_new_tokens):
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
            token = int(next_id.item())
            if token == tokenizer.eos_token_id:
                break
            generated.append(token)
            mask = torch.cat([mask, torch.ones_like(next_id)], dim=1)
            outputs = lm(input_ids=next_id, past_key_values=cache, attention_mask=mask,
                         use_cache=True)
            cache = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
    print(tokenizer.decode(generated, skip_special_tokens=True))


if __name__ == "__main__":
    main()
