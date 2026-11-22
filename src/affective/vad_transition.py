"""Minimal supervised VAD transition experiment. No language model or emotion rules.

Run: python -m affective.vad_transition train --toy
     python -m affective.vad_transition predict --checkpoint runs/vad-transition.pt \
       --current 0.2 0.4 -0.3 --observed 0.8 0.7 0.5
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn


class VADTransition(nn.Module):
    """6 inputs -> 16 hidden neurons -> 3 bounded VAD outputs."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 16), nn.Tanh(), nn.Linear(16, 3), nn.Tanh()
        )

    def forward(self, current: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        if current.shape != observed.shape or current.shape[-1] != 3:
            raise ValueError("current and observed must have matching [..., 3] shapes")
        return self.net(torch.cat((current, observed), dim=-1))


def vad(value: object, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must be an array of 3 numbers")
    if any(isinstance(x, bool) or not isinstance(x, (int, float))
           or not math.isfinite(x) or not -1 <= x <= 1 for x in value):
        raise ValueError(f"{name} must contain finite numbers between -1 and 1")
    return [float(x) for x in value]


def load_examples(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Human-labeled examples only; synthetic data uses explicit --toy flag."""
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            if not isinstance(obj, dict) or obj.get("source") != "human_reviewed":
                raise ValueError("source must be human_reviewed")
            current = vad(obj.get("current"), "current")
            observed = vad(obj.get("observed"), "observed")
            target = vad(obj.get("next"), "next")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
        rows.append((current + observed, target))
    if len(rows) < 10:
        raise ValueError("At least 10 reviewed transitions are required for this smoke experiment")
    return (torch.tensor([x for x, _ in rows], dtype=torch.float32),
            torch.tensor([y for _, y in rows], dtype=torch.float32))


def toy_examples(count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """An artificial known rule, solely for proving the training pipeline works."""
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.rand((count, 6), generator=generator) * 2 - 1
    targets = 0.65 * inputs[:, :3] + 0.35 * inputs[:, 3:]
    return inputs, targets


def fit(inputs: torch.Tensor, targets: torch.Tensor, *, seed: int = 17,
        epochs: int = 200) -> tuple[VADTransition, dict]:
    if inputs.ndim != 2 or inputs.shape[1] != 6 or targets.shape != (len(inputs), 3):
        raise ValueError("Expected training shapes [N, 6] and [N, 3]")
    if epochs < 1 or len(inputs) < 10:
        raise ValueError("Expected at least 10 examples and 1 epoch")
    if not torch.isfinite(inputs).all() or not torch.isfinite(targets).all():
        raise ValueError("Non-finite training values")
    if (inputs.abs() > 1).any() or (targets.abs() > 1).any():
        raise ValueError("VAD values must stay in [-1, 1]")
    torch.manual_seed(seed)
    permutation = torch.randperm(len(inputs))
    n_test = max(1, len(inputs) // 5)
    test_ids, train_ids = permutation[:n_test], permutation[n_test:]
    x_train, y_train = inputs[train_ids], targets[train_ids]
    x_test, y_test = inputs[test_ids], targets[test_ids]
    model = VADTransition()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x_train[:, :3], x_train[:, 3:])
        loss = nn.functional.mse_loss(prediction, y_train)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        learned = model(x_test[:, :3], x_test[:, 3:])
        test_mse = nn.functional.mse_loss(learned, y_test).item()
        baseline_mse = nn.functional.mse_loss(x_test[:, :3], y_test).item()
        train_mse = nn.functional.mse_loss(
            model(x_train[:, :3], x_train[:, 3:]), y_train).item()
    return model, {"train_mse": train_mse, "test_mse": test_mse,
                   "copy_current_baseline_mse": baseline_mse,
                   "train_examples": len(train_ids), "test_examples": len(test_ids)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Scratch 6->16->3 VAD transition experiment")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    source = train.add_mutually_exclusive_group(required=True)
    source.add_argument("--toy", action="store_true", help="Use an artificial rule to smoke-test training")
    source.add_argument("--data", type=Path, help="Reviewed .jsonl transition records")
    train.add_argument("--epochs", type=int, default=200)
    train.add_argument("--seed", type=int, default=17)
    train.add_argument("--out", type=Path, default=Path("runs/vad-transition.pt"))
    predict = sub.add_parser("predict")
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--current", type=float, nargs=3, required=True)
    predict.add_argument("--observed", type=float, nargs=3, required=True)
    args = parser.parse_args()
    if args.command == "train":
        if args.toy:
            inputs, targets = toy_examples(1000, args.seed)
        else:
            inputs, targets = load_examples(args.data)
        model, metrics = fit(inputs, targets, seed=args.seed, epochs=args.epochs)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"schema_version": 1, "state_dict": model.state_dict(),
                    "training_source": "toy" if args.toy else "human_reviewed"}, args.out)
        print(json.dumps({"checkpoint": str(args.out), "source": "toy" if args.toy else "human_reviewed",
                          **metrics}, indent=2))
    else:
        current = vad(args.current, "current")
        observed = vad(args.observed, "observed")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if checkpoint.get("schema_version") != 1:
            raise ValueError("Unsupported VAD checkpoint format")
        model = VADTransition()
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        with torch.no_grad():
            prediction = model(torch.tensor([current]), torch.tensor([observed]))[0].tolist()
        print(json.dumps({"current": current, "observed": observed, "predicted_next": prediction,
                          "state_delta": [round(p - c, 6) for p, c in zip(prediction, current)],
                          "trained_on": checkpoint.get("training_source")}, indent=2))


if __name__ == "__main__":
    main()
