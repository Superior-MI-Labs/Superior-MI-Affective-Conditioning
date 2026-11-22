"""One supervised VAD transition network: [current(3), observed(3)] -> next(3).

No emotion rules, pretrained language models, or alternate dynamics engines.
Synthetic targets are a software sanity check, not evidence of empathy.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn


class VADNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, 16), nn.Tanh(), nn.Linear(16, 3), nn.Tanh())

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[-1] != 6:
            raise ValueError("Expected last tensor dimension to be six VAD coordinates")
        return self.net(values)


def check_vad(value: object, field: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{field} must contain three numbers")
    if any(isinstance(n, bool) or not isinstance(n, (int, float))
           or not math.isfinite(n) or abs(n) > 1 for n in value):
        raise ValueError(f"{field} requires finite values in [-1, 1]")
    return [float(v) for v in value]


def load_reviewed(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            if not isinstance(item, dict) or item.get("source") != "human_reviewed":
                raise ValueError("source must be 'human_reviewed'")
            current = check_vad(item.get("current"), "current")
            observed = check_vad(item.get("observed"), "observed")
            target = check_vad(item.get("next"), "next")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{path}:{number}: {exc}") from exc
        rows.append((current + observed, target))
    if len(rows) < 10:
        raise ValueError("At least 10 reviewed examples are needed for this initial experiment")
    return (torch.tensor([a for a, _ in rows], dtype=torch.float32),
            torch.tensor([b for _, b in rows], dtype=torch.float32))


def synthetic_examples(count: int = 1000, seed: int = 17):
    """Known weighted-average toy rule ONLY to test learning and export."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.rand((count, 6), generator=generator) * 2 - 1
    y = 0.65 * x[:, :3] + 0.35 * x[:, 3:]
    return x, y


def train(x: torch.Tensor, y: torch.Tensor, *, epochs: int = 200, seed: int = 17):
    if x.ndim != 2 or x.shape[1] != 6 or y.shape != (len(x), 3):
        raise ValueError("Expected [N, 6] inputs and [N, 3] targets")
    if len(x) < 10 or epochs < 1:
        raise ValueError("Need at least 10 examples and one epoch")
    if not torch.isfinite(x).all() or not torch.isfinite(y).all() or (x.abs() > 1).any() or (y.abs() > 1).any():
        raise ValueError("All VAD coordinates must be finite and in [-1, 1]")
    torch.manual_seed(seed)
    indices = torch.randperm(len(x))
    n_test = max(1, len(x) // 5)
    tr, te = indices[n_test:], indices[:n_test]
    network = VADNetwork()
    optim = torch.optim.Adam(network.parameters(), lr=0.01)
    network.train()
    for _ in range(epochs):
        optim.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(network(x[tr]), y[tr])
        loss.backward()
        optim.step()
    network.eval()
    with torch.no_grad():
        metrics = {
            "train_mse": round(nn.functional.mse_loss(network(x[tr]), y[tr]).item(), 8),
            "test_mse": round(nn.functional.mse_loss(network(x[te]), y[te]).item(), 8),
            "copy_state_baseline_mse": round(nn.functional.mse_loss(x[te, :3], y[te]).item(), 8),
            "train_examples": len(tr),
            "test_examples": len(te),
        }
    return network, metrics


def web_artifact(network: VADNetwork, source: str, metrics: dict) -> dict:
    """Single-source export of PyTorch weights for read-only browser inference."""
    first, second = network.net[0], network.net[2]
    probe_input = [0.2, 0.4, -0.3, 0.8, 0.7, 0.5]
    with torch.no_grad():
        probe_output = network(torch.tensor([probe_input], dtype=torch.float32))[0].tolist()
    return {
        "schema_version": 1,
        "architecture": [6, 16, 3],
        "activation": "tanh",
        "source": source,
        "metrics": metrics,
        "layers": [
            {"weights": first.weight.detach().tolist(), "bias": first.bias.detach().tolist()},
            {"weights": second.weight.detach().tolist(), "bias": second.bias.detach().tolist()},
        ],
        "probe": {"input": probe_input, "output": probe_output},
    }


def save_model(path: Path, model: VADNetwork, source: str, metrics: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": 1, "state_dict": model.state_dict(),
                "source": source, "metrics": metrics}, path)


def restore_model(path: Path):
    record = torch.load(path, map_location="cpu", weights_only=True)
    if record.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint version")
    model = VADNetwork()
    model.load_state_dict(record["state_dict"], strict=True)
    model.eval()
    return model, record


def main():
    parser = argparse.ArgumentParser(description="Scratch-trained 6→16→3 VAD neural network")
    commands = parser.add_subparsers(dest="command", required=True)
    tr = commands.add_parser("train")
    group = tr.add_mutually_exclusive_group(required=True)
    group.add_argument("--toy", action="store_true", help="Train on artificial targets, not emotional observations")
    group.add_argument("--data", type=Path, help="Human-reviewed JSONL transitions")
    tr.add_argument("--epochs", type=int, default=200)
    tr.add_argument("--seed", type=int, default=17)
    tr.add_argument("--out", type=Path, default=Path("runs/vad.pt"))
    tr.add_argument("--web-out", type=Path, help="Export trained weights for the static web lab")
    exp = commands.add_parser("export")
    exp.add_argument("--checkpoint", type=Path, required=True)
    exp.add_argument("--out", type=Path, default=Path("web/model.json"))
    pred = commands.add_parser("predict")
    pred.add_argument("--checkpoint", type=Path, required=True)
    pred.add_argument("--current", type=float, nargs=3, required=True)
    pred.add_argument("--observed", type=float, nargs=3, required=True)
    args = parser.parse_args()
    if args.command == "train":
        x, y = synthetic_examples(seed=args.seed) if args.toy else load_reviewed(args.data)
        network, metrics = train(x, y, epochs=args.epochs, seed=args.seed)
        source = "synthetic_toy" if args.toy else "human_reviewed"
        save_model(args.out, network, source, metrics)
        if args.web_out:
            args.web_out.parent.mkdir(parents=True, exist_ok=True)
            args.web_out.write_text(json.dumps(web_artifact(network, source, metrics)) + "\n", encoding="utf-8")
        print(json.dumps({"checkpoint": str(args.out), "source": source, **metrics}, indent=2))
    elif args.command == "export":
        network, checkpoint = restore_model(args.checkpoint)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(web_artifact(network, checkpoint["source"], checkpoint["metrics"])) + "\n", encoding="utf-8")
        print(f"Exported {args.out}")
    elif args.command == "predict":
        current, observed = check_vad(args.current, "current"), check_vad(args.observed, "observed")
        network, checkpoint = restore_model(args.checkpoint)
        with torch.no_grad():
            result = network(torch.tensor([current + observed], dtype=torch.float32))[0].tolist()
        print(json.dumps({"current": current, "observed": observed,
                          "predicted_next": result, "delta": [p - c for p, c in zip(result, current)],
                          "trained_on": checkpoint["source"]}, indent=2))


if __name__ == "__main__":
    main()
