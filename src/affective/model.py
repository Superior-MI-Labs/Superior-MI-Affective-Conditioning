"""Learnable soft-prefix adapter. Base transformer and tokenizer remain intact."""
from __future__ import annotations

import torch
from torch import nn

from .features import FEATURE_DIM


class AffectProjector(nn.Module):
    def __init__(self, hidden_size: int, prefix_tokens: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.prefix_tokens = prefix_tokens
        self.net = nn.Sequential(
            nn.Linear(FEATURE_DIM, 128),
            nn.Tanh(),
            nn.Linear(128, prefix_tokens * hidden_size),
        )
        # Small initial prefix: do not overwhelm pretrained token embeddings.
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.zero_()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[-1] != FEATURE_DIM:
            raise ValueError(f"Expected [batch, {FEATURE_DIM}] feature tensor")
        result = self.net(features)
        return result.reshape(features.shape[0], self.prefix_tokens, self.hidden_size)


def prefix_input(lm, projector: AffectProjector, token_ids: torch.Tensor,
                 conditioning: torch.Tensor, labels: torch.Tensor | None = None):
    """Single explicit path for training and generation input preparation."""
    embeddings = lm.get_input_embeddings()(token_ids)
    prefix = projector(conditioning.to(device=token_ids.device, dtype=torch.float32))
    embedded = torch.cat([prefix.to(dtype=embeddings.dtype), embeddings], dim=1)
    attention = torch.ones(embedded.shape[:2], dtype=torch.long, device=token_ids.device)
    if labels is None:
        return {"inputs_embeds": embedded, "attention_mask": attention}
    ignored = torch.full((labels.shape[0], projector.prefix_tokens), -100,
                         device=labels.device, dtype=labels.dtype)
    return {"inputs_embeds": embedded, "attention_mask": attention,
            "labels": torch.cat([ignored, labels], dim=1)}
