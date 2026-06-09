"""Exponential moving average of model parameters.

Standard EMA used in diffusion / flow models (decay typically 0.9999). Keeps a
shadow copy of all parameters and updates it after each optimiser step. The
EMA weights are what we sample from at evaluation time, not the raw weights.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy

import torch
from torch import nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.step = 0
        self.shadow = deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.step += 1
        # Bias-corrected warmup so the shadow tracks the live model early on instead
        # of staying anchored to the random init (decay^step decays slowly otherwise).
        d = min(self.decay, (1 + self.step) / (10 + self.step))
        for p_ema, p in zip(self.shadow.parameters(), model.parameters(), strict=True):
            p_ema.mul_(d).add_(p.detach(), alpha=1.0 - d)
        # Buffers (none for GroupNorm) are copied, not averaged — fine for buffer-free norms.
        for b_ema, b in zip(self.shadow.buffers(), model.buffers(), strict=True):
            b_ema.copy_(b)

    @contextmanager
    def swap_into(self, model: nn.Module):
        """Temporarily replace ``model``'s params with the EMA shadow, then restore."""
        backup = [p.detach().clone() for p in model.parameters()]
        for p, p_ema in zip(model.parameters(), self.shadow.parameters(), strict=True):
            p.data.copy_(p_ema.data)
        try:
            yield model
        finally:
            for p, b in zip(model.parameters(), backup, strict=True):
                p.data.copy_(b)

    def state_dict(self) -> dict:
        return {"decay": self.decay, "step": self.step, "shadow": self.shadow.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.decay = state["decay"]
        self.step = state.get("step", 0)  # tolerate checkpoints written before warmup was added
        self.shadow.load_state_dict(state["shadow"])
