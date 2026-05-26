"""Training losses for Flow Matching and DDPM.

Both losses ultimately reduce to a per-element MSE between a network output and
a path-dependent target.  The only difference is *what* the network outputs:

* CFM (OT, VP):  v_θ(x_t, t) ≈ u_t(x_t | x1).
* DDPM:          ε_θ(x_t, t) ≈ x0  (the noise that was added to obtain x_t).

We therefore share a single helper, ``flow_matching_loss``, and expose two thin
wrappers that match the names used in the paper for clarity.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from flow_matching_b3.paths import _BaseGaussianPath


def _sample_t(batch: int, *, eps: float, device: torch.device, dtype: torch.dtype) -> Tensor:
    return torch.empty(batch, device=device, dtype=dtype).uniform_(eps, 1.0 - eps)


def flow_matching_loss(
    model: torch.nn.Module,
    x1: Tensor,
    path: _BaseGaussianPath,
    *,
    eps: float = 1e-5,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Generic objective: ``E_{t, x0, x1} ‖ model(x_t, t) - path.target(x0, x1, t) ‖²``.

    For OT/VP this is the CFM loss (eq. 9 / eq. 23 of Lipman 2023). For DDPM this
    is the noise-matching loss (eq. 45). The returned dict carries diagnostics
    useful for wandb logging (mean/std of the target, of x_t, etc.).
    """
    x0 = torch.randn_like(x1)
    t = _sample_t(x1.size(0), eps=eps, device=x1.device, dtype=x1.dtype)
    xt = path.sample_xt(x1, x0, t)
    target = path.target(x0, x1, t)
    pred = model(xt, t)
    loss = (pred - target).pow(2).mean()
    diag = {
        "loss": loss.detach(),
        "target_rms": target.detach().pow(2).mean().sqrt(),
        "xt_rms": xt.detach().pow(2).mean().sqrt(),
        "pred_rms": pred.detach().pow(2).mean().sqrt(),
    }
    return loss, diag


def cfm_loss(
    model: torch.nn.Module,
    x1: Tensor,
    path: _BaseGaussianPath,
    *,
    eps: float = 1e-5,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Conditional Flow Matching loss — for OT and VP paths."""
    if path.loss_kind != "cfm":
        raise ValueError(f"cfm_loss expects a CFM path, got {path.loss_kind!r}")
    return flow_matching_loss(model, x1, path, eps=eps)


def ddpm_loss(
    model: torch.nn.Module,
    x1: Tensor,
    path: _BaseGaussianPath,
    *,
    eps: float = 1e-5,
) -> tuple[Tensor, dict[str, Tensor]]:
    """DDPM noise-matching loss — eq. 45 of Lipman 2023, equivalent to Ho 2020."""
    if path.loss_kind != "ddpm":
        raise ValueError(f"ddpm_loss expects a DDPM path, got {path.loss_kind!r}")
    return flow_matching_loss(model, x1, path, eps=eps)


LossFn = Callable[
    [torch.nn.Module, Tensor, _BaseGaussianPath],
    tuple[Tensor, dict[str, Tensor]],
]


def get_loss_fn(path: _BaseGaussianPath) -> LossFn:
    """Pick the correct loss given the path. Both reduce to the same code path."""
    return cfm_loss if path.loss_kind == "cfm" else ddpm_loss
