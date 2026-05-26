"""ODE samplers for trained Flow Matching / DDPM models.

Sampling integrates ``dx/dt = u_t(x)`` from t=0 (noise) to t=1 (data). For
CFM-trained models the network output **is** ``u_t``. For DDPM-trained models
the network output is a noise prediction ``ε_θ(x_t, t)`` which we convert to
the equivalent vector field of the deterministic probability-flow ODE
(Lipman 2023, eq. 46):

    u_t(x) = -T'(1-t)/2 · (ε_θ(x, t)/σ_t - x).

Three fixed-step solvers (Euler, Midpoint = RK2, RK4) are implemented in pure
PyTorch for the FID-vs-NFE ablation (Fig. 7 of the paper). The adaptive
``dopri5`` solver is provided through ``torchdiffeq`` and used both for high-
quality sampling and for the NFE statistic reported in Table 1.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch
from torch import Tensor

from flow_matching_b3.paths import DDPMPath, VPPath, _BaseGaussianPath

SolverName = Literal["euler", "midpoint", "rk4", "dopri5"]


def _ddpm_eps_to_vector_field(
    eps_pred: Tensor, x: Tensor, t: Tensor, path: VPPath
) -> Tensor:
    """Convert a noise prediction into the probability-flow ODE vector field."""
    t_b = t.view(-1, *([1] * (x.dim() - 1)))
    alpha, sigma = path._alpha_sigma_bar(t_b)  # noqa: SLF001 — intentional package-internal use
    Tprime = path._T_prime(1.0 - t_b)  # noqa: SLF001
    score_like = eps_pred / sigma  # paper's "s_t" notation, eq. 43
    return -0.5 * Tprime * (score_like - x)


def make_vf(model: torch.nn.Module, path: _BaseGaussianPath) -> Callable[[Tensor, Tensor], Tensor]:
    """Return a closure ``vf(x, t) -> dx/dt`` for the given model and path.

    For CFM paths the model's raw output is the vector field. For DDPM the
    output is converted via the probability-flow identity.
    """
    if path.loss_kind == "cfm":

        def vf(x: Tensor, t: Tensor) -> Tensor:
            return model(x, t)

        return vf

    assert isinstance(path, DDPMPath)

    def vf(x: Tensor, t: Tensor) -> Tensor:
        eps = model(x, t)
        return _ddpm_eps_to_vector_field(eps, x, t, path)

    return vf


# ---------------------------------------------------------------------------
# Fixed-step solvers
# ---------------------------------------------------------------------------


@torch.no_grad()
def euler_sample(
    vf: Callable[[Tensor, Tensor], Tensor],
    x0: Tensor,
    *,
    nfe: int,
    t_start: float = 0.0,
    t_end: float = 1.0,
) -> Tensor:
    dt = (t_end - t_start) / nfe
    x = x0
    for k in range(nfe):
        t = torch.full((x.size(0),), t_start + k * dt, device=x.device, dtype=x.dtype)
        x = x + dt * vf(x, t)
    return x


@torch.no_grad()
def midpoint_sample(
    vf: Callable[[Tensor, Tensor], Tensor],
    x0: Tensor,
    *,
    nfe: int,
    t_start: float = 0.0,
    t_end: float = 1.0,
) -> Tensor:
    """RK2 / midpoint rule. Uses 2 function evaluations per step.

    ``nfe`` here is the **number of network evaluations**, so we run ``nfe // 2``
    steps to keep the comparison fair against Euler (Fig. 7 of the paper).
    """
    steps = max(nfe // 2, 1)
    dt = (t_end - t_start) / steps
    x = x0
    for k in range(steps):
        t = torch.full((x.size(0),), t_start + k * dt, device=x.device, dtype=x.dtype)
        k1 = vf(x, t)
        x_mid = x + 0.5 * dt * k1
        t_mid = t + 0.5 * dt
        k2 = vf(x_mid, t_mid)
        x = x + dt * k2
    return x


@torch.no_grad()
def rk4_sample(
    vf: Callable[[Tensor, Tensor], Tensor],
    x0: Tensor,
    *,
    nfe: int,
    t_start: float = 0.0,
    t_end: float = 1.0,
) -> Tensor:
    """Classical 4-stage Runge–Kutta. 4 evaluations per step."""
    steps = max(nfe // 4, 1)
    dt = (t_end - t_start) / steps
    x = x0
    for k in range(steps):
        t = torch.full((x.size(0),), t_start + k * dt, device=x.device, dtype=x.dtype)
        k1 = vf(x, t)
        k2 = vf(x + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = vf(x + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = vf(x + dt * k3, t + dt)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


# ---------------------------------------------------------------------------
# Adaptive solver (dopri5 via torchdiffeq)
# ---------------------------------------------------------------------------


class _NFECounter:
    """Tiny wrapper that counts vector-field evaluations during ODE solve."""

    def __init__(self, vf: Callable[[Tensor, Tensor], Tensor]):
        self.vf = vf
        self.n = 0

    def __call__(self, t: Tensor, x: Tensor) -> Tensor:
        self.n += 1
        # torchdiffeq passes scalar t; broadcast to batch dim expected by the model.
        t_batch = t.expand(x.size(0)) if t.dim() == 0 else t
        return self.vf(x, t_batch)


@torch.no_grad()
def dopri5_sample(
    vf: Callable[[Tensor, Tensor], Tensor],
    x0: Tensor,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    t_start: float = 0.0,
    t_end: float = 1.0,
) -> tuple[Tensor, int]:
    """Adaptive Dormand–Prince 5(4). Returns ``(x1, nfe_used)``."""
    from torchdiffeq import odeint  # local import to keep cold-start fast

    counter = _NFECounter(vf)
    ts = torch.tensor([t_start, t_end], device=x0.device, dtype=x0.dtype)
    out = odeint(counter, x0, ts, method="dopri5", atol=atol, rtol=rtol)
    return out[-1], counter.n


# ---------------------------------------------------------------------------
# Convenience dispatcher
# ---------------------------------------------------------------------------


def sample(
    model: torch.nn.Module,
    path: _BaseGaussianPath,
    *,
    shape: tuple[int, ...],
    nfe: int = 100,
    solver: SolverName = "dopri5",
    device: torch.device | str = "cuda",
    seed: int | None = None,
) -> tuple[Tensor, int]:
    """High-level entry point: noise → samples + actual NFE used."""
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    x0 = torch.randn(shape, device=device, generator=generator)
    vf = make_vf(model, path)
    if solver == "euler":
        return euler_sample(vf, x0, nfe=nfe), nfe
    if solver == "midpoint":
        return midpoint_sample(vf, x0, nfe=nfe), (nfe // 2) * 2
    if solver == "rk4":
        return rk4_sample(vf, x0, nfe=nfe), (nfe // 4) * 4
    if solver == "dopri5":
        return dopri5_sample(vf, x0)
    raise ValueError(f"Unknown solver: {solver!r}")
