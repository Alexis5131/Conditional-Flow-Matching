"""Trajectory-straightness metric (the project's own contribution).

The paper shows trajectory straightness only *qualitatively* (Fig. 4/6). Here we
**quantify** it. For an integrated sampling trajectory ``{x_{t_k}}`` going from
noise ``x_0`` (t=0) to the generated sample ``x_1`` (t=1), define the mean
deviation from the straight chord noise→data:

    C = (1/N) Σ_i (1/K) Σ_k ‖ x^{(i)}_{t_k} - [(1 - s_k) x^{(i)}_0 + s_k x^{(i)}_1] ‖_2

normalised per trajectory by the chord length ‖x_1 - x_0‖_2, where
``s_k = t_k / t_end`` is the time rescaled to [0, 1].

For the OT path the *conditional* trajectory is an exact straight line
(ψ''_t = 0), so a well-trained FM-OT field yields C ≈ 0; curved paths (VP/DDPM)
give a markedly larger C. A small C mechanically explains low NFE: the Euler
local truncation error scales like ‖ψ''‖ h², which vanishes for a straight path.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from flow_matching_b3.paths import VPPath, _BaseGaussianPath
from flow_matching_b3.sampling import SolverName, make_vf


@torch.no_grad()
def sample_trajectory(
    vf: Callable[[Tensor, Tensor], Tensor],
    x0: Tensor,
    *,
    nfe: int,
    solver: SolverName = "euler",
    t_start: float = 0.0,
    t_end: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Integrate ``dx/dt = vf(x, t)`` and record every step boundary.

    Returns ``(ts, xs)`` with ``ts`` of shape ``(K + 1,)`` and ``xs`` of shape
    ``(K + 1, *x0.shape)`` — the trajectory states including both endpoints.
    ``nfe`` counts network evaluations, so Midpoint runs ``nfe // 2`` steps and
    RK4 ``nfe // 4``, matching the fixed-step samplers in ``sampling.py``.
    """
    if solver == "euler":
        steps = nfe
    elif solver == "midpoint":
        steps = max(nfe // 2, 1)
    elif solver == "rk4":
        steps = max(nfe // 4, 1)
    else:
        raise ValueError(f"sample_trajectory supports euler/midpoint/rk4, got {solver!r}")

    dt = (t_end - t_start) / steps
    x = x0
    xs = [x0]
    ts = [t_start]
    for k in range(steps):
        t = torch.full((x.size(0),), t_start + k * dt, device=x.device, dtype=x.dtype)
        if solver == "euler":
            x = x + dt * vf(x, t)
        elif solver == "midpoint":
            k1 = vf(x, t)
            k2 = vf(x + 0.5 * dt * k1, t + 0.5 * dt)
            x = x + dt * k2
        else:  # rk4
            k1 = vf(x, t)
            k2 = vf(x + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = vf(x + 0.5 * dt * k2, t + 0.5 * dt)
            k4 = vf(x + dt * k3, t + dt)
            x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        xs.append(x)
        ts.append(t_start + (k + 1) * dt)
    return torch.tensor(ts, device=x0.device, dtype=x0.dtype), torch.stack(xs, dim=0)


@torch.no_grad()
def straightness(
    model: torch.nn.Module,
    path: _BaseGaussianPath,
    *,
    shape: tuple[int, ...],
    n_traj: int = 1000,
    nfe: int = 100,
    solver: SolverName = "euler",
    device: torch.device | str = "cuda",
    seed: int | None = None,
    sample_eps: float = 1e-5,
) -> float:
    """Mean trajectory straightness ``C`` (lower = straighter); see module docstring.

    ``shape`` is the per-sample shape (e.g. ``(3, 32, 32)`` for CIFAR-10, ``(2,)``
    for the 2-D toy). Generates ``n_traj`` noise samples, integrates them with the
    given fixed-step solver, and averages the normalised chord deviation.
    """
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    x0 = torch.randn((n_traj, *shape), device=device, generator=generator)

    vf = make_vf(model, path)
    # VP/DDPM fields are singular at t=1; stop at 1 - sample_eps like sampling.sample.
    t_end = 1.0 - sample_eps if isinstance(path, VPPath) else 1.0
    ts, xs = sample_trajectory(vf, x0, nfe=nfe, solver=solver, t_end=t_end)

    x_start, x_final = xs[0], xs[-1]  # (N, *shape)
    feat_dims = tuple(range(1, x0.dim()))  # feature dims relative to a (N, *shape) tensor
    s = (ts / t_end).view(-1, *([1] * x0.dim()))  # (K+1, 1, ...) rescaled to [0, 1]

    chord = x_final - x_start  # (N, *shape)
    line = x_start.unsqueeze(0) + s * chord.unsqueeze(0)  # (K+1, N, *shape)
    stacked_feat_dims = tuple(d + 1 for d in feat_dims)  # xs/line carry a leading (K+1) axis
    dev = torch.linalg.vector_norm(xs - line, dim=stacked_feat_dims)  # (K+1, N)
    chord_len = torch.linalg.vector_norm(chord, dim=feat_dims).clamp_min(1e-12)  # (N,)
    c_per_traj = dev.mean(dim=0) / chord_len  # (N,)
    return float(c_per_traj.mean())
