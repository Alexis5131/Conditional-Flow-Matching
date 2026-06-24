"""Tests for the trajectory-straightness metric C (``flow_matching_b3.metrics``).

C must be ~0 for a perfectly straight (constant-velocity) field and clearly
larger for a curved one. We use synthetic 2-D vector fields wrapped as ``nn.Module``
so the metric goes through the exact same ``make_vf`` / solver path as the real
CIFAR-10 model, but with an analytically known trajectory shape.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from flow_matching_b3.metrics import sample_trajectory, straightness
from flow_matching_b3.paths import OTPath
from flow_matching_b3.sampling import make_vf


class _ConstantField(nn.Module):
    """vf(x, t) = c — integral curves are straight lines, so C must be ~0."""

    def __init__(self, c: Tensor):
        super().__init__()
        self.register_buffer("c", c)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:  # noqa: ARG002
        return self.c.expand_as(x)


class _RotationField(nn.Module):
    """vf(x, t) = ω · (-x_2, x_1) — integral curves are circular arcs (curved)."""

    def __init__(self, omega: float = 2.0):
        super().__init__()
        self.omega = omega

    def forward(self, x: Tensor, t: Tensor) -> Tensor:  # noqa: ARG002
        return self.omega * torch.stack([-x[:, 1], x[:, 0]], dim=1)


def test_straightness_zero_for_constant_field() -> None:
    """A constant field produces exact straight lines → C ≈ 0."""
    model = _ConstantField(torch.tensor([1.5, -0.5]))
    c = straightness(
        model, OTPath(), shape=(2,), n_traj=256, nfe=50, solver="euler", device="cpu", seed=0
    )
    assert c < 1e-5, c


def test_straightness_larger_for_curved_field() -> None:
    """A rotation field bows the trajectory away from the chord → C clearly > 0."""
    straight = straightness(
        _ConstantField(torch.tensor([1.0, 1.0])),
        OTPath(),
        shape=(2,),
        n_traj=256,
        nfe=200,
        solver="euler",
        device="cpu",
        seed=0,
    )
    curved = straightness(
        _RotationField(omega=2.0),
        OTPath(),
        shape=(2,),
        n_traj=256,
        nfe=200,
        solver="euler",
        device="cpu",
        seed=0,
    )
    assert curved > 0.05, curved
    assert curved > 100 * straight, (straight, curved)


def test_sample_trajectory_records_endpoints_and_shape() -> None:
    """Trajectory tensor carries K+1 states; first state equals x0, NFE honoured per solver."""
    vf = make_vf(_ConstantField(torch.tensor([1.0, 0.0])), OTPath())
    x0 = torch.randn(8, 2)
    ts, xs = sample_trajectory(vf, x0, nfe=20, solver="euler")
    assert xs.shape == (21, 8, 2)
    assert ts.shape == (21,)
    assert torch.allclose(xs[0], x0)
    # RK4 does nfe // 4 steps → 5 steps → 6 recorded states.
    _, xs_rk4 = sample_trajectory(vf, x0, nfe=20, solver="rk4")
    assert xs_rk4.shape == (6, 8, 2)
