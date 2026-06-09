"""Tests for the ODE samplers and the DDPM ε→vector-field conversion.

``_ddpm_eps_to_vector_field`` is the only non-trivial DDPM-specific maths and is
otherwise unguarded: a wrong sign/factor would pass ``test_paths.py`` yet silently
corrupt DDPM sampling. We pin it against the analytic VP vector field, document the
t=1 singularity it carries, and check that the boundary fix keeps sampling bounded.
"""

from __future__ import annotations

import torch
from torch import nn

from flow_matching_b3.paths import DDPMPath, OTPath, VPPath
from flow_matching_b3.sampling import _ddpm_eps_to_vector_field, make_vf, rk4_sample, sample


def test_ddpm_eps_to_vf_matches_vp_target() -> None:
    """Feeding the true noise x0 as ε must reproduce the analytic VP field exactly."""
    torch.manual_seed(0)
    ddpm = DDPMPath(beta_min=0.1, beta_max=20.0)
    vp = VPPath(beta_min=0.1, beta_max=20.0)
    x1 = torch.randn(4, 3, 4, 4, dtype=torch.float64)
    x0 = torch.randn_like(x1)
    for t_val in (0.05, 0.2, 0.5, 0.8, 0.95):
        t = torch.full((x1.size(0),), t_val, dtype=torch.float64)
        xt = ddpm.sample_xt(x1, x0, t)
        vf = _ddpm_eps_to_vector_field(x0, xt, t, ddpm)  # ε := x0 (perfect predictor)
        target = vp.target(x0, x1, t)
        assert torch.allclose(vf, target, atol=1e-6, rtol=1e-6), t_val


def test_ddpm_field_singular_at_t1() -> None:
    """The 1/σ̄ factor explodes at t=1 — this is why sample() stops VP/DDPM at 1-eps."""
    ddpm = DDPMPath(beta_min=0.1, beta_max=20.0)
    x = torch.randn(2, 3, 4, 4)
    eps = torch.ones_like(x)
    vf_interior = _ddpm_eps_to_vector_field(eps, x, torch.full((2,), 1.0 - 1e-5), ddpm)
    vf_boundary = _ddpm_eps_to_vector_field(eps, x, torch.full((2,), 1.0), ddpm)
    assert vf_interior.abs().max() < 1e3
    assert vf_boundary.abs().max() > 1e4


class _ConstEps(nn.Module):
    """Dummy model whose ε prediction is a bounded constant (worst case for 1/σ̄)."""

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        return torch.ones_like(x)


def test_ddpm_boundary_fix_prevents_blowup() -> None:
    """RK4 stopping at 1-eps (the fix) is dramatically smaller than integrating to t=1."""
    torch.manual_seed(0)
    vf = make_vf(_ConstEps(), DDPMPath())
    x0 = torch.randn(2, 3, 8, 8)
    fixed = rk4_sample(vf, x0, nfe=20, t_end=1.0 - 1e-5)  # what sample() does for DDPM
    singular = rk4_sample(vf, x0, nfe=20, t_end=1.0)  # evaluates the 1/σ̄ spike at t=1
    assert torch.isfinite(fixed).all()
    assert fixed.abs().max().item() < 1000.0
    # Integrating to the singular t=1 boundary at least doubles the output magnitude;
    # the field-level spike is ~50× (see test_ddpm_field_singular_at_t1).
    assert singular.abs().max().item() > 2.0 * fixed.abs().max().item()


def test_ot_sampling_runs_to_t1() -> None:
    """OT has no boundary singularity; euler sampling runs and stays finite."""
    out, nfe = sample(
        _ConstEps(), OTPath(), shape=(2, 3, 8, 8), nfe=16, solver="euler", device="cpu", seed=0
    )
    assert torch.isfinite(out).all()
    assert nfe == 16
