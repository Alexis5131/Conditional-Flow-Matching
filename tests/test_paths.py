"""Sanity tests on conditional probability paths.

The contract we check for every CFM path is that the analytical target
``u_t(x_t | x1)`` equals the time-derivative of the flow map
``ψ_t(x0) = σ_t · x0 + μ_t``, computed by autodiff. If this fails for any path
the entire training is wrong.
"""

from __future__ import annotations

import torch

from flow_matching_b3.paths import DDPMPath, OTPath, VPPath


def _autodiff_target(path, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Compute d/dt ψ_t(x0) numerically via autodiff."""
    t = t.detach().clone().requires_grad_(True)
    xt = path.sample_xt(x1, x0, t)
    grads = torch.autograd.grad(
        outputs=xt, inputs=t, grad_outputs=torch.ones_like(xt), create_graph=False
    )[0]
    # autograd sums over the broadcast dims of xt; un-sum by recomputing per-time.
    # Easiest: compute Jacobian-vector products one t at a time. For the shapes
    # we use in tests (small B, tiny image), we just loop.
    return grads


def _per_sample_autodiff(path, x0, x1):
    """Element-wise time-derivative of ψ_t via repeated autograd."""
    out = torch.empty_like(x0)
    for i in range(x0.size(0)):
        t_i = torch.rand(1) * 0.9 + 0.05  # uniform in [0.05, 0.95]
        xi_0 = x0[i : i + 1]
        xi_1 = x1[i : i + 1]
        full = torch.zeros_like(xi_0[0])
        t2 = t_i.clone().requires_grad_(True)
        flat_xt2 = path.sample_xt(xi_1, xi_0, t2).reshape(-1)
        for j in range(flat_xt2.numel()):
            g = torch.autograd.grad(flat_xt2[j], t2, retain_graph=(j < flat_xt2.numel() - 1))[0]
            full.view(-1)[j] = g.squeeze()
        out[i] = full
    return out


# ---------------------------------------------------------------------------
# OT path
# ---------------------------------------------------------------------------


def test_ot_target_matches_autodiff() -> None:
    torch.manual_seed(0)
    path = OTPath(sigma_min=1e-4)
    x1 = torch.randn(3, 2, 4, 4)
    x0 = torch.randn_like(x1)
    target_autodiff = _per_sample_autodiff(path, x0, x1)
    # OTPath.target is independent of t, so we just compute it directly:
    target_analytical = path.target(x0, x1, torch.zeros(x0.size(0)))
    assert torch.allclose(target_autodiff, target_analytical, atol=1e-5, rtol=1e-5)


def test_ot_endpoints() -> None:
    path = OTPath(sigma_min=1e-4)
    x1 = torch.randn(4, 3, 8, 8)
    x0 = torch.randn_like(x1)
    t0 = torch.zeros(4)
    t1 = torch.ones(4)
    xt0 = path.sample_xt(x1, x0, t0)
    xt1 = path.sample_xt(x1, x0, t1)
    assert torch.allclose(xt0, x0, atol=1e-6)
    # At t=1, x_t ≈ x1 + σ_min · x0
    expected = x1 + 1e-4 * x0
    assert torch.allclose(xt1, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# VP path
# ---------------------------------------------------------------------------


def test_vp_target_matches_autodiff() -> None:
    torch.manual_seed(1)
    path = VPPath(beta_min=0.1, beta_max=20.0)
    B = 4
    x1 = torch.randn(B, 3, 4, 4, dtype=torch.float64)
    x0 = torch.randn_like(x1)
    t = torch.rand(B, dtype=torch.float64) * 0.9 + 0.05
    target_analytical = path.target(x0, x1, t)
    # Per-batch autodiff via vmap-like loop on a sum trick.
    target_autodiff = torch.zeros_like(x1)
    for i in range(B):
        t_i = t[i : i + 1].clone().requires_grad_(True)
        xi_1 = x1[i : i + 1]
        xi_0 = x0[i : i + 1]
        xt = path.sample_xt(xi_1, xi_0, t_i).reshape(-1)
        # Compute element-wise time derivative via grad over each component.
        for j in range(xt.numel()):
            g = torch.autograd.grad(xt[j], t_i, retain_graph=(j < xt.numel() - 1))[0]
            target_autodiff[i].view(-1)[j] = g.squeeze()
    assert torch.allclose(target_autodiff, target_analytical, atol=1e-7, rtol=1e-5)


def test_vp_endpoints() -> None:
    path = VPPath(beta_min=0.1, beta_max=20.0)
    x1 = torch.randn(2, 3, 4, 4)
    x0 = torch.randn_like(x1)
    # At t=1 (data side): α_0 = 1, σ = 0 → xt = x1
    xt1 = path.sample_xt(x1, x0, torch.ones(2))
    assert torch.allclose(xt1, x1, atol=1e-5)
    # At t=0 (noise side): α_1 small, σ ≈ 1 → xt ≈ x0
    xt0 = path.sample_xt(x1, x0, torch.zeros(2))
    # α_1 = exp(-T(1)/2) with T(1) = β_min + (β_max-β_min)/2 = 0.1 + 9.95 = 10.05
    # so α_1 ≈ exp(-5.025) ≈ 6.6e-3 — small but non-zero.
    err = (xt0 - x0).abs().mean().item()
    assert err < 0.05, err


# ---------------------------------------------------------------------------
# DDPM path
# ---------------------------------------------------------------------------


def test_ddpm_target_is_noise() -> None:
    path = DDPMPath(beta_min=0.1, beta_max=20.0)
    x1 = torch.randn(3, 3, 4, 4)
    x0 = torch.randn_like(x1)
    t = torch.rand(3) * 0.9 + 0.05
    assert torch.equal(path.target(x0, x1, t), x0)
    assert path.loss_kind == "ddpm"


def test_ddpm_shares_vp_geometry() -> None:
    """Same x_t for DDPMPath and VPPath at identical hyper-parameters."""
    vp = VPPath(beta_min=0.1, beta_max=20.0)
    ddpm = DDPMPath(beta_min=0.1, beta_max=20.0)
    x1 = torch.randn(2, 3, 4, 4)
    x0 = torch.randn_like(x1)
    t = torch.tensor([0.3, 0.7])
    assert torch.allclose(vp.sample_xt(x1, x0, t), ddpm.sample_xt(x1, x0, t))


# ---------------------------------------------------------------------------
# Shape / dtype safety
# ---------------------------------------------------------------------------


def test_batched_time_broadcasts_over_image_dims() -> None:
    for path in (OTPath(), VPPath(), DDPMPath()):
        x1 = torch.randn(5, 3, 8, 8)
        x0 = torch.randn_like(x1)
        t = torch.rand(5)
        xt = path.sample_xt(x1, x0, t)
        tg = path.target(x0, x1, t)
        assert xt.shape == x1.shape
        assert tg.shape == x1.shape
