"""Conditional probability paths for Flow Matching and DDPM.

Notation follows Lipman et al. (2023). For a data sample x1 and noise x0 ~ N(0, I):

    p_t(x | x1) = N(x | μ_t(x1), σ_t(x1)^2 I)
    x_t = σ_t(x1) * x0 + μ_t(x1)

The flow is ψ_t(x0) = x_t, and Theorem 3 gives the conditional vector field:

    u_t(x | x1) = (σ_t'/σ_t) * (x - μ_t) + μ_t'

This module exposes three paths sharing the interface ``ConditionalPath``:

* :class:`OTPath`   — Optimal-Transport path, μ_t = t·x1, σ_t = 1 - (1 - σmin)·t.
* :class:`VPPath`   — Variance-Preserving diffusion path (reversed time t=0 noise → t=1 data).
* :class:`DDPMPath` — same VP geometry, but the regression target is the noise x0 (Ho et al., 2020).

OT and VP are used with the CFM loss (target is u_t).  DDPMPath is used with the
noise-matching loss (target is x0).  Time is sampled uniformly in [ε, 1 - ε]
during training; ε defaults to 1e-5 and matters mostly for VP/DDPM where σ_t
collapses at the boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

PathType = Literal["ot", "vp", "ddpm"]


def _expand(t: Tensor, ref: Tensor) -> Tensor:
    """Broadcast a 1-D time tensor to match the trailing dims of ``ref``."""
    return t.view(-1, *([1] * (ref.dim() - 1)))


@dataclass
class _BaseGaussianPath:
    """Shared boilerplate for Gaussian conditional paths.

    Subclasses must implement ``mu_sigma`` returning (μ_t, σ_t) as broadcastable
    tensors with the same shape as x1, and ``target`` returning the regression
    target for the loss (vector field for CFM, noise for DDPM).
    """

    sigma_min: float = 1e-4

    def sample_xt(self, x1: Tensor, x0: Tensor, t: Tensor) -> Tensor:
        mu, sigma = self.mu_sigma(x1, t)
        return sigma * x0 + mu

    def mu_sigma(self, x1: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def target(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        raise NotImplementedError

    @property
    def loss_kind(self) -> Literal["cfm", "ddpm"]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Optimal Transport path (Lipman et al. 2023, eq. 20–23)
# ---------------------------------------------------------------------------


@dataclass
class OTPath(_BaseGaussianPath):
    """μ_t = t·x1, σ_t = 1 - (1 - σmin)·t.

    Conditional flow is ψ_t(x0) = (1 - (1-σmin)·t)·x0 + t·x1, whose derivative
    w.r.t. t — the target vector field along that flow — is the simple

        u_t(ψ_t(x0) | x1) = x1 - (1 - σmin)·x0.
    """

    sigma_min: float = 1e-4

    def mu_sigma(self, x1: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        t = _expand(t, x1)
        mu = t * x1
        sigma = 1.0 - (1.0 - self.sigma_min) * t
        return mu, sigma.expand_as(x1)

    def target(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        # u_t along ψ_t(x0); does not depend on t when expressed in (x0, x1).
        return x1 - (1.0 - self.sigma_min) * x0

    @property
    def loss_kind(self) -> Literal["cfm", "ddpm"]:
        return "cfm"


# ---------------------------------------------------------------------------
# Variance-Preserving diffusion path (Lipman 2023 eq. 18–19, reversed time)
# ---------------------------------------------------------------------------


@dataclass
class VPPath(_BaseGaussianPath):
    """Reversed VP diffusion: t=0 ↔ pure noise, t=1 ↔ data.

    With α_s = exp(-T(s)/2) and T(s) = β_min·s + ½ (β_max - β_min)·s²,

        μ_t(x1) = α_{1-t} · x1,
        σ_t    = sqrt(1 - α_{1-t}²).

    Theorem 3 gives (in terms of (x0, x1)):

        u_t = ᾱ'(t)·x1 + σ̄'(t)·x0,

    with  ᾱ(t) = α_{1-t},  ᾱ'(t) = T'(1-t)/2 · ᾱ(t)  (note the sign flip from
    reparameterising α at 1-t), and  σ̄'(t) = -ᾱ(t)·ᾱ'(t)/σ̄(t).
    """

    beta_min: float = 0.1
    beta_max: float = 20.0
    sigma_min: float = 0.0  # unused, kept for interface symmetry

    def _T(self, s: Tensor) -> Tensor:
        return self.beta_min * s + 0.5 * (self.beta_max - self.beta_min) * s.pow(2)

    def _T_prime(self, s: Tensor) -> Tensor:
        return self.beta_min + (self.beta_max - self.beta_min) * s

    def _alpha_sigma_bar(self, t: Tensor) -> tuple[Tensor, Tensor]:
        """Return (ᾱ(t), σ̄(t)) — α and σ evaluated at the reversed time 1-t."""
        s = 1.0 - t
        alpha = torch.exp(-0.5 * self._T(s))
        sigma = torch.sqrt(torch.clamp(1.0 - alpha.pow(2), min=1e-12))
        return alpha, sigma

    def _alpha_sigma_prime(self, t: Tensor, alpha: Tensor, sigma: Tensor) -> tuple[Tensor, Tensor]:
        """Derivatives w.r.t. t of (ᾱ, σ̄)."""
        s = 1.0 - t
        # d/dt α(1-t) = -α'(1-t) = -(-T'(1-t)/2 · α(1-t)) = +T'(1-t)/2 · α(1-t)
        alpha_prime = 0.5 * self._T_prime(s) * alpha
        sigma_prime = -alpha * alpha_prime / sigma
        return alpha_prime, sigma_prime

    def mu_sigma(self, x1: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        t_b = _expand(t, x1)
        alpha, sigma = self._alpha_sigma_bar(t_b)
        return alpha * x1, sigma.expand_as(x1)

    def target(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        t_b = _expand(t, x1)
        alpha, sigma = self._alpha_sigma_bar(t_b)
        alpha_p, sigma_p = self._alpha_sigma_prime(t_b, alpha, sigma)
        return alpha_p * x1 + sigma_p * x0

    @property
    def loss_kind(self) -> Literal["cfm", "ddpm"]:
        return "cfm"


# ---------------------------------------------------------------------------
# DDPM path (same VP geometry, but regression target is the noise x0)
# ---------------------------------------------------------------------------


@dataclass
class DDPMPath(VPPath):
    """Same geometry as :class:`VPPath` but the loss regresses the noise x0.

    The model output ε_θ(x_t, t) is asked to predict x0 directly
    (Ho et al. 2020, eq. 14; Lipman 2023 eq. 45).
    """

    def target(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        return x0

    @property
    def loss_kind(self) -> Literal["cfm", "ddpm"]:
        return "ddpm"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_path(name: PathType, **kwargs) -> _BaseGaussianPath:
    match name:
        case "ot":
            return OTPath(**kwargs)
        case "vp":
            return VPPath(**kwargs)
        case "ddpm":
            return DDPMPath(**kwargs)
        case _:
            raise ValueError(f"Unknown path type: {name!r}. Expected one of: ot, vp, ddpm.")
