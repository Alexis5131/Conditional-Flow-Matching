# Flow Matching for Generative Modeling — reproduction B3

Reproduction réduite de **Lipman et al., *Flow Matching for Generative Modeling*** (ICLR 2023, [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)) sur **CIFAR-10**, dans le cadre du projet de validation *Introduction to Deep Learning* (B3, Ambroise Bouru-Gazeau).

## Ce qu'on reproduit

Le triplet du Table 1 du papier sur CIFAR-10 (~36 M paramètres, 391 k steps) :

| Modèle           | Loss        | Cible                                     | FID papier |
|------------------|-------------|-------------------------------------------|------------|
| **DDPM**         | ε-matching  | $x_0$ (le bruit)                          | 7.48       |
| **FM-Diffusion** | CFM         | $u_t(x\|x_1)$ avec path VP                | 8.06       |
| **FM-OT**        | CFM         | $u_t(x\|x_1)$ avec path Optimal Transport | **6.35**   |

Et les figures clés : **Fig. 6** (trajectoires depuis les mêmes seeds), **Fig. 7** (FID vs NFE avec Euler/Mid/RK4), **Fig. 5** (FID au cours de l'entraînement).

## Structure

```
src/flow_matching_b3/
  paths.py        OT, VP, DDPM — μ_t, σ_t, target u_t
  unet.py         ADM-style U-Net (Dhariwal & Nichol 2021) + 2D MLP
  losses.py       CFM + ε-matching (same code path)
  sampling.py     Euler / Midpoint / RK4 / dopri5 (with NFE counter)
  ema.py          EMA shadow with swap-in context manager
  fid.py          clean-fid wrapper, CIFAR-10 train stats
  train.py        TrainConfig + train(cfg) — used by the notebook
tests/test_paths.py     analytical target vs autodiff (7 tests)
notebooks/
  01_sanity_2d.ipynb        — toy checkerboard (P2 du plan)
  02_train_cifar.ipynb      — ★ main training notebook (Colab)
  03_eval_ablations.ipynb   — load checkpoints, Table 1 + Fig. 5/6/7
```

## Quickstart local (smoke test)

```bash
uv sync --extra dev
uv run pytest tests/                # 7 tests sur les paths
uv run jupyter notebook notebooks/01_sanity_2d.ipynb
```

## Quickstart Colab (entraînement complet)

1. Ouvrir `notebooks/02_train_cifar.ipynb` dans Colab avec une **A100/H100**.
2. Régler la première cellule : `PATH_TYPE`, `RUN_NAME`, `DRIVE_ROOT`, `REPO_URL`.
3. Exécuter tout. Les checkpoints sont écrits dans `Drive/MyDrive/flow-matching-b3/<RUN_NAME>/`.
4. Répéter pour `PATH_TYPE ∈ {"ot", "vp", "ddpm"}` (≈ 30 h / run).
5. Ouvrir `notebooks/03_eval_ablations.ipynb` pour produire la table et les figures.

La reprise depuis le dernier checkpoint est automatique — si Colab coupe une session, relancer le notebook reprend là où ça en était.

## Hyper-paramètres (aligné Table 3 du papier, CIFAR-10)

| Param                | Valeur      |
|----------------------|-------------|
| U-Net channels       | 256         |
| Depth / mult         | 2 / (1,2,2,2) |
| Attention            | résolution 16, 4 heads × 64 |
| Optim                | Adam, lr=5e-4, wd=0 |
| LR schedule          | polynomial decay, warmup 45 k |
| Batch                | 256         |
| Steps                | 391 000     |
| EMA                  | 0.9999      |
| Précision            | FP32        |
| σ_min (OT)           | 1e-4        |
| β_min / β_max (VP)   | 0.1 / 20    |

## Tests

`uv run pytest tests/` — les 7 tests vérifient en particulier que la cible analytique $u_t(x \mid x_1)$ retournée par chaque path est égale à la dérivée temporelle de $\psi_t$ calculée par autodiff (tolérance 1e-5).

## Références

- Lipman, Chen, Ben-Hamu, Nickel, Le. *Flow Matching for Generative Modeling.* ICLR 2023.
- Dhariwal & Nichol. *Diffusion Models Beat GANs on Image Synthesis.* NeurIPS 2021 — pour l'architecture U-Net.
- Ho, Jain, Abbeel. *Denoising Diffusion Probabilistic Models.* NeurIPS 2020 — baseline DDPM.
- Chen, Rubanova, Bettencourt, Duvenaud. *Neural ODEs.* NeurIPS 2018 — `torchdiffeq`.
