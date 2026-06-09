# Flow Matching for Generative Modeling — reproduction B3

Reproduction réduite de **Lipman et al., *Flow Matching for Generative Modeling*** (ICLR 2023, [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)) sur **CIFAR-10**, dans le cadre du projet de validation *Introduction to Deep Learning* (B3, Ambroise Bouru-Gazeau).

## Ce qu'on reproduit

Le triplet du Table 1 du papier sur CIFAR-10 (~38 M paramètres, 391 k steps) :

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
tests/test_paths.py     analytical target vs autodiff
tests/test_sampling.py  DDPM ε→vector-field + fix de la singularité t=1
notebooks/
  01_sanity_2d.ipynb        — toy checkerboard (P2 du plan)
  02_train_cifar.ipynb      — ★ main training notebook (Colab)
  03_eval_ablations.ipynb   — load checkpoints, Table 1 + Fig. 5/6/7
```

## Quickstart local (smoke test)

```bash
uv sync --extra dev
uv run pytest tests/                # 11 tests (paths + samplers)
uv run jupyter notebook notebooks/01_sanity_2d.ipynb
```

## Quickstart Colab (entraînement complet)

1. Ouvrir `notebooks/02_train_cifar.ipynb` dans Colab avec une **A100/H100**.
2. Régler la première cellule : `PATH_TYPE`, `RUN_NAME`, `DRIVE_ROOT`, `REPO_URL`.
3. Exécuter tout. Les checkpoints sont écrits dans `Drive/MyDrive/flow-matching-b3/<RUN_NAME>/`.
4. Répéter pour `PATH_TYPE ∈ {"ot", "vp", "ddpm"}` (≈ 30 h / run).
5. Ouvrir `notebooks/03_eval_ablations.ipynb` pour produire la table et les figures.

La reprise depuis le dernier checkpoint est automatique — si Colab coupe une session, relancer le notebook reprend là où ça en était.

## Hyper-paramètres (lignée ADM / Dhariwal-Nichol — le papier sous-spécifie CIFAR-10)

| Param                | Valeur      |
|----------------------|-------------|
| U-Net base channels  | 128 (→ 256 à la résolution 16) |
| Params               | ~38.3 M     |
| Depth / mult         | 2 / (1,2,2,2) |
| Attention            | résolution 16, 4 heads × 64 (= pleine largeur 256) |
| Optim                | Adam (β=0.9/0.999, ε=1e-8), lr=1e-4, wd=0 |
| LR schedule          | warmup 45 k + decay linéaire — choix repo (ADM utilise lr constant, *à vérifier*) |
| Batch effectif       | 256 = batch physique × `accum_iter` (ex. 64 × 4 sur A100 40 Go) |
| Steps (optim)        | 391 000     |
| EMA                  | 0.9999 (avec warmup de decay) |
| Précision            | FP32        |
| σ_min (OT)           | 1e-4        |
| β_min / β_max (VP)   | 0.1 / 20    |
| Sampler (Table 1)    | dopri5, atol=rtol=1e-5 (valeur exacte du papier) |
| FID                  | clean-fid `mode="legacy_tensorflow"`, 50 k échantillons |

> **Note repro** : Lipman et al. n'optimisent pas l'archi pour CIFAR-10 et ne tabulent
> pas clairement lr/batch/steps. On suit le défaut ADM (lr 1e-4). Le FID est calculé en
> protocole `legacy_tensorflow` (Inception TF), seul comparable aux 6.35/8.06/7.48 — pas
> `mode="clean"`. Les chiffres mid-training/Fig. 7 à 10 k sont biaisés ↑ vs les 50 k de la table.

## Tests

`uv run pytest tests/` — les 11 tests vérifient en particulier que (a) la cible analytique $u_t(x \mid x_1)$ de chaque path égale la dérivée temporelle de $\psi_t$ par autodiff (tol. 1e-5), et (b) la conversion DDPM $\varepsilon \to$ champ de vitesse reproduit le champ VP analytique, avec la singularité en $t=1$ contournée par les samplers.

## Références

- Lipman, Chen, Ben-Hamu, Nickel, Le. *Flow Matching for Generative Modeling.* ICLR 2023.
- Dhariwal & Nichol. *Diffusion Models Beat GANs on Image Synthesis.* NeurIPS 2021 — pour l'architecture U-Net.
- Ho, Jain, Abbeel. *Denoising Diffusion Probabilistic Models.* NeurIPS 2020 — baseline DDPM.
- Chen, Rubanova, Bettencourt, Duvenaud. *Neural ODEs.* NeurIPS 2018 — `torchdiffeq`.
