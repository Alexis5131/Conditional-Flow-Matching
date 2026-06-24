# Flow Matching for Generative Modeling — reproduction B3

Reproduction réduite de **Lipman et al., *Flow Matching for Generative Modeling*** (ICLR 2023, [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)) sur **CIFAR-10**, dans le cadre du projet de validation *Introduction to Deep Learning* (B3, Ambroise Bouru-Gazeau).

> **Version resit** : projet resserré sur **FM-OT uniquement**, avec un **modèle beaucoup plus petit** (~6,9 M params au lieu de 38,3 M) pour réduire le coût de calcul. Ablation unique : la **métrique de rectitude** des trajectoires. VP/DDPM restent dans le code mais ne sont plus entraînés.

## Ce qu'on reproduit

La ligne **FM-OT** du Table 1 du papier sur CIFAR-10, avec un backbone réduit (~6,9 M params, 150 k steps) :

| Modèle    | Loss | Cible                                     | FID papier (modèle complet) |
|-----------|------|-------------------------------------------|------------------------------|
| **FM-OT** | CFM  | $u_t(x\|x_1) = x_1 - (1-\sigma_{\min})x_0$ | **6.35**                     |

On vise à reproduire la **tendance** (FID raisonnable + **NFE faible**), pas l'égalité absolue (modèle ~5,5× plus petit).

Figures clés : trajectoires de sampling FM-OT, FID au cours de l'entraînement, et l'**ablation rectitude** — $C$ vs NFE/solveur (axe A) et $C$ au fil de l'entraînement (axe B).

## Structure

```
src/flow_matching_b3/
  paths.py        OT (utilisé), VP, DDPM (hors-scope) — μ_t, σ_t, target u_t
  unet.py         ADM-style U-Net + adm_unet_cifar10_small (~6,9 M) + 2D MLP
  losses.py       CFM + ε-matching (same code path)
  sampling.py     Euler / Midpoint / RK4 / dopri5 (with NFE counter)
  metrics.py      rectitude C des trajectoires (straightness) ★ contribution
  ema.py          EMA shadow with swap-in context manager
  fid.py          clean-fid wrapper, CIFAR-10 train stats
  train.py        TrainConfig + train(cfg) — used by the notebook
tests/test_paths.py     analytical target vs autodiff
tests/test_sampling.py  DDPM ε→vector-field + fix de la singularité t=1
tests/test_metrics.py   rectitude C ≈ 0 sur champ droit, > 0 sur champ courbe
notebooks/
  01_sanity_2d.ipynb        — toy checkerboard
  02_fm_ot_colab.ipynb      — ★ notebook Colab complet : train FM-OT + FID + ablation rectitude (A/B)
```

## Quickstart local (smoke test)

```bash
uv sync --extra dev
uv run pytest tests/                # paths + samplers + rectitude
uv run jupyter notebook notebooks/01_sanity_2d.ipynb
```

## Quickstart Colab (tout-en-un)

Un seul notebook fait tout : **`notebooks/02_fm_ot_colab.ipynb`**.

1. Ouvrir dans Colab avec un **GPU NVIDIA** (`Exécution → Modifier le type d'exécution → GPU`).
2. La cellule **Setup** **clone le code depuis GitHub** (`REPO_URL`/`REPO_BRANCH`) et l'installe ; le Drive n'est monté que pour stocker les résultats.
3. **Exécuter tout** (`Exécution → Tout exécuter`) : setup → entraînement FM-OT (~150 k steps) → FID(50 k) + NFE → ablation rectitude (axes A/B) → figures.
4. Tout est persisté sur Drive dans `Drive/MyDrive/flow-matching-b3/fm-ot-cifar10/` : checkpoints (reprise auto), `results.json`, et figures dans `.../fm-ot-cifar10/pics/` — à copier dans `report/pics/` pour compiler le PDF.

La reprise depuis le dernier checkpoint est automatique — si Colab coupe une session, relancer le notebook reprend là où ça en était.

## Hyper-paramètres (lignée ADM / Dhariwal-Nichol — le papier sous-spécifie CIFAR-10)

| Param                | Valeur      |
|----------------------|-------------|
| U-Net base channels  | 64 (→ 128 à la résolution 16) |
| Params               | **6 949 187 (~6,9 M)** — `adm_unet_cifar10_small` |
| Depth / mult         | 2 / (1,2,2) |
| Attention            | résolution 16, 4 heads × 32 (= pleine largeur 128) |
| Optim                | Adam (β=0.9/0.999, ε=1e-8), lr=1e-4, wd=0 |
| LR schedule          | warmup 5 k + decay linéaire |
| Batch effectif       | 256 = batch physique × `accum_iter` (ex. 64 × 4) |
| Steps (optim)        | **150 000** (réduit pour le modèle plus petit) |
| EMA                  | 0.9999 (avec warmup de decay) |
| Précision            | FP32 en éval ; bf16 possible à l'entraînement A100 |
| σ_min (OT)           | 1e-4        |
| Sampler (Table 1)    | dopri5, atol=rtol=1e-5 (valeur exacte du papier) |
| FID                  | clean-fid `mode="legacy_tensorflow"`, 50 k échantillons |

> **Note repro** : modèle réduit à ~6,9 M params (vs 38,3 M) — le FID absolu sera supérieur
> au 6.35 du papier (backbone complet) ; on vise la tendance (NFE faible) et le comportement
> de la rectitude. Le FID est en protocole `legacy_tensorflow` (Inception TF), seul comparable
> au papier — pas `mode="clean"`. Les chiffres mid-training à 10 k sont biaisés ↑ vs les 50 k.

> **Perf A100** : activées par défaut (`tf32`, `channels_last`, `compile` dans `TrainConfig`,
> tous no-op sur CPU) — TF32 TensorCores pour le fp32, mémoire NHWC, `torch.compile` (fusion
> de kernels), `cudnn.benchmark`, prefetch dataloader. Le batch effectif 256 est préservé via
> accumulation. La cellule Setup du notebook affiche le GPU et la VRAM. bf16 (`dtype="bfloat16"`,
> activé par défaut sur GPU) est utilisé à l'entraînement ; le FID/éval reste en FP32.

## Tests

`uv run --extra dev pytest tests/` — vérifie en particulier que (a) la cible analytique $u_t(x \mid x_1)$ de chaque path égale la dérivée temporelle de $\psi_t$ par autodiff (tol. 1e-5), (b) la conversion DDPM $\varepsilon \to$ champ reproduit le champ VP analytique (singularité $t=1$ contournée), et (c) la métrique de rectitude $C$ vaut ≈ 0 sur un champ analytiquement droit et croît sur un champ courbe.

## Références

- Lipman, Chen, Ben-Hamu, Nickel, Le. *Flow Matching for Generative Modeling.* ICLR 2023.
- Dhariwal & Nichol. *Diffusion Models Beat GANs on Image Synthesis.* NeurIPS 2021 — pour l'architecture U-Net.
- Ho, Jain, Abbeel. *Denoising Diffusion Probabilistic Models.* NeurIPS 2020 — baseline DDPM.
- Chen, Rubanova, Bettencourt, Duvenaud. *Neural ODEs.* NeurIPS 2018 — `torchdiffeq`.
