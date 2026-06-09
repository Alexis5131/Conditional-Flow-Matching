"""Training loop for the CIFAR-10 reproduction.

Designed to be driven from a notebook cell. The :class:`TrainConfig` carries
every hyper-parameter (ADM/Dhariwal-Nichol CIFAR lineage; the paper itself
under-specifies CIFAR-10) and :func:`train` runs the loop end-to-end. Resumption
from any checkpoint under ``cfg.run_dir/ckpt_*.pt`` restores model/optim/EMA and
RNG state; the data-loader ordering is *not* replayed bit-exactly, so a resumed
run follows a different but statistically-equivalent data stream.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm

from flow_matching_b3.ema import EMA
from flow_matching_b3.losses import get_loss_fn
from flow_matching_b3.paths import PathType, get_path
from flow_matching_b3.unet import UNet, adm_unet_cifar10

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    path_type: PathType = "ot"
    run_dir: Path = Path("./runs/fm-ot")
    data_root: Path = Path("./data/cifar10")

    # Optimisation (ADM / Dhariwal-Nichol CIFAR-10 lineage)
    max_steps: int = 391_000  # number of optimiser steps
    batch_size: int = 64  # physical / micro-batch (sized to fit the GPU; see notebook smoke test)
    accum_iter: int = 4  # grad-accumulation micro-steps; effective batch = batch_size * accum_iter
    lr_peak: float = 1e-4  # ADM default (the paper inherits Dhariwal-Nichol and does not retune lr)
    lr_init: float = 1e-8
    warmup_steps: int = (
        45_000  # NB: ADM/DDPM use ~constant lr; warmup+decay is a repo choice (à vérifier)
    )
    poly_decay: bool = True

    # Path-specific
    sigma_min: float = 1e-4
    vp_beta_min: float = 0.1
    vp_beta_max: float = 20.0
    time_eps: float = 1e-5

    # EMA
    ema_decay: float = 0.9999

    # Logging / IO
    ckpt_every: int = 25_000
    log_every: int = 100
    eval_every: int = 25_000  # set to 0 to disable mid-training FID
    eval_n_samples: int = 10_000
    sample_grid_every: int = 5_000  # save a 8×8 PNG grid this often
    sample_batch: int = 64
    sample_nfe: int = 50

    # Hardware
    device: str = "cuda"
    dtype: str = "float32"  # "float32" | "bfloat16" | "float16"
    grad_clip: float = 1.0
    num_workers: int = 4

    # Misc
    seed: int = 42
    dropout: float = 0.0

    # WandB
    wandb_project: str = "flow-matching-b3"
    wandb_run_name: str | None = None
    use_wandb: bool = False

    def to_json(self) -> str:
        # Path objects don't serialise; convert.
        return json.dumps(
            {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(self).items()},
            indent=2,
        )


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------


def lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        frac = step / max(1, cfg.warmup_steps)
        return cfg.lr_init + (cfg.lr_peak - cfg.lr_init) * frac
    if not cfg.poly_decay:
        return cfg.lr_peak
    remain = max(0, cfg.max_steps - cfg.warmup_steps)
    decay_step = step - cfg.warmup_steps
    frac = min(1.0, decay_step / max(1, remain))
    return cfg.lr_peak + (cfg.lr_init - cfg.lr_peak) * frac


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def make_cifar10_loader(cfg: TrainConfig) -> DataLoader:
    """CIFAR-10 train loader. Images normalised to [-1, 1] like ADM."""
    tfm = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # [0, 1]
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # [-1, 1]
        ]
    )
    ds = datasets.CIFAR10(root=str(cfg.data_root), train=True, download=True, transform=tfm)
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )


def _infinite(loader: DataLoader):
    while True:
        yield from loader


# ---------------------------------------------------------------------------
# Build model / path / optim
# ---------------------------------------------------------------------------


def build(cfg: TrainConfig) -> tuple[UNet, Adam, EMA, object]:
    torch.manual_seed(cfg.seed)
    model = adm_unet_cifar10(dropout=cfg.dropout).to(cfg.device)
    optim = Adam(model.parameters(), lr=cfg.lr_peak, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    ema = EMA(model, decay=cfg.ema_decay)
    if cfg.path_type == "ot":
        path = get_path("ot", sigma_min=cfg.sigma_min)
    elif cfg.path_type == "vp":
        path = get_path("vp", beta_min=cfg.vp_beta_min, beta_max=cfg.vp_beta_max)
    elif cfg.path_type == "ddpm":
        path = get_path("ddpm", beta_min=cfg.vp_beta_min, beta_max=cfg.vp_beta_max)
    else:
        raise ValueError(cfg.path_type)
    return model, optim, ema, path


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------


def save_ckpt(
    path: Path,
    *,
    step: int,
    model: torch.nn.Module,
    optim: Adam,
    ema: EMA,
    cfg: TrainConfig,
    rng_state: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "ema": ema.state_dict(),
            "cfg": cfg.to_json(),
            "rng": rng_state,
        },
        path,
    )


def load_ckpt(
    path: Path, *, model: torch.nn.Module, optim: Adam, ema: EMA, device: str
) -> tuple[int, dict]:
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model"])
    optim.load_state_dict(state["optim"])
    ema.load_state_dict(state["ema"])
    return state["step"], state["rng"]


def find_latest_ckpt(run_dir: Path) -> Path | None:
    if not run_dir.exists():
        return None
    ckpts = sorted(run_dir.glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    return ckpts[-1] if ckpts else None


# ---------------------------------------------------------------------------
# Lightweight CSV metrics logger (Drive-friendly, independent of wandb)
# ---------------------------------------------------------------------------


def append_csv(path: Path, row: dict[str, float | int]) -> None:
    """Append one row to a CSV, writing the header if the file is new.

    Persists training/eval metrics next to the checkpoints (on Google Drive in
    Colab) so the curves survive a session disconnect even with wandb disabled.
    On resume the file already exists, so we just keep appending. The key order
    of ``row`` must stay stable across calls (it defines the columns).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a") as f:
        if write_header:
            f.write(",".join(row.keys()) + "\n")
        f.write(",".join(str(v) for v in row.values()) + "\n")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(cfg: TrainConfig, *, fid_fn: Callable[[torch.nn.Module], float] | None = None) -> Path:
    """Run the training loop. Returns the path of the final checkpoint."""
    cfg.run_dir = Path(cfg.run_dir)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    (cfg.run_dir / "config.json").write_text(cfg.to_json())

    model, optim, ema, path = build(cfg)
    loss_fn = get_loss_fn(path)

    # Optional wandb
    wb = None
    if cfg.use_wandb:
        import wandb

        wb = wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name or cfg.run_dir.name,
            config=json.loads(cfg.to_json()),
            dir=str(cfg.run_dir),
        )

    # Resume
    start_step = 0
    latest = find_latest_ckpt(cfg.run_dir)
    if latest is not None:
        start_step, rng = load_ckpt(latest, model=model, optim=optim, ema=ema, device=cfg.device)
        torch.set_rng_state(rng["cpu"])
        if torch.cuda.is_available() and "cuda" in rng:
            torch.cuda.set_rng_state(rng["cuda"])
        print(f"[resume] step={start_step} from {latest.name}")

    loader = make_cifar10_loader(cfg)
    data_iter = _infinite(loader)

    autocast_dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[cfg.dtype]
    use_amp = autocast_dtype != torch.float32
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.dtype == "float16")

    accum = max(1, cfg.accum_iter)
    eff_batch = cfg.batch_size * accum
    print(f"[train] physical batch={cfg.batch_size} × accum={accum} → effective batch={eff_batch}")

    model.train()
    t0 = time.time()
    pbar = tqdm(range(start_step, cfg.max_steps), initial=start_step, total=cfg.max_steps)
    for step in pbar:
        # One pbar iteration == one optimiser step over the effective batch.
        for g in optim.param_groups:
            g["lr"] = lr_at(step, cfg)

        optim.zero_grad(set_to_none=True)
        loss_sum = 0.0
        diag_sum: dict[str, float] = {}
        for _ in range(accum):
            x, _ = next(data_iter)
            x = x.to(cfg.device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=autocast_dtype, enabled=use_amp):
                loss, diag = loss_fn(model, x, path, eps=cfg.time_eps)
            scaled = loss / accum  # so accumulated grads match a single effective-batch step
            if cfg.dtype == "float16":
                scaler.scale(scaled).backward()
            else:
                scaled.backward()
            loss_sum += loss.item()
            for k, v in diag.items():
                diag_sum[k] = diag_sum.get(k, 0.0) + v.item()

        if cfg.dtype == "float16":
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optim)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()

        ema.update(model)

        if step % cfg.log_every == 0:
            speed = (step - start_step + 1) / max(1e-6, time.time() - t0)
            loss_mean = loss_sum / accum
            pbar.set_postfix(
                loss=f"{loss_mean:.4f}", lr=f"{lr_at(step, cfg):.2e}", it_s=f"{speed:.2f}"
            )
            append_csv(
                cfg.run_dir / "metrics.csv",
                {
                    "step": step,
                    "loss": round(loss_mean, 6),
                    "lr": round(lr_at(step, cfg), 10),
                    **{k: round(v / accum, 6) for k, v in diag_sum.items() if k != "loss"},
                    "it_per_s": round(speed, 3),
                    "elapsed_s": round(time.time() - t0, 1),
                },
            )
            if wb is not None:
                wb.log(
                    {
                        "train/loss": loss_mean,
                        "train/lr": lr_at(step, cfg),
                        **{f"train/{k}": v / accum for k, v in diag_sum.items()},
                        "train/it_per_s": speed,
                    },
                    step=step,
                )

        if (step + 1) % cfg.ckpt_every == 0 or (step + 1) == cfg.max_steps:
            rng_state = {"cpu": torch.get_rng_state()}
            if torch.cuda.is_available():
                rng_state["cuda"] = torch.cuda.get_rng_state()
            save_ckpt(
                cfg.run_dir / f"ckpt_{step + 1}.pt",
                step=step + 1,
                model=model,
                optim=optim,
                ema=ema,
                cfg=cfg,
                rng_state=rng_state,
            )

        if cfg.eval_every and (step + 1) % cfg.eval_every == 0 and fid_fn is not None:
            with ema.swap_into(model):
                model.eval()
                fid = fid_fn(model)
                model.train()
            append_csv(cfg.run_dir / "fid.csv", {"step": step + 1, "fid": round(fid, 4)})
            if wb is not None:
                wb.log({"eval/fid": fid}, step=step)
            print(f"[eval] step={step + 1} FID(EMA, {cfg.eval_n_samples} samples)={fid:.3f}")

    final = cfg.run_dir / f"ckpt_{cfg.max_steps}.pt"
    if wb is not None:
        wb.finish()
    return final


# ---------------------------------------------------------------------------
# Image utilities for the notebooks
# ---------------------------------------------------------------------------


def denorm(x: Tensor) -> Tensor:
    """Map a tensor from [-1, 1] back to [0, 1] for display / FID."""
    return (x.clamp(-1.0, 1.0) + 1.0) * 0.5


def make_grid_png(samples: Tensor, n_row: int = 8) -> Tensor:
    from torchvision.utils import make_grid

    return make_grid(denorm(samples), nrow=n_row)
