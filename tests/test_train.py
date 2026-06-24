"""Resume-path regression tests for checkpoint save/load.

Guards the bug where ``torch.load(map_location="cuda")`` moved the RNG state off
the CPU, making ``torch.set_rng_state`` raise "RNG state must be a torch.ByteTensor".
``load_ckpt`` must always return CPU uint8 ByteTensors so resume works.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.optim import Adam

from flow_matching_b3.ema import EMA
from flow_matching_b3.train import TrainConfig, load_ckpt, save_ckpt


def _tiny():
    model = nn.Linear(4, 4)
    optim = Adam(model.parameters(), lr=1e-3)
    ema = EMA(model, decay=0.99)
    return model, optim, ema


def test_load_ckpt_returns_usable_cpu_rng(tmp_path) -> None:
    model, optim, ema = _tiny()
    optim.step()  # populate optimiser state so the round trip is non-trivial
    rng_state = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        rng_state["cuda"] = torch.cuda.get_rng_state()

    ckpt = tmp_path / "ckpt_100.pt"
    save_ckpt(ckpt, step=100, model=model, optim=optim, ema=ema, cfg=TrainConfig(), rng_state=rng_state)

    m2, o2, e2 = _tiny()
    step, rng = load_ckpt(ckpt, model=m2, optim=o2, ema=e2, device="cpu")
    assert step == 100

    # The whole point: every RNG tensor is a CPU ByteTensor and set_rng_state accepts it.
    for key, tensor in rng.items():
        assert tensor.device.type == "cpu", key
        assert tensor.dtype == torch.uint8, key
    torch.set_rng_state(rng["cpu"])  # must not raise
