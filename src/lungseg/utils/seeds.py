"""Determinism helpers.

`set_global_determinism` wraps MONAI's `set_determinism` and exposes a
`seed_worker` function suitable for `torch.utils.data.DataLoader`'s
`worker_init_fn`. The legacy pipeline only called `set_determinism` at module
import which did not propagate the seed to multiprocessing workers; B4 will
hook these helpers into the trainer.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch
from monai.utils.misc import set_determinism


def set_global_determinism(seed: int) -> None:
    """Seed every RNG that touches the training stack."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_determinism(seed=seed)


def seed_worker(worker_id: int) -> None:
    """`worker_init_fn` for DataLoader workers. Derives the worker seed
    from the parent's `torch.initial_seed()` so each worker is deterministic
    yet uncorrelated with its siblings.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
