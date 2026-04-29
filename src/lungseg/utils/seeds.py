"""Ayudantes de determinismo.

`set_global_determinism` envuelve el `set_determinism` de MONAI y expone una
función `seed_worker` adecuada para el `worker_init_fn` de `torch.utils.data.DataLoader`.
La tubería heredada solo llamaba a `set_determinism` al importar el módulo,
lo que no propagaba la semilla a los trabajadores de multiprocesamiento; B4
conectará estos ayudantes al entrenador.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch
from monai.utils.misc import set_determinism


def set_global_determinism(seed: int) -> None:
    """Siembra cada RNG que toque la pila de entrenamiento."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_determinism(seed=seed)


def seed_worker(worker_id: int) -> None:
    """`worker_init_fn` para los trabajadores del DataLoader. Deriva la semilla del trabajador
    de la `torch.initial_seed()` del padre para que cada trabajador sea determinista
    pero no esté correlacionado con sus hermanos.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
