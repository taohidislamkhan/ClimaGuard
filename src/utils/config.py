"""Read ``params.yaml``. Stage scripts take every tunable value from here."""

from __future__ import annotations

import os
import random
from functools import lru_cache

import numpy as np
import yaml

from .paths import PARAMS


@lru_cache(maxsize=1)
def load_params() -> dict:
    with open(PARAMS, encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_everything(seed: int) -> None:
    """Pin the global RNGs. Every model also gets ``random_state=seed``."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
