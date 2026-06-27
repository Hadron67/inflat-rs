from dataclasses import dataclass
from typing import Callable

import numpy as np
import jax

from .expr import Expr, Symbol

@dataclass
class ScalarField:
    b: float
    mom_b: float
    phi: np.ndarray
    mom_phi: np.ndarray

@dataclass
class Params:
    dim: int
    kappa: float
    v: Callable[[Expr], Expr]

    def _apply_k1(self, field: ScalarField, dt: float):
        field.b += field.mom_b
