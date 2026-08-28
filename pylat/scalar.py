from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from pylat.jit.fn_wrapper import Wrapper

from .expr import Expr

lax = Wrapper()

@lax.jit()
def _plus_assign_dt(a, b, dt):
    a += b * dt

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
        _plus_assign_dt(field.b, field.mom_b, dt)
