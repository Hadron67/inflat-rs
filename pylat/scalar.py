from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .expr import Expr
from .jit.fn_wrapper import Wrapper

lax = Wrapper()

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
