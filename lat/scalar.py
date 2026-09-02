"""Lattice scalar field for single-field inflation.

Implements the split-Hamiltonian (Yoshida) evolution of :file:`scalar_note.md`:
the scalar field ``phi`` with conjugate momenta ``mom_phi`` lives on the
lattice, while the scale factor ``b`` and its momentum ``mom_b`` are global
degrees of freedom.
"""
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from symlat.expr import Expr, Symbol, coords, derivative
from symlat.jit.fn_wrapper import Wrapper, evaluate_expr

lax = Wrapper()

@dataclass
class ScalarField:
    b: np.ndarray
    mom_b: np.ndarray
    phi: np.ndarray
    mom_phi: np.ndarray

    @property
    def dim(self) -> int:
        return self.phi.ndim

# the field symbol used by the potential expressions of :class:`Params`
PHI = Symbol(('phi',))

@dataclass
class Params:
    spacing: tuple[float, ...]  # per-direction lattice spacing h_i = L_i/N_i
    size: tuple[int, ...]       # grid points N_i per direction
    kappa: float
    v: Expr                     # V(phi) (+ Lambda), over PHI and parameter symbols
    other_params: Any           # parameter values referenced by ``v``

    @property
    def dim(self) -> int:
        return len(self.spacing)

    @property
    def h(self) -> float:
        """The lattice spacing ``h`` of a uniform grid (scalar_note.md §5).

        Only meaningful when every ``spacing[i]`` is the same; the geometric
        mean equals that common value then.
        """
        return math.prod(self.spacing) ** (1 / self.dim)

    @property
    def h_vol(self) -> float:
        """The volume of one lattice cell, ``h_vol = prod_i h_i`` (scalar_note.md §1)."""
        return math.prod(self.spacing)

    @property
    def volume(self) -> float:
        """The total spatial volume ``V = prod_i L_i = prod_i (N_i h_i)``."""
        return math.prod(s * dx for s, dx in zip(self.size, self.spacing))

    def _potential_subs(self, field: ScalarField) -> dict[Expr, Any]:
        """Substitutions that evaluate :attr:`v` at every site of ``phi``.

        The parameter symbols of ``v`` are replaced by the traced values of
        ``other_params`` and the field symbol PHI by the traced ``field.phi``.
        """
        ret: dict[Expr, Any] = {Symbol(('param', k)): value for k, value in self.other_params.items()}
        ret[PHI] = field.phi
        return ret

    # --- split-Hamiltonian steps (see scalar_note.md §3) -------------------

    @lax.jit()
    def _apply_k1(self, field: ScalarField, dt: float):
        """K1: ``b <- b - kappa*d/(4*(d-1)*V) * pi_b * tau`` (§3.1)."""
        d = field.dim
        field.b -= self.kappa * d / 4 / (d - 1) / self.volume * field.mom_b * dt

    @lax.jit()
    def _apply_k2(self, field: ScalarField, dt: float):
        """K2: ``phi_i += p_i/(b^2*h_vol)*tau`` and
        ``pi_b += sum(p_i^2)/(b^3*h_vol)*tau`` (§3.2)."""
        field.phi += field.mom_phi / field.b ** 2 / self.h_vol * dt
        field.mom_b += np.sum(field.mom_phi ** 2) / field.b ** 3 / self.h_vol * dt

    @lax.jit()
    def _apply_k3(self, field: ScalarField, dt: float):
        """K3: ``p_i <- p_i - tau*(b^(2-4/d)*S_i + b^2*h_vol*V'(phi_i))`` and
        ``pi_b <- pi_b - tau*((1-2/d)*b^(1-4/d)*D + 2*b*h_vol*sum_i V(phi_i))``,
        with ``S_i`` and ``D`` the weighted nearest-neighbour sums of §3.3."""
        d = field.dim
        h_vol = self.h_vol
        subs = self._potential_subs(field)
        field.mom_phi -= dt * (
            field.b ** (2 - 4 / d) * neighbor_diff(field.phi, self.spacing, h_vol)
            + field.b ** 2 * h_vol * evaluate_expr(derivative(self.v, PHI), subs)
        )
        field.mom_b -= dt * (
            (1 - 2 / d) * field.b ** (1 - 4 / d) * np.sum(neighbor_square(field.phi, self.spacing, h_vol))
            + 2 * field.b * h_vol * np.sum(evaluate_expr(self.v, subs))
        )

def neighbor_diff(phi, spacing: tuple[float, ...], h_vol: float):
    """Per-site weighted neighbour sum ``S_i`` of scalar_note.md §3.3:
    ``sum_k (h_vol/h_k^2) sum_{j in nbr_k(i)} (phi_i - phi_j)``
    = ``sum_k (h_vol/h_k^2) (2 phi_i - phi_{i+1} - phi_{i-1})``."""
    return sum(
        h_vol / dx ** 2 * (2 * phi - np.roll(phi, 1, k) - np.roll(phi, -1, k))
        for k, dx in enumerate(spacing)
    )

def neighbor_square(phi, spacing: tuple[float, ...], h_vol: float):
    """Per-site weighted squared neighbour difference over the undirected pairs:
    ``sum_k (h_vol/h_k^2) (phi_{i+1} - phi_i)^2``.  Its site sum equals the
    ``sum_k (h_vol/h_k^2) sum_{<ij>_k} (phi_i - phi_j)^2`` of §3.3."""
    return sum(
        h_vol / dx ** 2 * (np.roll(phi, 1, k) - phi) ** 2
        for k, dx in enumerate(spacing)
    )

def effective_mom(coord, spacing, size):
    """Effective lattice momentum of a mode.

    Mirrors `effective_mom` in `inflat/src/scalar.rs`.
    """
    return np.sqrt(sum((np.sin(np.pi * c / n) / dx) ** 2 for c, n, dx in zip(coord, size, spacing) for c, n, dx in zip(coord, size, spacing))) * 2

def _rand_complex_normal(shape: tuple[int, ...]) -> np.ndarray:
    ret = np.zeros(shape, dtype=np.complex128)
    r1 = np.random.rand(*shape)
    r2 = np.random.rand(*shape)
    _rand_kernel(ret, r1, r2)
    return ret

@lax.jit()
def _rand_kernel(ret: np.ndarray, r1: np.ndarray, r2: np.ndarray):
    phase = np.cos(r1 * 2 * np.pi) + 1j * np.sin(r1 * 2 * np.pi)
    amp = np.sqrt(-np.log(r2) / 2)
    ret[:] = phase * amp

@lax.jit()
def _gen_phi_tilde_kernel(ret: np.ndarray, ladder: np.ndarray, spacing: tuple[float, ...]):
    x = coords(ret.shape)
    k_eff = effective_mom(x, ret.shape, spacing)


def populate_noise(size: tuple[int, ...], spacing: tuple[float, ...], a: float, v_a: float):
    phase = np.random.randn(*size) + np.random.randn(*size) * 1j
    amp = np.exp(-np.log(np.random.rand(*size)) / 2)
