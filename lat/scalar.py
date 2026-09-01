import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from symlat.expr import Expr, Symbol, coords, derivative
from symlat.jit.fn_wrapper import Wrapper

lax = Wrapper()

def laplacian(field, spacing):
    return sum((np.roll(field, 1, i) + np.roll(field, -1, i) - field * 2) / dx for i, dx in enumerate(spacing))

def derivative_square(field, spacing):
    return sum((np.roll(field, -1, i) - field) / dx for i, dx in enumerate(spacing))

@dataclass
class ScalarField:
    b: np.ndarray
    mom_b: np.ndarray
    phi: np.ndarray
    mom_phi: np.ndarray

    @property
    def dim(self) -> int:
        return self.phi.ndim

PHI = Symbol(('phi',))

@dataclass
class Params:
    spacing: tuple[float, ...]
    size: tuple[int, ...]
    kappa: float
    v: Expr
    other_params: Any

    @property
    def dim(self) -> int:
        return len(self.spacing)

    @property
    def h(self) -> float:
        """Lattice spacing ``h`` (the note assumes a uniform grid)."""
        return math.prod(self.spacing) ** (1 / self.dim)

    @property
    def h_d(self) -> float:
        """Volume of one lattice cell, ``h^d``."""
        return math.prod(self.spacing)

    @property
    def volume(self) -> float:
        """Total spatial volume ``V = L^d``."""
        return math.prod(s * dx for s, dx in zip(self.size, self.spacing))

    def _make_potential(self, expr: Expr, phi) -> Expr:
        reps: dict[Expr, Expr] = {Symbol(('param', k)): Expr.as_expr(v) for k, v in dict(self.other_params).items()}
        reps[PHI] = phi
        return expr.replace(reps)

    # --- split-Hamiltonian steps (see scalar_note.md) ----------------------
    # the array updates (phi, mom_phi) loop over the lattice, while the global
    # updates (b, mom_b) are scalar kernels, so each K step is split into one
    # jitted kernel per equation

    @lax.jit()
    def _k1(self, field: ScalarField, dt: float):
        """K1: b <- b - kappa*d/(4*(d-1)*V) * pi_b * tau."""
        d = field.dim
        field.b -= self.kappa * d / 4 / (d - 1) / self.volume * field.mom_b * dt

    @lax.jit()
    def _k2_phi(self, field: ScalarField, dt: float):
        """K2: phi_i <- phi_i + p_i/(b^2*h^d) * tau."""
        field.phi += field.mom_phi / field.b ** 2 / self.h_d * dt

    @lax.jit()
    def _k2_mom_b(self, field: ScalarField, dt: float):
        """K2: pi_b <- pi_b + sum(p_i^2)/(b^3*h^d) * tau."""
        field.mom_b += np.sum(field.mom_phi ** 2) / field.b ** 3 / self.h_d * dt

    @lax.jit()
    def _k3_mom_phi(self, field: ScalarField, dt: float):
        """K3: p_i <- p_i - tau*(b^(2-4/d)*h^(d-2)*sum_{j~i}(phi_i-phi_j) + b^2*h^d*V'(phi))."""
        d = field.dim
        h = self.h
        vd = self._make_potential(derivative(self.v, PHI), field.phi._expr)  # type: ignore[attr-defined]
        field.mom_phi += dt * (
            field.b ** (2 - 4 / d) * h ** (d - 1) * laplacian(field.phi, self.spacing)
            - field.b ** 2 * self.h_d * vd
        )

    @lax.jit()
    def _k3_mom_b(self, field: ScalarField, dt: float):
        """K3: pi_b <- pi_b - tau*((1-2/d)*b^(1-4/d)*h^(d-2)*sum_{<ij>}(phi_i-phi_j)^2
        + 2*b*h^d*sum_i(V(phi_i)+Lambda))."""
        d = field.dim
        # the potential at each site, with the parameter symbols and PHI
        # replaced by the parameter values and the traced field symbol
        v = self._make_potential(self.v, field.phi._expr)  # type: ignore[attr-defined]
        # np.sum only reduces probe expressions; wrap the bare expression in a
        # probe that normalizes to the expression itself (``phi * 0`` -> 0)
        v_probe = field.phi * 0 + v
        # sum of squared neighbor differences over undirected lattice pairs
        dphi2 = field.phi * 0
        for i in range(d):
            dphi2 = dphi2 + (np.roll(field.phi, 1, i) - field.phi) ** 2
        field.mom_b -= dt * (
            (1 - 2 / d) * field.b ** (1 - 4 / d) * self.h ** (d - 2) * np.sum(dphi2)
            + 2 * field.b * self.h_d * np.sum(v_probe)
        )

    def _apply_k1(self, field: ScalarField, dt: float):
        """The K1 step of the split Hamiltonian."""
        self._k1(field, dt)

    def _apply_k2(self, field: ScalarField, dt: float):
        """The K2 step of the split Hamiltonian."""
        self._k2_phi(field, dt)
        self._k2_mom_b(field, dt)

    def _apply_k3(self, field: ScalarField, dt: float):
        """The K3 step of the split Hamiltonian."""
        self._k3_mom_phi(field, dt)
        self._k3_mom_b(field, dt)

def effective_mom(coord, spacing, size):
    """Effective lattice momentum of a mode.

    Mirrors `effective_mom` in `inflat/src/scalar.rs`.
    """
    ret = 0.0
    for c, n, dx in zip(coord, size, spacing):
        aa = np.sin(np.pi * c / n)
        ret += aa * aa / dx / dx
    return np.sqrt(ret) * 2

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
