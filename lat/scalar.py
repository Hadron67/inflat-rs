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

from symlat.expr import Expr, Symbol, derivative
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
    """Effective lattice momentum (dispersion) of a mode.

    Mirrors `effective_mom` in `inflat/src/scalar.rs`: with the lattice
    Laplacian on a grid of ``size`` points and per-axis spacings ``spacing``,

        omega = 2 * sqrt(sum_i (sin(pi*c_i/n_i) / dx_i)^2)

    ``coord`` holds integer mode numbers ``c_i`` in ``[0, n_i)``; it may be a
    tuple of scalars or, for vectorized use, per-axis arrays of the mode-grid
    shape produced by ``np.indices(size)``.
    """
    return 2 * np.sqrt(sum((np.sin(np.pi * c / n) / dx) ** 2 for c, n, dx in zip(coord, size, spacing)))


def _rand_complex_normal(shape: tuple[int, ...]) -> np.ndarray:
    """I.i.d. complex Gaussians of variance 1/2 over ``shape``.

    Box--Muller: ``a = sqrt(-ln X / 2) e^{2 pi i Y}`` with ``X, Y`` uniform on
    ``[0, 1)``, i.e. both quadratures have variance 1/4 and ``E|a|^2 = 1/2``.
    """
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


def _conj_pair(amp: np.ndarray) -> np.ndarray:
    r"""Hermitian-symmetric completion ``c[m] = a[m] + conj(a[-m])``.

    Each mode index ``m`` is mapped to ``-m mod N`` along every axis, so that
    a field built from the coefficients ``c`` via ``sum_m c[m] e^{2 pi i m.n/N}``
    is real: the coefficient of ``e^{+ik.x}`` picks up ``a_k u + a_{-k}^* u*``,
    i.e. the ``a_k u_k + a_{-k}^\dagger u_k^*`` term of the mode expansion.
    """
    neg = amp
    for axis in range(amp.ndim):
        # flip maps m -> N-1-m, rolling by one then gives m -> N-m = -m mod N
        neg = np.roll(np.flip(neg, axis=axis), 1, axis=axis)
    return amp + np.conj(neg)


def populate_noise(
    size: tuple[int, ...],
    spacing: tuple[float, ...],
    a: float,
    v_a: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Realize the vacuum-noise initial condition of the scalar field.

    Mirrors `populate_noise` in `inflat/src/scalar.rs`.  The lattice field
    ``phi`` is evolved in cosmological time ``t`` on the FLRW background
    ``ds^2 = -dt^2 + a^2(t) dx^2`` (see scalar_note.md).  Its fluctuations are
    quantized through the canonically normalized field
    ``v = a^{(d-1)/2} phi``, whose sub-horizon mode functions in conformal
    time ``tau`` (``dt = a dtau``) are ``u_k(tau) = e^{-i omega_k tau} /
    sqrt(2 omega_k)``, ``omega`` being the lattice dispersion
    :func:`effective_mom`.  Sampling at ``tau = 0``,

        v(x) = 1/sqrt(V) sum_{k != 0} (a_k u_k + a_{-k}^* u_k^*) e^{i k.x},

    with i.i.d. complex Gaussian ``a_k`` of variance 1/2 (:func:`_rand_complex_normal`),
    the zero mode excluded and ``V`` the comoving volume, and converting from
    conformal to cosmological time yields the returned
    ``(noise_phi, noise_v_phi)`` = ``(phi, d phi/dt)`` at the initial time:

        phi      = v / a^n
        dphi/dt  = (v' - n v_a v) / a^(n+1),    n = (d-1)/2,  v_a = da/dt

    (at ``d = 3`` this reduces to the ``/a`` and ``/a^2`` factors of the Rust
    reference).  To seed a :class:`ScalarField` the caller combines these with
    a homogeneous background: ``phi += noise_phi`` and
    ``mom_phi += noise_v_phi * a**d * h_vol``.

    Args:
        size: grid points per direction, ``N_i``.
        spacing: lattice spacing per direction, ``h_i``.
        a: scale factor at the initial time.
        v_a: ``da/dt`` at the initial time (cosmological time).
    """
    dim = len(size)
    n_total = math.prod(size)
    volume = math.prod(n * h for n, h in zip(size, spacing))
    inv_sqrt_volume = 1.0 / math.sqrt(volume)
    # per-mode lattice frequency omega(m); the zero mode (omega = 0) is skipped
    omega = effective_mom(np.indices(size), spacing, size)
    u = np.divide(
        inv_sqrt_volume,
        np.sqrt(2 * omega),
        out=np.zeros_like(omega),
        where=omega > 0,
    )
    u_d = -1j * omega * u  # d u_k / d tau at tau = 0
    a_hat = _rand_complex_normal(size)  # a_k, E|a_k|^2 = 1/2
    # v and dv/dtau at tau = 0 (ifftn carries the 1/N of the inverse DFT)
    v = np.fft.ifftn(_conj_pair(a_hat * u) * n_total * inv_sqrt_volume).real
    v_prime = np.fft.ifftn(_conj_pair(a_hat * u_d) * n_total * inv_sqrt_volume).real
    n = (dim - 1) / 2
    noise_phi = v / a**n
    noise_v_phi = (v_prime - n * v_a * v) / a ** (n + 1)
    return noise_phi, noise_v_phi
