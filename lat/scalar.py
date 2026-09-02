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
    """Effective lattice momentum (dispersion) of a mode.

    Mirrors `effective_mom` in `inflat/src/scalar.rs`: with the lattice
    Laplacian on a grid of ``size`` points and per-axis spacings ``spacing``,

        omega = 2 * sqrt(sum_i (sin(pi*c_i/n_i) / dx_i)^2)

    ``coord`` holds integer mode numbers ``c_i`` in ``[0, n_i)``; it may be a
    tuple of scalars or, for vectorized use, per-axis arrays of the mode-grid
    shape produced by ``np.indices(size)``.
    """
    return 2 * np.sqrt(sum((np.sin(np.pi * c / n) / dx) ** 2 for c, n, dx in zip(coord, size, spacing)))


def _box_muller_amp(r1: np.ndarray, r2: np.ndarray):
    """A Box--Muller complex Gaussian ``a = sqrt(-ln X / 2) e^{2 pi i Y}``.

    ``r1``/``r2`` hold uniform draws on ``[0, 1)`` (``r1`` feeds the phase
    ``Y``, ``r2`` the amplitude ``X``), so ``E|a|^2 = 1/2``.  Works on plain
    arrays as well as on traced values inside a jitted function.
    """
    return np.sqrt(-np.log(r2) / 2) * (np.cos(2 * np.pi * r1) + 1j * np.sin(2 * np.pi * r1))


@lax.jit()
def _fill_noise_modes(v: np.ndarray, vp: np.ndarray, r1: np.ndarray, r2: np.ndarray, size: tuple[int, ...], spacing: tuple[float, ...]):
    """Fill the mode-space coefficient grids of ``v`` and ``dv/dtau`` in place.

    Everything is evaluated per mode index while the kernel walks the grid, so
    no intermediate mode-space arrays are materialised: the per-mode dispersion
    ``omega`` (see :func:`effective_mom`) is computed element-wise from the
    loop coordinates, the draws ``a_m`` are rebuilt by Box--Muller from the
    uniform ``r1``/``r2`` grids and ``a_{-m}`` from those grids read at the
    negated index (a flip plus a roll along every axis).  The grids hold the
    conjugate-pair coefficients of the mode expansion at ``tau = 0``, scaled by
    the number of modes so that an inverse FFT of them directly yields the
    real-space fields (the transform carries the ``1/N``).  The zero mode
    (``omega = 0``) is excluded with an ``If`` guard that keeps the reciprocal
    root ``1/sqrt(2 omega)`` unevaluated there.

    ``size``/``spacing`` are compile-time constants (the kernel is specialised
    per lattice geometry).
    """
    dim = len(size)
    n_total = math.prod(size)
    # the coordinates of the current mode index along every axis
    cs = coords(v.shape)
    # dispersion of the lattice Laplacian at this mode
    omega = effective_mom(cs, spacing, size)
    # u = 1/sqrt(V) / sqrt(2 omega) (the If keeps the zero mode, where
    # omega = 0, from evaluating the reciprocal root)
    inv_sqrt_volume = 1.0 / math.sqrt(math.prod(n * dx for n, dx in zip(size, spacing)))
    u = np.where(omega > 0, inv_sqrt_volume / (2 * omega).sqrt(), 0)
    # du/dtau at tau = 0 is -i omega u, purely imaginary
    u_d = (-1j) * (omega * u)
    # a_m is drawn from the current elements, a_{-m} from the negated indices;
    # conj(u_d) = -u_d turns the derivative's conjugate pair into a difference
    a = _box_muller_amp(r1, r2)
    neg_axes = tuple(range(dim))
    a_neg_c = np.conj(
        _box_muller_amp(np.roll(np.flip(r1), 1, axis=neg_axes), np.roll(np.flip(r2), 1, axis=neg_axes))
    )
    v[:] = (a + a_neg_c) * u * n_total
    vp[:] = (a - a_neg_c) * u_d * n_total


@lax.jit()
def _noise_to_cosmo_time(v: np.ndarray, vp: np.ndarray, k: float, an1: float, an: float):
    """Map the ``tau = 0`` phase-space fields into cosmological time, in place.

    With ``n = (d-1)/2``, ``an = a^n`` and ``an1 = a^(n+1)``,
    ``k = n (da/dt) / a^(n+1)``: ``phi = v / an`` and
    ``dphi/dt = (v' - n (da/dt) v) / an1``.
    """
    vp /= an1
    vp -= k * v
    v /= an


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

    with i.i.d. complex Gaussian ``a_k`` of variance 1/2 (Box--Muller),
    the zero mode excluded and ``V`` the comoving volume, and converting from
    conformal to cosmological time yields the returned
    ``(noise_phi, noise_v_phi)`` = ``(phi, d phi/dt)`` at the initial time:

        phi      = v / a^n
        dphi/dt  = (v' - n v_a v) / a^(n+1),    n = (d-1)/2,  v_a = da/dt

    (at ``d = 3`` this reduces to the ``/a`` and ``/a^2`` factors of the Rust
    reference).  To seed a :class:`ScalarField` the caller combines these with
    a homogeneous background: ``phi += noise_phi`` and
    ``mom_phi += noise_v_phi * a**d * h_vol``.

    To keep the temporary arrays down, the mode-coefficient grids of ``v`` and
    ``v'`` are computed directly by one jitted kernel
    (:func:`_fill_noise_modes`) into preallocated buffers -- including the
    Box--Muller draws and the per-mode dispersion -- and only the two inverse
    FFTs and the final in-place rescaling allocate additional space.

    Args:
        size: grid points per direction, ``N_i``.
        spacing: lattice spacing per direction, ``h_i``.
        a: scale factor at the initial time.
        v_a: ``da/dt`` at the initial time (cosmological time).
    """
    shape = tuple(size)
    # uniform draws shared by the modes of both fields
    r1 = np.random.rand(*shape)
    r2 = np.random.rand(*shape)
    # Hermitian mode-coefficient grids of v and dv/dtau, scaled by the number
    # of modes; one jitted kernel fills both without intermediate arrays
    v = np.empty(shape, dtype=np.complex128)
    vp = np.empty(shape, dtype=np.complex128)
    _fill_noise_modes(v, vp, r1, r2, size, spacing)
    # real-space v and dv/dtau at tau = 0, then the in-place conversion to the
    # cosmological-time phase-space fields
    n = (len(size) - 1) / 2
    an1 = a ** (n + 1)
    noise_phi = np.ascontiguousarray(np.fft.ifftn(v).real)
    noise_v_phi = np.ascontiguousarray(np.fft.ifftn(vp).real)
    _noise_to_cosmo_time(noise_phi, noise_v_phi, (n * v_a) / an1, an1, a**n)
    return noise_phi, noise_v_phi
