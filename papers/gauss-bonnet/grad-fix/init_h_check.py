"""Check which (laplacian, gradient) pair the logged initial H actually used.

Reconstructs the N=400 initial state (same BD spectrum) and solves the
Hamiltonian constraint for four variants of the two GB gradient terms:

    lap_xi  = <Delta phi * xi'>   (what lat.rs is *supposed* to give)
            = <(grad phi)^2 * xi'>  (what LatticeLaplacian::get actually gives)
    g2_xi2  = <(grad phi)^2 * xi''>  with grad^2 fwd or symmetric
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import gb_expr as G

XI0, PHI_C, XI1 = 1.13412e7, 9.6, 30.0
LAMBDA, F = 0.0065, 7.0
L_BOX = 400 * 447.4167505606408
PHI0, VPHI0, H_BG = 9.420104290699477, 3.05768324485467e-6, 2.7006884377829143e-5
H_LOG = 2.7190104429563513e-5


def V(p):
    return LAMBDA**4 * (1.0 + np.cos(p / F))


def xi(p):
    return XI0 * np.tanh(XI1 * (p - PHI_C))


def xid(p):
    s = np.cosh(XI1 * (p - PHI_C))
    return XI0 * XI1 / s / s


def xidd(p):
    t = np.tanh(XI1 * (p - PHI_C))
    s = np.cosh(XI1 * (p - PHI_C))
    return -2.0 * XI0 * XI1 * XI1 * t / s / s


def keff(n, dx):
    s1 = np.sin(np.pi * np.arange(n) / n) ** 2
    s = s1[:, None, None] + s1[None, :, None] + s1[None, None, :]
    s *= (2.0 / dx) ** 2
    return np.sqrt(s, out=s)


def init_state(n, dx, seed=10, v_a=H_BG):
    rng = np.random.default_rng(seed)
    c = (rng.standard_normal((n, n, n)) + 1j * rng.standard_normal((n, n, n))) / np.sqrt(2.0)
    neg = np.roll(c[::-1, ::-1, ::-1], 1, axis=(0, 1, 2))
    gp, gm = (c + np.conj(neg)) / np.sqrt(2.0), (c - np.conj(neg)) / np.sqrt(2.0)
    gp[0, 0, 0] = 0.0
    gm[0, 0, 0] = 0.0
    k = keff(n, dx)
    vol = (n * dx) ** 3
    ks = np.where(k > 0.0, k, 1.0)
    u = np.where(k > 0.0, 1.0 / np.sqrt(vol) / np.sqrt(2.0 * ks), 0.0)
    ud = np.where(k > 0.0, -1j * np.sqrt(ks / 2.0) / np.sqrt(vol), 0.0)
    phi = np.fft.fftn(gp * u).real
    vphi = np.fft.fftn(gm * ud).real - phi * v_a
    return phi, vphi


def grads(phi, dx):
    lap = gf = gb = 0.0
    for d in range(3):
        p, m = np.roll(phi, -1, axis=d), np.roll(phi, +1, axis=d)
        lap = lap + (p + m - 2.0 * phi) / dx / dx
        gf = gf + ((p - phi) / dx) ** 2
        gb = gb + ((m - phi) / dx) ** 2
    return lap, gf, gb


def main(n=400, seed=10):
    dx = L_BOX / n
    dphi, dvphi = init_state(n, dx, seed)
    phi, vphi = PHI0 + dphi, VPHI0 + dvphi
    lap, gf, gb = grads(phi, dx)
    gs = 0.5 * (gf + gb)
    x1, x2 = xid(phi), xidd(phi)
    common = dict(kappa=1.0, v_phi2=(vphi**2).mean(), V=V(phi).mean(),
                  vphi_xi=(vphi * x1).mean(), xi=xi(phi).mean())
    print(f"N={n} seed={seed}  <g_fwd>={gf.mean():.6e} <g_sym>={gs.mean():.6e}")
    print(f"  <lap*xi'>   = {(lap * x1).mean():+.6e}   (correct, after fixing LatticeLaplacian)")
    print(f"  <g_fwd*xi'> = {(gf * x1).mean():+.6e}   (what the current code gives)")
    print(f"  <g_sym*xi'> = {(gs * x1).mean():+.6e}")
    print(f"  <g_fwd*xi''>={ (gf * x2).mean():+.6e}  <g_sym*xi''>={(gs * x2).mean():+.6e}")
    print("  identity <lap*xi'> + <g_sym*xi''> = "
          f"{(lap * x1).mean() + (gs * x2).mean():+.3e}")
    rows = [
        ("current code        (lap<-g_fwd, g_fwd)", (gf * x1).mean(), gf.mean(), (gf * x2).mean()),
        ("grad fix only       (lap<-g_sym, g_sym)", (gs * x1).mean(), gs.mean(), (gs * x2).mean()),
        ("lap fix only        (lap,        g_fwd)", (lap * x1).mean(), gf.mean(), (gf * x2).mean()),
        ("both fixes          (lap,        g_sym)", (lap * x1).mean(), gs.mean(), (gs * x2).mean()),
    ]
    for name, lapxi, g, g2 in rows:
        H = G.solve_hubble(1.0, g, common["v_phi2"], common["V"], lapxi, g2,
                           common["vphi_xi"], common["xi"])
        print(f"  {name}: H={H:.10e}  H/H_bg-1={H/H_BG-1:+.3e}  vs log {H/H_LOG-1:+.2e}")


if __name__ == "__main__":
    main()
