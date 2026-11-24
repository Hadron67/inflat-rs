"""Extract verbatim return expressions of data::vv_phi / data::hubble2 /
data::hubble_constraint_coefs from inflat/src/gauss_bonnet.rs."""
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
src = (ROOT / "inflat/src/gauss_bonnet.rs").read_text().splitlines()


def grab(prefix):
    for line in src:
        if line.strip().startswith(prefix):
            return line.strip()
    raise SystemExit("not found: " + prefix)


vv = grab("(64.0 * d * laplacian_div_a2")
hb = grab("-1.0 / 4.0 * 1.0 / ((-1.0 + d)) * 1.0 / ((16.0 * d + (-2.0 + d)")

i0 = next(i for i, l in enumerate(src) if l.strip().startswith("pub fn hubble_constraint_coefs("))
i0 = next(i for i in range(i0, len(src)) if src[i].strip() == "[")
i1 = next(i for i in range(i0 + 1, len(src)) if src[i].strip() == "]")
body = " ".join(l.strip() for l in src[i0 + 1:i1])
coefs = [c.strip() for c in body.split(",") if c.strip()]

out = ["from math import *",
       "# d = 3 is the number of SPATIAL dimensions (GaussBonnetBInput.dim).",
       "d = 3.0", ""]
out += ["def vv_phi(kappa, a, v_a, phi, v_phi, laplacian_phi, derivative2_phi,",
        "            xi_phi, xi_d_phi, xi_dd_phi, pv_phi, pv_d_phi):",
        "    derivative2_div_a2 = derivative2_phi / a / a",
        "    laplacian_div_a2 = laplacian_phi / a / a",
        "    hubble = v_a / a",
        "    return " + vv, ""]
out += ["def hubble2(kappa, a, v_a, phi, v_phi, laplacian_phi, derivative2_phi,",
        "            xi_phi, xi_d_phi, xi_dd_phi, pv_phi, pv_d_phi):",
        "    laplacian_div_a2 = laplacian_phi / a / a",
        "    derivative2_div_a2 = derivative2_phi / a / a",
        "    hubble = v_a / a",
        "    return " + hb, ""]
out += ["def hubble_constraint_coefs(kappa, der2_phi, v_phi2, potential_phi,",
        "        laplacian_phi_xi_d_phi, der2_phi_xi_dd_phi, v_phi_xi_d_phi, xi_phi):"]
out += [f"    c{i} = {c}" for i, c in enumerate(coefs)]
out += ["    return [c0, c1, c2, c3, c4]", "",
        "def solve_hubble(kappa, der2, v_phi2, V, lap_xi, g2_xi2, vphi_xi, xi, guess=None):",
        "    h = guess if guess is not None else (kappa / d / (d - 1.0) * (der2 + v_phi2 + 2.0 * V))**0.5",
        "    for _ in range(200):",
        "        c = hubble_constraint_coefs(kappa, der2, v_phi2, V, lap_xi, g2_xi2, vphi_xi, xi)",
        "        fv = c[0] + c[2] * h**2 + c[3] * h**3 + c[4] * h**4",
        "        fp = 2 * c[2] * h + 3 * c[3] * h**2 + 4 * c[4] * h**3",
        "        step = fv / fp",
        "        h -= step",
        "        if abs(step) < 1e-14 * abs(h):",
        "            break",
        "    return h"]
(HERE / "gb_expr.py").write_text("\n".join(out) + "\n")
print("wrote", len(coefs), "coefs")
