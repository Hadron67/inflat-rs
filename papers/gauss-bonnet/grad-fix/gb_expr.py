from math import *
# d = 3 is the number of SPATIAL dimensions (GaussBonnetBInput.dim).
d = 3.0

def vv_phi(kappa, a, v_a, phi, v_phi, laplacian_phi, derivative2_phi,
            xi_phi, xi_d_phi, xi_dd_phi, pv_phi, pv_d_phi):
    derivative2_div_a2 = derivative2_phi / a / a
    laplacian_div_a2 = laplacian_phi / a / a
    hubble = v_a / a
    return (64.0 * d * laplacian_div_a2 + 16.0 * (-3.0 + d) * (-2.0 + d) * d * d * hubble * hubble * hubble * kappa * v_phi * xi_phi + 32.0 * (-2.0 + d) * laplacian_div_a2 * laplacian_div_a2 * kappa * xi_d_phi + (-3.0 + d) * (-2.0 + d) * (-2.0 + d) * (-1.0 + d) * d * d * hubble * hubble * hubble * hubble * hubble * hubble * kappa * xi_phi * xi_d_phi + -2.0 * (-3.0 + d) * (-2.0 + d) * (-2.0 + d) * (-1.0 + d) * d * d * hubble * hubble * hubble * hubble * hubble * kappa * v_phi * xi_d_phi * xi_d_phi + 8.0 * (-2.0 + d) * d * hubble * hubble * kappa * (2.0 * (-3.0 + d) * xi_phi * pv_d_phi + (2.0 * derivative2_div_a2 + -1.0 * derivative2_div_a2 * d + 5.0 * d * v_phi * v_phi + -2.0 * d * pv_phi) * xi_d_phi) + 2.0 * (-2.0 + d) * laplacian_div_a2 * kappa * (-8.0 * (-3.0 + d) * d * hubble * hubble * xi_phi + -32.0 * d * hubble * v_phi * xi_d_phi + -16.0 * pv_d_phi * xi_d_phi + (-3.0 + d) * (-2.0 + d) * (-1.0 + d) * d * hubble * hubble * hubble * hubble * xi_d_phi * xi_d_phi + 16.0 * derivative2_div_a2 * xi_dd_phi) + -32.0 * pv_d_phi * (2.0 * d + derivative2_div_a2 * (-2.0 + d) * kappa * xi_dd_phi) + 2.0 * (-2.0 + d) * (-1.0 + d) * d * hubble * hubble * hubble * hubble * xi_d_phi * (2.0 * (-1.0 + d) * d + (-2.0 + d) * kappa * (derivative2_div_a2 * (-3.0 + d) + -2.0 * d * v_phi * v_phi) * xi_dd_phi) + 32.0 * d * hubble * v_phi * (-2.0 * d + (-2.0 + d) * kappa * (pv_d_phi * xi_d_phi + -1.0 * derivative2_div_a2 * xi_dd_phi))) / (64.0 * d + 4.0 * (-2.0 + d) * kappa * (-4.0 * (-3.0 + d) * d * hubble * hubble * xi_phi + 8.0 * (laplacian_div_a2 + -1.0 * d * hubble * v_phi) * xi_d_phi + d * d * (2.0 + -3.0 * d + d * d) * hubble * hubble * hubble * hubble * xi_d_phi * xi_d_phi + 8.0 * derivative2_div_a2 * xi_dd_phi))

def hubble2(kappa, a, v_a, phi, v_phi, laplacian_phi, derivative2_phi,
            xi_phi, xi_d_phi, xi_dd_phi, pv_phi, pv_d_phi):
    laplacian_div_a2 = laplacian_phi / a / a
    derivative2_div_a2 = derivative2_phi / a / a
    hubble = v_a / a
    return -1.0 / 4.0 * 1.0 / ((-1.0 + d)) * 1.0 / ((16.0 * d + (-2.0 + d) * kappa * (-4.0 * (-3.0 + d) * d * hubble * hubble * xi_phi + xi_d_phi * (8.0 * laplacian_div_a2 + -8.0 * d * hubble * v_phi + (-2.0 + d) * (-1.0 + d) * d * d * hubble * hubble * hubble * hubble * xi_d_phi) + 8.0 * derivative2_div_a2 * xi_dd_phi))) * (-64.0 * d * kappa * pv_phi + 48.0 * (-2.0 + d) * (-1.0 + d) * d * hubble * hubble * hubble * kappa * v_phi * xi_d_phi + 16.0 * d * kappa * v_phi * v_phi * (2.0 + -1.0 * (-2.0 + d) * (-1.0 + d) * hubble * hubble * xi_dd_phi) + (-2.0 + d) * (32.0 * (-1.0 + d) * d * hubble * hubble + -32.0 * derivative2_div_a2 * kappa + (-1.0 + d) * hubble * hubble * kappa * (-4.0 * (-4.0 + d) * (-3.0 + d) * d * hubble * hubble * xi_phi + xi_d_phi * (-48.0 * laplacian_div_a2 + 16.0 * d * pv_d_phi + (-3.0 + d) * (-2.0 + d) * (-1.0 + d) * d * d * hubble * hubble * hubble * hubble * xi_d_phi) + 16.0 * derivative2_div_a2 * (-3.0 + d) * xi_dd_phi)))

def hubble_constraint_coefs(kappa, der2_phi, v_phi2, potential_phi,
        laplacian_phi_xi_d_phi, der2_phi_xi_dd_phi, v_phi_xi_d_phi, xi_phi):
    c0 = 1.0 / 2.0 * (der2_phi + v_phi2) + potential_phi
    c1 = 0.0
    c2 = -1.0 / 4.0 * (-1.0 + d) * 1.0 / (kappa) * (2.0 * d + (-2.0 + d) * kappa * (laplacian_phi_xi_d_phi + der2_phi_xi_dd_phi))
    c3 = 1.0 / 4.0 * (-2.0 + d) * (-1.0 + d) * d * v_phi_xi_d_phi
    c4 = 1.0 / 16.0 * (-3.0 + d) * (-2.0 + d) * (-1.0 + d) * d * xi_phi
    return [c0, c1, c2, c3, c4]

def solve_hubble(kappa, der2, v_phi2, V, lap_xi, g2_xi2, vphi_xi, xi, guess=None):
    h = guess if guess is not None else (kappa / d / (d - 1.0) * (der2 + v_phi2 + 2.0 * V))**0.5
    for _ in range(200):
        c = hubble_constraint_coefs(kappa, der2, v_phi2, V, lap_xi, g2_xi2, vphi_xi, xi)
        fv = c[0] + c[2] * h**2 + c[3] * h**3 + c[4] * h**4
        fp = 2 * c[2] * h + 3 * c[3] * h**2 + 4 * c[4] * h**3
        step = fv / fp
        h -= step
        if abs(step) < 1e-14 * abs(h):
            break
    return h
