use bincode::{Decode, Encode};
use libm::sqrt;
use random::Source;

use crate::{
    background::{
        BackgroundFn, BackgroundSolver, Dimension, Dt, Interpolate, Kappa, Phi, PhiD, ScaleFactor,
        ScaleFactorD,
    },
    c2fn::C2Fn,
    interpolate_fields,
    lat::{BoxLattice, Lattice, LatticeMut, LatticeParam, LatticeSupplier},
    scalar::populate_noise,
    util::{VecN, derivative_2, evaluate_polynomial},
};

pub mod data {
    use crate::{c2fn::C2Fn, util::newton_solve_polynomial};

    #[rustfmt::skip]
    pub fn vv_phi<V, Xi>(dim: usize, kappa: f64, a: f64, v_a: f64, phi: f64, v_phi: f64, laplacian_phi: f64, derivative2_phi: f64, v: &V, xi: &Xi) -> f64 where
        V: C2Fn<f64, Output = f64>,
        Xi: C2Fn<f64, Output = f64>,
    {
        let d = dim as f64;
        let derivative2_div_a2 = derivative2_phi / a / a;
        let laplacian_div_a2 = laplacian_phi / a / a;
        let hubble = v_a / a;
        let xi_phi = xi.value(phi);
        let xi_d_phi = xi.value_d(phi);
        let xi_dd_phi = xi.value_dd(phi);
        let pv_phi = v.value(phi);
        let pv_d_phi = v.value_d(phi);
        (64.0 * d * laplacian_div_a2 + 16.0 * (-3.0 + d) * (-2.0 + d) * d * d * hubble * hubble * hubble * kappa * v_phi * xi_phi + 32.0 * (-2.0 + d) * laplacian_div_a2 * laplacian_div_a2 * kappa * xi_d_phi + (-3.0 + d) * (-2.0 + d) * (-2.0 + d) * (-1.0 + d) * d * d * hubble * hubble * hubble * hubble * hubble * hubble * kappa * xi_phi * xi_d_phi + -2.0 * (-3.0 + d) * (-2.0 + d) * (-2.0 + d) * (-1.0 + d) * d * d * hubble * hubble * hubble * hubble * hubble * kappa * v_phi * xi_d_phi * xi_d_phi + 8.0 * (-2.0 + d) * d * hubble * hubble * kappa * (2.0 * (-3.0 + d) * xi_phi * pv_d_phi + (2.0 * derivative2_div_a2 + -1.0 * derivative2_div_a2 * d + 5.0 * d * v_phi * v_phi + -2.0 * d * pv_phi) * xi_d_phi) + 2.0 * (-2.0 + d) * laplacian_div_a2 * kappa * (-8.0 * (-3.0 + d) * d * hubble * hubble * xi_phi + -32.0 * d * hubble * v_phi * xi_d_phi + -16.0 * pv_d_phi * xi_d_phi + (-3.0 + d) * (-2.0 + d) * (-1.0 + d) * d * hubble * hubble * hubble * hubble * xi_d_phi * xi_d_phi + 16.0 * derivative2_div_a2 * xi_dd_phi) + -32.0 * pv_d_phi * (2.0 * d + derivative2_div_a2 * (-2.0 + d) * kappa * xi_dd_phi) + 2.0 * (-2.0 + d) * (-1.0 + d) * d * hubble * hubble * hubble * hubble * xi_d_phi * (2.0 * (-1.0 + d) * d + (-2.0 + d) * kappa * (derivative2_div_a2 * (-3.0 + d) + -2.0 * d * v_phi * v_phi) * xi_dd_phi) + 32.0 * d * hubble * v_phi * (-2.0 * d + (-2.0 + d) * kappa * (pv_d_phi * xi_d_phi + -1.0 * derivative2_div_a2 * xi_dd_phi))) / (64.0 * d + 4.0 * (-2.0 + d) * kappa * (-4.0 * (-3.0 + d) * d * hubble * hubble * xi_phi + 8.0 * (laplacian_div_a2 + -1.0 * d * hubble * v_phi) * xi_d_phi + d * d * (2.0 + -3.0 * d + d * d) * hubble * hubble * hubble * hubble * xi_d_phi * xi_d_phi + 8.0 * derivative2_div_a2 * xi_dd_phi))
    }

    /// Computes a'' / a
    #[rustfmt::skip]
    pub fn hubble2<V, Xi>(dim: usize, kappa: f64, a: f64, v_a: f64, phi: f64, v_phi: f64, laplacian_phi: f64, derivative2_phi: f64, v: &V, xi: &Xi) -> f64 where
        V: C2Fn<f64, Output = f64>,
        Xi: C2Fn<f64, Output = f64>,
    {
        let d = dim as f64;
        let laplacian_div_a2 = laplacian_phi / a / a;
        let derivative2_div_a2 = derivative2_phi / a / a;
        let hubble = v_a / a;
        let xi_phi = xi.value(phi);
        let xi_d_phi = xi.value_d(phi);
        let xi_dd_phi = xi.value_dd(phi);
        let pv_phi = v.value(phi);
        let pv_d_phi = v.value_d(phi);
        -1.0 / 4.0 * 1.0 / ((-1.0 + d)) * 1.0 / ((16.0 * d + (-2.0 + d) * kappa * (-4.0 * (-3.0 + d) * d * hubble * hubble * xi_phi + xi_d_phi * (8.0 * laplacian_div_a2 + -8.0 * d * hubble * v_phi + (-2.0 + d) * (-1.0 + d) * d * d * hubble * hubble * hubble * hubble * xi_d_phi) + 8.0 * derivative2_div_a2 * xi_dd_phi))) * (-64.0 * d * kappa * pv_phi + 48.0 * (-2.0 + d) * (-1.0 + d) * d * hubble * hubble * hubble * kappa * v_phi * xi_d_phi + 16.0 * d * kappa * v_phi * v_phi * (2.0 + -1.0 * (-2.0 + d) * (-1.0 + d) * hubble * hubble * xi_dd_phi) + (-2.0 + d) * (32.0 * (-1.0 + d) * d * hubble * hubble + -32.0 * derivative2_div_a2 * kappa + (-1.0 + d) * hubble * hubble * kappa * (-4.0 * (-4.0 + d) * (-3.0 + d) * d * hubble * hubble * xi_phi + xi_d_phi * (-48.0 * laplacian_div_a2 + 16.0 * d * pv_d_phi + (-3.0 + d) * (-2.0 + d) * (-1.0 + d) * d * d * hubble * hubble * hubble * hubble * xi_d_phi) + 16.0 * derivative2_div_a2 * (-3.0 + d) * xi_dd_phi)))
    }

    pub fn hubble_constraint_coefs<V, Xi>(
        dim: usize,
        kappa: f64,
        a: f64,
        phi: f64,
        v_phi: f64,
        laplacian_phi: f64,
        derivative2_phi: f64,
        v: &V,
        xi: &Xi,
    ) -> [f64; 5]
    where
        V: C2Fn<f64, Output = f64>,
        Xi: C2Fn<f64, Output = f64>,
    {
        let d = dim as f64;
        let laplacian_div_a2 = laplacian_phi / a / a;
        let derivative2_div_a2 = derivative2_phi / a / a;
        let xi_phi = xi.value(phi);
        let xi_d_phi = xi.value_d(phi);
        let xi_dd_phi = xi.value_dd(phi);
        let pv_phi = v.value(phi);
        [
            1.0 / 2.0 * (derivative2_div_a2 + v_phi * v_phi) + pv_phi,
            0.0,
            -1.0 / 4.0 * (-1.0 + d) * 1.0 / (kappa)
                * (2.0 * d
                    + (-2.0 + d)
                        * kappa
                        * (laplacian_div_a2 * xi_d_phi + derivative2_div_a2 * xi_dd_phi)),
            1.0 / 4.0 * (-2.0 + d) * (-1.0 + d) * d * v_phi * xi_d_phi,
            1.0 / 16.0 * (-3.0 + d) * (-2.0 + d) * (-1.0 + d) * d * xi_phi,
        ]
    }

    #[rustfmt::skip]
    pub fn epsilon<V, Xi>(dim: usize, kappa: f64, a: f64, v_a: f64, phi: f64, v_phi: f64, laplacian_phi: f64, derivative2_phi: f64, v: &V, xi: &Xi) -> f64 where
        V: C2Fn<f64, Output = f64>,
        Xi: C2Fn<f64, Output = f64>,
    {
        let d = dim as f64;
        let laplacian_div_a2 = laplacian_phi / a / a;
        let derivative2_div_a2 = derivative2_phi / a / a;
        let hubble = v_a / a;
        let xi_phi = xi.value(phi);
        let xi_d_phi = xi.value_d(phi);
        let xi_dd_phi = xi.value_dd(phi);
        let pv_d_phi = v.value_d(phi);
        (1.0 / (hubble * hubble) * kappa * (64.0 * derivative2_div_a2 + 16.0 * (-2.0 + d) * (-1.0 + d) * d * (1.0 + d) * hubble * hubble * hubble * v_phi * xi_d_phi + (-2.0 + d) * (-1.0 + d) * hubble * hubble * (xi_d_phi * (16.0 * d * pv_d_phi + (1.0 + d) * (-16.0 * laplacian_div_a2 + (-2.0 + d) * (-1.0 + d) * d * d * hubble * hubble * hubble * hubble * xi_d_phi)) + -16.0 * derivative2_div_a2 * xi_dd_phi) + 16.0 * d * v_phi * v_phi * (4.0 + -1.0 * (-2.0 + d) * (-1.0 + d) * hubble * hubble * xi_dd_phi))) / (4.0 * (-1.0 + d) * (16.0 * d + (-2.0 + d) * kappa * (-4.0 * (-3.0 + d) * d * hubble * hubble * xi_phi + xi_d_phi * (8.0 * laplacian_div_a2 + -8.0 * d * hubble * v_phi + (-2.0 + d) * (-1.0 + d) * d * d * hubble * hubble * hubble * hubble * xi_d_phi) + 8.0 * derivative2_div_a2 * xi_dd_phi)))
    }

    #[rustfmt::skip]
    pub fn perturbation_lag_coef_a_2<Xi>(dim: usize, kappa: f64, a: f64, v_a: f64, phi: f64, v_phi: f64, xi: &Xi) -> f64 where
        Xi: C2Fn<f64, Output = f64>,
    {
        let d = dim as f64;
        let hubble = v_a / a;
        let xi_phi = xi.value(phi);
        let xi_d_phi = xi.value_d(phi);
        (1.0 / (hubble * hubble) * v_phi * v_phi * (64.0 + -64.0 * (-2.0 + d) * hubble * kappa * v_phi * xi_d_phi + 16.0 * (-3.0 + d) * (-2.0 + d) * (-2.0 + d) * hubble * hubble * hubble * kappa * kappa * v_phi * xi_phi * xi_d_phi + -1.0 * (-2.0 + d) * (-2.0 + d) * (-2.0 + d) * d * (3.0 + -4.0 * d + d * d) * hubble * hubble * hubble * hubble * hubble * hubble * kappa * kappa * xi_phi * xi_d_phi * xi_d_phi + -2.0 * (-2.0 + d) * (-2.0 + d) * (-2.0 + d) * (-1.0 + d) * d * hubble * hubble * hubble * hubble * hubble * kappa * kappa * v_phi * xi_d_phi * xi_d_phi * xi_d_phi + 4.0 * (-2.0 + d) * (-2.0 + d) * hubble * hubble * hubble * hubble * kappa * ((-3.0 + d) * (-3.0 + d) * kappa * xi_phi * xi_phi + (-1.0 + d) * d * xi_d_phi * xi_d_phi) + -16.0 * (-2.0 + d) * hubble * hubble * kappa * (2.0 * (-3.0 + d) * xi_phi + -1.0 * (-2.0 + d) * kappa * v_phi * v_phi * xi_d_phi * xi_d_phi))) / (4.0 * (-4.0 + (6.0 + -5.0 * d + d * d) * hubble * hubble * kappa * xi_phi + 3.0 * (-2.0 + d) * hubble * kappa * v_phi * xi_d_phi) * (-4.0 + (6.0 + -5.0 * d + d * d) * hubble * hubble * kappa * xi_phi + 3.0 * (-2.0 + d) * hubble * kappa * v_phi * xi_d_phi))
    }

    #[rustfmt::skip]
    pub fn perturbation_lag_coef_c_2<V, Xi>(dim: usize, kappa: f64, a: f64, v_a: f64, phi: f64, v_phi: f64, v: &V, xi: &Xi) -> f64 where
        V: C2Fn<f64, Output = f64>,
        Xi: C2Fn<f64, Output = f64>,
    {
        let d = dim as f64;
        let hubble = v_a / a;
        let hubble2 = hubble2(dim, kappa, a, v_a, phi, v_phi, 0.0, 0.0, v, xi);
        let v_v_phi = vv_phi(dim, kappa, a, v_a, phi, v_phi, 0.0, 0.0, v, xi);
        let xi_phi = xi.value(phi);
        let xi_d_phi = xi.value_d(phi);
        let xi_dd_phi = xi.value_dd(phi);
        1.0 / 4.0 * (-1.0 + d) * 1.0 / (hubble * hubble) * 1.0 / (kappa) * 1.0 / ((-4.0 + (-2.0 + d) * hubble * kappa * ((-3.0 + d) * hubble * xi_phi + 3.0 * v_phi * xi_d_phi)) * (-4.0 + (-2.0 + d) * hubble * kappa * ((-3.0 + d) * hubble * xi_phi + 3.0 * v_phi * xi_d_phi))) * (64.0 * (hubble * hubble + -1.0 * hubble2) + -1.0 * (-2.0 + d) * hubble * kappa * ((-3.0 + d) * (-3.0 + d) * (-3.0 + d) * (-2.0 + d) * (-2.0 + d) * hubble * hubble * hubble * hubble * hubble * (hubble * hubble + -1.0 * hubble2) * kappa * kappa * xi_phi * xi_phi * xi_phi + 2.0 * (-8.0 * hubble * v_v_phi * xi_d_phi + 8.0 * v_phi * xi_d_phi * (7.0 * hubble * hubble + -6.0 * hubble2 + (-2.0 + d) * hubble * hubble * kappa * v_v_phi * xi_d_phi) + hubble * v_phi * v_phi * ((-2.0 + d) * kappa * xi_d_phi * xi_d_phi * (-2.0 * (12.0 + d) * hubble * hubble + 16.0 * hubble2 + -3.0 * (-2.0 + d) * hubble * hubble * kappa * v_v_phi * xi_d_phi) + -8.0 * xi_dd_phi) + -3.0 * (-2.0 + d) * (-2.0 + d) * hubble * hubble * hubble * kappa * kappa * v_phi * v_phi * v_phi * v_phi * xi_d_phi * xi_d_phi * xi_dd_phi + (-2.0 + d) * hubble * hubble * kappa * v_phi * v_phi * v_phi * xi_d_phi * ((-2.0 + d) * (3.0 + d) * hubble * hubble * kappa * xi_d_phi * xi_d_phi + 8.0 * xi_dd_phi)) + (-3.0 + d) * (-3.0 + d) * (-2.0 + d) * hubble * hubble * hubble * kappa * xi_phi * xi_phi * (12.0 * (-1.0 * hubble * hubble + hubble2) + (-2.0 + d) * hubble * kappa * (((7.0 * hubble * hubble + -6.0 * hubble2) * v_phi + -1.0 * hubble * v_v_phi) * xi_d_phi + -1.0 * hubble * v_phi * v_phi * xi_dd_phi)) + (-3.0 + d) * hubble * xi_phi * (48.0 * (hubble * hubble + -1.0 * hubble2) + (-2.0 + d) * hubble * kappa * (xi_d_phi * (8.0 * ((-7.0 * hubble * hubble + 6.0 * hubble2) * v_phi + hubble * v_v_phi) + (-2.0 + d) * hubble * kappa * v_phi * (((14.0 + d) * hubble * hubble + -10.0 * hubble2) * v_phi + -4.0 * hubble * v_v_phi) * xi_d_phi) + 4.0 * hubble * v_phi * v_phi * (2.0 + -1.0 * (-2.0 + d) * hubble * kappa * v_phi * xi_d_phi) * xi_dd_phi))))
    }

    /// Returns the metric scalar perturbation $A$
    #[rustfmt::skip]
    pub fn metric_perturbation_a<Xi>(
        dim: usize,
        kappa: f64,
        a: f64,
        v_a: f64,
        phi: f64,
        v_phi: f64,
        avg_phi: f64,
        avg_v_phi: f64,
        xi: &Xi,
    ) -> f64
    where
        Xi: C2Fn<f64, Output = f64>,
    {
        let d = dim as f64;
        let hubble = v_a / a;
        let xi_phi = xi.value(phi);
        let xi_d_phi = xi.value_d(phi);
        let xi_dd_phi = xi.value_dd(phi);
        let delta_phi = phi - avg_phi;
        let v_delta_phi = v_phi - avg_v_phi;
        (-1.0 * (2.0 + -3.0 * d + d * d) * hubble * kappa * (delta_phi * hubble + -1.0 * v_delta_phi) * xi_d_phi + delta_phi * 1.0 / (hubble) * kappa * v_phi * (-4.0 + (2.0 + -3.0 * d + d * d) * hubble * hubble * xi_dd_phi)) / ((-1.0 + d) * (-4.0 + (6.0 + -5.0 * d + d * d) * hubble * hubble * kappa * xi_phi + 3.0 * (-2.0 + d) * hubble * kappa * v_phi * xi_d_phi))
    }

    /// Returns the metric scalar perturbation $\partial_a B \partial^a B / a^2$
    #[rustfmt::skip]
    pub fn metric_perturbation_b<V, Xi>(
        dim: usize,
        kappa: f64,
        a: f64,
        v_a: f64,
        phi: f64,
        v_phi: f64,
        avg_phi: f64,
        avg_v_phi: f64,
        laplacian_phi: f64,
        v: &V,
        xi: &Xi,
    ) -> f64
    where
        V: C2Fn<f64, Output = f64>,
        Xi: C2Fn<f64, Output = f64>,
    {
        let d = dim as f64;
        let laplacian_div_a2 = laplacian_phi / a / a;
        let hubble = v_a / a;
        let xi_phi = xi.value(phi);
        let xi_d_phi = xi.value_d(phi);
        let xi_dd_phi = xi.value_dd(phi);
        let pv_d_phi = v.value_d(phi);
        let delta_phi = phi - avg_phi;
        let v_delta_phi = v_phi - avg_v_phi;
        let laplacian_delta_phi_div_a2 = laplacian_div_a2;
        (1.0 / (hubble * hubble) * kappa * ((-1.0 + d) * hubble * (-4.0 + (-3.0 + d) * (-2.0 + d) * hubble * hubble * kappa * xi_phi) * (16.0 * delta_phi * pv_d_phi + (-2.0 + d) * (-1.0 + d) * hubble * hubble * (delta_phi * d * (1.0 + d) * hubble * hubble + -4.0 * laplacian_delta_phi_div_a2) * xi_d_phi) + (-1.0 + d) * hubble * v_phi * (16.0 * (delta_phi * d * hubble + v_delta_phi) * (-4.0 + (-3.0 + d) * (-2.0 + d) * hubble * hubble * kappa * xi_phi) + 48.0 * delta_phi * (-2.0 + d) * hubble * kappa * pv_d_phi * xi_d_phi + (-2.0 + d) * (-2.0 + d) * (-1.0 + d) * hubble * hubble * hubble * kappa * (delta_phi * d * (7.0 + 3.0 * d) * hubble * hubble + -12.0 * laplacian_delta_phi_div_a2 + -4.0 * d * hubble * v_delta_phi) * xi_d_phi * xi_d_phi) + 16.0 * delta_phi * kappa * v_phi * v_phi * v_phi * (4.0 + -1.0 * (-2.0 + d) * (-1.0 + d) * hubble * hubble * xi_dd_phi) + 4.0 * (-2.0 + d) * (-1.0 + d) * hubble * hubble * kappa * v_phi * v_phi * xi_d_phi * (4.0 * delta_phi * (1.0 + 4.0 * d) * hubble + 8.0 * v_delta_phi + -1.0 * delta_phi * (-2.0 + d) * (-1.0 + d) * d * hubble * hubble * hubble * xi_dd_phi))) / (4.0 * (-1.0 + d) * (-1.0 + d) * (-4.0 + (-2.0 + d) * hubble * kappa * ((-3.0 + d) * hubble * xi_phi + 3.0 * v_phi * xi_d_phi)) * (-4.0 + (-2.0 + d) * hubble * kappa * ((-3.0 + d) * hubble * xi_phi + 3.0 * v_phi * xi_d_phi)))
    }

    pub fn solve_hubble<V, Xi>(
        dim: usize,
        kappa: f64,
        a: f64,
        phi: f64,
        v_phi: f64,
        derivative2_phi: f64,
        laplacian_phi: f64,
        v: &V,
        xi: &Xi,
    ) -> f64
    where
        V: C2Fn<f64, Output = f64>,
        Xi: C2Fn<f64, Output = f64>,
    {
        let d = dim as f64;
        let coefs = hubble_constraint_coefs(
            dim,
            kappa,
            a,
            phi,
            v_phi,
            laplacian_phi,
            derivative2_phi,
            v,
            xi,
        );
        let derivative2_div_a2 = derivative2_phi / a / a;
        let pv_phi = v.value(phi);
        let initial_h =
            (kappa / d / (d - 1.0) * (derivative2_div_a2 + v_phi * v_phi + 2.0 * pv_phi)).sqrt();
        newton_solve_polynomial(initial_h, &coefs, 1e-10)
    }
}

pub fn hubble_constraint<V, Xi>(
    dim: usize,
    kappa: f64,
    a: f64,
    v_a: f64,
    phi: f64,
    v_phi: f64,
    laplacian_phi: f64,
    derivative2_phi: f64,
    v: &V,
    xi: &Xi,
) -> f64
where
    V: C2Fn<f64, Output = f64>,
    Xi: C2Fn<f64, Output = f64>,
{
    evaluate_polynomial(
        v_a / a,
        &data::hubble_constraint_coefs(
            dim,
            kappa,
            a,
            phi,
            v_phi,
            laplacian_phi,
            derivative2_phi,
            v,
            xi,
        ),
    )
    // 1.0 / 2.0 * v_phi * v_phi + pv_phi + 1.0 / 16.0 * (-1.0 + d) * d * hubble * hubble * 1.0 / (kappa) * (-8.0 + (6.0 + -5.0 * d + d * d) * hubble * hubble * kappa * xi_phi) + 1.0 / 4.0 * d * (2.0 + -3.0 * d + d * d) * hubble * hubble * hubble * v_phi * xi_d_phi
}

#[derive(Debug, Clone, Copy, Encode, Decode)]
pub struct GaussBonnetBackgroundState {
    pub a: f64,
    pub v_a: f64,
    pub phi: f64,
    pub v_phi: f64,
    pub dt: f64,
    pub pert_c: f64,
    pub potential: f64,
    pub horizon: f64,
    pub pert_a: f64,
}

pub struct GaussBonnetBInput<V, Xi> {
    pub dim: usize,
    pub kappa: f64,
    pub v: V,
    pub xi: Xi,
}

impl<V, Xi> Dimension for GaussBonnetBInput<V, Xi> {
    fn dimension(&self) -> usize {
        self.dim
    }
}

impl<V, Xi> Kappa for GaussBonnetBInput<V, Xi> {
    fn kappa(&self) -> f64 {
        self.kappa
    }
}

impl GaussBonnetBackgroundState {
    pub fn init_slowroll3d<V, Xi>(input: &GaussBonnetBInput<V, Xi>, a: f64, phi: f64) -> Self
    where
        V: C2Fn<f64, Output = f64>,
        Xi: C2Fn<f64, Output = f64>,
    {
        assert_eq!(input.dim, 3);
        let hubble0 = sqrt(input.kappa * input.v.value(phi) / 3.0);
        let v_phi = -input.v.value_d(phi) / 3.0 / hubble0;
        let v_a =
            a * data::solve_hubble(3, input.kappa, a, phi, v_phi, 0.0, 0.0, &input.v, &input.xi);
        Self {
            a,
            v_a,
            phi,
            v_phi,
            dt: 0.0,
            potential: 0.0,
            horizon: 0.0,
            pert_a: 0.0,
            pert_c: 0.0,
        }
    }
    pub fn epsilon<V, Xi>(&self, input: &GaussBonnetBInput<V, Xi>) -> f64
    where
        V: C2Fn<f64, Output = f64>,
        Xi: C2Fn<f64, Output = f64>,
    {
        data::epsilon(
            input.dim,
            input.kappa,
            self.a,
            self.v_a,
            self.phi,
            self.v_phi,
            0.0,
            0.0,
            &input.v,
            &input.xi,
        )
    }
    pub fn hubble_constraint<V, Xi>(&self, input: &GaussBonnetBInput<V, Xi>) -> f64
    where
        V: C2Fn<f64, Output = f64>,
        Xi: C2Fn<f64, Output = f64>,
    {
        hubble_constraint(
            input.dim,
            input.kappa,
            self.a,
            self.v_a,
            self.phi,
            self.v_phi,
            0.0,
            0.0,
            &input.v,
            &input.xi,
        )
    }
    fn delta<V, Xi>(&self, input: &GaussBonnetBInput<V, Xi>) -> VecN<4, f64>
    where
        V: C2Fn<f64, Output = f64>,
        Xi: C2Fn<f64, Output = f64>,
    {
        VecN::new([
            self.v_a,
            self.v_phi,
            self.a
                * data::hubble2(
                    input.dim,
                    input.kappa,
                    self.a,
                    self.v_a,
                    self.phi,
                    self.v_phi,
                    0.0,
                    0.0,
                    &input.v,
                    &input.xi,
                ),
            data::vv_phi(
                input.dim,
                input.kappa,
                self.a,
                self.v_a,
                self.phi,
                self.v_phi,
                0.0,
                0.0,
                &input.v,
                &input.xi,
            ),
        ])
    }
    pub fn update_with(&self, dt: f64, vec: &VecN<4, f64>, real_dt: f64) -> Self {
        Self {
            a: self.a + dt * vec[0],
            phi: self.phi + dt * vec[1],
            v_a: self.v_a + dt * vec[2],
            v_phi: self.v_phi + dt * vec[3],
            dt: real_dt,
            potential: 0.0,
            horizon: 0.0,
            pert_a: 0.0,
            pert_c: 0.0,
        }
    }
    pub fn update<V, Xi>(&self, input: &GaussBonnetBInput<V, Xi>, dt: f64) -> Self
    where
        V: C2Fn<f64, Output = f64>,
        Xi: C2Fn<f64, Output = f64>,
    {
        let k1 = self.delta(input);
        let k2 = self.update_with(dt / 2.0, &k1, 0.0).delta(input);
        let k3 = self.update_with(dt / 2.0, &k2, 0.0).delta(input);
        let k4 = self.update_with(dt / 2.0, &k3, 0.0).delta(input);
        self.update_with(dt / 6.0, &(k1 + k2 * 2.0 + k3 * 2.0 + k4), dt)
    }
    pub fn calculate_pert_coefs<V, Xi>(data: &mut [Self], input: &GaussBonnetBInput<V, Xi>)
    where
        V: C2Fn<f64, Output = f64>,
        Xi: C2Fn<f64, Output = f64>,
    {
        let d = input.dim as f64;
        for state in &mut *data {
            state.pert_c = data::perturbation_lag_coef_c_2(
                input.dim,
                input.kappa,
                state.a,
                state.v_a,
                state.phi,
                state.v_phi,
                &input.v,
                &input.xi,
            )
            .sqrt();
            state.pert_a = data::perturbation_lag_coef_a_2(
                input.dim,
                input.kappa,
                state.a,
                state.v_a,
                state.phi,
                state.v_phi,
                &input.xi,
            )
            .sqrt();
        }
        for index in 1..data.len() - 1 {
            let cap_a = data[index].pert_a;
            let v_cap_a = (data[index + 1].pert_a - data[index - 1].pert_a)
                / (data[index - 1].dt + data[index].dt);
            let vv_cap_a = derivative_2(
                data[index - 1].dt,
                data[index].dt,
                data[index - 1].pert_a,
                data[index].pert_a,
                data[index + 1].pert_a,
            );
            let state = &mut data[index];
            let a = state.a;
            let hubble = state.v_a / a;
            let hubble2 = data::hubble2(
                input.dim,
                input.kappa,
                state.a,
                state.v_a,
                state.phi,
                state.v_phi,
                0.0,
                0.0,
                &input.v,
                &input.xi,
            );
            state.horizon = a
                * a
                * (1.0 / 4.0 * (-1.0 + d) * (-1.0 + d) * hubble * hubble
                    + 1.0 / 2.0 * (-1.0 + d) * hubble2
                    + 1.0 / (cap_a) * d * hubble * v_cap_a
                    + 1.0 / (cap_a) * vv_cap_a);
            state.potential = 1.0 / 4.0 * (-2.0 + d) * d * hubble * hubble
                + 1.0 / 2.0 * d * hubble2
                + 1.0 / (cap_a) * d * hubble * v_cap_a
                + 1.0 / (cap_a) * vv_cap_a;
        }
        data[0].horizon = data[1].horizon;
        data[0].potential = data[1].potential;
    }
}

impl ScaleFactor for GaussBonnetBackgroundState {
    fn scale_factor(&self) -> f64 {
        self.a
    }
}

impl ScaleFactorD for GaussBonnetBackgroundState {
    fn v_scale_factor(&self, _kappa: f64) -> f64 {
        self.v_a
    }
}

impl Phi for GaussBonnetBackgroundState {
    fn phi(&self) -> f64 {
        self.phi
    }
}

impl PhiD for GaussBonnetBackgroundState {
    fn v_phi(&self) -> f64 {
        self.v_phi
    }
}

impl Dt for GaussBonnetBackgroundState {
    fn dt(&self) -> f64 {
        self.dt
    }
}

impl Interpolate for GaussBonnetBackgroundState {
    fn interpolate(&self, other: &Self, l: f64) -> Self
    where
        Self: Sized,
    {
        interpolate_fields!(
            Self, self, other, l, a, v_a, phi, v_phi, potential, pert_c, horizon, pert_a
        )
    }
}

impl<V, Xi> BackgroundSolver for GaussBonnetBInput<V, Xi>
where
    V: C2Fn<f64, Output = f64>,
    Xi: C2Fn<f64, Output = f64>,
{
    type State = GaussBonnetBackgroundState;

    fn create_state(&self, a: f64, v_a: f64, phi: f64, v_phi: f64) -> Self::State {
        GaussBonnetBackgroundState {
            a,
            v_a,
            phi,
            v_phi,
            dt: 0.0,
            pert_c: 0.0,
            potential: 0.0,
            horizon: 0.0,
            pert_a: 0.0,
        }
    }

    fn update(&self, state: &mut Self::State, dt: f64) {
        *state = state.update(self, dt);
    }
}

pub trait GaussBonnetBInputProvider {
    type V;
    type Xi;
    fn input(&self) -> &GaussBonnetBInput<Self::V, Self::Xi>;
}

pub struct GaussBonnetScalarPerturbationPotential;
impl<Ctx> BackgroundFn<Ctx, GaussBonnetBackgroundState> for GaussBonnetScalarPerturbationPotential {
    type Output = f64;

    fn apply(&self, _context: &Ctx, state: &GaussBonnetBackgroundState, k: f64) -> Self::Output {
        let a = state.a;
        k / a * k / a * (state.pert_c / state.pert_a) - state.potential
    }
}

pub struct GaussBonnetScalarPerturbationCoef;
impl<Ctx> BackgroundFn<Ctx, GaussBonnetBackgroundState> for GaussBonnetScalarPerturbationCoef
where
    Ctx: Dimension,
{
    type Output = f64;

    fn apply(&self, context: &Ctx, state: &GaussBonnetBackgroundState, _k: f64) -> Self::Output {
        let ret = state.a.powf((context.dimension() as f64) / 2.0) * state.pert_a;
        1.0 / ret
    }
}

#[derive(Encode, Decode)]
pub struct GaussBonnetField<const D: usize> {
    pub a: f64,
    pub v_a: f64,
    pub phi: BoxLattice<D, [f64; 2]>,
}

impl<const D: usize> GaussBonnetField<D> {
    pub fn zero(size: VecN<D, usize>) -> Self {
        Self {
            a: 0.0,
            v_a: 0.0,
            phi: BoxLattice::constant(size, [0.0, 0.0]),
        }
    }
    pub fn init<V, Xi>(&mut self, a: f64, phi: f64, v_phi: f64, input: &GaussBonnetBInput<V, Xi>)
    where
        V: C2Fn<f64, Output = f64>,
        Xi: C2Fn<f64, Output = f64>,
    {
        self.a = a;
        self.v_a =
            a * data::solve_hubble(D, input.kappa, a, phi, v_phi, 0.0, 0.0, &input.v, &input.xi);
        self.phi.par_fill([phi, v_phi]);
    }
    pub fn assign(&mut self, other: &Self) {
        self.a = other.a;
        self.v_a = other.v_a;
        self.phi.par_assign(&other.phi);
    }
    pub fn add(&mut self, other: &Self, factor: f64) {
        self.a += other.a * factor;
        self.v_a += other.v_a * factor;
        self.phi.par_for_each_mut(|ptr, index, coord| {
            let phi0 = other.phi.get(index, coord);
            ptr[0] += phi0[0] * factor;
            ptr[1] += phi0[1] * factor;
        });
    }
    pub fn delta<V, Xi>(
        &mut self,
        field: &Self,
        input: &GaussBonnetBInput<V, Xi>,
        lattice: &LatticeParam<D>,
    ) where
        V: C2Fn<f64, Output = f64> + Sync,
        Xi: C2Fn<f64, Output = f64> + Sync,
    {
        let phi = field.phi.view().map(|f| f[0]);
        let v_phi = field.phi.view().map(|f| f[1]);
        self.a = field.v_a;
        self.v_a = field.a
            * data::hubble2(
                input.dim,
                input.kappa,
                field.a,
                field.v_a,
                phi.average(),
                v_phi.average(),
                phi.as_ref().laplacian(&lattice.spacing).average(),
                phi.as_ref().derivative_square(&lattice.spacing).average(),
                &input.v,
                &input.xi,
            );
        self.phi.par_for_each_mut(|ptr, index, coord| {
            ptr[0] = v_phi.get(index, coord);
            ptr[1] = data::vv_phi(
                input.dim,
                input.kappa,
                field.a,
                field.v_a,
                phi.get(index, coord),
                v_phi.get(index, coord),
                phi.laplacian_at(coord, &lattice.spacing),
                phi.derivative_square_at(coord, &lattice.spacing),
                &input.v,
                &input.xi,
            );
        });
    }
    pub fn metric_perturbations<V, Xi>(
        &self,
        input: &GaussBonnetBInput<V, Xi>,
        lattice: &LatticeParam<D>,
    ) -> (f64, f64)
    where
        V: C2Fn<f64, Output = f64> + Sync,
        Xi: C2Fn<f64, Output = f64> + Sync,
    {
        let phi = self.phi.view().map(|f| f[0]);
        let v_phi = self.phi.view().map(|f| f[1]);
        let avg_phi = phi.average();
        let avg_v_phi = v_phi.average();
        let pert_a = LatticeSupplier::new(*self.phi.dim(), |index, coord| {
            data::metric_perturbation_a(
                input.dim,
                input.kappa,
                self.a,
                self.v_a,
                phi.get(index, coord),
                v_phi.get(index, coord),
                avg_phi,
                avg_v_phi,
                &input.xi,
            )
        })
        .average();
        let pert_b = LatticeSupplier::new(*self.phi.dim(), |index, coord| {
            data::metric_perturbation_b(
                input.dim,
                input.kappa,
                self.a,
                self.v_a,
                phi.get(index, coord),
                v_phi.get(index, coord),
                avg_phi,
                avg_v_phi,
                phi.laplacian_at(coord, &lattice.spacing),
                &input.v,
                &input.xi,
            )
        })
        .average();
        (pert_a, pert_b)
    }
    pub fn hubble_constraint<V, Xi>(
        &self,
        input: &GaussBonnetBInput<V, Xi>,
        lattice: &LatticeParam<D>,
    ) -> f64
    where
        V: C2Fn<f64, Output = f64>,
        Xi: C2Fn<f64, Output = f64>,
    {
        let phi = self.phi.view().map(|f| f[0]);
        let v_phi = self.phi.view().map(|f| f[1]);
        hubble_constraint(
            D,
            input.kappa,
            self.a,
            self.v_a,
            phi.average(),
            v_phi.average(),
            phi.as_ref().laplacian(&lattice.spacing).average(),
            phi.as_ref().derivative_square(&lattice.spacing).average(),
            &input.v,
            &input.xi,
        )
    }
    pub fn populate_noise<S, V, Xi>(
        &mut self,
        source: &mut S,
        input: &GaussBonnetBInput<V, Xi>,
        lattice: &LatticeParam<D>,
    ) where
        S: Source,
        V: C2Fn<f64, Output = f64>,
        Xi: C2Fn<f64, Output = f64>,
    {
        let mut phi = BoxLattice::zeros(lattice.size);
        let mut v_phi = BoxLattice::zeros(lattice.size);
        populate_noise(&lattice, self.a, self.v_a, source, &mut phi, &mut v_phi);
        self.phi.par_for_each_mut(|ptr, index, coord| {
            ptr[0] += phi.get(index, coord).re;
            ptr[1] += v_phi.get(index, coord).re;
        });
        let phi = self.phi.view().map(|f| f[0]);
        let v_phi = self.phi.view().map(|f| f[1]);
        self.v_a = self.a
            * data::solve_hubble(
                input.dim,
                input.kappa,
                self.a,
                phi.average(),
                v_phi.average(),
                phi.as_ref().derivative_square(&lattice.spacing).average(),
                phi.as_ref().laplacian(&lattice.spacing).average(),
                &input.v,
                &input.xi,
            );
    }
}

pub struct GaussBonnetFieldSimulator<'a, 'b, const D: usize, V, Xi> {
    pub lattice: &'a LatticeParam<D>,
    pub input: &'b GaussBonnetBInput<V, Xi>,
    pub field: GaussBonnetField<D>,
    delta: GaussBonnetField<D>,
    k1: GaussBonnetField<D>,
    k2: GaussBonnetField<D>,
    k3: GaussBonnetField<D>,
    k4: GaussBonnetField<D>,
}

impl<'a, 'b, const D: usize, V, Xi> GaussBonnetFieldSimulator<'a, 'b, D, V, Xi> {
    pub fn new(
        lattice: &'a LatticeParam<D>,
        input: &'b GaussBonnetBInput<V, Xi>,
        field: GaussBonnetField<D>,
    ) -> Self {
        let dim = lattice.size;
        Self {
            lattice,
            input,
            field,
            delta: GaussBonnetField::zero(dim),
            k1: GaussBonnetField::zero(dim),
            k2: GaussBonnetField::zero(dim),
            k3: GaussBonnetField::zero(dim),
            k4: GaussBonnetField::zero(dim),
        }
    }
    pub fn update(&mut self, dt: f64)
    where
        V: C2Fn<f64, Output = f64> + Sync,
        Xi: C2Fn<f64, Output = f64> + Sync,
    {
        self.k1.delta(&self.field, &self.input, &self.lattice);

        self.delta.assign(&self.field);
        self.delta.add(&self.k1, dt / 2.0);
        self.k2.delta(&self.delta, &self.input, &self.lattice);

        self.delta.assign(&self.field);
        self.delta.add(&self.k2, dt / 2.0);
        self.k3.delta(&self.delta, &self.input, &self.lattice);

        self.delta.assign(&self.field);
        self.delta.add(&self.k3, dt);
        self.k4.delta(&self.delta, &self.input, &self.lattice);

        self.field.add(&self.k1, dt / 6.0);
        self.field.add(&self.k2, dt / 3.0);
        self.field.add(&self.k3, dt / 3.0);
        self.field.add(&self.k4, dt / 6.0);
    }
}
