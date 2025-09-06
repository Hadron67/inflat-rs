use std::{
    collections::HashMap,
    f64::consts::PI,
    fmt::Debug,
    iter::zip,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
    time::{Duration, SystemTime},
};

use bincode::{Decode, Encode};
use libm::{fmin, pow, sin, sqrt};
use num_complex::ComplexFloat;
use plotly::{
    Layout, Plot, Scatter,
    common::ExponentFormat,
    layout::{Axis, AxisType, LayoutGrid},
};
use random::Source;
use rustfft::{FftDirection, num_complex::Complex64};

use crate::{
    background::{
        BINCODE_CONFIG, BackgroundSolver, BackgroundStateInput, Kappa, Phi, PhiD,
        ScaleFactor, ScaleFactorD, ScaleFactorMut, spectrum_factor,
    },
    c2fn::C2Fn,
    fft::DftNDPlan,
    lat::{BoxLattice, Lattice, LatticeMut, LatticeParam, LatticeRef, LatticeScalarMul},
    util::{
        self, Hms, ParamRange, RateLimiter, TimeEstimator, VecN, decode_from_file, encode_to_file,
        lazy_file_opt, limit_length, plot_spectrum, remove_first_and_last,
    },
};

#[derive(Encode, Decode)]
pub struct ScalarFieldState<const D: usize> {
    pub b: f64,
    pub mom_b: f64,
    pub phi: BoxLattice<D, f64>,
    pub mom_phi: BoxLattice<D, f64>,
}

pub struct ScalarFieldParams<const D: usize, F> {
    pub kappa: f64,
    pub potential: F,
    pub lattice: LatticeParam<D>,
}

impl<const D: usize, F> ScalarFieldParams<D, F> {
    pub fn a_to_b(a: f64, kappa: f64) -> f64 {
        let d = D as f64;
        2.0 * sqrt((d - 1.0) / d / kappa) * a.powf(d / 2.0)
    }
    pub fn b_to_a(b: f64, kappa: f64) -> f64 {
        let d = D as f64;
        pow(b / 2.0 * sqrt(kappa * d / (d - 1.0)), 2.0 / d)
    }
    pub fn v_a_to_mom_b(v_a: f64, a: f64, kappa: f64) -> f64 {
        let d = D as f64;
        sqrt((d - 1.0) / d / kappa) * d * a.powf(d / 2.0 - 1.0) * v_a
    }
    pub fn v_a_from_mom_b(mom_b: f64, a: f64, kappa: f64) -> f64 {
        let d = D as f64;
        mom_b / (sqrt((d - 1.0) / d / kappa) * d * a.powf(d / 2.0 - 1.0))
    }
    pub fn mom_b_to_v_a(mom_b: f64, b: f64, kappa: f64) -> f64 {
        let d = D as f64;
        let v_b = -mom_b;
        pow(d * kappa / (d - 1.0) / 4.0, 1.0 / d) * 2.0 / d * b.powf(2.0 / d - 1.0) * v_b
    }
    fn v_a_from_hubble_constraint(&self, field: &ScalarFieldState<D>) -> f64
    where
        F: C2Fn<f64, Output = f64> + Send + Sync,
    {
        let d = D as f64;
        let a0 = Self::b_to_a(field.b, self.kappa);
        let v_a2 = (field
            .phi
            .view()
            .derivative_square(&self.lattice.spacing)
            .average()
            + field
                .mom_phi
                .view()
                .map(|f| {
                    let c = f * a0.powi(2 - 2 * (D as i32));
                    c * c
                })
                .average()
            + field.phi.view().map(|f| self.potential.value(f)).average() * 2.0 * a0 * a0)
            * self.kappa
            / d
            / (d - 1.0);
        -v_a2.sqrt()
    }
    pub fn init_slowroll(&self, field: &mut ScalarFieldState<D>, a0: f64, phi0: f64)
    where
        F: C2Fn<f64, Output = f64>,
    {
        let d = D as f64;
        let v_a = a0 * sqrt(2.0 * self.kappa / d / (d - 1.0) * self.potential.value(phi0));
        let v_phi = -a0 * self.potential.value(phi0) / d / v_a;
        let mom_phi = v_phi * a0.powi(D as i32);
        field.phi.for_each(|ptr, _, _| *ptr = phi0);
        field.mom_phi.for_each(|ptr, _, _| *ptr = mom_phi);
    }
    pub fn init(&self, field: &mut ScalarFieldState<D>, a0: f64, phi0: f64, v_phi0: f64)
    where
        F: C2Fn<f64, Output = f64> + Send + Sync,
    {
        let mom_phi = v_phi0 * a0.powi(D as i32);
        field.b = Self::a_to_b(a0, self.kappa);
        field.phi.par_for_each_mut(|ptr, _, _| *ptr = phi0);
        field.mom_phi.par_for_each_mut(|ptr, _, _| *ptr = mom_phi);
        field.mom_b = Self::v_a_to_mom_b(self.v_a_from_hubble_constraint(&field), a0, self.kappa);
    }
    pub fn apply_k1(&self, field: &mut ScalarFieldState<D>, dt: f64) {
        field.b -= dt * field.mom_b;
    }
    pub fn apply_k2(&self, field: &mut ScalarFieldState<D>, dt: f64) {
        let d = D as f64;
        let b = field.b;
        field.mom_b += dt * 4.0 * (d - 1.0) / d / self.kappa
            * field.mom_phi.view().map(|f| f / b * f / b / b).average();
        field.phi.par_add_assign(
            &field
                .mom_phi
                .view()
                .mul_scalar(dt * 4.0 * (d - 1.0) / d / b / b / self.kappa),
        );
    }
    pub fn apply_k3(&self, field: &mut ScalarFieldState<D>, dt: f64)
    where
        F: C2Fn<f64, Output = f64> + Send + Sync,
    {
        let d = D as f64;
        let a = pow(2.0, 4.0 / d - 2.0)
            * pow((d - 1.0) / d, 2.0 / d - 1.0)
            * self.kappa.powf(1.0 - 2.0 / d);
        field.mom_b -= 0.5
            * a
            * (2.0 - 4.0 / d)
            * field.b.powf(1.0 - 4.0 / d)
            * field
                .phi
                .view()
                .derivative_square(&self.lattice.spacing)
                .average()
            + d * field.b * self.kappa / 2.0 / (d - 1.0)
                * field.phi.view().map(|f| self.potential.value(f)).average();
        field.mom_phi.par_for_each_mut(|ptr, index, coord| {
            let dd = a
                * field.b.powf(2.0 - 4.0 / d)
                * field.phi.laplacian_at(coord, &self.lattice.spacing)
                - d / 4.0 / (d - 1.0)
                    * field.b
                    * field.b
                    * self.kappa
                    * self.potential.value_d(field.phi.get(index, coord));
            *ptr += dd * dt;
        });
    }
    pub fn apply_full_k_order2(&self, field: &mut ScalarFieldState<D>, dt: f64)
    where
        F: C2Fn<f64, Output = f64> + Send + Sync,
    {
        self.apply_k1(field, dt / 2.0);
        self.apply_k2(field, dt / 2.0);
        self.apply_k3(field, dt);
        self.apply_k2(field, dt / 2.0);
        self.apply_k1(field, dt / 2.0);
    }
    pub fn scale_factor(&self, field: &ScalarFieldState<D>) -> f64 {
        let d = D as f64;
        let v = sqrt(d / (d - 1.0) * self.kappa);
        pow(field.b * v / 2.0, 2.0 / d)
    }
    pub fn v_a(&self, field: &ScalarFieldState<D>) -> f64 {
        Self::mom_b_to_v_a(field.mom_b, field.b, self.kappa)
    }
    pub fn hubble(&self, field: &ScalarFieldState<D>) -> f64 {
        self.v_a(field) / self.scale_factor(field)
    }
    pub fn v_phi_from_mom_phi(&self, b: f64, mom_phi: f64) -> f64 {
        let d = D as f64;
        4.0 * (d - 1.0) / d / self.kappa / b / b * mom_phi
    }
    pub fn v_phi_average(&self, field: &ScalarFieldState<D>) -> f64
    where
        F: Send + Sync,
    {
        field
            .mom_phi
            .view()
            .map(|mom_phi| self.v_phi_from_mom_phi(field.b, mom_phi))
            .average()
    }
}

impl<const D: usize, C> ScaleFactorD<C> for ScalarFieldState<D>
where
    C: Kappa,
{
    fn v_scale_factor(&self, ctx: &C) -> f64 {
        let a = self.scale_factor(ctx);
        scalar_field_simulator::mom_b_to_v_a(self.mom_b, D, a, ctx.kappa())
    }
}

pub fn effective_mom<const D: usize>(
    mode: &VecN<D, usize>,
    dx: &VecN<D, f64>,
    size: &VecN<D, usize>,
) -> f64 {
    let mut ret = 0f64;
    for i in 0..D {
        let l = mode[i] as f64;
        let n = size[i] as f64;
        let aa = sin(PI * l / n);
        ret += aa * aa / dx[i] / dx[i];
    }
    2.0 * ret.sqrt()
}

pub fn populate_noise<const D: usize, S>(
    lattice: &LatticeParam<D>,
    a: f64,
    v_a: f64,
    source: &mut S,
    noise_phi: &mut BoxLattice<D, Complex64>,
    noise_v_phi: &mut BoxLattice<D, Complex64>,
) where
    S: Source,
{
    let dim = &lattice.size;
    let inv_sqrt_volumn = 1.0 / lattice.volumn().sqrt();

    for index in 1..dim.product() {
        let coord = dim.decode_coord(index);
        let rev_coord = coord.flip(&dim);
        let phase = 2.0 * PI * source.read_f64();
        let m = sqrt(-source.read_f64().ln() / 2.0);
        let al = m * Complex64::new(phase.cos(), phase.sin());
        let k_eff = effective_mom(&coord, &lattice.spacing, &lattice.size);
        let u = inv_sqrt_volumn * Complex64::new(1.0 / sqrt(2.0 * k_eff), 0.0);
        let u_d = inv_sqrt_volumn * Complex64::new(0.0, -sqrt(k_eff / 2.0));
        *noise_phi.get_mut(index, &coord) += al * u;
        *noise_v_phi.get_mut(index, &coord) += al * u_d;
        *noise_phi.get_mut_by_coord(&rev_coord) += al.conj() * u.conj();
        *noise_v_phi.get_mut_by_coord(&rev_coord) += al.conj() * u_d.conj();
    }
    let fft = DftNDPlan::new(lattice.size.value, FftDirection::Forward);
    fft.transform_inplace(noise_phi);
    fft.transform_inplace(noise_v_phi);
    noise_v_phi.par_for_each_mut(|ptr, index, coord| {
        *ptr = *ptr / a / a - noise_phi.get(index, coord) * (v_a / a / a);
    });
    noise_phi.par_for_each_mut(|ptr, _, _| *ptr /= a);
}

pub struct ScalarFieldSimulator<'a, const D: usize, F> {
    pub input: &'a BackgroundStateInput<F>,
    pub lattice: LatticeParam<D>,
}

pub mod scalar_field_simulator {
    use libm::{pow, sqrt};

    pub fn a_to_b(a: f64, d: usize, kappa: f64) -> f64 {
        let d = d as f64;
        2.0 * sqrt((d - 1.0) / d / kappa) * a.powf(d / 2.0)
    }
    pub fn b_to_a(b: f64, d: usize, kappa: f64) -> f64 {
        let d = d as f64;
        pow(b / 2.0 * sqrt(kappa * d / (d - 1.0)), 2.0 / d)
    }
    pub fn v_a_to_mom_b(v_a: f64, d: usize, a: f64, kappa: f64) -> f64 {
        let d = d as f64;
        sqrt((d - 1.0) / d / kappa) * d * a.powf(d / 2.0 - 1.0) * v_a
    }
    pub fn mom_b_to_v_a(mom_b: f64, d: usize, a: f64, kappa: f64) -> f64 {
        let d = d as f64;
        -mom_b / (sqrt((d - 1.0) / d / kappa) * d * a.powf(d / 2.0 - 1.0))
    }
    pub fn mom_phi_to_v_phi_factor(d: usize, b: f64, kappa: f64) -> f64 {
        let d = d as f64;
        4.0 * (d - 1.0) / d / kappa / b / b
    }
}

impl<'a, const D: usize, F> ScalarFieldSimulator<'a, D, F> {
    pub fn v_a_to_mom_b(v_a: f64, a: f64, kappa: f64) -> f64 {
        let d = D as f64;
        sqrt((d - 1.0) / d / kappa) * d * a.powf(d / 2.0 - 1.0) * v_a
    }
    pub fn mom_b_to_v_a(mom_b: f64, b: f64, kappa: f64) -> f64 {
        let d = D as f64;
        let v_b = -mom_b;
        pow(d * kappa / (d - 1.0) / 4.0, 1.0 / d) * 2.0 / d * b.powf(2.0 / d - 1.0) * v_b
    }
    fn v_a_from_hubble_constraint(&self, field: &ScalarFieldState<D>) -> f64
    where
        F: C2Fn<f64, Output = f64> + Sync,
    {
        let d = D as f64;
        let a0 = scalar_field_simulator::b_to_a(field.b, D, self.input.kappa);
        let v_a2 = (field
            .phi
            .view()
            .derivative_square(&self.lattice.spacing)
            .average()
            + field
                .mom_phi
                .view()
                .map(|f| {
                    let c = f * a0.powi(2 - 2 * (D as i32));
                    c * c
                })
                .average()
            + field
                .phi
                .view()
                .map(|f| self.input.potential.value(f))
                .average()
                * 2.0
                * a0
                * a0)
            * self.input.kappa
            / d
            / (d - 1.0);
        -v_a2.sqrt()
    }
    pub fn init_slowroll(&self, field: &mut ScalarFieldState<D>, a0: f64, phi0: f64)
    where
        F: C2Fn<f64, Output = f64>,
    {
        let d = D as f64;
        let v_a =
            a0 * sqrt(2.0 * self.input.kappa / d / (d - 1.0) * self.input.potential.value(phi0));
        let v_phi = -a0 * self.input.potential.value(phi0) / d / v_a;
        let mom_phi = v_phi * a0.powi(D as i32);
        field.phi.for_each(|ptr, _, _| *ptr = phi0);
        field.mom_phi.for_each(|ptr, _, _| *ptr = mom_phi);
    }
    pub fn init(&self, field: &mut ScalarFieldState<D>, a0: f64, phi0: f64, v_phi0: f64)
    where
        F: C2Fn<f64, Output = f64> + Send + Sync,
    {
        let mom_phi = v_phi0 * a0.powi(D as i32);
        field.b = scalar_field_simulator::a_to_b(a0, D, self.input.kappa);
        field.phi.par_for_each_mut(|ptr, _, _| *ptr = phi0);
        field.mom_phi.par_for_each_mut(|ptr, _, _| *ptr = mom_phi);
        field.mom_b = Self::v_a_to_mom_b(
            self.v_a_from_hubble_constraint(&field),
            a0,
            self.input.kappa,
        );
    }
    pub fn apply_k1(&self, field: &mut ScalarFieldState<D>, dt: f64) {
        field.b -= dt * field.mom_b;
    }
    pub fn apply_k2(&self, field: &mut ScalarFieldState<D>, dt: f64) {
        let d = D as f64;
        let b = field.b;
        field.mom_b += dt * 4.0 * (d - 1.0) / d / self.input.kappa
            * field.mom_phi.view().map(|f| f / b * f / b / b).average();
        field.phi.par_add_assign(
            &field
                .mom_phi
                .view()
                .mul_scalar(dt * 4.0 * (d - 1.0) / d / b / b / self.input.kappa),
        );
    }
    pub fn apply_k3(&self, field: &mut ScalarFieldState<D>, dt: f64)
    where
        F: C2Fn<f64, Output = f64> + Sync,
    {
        let d = D as f64;
        let a = pow(2.0, 4.0 / d - 2.0)
            * pow((d - 1.0) / d, 2.0 / d - 1.0)
            * self.input.kappa.powf(1.0 - 2.0 / d);
        let phid_term = 0.5
            * a
            * (2.0 - 4.0 / d)
            * field.b.powf(1.0 - 4.0 / d)
            * field
                .phi
                .as_ref()
                .derivative_square(&self.lattice.spacing)
                .average();
        // let phid_term = 0.0; // XXX: ignoring derivative contribution?
        field.mom_b -= phid_term
            + d * field.b * self.input.kappa / 2.0 / (d - 1.0)
                * field
                    .phi
                    .view()
                    .map(|f| self.input.potential.value(f))
                    .average();
        field.mom_phi.par_for_each_mut(|ptr, index, coord| {
            let dd = a
                * field.b.powf(2.0 - 4.0 / d)
                * field.phi.laplacian_at(coord, &self.lattice.spacing)
                - d / 4.0 / (d - 1.0)
                    * field.b
                    * field.b
                    * self.input.kappa
                    * self.input.potential.value_d(field.phi.get(index, coord));
            *ptr += dd * dt;
        });
    }
    pub fn apply_full_k_order2(&self, field: &mut ScalarFieldState<D>, dt: f64)
    where
        F: C2Fn<f64, Output = f64> + Sync,
    {
        self.apply_k1(field, dt / 2.0);
        self.apply_k2(field, dt / 2.0);
        self.apply_k3(field, dt);
        self.apply_k2(field, dt / 2.0);
        self.apply_k1(field, dt / 2.0);
    }
    pub fn scale_factor(&self, field: &ScalarFieldState<D>) -> f64 {
        let d = D as f64;
        let v = sqrt(d / (d - 1.0) * self.input.kappa);
        pow(field.b * v / 2.0, 2.0 / d)
    }
    pub fn v_a(&self, field: &ScalarFieldState<D>) -> f64 {
        Self::mom_b_to_v_a(field.mom_b, field.b, self.input.kappa)
    }
    pub fn hubble(&self, field: &ScalarFieldState<D>) -> f64 {
        self.v_a(field) / self.scale_factor(field)
    }
    pub fn v_phi_from_mom_phi(&self, b: f64, mom_phi: f64) -> f64 {
        let d = D as f64;
        4.0 * (d - 1.0) / d / self.input.kappa / b / b * mom_phi
    }
    pub fn v_phi_average(&self, field: &ScalarFieldState<D>) -> f64
    where
        F: Send + Sync,
    {
        field
            .mom_phi
            .view()
            .map(|mom_phi| self.v_phi_from_mom_phi(field.b, mom_phi))
            .average()
    }
}

impl<'a, const D: usize, F> LatticeSimulator for ScalarFieldSimulator<'a, D, F>
where
    F: C2Fn<f64, Output = f64> + Sync,
{
    type LatticeState = ScalarFieldState<D>;

    fn update_lattice_state(&mut self, field: &mut Self::LatticeState, dt: f64) {
        self.apply_full_k_order2(field, dt);
    }
    fn hubble_constraint(&self, field: &Self::LatticeState) -> f64 {
        let d = D as f64;
        let a = field.scale_factor(self.input);
        let v_a = field.v_scale_factor(self.input);
        let hubble = v_a / a;
        let phi = field.phi_field(self.input);
        let v_phi = field.v_phi_field(self.input);
        let kappa = self.input.kappa;
        -(d - 1.0) * d * hubble * hubble / kappa
            + phi
                .as_ref()
                .derivative_square(&self.lattice.spacing)
                .average()
                / a
                / a
            + v_phi.as_ref().map(|f| f * f).average()
            + 2.0
                * phi
                    .as_ref()
                    .map(|f| self.input.potential.value(f))
                    .average()
    }
}

pub struct ScalarFieldSimulatorCreator;

impl<const D: usize, F> LatticeSimulatorCreator<D, BackgroundStateInput<F>>
    for ScalarFieldSimulatorCreator
{
    type Simulator<'a>
        = ScalarFieldSimulator<'a, D, F>
    where
        F: 'a;

    fn create<'a>(
        &self,
        lattice: LatticeParam<D>,
        input: &'a BackgroundStateInput<F>,
    ) -> Self::Simulator<'a>
    where
        BackgroundStateInput<F>: 'a,
    {
        ScalarFieldSimulator { lattice, input }
    }
}

impl<'a, const D: usize, S, F> LatticeInitializer<S> for ScalarFieldSimulator<'a, D, F>
where
    S: ScaleFactor<BackgroundStateInput<F>>
        + ScaleFactorD<BackgroundStateInput<F>>
        + Phi<BackgroundStateInput<F>>
        + PhiD<BackgroundStateInput<F>>,
    F: C2Fn<f64, Output = f64> + Sync,
{
    type LatticeState = ScalarFieldState<D>;
    fn init_from_state(&self, field: &mut Self::LatticeState, state: &S) {
        let a0 = state.scale_factor(self.input);
        let phi0 = state.phi(self.input);
        let v_phi0 = state.v_phi(self.input);
        let mom_phi = v_phi0 * a0.powi(D as i32);
        field.b = scalar_field_simulator::a_to_b(a0, D, self.input.kappa);
        field.phi.par_for_each_mut(|ptr, _, _| *ptr = phi0);
        field.mom_phi.par_for_each_mut(|ptr, _, _| *ptr = mom_phi);
        field.mom_b = Self::v_a_to_mom_b(
            self.v_a_from_hubble_constraint(&field),
            a0,
            self.input.kappa,
        );
    }
}

impl<'a, const D: usize, F> LatticeNoiseGenerator for ScalarFieldSimulator<'a, D, F>
where
    F: C2Fn<f64, Output = f64> + Sync,
{
    type LatticeState = ScalarFieldState<D>;
    fn populate_noise<S: Source>(&self, field: &mut Self::LatticeState, rand: &mut S) {
        let a = field.scale_factor(self.input);
        let v_a = field.v_scale_factor(self.input);
        let mut noise_phi = BoxLattice::<D, Complex64>::zeros(self.lattice.size);
        let mut noise_v_phi = BoxLattice::<D, Complex64>::zeros(self.lattice.size);
        populate_noise(
            &self.lattice,
            a,
            v_a,
            rand,
            &mut noise_phi,
            &mut noise_v_phi,
        );

        let kappa = self.input.kappa;

        let a_dimx = scalar_field_simulator::b_to_a(field.b, D, kappa).powi(D as i32);
        field.phi.par_add_assign(&noise_phi.view().map(|f| f.re));
        field
            .mom_phi
            .par_add_assign(&noise_v_phi.view().map(|f| f.re * a_dimx));
        // XXX: ignoring effects on Hubble constraint
        field.mom_b = Self::v_a_to_mom_b(
            self.v_a_from_hubble_constraint(field),
            scalar_field_simulator::b_to_a(field.b, D, kappa),
            kappa,
        );
    }
}

impl<const D: usize, C> ScaleFactor<C> for ScalarFieldState<D>
where
    C: Kappa,
{
    fn scale_factor(&self, ctx: &C) -> f64 {
        scalar_field_simulator::b_to_a(self.b, D, ctx.kappa())
    }
}

pub struct SpectrumEntry {
    pub k: f64,
    pub amp: f64,
    pub error: f64,
}

struct MomBin {
    pub total_k_eff: f64,
    pub total_amp: f64,
    pub total_amp_square: f64,
    pub count: usize,
}

impl MomBin {
    pub fn zero() -> Self {
        Self {
            total_k_eff: 0.0,
            total_amp: 0.0,
            total_amp_square: 0.0,
            count: 0,
        }
    }
    pub fn add(&mut self, k_eff: f64, amp: f64) {
        self.total_amp += amp;
        self.total_amp_square += amp * amp;
        self.total_k_eff += k_eff;
        self.count += 1;
    }
    pub fn average(&self, dim: usize) -> (f64, f64) {
        let k_eff = self.total_k_eff / (self.count as f64);
        let amp = self.total_amp / (self.count as f64);
        (k_eff, k_eff.powi(dim as i32) * spectrum_factor(dim) * amp)
    }
}

fn normalize_mom_mode<const D: usize>(
    mom: VecN<D, usize>,
    size: &VecN<D, usize>,
) -> VecN<D, usize> {
    VecN::new({
        let mut c = mom.value;
        for (cc, size) in zip(&mut c, size) {
            if *cc > size / 2 {
                *cc = size - *cc;
            }
        }
        c.sort();
        c
    })
}

pub fn spectrum_with_scratch<const D: usize, Phi>(
    phi: &Phi,
    lattice: &LatticeParam<D>,
    scratch_field: &mut BoxLattice<D, Complex64>,
) -> Vec<(f64, f64)>
where
    Phi: Lattice<D, f64> + Sync,
{
    let dft = DftNDPlan::new(lattice.size.value, FftDirection::Inverse);
    scratch_field.par_assign(&{
        let phi2 = phi.as_ref();
        let avg_phi = phi2.average();
        phi2.map(move |f| (f - avg_phi).into())
    });
    dft.transform_inplace(scratch_field);
    let factor = lattice.spacing.product() / (lattice.size.product() as f64);
    let mut spectrum_bins = HashMap::new();
    let dim = lattice.size;
    for index in 0..dim.product() {
        let coord = dim.decode_coord(index);
        let rev_coord = coord.flip(&dim);
        let mode = normalize_mom_mode(coord, &lattice.size);
        let l = (mode.value.iter().map(|f| f * f).sum::<usize>() as f64)
            .sqrt()
            .round() as usize;
        if l != 0 {
            let value = scratch_field.get_by_coord(&coord) * scratch_field.get_by_coord(&rev_coord);
            if !spectrum_bins.contains_key(&l) {
                spectrum_bins.insert(l, MomBin::zero());
            }
            spectrum_bins.get_mut(&l).unwrap().add(
                effective_mom(&mode, &lattice.spacing, &lattice.size),
                factor * value.re,
            );
        }
    }
    let mut ret = spectrum_bins
        .iter()
        .map(|(_, bin)| bin.average(D))
        .collect::<Vec<_>>();
    ret.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    ret
}

pub fn spectrum<const D: usize, Phi>(phi: &Phi, lattice: &LatticeParam<D>) -> Vec<(f64, f64)>
where
    Phi: Lattice<D, f64> + Sync,
{
    let mut scratch_field = BoxLattice::zeros(lattice.size);
    spectrum_with_scratch(phi, lattice, &mut scratch_field)
}

pub trait ScalarEffectivePotential {
    fn scalar_eff_potential(&self, a: f64, v_a: f64, phi: f64, v_phi: f64) -> f64;
}

pub fn construct_zeta_inplace<const D: usize, Zeta, Phi1, Phi2, Solver, M>(
    zeta: &mut Zeta,
    a: f64,
    v_a: f64,
    phi: &Phi1,
    v_phi: &Phi2,
    reference_phi: f64,
    dt_segs: usize,
    max_dt: f64,
    solver: &Solver,
    step_monitor: M,
) -> bool
where
    Phi1: Lattice<D, f64> + Sync,
    Phi2: Lattice<D, f64> + Sync,
    Zeta: LatticeMut<D, f64>,
    Solver: BackgroundSolver + Kappa + Sync,
    Solver::State: Clone + Phi<Solver> + PhiD<Solver> + ScaleFactor<Solver> + Debug,
    M: Fn(usize, usize) + Sync,
{
    let total_count = phi.dim().product();
    let done_count = AtomicUsize::new(0);
    let mut terminated = AtomicBool::new(false);
    let sm = &step_monitor;
    let tm = &terminated;
    zeta.par_assign(&phi.as_ref().zip(v_phi.as_ref()).map(move |(phi, v_phi)| {
        if tm.load(Ordering::SeqCst) {
            reference_phi
        } else {
            let dt = fmin(max_dt, (reference_phi - phi) / v_phi / (dt_segs as f64));
            let mut state = solver.create_state(a, v_a, phi, v_phi);
            solver.evaluate_to_phi(&mut state, dt, reference_phi);
            sm(done_count.fetch_add(1, Ordering::SeqCst) + 1, total_count);
            let ret = (state.scale_factor(solver) / a).ln();
            if ret.is_infinite() || ret.is_nan() {
                println!(
                    "[warning/zeta] found NaN, a = {}, v_a = {}, phi = {}, v_phi = {}, final_state = {:?}",
                    a, v_a, phi, v_phi, &state
                );
            }
            ret
        }
    }));
    !*terminated.get_mut()
}

pub fn construct_zeta<const D: usize, Phi1, Phi2, Solver, M>(
    a: f64,
    v_a: f64,
    phi: &Phi1,
    v_phi: &Phi2,
    reference_phi: f64,
    dt_segs: usize,
    max_dt: f64,
    solver: &Solver,
    step_monitor: M,
) -> BoxLattice<D, f64>
where
    Phi1: Lattice<D, f64> + Sync,
    Phi2: Lattice<D, f64> + Sync,
    Solver: BackgroundSolver + Kappa + Sync,
    Solver::State: Clone + Phi<Solver> + PhiD<Solver> + ScaleFactor<Solver> + Debug,
    M: Fn(usize, usize) + Sync,
{
    let mut zeta = BoxLattice::zeros(*phi.dim());
    construct_zeta_inplace(
        &mut zeta,
        a,
        v_a,
        phi,
        v_phi,
        reference_phi,
        dt_segs,
        max_dt,
        solver,
        step_monitor,
    );
    zeta
}

pub trait LatticePhi<const D: usize, C> {
    type Phi<'a>: Lattice<D, f64>
    where
        Self: 'a;
    fn phi_field<'a>(&'a self, ctx: &C) -> Self::Phi<'a>;

    fn phi_average<'a>(&'a self, ctx: &C) -> f64
    where
        Self::Phi<'a>: Sync,
    {
        self.phi_field(ctx).average()
    }
}

pub trait LatticePhiD<const D: usize, C> {
    type PhiD<'a>: Lattice<D, f64>
    where
        Self: 'a;
    fn v_phi_field<'a>(&'a self, ctx: &C) -> Self::PhiD<'a>;

    fn v_phi_average<'a>(&'a self, ctx: &C) -> f64
    where
        Self::PhiD<'a>: Sync,
    {
        self.v_phi_field(ctx).average()
    }
}

impl<const D: usize, C> LatticePhi<D, C> for ScalarFieldState<D> {
    type Phi<'a> = LatticeRef<'a, BoxLattice<D, f64>>;

    fn phi_field<'a>(&'a self, _: &C) -> Self::Phi<'a> {
        self.phi.as_ref()
    }
}

impl<const D: usize, C> LatticePhiD<D, C> for ScalarFieldState<D>
where
    C: Kappa,
{
    type PhiD<'a>
        = LatticeScalarMul<D, f64, LatticeRef<'a, BoxLattice<D, f64>>>
    where
        Self: 'a;

    fn v_phi_field<'a>(&'a self, ctx: &C) -> Self::PhiD<'a> {
        self.mom_phi
            .as_ref()
            .mul_scalar(scalar_field_simulator::mom_phi_to_v_phi_factor(
                D,
                self.b,
                ctx.kappa(),
            ))
    }
}

pub trait LatticeSimulatorCreator<const D: usize, I> {
    type Simulator<'a>
    where
        I: 'a;
    fn create<'a>(&self, lattice: LatticeParam<D>, input: &'a I) -> Self::Simulator<'a>
    where
        I: 'a;
}

pub trait LatticeSimulator {
    type LatticeState: ?Sized;
    fn update_lattice_state(&mut self, field: &mut Self::LatticeState, dt: f64);
    fn metric_perturbations(&self, _field: &Self::LatticeState) -> (f64, f64) {
        (0.0, 0.0)
    }
    fn hubble_constraint(&self, _field: &Self::LatticeState) -> f64 {
        0.0
    }
}

pub trait LatticeInitializer<S: ?Sized> {
    type LatticeState: ?Sized;
    fn init_from_state(&self, field: &mut Self::LatticeState, state: &S);
}

pub trait LatticeNoiseGenerator {
    type LatticeState: ?Sized;
    fn populate_noise<S: Source>(&self, field: &mut Self::LatticeState, rand: &mut S);
}

pub trait LatticeState<const D: usize> {
    fn zero(size: VecN<D, usize>) -> Self
    where
        Self: Sized;
}

impl<const D: usize> LatticeState<D> for ScalarFieldState<D> {
    fn zero(size: VecN<D, usize>) -> Self
    where
        Self: Sized,
    {
        Self {
            phi: BoxLattice::zeros(size),
            mom_phi: BoxLattice::zeros(size),
            mom_b: 0.0,
            b: 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy, Encode, Decode)]
pub struct LatticeMeasurables {
    pub a: f64,
    pub v_a: f64,
    pub phi: f64,
    pub v_phi: f64,
    pub hubble_constraint: f64,
    pub metric_perts: (f64, f64),
}

#[derive(Debug)]
pub struct LatticeInput<const D: usize, S> {
    pub lattice: LatticeParam<D>,
    pub state: S,
    pub dt: f64,
    pub end_n: f64,
    pub k_unit: f64,
    pub no_noise: bool,
    pub seed: u64,
}

#[derive(Encode, Decode)]
pub struct LatticeOutputData<const D: usize, S> {
    pub measurables: Vec<LatticeMeasurables>,
    pub final_state: S,
    pub spectrum_k: Vec<f64>,
    pub spectrums: Vec<(f64, Vec<f64>)>,
    pub zeta_spectrum: Vec<f64>,
    pub final_zeta: BoxLattice<D, f64>,
}

impl<const D: usize, S> LatticeInput<D, S> {
    pub fn from_background_and_k_normalized<C: ?Sized>(
        background: &[S],
        ctx: &C,
        start_k: f64,
        subhorizon_tolerance: f64,
        superhorizon_tolerance: f64,
        lattice_size: usize,
        dt: f64,
    ) -> Self
    where
        S: ScaleFactor<C> + ScaleFactorD<C> + ScaleFactorMut<C> + Clone,
    {
        let starting_horizon = start_k / subhorizon_tolerance;
        let end_k = start_k * sqrt(D as f64) / PI * (lattice_size as f64);
        let end_horizon = end_k * superhorizon_tolerance;
        let start_state = background
            .iter()
            .find(|state| state.v_scale_factor(ctx) >= starting_horizon)
            .unwrap();
        let start_k_state = background
            .iter()
            .find(|state| state.v_scale_factor(ctx) >= start_k)
            .unwrap();
        let end_state = background
            .iter()
            .find(|state| state.v_scale_factor(ctx) >= end_horizon)
            .unwrap();
        let start_hubble = start_state.hubble(ctx);
        let normalized_start_k = start_k_state.v_scale_factor(ctx) / start_state.scale_factor(ctx);
        let dx = 2.0 * PI / normalized_start_k / (lattice_size as f64);
        let lattice = LatticeParam {
            size: VecN::new([lattice_size; D]),
            spacing: VecN::new([dx; D]),
        };
        let mut state = start_state.clone();
        state.set_scale_factor(ctx, 1.0, start_hubble);
        Self {
            lattice,
            state,
            dt,
            end_n: (end_state.scale_factor(ctx) / start_state.scale_factor(ctx)).ln(),
            k_unit: start_k / normalized_start_k,
            no_noise: false,
            seed: 10,
        }
    }
    pub fn run<'a, LatState, SimCreator, I>(
        &self,
        simulator_creator: &SimCreator,
        input: &'a I,
        out_file: &str,
        create: bool,
        spectrum_count: usize,
    ) -> util::Result<LatticeOutputData<D, LatState>>
    where
        SimCreator: LatticeSimulatorCreator<D, I>,
        SimCreator::Simulator<'a>: LatticeSimulator<LatticeState = LatState>
            + LatticeNoiseGenerator<LatticeState = LatState>
            + LatticeInitializer<S, LatticeState = LatState>,
        LatState: LatticeState<D>
            + Decode<()>
            + Encode
            + ScaleFactor<I>
            + ScaleFactorD<I>
            + LatticePhi<D, I>
            + LatticePhiD<D, I>,
        for<'b> LatState::Phi<'b>: Sync,
        for<'b> LatState::PhiD<'b>: Sync,
        S: ScaleFactor<I> + ScaleFactorD<I> + Phi<I> + PhiD<I>,
        I: ScalarEffectivePotential + BackgroundSolver + Kappa + Sync,
        I::State: Clone + Phi<I> + PhiD<I> + ScaleFactor<I> + ScaleFactorD<I> + Debug,
        Self: Debug,
    {
        lazy_file_opt(
            &format!("{}.bincode", out_file),
            BINCODE_CONFIG,
            create,
            || {
                println!("[lattice] input = {{H = {:e}, phi = {:e}, v_phi = {:e}}}", self.state.hubble(input), self.state.phi(input), self.state.v_phi(input));
                let mut simulator = simulator_creator.create(self.lattice, input);
                let checkpoint_file_name = format!("{}.checkpoint.bincode", out_file);
                let mut lattice_state = if let Ok(state) =
                    decode_from_file(&checkpoint_file_name, BINCODE_CONFIG)
                {
                    println!(
                        "[lattice] read from previous saved state {}",
                        &checkpoint_file_name
                    );
                    state
                } else {
                    let mut lattice_state = LatState::zero(self.lattice.size);
                    simulator.init_from_state(&mut lattice_state, &self.state);
                    if !self.no_noise {
                        simulator
                            .populate_noise(&mut lattice_state, &mut random::default(self.seed));
                    }
                    lattice_state
                };
                println!(
                    "[lattice] background H = {}, initial H = {}, dx = {}",
                    self.state.hubble(input),
                    lattice_state.hubble(input),
                    self.lattice.spacing[0],
                );
                let mut spectrum_scratch = BoxLattice::zeros(self.lattice.size);
                let initial_spectrum = spectrum_with_scratch(
                    &lattice_state.phi_field(input),
                    &self.lattice,
                    &mut spectrum_scratch,
                );
                let spectrum_k = initial_spectrum.iter().map(|f| f.0 * self.k_unit).collect();
                let mut spectrums = vec![(self.state.scale_factor(input).ln(), {
                    let zeta_factor = lattice_state.scale_factor(input);
                    initial_spectrum
                        .iter()
                        .map(|f| f.1 * zeta_factor * zeta_factor)
                        .collect()
                })];
                let n_range = self.state.scale_factor(input).ln()..self.end_n;
                let spectrum_delta_n = (n_range.end - n_range.start) / (spectrum_count as f64);
                let mut next_spectrum_n = n_range.start + spectrum_delta_n;
                let mut measurables = vec![];
                let mut rate_limiter = RateLimiter::new(Duration::from_millis(2000));
                let mut time_estimator = TimeEstimator::new(n_range.clone(), 100);
                let mut last_checkpoint_time = SystemTime::now();
                while lattice_state.scale_factor(input).ln() < n_range.end {
                    simulator.update_lattice_state(&mut lattice_state, self.dt);
                    time_estimator.update(lattice_state.scale_factor(input).ln());
                    if let Ok(elapsed) = last_checkpoint_time.elapsed() {
                        if elapsed.as_secs() >= 600 {
                            match encode_to_file(
                                &checkpoint_file_name,
                                BINCODE_CONFIG,
                                &lattice_state,
                            ) {
                                Ok(_) => {
                                    println!("[lattice] saved checkpoint {}", &checkpoint_file_name)
                                }
                                Err(err) => println!(
                                    "[lattice] failed to save checkpoint {}: {}",
                                    &checkpoint_file_name, &err
                                ),
                            }
                            last_checkpoint_time = SystemTime::now();
                        }
                    } else {
                        last_checkpoint_time = SystemTime::now();
                    }
                    let state = LatticeMeasurables {
                        a: lattice_state.scale_factor(input),
                        v_a: lattice_state.v_scale_factor(input),
                        phi: lattice_state.phi_average(input),
                        v_phi: lattice_state.v_phi_average(input),
                        metric_perts: simulator.metric_perturbations(&lattice_state),
                        hubble_constraint: simulator.hubble_constraint(&lattice_state),
                    };
                    rate_limiter.run(|| {
                        println!(
                            "[lattice] eta remaining = {}, step = {}, measurables = {:?}",
                            Hms::from_secs(time_estimator.remaining_secs()),
                            measurables.len(),
                            &state
                        )
                    });
                    measurables.push(state);
                    if lattice_state.scale_factor(input).ln() >= next_spectrum_n {
                        next_spectrum_n += spectrum_delta_n;
                        let zeta_factor =
                            lattice_state.hubble(input) / lattice_state.v_phi_average(input);
                        let spec = spectrum_with_scratch(
                            &lattice_state.phi_field(input),
                            &self.lattice,
                            &mut spectrum_scratch,
                        );
                        spectrums.push((
                            lattice_state.scale_factor(input).ln(),
                            spec.iter()
                                .map(|f| f.1 * zeta_factor * zeta_factor)
                                .collect(),
                        ));
                    }
                }
                let mut zeta = BoxLattice::zeros(self.lattice.size);
                let zeta_spectrum = {
                    let percentage = AtomicUsize::new(0);
                    let denom = 100usize;
                    // let reference_phi_old = simulator.field.phi.as_ref().map(|f| f[0]).max().1;
                    let reference_phi = {
                        let a = lattice_state.scale_factor(input);
                        let v_a = lattice_state.v_scale_factor(input);
                        let coord = lattice_state
                            .phi_field(input)
                            .zip(lattice_state.v_phi_field(input))
                            .map(move |(phi, v_phi)| input.scalar_eff_potential(a, v_a, phi, v_phi))
                            .min()
                            .0;
                        lattice_state.phi_field(input).get_by_coord(&coord)
                    };
                    println!(
                        "[lattice] calculating spectrum, reference_phi = {}",
                        reference_phi
                    );
                    construct_zeta_inplace(
                        &mut zeta,
                        lattice_state.scale_factor(input),
                        lattice_state.v_scale_factor(input),
                        &lattice_state.phi_field(input),
                        &lattice_state.v_phi_field(input),
                        reference_phi,
                        100,
                        10.0,
                        input,
                        |count, total| {
                            let p =
                                ((count as f64) / (total as f64) * (denom as f64)).floor() as usize;
                            let c = percentage.fetch_max(p, std::sync::atomic::Ordering::SeqCst);
                            if p != c {
                                println!("[zeta] {:.2}%", (p as f64) / ((denom / 100) as f64));
                            }
                        },
                    );
                    spectrum_with_scratch(&zeta, &self.lattice, &mut spectrum_scratch)
                        .iter()
                        .map(|f| f.1)
                        .collect()
                };
                LatticeOutputData {
                    measurables,
                    spectrum_k,
                    spectrums,
                    final_state: lattice_state,
                    final_zeta: zeta,
                    zeta_spectrum,
                }
            },
        )
    }
}

impl<const D: usize, S> LatticeOutputData<D, S> {
    pub fn plot_background(&self, out_file: &str) {
        let mut efolding = vec![];
        let mut phi = vec![];
        let mut v_phi = vec![];
        let mut hubble = vec![];
        let mut pert_a = vec![];
        let mut pert_b = vec![];
        let mut hubble_constraint = vec![];
        for state in limit_length(&self.measurables, 500000) {
            efolding.push(state.a.ln());
            phi.push(state.phi);
            v_phi.push(state.v_phi);
            hubble.push(state.v_a / state.a);
            pert_a.push(state.metric_perts.0.abs());
            pert_b.push(state.metric_perts.1.abs());
            hubble_constraint.push(state.hubble_constraint.abs());
        }
        let mut plot = Plot::new();
        plot.add_trace(Scatter::new(efolding.clone(), phi).name("phi"));
        plot.add_trace(
            Scatter::new(efolding.clone(), v_phi)
                .name("v_phi")
                .y_axis("y2"),
        );
        plot.add_trace(
            Scatter::new(efolding.clone(), hubble)
                .name("H")
                .y_axis("y3"),
        );
        plot.add_trace(
            Scatter::new(efolding.clone(), pert_a)
                .name("|A|")
                .y_axis("y4"),
        );
        plot.add_trace(
            Scatter::new(efolding.clone(), pert_b)
                .name("|\\Box B / a^2|")
                .y_axis("y5"),
        );
        plot.add_trace(
            Scatter::new(efolding.clone(), hubble_constraint)
                .name("hubble_constraint")
                .y_axis("y6"),
        );
        plot.set_layout(
            Layout::new()
                .grid(LayoutGrid::new().rows(6).columns(1))
                .x_axis(Axis::new().exponent_format(ExponentFormat::Power))
                .y_axis(Axis::new().exponent_format(ExponentFormat::Power))
                .y_axis2(Axis::new().exponent_format(ExponentFormat::Power))
                .y_axis3(Axis::new().exponent_format(ExponentFormat::Power))
                .y_axis4(
                    Axis::new()
                        .type_(AxisType::Log)
                        .exponent_format(ExponentFormat::Power),
                )
                .y_axis5(
                    Axis::new()
                        .type_(AxisType::Log)
                        .exponent_format(ExponentFormat::Power),
                )
                .y_axis6(
                    Axis::new()
                        .type_(AxisType::Log)
                        .exponent_format(ExponentFormat::Power),
                )
                .height(1200),
        );
        plot.write_html(out_file);
    }
    pub fn plot_spectrums(&self, out_file: &str, k_star: f64) {
        let mut plot = Plot::new();
        let ks = self
            .spectrum_k
            .iter()
            .map(|k| k / k_star)
            .collect::<Vec<_>>();
        for (n, spec) in &self.spectrums {
            plot.add_trace(Scatter::new(ks.clone(), spec.clone()).name(&format!("N = {}", n)));
        }
        plot.add_trace(
            Scatter::new(ks, self.zeta_spectrum.clone())
                .name("final")
                .y_axis("y2"),
        );
        plot.set_layout(
            Layout::new()
                .grid(LayoutGrid::new().rows(2).columns(1))
                .x_axis(
                    Axis::new()
                        .type_(AxisType::Log)
                        .exponent_format(ExponentFormat::Power),
                )
                .y_axis(
                    Axis::new()
                        .type_(AxisType::Log)
                        .exponent_format(ExponentFormat::Power),
                )
                .y_axis2(
                    Axis::new()
                        .type_(AxisType::Log)
                        .exponent_format(ExponentFormat::Power),
                )
                .height(1600),
        );
        plot.write_html(out_file);
    }
    pub fn plot_all(
        &self,
        out_dir: &str,
        name: &str,
        k_unit: f64,
        spectrum_k_range: ParamRange<f64>,
        linear_spectrum: &[f64],
    ) {
        self.plot_background(&format!("{}/{}.lattice.background.html", out_dir, name));
        self.plot_spectrums(
            &format!("{}/{}.lattice.spectrums.html", out_dir, name),
            k_unit,
        );
        plot_spectrum(
            &format!("{}/{}.lattice.combined_spectrum.html", out_dir, name),
            &[
                (
                    "tree",
                    &spectrum_k_range.as_logspace().collect::<Vec<_>>(),
                    linear_spectrum,
                ),
                (
                    "lattice",
                    remove_first_and_last(&self.spectrum_k),
                    remove_first_and_last(&self.zeta_spectrum),
                ),
            ],
            k_unit,
        );
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        lat::{BoxLattice, Lattice, LatticeParam},
        scalar::populate_noise,
        util::VecN,
    };

    fn run_noise_test(size: usize, l: f64) -> (f64, f64) {
        let dx = l / (size as f64);
        let lattice = LatticeParam {
            spacing: VecN::new([dx; 3]),
            size: VecN::new([size; 3]),
        };
        let mut noise_phi = BoxLattice::zeros(lattice.size);
        let mut noise_v_phi = BoxLattice::zeros(lattice.size);
        populate_noise(
            &lattice,
            10.0,
            1e-2,
            &mut random::default(1),
            &mut noise_phi,
            &mut noise_v_phi,
        );
        (
            noise_phi.as_ref().map(|f| f.re * f.re).average(),
            noise_phi
                .as_ref()
                .map(|f| f.re)
                .derivative_square(&lattice.spacing)
                .average(),
        )
    }

    #[test]
    fn noise() {
        let (phi1, ds1) = run_noise_test(16, 1.0);
        let (phi2, ds2) = run_noise_test(64, 1.0);
        println!("phi1 = {phi1}, phi2 = {phi2}, ds1 = {ds1}, ds2 = {ds2}");
        assert!((phi1 - phi2).abs() < 1e-10);
        assert!((ds1 - ds2).abs() < 1e-10);
    }
}
