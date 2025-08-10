use std::{
    collections::HashMap,
    f64::consts::PI,
    fmt::Debug,
    iter::zip,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};

use bincode::{Decode, Encode};
use libm::{fmin, pow, sin, sqrt};
use num_complex::ComplexFloat;
use random::Source;
use rustfft::{FftDirection, num_complex::Complex64};

use crate::{
    background::{BackgroundSolver, Kappa, Phi, PhiD, ScaleFactor, spectrum_factor},
    c2fn::C2Fn,
    fft::DftNDPlan,
    lat::{BoxLattice, Lattice, LatticeMut, LatticeParam},
    util::VecN,
};

#[derive(Encode, Decode)]
pub struct ScalarFieldState<const D: usize> {
    pub b: f64,
    pub mom_b: f64,
    pub phi: BoxLattice<D, f64>,
    pub mom_phi: BoxLattice<D, f64>,
}

impl<const D: usize> ScalarFieldState<D> {
    pub fn zeros(dim: VecN<D, usize>) -> Self {
        Self {
            phi: BoxLattice::zeros(dim),
            mom_phi: BoxLattice::zeros(dim),
            mom_b: 0.0,
            b: 0.0,
        }
    }
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
    pub fn populate_noise<S: Source>(&self, source: &mut S, field: &mut ScalarFieldState<D>)
    where
        F: C2Fn<f64, Output = f64> + Send + Sync,
    {
        let a = Self::b_to_a(field.b, self.kappa);
        let v_a = Self::v_a_from_mom_b(field.mom_b, a, self.kappa);
        let mut noise_phi = BoxLattice::<D, Complex64>::zeros(self.lattice.size);
        let mut noise_v_phi = BoxLattice::<D, Complex64>::zeros(self.lattice.size);
        populate_noise(
            &self.lattice,
            a,
            v_a,
            source,
            &mut noise_phi,
            &mut noise_v_phi,
        );

        let a_dimx = Self::b_to_a(field.b, self.kappa).powi(D as i32);
        field.phi.par_add_assign(&noise_phi.view().map(|f| f.re));
        field
            .mom_phi
            .par_add_assign(&noise_v_phi.view().map(|f| f.re * a_dimx));
        field.mom_b = Self::v_a_to_mom_b(
            self.v_a_from_hubble_constraint(field),
            Self::b_to_a(field.b, self.kappa),
            self.kappa,
        );
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

fn effective_mom<const D: usize>(
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
    Solver::State: Clone + Phi + PhiD + ScaleFactor + Debug,
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
            let ret = (state.scale_factor() / a).ln();
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
    Solver::State: Clone + Phi + PhiD + ScaleFactor + Debug,
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
