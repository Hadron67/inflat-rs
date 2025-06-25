use std::f64::consts::PI;

use bincode::{Decode, Encode};
use libm::{pow, sin, sqrt};
use random::Source;
use rustfft::{FftDirection, num_complex::Complex64};

use crate::{
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
        v_a2.sqrt()
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
        let dim = *field.phi.dim();
        let volumn = self.lattice.spacing.product() * (self.lattice.size.product() as f64);
        let mut noise_phi = BoxLattice::<D, Complex64>::zeros(self.lattice.size);
        let mut noise_v_phi = BoxLattice::<D, Complex64>::zeros(self.lattice.size);
        for index in 0..dim.product() {
            let coord = dim.decode_coord(index);
            let rev_coord = coord.flip(&dim);
            let phase = 2.0 * PI * source.read_f64();
            let m = sqrt(-source.read_f64().ln() / 2.0);
            let al = m * Complex64::new(phase.cos(), phase.sin());
            let k_eff = effective_mom(&coord, &self.lattice.spacing, &self.lattice.size);
            let u = volumn.sqrt() * Complex64::new(1.0 / sqrt(2.0 * k_eff), 0.0);
            let u_d = volumn.sqrt() * Complex64::new(0.0, -sqrt(k_eff / 2.0));
            *noise_phi.get_mut(index, &coord) += u * al;
            *noise_v_phi.get_mut(index, &coord) += u_d * al;
            *noise_phi.get_mut_by_coord(&rev_coord) += al.conj() * u.conj();
            *noise_v_phi.get_mut_by_coord(&rev_coord) += al.conj() * u_d.conj();
        }
        let fft = DftNDPlan::new(self.lattice.size.value, FftDirection::Forward);
        fft.transform_inplace(&mut noise_phi);
        fft.transform_inplace(&mut noise_v_phi);
        let a_dimx = Self::b_to_a(field.b, self.kappa).powi(D as i32);
        field.phi.par_add_assign(&noise_phi.view().map(|f| f.re));
        field
            .mom_phi
            .par_add_assign(&noise_v_phi.view().map(|f| f.re * a_dimx));
        field.mom_b = -self.v_a_from_hubble_constraint(field);
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
            * field.b.powf(2.0 - 4.0 / d)
            * self.kappa.powf(1.0 - 2.0 / d);
        field.mom_b -= d * field.b * self.kappa / 2.0 / (d - 1.0)
            * field.phi.view().map(|f| self.potential.value(f)).average();
        field.mom_phi.par_for_each_mut(|ptr, index, coord| {
            let dd = a * field.phi.laplacian_at(coord, &self.lattice.spacing)
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
    pub fn v_phi(&self, field: &ScalarFieldState<D>) -> f64
    where
        F: Send + Sync,
    {
        let d = D as f64;
        field
            .mom_phi
            .view()
            .map(|mom_phi| 4.0 * (d - 1.0) / d / self.kappa / field.b / field.b * mom_phi)
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
