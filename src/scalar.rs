use std::{f64::consts::PI, ops::AddAssign};

use bincode::{Decode, Encode};
use libm::{cos, log, pow, sin, sqrt};
use random::Source;
use rustfft::{FftDirection, num_complex::Complex64};

use crate::{
    c2fn::C2Fn, fft::Dft3DPlan, field::{Dim, LatticeLike, LatticeMutLike, Vec3i}, lat::{BoxLattice, Lattice, LatticeMut, LatticeParam}
};

// #[derive(Clone, Copy)]
// pub struct SimulateParams {
//     pub time_step: f64,
//     pub kappa: f64,
//     pub dim: Dim,
//     pub lattice_spacing: f64,
//     pub initial_scale_factor: f64,
//     pub initial_phi: f64,
//     pub initial_d_phi: f64,
//     pub random_seed: u64,
// }

// pub struct FullSimulateParams<
//     T1: (Fn(f64) -> f64) + Send + Sync,
//     T2: (Fn(f64) -> f64) + Send + Sync,
// > {
//     pub params: SimulateParams,
//     pub potential: T1,
//     pub potential_d: T2,
// }

// pub struct Simulator<T1: (Fn(f64) -> f64) + Send + Sync, T2: (Fn(f64) -> f64) + Send + Sync> {
//     pub params: SimulateParams,
//     pub potential: T1,
//     pub potential_d: T2,
//     pub phi: Lattice<f64>,
//     conformal_time: f64,
//     time: f64,

//     mom_phi: Lattice<f64>,

//     scale_factor: f64,
//     mom_scale_factor: f64,

//     // used for computing slowroll parametre
//     last_mom_scale_factor: f64,

//     dft3d_plan: Dft3DPlan<f64>,
// }

// #[derive(Clone, Copy, Debug)]
// pub struct Measurables {
//     pub hubble_constraint: f64,
//     pub hamitonian: f64,
//     pub hubble: f64,
//     pub phi: f64,
//     pub phi_d: f64,
//     pub slowroll_epsilon: f64,
//     pub efolding: f64,
// }

// fn effective_mom(mode: Vec3i, dim: Dim) -> f64 {
//     let sx = sin(PI * (mode.x as f64) / (dim.x as f64));
//     let sy = sin(PI * (mode.y as f64) / (dim.y as f64));
//     let sz = sin(PI * (mode.z as f64) / (dim.z as f64));
//     sqrt(sx * sx + sy * sy + sz * sz) * 2.0
// }

// impl<T1: (Fn(f64) -> f64) + Send + Sync, T2: (Fn(f64) -> f64) + Send + Sync> Simulator<T1, T2> {
//     pub fn new(params: FullSimulateParams<T1, T2>) -> Self {
//         let dim = params.params.dim;
//         let mut ret = Simulator {
//             potential: params.potential,
//             potential_d: params.potential_d,
//             conformal_time: 0.0,
//             time: 0.0,
//             params: params.params,
//             phi: Lattice::new(dim),
//             mom_phi: Lattice::new(dim),

//             scale_factor: 1.0,
//             mom_scale_factor: 0.0,

//             last_mom_scale_factor: 0.0,

//             dft3d_plan: Dft3DPlan::new(dim, FftDirection::Forward),
//         };
//         ret.initialize(
//             params.params.initial_scale_factor,
//             params.params.initial_phi,
//             params.params.initial_d_phi,
//         );
//         ret
//     }
//     fn apply_k1(&mut self, delta_t: f64) {
//         let volumn = self.phi.dim().total_size() as f64;
//         if self.params.kappa != 0.0 {
//             self.scale_factor +=
//                 -self.mom_scale_factor * self.params.kappa / volumn / 6.0 * delta_t;
//         }
//     }
//     fn apply_k2(&mut self, delta_t: f64) {
//         let summed_mom_phi2 = self.mom_phi.map(|f| f * f).sum();
//         let a = self.scale_factor;
//         let a2 = a * a;
//         self.mom_scale_factor += summed_mom_phi2 / a / a2 * delta_t;
//         self.phi
//             .ref_mut()
//             .par_add_assign(self.mom_phi.mul_scalar(delta_t / a2));
//     }
//     fn apply_k3(&mut self, delta_t: f64) {
//         let dx = self.params.lattice_spacing;
//         let summed_d2_phi = self.phi.derivative_square().sum() / dx / dx;
//         let summed_potential = self.phi.map(&self.potential).sum();
//         let a = self.scale_factor;
//         let a2 = a * a;
//         let a4 = a2 * a2;
//         self.mom_scale_factor += (-summed_d2_phi * a - 4.0 * summed_potential * a2 * a) * delta_t;
//         let v_d = self.phi.map(&self.potential_d);
//         self.mom_phi.ref_mut().par_add_assign(
//             self.phi
//                 .laplacian()
//                 .mul_scalar(a2 / dx / dx)
//                 .add(v_d.mul_scalar(-a4))
//                 .mul_scalar(delta_t),
//         );
//     }
//     fn apply_full_k_order2(&mut self, delta_t: f64) {
//         self.apply_k1(delta_t / 2.0);
//         self.apply_k2(delta_t / 2.0);
//         self.apply_k3(delta_t);
//         self.apply_k2(delta_t / 2.0);
//         self.apply_k1(delta_t / 2.0);
//     }
//     fn apply_full_k_order4(&mut self, delta_t: f64) {
//         let beta = 0.7400789501051268; // 2 - 2 ^ (1/3)
//         self.apply_full_k_order2(delta_t / beta);
//         self.apply_full_k_order2(delta_t * (1.0 - 2.0 / beta));
//         self.apply_full_k_order2(delta_t / beta);
//     }
//     pub fn step(&mut self) {
//         let dt = self.params.time_step;
//         self.last_mom_scale_factor = self.mom_scale_factor;
//         self.apply_full_k_order4(dt);
//         self.conformal_time += dt;
//         self.time += dt * self.scale_factor;
//     }
//     pub fn initialize(&mut self, a: f64, phi: f64, d_phi: f64) {
//         let volumn = self.params.dim.total_size() as f64;
//         let dx = self.params.lattice_spacing;
//         self.phi.ref_mut().par_map_mut(|_, _| phi);
//         let mom_phi = d_phi * a * a;
//         self.mom_phi.ref_mut().par_map_mut(|_, _| mom_phi);
//         // TODO: initialize fluctuations
//         let averaged_mom_phi2 = self.mom_phi.map(|f| f * f).average();
//         let averaged_d2_phi = self.phi.derivative_square().average() / dx / dx;
//         let averaged_potential = self.phi.map(&self.potential).average();
//         let a2 = a * a;
//         let a4 = a2 * a2;
//         let a6 = a2 * a4;
//         self.mom_scale_factor = -volumn
//             * sqrt(
//                 6.0 * (averaged_mom_phi2 + averaged_d2_phi * a4 + 2.0 * averaged_potential * a6)
//                     / self.params.kappa,
//             )
//             / a;
//         self.populate_noise();
//         self.mom_scale_factor = -volumn
//             * sqrt(
//                 6.0 * (averaged_mom_phi2 + averaged_d2_phi * a4 + 2.0 * averaged_potential * a6)
//                     / self.params.kappa,
//             )
//             / a;
//         self.time = 0.0;
//         self.conformal_time = 0.0;
//     }
//     fn populate_noise(&mut self) {
//         let mut noise_phi = Lattice::<Complex64>::new(self.params.dim);
//         let mut noise_d_phi = Lattice::<Complex64>::new(self.params.dim);
//         let mut rand = random::default(self.params.random_seed);
//         let dx = self.params.lattice_spacing;
//         let inv_sqrt_volumn = 1.0 / sqrt((self.params.dim.total_size() as f64) * dx * dx * dx);
//         for index in 0..self.params.dim.total_size() {
//             let coord = self.params.dim.index_to_coord(index);
//             let index_coord_flipped = self
//                 .params
//                 .dim
//                 .coord_to_index(coord.flip_wrap_around(self.params.dim.try_into().unwrap()));
//             let rand_mag = sqrt(-2.0 * log(rand.read_f64()));
//             let rand_theta = 2.0 * PI * rand.read_f64();
//             let rand_a = Complex64::new(rand_mag * cos(rand_theta), rand_mag * sin(rand_theta));
//             let keff = effective_mom(coord, self.params.dim) / dx;
//             if index > 0 {
//                 let n_phi = rand_a / sqrt(2.0 * keff) * inv_sqrt_volumn;
//                 let n_phi_d = -Complex64::new(0.0, keff) * n_phi;
//                 noise_phi.ref_mut().set_by_index(index, n_phi);
//                 noise_d_phi.ref_mut().set_by_index(index, n_phi_d);
//                 if index_coord_flipped > index {
//                     noise_phi
//                         .ref_mut()
//                         .set_by_index(index_coord_flipped, n_phi.conj());
//                     noise_d_phi
//                         .ref_mut()
//                         .set_by_index(index_coord_flipped, n_phi_d.conj());
//                 }
//             } else {
//                 noise_phi.ref_mut().set_by_index(0, 0.0.into());
//                 noise_d_phi.ref_mut().set_by_index(0, 0.0.into());
//             }
//         }
//         self.dft3d_plan.transform(&mut noise_phi);
//         self.dft3d_plan.transform(&mut noise_d_phi);
//         println!(
//             "noise profile = {}, {}",
//             noise_phi.data()[1],
//             noise_d_phi.data()[1]
//         );
//         self.phi
//             .ref_mut()
//             .par_add_assign(noise_phi.map(|f| f.re / self.scale_factor).flip());
//         let scale_factor_d = self.scale_factor_d();
//         self.mom_phi.ref_mut().par_add_assign(
//             noise_d_phi
//                 .map(|f| f.re)
//                 .mul_scalar(self.scale_factor)
//                 .add(noise_phi.map(|f| f.re).mul_scalar(-scale_factor_d))
//                 .flip(),
//         );
//     }

//     pub fn measure(&self) -> Measurables {
//         let volumn = self.params.dim.total_size() as f64;
//         let dx = self.params.lattice_spacing;
//         let a = self.scale_factor;
//         let v_scale_factor = -self.mom_scale_factor * self.params.kappa / volumn / 6.0;
//         let comv_hubble = v_scale_factor / a;
//         let v_comv_hubble = -0.5 * comv_hubble * comv_hubble
//             + self.params.kappa * self.phi.derivative_square().average() / dx / dx / 12.0
//             - self.params.kappa * self.mom_phi.map(|phi| phi * phi).average() / 4.0 / a / a / a / a
//             + 0.5 * self.params.kappa * self.phi.map(&self.potential).average() * a * a;
//         let scale_factor_ham = if self.params.kappa != 0.0 {
//             -self.mom_scale_factor * self.mom_scale_factor / volumn * self.params.kappa / 12.0
//         } else {
//             0.0
//         };
//         let hamitonian = scale_factor_ham
//             + self.mom_phi.map(|phi| phi * phi).sum() / 2.0 / a / a
//             + self.phi.derivative_square().sum() * a * a / dx / dx / 2.0
//             + self.phi.map(&self.potential).sum() * a * a * a * a;
//         let hubble_constraint = self.phi.derivative_square().average() / dx / dx / 4.0
//             + self.mom_phi.map(|mp| mp * mp).average() / a / a / a / a / 4.0
//             - self.mom_scale_factor / volumn * self.mom_scale_factor / volumn * self.params.kappa
//                 / a
//                 / a
//                 / 24.0
//             + self.phi.map(&self.potential).average() * a * a / 2.0;
//         Measurables {
//             hubble_constraint,
//             hamitonian,
//             hubble: comv_hubble / a,
//             phi: (&self.phi).average(),
//             phi_d: self.mom_phi.map(|mom_phi| mom_phi / a / a / a).average(),
//             slowroll_epsilon: 1.0 - v_comv_hubble / comv_hubble / comv_hubble,
//             efolding: log(a),
//         }
//     }
//     pub fn scale_factor(&self) -> f64 {
//         self.scale_factor
//     }
//     pub fn efolding(&self) -> f64 {
//         log(self.scale_factor)
//     }
//     pub fn scale_factor_d(&self) -> f64 {
//         -self.mom_scale_factor * self.params.kappa / 6.0 / (self.params.dim.total_size() as f64)
//     }
// }

#[derive(Encode, Decode)]
pub struct ScalarFieldState<const D: usize> {
    pub phi: BoxLattice<D, f64>,
    pub mom_phi: BoxLattice<D, f64>,
    pub b: f64,
    pub mom_b: f64,
}

pub struct ScalarFieldParams<const D: usize, F> {
    pub kappa: f64,
    pub potential: F,
    pub lattice: LatticeParam<D>,
}

impl<const D: usize, F> ScalarFieldParams<D, F> {
    pub fn apply_k1(&self, field: &mut ScalarFieldState<D>, dt: f64) {
        field.b -= dt * field.mom_b;
    }
    pub fn apply_k2(&self, field: &mut ScalarFieldState<D>, dt: f64) {
        let d = D as f64;
        let b = field.b;
        field.mom_b += dt * 4.0 * (d - 1.0) / d / self.kappa * field.mom_phi.view().map(|f|f / b * f / b / b).average();
        field.phi.par_add_assign(&field.mom_phi.view().mul_scalar(dt * 4.0 * (d - 1.0) / d / b / b / self.kappa));
    }
    pub fn apply_k3(&self, field: &mut ScalarFieldState<D>, dt: f64) where
        F: C2Fn<f64, Output = f64> + Send + Sync,
    {
        let d = D as f64;
        let a = pow(2.0, 4.0 / d - 2.0) * pow((d - 1.0) / d, 2.0 / d - 1.0) * field.b.powf(2.0 - 4.0 / d) * self.kappa.powf(1.0 - 2.0 / d);
        field.mom_b -= d * field.b * self.kappa / 2.0 / (d - 1.0) * field.phi.view().map(|f|self.potential.value(f)).average();
        field.mom_phi.par_for_each_mut(|ptr, index, coord|{
            let dd = a * field.phi.laplacian_at(coord, &self.lattice.spacing) - d / 4.0 / (d - 1.0) * field.b * field.b * self.kappa * self.potential.value_d(field.phi.get(index, coord));
            *ptr += dd * dt;
        });
    }
    pub fn apply_full_k_order2(&self, field: &mut ScalarFieldState<D>, dt: f64) where
        F: C2Fn<f64, Output = f64> + Send + Sync,
    {
        self.apply_k1(field, dt / 2.0);
        self.apply_k2(field, dt / 2.0);
        self.apply_k3(field, dt);
        self.apply_k2(field, dt / 2.0);
        self.apply_k1(field, dt / 2.0);
    }
}
