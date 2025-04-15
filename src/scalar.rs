
use libm::{log, sqrt};

use crate::field::{Dim, Lattice, LatticeLike, LatticeMutLike};

#[derive(Clone, Copy)]
pub struct SimulateParams {
    pub time_step: f64,
    pub kappa: f64,
    pub dim: Dim,
    pub lattice_spacing: f64,
    pub initial_scale_factor: f64,
    pub initial_phi: f64,
    pub initial_d_phi: f64,
}

pub struct FullSimulateParams<T1: (Fn(f64) -> f64) + Send + Sync, T2: (Fn(f64) -> f64) + Send + Sync> {
    pub params: SimulateParams,
    pub potential: T1,
    pub potential_d: T2,
}

pub struct Simulator<T1: (Fn(f64) -> f64) + Send + Sync, T2: (Fn(f64) -> f64) + Send + Sync> {
    pub params: SimulateParams,
    pub potential: T1,
    pub potential_d: T2,
    pub phi: Lattice<f64>,
    conformal_time: f64,
    time: f64,

    mom_phi: Lattice<f64>,

    scale_factor: f64,
    mom_scale_factor: f64,

    // used for computing slowroll parametre
    last_mom_scale_factor: f64,
}

#[macro_export]
macro_rules! fdsl {
    ($input:tt) => {
        |index| {
            fdsl!(@inner index, $input)
        }
    };
    (@inner $index:ident, f!($expr:expr) $($more:tt)*) => {
        $expr[$index] fdsl!(@inner $index, $($more)*)
    };
    (@inner $index:ident, nabla!($expr:expr) $($more:tt)*) => {
        $expr[$index] fdsl!(@inner $index, $($more)*)
    };
    (@inner $index:ident, $other:tt $($more:tt)*) => {
        $other fdsl!(@inner $index, $($more)*)
    };
}

#[derive(Clone, Copy, Debug)]
pub struct Measurables {
    pub hubble_constraint: f64,
    pub hamitonian: f64,
    pub hubble: f64,
    pub phi: f64,
    pub phi_d: f64,
    pub slowroll_epsilon: f64,
    pub efolding: f64,
}

impl<T1: (Fn(f64) -> f64) + Send + Sync, T2: (Fn(f64) -> f64) + Send + Sync> Simulator<T1, T2> {
    pub fn new(params: FullSimulateParams<T1, T2>) -> Self {
        let dim = params.params.dim;
        let mut ret = Simulator {
            potential: params.potential,
            potential_d: params.potential_d,
            conformal_time: 0.0,
            time: 0.0,
            params: params.params,
            phi: Lattice::new(dim),
            mom_phi: Lattice::new(dim),

            scale_factor: 1.0,
            mom_scale_factor: 0.0,

            last_mom_scale_factor: 0.0,
        };
        ret.initialize(params.params.initial_scale_factor, params.params.initial_phi, params.params.initial_d_phi);
        ret
    }
    fn apply_k1(&mut self, delta_t: f64) {
        let volumn = self.phi.dim().total_size() as f64;
        if self.params.kappa != 0.0 {
            self.scale_factor += -self.mom_scale_factor * self.params.kappa / volumn / 6.0 * delta_t;
        }
    }
    fn apply_k2(&mut self, delta_t: f64) {
        // let summed_mom_phi2 = self.mom_phi.par_iter().cloned().map(|mp|mp * mp).sum::<f64>();
        let summed_mom_phi2 = self.mom_phi.map(|f|f * f).sum();
        let a = self.scale_factor;
        let a2 = a * a;
        self.mom_scale_factor += summed_mom_phi2 / a / a2 * delta_t;
        self.phi.ref_mut().par_add_assign(self.mom_phi.mul_scalar(delta_t / a2));
    }
    fn apply_k3(&mut self, delta_t: f64) {
        // let len = self.phi.len();
        let summed_d2_phi = self.phi.derivative_square().sum();
        let summed_potential = self.phi.map(&self.potential).sum();
        let a = self.scale_factor;
        let a2 = a * a;
        let a4 = a2 * a2;
        self.mom_scale_factor += (-summed_d2_phi * a - 4.0 * summed_potential * a2 * a) * delta_t;
        let v_d = self.phi.map(&self.potential_d);
        self.mom_phi.ref_mut().par_add_assign(
            self.phi.laplacian().mul_scalar(a2)
                .add(v_d.mul_scalar(-a4)).mul_scalar(delta_t)
        );
    }
    fn apply_full_k_order2(&mut self, delta_t: f64) {
        self.apply_k1(delta_t / 2.0);
        self.apply_k2(delta_t / 2.0);
        self.apply_k3(delta_t);
        self.apply_k2(delta_t / 2.0);
        self.apply_k1(delta_t / 2.0);
    }
    fn apply_full_k_order4(&mut self, delta_t: f64) {
        let beta = 0.7400789501051268; // 2 - 2 ^ (1/3)
        self.apply_full_k_order2(delta_t / beta);
        self.apply_full_k_order2(delta_t * (1.0 - 2.0 / beta));
        self.apply_full_k_order2(delta_t / beta);
    }
    pub fn step(&mut self) {
        let dt = self.params.time_step;
        self.last_mom_scale_factor = self.mom_scale_factor;
        self.apply_full_k_order4(dt);
        self.conformal_time += dt;
        self.time += dt * self.scale_factor;
    }
    pub fn initialize(&mut self, a: f64, phi: f64, d_phi: f64) {
        let volumn = self.params.dim.total_size() as f64;
        self.phi.ref_mut().par_map_mut(|_, _|phi);
        let mom_phi = d_phi * a * a;
        self.mom_phi.ref_mut().par_map_mut(|_, _|mom_phi);
        // TODO: initialize fluctuations
        let averaged_mom_phi2 = self.mom_phi.map(|f|f * f).average();
        let averaged_d2_phi = self.phi.derivative_square().average();
        let averaged_potential = self.phi.map(&self.potential).average();
        let a2 = a * a;
        let a4 = a2 * a2;
        let a6 = a2 * a4;
        self.mom_scale_factor = -volumn * sqrt(6.0 * (averaged_mom_phi2 + averaged_d2_phi * a4 + 2.0 * averaged_potential * a6) / self.params.kappa) / a;
        self.time = 0.0;
        self.conformal_time = 0.0;
    }

    pub fn measure(&self) -> Measurables {
        let volumn = self.params.dim.total_size() as f64;
        let a = self.scale_factor;
        let v_scale_factor = -self.mom_scale_factor * self.params.kappa / volumn / 6.0;
        let comv_hubble = v_scale_factor / a;
        let v_comv_hubble = -0.5 * comv_hubble * comv_hubble
            + self.params.kappa * self.phi.derivative_square().average() / 12.0
            - self.params.kappa * self.mom_phi.map(|phi|phi * phi).average() / 4.0 / a / a / a / a
            + 0.5 * self.params.kappa * self.phi.map(&self.potential).average() * a * a;
        let scale_factor_ham = if self.params.kappa != 0.0 { -self.mom_scale_factor * self.mom_scale_factor / volumn * self.params.kappa / 12.0 } else { 0.0 };
        let hamitonian = scale_factor_ham
            + self.mom_phi.map(|phi|phi * phi).sum() / 2.0 / a / a
            + self.phi.derivative_square().sum() * a * a / 2.0
            + self.phi.map(&self.potential).sum() * a * a * a * a;
        let hubble_constraint = self.phi.derivative_square().average() / 4.0
            + self.mom_phi.map(|mp|mp * mp).average() / a / a / a / a / 4.0
            - self.mom_scale_factor / volumn * self.mom_scale_factor / volumn * self.params.kappa / a / a / 24.0
            + self.phi.map(&self.potential).average() * a * a / 2.0;
        Measurables {
            hubble_constraint,
            hamitonian,
            hubble: comv_hubble / a,
            phi: (&self.phi).average(),
            phi_d: self.mom_phi.map(|mom_phi|mom_phi / a / a / a).average(),
            slowroll_epsilon: 1.0 - v_comv_hubble / comv_hubble / comv_hubble,
            efolding: log(a),
        }
    }
    pub fn scale_factor(&self) -> f64 { self.scale_factor }
    pub fn efolding(&self) -> f64 { log(self.scale_factor) }
}
