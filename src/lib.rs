use libm::{log, sqrt};
use num_traits::{Num, NumAssign};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator};


#[derive(Copy, Clone)]
pub struct Dim {
    pub x: usize,
    pub y: usize,
    pub z: usize,
}

impl Dim {
    pub fn coord_to_index(self, coord: Vec3<i32>) -> usize {
        let x: usize = coord.x.try_into().unwrap();
        let y: usize = coord.y.try_into().unwrap();
        let z: usize = coord.z.try_into().unwrap();
        x + self.x * (y + self.y * z)
    }
    pub fn index_to_coord(self, mut index: usize) -> Vec3<i32> {
        let x = (index % self.x).try_into().unwrap();
        index /= self.x;
        let y = (index % self.y).try_into().unwrap();
        index /= self.y;
        let z = index.try_into().unwrap();
        Vec3 { x, y, z }
    }
    pub fn total_size(self) -> usize {
        self.x * self.y * self.z
    }
}

#[derive(Copy, Clone)]
pub struct Vec3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T: NumAssign + PartialOrd> Vec3<T> {
    pub fn shift_x_wrap(self, by: T, size: T) -> Self {
        let mut x = self.x + by;
        if x < T::zero() {
            x += size;
        } else if x >= size {
            x -= size;
        }
        Self {
            x,
            y: self.y,
            z: self.z,
        }
    }
    pub fn shift_y_wrap(self, by: T, size: T) -> Self {
        let mut y = self.y + by;
        if y < T::zero() {
            y += size;
        } else if y >= size {
            y -= size;
        }
        Self {
            x: self.x,
            y,
            z: self.z,
        }
    }
    pub fn shift_z_wrap(self, by: T, size: T) -> Self {
        let mut z = self.z + by;
        if z < T::zero() {
            z += size;
        } else if z >= size {
            z -= size;
        }
        Self {
            x: self.x,
            y: self.y,
            z,
        }
    }
    pub fn inner(self, other: Self) -> T {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
}

fn derivative<T: Num + Copy + Clone + From<i32>>(field: &[T], dim: Dim, coord: Vec3<i32>) -> Vec3<T> {
    let size_x = dim.x.try_into().unwrap();
    let size_y = dim.y.try_into().unwrap();
    let size_z = dim.z.try_into().unwrap();
    Vec3 {
        x: (field[dim.coord_to_index(coord.shift_x_wrap(1, size_x))] - field[dim.coord_to_index(coord.shift_x_wrap(-1, size_x))]) / 2.into(),
        y: (field[dim.coord_to_index(coord.shift_y_wrap(1, size_y))] - field[dim.coord_to_index(coord.shift_y_wrap(-1, size_y))]) / 2.into(),
        z: (field[dim.coord_to_index(coord.shift_z_wrap(1, size_z))] - field[dim.coord_to_index(coord.shift_z_wrap(-1, size_z))]) / 2.into(),
    }
}

fn laplacian<T: NumAssign + Copy + Clone + From<i32>>(field: &[T], dim: Dim, coord: Vec3<i32>) -> T {
    let size_x = dim.x.try_into().unwrap();
    let size_y = dim.y.try_into().unwrap();
    let size_z = dim.z.try_into().unwrap();
    let coords = [
        dim.coord_to_index(coord.shift_x_wrap(-1, size_x)),
        dim.coord_to_index(coord.shift_x_wrap(1, size_x)),
        dim.coord_to_index(coord.shift_y_wrap(-1, size_y)),
        dim.coord_to_index(coord.shift_y_wrap(1, size_y)),
        dim.coord_to_index(coord.shift_z_wrap(-1, size_z)),
        dim.coord_to_index(coord.shift_z_wrap(1, size_z)),
    ];
    let mut ret = field[dim.coord_to_index(coord)] * (-6).into();
    for c in coords {
        ret += field[c];
    }
    ret
}

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
    pub phi: Box<[f64]>,
    conformal_time: f64,
    time: f64,

    mom_phi: Box<[f64]>,

    scale_factor: f64,
    mom_scale_factor: f64,

    // used for computing slowroll parametre
    last_mom_scale_factor: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct Measurables {
    pub hubble: f64,
    pub phi: f64,
    pub phi_d: f64,
    pub slowroll_epsilon: f64,
    pub efolding: f64,
}

impl<T1: (Fn(f64) -> f64) + Send + Sync, T2: (Fn(f64) -> f64) + Send + Sync> Simulator<T1, T2> {
    pub fn new(params: FullSimulateParams<T1, T2>) -> Self {
        let len = params.params.dim.total_size();
        let mut ret = Simulator {
            potential: params.potential,
            potential_d: params.potential_d,
            conformal_time: 0.0,
            time: 0.0,
            params: params.params,
            phi: unsafe { Box::new_uninit_slice(len).assume_init() },
            mom_phi: unsafe { Box::new_uninit_slice(len).assume_init() },

            scale_factor: 1.0,
            mom_scale_factor: 0.0,

            last_mom_scale_factor: 0.0,
        };
        ret.initialize(params.params.initial_scale_factor, params.params.initial_phi, params.params.initial_d_phi);
        ret
    }
    fn apply_k1(&mut self, delta_t: f64) {
        self.scale_factor += -self.mom_scale_factor * delta_t / 6.0;
    }
    fn apply_k2(&mut self, delta_t: f64) {
        let volumn = self.params.dim.total_size() as f64;
        let averaged_mom_phi = self.mom_phi.par_iter().cloned().reduce(||0.0, |a, b|a + b) / volumn;
        let a = self.scale_factor;
        let a2 = a * a;
        let len = self.phi.len();
        self.mom_scale_factor += averaged_mom_phi * averaged_mom_phi / a / a2 * delta_t;
        self.phi.par_iter_mut().zip(0..len).for_each(|(phi, index)| {
            *phi += self.mom_phi[index] / a2 * delta_t;
        });
    }
    fn apply_k3(&mut self, delta_t: f64) {
        let len = self.phi.len();
        let volumn = self.params.dim.total_size() as f64;
        let averaged_d2_phi = (0..len).into_par_iter().map(|index| {
            let d = derivative(&self.phi, self.params.dim, self.params.dim.index_to_coord(index));
            d.inner(d)
        }).reduce(||0.0, |a, b|a + b) / volumn;
        let averaged_potential = self.phi.par_iter().cloned().map(|phi|(self.potential)(phi)).reduce(||0.0, |a, b|a + b) / volumn;
        let a = self.scale_factor;
        let a2 = a * a;
        let a4 = a2 * a2;
        self.mom_scale_factor += (-averaged_d2_phi * a - 4.0 * averaged_potential * a2 * a) * delta_t;
        self.mom_phi.par_iter_mut().zip(0..len).for_each(|(mom_phi, index)| {
            let lap = laplacian(&self.phi, self.params.dim, self.params.dim.index_to_coord(index));
            *mom_phi += (lap * a2 - a4 * (self.potential_d)(self.phi[index])) * delta_t;
        });
    }
    fn update_mom(&mut self) {
        let len = self.phi.len();
        let volumn = self.params.dim.total_size() as f64;
        let dim = self.params.dim;
        let dx = self.params.lattice_spacing;
        let dt = self.params.time_step;
        let a = self.scale_factor;
        let a3 = a * a * a;
        let a4 = a3 * a;
        let averaged_d2_phi = (0..len).into_par_iter().map(|index| {
            let d = derivative(&self.phi, self.params.dim, self.params.dim.index_to_coord(index));
            d.inner(d)
        }).reduce(||0.0, |a, b|a + b) / volumn;
        let averaged_potential = self.phi.par_iter().cloned().map(|phi|(self.potential)(phi)).reduce(||0.0, |a, b|a + b) / volumn;
        self.mom_phi.par_iter_mut().zip(0..len).for_each(|(mom_phi, index)| {
            let coord = dim.index_to_coord(index);
            let phi = self.phi[index];
            let a_phi = -laplacian(&self.phi, dim, coord) / dx / dx * a + a4 * (self.potential_d)(phi);
            *mom_phi -= a_phi * dt / 2.0;
        });
        let averaged_mom_phi = self.mom_phi.par_iter().cloned().reduce(||0.0, |a, b|a + b) / volumn;
        let a_scale_factor = -averaged_mom_phi * averaged_mom_phi / a3 + averaged_d2_phi * a + 4.0 * averaged_potential * a3;
        self.mom_scale_factor -= a_scale_factor * dt / 2.0;
    }
    pub fn step0(&mut self) {
        let len = self.phi.len();
        let dt = self.params.time_step;
        self.update_mom();
        let new_scale_factor = self.scale_factor - self.mom_scale_factor * self.params.kappa / 6.0 * dt;
        let middle_scale_factor = (self.scale_factor + new_scale_factor) / 2.0;
        self.scale_factor = new_scale_factor;
        self.phi.par_iter_mut().zip(0..len).for_each(|(phi, index)| {
            *phi += self.mom_phi[index] / middle_scale_factor / middle_scale_factor * dt;
        });
        self.update_mom();
    }
    pub fn step(&mut self) {
        let dt = self.params.time_step;
        self.last_mom_scale_factor = self.mom_scale_factor;
        self.apply_k1(dt / 2.0);
        self.apply_k2(dt / 2.0);
        self.apply_k3(dt);
        self.apply_k2(dt / 2.0);
        self.apply_k1(dt / 2.0);
        self.conformal_time += dt;
        self.time += dt * self.scale_factor;
    }
    pub fn initialize(&mut self, a: f64, phi: f64, d_phi: f64) {
        let len = self.phi.len();
        let volumn = self.params.dim.total_size() as f64;
        self.phi.par_iter_mut().for_each(|phi0|*phi0 = phi);
        let mom_phi = d_phi * a * a;
        self.mom_phi.par_iter_mut().for_each(|mom_phi0|*mom_phi0 = mom_phi);
        // TODO: initialize fluctuations
        let averaged_mom_phi2 = self.mom_phi.par_iter().cloned().map(|mom_phi|mom_phi * mom_phi).reduce(||0.0, |a, b|a + b) / volumn;
        let averaged_d2_phi = (0..len).into_par_iter().map(|index| {
            let d = derivative(&self.phi, self.params.dim, self.params.dim.index_to_coord(index));
            d.inner(d)
        }).reduce(||0.0, |a, b|a + b) / volumn;
        let averaged_potential = self.phi.par_iter().cloned().map(|phi|(self.potential)(phi)).reduce(||0.0, |a, b|a + b) / volumn;
        let a2 = a * a;
        let a4 = a2 * a2;
        let a6 = a2 * a4;
        self.mom_scale_factor = -sqrt(6.0 * (averaged_mom_phi2 + averaged_d2_phi * a4 + 2.0 * averaged_potential * a6) / self.params.kappa) / a;
        self.time = 0.0;
        self.conformal_time = 0.0;
    }
    pub fn measure(&self) -> Measurables {
        let volumn = self.params.dim.total_size() as f64;
        let a = self.scale_factor;
        let v_scale_factor = -self.mom_scale_factor * self.params.kappa / 6.0;
        let a_scale_factor = -self.params.kappa / 6.0 * (self.mom_scale_factor - self.last_mom_scale_factor) / self.params.time_step;
        let v_comv_hubble = (a_scale_factor * a - v_scale_factor * v_scale_factor) / a / a;
        let comv_hubble = v_scale_factor / a;
        Measurables {
            hubble: comv_hubble / a,
            phi: self.phi.par_iter().cloned().reduce(||0.0, |a, b|a + b) / volumn,
            phi_d: self.mom_phi.par_iter().cloned().map(|mom_phi|mom_phi / a / a).reduce(||0.0, |a, b|a + b) / volumn,
            slowroll_epsilon: 1.0 - v_comv_hubble / comv_hubble / comv_hubble,
            efolding: log(a),
        }
    }
    pub fn scale_factor(&self) -> f64 { self.scale_factor }
    pub fn efolding(&self) -> f64 { log(self.scale_factor) }
}
