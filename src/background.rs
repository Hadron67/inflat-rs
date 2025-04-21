use std::{cmp::max, io::{self, Write}, iter::zip, ops::{Add, Mul, Range, Sub}, sync::{atomic::{AtomicUsize, Ordering}, Mutex}, time::SystemTime};

use bincode::{Decode, Encode};
use libm::{fmax, fmin, log, pow, sqrt};
use num_complex::{Complex64, ComplexFloat};
use rayon::{iter::{IntoParallelIterator, ParallelIterator}, vec};

pub struct BackgroundInput<F1, F2, F3> {
    pub kappa: f64,
    pub scale_factor: f64,
    pub phi: f64,
    pub phi_d: f64,
    pub potential: F1,
    pub potential_d: F2,
    pub potential_dd: F3,
}

pub struct BackgroundSimulator<F1, F2, F3> {
    pub kappa: f64,
    pub potential: F1,
    pub potential_d: F2,
    pub potential_dd: F3,
    pub scale_factor: f64,
    pub mom_scale_factor: f64,
    pub phi: f64,
    pub mom_phi: f64,
}

#[derive(Default)]
pub struct BackgroundMeasurables {
    pub phi: Vec<f64>,
    pub d_phi: Vec<f64>,
    pub scale_factor: Vec<f64>,
    pub hubble_constraint: Vec<f64>,
    pub hubble: Vec<f64>,
    pub epsilon: Vec<f64>,
}

#[derive(Default, Encode, Decode)]
pub struct BackgroundOutput {
    pub dt: Vec<f64>,
    pub effective_mass: Vec<f64>,
    pub scale_factor: Vec<f64>,
    pub d_scale_factor: Vec<f64>,
    pub phi: Vec<f64>,
    pub phi_d: Vec<f64>,
    pub hamitonian: Vec<f64>,
}

pub trait Natural {
    type Ascensor: Natural;
    const VALUE: usize;
    const IS_ZERO: bool;
}

pub struct Zero;
pub struct One;
pub struct Two;

macro_rules! impl_natural {
    ($ty:ident, $zero:expr, $value:expr, $ascensor:ident) => {
        impl Natural for $ty {
            type Ascensor = $ascensor;
            const IS_ZERO: bool = $zero;
            const VALUE: usize = $value;
        }
    };
}
impl_natural!(Zero, true, 0, Zero);
impl_natural!(One, false, 1, Zero);
impl_natural!(Two, false, 2, One);

impl<F1, F2, F3> BackgroundSimulator<F1, F2, F3> where
    F1: Fn(f64) -> f64,
    F2: Fn(f64) -> f64,
    F3: Fn(f64) -> f64,
{
    pub fn new(params: BackgroundInput<F1, F2, F3>) -> Self {
        let mut ret = Self {
            kappa: params.kappa,
            potential: params.potential,
            potential_d: params.potential_d,
            potential_dd: params.potential_dd,
            scale_factor: params.scale_factor,
            phi: params.phi,
            mom_phi: 0.0,
            mom_scale_factor: 0.0,
        };
        ret.initialize(params.phi, params.phi_d, params.scale_factor);
        ret
    }
    pub fn initialize(&mut self, phi: f64, d_phi: f64, scale_factor: f64) {
        let a2 = scale_factor * scale_factor;
        let a6 = a2 * a2 * a2;
        self.phi = phi;
        self.mom_phi = d_phi * scale_factor * scale_factor;
        self.scale_factor = scale_factor;
        self.mom_scale_factor = -sqrt((self.mom_phi * self.mom_phi + 2.0 * (self.potential)(self.phi) * a6) * 6.0 / self.kappa / a2);
    }
    fn apply_k1(&mut self, dt: f64) {
        self.scale_factor -= self.mom_scale_factor * self.kappa / 6.0 * dt;
    }
    fn apply_k2(&mut self, dt: f64) {
        self.mom_scale_factor += self.mom_phi * self.mom_phi / self.scale_factor / self.scale_factor / self.scale_factor * dt;
        self.phi += self.mom_phi / self.scale_factor / self.scale_factor * dt;
    }
    fn apply_k3(&mut self, dt: f64) {
        let a = self.scale_factor;
        self.mom_scale_factor += -4.0 * (self.potential)(self.phi) * a * a * a * dt;
        self.mom_phi += -a * a * a * a * (self.potential_d)(self.phi) * dt;
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
    fn apply_full_k_order_n(&mut self, dt: f64, n: usize) {
        if n == 4 {
            self.apply_full_k_order4(dt);
        } else {
            let beta = 2.0 - pow(2.0, 1.0 / ((n - 1) as f64));
            self.apply_full_k_order_n(dt / beta, n - 2);
            self.apply_full_k_order_n(dt * (1.0 - 2.0 / beta), n - 2);
            self.apply_full_k_order_n(dt / beta, n - 2);
        }
    }
    pub fn step(&mut self, delta_t: f64) {
        self.apply_full_k_order_n(delta_t, 8);
    }
    pub fn run_with_dn<P1>(&mut self, dn: f64, max_dt: f64, stop_predicate: P1) -> (BackgroundMeasurables, BackgroundOutput) where
        P1: FnMut(&Self) -> bool,
    {
        self.run(|s|{
            fmin(-dn * s.scale_factor / s.mom_scale_factor / s.kappa * 6.0, max_dt)
        }, stop_predicate)
    }
    pub fn run_with_da<P1>(&mut self, da: f64, min_dt: f64, stop_predicate: P1) -> (BackgroundMeasurables, BackgroundOutput) where
        P1: FnMut(&Self) -> bool,
    {
        self.run(|s|{
            fmin(-da / s.mom_scale_factor / s.kappa * 6.0, min_dt)
        }, stop_predicate)
    }
    pub fn effective_mass(&self) -> f64 {
        let pi_a = self.mom_scale_factor;
        let pi_phi = self.mom_phi;
        let a = self.scale_factor;
        let a2 = a * a;
        let a4 = a2 * a2;
        let v = (self.potential)(self.phi);
        let vd = (self.potential_d)(self.phi);
        let vdd = (self.potential_dd)(self.phi);
        pi_a * pi_a * self.kappa * self.kappa / a2 / 36.0
            + 5.0 * self.kappa * v * a2
            - 72.0 * v * v * a / pi_a * a / pi_a * a4
            - 12.0 * pi_phi * a * vd / pi_a
            + a2 * vdd
    }
    pub fn run<P1, P2>(&mut self, mut dt_provider: P1, mut stop_predicate: P2) -> (BackgroundMeasurables, BackgroundOutput) where
        P1: FnMut(&Self) -> f64,
        P2: FnMut(&Self) -> bool,
    {
        let mut measurables = BackgroundMeasurables {
            ..Default::default()
        };
        let mut output = BackgroundOutput {
            ..Default::default()
        };
        let mut steps = 1;
        let mut last_log_a = self.scale_factor;
        let mut last_log_time = SystemTime::now();
        while !stop_predicate(self) {
            let mass_eff = self.effective_mass();
            if self.scale_factor / last_log_a >= 1.01 {
                let a = self.scale_factor;
                let a2 = a * a;
                let a4 = a2 * a2;
                let a6 = a4 * a2;
                let d_a = -self.mom_scale_factor * self.kappa / 6.0;
                let hubble_contraint = self.mom_phi * self.mom_phi / 4.0 / a6
                    -self.mom_scale_factor * self.mom_scale_factor / 24.0 / a4
                    + 0.5 * (self.potential)(self.phi);
                let epsilon = 18.0 * self.mom_phi * self.mom_phi / self.mom_scale_factor / self.mom_scale_factor / self.kappa / a2;
                measurables.phi.push(self.phi);
                measurables.scale_factor.push(self.scale_factor);
                measurables.hubble_constraint.push(hubble_contraint);
                measurables.epsilon.push(epsilon);
                measurables.hubble.push(d_a / a2);
                measurables.d_phi.push(self.mom_phi / a2 / a);
                last_log_a = self.scale_factor;
                if last_log_time.elapsed().map(|f|f.as_millis() >= 100).unwrap_or(true) {
                    println!("step = {}, a = {}, hc = {}", steps, self.scale_factor, hubble_contraint);
                    last_log_time = SystemTime::now();
                }
            }
            let dt = dt_provider(self);
            output.effective_mass.push(mass_eff);
            output.dt.push(dt);
            output.scale_factor.push(self.scale_factor);
            output.d_scale_factor.push(-self.mom_scale_factor * self.kappa / 6.0);
            output.phi.push(self.phi);
            output.phi_d.push(self.mom_phi / self.scale_factor / self.scale_factor);
            self.step(dt);
            steps += 1;
        }
        (measurables, output)
    }
}

pub struct ScalarPerturbationSimulator<'a> {
    pub input: &'a BackgroundOutput,
    pub k: f64,
    pub u: Complex64,
    pub mom_u: Complex64,
    time_cursor: usize,
    local_time: f64,
    time_accumulator: f64,
}

#[derive(Default)]
pub struct ScalarPerturbationOutput {
    pub k: f64,
    pub scale_factor: Vec<f64>,
    pub dt: Vec<f64>,
    pub time: Vec<f64>,
    pub u: Vec<Complex64>,
    pub mom_u: Vec<Complex64>,
    pub potential: Vec<f64>,
    pub effective_mass: Vec<f64>,
    pub perturbation: Vec<f64>,
    pub z: Vec<f64>,
}

impl<'a> ScalarPerturbationSimulator<'a> {
    pub fn new(input: &'a BackgroundOutput, k: f64) -> Self {
        Self {
            input,
            k,
            u: Complex64::new(1.0 / sqrt(2.0 * k), 0.0),
            mom_u: Complex64::new(0.0, sqrt(k / 2.0)),
            local_time: 0.0,
            time_cursor: 0,
            time_accumulator: 0.0,
        }
    }
    pub fn scan_spectrum<F>(creator: F, k_range: Range<f64>, n_range: Range<f64>, count: usize, n: usize) -> Vec<(f64, f64)> where
        F: Fn() -> Self + Sync,
    {
        let done_count = AtomicUsize::new(0);
        (0..count).into_par_iter().map(|i|{
            let l = (i as f64) / ((count - 1) as f64);
            let k = pow(k_range.start, 1.0 - l) * pow(k_range.end, l);
            let mut sim = creator();
            let output = sim.run::<true>(n_range.start, Some(n_range.end), n);
            println!("[spectrum] ({}/{}) n = {}, k = {}, done", done_count.fetch_add(1, Ordering::Acquire) + 1, count, i, k);
            (k, *output.perturbation.last().unwrap())
        }).collect::<Vec<_>>()
    }
    fn advance_time(&mut self, dt: f64) {
        let mut new_time = self.local_time + dt;
        while new_time >= self.input.dt[self.time_cursor] {
            new_time -= self.input.dt[self.time_cursor];
            self.time_cursor += 1;
        }
        self.local_time = new_time;
        self.time_accumulator += dt;

    }
    fn effective_mass(&self) -> f64 {
        self.time_interpolate(&self.input.effective_mass)
    }
    fn scale_factor(&self) -> f64 {
        self.time_interpolate(&self.input.scale_factor)
    }
    fn apply_k1(&mut self, dt: f64) {
        self.mom_u += -(self.k * self.k + self.effective_mass()) * self.u.conj() * dt;
    }
    fn apply_k2(&mut self, dt: f64) {
        self.u += self.mom_u.conj() * dt;
    }
    fn apply_full_k_order2(&mut self, dt: f64) {
        self.apply_k1(dt / 2.0);
        self.apply_k2(dt);
        self.apply_k1(dt / 2.0);
    }
    fn apply_full_k_order_n(&mut self, dt: f64, n: usize) {
        if n == 2 {
            self.apply_full_k_order2(dt);
        } else {
            let beta = 2.0 - pow(2.0, 1.0 / ((n - 1) as f64));
            self.apply_full_k_order_n(dt / beta, n - 2);
            self.apply_full_k_order_n(dt * (1.0 - 2.0 / beta), n - 2);
            self.apply_full_k_order_n(dt / beta, n - 2);
        }
    }
    fn move_to_start_time(&mut self) {
        let k2 = self.k * self.k;
        let mut i = 0;
        while k2 / self.input.d_scale_factor[i] * self.input.scale_factor[i] >= 1e3 && i < self.input.effective_mass.len() {
            i += 1;
        }
        self.time_cursor = i;
        self.local_time = 0.0;
    }
    fn move_to_n(&mut self, n: f64) {
        self.time_cursor = self.input.scale_factor.iter().cloned().zip(0usize..).find(|(v, _)|{
            log(*v) >= n
        }).unwrap().1;
        self.local_time = 0.0;
    }
    fn time_interpolate<T>(&self, arr: &[T]) -> T where
        T: Add<T, Output = T> + Mul<f64, Output = T> + Sub<T, Output = T> + Copy,
    {
        let t = self.local_time / self.input.dt[self.time_cursor];
        assert!(t <= 1.0);
        let m1 = arr[self.time_cursor];
        let m2 = arr[self.time_cursor + 1];
        m1 + (m2 - m1) * t
    }
    pub fn run<const SILENCE: bool>(&mut self, start_n: f64, end_n: Option<f64>, n: usize) -> ScalarPerturbationOutput {
        let k2 = self.k * self.k;
        // self.move_to_start_time();
        self.move_to_n(start_n);
        if !SILENCE {
            println!("start time k^2 / (aH) = {}", k2 / self.input.d_scale_factor[self.time_cursor] * self.input.scale_factor[self.time_cursor]);
        }
        let mut output: ScalarPerturbationOutput = ScalarPerturbationOutput {
            k: self.k,
            ..Default::default()
        };
        let mut steps = 1;
        let mut last_display_time = SystemTime::now();
        let mut last_efolding = 0.0;
        while end_n.map(|e|log(self.scale_factor()) <= e).unwrap_or(true) && k2 / self.input.d_scale_factor[self.time_cursor] * self.input.scale_factor[self.time_cursor] >= 1e-3 {
            // let dt = fmin(du / (self.k * self.k + self.effective_mass()).abs().abs(), self.input.dt[self.time_cursor]);
            let dt = self.input.dt[self.time_cursor];
            let efolding = log(self.scale_factor());
            if efolding - last_efolding >= 0.00001 {
                last_efolding = efolding;
                output.u.push(self.u);
                output.scale_factor.push(self.scale_factor());
                output.dt.push(dt);
                output.time.push(self.time_accumulator);
                output.potential.push(k2 + self.effective_mass());
                output.effective_mass.push(self.effective_mass());
                output.mom_u.push(self.mom_u);
                let u = self.u.abs();
                let a = self.scale_factor();
                let a_d = self.time_interpolate(&self.input.d_scale_factor);
                let phi_d = self.time_interpolate(&self.input.phi_d);
                output.perturbation.push(u * a_d / a / a / phi_d);
                output.z.push(-a * a * phi_d / a_d);
                if !SILENCE && last_display_time.elapsed().map(|f|f.as_millis() >= 100).unwrap_or(true) {
                    println!("step = {}, |u| = {}, mass = {}, k^2 / mass = {}, N = {}", steps, self.u.abs(), self.effective_mass(), self.k * self.k / self.effective_mass(), log(self.scale_factor()));
                    last_display_time = SystemTime::now();
                }
            }
            self.apply_full_k_order_n(dt, n);
            self.advance_time(dt);
            steps += 1;
        }
        output
    }
}