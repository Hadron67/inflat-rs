use std::{
    f64::consts::PI,
    fs::File,
    io::{BufReader, BufWriter},
    marker::PhantomData,
    ops::{AddAssign, Div, Index, Mul},
    process::Output,
    sync::atomic::{AtomicUsize, Ordering},
    time::SystemTime,
};

use bincode::{
    Decode, Encode,
    config::{Configuration, standard},
    de, decode_from_std_read, encode_into_std_write,
};
use libm::{exp, fmin, pow, sqrt};
use num_complex::{Complex64, ComplexFloat};
use plotly::{
    Layout, Plot, Scatter,
    common::ExponentFormat,
    layout::{Axis, AxisType, GridPattern, LayoutGrid},
};
use rayon::{
    iter::{IntoParallelIterator, ParallelIterator},
};

use crate::{
    c2fn::{C2Fn, C2Fn2},
    util::{lazy_file, limit_length, linear_interp, power_interp},
};

#[derive(Debug, Clone, Copy)]
pub struct OutputSelector {
    pub dn: Option<f64>,
    pub start_n: Option<f64>,
    pub end_n: Option<f64>,
}

impl Default for OutputSelector {
    fn default() -> Self {
        Self {
            dn: Some(0.00001),
            start_n: Default::default(),
            end_n: Default::default(),
        }
    }
}

impl OutputSelector {
    pub fn test(&self, last_n: f64, n: f64) -> bool {
        self.dn.map(|dn| n >= last_n + dn).unwrap_or(true)
            && self.start_n.map(|f| n >= f).unwrap_or(true)
            && self.end_n.map(|f| n <= f).unwrap_or(true)
    }
}

pub struct BackgroundInput<F1, F2, F3> {
    pub kappa: f64,
    pub scale_factor: f64,
    pub phi: f64,
    pub phi_d: f64,
    pub potential: F1,
    pub potential_d: F2,
    pub potential_dd: F3,
}

pub trait TimeStateData {
    fn interpolate(&self, other: &Self, l: f64) -> Self
    where
        Self: Sized;
}

pub trait Dt {
    fn dt(&self) -> f64;
}

pub trait ScaleFactor {
    fn scale_factor(&self) -> f64;
}

pub trait ScaleFactorD {
    fn v_scale_factor(&self) -> f64;
}

#[derive(Clone, Copy, Encode, Decode, Debug)]
pub struct BackgroundState {
    pub phi: f64,
    pub mom_phi: f64,
    pub a: f64,
    pub mom_a: f64,
    pub dt: f64,
}

impl BackgroundState {
    pub fn init<P: C2Fn<f64>>(kappa: f64, phi: f64, phi_d: f64, a: f64, potential: &P) -> Self {
        let a2 = a * a;
        let a6 = a2 * a2 * a2;
        let mom_phi = phi_d * a * a;
        let mom_a = -sqrt((mom_phi * mom_phi + 2.0 * potential.value(phi) * a6) * 6.0 / kappa) / a;
        Self {
            phi,
            mom_phi,
            a,
            mom_a,
            dt: 0.0,
        }
    }
    pub fn init_slowroll<P: C2Fn<f64>>(kappa: f64, phi: f64, a: f64, potential: &P) -> Self {
        let phi_d = -potential.value_d(phi) / sqrt(3.0 * kappa * potential.value(phi)) * a;
        Self::init(kappa, phi, phi_d, a, potential)
    }
    pub fn v_phi(&self) -> f64 {
        self.mom_phi / self.a / self.a
    }
    pub fn dphi_dt(&self) -> f64 {
        self.v_phi() / self.a
    }
    pub fn v_a(&self, kappa: f64) -> f64 {
        -self.mom_a / 6.0 * kappa
    }
    pub fn vv_a<F: C2Fn<f64>>(&self, kappa: f64, potential: &F) -> f64 {
        -self.mom_a / self.a * self.mom_a * kappa * kappa / 36.0
            + kappa * potential.value(self.phi) * self.a * self.a * self.a
    }
    pub fn comoving_hubble(&self, kappa: f64) -> f64 {
        self.v_a(kappa) / self.a
    }
    pub fn spectrum_k_scale_mpc(&self, kappa: f64) -> f64 {
        0.05 / self.comoving_hubble(kappa)
    }
    pub fn spectrum_k_scale_hz(&self, kappa: f64) -> f64 {
        1.547e-15 * self.spectrum_k_scale_mpc(kappa)
    }
    pub fn z(&self, kappa: f64) -> f64 {
        self.a * self.a * self.v_phi() / self.v_a(kappa)
    }
    fn apply_k1(&mut self, dt: f64, kappa: f64) {
        self.a -= self.mom_a * kappa / 6.0 * dt;
    }
    fn apply_k2(&mut self, dt: f64) {
        self.mom_a += self.mom_phi * self.mom_phi / self.a / self.a / self.a * dt;
        self.phi += self.mom_phi / self.a / self.a * dt;
    }
    fn apply_k3<F: C2Fn<f64>>(&mut self, dt: f64, potential: &F) {
        let a = self.a;
        self.mom_a += -4.0 * potential.value(self.phi) * a * a * a * dt;
        self.mom_phi += -a * a * a * a * potential.value_d(self.phi) * dt;
    }
    fn apply_full_k_order2<F: C2Fn<f64>>(&mut self, delta_t: f64, kappa: f64, potential: &F) {
        self.apply_k1(delta_t / 2.0, kappa);
        self.apply_k2(delta_t / 2.0);
        self.apply_k3(delta_t, potential);
        self.apply_k2(delta_t / 2.0);
        self.apply_k1(delta_t / 2.0, kappa);
    }
    pub fn update<F: C2Fn<f64>>(&mut self, delta_t: f64, order: usize, kappa: f64, potential: &F) {
        if order == 2 {
            self.apply_full_k_order2(delta_t, kappa, potential);
        } else {
            let beta = 2.0 - pow(2.0, 1.0 / ((order - 1) as f64));
            self.update(delta_t / beta, order - 2, kappa, potential);
            self.update(delta_t * (1.0 - 2.0 / beta), order - 2, kappa, potential);
            self.update(delta_t / beta, order - 2, kappa, potential);
        }
    }
    pub fn scalar_effective_mass<F: C2Fn<f64>>(&self, kappa: f64, potential: &F) -> f64 {
        let pi_a = self.mom_a;
        let pi_phi = self.mom_phi;
        let a = self.a;
        let a2 = a * a;
        let a4 = a2 * a2;
        let a6 = a2 * a4;
        let v = potential.value(self.phi);
        let vd = potential.value_d(self.phi);
        let vdd = potential.value_dd(self.phi);
        pi_a * pi_a * kappa * kappa / a2 / 36.0 + 5.0 * kappa * v * a2
            - 72.0 * v * v * a6 / pi_a / pi_a
            - 12.0 * pi_phi * a * vd / pi_a
            + a2 * vdd
    }
    pub fn hubble_constraint<F: C2Fn<f64>>(&self, potential: &F) -> f64 {
        let a = self.a;
        let a2 = a * a;
        let a4 = a2 * a2;
        let a6 = a2 * a4;
        self.mom_phi * self.mom_phi / 4.0 / a6 - self.mom_a * self.mom_a / 24.0 / a4
            + 0.5 * potential.value(self.phi)
    }
    pub fn epsilon(&self, kappa: f64) -> f64 {
        let a2 = self.a * self.a;
        18.0 * self.mom_phi * self.mom_phi / self.mom_a / self.mom_a / kappa / a2
    }
    pub fn delta<F: C2Fn<f64>>(&self, kappa: f64, potential: &F) -> f64 {
        let a = self.a;
        let a2 = a * a;
        let a4 = a2 * a2;
        let a5 = a4 * a;
        3.0 - 6.0 * a5 * potential.value_d(self.phi) / self.mom_phi / self.mom_a / kappa
    }
    pub fn simulate<F, Cond, Step>(
        &self,
        kappa: f64,
        potential: &F,
        dn: f64,
        min_dt: f64,
        order: usize,
        mut stop_condition: Cond,
        mut step_monitor: Step,
    ) -> Vec<Self>
    where
        F: C2Fn<f64>,
        Cond: FnMut(&Self) -> bool,
        Step: FnMut(&Self),
    {
        let mut state = *self;
        let mut ret = vec![state];
        while !stop_condition(&state) {
            state.dt = fmin(-dn * state.a / state.mom_a / kappa * 6.0, min_dt);
            state.update(state.dt, order, kappa, potential);
            step_monitor(&state);
            ret.push(state);
        }
        ret
    }
}

impl TimeStateData for BackgroundState {
    fn interpolate(&self, other: &Self, l: f64) -> Self {
        Self {
            phi: linear_interp(self.phi, other.phi, l),
            mom_phi: linear_interp(self.mom_phi, other.mom_phi, l),
            a: linear_interp(self.a, other.a, l),
            mom_a: linear_interp(self.mom_a, other.mom_a, l),
            dt: self.dt,
        }
    }
}

impl Dt for BackgroundState {
    fn dt(&self) -> f64 {
        self.dt
    }
}

#[derive(Debug, Clone, Copy, Encode, Decode)]
pub struct TwoFieldBackgroundState {
    pub a: f64,
    pub mom_a: f64,
    pub phi: f64,
    pub mom_phi: f64,
    pub chi: f64,
    pub mom_chi: f64,
    pub dt: f64,
}

impl Dt for TwoFieldBackgroundState {
    fn dt(&self) -> f64 {
        self.dt
    }
}

macro_rules! interpolate_fields {
    ($ty:ident, $str1:expr, $str2:expr, $l: expr, $($field:ident),*) => {
        $ty {
            $($field: linear_interp($str1.$field, $str2.$field, $l)),*,
            dt: $str1.dt * (1.0 - $l),
        }
    };
}

pub struct TwoFieldBackgroundInput<F1, F2> {
    pub kappa: f64,
    pub b: F1,
    pub v: F2,
}

impl TwoFieldBackgroundState {
    pub fn init<F1, F2>(
        a: f64,
        phi: f64,
        v_phi: f64,
        chi: f64,
        v_chi: f64,
        input: &TwoFieldBackgroundInput<F1, F2>,
    ) -> Self
    where
        F1: C2Fn<f64>,
        F2: C2Fn2<f64, f64, Ret = f64>,
    {
        let a2 = a * a;
        let a4 = a2 * a2;
        let a6 = a2 * a4;
        let mom_phi = v_phi * a * a;
        let mom_chi = v_chi * a * a * exp(2.0 * input.b.value(phi));
        let mom_a = -sqrt(
            (exp(-2.0 * input.b.value(phi)) * mom_chi * mom_chi
                + mom_phi * mom_phi
                + 2.0 * input.v.value_00(phi, chi) * a6)
                * 6.0
                / input.kappa,
        ) / a;
        Self {
            a,
            mom_a,
            phi,
            mom_phi,
            chi,
            mom_chi,
            dt: 0.0,
        }
    }
    pub fn init_slowroll<F1, F2>(
        a: f64,
        phi: f64,
        chi: f64,
        v_chi: f64,
        input: &TwoFieldBackgroundInput<F1, F2>,
    ) -> Self
    where
        F1: C2Fn<f64>,
        F2: C2Fn2<f64, f64, Ret = f64>,
    {
        let a2 = a * a;
        let a4 = a2 * a2;
        let a6 = a4 * a2;
        let mom_chi = v_chi * a * a * exp(2.0 * input.b.value(phi));
        let mom_a = -sqrt(
            (exp(-2.0 * input.b.value(phi)) * mom_chi * mom_chi
                + 2.0 * input.v.value_00(phi, chi) * a6)
                * 6.0
                / input.kappa,
        ) / a;
        let mom_phi = 2.0 * a4 * a * input.v.value_10(phi, chi) / mom_a / input.kappa
            - 2.0 * exp(-2.0 * input.b.value(phi)) * mom_chi * mom_chi * input.b.value_d(phi)
                / mom_a
                / input.kappa
                / a;
        Self {
            a,
            mom_a,
            phi,
            mom_phi,
            chi,
            mom_chi,
            dt: 0.0,
        }
    }
    fn apply_k1<F1, F2>(&mut self, dt: f64, input: &TwoFieldBackgroundInput<F1, F2>)
    where
        F1: C2Fn<f64>,
        F2: C2Fn2<f64, f64, Ret = f64>,
    {
        self.a += -dt * self.mom_a * input.kappa / 6.0;
    }
    fn apply_k2<F1, F2>(&mut self, dt: f64, input: &TwoFieldBackgroundInput<F1, F2>)
    where
        F1: C2Fn<f64>,
        F2: C2Fn2<f64, f64, Ret = f64>,
    {
        let a2 = self.a * self.a;
        let a3 = self.a * a2;
        self.mom_phi += dt
            * exp(-2.0 * input.b.value(self.phi))
            * self.mom_chi
            * self.mom_chi
            * input.b.value_d(self.phi)
            / a2;
        self.chi += dt * exp(-2.0 * input.b.value(self.phi)) * self.mom_chi / a2;
        self.mom_a += dt * exp(-2.0 * input.b.value(self.phi)) * self.mom_chi * self.mom_chi / a3;
    }
    fn apply_k3(&mut self, dt: f64) {
        self.phi += dt * self.mom_phi / self.a / self.a;
        self.mom_a += dt * self.mom_phi * self.mom_phi / self.a / self.a / self.a;
    }
    fn apply_k4<F1, F2>(&mut self, dt: f64, input: &TwoFieldBackgroundInput<F1, F2>)
    where
        F1: C2Fn<f64>,
        F2: C2Fn2<f64, f64, Ret = f64>,
    {
        let a2 = self.a * self.a;
        let a4 = a2 * a2;
        self.mom_phi += -dt * a4 * input.v.value_10(self.phi, self.chi);
        self.mom_chi += -dt * a4 * input.v.value_01(self.phi, self.chi);
        self.mom_a += -dt * 4.0 * a2 * self.a * input.v.value_00(self.phi, self.chi);
    }
    pub fn apply_full_k_order2<F1, F2>(&mut self, dt: f64, input: &TwoFieldBackgroundInput<F1, F2>)
    where
        F1: C2Fn<f64>,
        F2: C2Fn2<f64, f64, Ret = f64>,
    {
        self.apply_k1(dt / 2.0, input);
        self.apply_k2(dt / 2.0, input);
        self.apply_k3(dt / 2.0);
        self.apply_k4(dt, input);
        self.apply_k3(dt / 2.0);
        self.apply_k2(dt / 2.0, input);
        self.apply_k1(dt / 2.0, input);
    }
    pub fn epsilon<F1, F2>(&self, input: &TwoFieldBackgroundInput<F1, F2>) -> f64
    where
        F1: C2Fn<f64>,
        F2: C2Fn2<f64, f64, Ret = f64>,
    {
        3.0 - 36.0 * input.v.value_00(self.phi, self.chi) * self.a * self.a * self.a * self.a
            / input.kappa
            / self.mom_a
            / self.mom_a
    }
    pub fn hubble_constraint<F1, F2>(&self, input: &TwoFieldBackgroundInput<F1, F2>) -> f64
    where
        F1: C2Fn<f64>,
        F2: C2Fn2<f64, f64, Ret = f64>,
    {
        let a2 = self.a * self.a;
        let a4 = a2 * a2;
        let a6 = a4 * a2;
        0.5 * input.v.value_00(self.phi, self.chi)
            + exp(-2.0 * input.b.value(self.phi)) * self.mom_chi * self.mom_chi / 4.0 / a6
            + self.mom_phi * self.mom_phi / 4.0 / a6
            - self.mom_a * self.mom_a * input.kappa / 24.0 / a4
    }
    #[rustfmt::skip]
    pub fn intermediate_potential<F1, F2>(&self, input: &TwoFieldBackgroundInput<F1, F2>, k: f64, alpha: f64, lambda: f64) -> f64 where
        F1: C2Fn<f64>,
        F2: C2Fn2<f64, f64, Ret = f64>,
    {
        5.0 / 36.0 * exp(-2.0 * input.b.value(self.phi)) * self.mom_chi * k * self.mom_a * self.mom_a * alpha * input.kappa * input.kappa * input.kappa * lambda * 1.0 / self.a / self.a / self.a / self.a / self.a / self.a + exp(-2.0 * input.b.value(self.phi)) * self.mom_chi * k * k * k * alpha * input.kappa * lambda * 1.0 / self.a / self.a / self.a / self.a + -1.0 * exp(-2.0 * input.b.value(self.phi)) * self.mom_chi * k * alpha * input.kappa * input.kappa * lambda * input.v.value_00(self.phi,self.chi) * 1.0 / self.a / self.a + -1.0 / 3.0 * exp(-2.0 * input.b.value(self.phi)) * self.mom_chi * k * self.mom_phi * self.mom_a * alpha * input.kappa * input.kappa * lambda * 1.0 / self.a / self.a / self.a / self.a / self.a / self.a / self.a * input.b.value_d(self.phi) + -1.0 / 6.0 * exp(-2.0 * input.b.value(self.phi)) * k * self.mom_a * alpha * input.kappa * input.kappa * lambda * 1.0 / self.a * input.v.value_01(self.phi,self.chi)
    }
    #[rustfmt::skip]
    pub fn normalized_potential<F1, F2>(&self, input: &TwoFieldBackgroundInput<F1, F2>, k: f64, alpha: f64, lambda: f64) -> f64 where
        F1: C2Fn<f64>,
        F2: C2Fn2<f64, f64, Ret = f64>,
    {
        let num = 1.0 / 3.0 * exp(-4.0 * input.b.value(self.phi)) * self.mom_chi * self.mom_chi * k * k * self.mom_a * self.mom_a * alpha * alpha * input.kappa * input.kappa * input.kappa * input.kappa * lambda * lambda * 1.0 / self.a / self.a / self.a / self.a / self.a / self.a / self.a / self.a / self.a / self.a + 2.0 / 3.0 * exp(-2.0 * input.b.value(self.phi)) * self.mom_chi * k * self.mom_a * self.mom_a * alpha * input.kappa * input.kappa * input.kappa * lambda * 1.0 / self.a / self.a / self.a / self.a / self.a / self.a + -4.0 * exp(-4.0 * input.b.value(self.phi)) * self.mom_chi * self.mom_chi * k * k * alpha * alpha * input.kappa * input.kappa * input.kappa * lambda * lambda * input.v.value_00(self.phi,self.chi) * 1.0 / self.a / self.a / self.a / self.a / self.a / self.a + -8.0 / 3.0 * exp(-4.0 * input.b.value(self.phi)) * self.mom_chi * self.mom_chi * k * k * self.mom_phi * self.mom_a * alpha * alpha * input.kappa * input.kappa * input.kappa * lambda * lambda * 1.0 / self.a / self.a / self.a / self.a / self.a / self.a / self.a / self.a / self.a / self.a / self.a * input.b.value_d(self.phi) + -16.0 / 3.0 * exp(-2.0 * input.b.value(self.phi)) * self.mom_chi * k * self.mom_phi * self.mom_a * alpha * input.kappa * input.kappa * lambda * 1.0 / self.a / self.a / self.a / self.a / self.a / self.a / self.a * input.b.value_d(self.phi) + 4.0 * exp(-6.0 * input.b.value(self.phi)) * self.mom_chi * self.mom_chi * self.mom_chi * self.mom_chi * k * k * alpha * alpha * input.kappa * input.kappa * lambda * lambda * 1.0 / self.a / self.a / self.a / self.a / self.a / self.a / self.a / self.a / self.a / self.a / self.a / self.a * input.b.value_d(self.phi) * input.b.value_d(self.phi) + 12.0 * exp(-4.0 * input.b.value(self.phi)) * self.mom_chi * self.mom_chi * k * k * self.mom_phi * self.mom_phi * alpha * alpha * input.kappa * input.kappa * lambda * lambda * 1.0 / self.a / self.a / self.a / self.a / self.a / self.a / self.a / self.a / self.a / self.a / self.a / self.a * input.b.value_d(self.phi) * input.b.value_d(self.phi) + -4.0 / 3.0 * exp(-4.0 * input.b.value(self.phi)) * self.mom_chi * self.mom_chi * k * k * self.mom_a * self.mom_a * alpha * alpha * input.kappa * input.kappa * input.kappa * lambda * lambda * 1.0 / self.a / self.a / self.a / self.a / self.a / self.a / self.a / self.a / self.a / self.a * input.b.value_d(self.phi) * input.b.value_d(self.phi) + 4.0 * exp(-4.0 * input.b.value(self.phi)) * self.mom_chi * self.mom_chi * self.mom_chi * k * alpha * input.kappa * lambda * 1.0 / self.a / self.a / self.a / self.a / self.a / self.a / self.a / self.a * input.b.value_d(self.phi) * input.b.value_d(self.phi) + 16.0 * exp(-2.0 * input.b.value(self.phi)) * self.mom_chi * k * self.mom_phi * self.mom_phi * alpha * input.kappa * lambda * 1.0 / self.a / self.a / self.a / self.a / self.a / self.a / self.a / self.a * input.b.value_d(self.phi) * input.b.value_d(self.phi) + -4.0 / 3.0 * exp(-2.0 * input.b.value(self.phi)) * self.mom_chi * k * self.mom_a * self.mom_a * alpha * input.kappa * input.kappa * lambda * 1.0 / self.a / self.a / self.a / self.a / self.a / self.a * input.b.value_d(self.phi) * input.b.value_d(self.phi) + 16.0 * exp(-4.0 * input.b.value(self.phi)) * self.mom_chi * self.mom_chi * k * k * alpha * alpha * input.kappa * input.kappa * lambda * lambda * input.v.value_00(self.phi,self.chi) * 1.0 / self.a / self.a / self.a / self.a / self.a / self.a * input.b.value_d(self.phi) * input.b.value_d(self.phi) + 16.0 * exp(-2.0 * input.b.value(self.phi)) * self.mom_chi * k * alpha * input.kappa * lambda * input.v.value_00(self.phi,self.chi) * 1.0 / self.a / self.a * input.b.value_d(self.phi) * input.b.value_d(self.phi) + 4.0 * exp(-6.0 * input.b.value(self.phi)) * self.mom_chi * self.mom_chi * self.mom_chi * self.mom_chi * k * k * alpha * alpha * input.kappa * input.kappa * lambda * lambda * 1.0 / self.a / self.a / self.a / self.a / self.a / self.a / self.a / self.a / self.a / self.a / self.a / self.a * input.b.value_dd(self.phi) + -2.0 / 3.0 * exp(-4.0 * input.b.value(self.phi)) * self.mom_chi * self.mom_chi * k * k * self.mom_a * self.mom_a * alpha * alpha * input.kappa * input.kappa * input.kappa * lambda * lambda * 1.0 / self.a / self.a / self.a / self.a / self.a / self.a / self.a / self.a / self.a / self.a * input.b.value_dd(self.phi) + 4.0 * exp(-4.0 * input.b.value(self.phi)) * self.mom_chi * self.mom_chi * self.mom_chi * k * alpha * input.kappa * lambda * 1.0 / self.a / self.a / self.a / self.a / self.a / self.a / self.a / self.a * input.b.value_dd(self.phi) + -2.0 / 3.0 * exp(-2.0 * input.b.value(self.phi)) * self.mom_chi * k * self.mom_a * self.mom_a * alpha * input.kappa * input.kappa * lambda * 1.0 / self.a / self.a / self.a / self.a / self.a / self.a * input.b.value_dd(self.phi) + 8.0 * exp(-4.0 * input.b.value(self.phi)) * self.mom_chi * self.mom_chi * k * k * alpha * alpha * input.kappa * input.kappa * lambda * lambda * input.v.value_00(self.phi,self.chi) * 1.0 / self.a / self.a / self.a / self.a / self.a / self.a * input.b.value_dd(self.phi) + 8.0 * exp(-2.0 * input.b.value(self.phi)) * self.mom_chi * k * alpha * input.kappa * lambda * input.v.value_00(self.phi,self.chi) * 1.0 / self.a / self.a * input.b.value_dd(self.phi) + 2.0 / 3.0 * exp(-4.0 * input.b.value(self.phi)) * self.mom_chi * k * k * self.mom_a * alpha * alpha * input.kappa * input.kappa * input.kappa * lambda * lambda * 1.0 / self.a / self.a / self.a / self.a / self.a * input.v.value_01(self.phi,self.chi) + -2.0 / 3.0 * exp(-2.0 * input.b.value(self.phi)) * k * self.mom_a * alpha * input.kappa * input.kappa * lambda * 1.0 / self.a * input.v.value_01(self.phi,self.chi) + 4.0 * exp(-4.0 * input.b.value(self.phi)) * self.mom_chi * k * k * self.mom_phi * alpha * alpha * input.kappa * input.kappa * lambda * lambda * 1.0 / self.a / self.a / self.a / self.a / self.a / self.a * input.b.value_d(self.phi) * input.v.value_01(self.phi,self.chi) + 8.0 * exp(-2.0 * input.b.value(self.phi)) * k * self.mom_phi * alpha * input.kappa * lambda * 1.0 / self.a / self.a * input.b.value_d(self.phi) * input.v.value_01(self.phi,self.chi) + -1.0 * exp(-4.0 * input.b.value(self.phi)) * k * k * alpha * alpha * input.kappa * input.kappa * lambda * lambda * input.v.value_01(self.phi,self.chi) * input.v.value_01(self.phi,self.chi) + -2.0 * exp(-6.0 * input.b.value(self.phi)) * self.mom_chi * self.mom_chi * k * k * alpha * alpha * input.kappa * input.kappa * lambda * lambda * 1.0 / self.a / self.a / self.a / self.a / self.a / self.a * input.v.value_02(self.phi,self.chi) + -2.0 * exp(-4.0 * input.b.value(self.phi)) * self.mom_chi * k * alpha * input.kappa * lambda * 1.0 / self.a / self.a * input.v.value_02(self.phi,self.chi) + 4.0 * exp(-4.0 * input.b.value(self.phi)) * self.mom_chi * self.mom_chi * k * k * alpha * alpha * input.kappa * input.kappa * lambda * lambda * 1.0 / self.a / self.a / self.a / self.a / self.a / self.a * input.b.value_d(self.phi) * input.v.value_10(self.phi,self.chi) + 4.0 * exp(-2.0 * input.b.value(self.phi)) * self.mom_chi * k * alpha * input.kappa * lambda * 1.0 / self.a / self.a * input.b.value_d(self.phi) * input.v.value_10(self.phi,self.chi) + -2.0 * exp(-4.0 * input.b.value(self.phi)) * self.mom_chi * k * k * self.mom_phi * alpha * alpha * input.kappa * input.kappa * lambda * lambda * 1.0 / self.a / self.a / self.a / self.a / self.a / self.a * input.v.value_11(self.phi,self.chi) + -2.0 * exp(-2.0 * input.b.value(self.phi)) * k * self.mom_phi * alpha * input.kappa * lambda * 1.0 / self.a / self.a * input.v.value_11(self.phi,self.chi);
        let den = 2.0 + 2.0 * exp(-2.0 * input.b.value(self.phi)) * self.mom_chi * k * alpha * input.kappa * lambda * 1.0 / self.a / self.a / self.a / self.a;
        num / den / den
    }
    pub fn normalized_horizon_potential<F1, F2>(&self, input: &TwoFieldBackgroundInput<F1, F2>, k: f64, alpha: f64, lambda: f64) -> f64 where
        F1: C2Fn<f64>,
        F2: C2Fn2<f64, f64, Ret = f64>,
    {
        let num = 1.0 / 9.0 * self.mom_a * self.mom_a * input.kappa * input.kappa * self.a * self.a + -4.0 * input.kappa * input.v.value_00(self.phi,self.chi) * self.a * self.a * self.a * self.a * self.a * self.a;
        let den = 2.0 * exp(-2.0 * input.b.value(self.phi)) * self.mom_chi * k * alpha * input.kappa * lambda * 1.0 / self.a / self.a + self.a * self.a;
        num / den / den
    }
    pub fn vv_chi<F1, F2>(&self, input: &TwoFieldBackgroundInput<F1, F2>) -> f64 where
        F1: C2Fn<f64>,
        F2: C2Fn2<f64, f64, Ret = f64>,
    {
        1.0 / 3.0 * exp(-2.0 * input.b.value(self.phi)) * self.mom_chi * self.mom_a * input.kappa * 1.0 / self.a / self.a / self.a + -2.0 * exp(-2.0 * input.b.value(self.phi)) * self.mom_chi * self.mom_phi * 1.0 / self.a / self.a / self.a / self.a * input.b.value_d(self.phi) + -1.0 * exp(-2.0 * input.b.value(self.phi)) * self.a * self.a * input.v.value_01(self.phi,self.chi)
    }
    #[rustfmt::skip]
    pub fn vvv_chi<F1, F2>(&self, input: &TwoFieldBackgroundInput<F1, F2>) -> f64 where
        F1: C2Fn<f64>,
        F2: C2Fn2<f64, f64, Ret = f64>,
    {
        -2.0 * exp(-2.0 * input.b.value(self.phi)) * self.mom_chi * input.kappa * input.v.value_00(self.phi,self.chi) + 2.0 / 9.0 * exp(-2.0 * input.b.value(self.phi)) * self.mom_chi * self.mom_a * self.mom_a * input.kappa * input.kappa * 1.0 / self.a / self.a / self.a / self.a + -2.0 * exp(-2.0 * input.b.value(self.phi)) * self.mom_chi * self.mom_phi * self.mom_a * input.kappa * 1.0 / self.a / self.a / self.a / self.a / self.a * input.b.value_d(self.phi) + 8.0 * exp(-2.0 * input.b.value(self.phi)) * self.mom_chi * input.v.value_00(self.phi,self.chi) * input.b.value_d(self.phi) * input.b.value_d(self.phi) + 2.0 * exp(-4.0 * input.b.value(self.phi)) * self.mom_chi * self.mom_chi * self.mom_chi * 1.0 / self.a / self.a / self.a / self.a / self.a / self.a * input.b.value_d(self.phi) * input.b.value_d(self.phi) + 8.0 * exp(-2.0 * input.b.value(self.phi)) * self.mom_chi * self.mom_phi * self.mom_phi * 1.0 / self.a / self.a / self.a / self.a / self.a / self.a * input.b.value_d(self.phi) * input.b.value_d(self.phi) + -2.0 / 3.0 * exp(-2.0 * input.b.value(self.phi)) * self.mom_chi * self.mom_a * self.mom_a * input.kappa * 1.0 / self.a / self.a / self.a / self.a * input.b.value_d(self.phi) * input.b.value_d(self.phi) + 4.0 * exp(-2.0 * input.b.value(self.phi)) * self.mom_chi * input.v.value_00(self.phi,self.chi) * input.b.value_dd(self.phi) + 2.0 * exp(-4.0 * input.b.value(self.phi)) * self.mom_chi * self.mom_chi * self.mom_chi * 1.0 / self.a / self.a / self.a / self.a / self.a / self.a * input.b.value_dd(self.phi) + -1.0 / 3.0 * exp(-2.0 * input.b.value(self.phi)) * self.mom_chi * self.mom_a * self.mom_a * input.kappa * 1.0 / self.a / self.a / self.a / self.a * input.b.value_dd(self.phi) + 4.0 * exp(-2.0 * input.b.value(self.phi)) * self.mom_phi * input.b.value_d(self.phi) * input.v.value_01(self.phi,self.chi) + -1.0 * exp(-4.0 * input.b.value(self.phi)) * self.mom_chi * input.v.value_02(self.phi,self.chi) + 2.0 * exp(-2.0 * input.b.value(self.phi)) * self.mom_chi * input.b.value_d(self.phi) * input.v.value_10(self.phi,self.chi) + -1.0 * exp(-2.0 * input.b.value(self.phi)) * self.mom_phi * input.v.value_11(self.phi,self.chi)
    }
    pub fn v_chi<F1, F2>(&self, input: &TwoFieldBackgroundInput<F1, F2>) -> f64
    where
        F1: C2Fn<f64>,
        F2: C2Fn2<f64, f64, Ret = f64>,
    {
        exp(-2.0 * input.b.value(self.phi)) * self.mom_chi / self.a / self.a
    }
    pub fn vv_a<F1, F2>(&self, input: &TwoFieldBackgroundInput<F1, F2>) -> f64
    where
        F1: C2Fn<f64>,
        F2: C2Fn2<f64, f64, Ret = f64>,
    {
        -1.0 / 36.0 * self.mom_a * self.mom_a * input.kappa * input.kappa * 1.0 / self.a
            + input.kappa * input.v.value_00(self.phi, self.chi) * self.a * self.a * self.a
    }
    pub fn update<F1, F2>(&mut self, dt: f64, input: &TwoFieldBackgroundInput<F1, F2>)
    where
        F1: C2Fn<f64>,
        F2: C2Fn2<f64, f64, Ret = f64>,
    {
        self.apply_full_k_order2(dt, input);
    }
    pub fn simulate<F1, F2, Cond, Step>(
        &self,
        input: &TwoFieldBackgroundInput<F1, F2>,
        dn: f64,
        min_dt: f64,
        mut stop_condition: Cond,
        mut step_monitor: Step,
    ) -> Vec<Self>
    where
        F1: C2Fn<f64>,
        F2: C2Fn2<f64, f64, Ret = f64>,
        Cond: FnMut(&Self) -> bool,
        Step: FnMut(&Self),
    {
        let mut state = *self;
        let mut ret = vec![state];
        while !stop_condition(&state) {
            state.dt = fmin(-dn * state.a / state.mom_a / input.kappa * 6.0, min_dt);
            state.update(state.dt, input);
            step_monitor(&state);
            ret.push(state);
        }
        ret
    }
}

impl TimeStateData for TwoFieldBackgroundState {
    fn interpolate(&self, other: &Self, l: f64) -> Self {
        interpolate_fields!(Self, self, other, l, a, mom_a, phi, mom_phi, chi, mom_chi)
    }
}

/// For simulating Hamitonian of the form 1/(2m) |p|^2 + 1/2 k |x|^2
#[derive(Debug, Clone, Copy)]
pub struct HamitonianState<T> {
    pub x: T,
    pub mom: T,
}

impl HamitonianState<Complex64> {
    pub fn init_bd_vacuum(mass: f64, k: f64) -> Self {
        Self {
            x: Complex64::new(1.0 / sqrt(2.0 * k), 0.0),
            mom: Complex64::new(0.0, mass * sqrt(k / 2.0)),
        }
    }
}

impl<T> HamitonianState<T>
where
    T: ComplexFloat,
{
    fn apply_k1(&mut self, dt: f64, mass: T)
    where
        T: AddAssign<T> + Mul<f64, Output = T>,
    {
        self.x += self.mom.conj() / mass * dt;
    }
    fn apply_k2(&mut self, dt: f64, k: T)
    where
        T: AddAssign<T> + Mul<f64, Output = T>,
    {
        self.mom += -self.x.conj() * k * dt;
    }
    fn apply_full_k_order2(&mut self, dt: f64, mass: T, k: T)
    where
        T: AddAssign<T> + Mul<f64, Output = T>,
    {
        self.apply_k1(dt / 2.0, mass);
        self.apply_k2(dt, k);
        self.apply_k1(dt / 2.0, mass);
    }
}

pub trait SequentialAccess<Idx, T> {
    fn advance(&mut self, by: Idx);
}

pub struct LinearInterpolator {
    cursor: usize,
    local_time: f64,
}

impl LinearInterpolator {
    pub fn advance<I>(&mut self, states: &I, by: f64)
    where
        I: Index<usize> + ?Sized,
        I::Output: Dt + Sized,
    {
        self.local_time += by;
        while self.local_time >= states[self.cursor].dt() {
            self.local_time -= states[self.cursor].dt();
            self.cursor += 1;
        }
    }
    pub fn get<I>(&self, states: &I) -> I::Output
    where
        I: Index<usize> + ?Sized,
        I::Output: TimeStateData + Dt + Sized,
    {
        let s = &states[self.cursor];
        s.interpolate(s, self.local_time / s.dt())
    }
}

pub trait BackgroundFn<Context: ?Sized, Background: ?Sized> {
    fn apply(&self, context: &Context, state: &Background, k: f64) -> f64;
}

pub struct HamitonianSimulator<'a, 'c, Ctx: ?Sized, I: ?Sized, B, Mass, SubHorizon, SuperHorizon> {
    context: &'c Ctx,
    length: usize,
    background_state: &'a I,
    _background_elem: PhantomData<&'a [B]>,
    time_interpolator: LinearInterpolator,
    state: HamitonianState<Complex64>,
    mass: Mass,
    subhorizon_potential: SubHorizon,
    superhorizon_potential: SuperHorizon,
}

impl<'a, 'c, Ctx, I, B, Mass, SubHorizon, SuperHorizon>
    HamitonianSimulator<'a, 'c, Ctx, I, B, Mass, SubHorizon, SuperHorizon>
where
    I: Index<usize, Output = B> + ?Sized,
    Ctx: ?Sized,
{
    pub fn new(
        context: &'c Ctx,
        length: usize,
        background_state: &'a I,
        mass: Mass,
        subhorizon_potential: SubHorizon,
        superhorizon_potential: SuperHorizon,
    ) -> Self {
        Self {
            context,
            length,
            background_state,
            _background_elem: PhantomData,
            time_interpolator: LinearInterpolator {
                cursor: 0,
                local_time: 0.0,
            },
            state: HamitonianState {
                x: 0.0.into(),
                mom: 0.0.into(),
            },
            mass,
            subhorizon_potential,
            superhorizon_potential,
        }
    }
}

impl<'a, 'c, Ctx, I, B, Mass, SubHorizon, SuperHorizon>
    HamitonianSimulator<'a, 'c, Ctx, I, B, Mass, SubHorizon, SuperHorizon>
where
    Ctx: ?Sized,
    I: Index<usize, Output = B> + ?Sized,
    B: TimeStateData + Dt,
    Mass: BackgroundFn<Ctx, B>,
    SubHorizon: BackgroundFn<Ctx, B>,
    SuperHorizon: BackgroundFn<Ctx, B>,
{
    fn horizon_ratio(&self, index: usize, k: f64) -> f64 {
        let state = &self.background_state[index];
        k * k
            / (self.subhorizon_potential.apply(self.context, &state, k)
                + self.superhorizon_potential.apply(self.context, &state, k))
            .abs()
    }
    fn first_horizon_ratio_index(&self, k: f64, tolerance: f64) -> usize {
        let mut index = 0usize;
        while index < self.length && self.horizon_ratio(index, k) > tolerance {
            index += 1;
        }
        index
    }
    pub fn run<F>(&mut self, k: f64, da: f64, tolerance: f64, mut consumer: F)
    where
        F: FnMut(&mut Self, &B, &HamitonianState<Complex64>, f64, f64),
    {
        let start_index = self.first_horizon_ratio_index(k,tolerance);
        let end_index = self.first_horizon_ratio_index(k, 1.0 / tolerance);
        println!("start i = {}, end i = {}", start_index, end_index);
        self.time_interpolator = LinearInterpolator {
            cursor: start_index,
            local_time: 0.0,
        };
        self.state = HamitonianState::init_bd_vacuum(
            self.mass.apply(self.context, &self.background_state[0], k),
            k,
        );
        while self.time_interpolator.cursor < end_index {
            let background_state = self.time_interpolator.get(self.background_state);
            let mass = self.mass.apply(self.context, &background_state, k);
            let potential = k * k
                + self
                    .subhorizon_potential
                    .apply(self.context, &background_state, k)
                + self
                    .superhorizon_potential
                    .apply(self.context, &background_state, k);
            let dt = fmin(background_state.dt(), da / potential.abs().sqrt());
            let mut state = self.state;
            state.apply_full_k_order2(dt, mass.into(), potential.into());
            self.state = state;
            consumer(self, &background_state, &state, potential, dt);
            self.time_interpolator.advance(self.background_state, dt);
        }
    }
    // pub fn spectrum(&mut self, k_range: (f64, f64), count: usize, da: f64) {
    //     let done_count = AtomicUsize::new(0);
    //     (0..count).into_par_iter().map(|i|{
    //         let k = power_interp(k_range.0, k_range.1, ((i - 1) as f64) / ((count - 1) as f64));

    //     })
    // }
}

pub trait PerturbationParams {
    fn constant_term(&self, background: &BackgroundState, k: f64) -> f64;
    fn intermediate_term(&self, background: &BackgroundState, k: f64) -> f64;
    fn horizon_term(&self, background: &BackgroundState, k: f64) -> f64;
    fn perturbation(&self, u: Complex64, background: &BackgroundState) -> Complex64;
    fn potential(&self, background: &BackgroundState, k: f64) -> f64 {
        self.constant_term(background, k)
            + self.intermediate_term(background, k)
            + self.horizon_term(background, k)
    }
}

pub struct ScalarPerturbation2<'a, F> {
    pub kappa: f64,
    pub potential: &'a F,
}

impl<'a, F: C2Fn<f64>> PerturbationParams for ScalarPerturbation2<'a, F> {
    fn perturbation(&self, u: Complex64, background: &BackgroundState) -> Complex64 {
        -u / background.z(self.kappa)
    }

    fn constant_term(&self, _background: &BackgroundState, k: f64) -> f64 {
        k * k
    }

    fn intermediate_term(&self, _background: &BackgroundState, _k: f64) -> f64 {
        0.0
    }

    fn horizon_term(&self, background: &BackgroundState, k: f64) -> f64 {
        background.scalar_effective_mass(self.kappa, self.potential)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PerturbationState {
    pub u: Complex64,
    pub mom_u: Complex64,
}

impl PerturbationState {
    pub fn init<P: PerturbationParams>(k: f64, background: &BackgroundState, params: &P) -> Self {
        let omega = params.constant_term(background, k).sqrt();
        Self {
            u: Complex64::new(1.0 / sqrt(2.0 * omega), 0.0),
            mom_u: Complex64::new(0.0, sqrt(omega / 2.0)),
        }
    }
    pub fn apply_k1<P: PerturbationParams>(
        &mut self,
        dt: f64,
        k: f64,
        background: &BackgroundState,
        params: &P,
    ) {
        // self.mom_u += self.u.conj() * (-(self.simulator.k * self.simulator.k + self.simulator.effective_mass()) * dt);
        let v = params.potential(background, k);
        self.mom_u += self.u.conj() * (-v * dt);
    }
    pub fn apply_k2(&mut self, dt: f64) {
        self.u += self.mom_u.conj() * dt;
    }
    pub fn apply_full_k_order2<P: PerturbationParams>(
        &mut self,
        dt: f64,
        k: f64,
        background: &BackgroundState,
        params: &P,
    ) {
        self.apply_k1(dt / 2.0, k, background, params);
        self.apply_k2(dt);
        self.apply_k1(dt / 2.0, k, background, params);
    }
    pub fn apply_full_k_order_n<P: PerturbationParams>(
        &mut self,
        dt: f64,
        order: usize,
        k: f64,
        background: &BackgroundState,
        params: &P,
    ) {
        if order == 1 {
            self.apply_k1(dt, k, background, params);
            self.apply_k2(dt);
        } else if order == 2 {
            self.apply_full_k_order2(dt, k, background, params);
        } else {
            let beta = 2.0 - pow(2.0, 1.0 / ((order - 1) as f64));
            self.apply_full_k_order_n(dt / beta, order - 2, k, background, params);
            self.apply_full_k_order_n(dt * (1.0 - 2.0 / beta), order - 2, k, background, params);
            self.apply_full_k_order_n(dt / beta, order - 2, k, background, params);
        }
    }
}

pub struct PerturbationSimulator<'a, 'b, P> {
    pub k: f64,
    pub background: &'a [BackgroundState],
    param: &'b P,
    cursor: usize,
    local_time: f64,
}

impl<'a, 'b, P> PerturbationSimulator<'a, 'b, P>
where
    P: PerturbationParams,
{
    pub fn new(k: f64, background: &'a [BackgroundState], param: &'b P) -> Self {
        Self {
            k,
            background,
            param,
            cursor: 0,
            local_time: 0.0,
        }
    }
    pub fn advance(&mut self, dt: f64) {
        self.local_time += dt;
        while self.local_time >= self.background[self.cursor].dt {
            self.local_time -= self.background[self.cursor].dt;
            self.cursor += 1;
        }
    }
    pub fn get_background(&self) -> BackgroundState {
        let b1 = &self.background[self.cursor];
        let b2 = &self.background[self.cursor + 1];
        b1.interpolate(b2, self.local_time / b1.dt)
    }
    fn horizon_ratio(&self, index: usize) -> f64 {
        let b = &self.background[index];
        (self.param.horizon_term(b, self.k)
            / (self.param.constant_term(b, self.k) + self.param.intermediate_term(b, self.k)))
        .abs()
    }
    fn horizon_partial_ratio(&self, index: usize) -> f64 {
        let b = &self.background[index];
        (self.param.horizon_term(b, self.k) / self.param.constant_term(b, self.k)).abs()
    }
    fn constant_term_ratio(&self, index: usize) -> f64 {
        let b = &self.background[index];
        (self.param.constant_term(b, self.k)
            / (self.param.intermediate_term(b, self.k) + self.param.horizon_term(b, self.k)))
        .abs()
    }
    pub fn get_start_index(&self, max_n: Option<f64>, horizon_tolerance: f64) -> usize {
        let mut ret = 0usize;
        while ret < self.background.len()
            && max_n
                .map(|n| self.background[ret].a.ln() < n)
                .unwrap_or(true)
            && self.constant_term_ratio(ret) > horizon_tolerance
        {
            ret += 1;
        }
        ret
    }
    pub fn get_end_index(&self, min_n: Option<f64>, horizon_tolerance: f64) -> usize {
        let mut ret = 0usize;
        while ret < self.background.len()
            && (min_n
                .map(|n| self.background[ret].a.ln() < n)
                .unwrap_or(false)
                || self.horizon_ratio(ret) < horizon_tolerance
                || self.horizon_partial_ratio(ret) < horizon_tolerance)
        {
            ret += 1;
        }
        ret
    }
    pub fn get_next_dt(&self, du: f64, background: &BackgroundState) -> f64 {
        let k = self.k;
        let potential = self.param.potential(background, k);
        fmin(du / potential.abs().sqrt(), background.dt)
    }
    pub fn run<F>(
        &mut self,
        n_range: (Option<f64>, Option<f64>),
        du: f64,
        order: usize,
        mut output_consumer: F,
    ) -> PerturbationState
    where
        F: FnMut(&BackgroundState, &PerturbationState),
    {
        let k = self.k;
        self.cursor = self.get_start_index(n_range.0, 1e4);
        let end_i = self.get_end_index(n_range.1, 1e4);
        self.local_time = 0.0;
        let mut pert_state = PerturbationState::init(k, &self.background[self.cursor], self.param);
        while self.cursor <= end_i {
            let background = self.get_background();
            let dt = self.get_next_dt(du, &background);
            pert_state.apply_full_k_order_n(dt, order, k, &background, self.param);
            output_consumer(&background, &pert_state);
            self.advance(dt);
        }
        pert_state
    }
}

#[derive(Clone, Copy)]
pub struct SpectrumSetting {
    pub k_range: (f64, f64),
    pub n_range: (Option<f64>, Option<f64>),
    pub count: usize,
}

pub fn scan_spectrum<P>(
    background: &[BackgroundState],
    pert_param: &P,
    setting: &SpectrumSetting,
    k_scale: f64,
) -> Vec<(f64, f64)>
where
    P: PerturbationParams + Send + Sync,
{
    let done = AtomicUsize::new(0);
    (0..setting.count)
        .into_par_iter()
        .map(|i| {
            let k = power_interp(
                setting.k_range.0,
                setting.k_range.1,
                (i as f64) / ((setting.count - 1) as f64),
            );
            let mut sim = PerturbationSimulator::new(k, background, pert_param);
            let pert_state = sim.run(setting.n_range, 0.01, 2, |_, _| {});
            let done0 = done.fetch_add(1, Ordering::SeqCst) + 1;
            println!("[spectrum] ({}/{}) k = {}", done0, setting.count, k);
            let r = pert_param
                .perturbation(pert_state.u, &sim.get_background())
                .abs();
            (k / k_scale, r * r * k * k * k / 2.0 / PI / PI)
        })
        .collect::<Vec<_>>()
}

pub struct InputData<'name, 'pot, F, P> {
    pub name: &'name str,
    pub kappa: f64,
    pub phi0: f64,
    pub a0: f64,
    pub potential: &'pot F,
    pub pert_param: P,
}

pub const BINCODE_CONFIG: Configuration = standard();

pub struct Context<'name, 'pot, 'dir, 'input, F, P> {
    input_data: &'input InputData<'name, 'pot, F, P>,
    plot_max_length: usize,
    out_dir: &'dir str,
    background_data: Option<Vec<BackgroundState>>,
    order: usize,
}

impl<'name, 'pot, 'dir, 'input, F, P> Context<'name, 'pot, 'dir, 'input, F, P>
where
    F: C2Fn<f64>,
    P: PerturbationParams,
{
    pub fn new(
        out_dir: &'dir str,
        plot_max_length: usize,
        order: usize,
        input_data: &'input InputData<'name, 'pot, F, P>,
    ) -> Self {
        Self {
            input_data,
            plot_max_length,
            out_dir,
            background_data: None,
            order,
        }
    }
    fn background_data_file_name(&self) -> String {
        self.out_dir.to_owned() + "/" + self.input_data.name + ".background.bincode"
    }
    pub fn load_input_data(&mut self) {
        if self.background_data.is_none() {
            self.background_data = Some({
                decode_from_std_read(
                    &mut BufReader::new(File::open(self.background_data_file_name()).unwrap()),
                    BINCODE_CONFIG,
                )
                .unwrap()
            });
        }
    }
    pub fn get_background(&self) -> &[BackgroundState] {
        self.background_data.as_ref().unwrap()
    }
    pub fn run_background(&mut self, dn: f64, min_dt: f64) {
        let kappa = self.input_data.kappa;
        let initial = BackgroundState::init_slowroll(
            kappa,
            self.input_data.phi0,
            self.input_data.a0,
            self.input_data.potential,
        );
        let mut last_log_time = SystemTime::now();
        let result = initial.simulate(
            kappa,
            self.input_data.potential,
            dn,
            min_dt,
            self.order,
            |s| s.epsilon(kappa) >= 1.0,
            |s| {
                let now = SystemTime::now();
                if last_log_time
                    .elapsed()
                    .map(|s| s.as_millis() > 100)
                    .unwrap_or(false)
                {
                    last_log_time = now;
                    println!("{:?}", s);
                }
            },
        );
        {
            encode_into_std_write(
                &result,
                &mut BufWriter::new(File::create(self.background_data_file_name()).unwrap()),
                BINCODE_CONFIG,
            )
            .unwrap();
        }
        let result = limit_length(result, self.plot_max_length);
        {
            let mut plot = Plot::new();
            let mut efoldings = vec![];
            let mut phi = vec![];
            let mut v_phi = vec![];
            let mut dphi_dt = vec![];
            let mut epsilon = vec![];
            let mut delta = vec![];
            let mut hubble_constraint = vec![];
            let mut effective_mass = vec![];
            let mut tensor_effective_mass = vec![];

            for elem in result {
                efoldings.push(elem.a.ln());
                phi.push(elem.phi);
                v_phi.push(elem.v_phi().abs());
                dphi_dt.push((elem.v_phi() / elem.a).abs());
                epsilon.push(elem.epsilon(kappa));
                delta.push(elem.delta(kappa, self.input_data.potential));
                hubble_constraint.push(elem.hubble_constraint(self.input_data.potential));
                effective_mass.push(-elem.scalar_effective_mass(kappa, self.input_data.potential));
                tensor_effective_mass.push(elem.vv_a(kappa, self.input_data.potential) / elem.a);
            }
            plot.add_trace(Scatter::new(efoldings.clone(), phi).name("phi"));
            plot.add_trace(
                Scatter::new(efoldings.clone(), epsilon)
                    .name("epsilon")
                    .x_axis("x1")
                    .y_axis("y2"),
            );
            plot.add_trace(
                Scatter::new(efoldings.clone(), delta)
                    .name("delta")
                    .x_axis("x1")
                    .y_axis("y3"),
            );
            plot.add_trace(
                Scatter::new(efoldings.clone(), dphi_dt)
                    .name("|dphi/dt|")
                    .x_axis("x1")
                    .y_axis("y4"),
            );
            plot.add_trace(
                Scatter::new(efoldings.clone(), v_phi)
                    .name("|v_phi|")
                    .x_axis("x1")
                    .y_axis("y5"),
            );
            plot.add_trace(
                Scatter::new(efoldings.clone(), effective_mass)
                    .name("m^2")
                    .x_axis("x1")
                    .y_axis("y6"),
            );
            plot.add_trace(
                Scatter::new(efoldings.clone(), tensor_effective_mass)
                    .name("tensor m^2")
                    .x_axis("x1")
                    .y_axis("y6"),
            );
            plot.add_trace(
                Scatter::new(efoldings.clone(), hubble_constraint)
                    .name("hubble constraint")
                    .x_axis("x1")
                    .y_axis("y7"),
            );
            plot.set_layout(
                Layout::new()
                    .grid(
                        LayoutGrid::new()
                            .rows(7)
                            .columns(1)
                            .pattern(GridPattern::Coupled),
                    )
                    .y_axis(Axis::new().exponent_format(ExponentFormat::Power))
                    .y_axis2(
                        Axis::new()
                            .type_(AxisType::Log)
                            .exponent_format(ExponentFormat::Power),
                    )
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
                    .y_axis7(
                        Axis::new()
                            .type_(AxisType::Log)
                            .exponent_format(ExponentFormat::Power),
                    )
                    .height(1400),
            );
            plot.write_html(
                self.out_dir.to_owned() + "/" + self.input_data.name + ".background.html",
            );
        }
    }
    pub fn run_perturbation(&mut self, k: f64, n_range: (Option<f64>, Option<f64>)) {
        self.load_input_data();
        let background = self.background_data.as_ref().unwrap();
        let mut sim = PerturbationSimulator::new(k, background, &self.input_data.pert_param);
        let output_selector = OutputSelector::default();
        let mut efolding = vec![];
        let mut re_u = vec![];
        let mut im_u = vec![];
        let mut u = vec![];
        let mut r = vec![];

        let mut last_n: Option<f64> = None;
        let mut last_log_time = SystemTime::now();
        sim.run(n_range, 0.01, self.order, |b, u1| {
            let n = b.a.ln();
            if last_n.map(|la| output_selector.test(la, n)).unwrap_or(true) {
                last_n = Some(n);
                efolding.push(n);
                re_u.push(u1.u.re);
                im_u.push(u1.u.im);
                u.push(u1.u.abs());
                r.push(self.input_data.pert_param.perturbation(u1.u, b).abs());
            }
            if last_log_time.elapsed().unwrap().as_millis() > 100 {
                last_log_time = SystemTime::now();
                println!("N = {}, pert = {:?}", n, u1);
            }
        });
        {
            let mut plot = Plot::new();
            plot.add_trace(
                Scatter::new(efolding.clone(), re_u)
                    .name("re u")
                    .x_axis("x1")
                    .y_axis("y1"),
            );
            plot.add_trace(
                Scatter::new(efolding.clone(), im_u)
                    .name("im u")
                    .x_axis("x1")
                    .y_axis("y1"),
            );
            plot.add_trace(
                Scatter::new(efolding.clone(), u)
                    .name("u")
                    .x_axis("x1")
                    .y_axis("y2"),
            );
            plot.add_trace(
                Scatter::new(efolding.clone(), r)
                    .name("r")
                    .x_axis("x1")
                    .y_axis("y3"),
            );
            plot.set_layout(
                Layout::new()
                    .grid(
                        LayoutGrid::new()
                            .rows(3)
                            .columns(1)
                            .pattern(GridPattern::Coupled),
                    )
                    .y_axis2(
                        Axis::new()
                            .type_(AxisType::Log)
                            .exponent_format(ExponentFormat::Power),
                    )
                    .y_axis3(
                        Axis::new()
                            .type_(AxisType::Log)
                            .exponent_format(ExponentFormat::Power),
                    )
                    .height(800),
            );
            plot.write_html(
                self.out_dir.to_owned() + "/" + self.input_data.name + ".perturbation.scalar.html",
            );
        }
    }
    pub fn run_spectrum(&mut self, label: &str, setting: &SpectrumSetting) -> Vec<(f64, f64)>
    where
        P: Send + Sync,
        F: Send + Sync,
    {
        let out_file = format!(
            "{}/{}.spectrum.{}.bincode",
            self.out_dir, self.input_data.name, label
        );
        lazy_file(&out_file, BINCODE_CONFIG, || {
            self.load_input_data();
            let background = self.background_data.as_ref().unwrap();
            scan_spectrum(background, &self.input_data.pert_param, setting, 1.0)
        })
        .unwrap()
    }
    pub fn plot_spectrum(&mut self, spectrum: &[(f64, f64)], label: &str) {
        let mut plot = Plot::new();
        plot.add_trace(Scatter::new(
            spectrum.iter().map(|f| f.0).collect(),
            spectrum.iter().map(|f| f.1).collect(),
        ));
        plot.set_layout(
            Layout::new()
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
                .height(800),
        );
        plot.write_html(
            self.out_dir.to_owned() + "/" + self.input_data.name + ".spectrum." + label + ".html",
        );
    }
    pub fn spectrum_k_scale_hz(&self) -> f64 {
        self.spectrum_k_scale_mpc() * 1.547e-15
    }
    pub fn spectrum_k_scale_mpc(&self) -> f64 {
        0.05 / self.get_background()[0].comoving_hubble(self.input_data.kappa)
    }
}
