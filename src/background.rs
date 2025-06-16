use std::{
    cmp::{max, min}, f64::consts::PI, marker::PhantomData, ops::{AddAssign, Index, Mul}, sync::atomic::{AtomicUsize, Ordering}
};

use bincode::{
    Decode, Encode,
    config::{Configuration, standard},
};
use libm::{exp, fmin, log, pow, sqrt};
use num_complex::{Complex64, ComplexFloat};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{
    c2fn::{C2Fn, C2Fn2},
    util::{self, ParamRange, VecN, first_index_of, lazy_file, linear_interp},
};

macro_rules! interpolate_fields {
    ($ty:ident, $str1:expr, $str2:expr, $l: expr, $($field:ident),*) => {
        $ty {
            $($field: linear_interp($str1.$field, $str2.$field, $l)),*,
            dt: $str1.dt * (1.0 - $l),
        }
    };
}

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
    fn v_scale_factor(&self, kappa: f64) -> f64;
    fn mom_unit_coef_mpc(&self, kappa: f64, scale: f64) -> f64 {
        scale / self.v_scale_factor(kappa)
    }
    fn mom_unit_coef_hz(&self, kappa: f64, scale: f64) -> f64 {
        1.547e-15 * self.mom_unit_coef_mpc(kappa, scale)
    }
}

pub trait Kappa {
    fn kappa(&self) -> f64;
}

pub struct BackgroundStateInput<F> {
    pub kappa: f64,
    pub potential: F,
}

impl<F> Kappa for BackgroundStateInput<F> {
    fn kappa(&self) -> f64 {
        self.kappa
    }
}

/// The 'potential' part of scalar perturbation
/// given by y'' / y, where y = a z, z = a \sqrt{a} \phi' / a'
pub trait ZPotential<Ctx: ?Sized> {
    fn z_potential(&self, context: &Ctx) -> f64;
}

pub trait BackgroundStateInputProvider {
    type F;
    fn input(&self) -> &BackgroundStateInput<Self::F>;
}

impl<F> BackgroundStateInputProvider for BackgroundStateInput<F> {
    type F = F;

    fn input(&self) -> &BackgroundStateInput<Self::F> {
        self
    }
}

#[derive(Clone, Copy, Encode, Decode, Debug)]
pub struct BackgroundState {
    pub phi: f64,
    pub mom_phi: f64,
    pub b: f64,
    pub mom_b: f64,
    pub dt: f64,
}

impl BackgroundState {
    pub fn init<P: C2Fn<f64, Output = f64>>(
        phi: f64,
        v_phi: f64,
        a: f64,
        input: &BackgroundStateInput<P>,
    ) -> Self {
        let b = 2.0 / 3.0 * a * a.sqrt();
        let mom_phi = 9.0 / 4.0 * v_phi * b * b;
        let mom_b =
            -a * sqrt(6.0 / input.kappa * a * (v_phi * v_phi + 2.0 * input.potential.value(phi)));
        Self {
            phi,
            mom_phi,
            b,
            mom_b,
            dt: 0.0,
        }
    }
    pub fn init_slowroll<P: C2Fn<f64, Output = f64>>(
        a: f64,
        phi: f64,
        input: &BackgroundStateInput<P>,
    ) -> Self {
        let phi_d = -input.potential.value_d(phi)
            / sqrt(3.0 * input.kappa * input.potential.value(phi))
            * a;
        Self::init(phi, phi_d, a, input)
    }
    pub fn v_a<F>(&self, input: &BackgroundStateInput<F>) -> f64 {
        self.v_scale_factor(input.kappa)
    }
    fn apply_k1(&mut self, dt: f64, kappa: f64) {
        self.b -= self.mom_b * kappa / 6.0 * dt;
    }
    fn apply_k2(&mut self, dt: f64) {
        self.mom_b += 4.0 / 9.0 * self.mom_phi * self.mom_phi / self.b / self.b / self.b * dt;
        self.phi += 4.0 / 9.0 * self.mom_phi / self.b / self.b * dt;
    }
    fn apply_k3<F: C2Fn<f64, Output = f64>>(&mut self, dt: f64, potential: &F) {
        self.mom_b += -dt * 4.5 * self.b * potential.value(self.phi);
        self.mom_phi += -dt * 2.25 * self.b * self.b * potential.value_d(self.phi);
    }
    fn apply_full_k_order2<F: C2Fn<f64, Output = f64>>(
        &mut self,
        delta_t: f64,
        kappa: f64,
        potential: &F,
    ) {
        self.apply_k1(delta_t / 2.0, kappa);
        self.apply_k2(delta_t / 2.0);
        self.apply_k3(delta_t, potential);
        self.apply_k2(delta_t / 2.0);
        self.apply_k1(delta_t / 2.0, kappa);
    }
    pub fn update<F: C2Fn<f64, Output = f64>>(
        &mut self,
        delta_t: f64,
        order: usize,
        input: &BackgroundStateInput<F>,
    ) {
        if order == 2 {
            self.apply_full_k_order2(delta_t, input.kappa, &input.potential);
        } else {
            let beta = 2.0 - pow(2.0, 1.0 / ((order - 1) as f64));
            self.update(delta_t / beta, order - 2, input);
            self.update(delta_t * (1.0 - 2.0 / beta), order - 2, input);
            self.update(delta_t / beta, order - 2, input);
        }
    }
    pub fn hubble_constraint<F: C2Fn<f64, Output = f64>>(
        &self,
        input: &BackgroundStateInput<F>,
    ) -> f64 {
        -3.0 / 2.0 * 1.0 / self.a() / self.a() * 1.0 / input.kappa
            * self.v_a(input)
            * self.v_a(input)
            + 1.0 / 4.0 * self.v_phi() * self.v_phi()
            + 1.0 / 2.0 * input.potential.value(self.phi)
    }
    pub fn epsilon<F>(&self, input: &BackgroundStateInput<F>) -> f64 {
        let a = self.a();
        let v_a = self.v_a(input);
        let v_phi = self.v_phi();
        input.kappa * v_phi * v_phi * a * a / v_a / v_a
    }
    fn a(&self) -> f64 {
        pow(1.5 * self.b, 2.0 / 3.0)
    }
    #[rustfmt::skip]
    pub fn zdd_z<F>(&self, input: &BackgroundStateInput<F>) -> f64 where
        F: C2Fn<f64, Output = f64>,
    {
        -1.0 * self.a() * self.v_phi() + 3.0 * self.v_a(input) * self.v_phi() + 1.0 / 2.0 * self.a() * self.a() * self.a() * input.kappa * 1.0 / self.v_a(input) / self.v_a(input) * self.v_phi() * self.v_phi() * self.v_phi() + -2.0 * self.a() * self.a() * input.kappa * 1.0 / self.v_a(input) * self.v_phi() * self.v_phi() * self.v_phi() + 1.0 / 2.0 * self.a() * self.a() * self.a() * self.a() * input.kappa * input.kappa * 1.0 / self.v_a(input) / self.v_a(input) / self.v_a(input) * self.v_phi() * self.v_phi() * self.v_phi() * self.v_phi() * self.v_phi() + -1.0 * self.a() * self.a() * self.a() * input.kappa * 1.0 / self.v_a(input) / self.v_a(input) * self.v_phi() * self.v_phi() * input.potential.value_d(self.phi) + -1.0 * self.a() * self.a() * 1.0 / self.v_a(input) * self.v_phi() * input.potential.value_dd(self.phi)
    }
    pub fn simulate<F, Cond, Step>(
        &self,
        input: &BackgroundStateInput<F>,
        dn: f64,
        max_dt: f64,
        order: usize,
        mut stop_condition: Cond,
        mut step_monitor: Step,
    ) -> Vec<Self>
    where
        F: C2Fn<f64, Output = f64>,
        Cond: FnMut(&Self) -> bool,
        Step: FnMut(&Self),
    {
        let mut state = *self;
        let mut ret = vec![state];
        while !stop_condition(&state) {
            state.dt = fmin(-dn * state.b / state.mom_b / input.kappa * 6.0, max_dt);
            state.update(state.dt, order, input);
            step_monitor(&state);
            ret.push(state);
        }
        ret
    }
}

impl TimeStateData for BackgroundState {
    fn interpolate(&self, other: &Self, l: f64) -> Self {
        interpolate_fields!(Self, self, other, l, b, mom_b, phi, mom_phi)
    }
}

impl Dt for BackgroundState {
    fn dt(&self) -> f64 {
        self.dt
    }
}

impl ScaleFactor for BackgroundState {
    fn scale_factor(&self) -> f64 {
        self.a()
    }
}

impl ScaleFactorD for BackgroundState {
    fn v_scale_factor(&self, kappa: f64) -> f64 {
        -self.mom_b * kappa / 6.0 / self.scale_factor().sqrt()
    }
}

impl<Ctx> ScaleFactorDD<Ctx> for BackgroundState
where
    Ctx: ?Sized + BackgroundStateInputProvider,
    Ctx::F: C2Fn<f64, Output = f64>,
{
    fn vv_scale_factor(&self, context: &Ctx) -> f64 {
        let input = context.input();
        1.0 / self.a() * self.v_a(input) * self.v_a(input)
            + -1.0 / 2.0 * self.a() * input.kappa * self.v_phi() * self.v_phi()
    }
}

impl<Ctx> ZPotential<Ctx> for BackgroundState
where
    Ctx: ?Sized + BackgroundStateInputProvider,
    Ctx::F: C2Fn<f64, Output = f64>,
{
    fn z_potential(&self, context: &Ctx) -> f64 {
        let input = context.input();
        9.0 / 4.0 * 1.0 / self.a() / self.a() * self.v_a(input) * self.v_a(input)
            + -15.0 / 4.0 * input.kappa * self.v_phi() * self.v_phi()
            + 1.0 / 2.0 * self.a() * self.a() * input.kappa * input.kappa * 1.0
                / self.v_a(input)
                / self.v_a(input)
                * self.v_phi()
                * self.v_phi()
                * self.v_phi()
                * self.v_phi()
            + -2.0 * self.a() * input.kappa * 1.0 / self.v_a(input)
                * self.v_phi()
                * input.potential.value_d(self.phi)
            + -1.0 * input.potential.value_dd(self.phi)
    }
}

impl PhiD for BackgroundState {
    fn v_phi(&self) -> f64 {
        4.0 / 9.0 * self.mom_phi / self.b / self.b
    }
}

#[derive(Debug, Clone, Copy, Encode, Decode)]
pub struct TwoFieldBackgroundState {
    pub b: f64,
    pub mom_b: f64,
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
        F1: C2Fn<f64, Output = f64>,
        F2: C2Fn2<f64, f64, Ret = f64>,
    {
        let b = 2.0 / 3.0 * pow(a, 3.0 / 2.0);
        let mom_phi = v_phi * 9.0 / 4.0 * b * b;
        let mom_chi = v_chi * 9.0 / 4.0 * b * b * exp(2.0 * input.b.value(phi));
        let mom_b = -sqrt(
            (v_phi * v_phi
                + exp(2.0 * input.b.value(phi)) * v_chi * v_chi
                + 2.0 * input.v.value_00(phi, chi))
                * 3.0
                / 2.0
                / input.kappa,
        ) * 3.0
            * b;
        Self {
            b,
            mom_b,
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
        F1: C2Fn<f64, Output = f64>,
        F2: C2Fn2<f64, f64, Ret = f64>,
    {
        let v_phi = (exp(2.0 * input.b.value(phi)) * v_chi * v_chi * input.b.value_d(phi)
            - input.v.value_10(phi, chi))
            / sqrt(3.0 * input.kappa * input.v.value_00(phi, chi));
        Self::init(a, phi, v_phi, chi, v_chi, input)
    }
    fn apply_k1<F1, F2>(&mut self, dt: f64, input: &TwoFieldBackgroundInput<F1, F2>)
    where
        F1: C2Fn<f64, Output = f64>,
        F2: C2Fn2<f64, f64, Ret = f64>,
    {
        self.b += -dt * self.mom_b * input.kappa / 6.0;
    }
    fn apply_k2<F1, F2>(&mut self, dt: f64, input: &TwoFieldBackgroundInput<F1, F2>)
    where
        F1: C2Fn<f64, Output = f64>,
        F2: C2Fn2<f64, f64, Ret = f64>,
    {
        let b2 = self.b * self.b;
        self.mom_b += dt * 4.0 / 9.0 * exp(-2.0 * input.b.value(self.phi)) * self.mom_chi / self.b
            * self.mom_chi
            / b2;
        self.mom_phi += dt * 4.0 / 9.0 * exp(-2.0 * input.b.value(self.phi)) * self.mom_chi
            / self.b
            * self.mom_chi
            / self.b
            * input.b.value_d(self.phi);
        self.chi += dt * 4.0 / 9.0 * exp(-2.0 * input.b.value(self.phi)) * self.mom_chi / b2;
    }
    fn apply_k3(&mut self, dt: f64) {
        self.mom_b += dt * 4.0 / 9.0 * self.mom_phi / self.b * self.mom_phi / self.b / self.b;
        self.phi += dt * 4.0 / 9.0 * self.mom_phi / self.b / self.b;
    }
    fn apply_k4<F1, F2>(&mut self, dt: f64, input: &TwoFieldBackgroundInput<F1, F2>)
    where
        F1: C2Fn<f64, Output = f64>,
        F2: C2Fn2<f64, f64, Ret = f64>,
    {
        let b2 = self.b * self.b;
        self.mom_b += -dt * 4.5 * self.b * input.v.value_00(self.phi, self.chi);
        self.mom_phi += -dt * 9.0 / 4.0 * b2 * input.v.value_10(self.phi, self.chi);
        self.mom_chi += -dt * 9.0 / 4.0 * b2 * input.v.value_01(self.phi, self.chi);
    }
    pub fn apply_full_k_order2<F1, F2>(&mut self, dt: f64, input: &TwoFieldBackgroundInput<F1, F2>)
    where
        F1: C2Fn<f64, Output = f64>,
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
        F1: C2Fn<f64, Output = f64>,
        F2: C2Fn2<f64, f64, Ret = f64>,
    {
        3.0 + -81.0 * 1.0 / self.mom_b / self.mom_b * 1.0 / input.kappa
            * input.v.value_00(self.phi, self.chi)
            * self.b
            * self.b
    }
    pub fn hubble_constraint<F1, F2>(&self, input: &TwoFieldBackgroundInput<F1, F2>) -> f64
    where
        F1: C2Fn<f64, Output = f64>,
        F2: C2Fn2<f64, f64, Ret = f64>,
    {
        1.0 / 4.0 * self.v_phi() * self.v_phi()
            + 1.0 / 4.0 * exp(2.0 * input.b.value(self.phi)) * self.v_chi(input) * self.v_chi(input)
            + 1.0 / 2.0 * input.v.value_00(self.phi, self.chi)
            + -3.0 / 2.0 * 1.0 / input.kappa * 1.0 / self.a() / self.a()
                * self.v_a(input)
                * self.v_a(input)
    }
    pub fn v_chi<F1, F2>(&self, input: &TwoFieldBackgroundInput<F1, F2>) -> f64
    where
        F1: C2Fn<f64, Output = f64>,
        F2: C2Fn2<f64, f64, Ret = f64>,
    {
        4.0 / 9.0 * exp(-2.0 * input.b.value(self.phi)) * self.mom_chi * 1.0 / self.b / self.b
    }
    pub fn vv_chi<F1, F2>(&self, input: &TwoFieldBackgroundInput<F1, F2>) -> f64
    where
        F1: C2Fn<f64, Output = f64>,
        F2: C2Fn2<f64, f64, Ret = f64>,
    {
        -2.0 * self.v_phi() * self.v_chi(input) * input.b.value_d(self.phi)
            + -3.0 * self.v_chi(input) * 1.0 / self.a() * self.v_a(input)
            + -1.0 * exp(-2.0 * input.b.value(self.phi)) * input.v.value_01(self.phi, self.chi)
    }
    pub fn a(&self) -> f64 {
        pow(1.5 * self.b, 2.0 / 3.0)
    }
    pub fn v_a<F1, F2>(&self, input: &TwoFieldBackgroundInput<F1, F2>) -> f64 {
        self.v_b(input) / self.a().sqrt()
    }
    pub fn v_b<F1, F2>(&self, input: &TwoFieldBackgroundInput<F1, F2>) -> f64 {
        -self.mom_b * input.kappa / 6.0
    }
    #[rustfmt::skip]
    pub fn dcs_horizon<F1, F2>(
        &self,
        input: &TwoFieldBackgroundInput<F1, F2>,
        k: f64,
        alpha: f64,
        lambda: f64,
    ) -> f64
    where
        F1: C2Fn<f64, Output = f64>,
        F2: C2Fn2<f64, f64, Ret = f64>,
    {
        let num = -4.0 * self.v_a(input) * self.v_a(input)
            + 24.0 * k * 1.0 / self.a() * alpha * input.kappa * lambda * self.v_a(input) * self.v_a(input) * self.v_chi(input)
            + 12.0 * k * k * 1.0 / self.a() / self.a()
                * alpha
                * alpha
                * input.kappa
                * input.kappa
                * lambda
                * lambda
                * self.v_a(input)
                * self.v_a(input)
                * self.v_chi(input)
                * self.v_chi(input)
            + 4.0 * self.a() * self.a() * input.kappa * input.v.value_00(self.phi, self.chi)
            + -4.0
                * k
                * k
                * alpha
                * alpha
                * input.kappa
                * input.kappa
                * input.kappa
                * lambda
                * lambda
                * self.v_chi(input)
                * self.v_chi(input)
                * input.v.value_00(self.phi, self.chi)
            + 32.0
                * k
                * alpha
                * input.kappa
                * lambda
                * self.v_a(input)
                * self.v_phi()
                * self.v_chi(input)
                * input.b.value_d(self.phi)
            + 16.0 * k * k * 1.0 / self.a()
                * alpha
                * alpha
                * input.kappa
                * input.kappa
                * lambda
                * lambda
                * self.v_a(input)
                * self.v_phi()
                * self.v_chi(input)
                * self.v_chi(input)
                * input.b.value_d(self.phi)
            + -48.0 * k * 1.0 / self.a()
                * alpha
                * lambda
                * self.v_a(input)
                * self.v_a(input)
                * self.v_chi(input)
                * input.b.value_d(self.phi)
                * input.b.value_d(self.phi)
            + 16.0
                * k
                * self.a()
                * alpha
                * input.kappa
                * lambda
                * self.v_phi()
                * self.v_phi()
                * self.v_chi(input)
                * input.b.value_d(self.phi)
                * input.b.value_d(self.phi)
            + -48.0 * k * k * 1.0 / self.a() / self.a()
                * alpha
                * alpha
                * input.kappa
                * lambda
                * lambda
                * self.v_a(input)
                * self.v_a(input)
                * self.v_chi(input)
                * self.v_chi(input)
                * input.b.value_d(self.phi)
                * input.b.value_d(self.phi)
            + 12.0
                * k
                * k
                * alpha
                * alpha
                * input.kappa
                * input.kappa
                * lambda
                * lambda
                * self.v_phi()
                * self.v_phi()
                * self.v_chi(input)
                * self.v_chi(input)
                * input.b.value_d(self.phi)
                * input.b.value_d(self.phi)
            + 4.0
                * exp(2.0 * input.b.value(self.phi))
                * k
                * self.a()
                * alpha
                * input.kappa
                * lambda
                * self.v_chi(input)
                * self.v_chi(input)
                * self.v_chi(input)
                * input.b.value_d(self.phi)
                * input.b.value_d(self.phi)
            + 4.0
                * exp(2.0 * input.b.value(self.phi))
                * k
                * k
                * alpha
                * alpha
                * input.kappa
                * input.kappa
                * lambda
                * lambda
                * self.v_chi(input)
                * self.v_chi(input)
                * self.v_chi(input)
                * self.v_chi(input)
                * input.b.value_d(self.phi)
                * input.b.value_d(self.phi)
            + 16.0
                * k
                * self.a()
                * alpha
                * input.kappa
                * lambda
                * self.v_chi(input)
                * input.v.value_00(self.phi, self.chi)
                * input.b.value_d(self.phi)
                * input.b.value_d(self.phi)
            + 16.0
                * k
                * k
                * alpha
                * alpha
                * input.kappa
                * input.kappa
                * lambda
                * lambda
                * self.v_chi(input)
                * self.v_chi(input)
                * input.v.value_00(self.phi, self.chi)
                * input.b.value_d(self.phi)
                * input.b.value_d(self.phi)
            + -24.0 * k * 1.0 / self.a()
                * alpha
                * lambda
                * self.v_a(input)
                * self.v_a(input)
                * self.v_chi(input)
                * input.b.value_dd(self.phi)
            + -24.0 * k * k * 1.0 / self.a() / self.a()
                * alpha
                * alpha
                * input.kappa
                * lambda
                * lambda
                * self.v_a(input)
                * self.v_a(input)
                * self.v_chi(input)
                * self.v_chi(input)
                * input.b.value_dd(self.phi)
            + 4.0
                * exp(2.0 * input.b.value(self.phi))
                * k
                * self.a()
                * alpha
                * input.kappa
                * lambda
                * self.v_chi(input)
                * self.v_chi(input)
                * self.v_chi(input)
                * input.b.value_dd(self.phi)
            + 4.0
                * exp(2.0 * input.b.value(self.phi))
                * k
                * k
                * alpha
                * alpha
                * input.kappa
                * input.kappa
                * lambda
                * lambda
                * self.v_chi(input)
                * self.v_chi(input)
                * self.v_chi(input)
                * self.v_chi(input)
                * input.b.value_dd(self.phi)
            + 8.0
                * k
                * self.a()
                * alpha
                * input.kappa
                * lambda
                * self.v_chi(input)
                * input.v.value_00(self.phi, self.chi)
                * input.b.value_dd(self.phi)
            + 8.0
                * k
                * k
                * alpha
                * alpha
                * input.kappa
                * input.kappa
                * lambda
                * lambda
                * self.v_chi(input)
                * self.v_chi(input)
                * input.v.value_00(self.phi, self.chi)
                * input.b.value_dd(self.phi)
            + 4.0
                * exp(-2.0 * input.b.value(self.phi))
                * k
                * alpha
                * input.kappa
                * lambda
                * self.v_a(input)
                * input.v.value_01(self.phi, self.chi)
            + -4.0 * exp(-2.0 * input.b.value(self.phi)) * k * k * 1.0 / self.a()
                * alpha
                * alpha
                * input.kappa
                * input.kappa
                * lambda
                * lambda
                * self.v_a(input)
                * self.v_chi(input)
                * input.v.value_01(self.phi, self.chi)
            + 8.0
                * exp(-2.0 * input.b.value(self.phi))
                * k
                * self.a()
                * alpha
                * input.kappa
                * lambda
                * self.v_phi()
                * input.b.value_d(self.phi)
                * input.v.value_01(self.phi, self.chi)
            + 4.0
                * exp(-2.0 * input.b.value(self.phi))
                * k
                * k
                * alpha
                * alpha
                * input.kappa
                * input.kappa
                * lambda
                * lambda
                * self.v_phi()
                * self.v_chi(input)
                * input.b.value_d(self.phi)
                * input.v.value_01(self.phi, self.chi)
            + -1.0
                * exp(-4.0 * input.b.value(self.phi))
                * k
                * k
                * alpha
                * alpha
                * input.kappa
                * input.kappa
                * lambda
                * lambda
                * input.v.value_01(self.phi, self.chi)
                * input.v.value_01(self.phi, self.chi)
            + -2.0
                * exp(-2.0 * input.b.value(self.phi))
                * k
                * self.a()
                * alpha
                * input.kappa
                * lambda
                * self.v_chi(input)
                * input.v.value_02(self.phi, self.chi)
            + -2.0
                * exp(-2.0 * input.b.value(self.phi))
                * k
                * k
                * alpha
                * alpha
                * input.kappa
                * input.kappa
                * lambda
                * lambda
                * self.v_chi(input)
                * self.v_chi(input)
                * input.v.value_02(self.phi, self.chi)
            + 4.0
                * k
                * self.a()
                * alpha
                * input.kappa
                * lambda
                * self.v_chi(input)
                * input.b.value_d(self.phi)
                * input.v.value_10(self.phi, self.chi)
            + 4.0
                * k
                * k
                * alpha
                * alpha
                * input.kappa
                * input.kappa
                * lambda
                * lambda
                * self.v_chi(input)
                * self.v_chi(input)
                * input.b.value_d(self.phi)
                * input.v.value_10(self.phi, self.chi)
            + -2.0
                * exp(-2.0 * input.b.value(self.phi))
                * k
                * self.a()
                * alpha
                * input.kappa
                * lambda
                * self.v_phi()
                * input.v.value_11(self.phi, self.chi)
            + -2.0
                * exp(-2.0 * input.b.value(self.phi))
                * k
                * k
                * alpha
                * alpha
                * input.kappa
                * input.kappa
                * lambda
                * lambda
                * self.v_phi()
                * self.v_chi(input)
                * input.v.value_11(self.phi, self.chi);
        let den = 2.0 + 2.0 * k * 1.0 / self.a() * alpha * input.kappa * lambda * self.v_chi(input);
        num / den / den
    }
    pub fn dcs_fa<F1, F2>(
        &self,
        input: &TwoFieldBackgroundInput<F1, F2>,
        k: f64,
        alpha: f64,
        lambda: f64,
    ) -> f64
    where
        F1: C2Fn<f64, Output = f64>,
        F2: C2Fn2<f64, f64, Ret = f64>,
    {
        let a = self.a();
        a * sqrt(a + k * lambda * alpha * input.kappa * self.v_chi(input))
            / 2.0
            / input.kappa.sqrt()
    }
    pub fn dcs_fa_potential<F1, F2>(
        &self,
        input: &TwoFieldBackgroundInput<F1, F2>,
        k: f64,
        alpha: f64,
        lambda: f64,
    ) -> f64
    where
        F1: C2Fn<f64, Output = f64>,
        F2: C2Fn2<f64, f64, Ret = f64>,
    {
        let num = -9.0 * 1.0 / self.a() / self.a() * self.v_a(input) * self.v_a(input)
            + 14.0 * k * 1.0 / self.a() / self.a() / self.a()
                * alpha
                * input.kappa
                * lambda
                * self.v_a(input)
                * self.v_a(input)
                * self.v_chi(input)
            + 7.0 * k * k * 1.0 / self.a() / self.a() / self.a() / self.a()
                * alpha
                * alpha
                * input.kappa
                * input.kappa
                * lambda
                * lambda
                * self.v_a(input)
                * self.v_a(input)
                * self.v_chi(input)
                * self.v_chi(input)
            + 6.0 * input.kappa * input.v.value_00(self.phi, self.chi)
            + 4.0 * k * 1.0 / self.a()
                * alpha
                * input.kappa
                * input.kappa
                * lambda
                * self.v_chi(input)
                * input.v.value_00(self.phi, self.chi)
            + -2.0 * k * k * 1.0 / self.a() / self.a()
                * alpha
                * alpha
                * input.kappa
                * input.kappa
                * input.kappa
                * lambda
                * lambda
                * self.v_chi(input)
                * self.v_chi(input)
                * input.v.value_00(self.phi, self.chi)
            + 32.0 * k * 1.0 / self.a() / self.a()
                * alpha
                * input.kappa
                * lambda
                * self.v_a(input)
                * self.v_phi()
                * self.v_chi(input)
                * input.b.value_d(self.phi)
            + 16.0 * k * k * 1.0 / self.a() / self.a() / self.a()
                * alpha
                * alpha
                * input.kappa
                * input.kappa
                * lambda
                * lambda
                * self.v_a(input)
                * self.v_phi()
                * self.v_chi(input)
                * self.v_chi(input)
                * input.b.value_d(self.phi)
            + -48.0 * k * 1.0 / self.a() / self.a() / self.a()
                * alpha
                * lambda
                * self.v_a(input)
                * self.v_a(input)
                * self.v_chi(input)
                * input.b.value_d(self.phi)
                * input.b.value_d(self.phi)
            + 16.0 * k * 1.0 / self.a()
                * alpha
                * input.kappa
                * lambda
                * self.v_phi()
                * self.v_phi()
                * self.v_chi(input)
                * input.b.value_d(self.phi)
                * input.b.value_d(self.phi)
            + -48.0 * k * k * 1.0 / self.a() / self.a() / self.a() / self.a()
                * alpha
                * alpha
                * input.kappa
                * lambda
                * lambda
                * self.v_a(input)
                * self.v_a(input)
                * self.v_chi(input)
                * self.v_chi(input)
                * input.b.value_d(self.phi)
                * input.b.value_d(self.phi)
            + 12.0 * k * k * 1.0 / self.a() / self.a()
                * alpha
                * alpha
                * input.kappa
                * input.kappa
                * lambda
                * lambda
                * self.v_phi()
                * self.v_phi()
                * self.v_chi(input)
                * self.v_chi(input)
                * input.b.value_d(self.phi)
                * input.b.value_d(self.phi)
            + 4.0 * exp(2.0 * input.b.value(self.phi)) * k * 1.0 / self.a()
                * alpha
                * input.kappa
                * lambda
                * self.v_chi(input)
                * self.v_chi(input)
                * self.v_chi(input)
                * input.b.value_d(self.phi)
                * input.b.value_d(self.phi)
            + 4.0 * exp(2.0 * input.b.value(self.phi)) * k * k * 1.0 / self.a() / self.a()
                * alpha
                * alpha
                * input.kappa
                * input.kappa
                * lambda
                * lambda
                * self.v_chi(input)
                * self.v_chi(input)
                * self.v_chi(input)
                * self.v_chi(input)
                * input.b.value_d(self.phi)
                * input.b.value_d(self.phi)
            + 16.0 * k * 1.0 / self.a()
                * alpha
                * input.kappa
                * lambda
                * self.v_chi(input)
                * input.v.value_00(self.phi, self.chi)
                * input.b.value_d(self.phi)
                * input.b.value_d(self.phi)
            + 16.0 * k * k * 1.0 / self.a() / self.a()
                * alpha
                * alpha
                * input.kappa
                * input.kappa
                * lambda
                * lambda
                * self.v_chi(input)
                * self.v_chi(input)
                * input.v.value_00(self.phi, self.chi)
                * input.b.value_d(self.phi)
                * input.b.value_d(self.phi)
            + -24.0 * k * 1.0 / self.a() / self.a() / self.a()
                * alpha
                * lambda
                * self.v_a(input)
                * self.v_a(input)
                * self.v_chi(input)
                * input.b.value_dd(self.phi)
            + -24.0 * k * k * 1.0 / self.a() / self.a() / self.a() / self.a()
                * alpha
                * alpha
                * input.kappa
                * lambda
                * lambda
                * self.v_a(input)
                * self.v_a(input)
                * self.v_chi(input)
                * self.v_chi(input)
                * input.b.value_dd(self.phi)
            + 4.0 * exp(2.0 * input.b.value(self.phi)) * k * 1.0 / self.a()
                * alpha
                * input.kappa
                * lambda
                * self.v_chi(input)
                * self.v_chi(input)
                * self.v_chi(input)
                * input.b.value_dd(self.phi)
            + 4.0 * exp(2.0 * input.b.value(self.phi)) * k * k * 1.0 / self.a() / self.a()
                * alpha
                * alpha
                * input.kappa
                * input.kappa
                * lambda
                * lambda
                * self.v_chi(input)
                * self.v_chi(input)
                * self.v_chi(input)
                * self.v_chi(input)
                * input.b.value_dd(self.phi)
            + 8.0 * k * 1.0 / self.a()
                * alpha
                * input.kappa
                * lambda
                * self.v_chi(input)
                * input.v.value_00(self.phi, self.chi)
                * input.b.value_dd(self.phi)
            + 8.0 * k * k * 1.0 / self.a() / self.a()
                * alpha
                * alpha
                * input.kappa
                * input.kappa
                * lambda
                * lambda
                * self.v_chi(input)
                * self.v_chi(input)
                * input.v.value_00(self.phi, self.chi)
                * input.b.value_dd(self.phi)
            + 4.0 * exp(-2.0 * input.b.value(self.phi)) * k * 1.0 / self.a() / self.a()
                * alpha
                * input.kappa
                * lambda
                * self.v_a(input)
                * input.v.value_01(self.phi, self.chi)
            + -4.0 * exp(-2.0 * input.b.value(self.phi)) * k * k * 1.0
                / self.a()
                / self.a()
                / self.a()
                * alpha
                * alpha
                * input.kappa
                * input.kappa
                * lambda
                * lambda
                * self.v_a(input)
                * self.v_chi(input)
                * input.v.value_01(self.phi, self.chi)
            + 8.0 * exp(-2.0 * input.b.value(self.phi)) * k * 1.0 / self.a()
                * alpha
                * input.kappa
                * lambda
                * self.v_phi()
                * input.b.value_d(self.phi)
                * input.v.value_01(self.phi, self.chi)
            + 4.0 * exp(-2.0 * input.b.value(self.phi)) * k * k * 1.0 / self.a() / self.a()
                * alpha
                * alpha
                * input.kappa
                * input.kappa
                * lambda
                * lambda
                * self.v_phi()
                * self.v_chi(input)
                * input.b.value_d(self.phi)
                * input.v.value_01(self.phi, self.chi)
            + -1.0 * exp(-4.0 * input.b.value(self.phi)) * k * k * 1.0 / self.a() / self.a()
                * alpha
                * alpha
                * input.kappa
                * input.kappa
                * lambda
                * lambda
                * input.v.value_01(self.phi, self.chi)
                * input.v.value_01(self.phi, self.chi)
            + -2.0 * exp(-2.0 * input.b.value(self.phi)) * k * 1.0 / self.a()
                * alpha
                * input.kappa
                * lambda
                * self.v_chi(input)
                * input.v.value_02(self.phi, self.chi)
            + -2.0 * exp(-2.0 * input.b.value(self.phi)) * k * k * 1.0 / self.a() / self.a()
                * alpha
                * alpha
                * input.kappa
                * input.kappa
                * lambda
                * lambda
                * self.v_chi(input)
                * self.v_chi(input)
                * input.v.value_02(self.phi, self.chi)
            + 4.0 * k * 1.0 / self.a()
                * alpha
                * input.kappa
                * lambda
                * self.v_chi(input)
                * input.b.value_d(self.phi)
                * input.v.value_10(self.phi, self.chi)
            + 4.0 * k * k * 1.0 / self.a() / self.a()
                * alpha
                * alpha
                * input.kappa
                * input.kappa
                * lambda
                * lambda
                * self.v_chi(input)
                * self.v_chi(input)
                * input.b.value_d(self.phi)
                * input.v.value_10(self.phi, self.chi)
            + -2.0 * exp(-2.0 * input.b.value(self.phi)) * k * 1.0 / self.a()
                * alpha
                * input.kappa
                * lambda
                * self.v_phi()
                * input.v.value_11(self.phi, self.chi)
            + -2.0 * exp(-2.0 * input.b.value(self.phi)) * k * k * 1.0 / self.a() / self.a()
                * alpha
                * alpha
                * input.kappa
                * input.kappa
                * lambda
                * lambda
                * self.v_phi()
                * self.v_chi(input)
                * input.v.value_11(self.phi, self.chi);
        let den = 2.0 + 2.0 * k * 1.0 / self.a() * alpha * input.kappa * lambda * self.v_chi(input);
        num / den / den
    }
    #[rustfmt::skip]
    pub fn vvv_chi<F1, F2>(&self, input: &TwoFieldBackgroundInput<F1, F2>) -> f64
    where
        F1: C2Fn<f64, Output = f64>,
        F2: C2Fn2<f64, f64, Ret = f64>,
    {
        18.0 * 1.0 / self.a() / self.a() * self.v_a(input) * self.v_a(input) * self.v_chi(input) + -3.0 * input.kappa * self.v_chi(input) * input.v.value_00(self.phi, self.chi) + 18.0 * 1.0 / self.a() * self.v_a(input) * self.v_phi() * self.v_chi(input) * input.b.value_d(self.phi) + -24.0 * 1.0 / self.a() / self.a() * 1.0 / input.kappa * self.v_a(input) * self.v_a(input) * self.v_chi(input) * input.b.value_d(self.phi) * input.b.value_d(self.phi)
            + 8.0
                * self.v_phi()
                * self.v_phi()
                * self.v_chi(input)
                * input.b.value_d(self.phi)
                * input.b.value_d(self.phi)
            + 2.0
                * exp(2.0 * input.b.value(self.phi))
                * self.v_chi(input)
                * self.v_chi(input)
                * self.v_chi(input)
                * input.b.value_d(self.phi)
                * input.b.value_d(self.phi)
            + 8.0
                * self.v_chi(input)
                * input.v.value_00(self.phi, self.chi)
                * input.b.value_d(self.phi)
                * input.b.value_d(self.phi)
            + -12.0 * 1.0 / self.a() / self.a() * 1.0 / input.kappa
                * self.v_a(input)
                * self.v_a(input)
                * self.v_chi(input)
                * input.b.value_dd(self.phi)
            + 2.0
                * exp(2.0 * input.b.value(self.phi))
                * self.v_chi(input)
                * self.v_chi(input)
                * self.v_chi(input)
                * input.b.value_dd(self.phi)
            + 4.0
                * self.v_chi(input)
                * input.v.value_00(self.phi, self.chi)
                * input.b.value_dd(self.phi)
            + 3.0 * exp(-2.0 * input.b.value(self.phi)) * 1.0 / self.a()
                * self.v_a(input)
                * input.v.value_01(self.phi, self.chi)
            + 4.0
                * exp(-2.0 * input.b.value(self.phi))
                * self.v_phi()
                * input.b.value_d(self.phi)
                * input.v.value_01(self.phi, self.chi)
            + -1.0
                * exp(-2.0 * input.b.value(self.phi))
                * self.v_chi(input)
                * input.v.value_02(self.phi, self.chi)
            + 2.0
                * self.v_chi(input)
                * input.b.value_d(self.phi)
                * input.v.value_10(self.phi, self.chi)
            + -1.0
                * exp(-2.0 * input.b.value(self.phi))
                * self.v_phi()
                * input.v.value_11(self.phi, self.chi)
    }
    pub fn update<F1, F2>(&mut self, dt: f64, input: &TwoFieldBackgroundInput<F1, F2>)
    where
        F1: C2Fn<f64, Output = f64>,
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
        F1: C2Fn<f64, Output = f64>,
        F2: C2Fn2<f64, f64, Ret = f64>,
        Cond: FnMut(&Self) -> bool,
        Step: FnMut(&Self, f64),
    {
        let mut time = 0.0;
        let mut state = *self;
        let mut ret = vec![state];
        while !stop_condition(&state) {
            state.dt = fmin(-dn * state.b / state.mom_b / input.kappa * 6.0, min_dt);
            state.update(state.dt, input);
            step_monitor(&state, time);
            ret.push(state);
            time += state.dt;
        }
        ret
    }
}

impl TimeStateData for TwoFieldBackgroundState {
    fn interpolate(&self, other: &Self, l: f64) -> Self {
        interpolate_fields!(Self, self, other, l, b, mom_b, phi, mom_phi, chi, mom_chi)
    }
}

impl ScaleFactor for TwoFieldBackgroundState {
    fn scale_factor(&self) -> f64 {
        pow(1.5 * self.b, 2.0 / 3.0)
    }
}

impl ScaleFactorD for TwoFieldBackgroundState {
    fn v_scale_factor(&self, kappa: f64) -> f64 {
        -self.mom_b * kappa / 6.0 / self.scale_factor().sqrt()
    }
}

pub trait TwoFieldBackgroundInputProvider {
    type F1;
    type F2;
    fn input(&self) -> &TwoFieldBackgroundInput<Self::F1, Self::F2>;
}

pub trait PhiD {
    fn v_phi(&self) -> f64;
}

impl<F1, F2> TwoFieldBackgroundInputProvider for TwoFieldBackgroundInput<F1, F2> {
    type F1 = F1;
    type F2 = F2;
    fn input(&self) -> &TwoFieldBackgroundInput<Self::F1, Self::F2> {
        self
    }
}

impl PhiD for TwoFieldBackgroundState {
    fn v_phi(&self) -> f64 {
        4.0 / 9.0 * self.mom_phi * 1.0 / self.b / self.b
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
    type Output;
    fn apply(&self, context: &Context, state: &Background, k: f64) -> Self::Output;
}

pub struct DefaultPerturbationInitializer;
impl<Ctx, B> BackgroundFn<Ctx, B> for DefaultPerturbationInitializer
where
    B: ScaleFactorD + ScaleFactor,
    Ctx: Kappa,
{
    type Output = (Complex64, Complex64);

    fn apply(&self, context: &Ctx, state: &B, k: f64) -> Self::Output {
        let a = state.scale_factor();
        let v_a = state.v_scale_factor(context.kappa());
        let h0 = 1.0 / sqrt(2.0 * k);
        let x = Complex64::new(h0 * a.sqrt(), 0.0);
        let v = Complex64::new(v_a / a.sqrt() / 2.0 * h0, -k * h0 / a.sqrt());
        (x, v)
    }
}

pub struct ScalarPerturbationPotential;
impl<Ctx, B> BackgroundFn<Ctx, B> for ScalarPerturbationPotential
where
    B: ScaleFactor + ZPotential<Ctx>,
{
    type Output = f64;

    fn apply(&self, context: &Ctx, state: &B, k: f64) -> Self::Output {
        let a = state.scale_factor();
        k / a * k / a - state.z_potential(context)
    }
}

pub struct NymtgTensorPerturbationPotential {
    pub lambda: f64,
    pub alpha: f64,
}

pub trait ScaleFactorDD<Ctx: ?Sized> {
    fn vv_scale_factor(&self, context: &Ctx) -> f64;
}

impl<Ctx> ScaleFactorDD<Ctx> for TwoFieldBackgroundState
where
    Ctx: ?Sized + TwoFieldBackgroundInputProvider,
    Ctx::F2: C2Fn2<f64, f64, Ret = f64>,
{
    fn vv_scale_factor(&self, context: &Ctx) -> f64 {
        let input = context.input();
        let a = self.a();
        let v_a = self.v_a(input);
        -2.0 * v_a / a * v_a + a * input.kappa * input.v.value_00(self.phi, self.chi)
    }
}

impl<Ctx, B> BackgroundFn<Ctx, B> for NymtgTensorPerturbationPotential
where
    Ctx: ?Sized + Kappa,
    B: ScaleFactorDD<Ctx> + ScaleFactor + ScaleFactorD + PhiD,
{
    type Output = f64;

    fn apply(&self, context: &Ctx, state: &B, k: f64) -> Self::Output {
        let kappa = context.kappa();
        let a = state.scale_factor();
        let v_a = state.v_scale_factor(kappa);
        let vv_a = state.vv_scale_factor(context);
        k / a * (k / a + self.lambda * self.alpha * state.v_phi())
            - 0.75 * v_a / a * v_a / a
            - 1.5 * kappa * vv_a / a
    }
}

pub struct CubicScaleFactor;
impl<Ctx, B> BackgroundFn<Ctx, B> for CubicScaleFactor
where
    B: ScaleFactor,
    Ctx: Kappa,
{
    type Output = f64;

    fn apply(&self, context: &Ctx, state: &B, _k: f64) -> Self::Output {
        let a = state.scale_factor();
        let kappa = context.kappa();
        2.0 * kappa.sqrt() / sqrt(a * a * a)
    }
}

pub struct ScalarPerturbationFactor;
impl<Ctx, B> BackgroundFn<Ctx, B> for ScalarPerturbationFactor
where
    Ctx: Kappa,
    B: ScaleFactor + ScaleFactorD + PhiD,
{
    type Output = f64;

    fn apply(&self, context: &Ctx, state: &B, _k: f64) -> Self::Output {
        let a = state.scale_factor();
        let f = a * a / state.v_scale_factor(context.kappa()) * a.sqrt() * state.v_phi();
        1.0 / f
    }
}

pub trait IndexRangeSelector<Ctx: ?Sized, B: ?Sized> {
    fn select<I>(&self, context: &Ctx, background: &I, length: usize, k: f64) -> (usize, usize)
    where
        I: ?Sized + Index<usize, Output = B>;
}

pub struct HorizonSelector {
    tolerance: f64,
}

impl HorizonSelector {
    pub fn new(tolerance: f64) -> Self {
        Self { tolerance }
    }
}

impl<Ctx, B> IndexRangeSelector<Ctx, B> for HorizonSelector
where
    B: ?Sized + ScaleFactorD,
    Ctx: ?Sized + Kappa,
{
    fn select<I>(&self, context: &Ctx, background: &I, length: usize, k: f64) -> (usize, usize)
    where
        I: ?Sized + Index<usize, Output = B>,
    {
        let kappa = context.kappa();
        let start = first_index_of(background, 0..length, |s| {
            k / s.v_scale_factor(kappa) < self.tolerance
        })
        .unwrap_or(0);
        let end = first_index_of(background, 0..length, |s| {
            k / s.v_scale_factor(kappa) < 1.0 / self.tolerance
        })
        .unwrap_or(length - 1);
        (start, end)
    }
}

pub struct HorizonSelectorWithExlusion {
    tolerance: f64,
    excluded_n_range: (f64, f64),
}
impl HorizonSelectorWithExlusion {
    pub fn new(tolerance: f64, excluded_n_range: (f64, f64)) -> Self {
        Self {
            tolerance,
            excluded_n_range,
        }
    }
}

impl<Ctx, B> IndexRangeSelector<Ctx, B> for HorizonSelectorWithExlusion
where
    B: ?Sized + ScaleFactorD + ScaleFactor,
    Ctx: ?Sized + Kappa,
{
    fn select<I>(&self, context: &Ctx, background: &I, length: usize, k: f64) -> (usize, usize)
    where
        I: ?Sized + Index<usize, Output = B>,
    {
        let kappa = context.kappa();
        let start = first_index_of(background, 0..length, |s| {
            k / s.v_scale_factor(kappa) < self.tolerance
        })
        .unwrap_or(0);
        let end = first_index_of(background, 0..length, |s| {
            k / s.v_scale_factor(kappa) < 1.0 / self.tolerance
        })
        .unwrap_or(length - 1);
        let range = start..end;
        let start_n = first_index_of(background, 0..length, |s|s.scale_factor().ln() > self.excluded_n_range.0).unwrap();
        let end_n = first_index_of(background, 0..length, |s|s.scale_factor().ln() > self.excluded_n_range.1).unwrap();
        let range0 = start_n..end_n;
        if range.contains(&start_n) || range.contains(&end_n) || range0.contains(&start) || range0.contains(&end) {
            (min(start, start_n), max(end, end_n))
        } else {
            (start, end)
        }
    }
}

pub struct HamitonianSimulator<
    'a,
    'b,
    Ctx: ?Sized,
    I: ?Sized,
    B,
    Initializer,
    Potential,
    RangeSelector,
    PertCoef,
> {
    context: &'a Ctx,
    length: usize,
    background_state: &'b I,
    _background_elem: PhantomData<&'b [B]>,
    potential: Potential,
    range_selector: RangeSelector,
    initializer: Initializer,
    pert_coef: PertCoef,
}

impl<'a, 'b, Ctx, I, B, Initializer, Potential, RangeSelector, PertCoef>
    HamitonianSimulator<'a, 'b, Ctx, I, B, Initializer, Potential, RangeSelector, PertCoef>
where
    I: Index<usize, Output = B> + ?Sized,
    Ctx: ?Sized,
{
    pub fn new(
        context: &'a Ctx,
        length: usize,
        background_state: &'b I,
        initializer: Initializer,
        potential: Potential,
        range_selector: RangeSelector,
        pert_coef: PertCoef,
    ) -> Self {
        Self {
            context,
            length,
            background_state,
            _background_elem: PhantomData,
            initializer,
            potential,
            range_selector,
            pert_coef,
        }
    }
}

impl<'a, 'b, Ctx, I, B, Initializer, Potential, Horizon, PertCoef>
    HamitonianSimulator<'a, 'b, Ctx, I, B, Initializer, Potential, Horizon, PertCoef>
where
    Ctx: ?Sized,
    I: Index<usize, Output = B> + ?Sized,
    B: TimeStateData + Dt + ScaleFactor,
    Potential: BackgroundFn<Ctx, B, Output = f64>,
    Horizon: IndexRangeSelector<Ctx, B>,
    Initializer: BackgroundFn<Ctx, B, Output = (Complex64, Complex64)>,
    PertCoef: BackgroundFn<Ctx, B, Output = f64>,
{
    // fn get_eval_index_range(&self, k: f64) -> (usize, usize) {
    //     let mut begin = 0usize;
    //     let mut end = self.length - 1;
    //     let mut last_flag = false;
    //     for i in 0..self.length {
    //         let flag = self
    //             .range_selector
    //             .apply(self.context, &self.background_state[i], k);
    //         if !last_flag && flag {
    //             begin = i;
    //         }
    //         if last_flag && !flag {
    //             end = i;
    //         }
    //         last_flag = flag;
    //     }
    //     (begin, end)
    // }
    pub fn run<F>(&self, k: f64, da: f64, mut consumer: F) -> Complex64
    where
        F: FnMut(&Self, &B, &HamitonianState<Complex64>, Complex64, f64, f64),
    {
        let (start_index, end_index) =
            self.range_selector
                .select(self.context, self.background_state, self.length, k);
        println!("start n = {}, end n = {}", self.background_state[start_index].scale_factor().ln(), self.background_state[end_index].scale_factor().ln());
        let mut time_interpolator = LinearInterpolator {
            cursor: start_index,
            local_time: 0.0,
        };
        let (x0, v0) = self
            .initializer
            .apply(self.context, &self.background_state[start_index], k);
        let mut state = HamitonianState {
            x: x0,
            mom: v0.conj(),
        };
        while time_interpolator.cursor < end_index {
            let background_state = time_interpolator.get(self.background_state);
            let potential = self.potential.apply(self.context, &background_state, k);
            let dt = if potential > 0.0 {
                fmin(da / potential.sqrt(), background_state.dt())
            } else {
                background_state.dt()
            };
            state.apply_full_k_order2(dt, 1.0.into(), potential.into());
            consumer(
                self,
                &background_state,
                &state,
                state.x * self.pert_coef.apply(self.context, &background_state, k),
                potential,
                dt,
            );
            time_interpolator.advance(self.background_state, dt);
        }
        let background_state = time_interpolator.get(self.background_state);
        state.x * self.pert_coef.apply(self.context, &background_state, k)
    }
    pub fn spectrum(&self, k_range: ParamRange<f64>, da: f64) -> Vec<f64>
    where
        Self: Send + Sync,
    {
        let done_count = AtomicUsize::new(0);
        (0..k_range.count)
            .into_par_iter()
            .map(|i| {
                let k = k_range.log_interp(i);
                let state = self.run(k, da, |_, _, _, _, _, _| {}).abs();
                println!(
                    "[spectrum]({}/{}) k = {}",
                    done_count.fetch_add(1, Ordering::SeqCst) + 1,
                    k_range.count,
                    k
                );
                k * k * k / 2.0 / PI * state * state
            })
            .collect::<Vec<_>>()
    }
    pub fn spectrum_with_cache(
        &self,
        fname: &str,
        k_range: ParamRange<f64>,
        da: f64,
    ) -> util::Result<Vec<f64>>
    where
        Self: Send + Sync,
    {
        lazy_file(fname, BINCODE_CONFIG, || self.spectrum(k_range, da))
    }
}

pub const BINCODE_CONFIG: Configuration = standard();

/// Theory used by 2406.16549.
/// However, a symplectic method seems impossible, so we use 4th order Runge-Kutta method.
#[derive(Clone, Copy, Debug, Encode, Decode)]
pub struct BiNymtgBackgroundState {
    pub a: f64,
    pub v_a: f64,
    pub phi: f64,
    pub v_phi: f64,
    pub chi: f64,
    pub v_chi: f64,
    pub dt: f64,
}

pub struct BiNymtgBackgroundStateInput<Alpha, Beta, V> {
    pub kappa: f64,
    pub dim: f64,
    pub alpha: Alpha,
    pub beta: Beta,
    pub potential_v: V,
}

impl BiNymtgBackgroundState {
    pub fn init<Alpha, Beta, V>(
        a: f64,
        phi: f64,
        v_phi: f64,
        chi: f64,
        v_chi: f64,
        input: &BiNymtgBackgroundStateInput<Alpha, Beta, V>,
    ) -> Self
    where
        Alpha: C2Fn<f64, Output = f64>,
        Beta: C2Fn<f64, Output = f64>,
        V: C2Fn2<f64, f64, Ret = f64>,
    {
        let part = (1.0 / ((-2.0) * (input.dim) + (2.0) * ((input.dim) * (input.dim))))
            * (input.kappa)
            * ((2.0) * ((v_chi) * (v_chi))
                + (4.0) * (input.potential_v.value_00(phi, chi))
                + (2.0) * ((v_phi) * (v_phi)) * (input.alpha.value(phi))
                + (3.0) * ((v_phi) * (v_phi) * (v_phi) * (v_phi)) * (input.beta.value(phi)));
        let v_a = a * part.sqrt();
        Self {
            a,
            v_a,
            phi,
            v_phi,
            chi,
            v_chi,
            dt: 0.0,
        }
    }
    pub fn init_slowroll<Alpha, Beta, V>(
        time: f64,
        a: f64,
        q: f64,
        v0: f64,
        chi: f64,
        v_chi: f64,
        input: &BiNymtgBackgroundStateInput<Alpha, Beta, V>,
    ) -> Self
    where
        Alpha: C2Fn<f64, Output = f64>,
        Beta: C2Fn<f64, Output = f64>,
        V: C2Fn2<f64, f64, Ret = f64>,
    {
        let epsilon = 1.0 / q;
        let v_phi = sqrt(2.0 / epsilon) / (-time);
        let phi = sqrt(2.0 / epsilon)
            * log(sqrt(epsilon - 3.0) / epsilon / v0.sqrt() / input.kappa.sqrt() / (-time));
        let v_a = 1.0 / epsilon / time;
        Self {
            a,
            v_a,
            phi,
            v_phi,
            chi,
            v_chi,
            dt: 0.0,
        }
    }
    pub fn vv<Alpha, Beta, V>(
        &self,
        input: &BiNymtgBackgroundStateInput<Alpha, Beta, V>,
    ) -> (f64, f64, f64)
    where
        Alpha: C2Fn<f64, Output = f64>,
        Beta: C2Fn<f64, Output = f64>,
        V: C2Fn2<f64, f64, Ret = f64>,
    {
        let vv_a =
            (-1.0) * (1.0 / (-1.0 + input.dim)) * (1.0 / (self.a)) * ((self.v_a) * (self.v_a))
                + (3.0 / 2.0)
                    * (1.0 / (-1.0 + input.dim))
                    * (input.dim)
                    * (1.0 / (self.a))
                    * ((self.v_a) * (self.v_a))
                + (-1.0 / 2.0)
                    * (1.0 / (-1.0 + input.dim))
                    * ((input.dim) * (input.dim))
                    * (1.0 / (self.a))
                    * ((self.v_a) * (self.v_a))
                + (-1.0 / 2.0)
                    * (1.0 / (-1.0 + input.dim))
                    * (self.a)
                    * (input.kappa)
                    * ((self.v_chi) * (self.v_chi))
                + (1.0 / (-1.0 + input.dim))
                    * (self.a)
                    * (input.kappa)
                    * (input.potential_v.value_00(self.phi, self.chi))
                + (-1.0 / 2.0)
                    * (1.0 / (-1.0 + input.dim))
                    * (self.a)
                    * (input.kappa)
                    * ((self.v_phi) * (self.v_phi))
                    * (input.alpha.value(self.phi))
                + (-1.0 / 4.0)
                    * (1.0 / (-1.0 + input.dim))
                    * (self.a)
                    * (input.kappa)
                    * ((self.v_phi) * (self.v_phi) * (self.v_phi) * (self.v_phi))
                    * (input.beta.value(self.phi));
        let vv_phi = ((-4.0)
            * (input.dim)
            * (1.0 / (self.a))
            * (self.v_a)
            * (self.v_phi)
            * (input.alpha.value(self.phi))
            + (-4.0)
                * (input.dim)
                * (1.0 / (self.a))
                * (self.v_a)
                * ((self.v_phi) * (self.v_phi) * (self.v_phi))
                * (input.beta.value(self.phi))
            + (-2.0) * ((self.v_phi) * (self.v_phi)) * (input.alpha.value_d(self.phi))
            + (-3.0)
                * ((self.v_phi) * (self.v_phi) * (self.v_phi) * (self.v_phi))
                * (input.beta.value_d(self.phi))
            + (-4.0) * (input.potential_v.value_10(self.phi, self.chi)))
            / ((4.0) * (input.alpha.value(self.phi))
                + (12.0) * ((self.v_phi) * (self.v_phi)) * (input.beta.value(self.phi)));
        let vv_chi = (-1.0)
            * (1.0 / (self.a))
            * ((input.dim) * (self.v_a) * (self.v_chi)
                + (self.a) * (input.potential_v.value_01(self.phi, self.chi)));
        (vv_a, vv_phi, vv_chi)
    }
    fn delta<Alpha, Beta, V>(
        &self,
        input: &BiNymtgBackgroundStateInput<Alpha, Beta, V>,
    ) -> VecN<6, f64>
    where
        Alpha: C2Fn<f64, Output = f64>,
        Beta: C2Fn<f64, Output = f64>,
        V: C2Fn2<f64, f64, Ret = f64>,
    {
        let vv = self.vv(input);
        VecN::new([self.v_a, self.v_phi, self.v_chi, vv.0, vv.1, vv.2])
    }
    fn update_inplace_with(&mut self, dt: f64, delta: &VecN<6, f64>) {
        self.a += delta[0] * dt;
        self.phi += delta[1] * dt;
        self.chi += delta[2] * dt;
        self.v_a += delta[3] * dt;
        self.v_phi += delta[4] * dt;
        self.v_chi += delta[5] * dt;
    }
    fn update_with(&self, dt: f64, delta: &VecN<6, f64>) -> Self {
        let mut ret = *self;
        ret.update_inplace_with(dt, delta);
        ret
    }
    pub fn update<Alpha, Beta, V>(
        &mut self,
        dt: f64,
        input: &BiNymtgBackgroundStateInput<Alpha, Beta, V>,
    ) where
        Alpha: C2Fn<f64, Output = f64>,
        Beta: C2Fn<f64, Output = f64>,
        V: C2Fn2<f64, f64, Ret = f64>,
    {
        let k1 = self.delta(input);
        let k2 = self.update_with(dt / 2.0, &k1).delta(input);
        let k3 = self.update_with(dt / 2.0, &k2).delta(input);
        let k4 = self.update_with(dt, &k3).delta(input);
        self.update_inplace_with(dt / 6.0, &(k1 + k2 * 2.0 + k3 * 2.0 + k4));
    }
    pub fn simulate<Alpha, Beta, V, Cond, Step>(
        &self,
        input: &BiNymtgBackgroundStateInput<Alpha, Beta, V>,
        dt: f64,
        mut stop_condition: Cond,
        mut step_monitor: Step,
    ) -> Vec<Self>
    where
        Alpha: C2Fn<f64, Output = f64>,
        Beta: C2Fn<f64, Output = f64>,
        V: C2Fn2<f64, f64, Ret = f64>,
        Cond: FnMut(&Self) -> bool,
        Step: FnMut(&Self),
    {
        let mut state = *self;
        let mut ret = vec![state];
        while !stop_condition(&state) {
            state.dt = dt;
            state.update(dt, input);
            ret.push(state);
            step_monitor(&state);
        }
        ret
    }
    pub fn epsilon<Alpha, Beta, V>(
        &self,
        input: &BiNymtgBackgroundStateInput<Alpha, Beta, V>,
    ) -> f64
    where
        Alpha: C2Fn<f64, Output = f64>,
        Beta: C2Fn<f64, Output = f64>,
        V: C2Fn2<f64, f64, Ret = f64>,
    {
        (1.0 / (-1.0 + input.dim))
            * ((self.a) * (self.a))
            * (input.kappa)
            * (1.0 / (self.v_a) / (self.v_a))
            * ((self.v_chi) * (self.v_chi)
                + ((self.v_phi) * (self.v_phi)) * (input.alpha.value(self.phi))
                + ((self.v_phi) * (self.v_phi) * (self.v_phi) * (self.v_phi))
                    * (input.beta.value(self.phi)))
    }
    pub fn hubble_constraint<Alpha, Beta, V>(
        &self,
        input: &BiNymtgBackgroundStateInput<Alpha, Beta, V>,
    ) -> f64
    where
        Alpha: C2Fn<f64, Output = f64>,
        Beta: C2Fn<f64, Output = f64>,
        V: C2Fn2<f64, f64, Ret = f64>,
    {
        (1.0 / 4.0)
            * (input.dim)
            * (1.0 / (self.a) / (self.a))
            * (1.0 / (input.kappa))
            * ((self.v_a) * (self.v_a))
            + (-1.0 / 4.0)
                * ((input.dim) * (input.dim))
                * (1.0 / (self.a) / (self.a))
                * (1.0 / (input.kappa))
                * ((self.v_a) * (self.v_a))
            + (1.0 / 4.0) * ((self.v_chi) * (self.v_chi))
            + (1.0 / 2.0) * (input.potential_v.value_00(self.phi, self.chi))
            + (1.0 / 4.0) * ((self.v_phi) * (self.v_phi)) * (input.alpha.value(self.phi))
            + (3.0 / 8.0)
                * ((self.v_phi) * (self.v_phi) * (self.v_phi) * (self.v_phi))
                * (input.beta.value(self.phi))
    }
}

impl Dt for BiNymtgBackgroundState {
    fn dt(&self) -> f64 {
        self.dt
    }
}

impl ScaleFactor for BiNymtgBackgroundState {
    fn scale_factor(&self) -> f64 {
        self.a
    }
}

impl ScaleFactorD for BiNymtgBackgroundState {
    fn v_scale_factor(&self, _kappa: f64) -> f64 {
        self.v_a
    }
}

pub trait BiNymtgBackgroundStateInputProvider {
    type Alpha;
    type Beta;
    type V;
    fn input(&self) -> &BiNymtgBackgroundStateInput<Self::Alpha, Self::Beta, Self::V>;
}

impl<Alpha, Beta, V> BiNymtgBackgroundStateInputProvider
    for BiNymtgBackgroundStateInput<Alpha, Beta, V>
{
    type Alpha = Alpha;

    type Beta = Beta;

    type V = V;

    fn input(&self) -> &BiNymtgBackgroundStateInput<Self::Alpha, Self::Beta, Self::V> {
        self
    }
}

impl<Ctx> ScaleFactorDD<Ctx> for BiNymtgBackgroundState
where
    Ctx: ?Sized + BiNymtgBackgroundStateInputProvider,
    Ctx::Alpha: C2Fn<f64, Output = f64>,
    Ctx::Beta: C2Fn<f64, Output = f64>,
    Ctx::V: C2Fn2<f64, f64, Ret = f64>,
{
    fn vv_scale_factor(&self, context: &Ctx) -> f64 {
        self.vv(context.input()).0
    }
}

impl PhiD for BiNymtgBackgroundState {
    fn v_phi(&self) -> f64 {
        self.v_phi
    }
}

impl TimeStateData for BiNymtgBackgroundState {
    fn interpolate(&self, other: &Self, l: f64) -> Self
    where
        Self: Sized,
    {
        interpolate_fields!(Self, self, other, l, a, v_a, phi, v_phi, chi, v_chi)
    }
}
