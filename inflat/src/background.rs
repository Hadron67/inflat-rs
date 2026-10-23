use std::{
    cmp::{max, min},
    f64::consts::PI,
    fmt::Debug,
    marker::PhantomData,
    ops::{AddAssign, Index, Mul},
    sync::atomic::{AtomicUsize, Ordering},
};

use bincode::{
    Decode, Encode,
    config::{Configuration, standard},
};
use libm::{exp, fmin, log, pow, sqrt};
use num_complex::{Complex64, ComplexFloat};
use plotly::{
    Layout, Plot, Scatter,
    common::ExponentFormat,
    layout::{Axis, LayoutGrid},
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{
    c2fn::{C2Fn, C2Fn2},
    scalar::ScalarEffectivePotential,
    util::{self, ParamRange, VecN, first_index_of, half_int_gamma, lazy_file, log_interp},
};

#[macro_export]
macro_rules! interpolate_fields {
    ($ty:ident, $str1:expr, $str2:expr, $l: expr, $($field:ident),*) => {
        $ty {
            $($field: $crate::util::linear_interp($str1.$field, $str2.$field, $l)),*,
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

pub trait BackgroundSolver {
    type State;
    fn create_state(&self, a: f64, v_a: f64, phi: f64, v_phi: f64) -> Self::State;
    fn update(&self, state: &mut Self::State, dt: f64);
    fn simulate<S, M>(
        &self,
        initial_state: Self::State,
        dt: f64,
        mut stop_condition: S,
        mut step: M,
    ) -> Vec<Self::State>
    where
        Self::State: Clone + DtMut,
        S: FnMut(&Self::State) -> bool,
        M: FnMut(&Self::State),
    {
        let mut state = initial_state;
        let mut ret = vec![state.clone()];
        while !stop_condition(&state) {
            state.set_dt(dt);
            self.update(&mut state, dt);
            step(&state);
            ret.push(state.clone());
        }
        ret
    }
    fn evaluate_to_phi(&self, state: &mut Self::State, dt: f64, phi_goal: f64)
    where
        Self::State: Clone + Phi<Self> + PhiD<Self> + Debug,
        Self: Kappa,
    {
        while (phi_goal - state.phi(self)) * state.v_phi(self).signum() > 0.0 {
            self.update(state, dt);
        }
    }
}

pub trait Interpolate {
    fn interpolate(&self, other: &Self, l: f64) -> Self
    where
        Self: Sized;
}

pub trait Dimension {
    fn dimension(&self) -> usize;
}

impl Dimension for usize {
    fn dimension(&self) -> usize {
        *self
    }
}

pub trait Dt {
    fn dt(&self) -> f64;
}

pub trait DtMut {
    fn set_dt(&mut self, dt: f64);
}

pub trait ScaleFactor<Ctx: ?Sized> {
    fn scale_factor(&self, ctx: &Ctx) -> f64;
}

pub trait ScaleFactorMut<C: ?Sized> {
    fn set_scale_factor(&mut self, ctx: &C, a: f64, v_a: f64);
}

pub const MPC_HZ: f64 = 1.547e-15;

pub trait ScaleFactorD<Ctx: ?Sized> {
    fn v_scale_factor(&self, ctx: &Ctx) -> f64;
    fn mom_unit_coef_mpc(&self, ctx: &Ctx, scale: f64) -> f64 {
        scale / self.v_scale_factor(ctx)
    }
    fn mom_unit_coef_hz(&self, ctx: &Ctx, scale: f64) -> f64 {
        MPC_HZ * self.mom_unit_coef_mpc(ctx, scale)
    }
    fn hubble(&self, ctx: &Ctx) -> f64
    where
        Self: ScaleFactor<Ctx>,
    {
        self.v_scale_factor(ctx) / self.scale_factor(ctx)
    }
}

pub trait Kappa {
    fn kappa(&self) -> f64;
}

pub trait SlowrollKappa<S: ?Sized> {
    fn slowroll_kappa(&self, state: &S) -> f64;
    fn scalar_spectral_power(&self, state: &S) -> f64
    where
        Self: SlowrollEpsilon<S>,
    {
        -2.0 * self.slowroll_epsilon(state) - self.slowroll_kappa(state)
    }
}

pub trait SlowrollEpsilon<S: ?Sized> {
    fn slowroll_epsilon(&self, state: &S) -> f64;
    fn scalar_spectral_amplitude(&self, state: &S) -> f64
    where
        Self: Kappa,
        S: ScaleFactor<Self> + ScaleFactorD<Self>,
    {
        let hubble = state.hubble(self);
        self.kappa() * hubble * hubble / self.slowroll_epsilon(state) / 8.0 / PI / PI
    }
}

pub trait HubbleConstraint<S: ?Sized> {
    fn hubble_constraint(&self, state: &S) -> f64;
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

impl<S, F> SlowrollKappa<S> for BackgroundStateInput<F>
where
    S: ScaleFactor<Self> + ScaleFactorD<Self> + Phi<Self> + PhiD<Self>,
    F: C2Fn<f64, Output = f64>,
{
    fn slowroll_kappa(&self, state: &S) -> f64 {
        let hubble = state.hubble(self);
        let phi = state.phi(self);
        let v_phi = state.v_phi(self);
        self.kappa * v_phi * v_phi / hubble / hubble
            - 2.0 * self.potential.value_d(phi) / hubble / v_phi
            - 6.0
    }
}

impl<S, F> SlowrollEpsilon<S> for BackgroundStateInput<F>
where
    S: ScaleFactor<Self> + ScaleFactorD<Self> + PhiD<Self>,
{
    fn slowroll_epsilon(&self, state: &S) -> f64 {
        let hubble = state.hubble(self);
        let v_phi = state.v_phi(self);
        self.kappa * v_phi * v_phi / hubble / hubble / 2.0
    }
}

impl<S, F> HubbleConstraint<S> for BackgroundStateInput<F>
where
    S: ScaleFactor<Self> + ScaleFactorD<Self> + Phi<Self> + PhiD<Self>,
    F: C2Fn<f64, Output = f64>,
{
    fn hubble_constraint(&self, state: &S) -> f64 {
        let hubble = state.hubble(self);
        let phi = state.phi(self);
        let v_phi = state.v_phi(self);
        -1.5 * hubble * hubble / self.kappa + 0.25 * v_phi * v_phi + 0.5 * self.potential.value(phi)
    }
}

impl<F> BackgroundSolver for BackgroundStateInput<F>
where
    F: C2Fn<f64, Output = f64>,
{
    type State = BackgroundState;

    fn create_state(&self, a: f64, _v_a: f64, phi: f64, v_phi: f64) -> Self::State {
        BackgroundState::init_normal(phi, v_phi, a, self)
    }

    fn update(&self, state: &mut Self::State, dt: f64) {
        state.update(dt, 4, self);
    }
}

impl<F> ScalarEffectivePotential for BackgroundStateInput<F>
where
    F: C2Fn<f64, Output = f64>,
{
    fn scalar_eff_potential(&self, _a: f64, _v_a: f64, phi: f64, _v_phi: f64) -> f64 {
        self.potential.value(phi)
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
    pub fn a_to_b(a: f64) -> f64 {
        pow(a, 1.5) / 1.5
    }
    pub fn b_to_a(b: f64) -> f64 {
        pow(1.5 * b, 1.0 / 1.5)
    }
    pub fn v_a_to_mom_b(v_a: f64, kappa: f64, a: f64) -> f64 {
        -v_a * a.sqrt() * 6.0 / kappa
    }
    pub fn mom_b_to_v_a(mom_b: f64, kappa: f64, a: f64) -> f64 {
        -mom_b * kappa / 6.0 / a.sqrt()
    }
    pub fn mom_phi_to_v_phi(mom_phi: f64, b: f64) -> f64 {
        4.0 / 9.0 * mom_phi / b / b
    }
    pub fn v_phi_to_mom_phi(v_phi: f64, b: f64) -> f64 {
        v_phi * b * b * 9.0 / 4.0
    }
    pub fn init_normal<P: C2Fn<f64, Output = f64>>(
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
        Self::init_normal(phi, phi_d, a, input)
    }
    pub fn v_a<F>(&self, input: &BackgroundStateInput<F>) -> f64 {
        self.v_scale_factor(input)
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
    pub fn epsilon<F>(&self, input: &BackgroundStateInput<F>) -> f64 {
        let a = self.a();
        let v_a = self.v_a(input);
        let v_phi = self.v_phi(&());
        input.kappa * v_phi * v_phi * a * a / v_a / v_a
    }
    fn a(&self) -> f64 {
        pow(1.5 * self.b, 2.0 / 3.0)
    }
    #[rustfmt::skip]
    pub fn zdd_z<F>(&self, input: &BackgroundStateInput<F>) -> f64 where
        F: C2Fn<f64, Output = f64>,
    {
        -1.0 * self.a() * self.v_phi(&()) + 3.0 * self.v_a(input) * self.v_phi(&()) + 1.0 / 2.0 * self.a() * self.a() * self.a() * input.kappa * 1.0 / self.v_a(input) / self.v_a(input) * self.v_phi(&()) * self.v_phi(&()) * self.v_phi(&()) + -2.0 * self.a() * self.a() * input.kappa * 1.0 / self.v_a(input) * self.v_phi(&()) * self.v_phi(&()) * self.v_phi(&()) + 1.0 / 2.0 * self.a() * self.a() * self.a() * self.a() * input.kappa * input.kappa * 1.0 / self.v_a(input) / self.v_a(input) / self.v_a(input) * self.v_phi(&()) * self.v_phi(&()) * self.v_phi(&()) * self.v_phi(&()) * self.v_phi(&()) + -1.0 * self.a() * self.a() * self.a() * input.kappa * 1.0 / self.v_a(input) / self.v_a(input) * self.v_phi(&()) * self.v_phi(&()) * input.potential.value_d(self.phi) + -1.0 * self.a() * self.a() * 1.0 / self.v_a(input) * self.v_phi(&()) * input.potential.value_dd(self.phi)
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

impl Interpolate for BackgroundState {
    fn interpolate(&self, other: &Self, l: f64) -> Self {
        interpolate_fields!(Self, self, other, l, b, mom_b, phi, mom_phi)
    }
}

impl Dt for BackgroundState {
    fn dt(&self) -> f64 {
        self.dt
    }
}

impl DtMut for BackgroundState {
    fn set_dt(&mut self, dt: f64) {
        self.dt = dt;
    }
}

impl<C> Phi<C> for BackgroundState {
    fn phi(&self, _: &C) -> f64 {
        self.phi
    }
}

impl<C> ScaleFactor<C> for BackgroundState {
    fn scale_factor(&self, _: &C) -> f64 {
        self.a()
    }
}

impl<C> ScaleFactorMut<C> for BackgroundState
where
    C: Kappa,
{
    fn set_scale_factor(&mut self, ctx: &C, a: f64, v_a: f64) {
        let v_phi = self.v_phi(ctx);
        self.b = BackgroundState::a_to_b(a);
        self.mom_b = BackgroundState::v_a_to_mom_b(v_a, ctx.kappa(), a);
        self.mom_phi = BackgroundState::v_phi_to_mom_phi(v_phi, self.b);
    }
}

impl<C: Kappa> ScaleFactorD<C> for BackgroundState {
    fn v_scale_factor(&self, ctx: &C) -> f64 {
        -self.mom_b * ctx.kappa() / 6.0 / self.scale_factor(&()).sqrt()
    }
}

impl<Ctx> ScaleFactorDD<Ctx> for BackgroundState
where
    Ctx: ?Sized + BackgroundStateInputProvider,
    Ctx::F: C2Fn<f64, Output = f64>,
{
    fn vv_scale_factor(&self, context: &Ctx) -> f64 {
        let input = context.input();
        let v_phi = self.v_phi(&());
        1.0 / self.a() * self.v_a(input) * self.v_a(input)
            + -1.0 / 2.0 * self.a() * input.kappa * v_phi * v_phi
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
            + -15.0 / 4.0 * input.kappa * self.v_phi(&()) * self.v_phi(&())
            + 1.0 / 2.0 * self.a() * self.a() * input.kappa * input.kappa * 1.0
                / self.v_a(input)
                / self.v_a(input)
                * self.v_phi(&())
                * self.v_phi(&())
                * self.v_phi(&())
                * self.v_phi(&())
            + -2.0 * self.a() * input.kappa * 1.0 / self.v_a(input)
                * self.v_phi(&())
                * input.potential.value_d(self.phi)
            + -1.0 * input.potential.value_dd(self.phi)
    }
}

impl<C> PhiD<C> for BackgroundState {
    fn v_phi(&self, _: &C) -> f64 {
        4.0 / 9.0 * self.mom_phi / self.b / self.b
    }
}

pub fn spectrum_k_range_from_background<B, C>(
    background: &[B],
    ctx: &C,
    k_range_in: ParamRange<f64>,
    horizon_tolerance: f64,
) -> ParamRange<f64>
where
    B: ScaleFactorD<C>,
{
    let min_k = background[0].v_scale_factor(ctx) * horizon_tolerance;
    let max_k = background.last().unwrap().v_scale_factor(ctx) / horizon_tolerance;
    ParamRange::new(
        log_interp(min_k, max_k, k_range_in.start),
        log_interp(min_k, max_k, k_range_in.end),
        k_range_in.count,
    )
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

impl<F1, F2> Kappa for TwoFieldBackgroundInput<F1, F2> {
    fn kappa(&self) -> f64 {
        self.kappa
    }
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
        1.0 / 4.0 * self.v_phi(&()) * self.v_phi(&())
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
        -2.0 * self.v_phi(&()) * self.v_chi(input) * input.b.value_d(self.phi)
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
                * self.v_phi(&())
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
                * self.v_phi(&())
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
                * self.v_phi(&())
                * self.v_phi(&())
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
                * self.v_phi(&())
                * self.v_phi(&())
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
                * self.v_phi(&())
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
                * self.v_phi(&())
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
                * self.v_phi(&())
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
                * self.v_phi(&())
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
                * self.v_phi(&())
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
                * self.v_phi(&())
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
                * self.v_phi(&())
                * self.v_phi(&())
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
                * self.v_phi(&())
                * self.v_phi(&())
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
                * self.v_phi(&())
                * input.b.value_d(self.phi)
                * input.v.value_01(self.phi, self.chi)
            + 4.0 * exp(-2.0 * input.b.value(self.phi)) * k * k * 1.0 / self.a() / self.a()
                * alpha
                * alpha
                * input.kappa
                * input.kappa
                * lambda
                * lambda
                * self.v_phi(&())
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
                * self.v_phi(&())
                * input.v.value_11(self.phi, self.chi)
            + -2.0 * exp(-2.0 * input.b.value(self.phi)) * k * k * 1.0 / self.a() / self.a()
                * alpha
                * alpha
                * input.kappa
                * input.kappa
                * lambda
                * lambda
                * self.v_phi(&())
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
        18.0 * 1.0 / self.a() / self.a() * self.v_a(input) * self.v_a(input) * self.v_chi(input) + -3.0 * input.kappa * self.v_chi(input) * input.v.value_00(self.phi, self.chi) + 18.0 * 1.0 / self.a() * self.v_a(input) * self.v_phi(&()) * self.v_chi(input) * input.b.value_d(self.phi) + -24.0 * 1.0 / self.a() / self.a() * 1.0 / input.kappa * self.v_a(input) * self.v_a(input) * self.v_chi(input) * input.b.value_d(self.phi) * input.b.value_d(self.phi)
            + 8.0
                * self.v_phi(&())
                * self.v_phi(&())
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
                * self.v_phi(&())
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
                * self.v_phi(&())
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

impl Interpolate for TwoFieldBackgroundState {
    fn interpolate(&self, other: &Self, l: f64) -> Self {
        interpolate_fields!(Self, self, other, l, b, mom_b, phi, mom_phi, chi, mom_chi)
    }
}

impl<C> ScaleFactor<C> for TwoFieldBackgroundState {
    fn scale_factor(&self, _: &C) -> f64 {
        pow(1.5 * self.b, 2.0 / 3.0)
    }
}

impl<C: Kappa> ScaleFactorD<C> for TwoFieldBackgroundState {
    fn v_scale_factor(&self, ctx: &C) -> f64 {
        -self.mom_b * ctx.kappa() / 6.0 / self.scale_factor(&()).sqrt()
    }
}

pub trait TwoFieldBackgroundInputProvider {
    type F1;
    type F2;
    fn input(&self) -> &TwoFieldBackgroundInput<Self::F1, Self::F2>;
}

pub trait PhiD<C: ?Sized> {
    fn v_phi(&self, ctx: &C) -> f64;
}

pub trait Phi<C: ?Sized> {
    fn phi(&self, ctx: &C) -> f64;
}

impl<F1, F2> TwoFieldBackgroundInputProvider for TwoFieldBackgroundInput<F1, F2> {
    type F1 = F1;
    type F2 = F2;
    fn input(&self) -> &TwoFieldBackgroundInput<Self::F1, Self::F2> {
        self
    }
}

impl<C> PhiD<C> for TwoFieldBackgroundState {
    fn v_phi(&self, _: &C) -> f64 {
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
        I::Output: Interpolate + Dt + Sized,
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
    B: ScaleFactorD<Ctx> + ScaleFactor<Ctx>,
{
    type Output = (Complex64, Complex64);

    fn apply(&self, context: &Ctx, state: &B, k: f64) -> Self::Output {
        let a = state.scale_factor(context);
        let v_a = state.v_scale_factor(context);
        let h0 = 1.0 / sqrt(2.0 * k);
        let x = Complex64::new(h0 * a.sqrt(), 0.0);
        let v = Complex64::new(v_a / a.sqrt() / 2.0 * h0, -k * h0 / a.sqrt());
        (x, v)
    }
}

pub struct ScalarPerturbationPotential;
impl<Ctx, B> BackgroundFn<Ctx, B> for ScalarPerturbationPotential
where
    B: ScaleFactor<Ctx> + ZPotential<Ctx>,
{
    type Output = f64;

    fn apply(&self, context: &Ctx, state: &B, k: f64) -> Self::Output {
        let a = state.scale_factor(context);
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
    B: ScaleFactorDD<Ctx> + ScaleFactor<Ctx> + ScaleFactorD<Ctx> + PhiD<Ctx>,
{
    type Output = f64;

    fn apply(&self, context: &Ctx, state: &B, k: f64) -> Self::Output {
        let kappa = context.kappa();
        let a = state.scale_factor(context);
        let v_a = state.v_scale_factor(context);
        let vv_a = state.vv_scale_factor(context);
        let v_phi = state.v_phi(context);
        k / a * (k / a + self.lambda * self.alpha * v_phi)
            - 0.75 * v_a / a * v_a / a
            - 1.5 * kappa * vv_a / a
    }
}

pub struct CubicScaleFactor;
impl<Ctx, B> BackgroundFn<Ctx, B> for CubicScaleFactor
where
    B: ScaleFactor<Ctx>,
    Ctx: Kappa,
{
    type Output = f64;

    fn apply(&self, context: &Ctx, state: &B, _k: f64) -> Self::Output {
        let a = state.scale_factor(context);
        let kappa = context.kappa();
        2.0 * kappa.sqrt() / sqrt(a * a * a)
    }
}

pub struct ScalarPerturbationFactor;
impl<Ctx, B> BackgroundFn<Ctx, B> for ScalarPerturbationFactor
where
    Ctx: ?Sized,
    B: ScaleFactor<Ctx> + ScaleFactorD<Ctx> + PhiD<Ctx>,
{
    type Output = f64;

    fn apply(&self, context: &Ctx, state: &B, _k: f64) -> Self::Output {
        let a = state.scale_factor(context);
        let f = a * a / state.v_scale_factor(context) * a.sqrt() * state.v_phi(context);
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
    B: ?Sized + ScaleFactorD<Ctx>,
    Ctx: ?Sized + Kappa,
{
    fn select<I>(&self, context: &Ctx, background: &I, length: usize, k: f64) -> (usize, usize)
    where
        I: ?Sized + Index<usize, Output = B>,
    {
        let start = first_index_of(background, 0..length, |s| {
            k / s.v_scale_factor(context) < self.tolerance
        })
        .unwrap_or(0);
        let end = first_index_of(background, 0..length, |s| {
            k / s.v_scale_factor(context) < 1.0 / self.tolerance
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
    B: ?Sized + ScaleFactorD<Ctx> + ScaleFactor<Ctx>,
    Ctx: ?Sized,
{
    fn select<I>(&self, context: &Ctx, background: &I, length: usize, k: f64) -> (usize, usize)
    where
        I: ?Sized + Index<usize, Output = B>,
    {
        let start = first_index_of(background, 0..length, |s| {
            k / s.v_scale_factor(context) < self.tolerance
        })
        .unwrap_or(0);
        let end = first_index_of(background, 0..length, |s| {
            k / s.v_scale_factor(context) < 1.0 / self.tolerance
        })
        .unwrap_or(length - 1);
        let range = start..end;
        let start_n = first_index_of(background, 0..length, |s| {
            s.scale_factor(context).ln() > self.excluded_n_range.0
        })
        .unwrap();
        let end_n = first_index_of(background, 0..length, |s| {
            s.scale_factor(context).ln() > self.excluded_n_range.1
        })
        .unwrap();
        let range0 = start_n..end_n;
        if range.contains(&start_n)
            || range.contains(&end_n)
            || range0.contains(&start)
            || range0.contains(&end)
        {
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
    B: Interpolate + Dt + ScaleFactor<Ctx>,
    Potential: BackgroundFn<Ctx, B, Output = f64>,
    Horizon: IndexRangeSelector<Ctx, B>,
    Initializer: BackgroundFn<Ctx, B, Output = (Complex64, Complex64)>,
    PertCoef: BackgroundFn<Ctx, B, Output = f64>,
{
    pub fn run<F>(&self, k: f64, da: f64, mut consumer: F) -> Complex64
    where
        F: FnMut(&Self, &B, &HamitonianState<Complex64>, Complex64, f64, f64),
    {
        let (start_index, end_index) =
            self.range_selector
                .select(self.context, self.background_state, self.length, k);
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
                fmin(da / potential.sqrt(), background_state.dt())
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
    pub fn spectrum(&self, k_range: ParamRange<f64>, da: f64, quite: bool) -> Vec<f64>
    where
        Self: Send + Sync,
    {
        let done_count = AtomicUsize::new(0);
        (0..k_range.count)
            .into_par_iter()
            .map(|i| {
                let k = k_range.log_interp(i);
                let state = self.run(k, da, |_, _, _, _, _, _| {}).abs();
                if !quite {
                    println!(
                        "[spectrum]({}/{}) k = {:e}",
                        done_count.fetch_add(1, Ordering::SeqCst) + 1,
                        k_range.count,
                        k
                    );
                }
                k * k * k / 2.0 / PI / PI * state * state
            })
            .collect::<Vec<_>>()
    }
    pub fn spectrum_with_cache(
        &self,
        fname: &str,
        k_range: ParamRange<f64>,
        da: f64,
        quite: bool,
    ) -> util::Result<Vec<f64>>
    where
        Self: Send + Sync,
    {
        lazy_file(fname, BINCODE_CONFIG, || self.spectrum(k_range, da, quite))
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

impl<C> ScaleFactor<C> for BiNymtgBackgroundState {
    fn scale_factor(&self, _: &C) -> f64 {
        self.a
    }
}

impl<C> ScaleFactorD<C> for BiNymtgBackgroundState {
    fn v_scale_factor(&self, _: &C) -> f64 {
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

impl<C> PhiD<C> for BiNymtgBackgroundState {
    fn v_phi(&self, _: &C) -> f64 {
        self.v_phi
    }
}

impl Interpolate for BiNymtgBackgroundState {
    fn interpolate(&self, other: &Self, l: f64) -> Self
    where
        Self: Sized,
    {
        interpolate_fields!(Self, self, other, l, a, v_a, phi, v_phi, chi, v_chi)
    }
}

/// Computes the coefficient of spectrum, given by $2^{1 - d}\pi^{-d/2} / \Gamma(d/2)$
pub fn spectrum_factor(dim: usize) -> f64 {
    let d = dim as f64;
    2.0.powi(1 - (dim as i32)) * PI.powf(-d / 2.0) / half_int_gamma(dim as u32)
}

pub fn plot_background<'a, I, B, C>(out_file: &str, background: I, ctx: &C)
where
    I: IntoIterator<Item = &'a B> + 'a,
    B: 'a,
    I::Item: ScaleFactor<C> + ScaleFactorD<C> + Phi<C> + PhiD<C>,
{
    let mut efolding = vec![];
    let mut v_a = vec![];
    let mut hubble = vec![];
    let mut phi = vec![];
    let mut v_phi = vec![];
    for b in background {
        efolding.push(b.scale_factor(ctx));
        v_a.push(b.v_scale_factor(ctx));
        hubble.push(b.hubble(ctx));
        phi.push(b.phi(ctx));
        v_phi.push(b.v_phi(ctx));
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
    plot.add_trace(Scatter::new(efolding.clone(), v_a).name("v_a").y_axis("y4"));
    plot.set_layout(
        Layout::new()
            .grid(LayoutGrid::new().columns(1).rows(4))
            .x_axis(Axis::new().exponent_format(ExponentFormat::Power))
            .y_axis(Axis::new().exponent_format(ExponentFormat::Power))
            .y_axis2(Axis::new().exponent_format(ExponentFormat::Power))
            .y_axis3(Axis::new().exponent_format(ExponentFormat::Power))
            .y_axis4(Axis::new().exponent_format(ExponentFormat::Power))
            .height(1000),
    );
    plot.write_html(out_file);
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use crate::background::spectrum_factor;

    #[test]
    fn spec_factor() {
        assert_approx_eq::assert_approx_eq!(spectrum_factor(3), 1.0 / 2.0 / PI / PI);
    }
}
