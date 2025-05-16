use std::{cmp::max, collections::VecDeque, f64::consts::PI, fs::File, io::{self, BufReader, BufWriter, Write}, iter::zip, marker::PhantomData, ops::{Add, Mul, Range, Sub}, sync::{atomic::{AtomicUsize, Ordering}, Mutex}, time::SystemTime};

use bincode::{config::{standard, Configuration}, decode_from_std_read, encode_into_std_write, Decode, Encode};
use libm::{fmax, fmin, log, pow, sqrt};
use num_complex::{Complex64, ComplexFloat};
use num_traits::{abs, real::Real, Float};
use plotly::{layout::{Axis, AxisType, GridPattern, LayoutGrid}, Layout, Plot, Scatter};
use rayon::{iter::{IntoParallelIterator, ParallelIterator}, vec};

use crate::{scalar, util::{limit_length, linear_interp, power_interp}};

pub trait C2Fn {
    fn value(&self, phi: f64) -> f64;
    fn value_d(&self, phi: f64) -> f64;
    fn value_dd(&self, phi: f64) -> f64;
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

#[derive(Clone, Copy, Encode, Decode, Debug)]
pub struct BackgroundState {
    pub phi: f64,
    pub mom_phi: f64,
    pub a: f64,
    pub mom_a: f64,
    pub dt: f64,
}

impl BackgroundState {
    pub fn init<P: C2Fn>(kappa: f64, phi: f64, phi_d: f64, a: f64, potential: &P) -> Self {
        let a2 = a * a;
        let a6 = a2 * a2 * a2;
        let mom_phi = phi_d * a * a;
        let mom_a = -sqrt((mom_phi * mom_phi + 2.0 * potential.value(phi) * a6) * 6.0 / kappa) / a;
        Self { phi, mom_phi, a, mom_a, dt: 0.0 }
    }
    pub fn v_phi(&self) -> f64 {
        self.mom_phi / self.a / self.a
    }
    pub fn v_a(&self, kappa: f64) -> f64 {
        -self.mom_a / 6.0 * kappa
    }
    pub fn z(&self, kappa: f64) -> f64 {
        self.a * self.a * self.v_phi() / self.v_a(kappa)
    }
    pub fn interpolate(&self, other: &Self, l: f64) -> Self {
        Self {
            phi: linear_interp(self.phi, other.phi, l),
            mom_phi: linear_interp(self.mom_phi, other.mom_phi, l),
            a: linear_interp(self.a, other.a, l),
            mom_a: linear_interp(self.mom_a, other.mom_a, l),
            dt: self.dt,
        }
    }
    fn apply_k1(&mut self, dt: f64, kappa: f64) {
        self.a -= self.mom_a * kappa / 6.0 * dt;
    }
    fn apply_k2(&mut self, dt: f64) {
        self.mom_a += self.mom_phi * self.mom_phi / self.a / self.a / self.a * dt;
        self.phi += self.mom_phi / self.a / self.a * dt;
    }
    fn apply_k3<F: C2Fn>(&mut self, dt: f64, potential: &F) {
        let a = self.a;
        self.mom_a += -4.0 * potential.value(self.phi) * a * a * a * dt;
        self.mom_phi += -a * a * a * a * potential.value_d(self.phi) * dt;
    }
    fn apply_full_k_order2<F: C2Fn>(&mut self, delta_t: f64, kappa: f64, potential: &F) {
        self.apply_k1(delta_t / 2.0, kappa);
        self.apply_k2(delta_t / 2.0);
        self.apply_k3(delta_t, potential);
        self.apply_k2(delta_t / 2.0);
        self.apply_k1(delta_t / 2.0, kappa);
    }
    pub fn update<F: C2Fn>(&mut self, delta_t: f64, order: usize, kappa: f64, potential: &F) {
        if order == 2 {
            self.apply_full_k_order2(delta_t, kappa, potential);
        } else {
            let beta = 2.0 - pow(2.0, 1.0 / ((order - 1) as f64));
            self.update(delta_t / beta, order - 2, kappa, potential);
            self.update(delta_t * (1.0 - 2.0 / beta), order - 2, kappa, potential);
            self.update(delta_t / beta, order - 2, kappa, potential);
        }
    }
    pub fn scalar_effective_mass<F: C2Fn>(&self, kappa: f64, potential: &F) -> f64 {
        let pi_a = self.mom_a;
        let pi_phi = self.mom_phi;
        let a = self.a;
        let a2 = a * a;
        let a4 = a2 * a2;
        let a6 = a2 * a4;
        let v = potential.value(self.phi);
        let vd = potential.value_d(self.phi);
        let vdd = potential.value_dd(self.phi);
        pi_a * pi_a * kappa * kappa / a2 / 36.0
            + 5.0 * kappa * v * a2
            - 72.0 * v * v * a6 / pi_a / pi_a
            - 12.0 * pi_phi * a * vd / pi_a
            + a2 * vdd
    }
    pub fn hubble_constraint<F: C2Fn>(&self, potential: &F) -> f64 {
        let a = self.a;
        let a2 = a * a;
        let a4 = a2 * a2;
        let a6 = a2 * a4;
        self.mom_phi * self.mom_phi / 4.0 / a6
            -self.mom_a * self.mom_a / 24.0 / a4
            + 0.5 * potential.value(self.phi)
    }
    pub fn epsilon(&self, kappa: f64) -> f64 {
        let a2 = self.a * self.a;
        18.0 * self.mom_phi * self.mom_phi / self.mom_a / self.mom_a / kappa / a2
    }
    pub fn simulate<F, Cond, Step>(&self, kappa: f64, potential: &F, dn: f64, min_dt: f64, mut stop_condition: Cond, mut step_monitor: Step) -> Vec<Self> where
        F: C2Fn,
        Cond: FnMut(&Self) -> bool,
        Step: FnMut(&Self),
    {
        let mut state = *self;
        let mut ret = vec![state];
        while !stop_condition(&state) {
            state.dt = fmin(-dn * state.a / state.mom_a / kappa * 6.0, min_dt);
            state.update(state.dt, 4, kappa, potential);
            step_monitor(&state);
            ret.push(state);
        }
        ret
    }
}

pub trait PerturbationParams {
    fn effective_mass(&self, background: &BackgroundState) -> f64;
    fn sound_speed(&self, background: &BackgroundState) -> f64;
}

pub struct ScalarPerturbation2<'a, F> {
    pub kappa: f64,
    pub potential: &'a F,
}

impl<'a, F: C2Fn> PerturbationParams for ScalarPerturbation2<'a, F> {
    fn effective_mass(&self, background: &BackgroundState) -> f64 {
        background.scalar_effective_mass(self.kappa, self.potential)
    }

    fn sound_speed(&self, _background: &BackgroundState) -> f64 {
        1.0
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PerturbationState {
    pub u: Complex64,
    pub mom_u: Complex64,
}

impl PerturbationState {
    pub fn init<P: PerturbationParams>(k: f64, background: &BackgroundState, params: &P) -> Self {
        let cs = params.sound_speed(background);
        Self {
            u: Complex64::new(1.0 / sqrt(cs * 2.0 * k), 0.0),
            mom_u: Complex64::new(0.0, sqrt(cs * k / 2.0)),
        }
    }
    pub fn apply_k1<P: PerturbationParams>(&mut self, dt: f64, k: f64, background: &BackgroundState, params: &P) {
        // self.mom_u += self.u.conj() * (-(self.simulator.k * self.simulator.k + self.simulator.effective_mass()) * dt);
        let v = k * k * params.sound_speed(background) + params.effective_mass(background);
        self.mom_u += self.u.conj() * (-v * dt);
    }
    pub fn apply_k2(&mut self, dt: f64) {
        self.u += self.mom_u.conj() * dt;
    }
    pub fn apply_full_k_order2<P: PerturbationParams>(&mut self, dt: f64, k: f64, background: &BackgroundState, params: &P) {
        self.apply_k1(dt / 2.0, k, background, params);
        self.apply_k2(dt);
        self.apply_k1(dt / 2.0, k, background, params);
    }
    pub fn apply_full_k_order_n<P: PerturbationParams>(&mut self, dt: f64, order: usize, k: f64, background: &BackgroundState, params: &P) {
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

impl<'a, 'b, P> PerturbationSimulator<'a, 'b, P> where
    P: PerturbationParams,
{
    pub fn new(k: f64, background: &'a [BackgroundState], param: &'b P) -> Self {
        Self {k, background, param, cursor: 0, local_time: 0.0}
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
        let k = self.k * self.param.sound_speed(&self.background[index]).sqrt();
        k / self.param.effective_mass(&self.background[index]).abs().sqrt()
    }
    pub fn get_start_index(&self, max_n: Option<f64>, horizon_tolerance: f64) -> usize {
        let mut ret = 0usize;
        while ret < self.background.len() && max_n.map(|n|self.background[ret].a.ln() < n).unwrap_or(true) && self.horizon_ratio(ret) > horizon_tolerance {
            ret += 1;
        }
        ret
    }
    pub fn get_end_index(&self, min_n: Option<f64>, horizon_tolerance: f64) -> usize {
        let mut ret = 0usize;
        while ret < self.background.len() && (min_n.map(|n|self.background[ret].a.ln() < n).unwrap_or(false) || self.horizon_ratio(ret) > 1.0 / horizon_tolerance) {
            ret += 1;
        }
        ret
    }
    pub fn get_next_dt(&self, du: f64, background: &BackgroundState) -> f64 {
        let k = self.k;
        let potential = k * k * self.param.sound_speed(background) + self.param.effective_mass(background);
        fmin(du / potential.abs().sqrt(), background.dt)
    }
    pub fn run<F>(&mut self, n_range: (Option<f64>, Option<f64>), du: f64, order: usize, mut output_consumer: F) -> PerturbationState where
        F: FnMut(&BackgroundState, &PerturbationState)
    {
        let k = self.k;
        self.cursor = self.get_start_index(n_range.0, 1000.0);
        let end_i = self.get_end_index(n_range.1, 1000.0);
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

pub fn scan_spectrum<P>(background: &[BackgroundState], pert_param: &P, k_range: (f64, f64), n_range: (Option<f64>, Option<f64>), count: usize) -> Vec<(f64, BackgroundState, PerturbationState)> where
    P: PerturbationParams + Send + Sync,
{
    let done = AtomicUsize::new(0);
    (0..count).into_par_iter().map(|i| {
        let k = power_interp(k_range.0, k_range.1, (i as f64) / ((count - 1) as f64));
        let mut sim = PerturbationSimulator::new(k, background, pert_param);
        let pert_state = sim.run(n_range, 0.01, 2, |_, _|{});
        let done0 = done.fetch_add(1, Ordering::SeqCst) + 1;
        println!("[spectrum] ({}/{}) k = {}", done0, count, k);
        (k, sim.get_background(), pert_state)
    }).collect::<Vec<_>>()
}

pub struct InputData<'name, 'pot, F, P> {
    pub name: &'name str,
    pub kappa: f64,
    pub phi0: f64,
    pub a0: f64,
    pub potential: &'pot F,
    pub pert_param: P,
}

const BINCODE_CONFIG: Configuration = standard();

pub struct Context<'name, 'pot, 'dir, 'input, F, P> {
    input_data: &'input InputData<'name, 'pot, F, P>,
    plot_max_length: usize,
    out_dir: &'dir str,
    background_data: Option<Vec<BackgroundState>>,
}

impl<'name, 'pot, 'dir, 'input, F, P> Context<'name, 'pot, 'dir, 'input, F, P> where
    F: C2Fn,
    P: PerturbationParams,
{
    pub fn new(out_dir: &'dir str, plot_max_length: usize, input_data: &'input InputData<'name, 'pot, F, P>) -> Self {
        Self {
            input_data,
            plot_max_length,
            out_dir,
            background_data: None,
        }
    }
    fn background_data_file_name(&self) -> String {
        self.out_dir.to_owned() + "/" + self.input_data.name + "background.bincode"
    }
    fn load_input_data(&mut self) {
        if self.background_data.is_none() {
            self.background_data = Some({
                decode_from_std_read(&mut BufReader::new(File::open(self.background_data_file_name()).unwrap()), BINCODE_CONFIG).unwrap()
            });
        }
    }
    pub fn run_background(&mut self, dn: f64, min_dt: f64) {
        let kappa = self.input_data.kappa;
        let initial = BackgroundState::init(kappa, self.input_data.phi0, 0.0, self.input_data.a0, self.input_data.potential);
        let mut last_log_time = SystemTime::now();
        let result = initial.simulate(kappa, self.input_data.potential, dn, min_dt, |s|s.phi < 0.1, |s| {
            let now = SystemTime::now();
            if last_log_time.elapsed().map(|s|s.as_millis() > 100).unwrap_or(false) {
                last_log_time = now;
                println!("{:?}", s);
            }
        });
        {
            encode_into_std_write(&result, &mut BufWriter::new(File::create(self.background_data_file_name()).unwrap()), BINCODE_CONFIG).unwrap();
        }
        let result = limit_length(result, self.plot_max_length);
        {
            let mut plot = Plot::new();
            let mut efoldings = vec![];
            let mut phi = vec![];
            let mut epsilon = vec![];
            let mut hubble_constraint = vec![];
            let mut effective_mass = vec![];
            for elem in result {
                efoldings.push(elem.a.ln());
                phi.push(elem.phi);
                epsilon.push(elem.epsilon(kappa));
                hubble_constraint.push(elem.hubble_constraint(self.input_data.potential));
                effective_mass.push(-elem.scalar_effective_mass(kappa, self.input_data.potential));
            }
            plot.add_trace(Scatter::new(efoldings.clone(), phi).name("phi"));
            plot.add_trace(Scatter::new(efoldings.clone(), epsilon).name("epsilon").x_axis("x1").y_axis("y2"));
            plot.add_trace(Scatter::new(efoldings.clone(), effective_mass).name("m^2").x_axis("x1").y_axis("y3"));
            plot.add_trace(Scatter::new(efoldings.clone(), hubble_constraint).name("hubble constraint").x_axis("x1").y_axis("y4"));
            plot.set_layout(
                Layout::new()
                    .grid(
                        LayoutGrid::new()
                            .rows(4)
                            .columns(1)
                            .pattern(GridPattern::Coupled)
                    )
                    .y_axis2(Axis::new().type_(AxisType::Log))
                    .y_axis3(Axis::new().type_(AxisType::Log))
                    .y_axis4(Axis::new().type_(AxisType::Log))
                    .height(1400)
            );
            plot.write_html(self.out_dir.to_owned() + "/" + self.input_data.name + ".background.html");
        }
    }
    pub fn run_perturbation(&mut self, k: f64, n_range: (Option<f64>, Option<f64>)) {
        self.load_input_data();
        let background = self.background_data.as_ref().unwrap();
        let pert_param = ScalarPerturbation2 {
            kappa: self.input_data.kappa,
            potential: self.input_data.potential,
        };
        let mut sim = PerturbationSimulator::new(k, background, &pert_param);
        let output_selector = OutputSelector::default();
        let mut efolding = vec![];
        let mut u = vec![];
        let mut r = vec![];

        let mut last_n = 0.0;
        let mut last_log_time = SystemTime::now();
        sim.run(n_range, 0.01, 2, |b, u1| {
            let n = b.a.ln();
            if output_selector.test(last_n, n) {
                last_n = n;
                efolding.push(n);
                u.push(u1.u.abs());
                r.push(-u1.u.abs() / b.z(1.0));
            }
            if last_log_time.elapsed().unwrap().as_millis() > 100 {
                last_log_time = SystemTime::now();
                println!("N = {}, pert = {:?}", n, u1);
            }
        });
        {
            let mut plot = Plot::new();
            plot.add_trace(Scatter::new(efolding.clone(), u).name("u"));
            plot.add_trace(Scatter::new(efolding.clone(), r).name("r").x_axis("x1").y_axis("y2"));
            plot.set_layout(
                Layout::new()
                    .grid(
                        LayoutGrid::new()
                            .rows(2)
                            .columns(1)
                            .pattern(GridPattern::Coupled)
                    )
                    .y_axis(Axis::new().type_(AxisType::Log))
                    .y_axis2(Axis::new().type_(AxisType::Log))
                    .height(800)
            );
            plot.write_html(self.out_dir.to_owned() + "/" + self.input_data.name + ".perturbation.scalar.html");
        }
    }
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
    pub dphi_dt: Vec<f64>,
    pub scale_factor: Vec<f64>,
    pub hubble_constraint: Vec<f64>,
    pub hubble: Vec<f64>,
    pub epsilon: Vec<f64>,
}

#[derive(Default, Encode, Decode)]
pub struct BackgroundOutput {
    pub dt: Vec<f64>,
    pub effective_mass: Vec<f64>,
    pub tensor_effective_mass: Vec<f64>,
    pub scale_factor: Vec<f64>,
    pub z: Vec<f64>,
    pub phi: Vec<f64>,
    pub d_phi: Vec<f64>,
    pub kappa: f64,
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
    F1: C2Fn,
    F2: C2Fn,
    F3: C2Fn,
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
        self.mom_scale_factor = -sqrt((self.mom_phi * self.mom_phi + 2.0 * self.potential.value(self.phi) * a6) * 6.0 / self.kappa) / scale_factor;
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
        self.mom_scale_factor += -4.0 * self.potential.value(self.phi) * a * a * a * dt;
        self.mom_phi += -a * a * a * a * self.potential_d.value(self.phi) * dt;
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
        let a6 = a2 * a4;
        let v = self.potential.value(self.phi);
        let vd = self.potential_d.value(self.phi);
        let vdd = self.potential_dd.value(self.phi);
        pi_a * pi_a * self.kappa * self.kappa / a2 / 36.0
            + 5.0 * self.kappa * v * a2
            - 72.0 * v * v * a6 / pi_a / pi_a
            - 12.0 * pi_phi * a * vd / pi_a
            + a2 * vdd
    }
    pub fn tensor_effective_mass(&self) -> f64 {
        let pi_a = self.mom_scale_factor;
        let v = self.potential.value(self.phi);
        let a = self.scale_factor;
        let a2 = a * a;
        let kappa = self.kappa;
        pi_a * pi_a * kappa * kappa  / 36.0 / a2 - kappa * v * a2
    }
    pub fn run<P1, P2>(&mut self, mut dt_provider: P1, mut stop_predicate: P2) -> (BackgroundMeasurables, BackgroundOutput) where
        P1: FnMut(&Self) -> f64,
        P2: FnMut(&Self) -> bool,
    {
        let mut measurables = BackgroundMeasurables {
            ..Default::default()
        };
        let mut output = BackgroundOutput {
            kappa: self.kappa,
            ..Default::default()
        };
        println!("d_a = {}", -self.mom_scale_factor * self.kappa / 6.0);
        let mut steps = 1;
        let mut last_log_a = self.scale_factor;
        let mut time = 0.0;
        let mut last_log_time = SystemTime::now();
        while !stop_predicate(self) {
            let mass_eff = self.effective_mass();
            let a = self.scale_factor;
            let a2 = a * a;
            let a4 = a2 * a2;
            let a6 = a4 * a2;
            let d_a = -self.mom_scale_factor * self.kappa / 6.0;
            let d_phi = self.mom_phi / a2;
            // \dot{\phi}, NOT \phi' !!!
            let dphi_dt = d_phi / a;
            if self.scale_factor / last_log_a >= 1.01 {
                let hubble_contraint = self.mom_phi * self.mom_phi / 4.0 / a6
                    -self.mom_scale_factor * self.mom_scale_factor / 24.0 / a4
                    + 0.5 * self.potential.value(self.phi);
                let epsilon = 18.0 * self.mom_phi * self.mom_phi / self.mom_scale_factor / self.mom_scale_factor / self.kappa / a2;
                measurables.phi.push(self.phi);
                measurables.scale_factor.push(self.scale_factor);
                measurables.hubble_constraint.push(hubble_contraint);
                measurables.epsilon.push(epsilon);
                measurables.hubble.push(d_a / a2);
                measurables.dphi_dt.push(dphi_dt);
                last_log_a = self.scale_factor;
                if last_log_time.elapsed().map(|f|f.as_millis() >= 100).unwrap_or(true) {
                    println!("step = {}, time = {}, a = {}, phi = {}, hc = {}", steps, time, self.scale_factor, self.phi, hubble_contraint);
                    last_log_time = SystemTime::now();
                }
            }
            let dt = dt_provider(self);
            output.effective_mass.push(mass_eff);
            output.tensor_effective_mass.push(self.tensor_effective_mass());
            output.dt.push(dt);
            output.scale_factor.push(self.scale_factor);
            output.z.push(a * a * d_phi / d_a);
            output.phi.push(self.phi);
            output.d_phi.push(d_phi);
            self.step(dt);
            steps += 1;
            time += dt;
        }
        (measurables, output)
    }
}

pub trait PerturbationParameters {
    fn effective_mass(&self, sim: &ScalarPerturbationSimulator<'_, Self>) -> f64 where Self: Sized;
    fn sound_speed(&self, dim: &ScalarPerturbationSimulator<'_, Self>) -> f64 where Self: Sized;
    fn perturbation(&self, u: Complex64, sim: &ScalarPerturbationSimulator<'_, Self>) -> Complex64 where Self: Sized;
}

#[derive(Clone, Copy)]
pub struct ScalarPerturbation;
impl PerturbationParameters for ScalarPerturbation {
    fn perturbation(&self, u: Complex64, sim: &ScalarPerturbationSimulator<'_, Self>) -> Complex64 where Self: Sized {
        -u / sim.input_z()
    }

    fn effective_mass(&self, sim: &ScalarPerturbationSimulator<'_, Self>) -> f64 where Self: Sized {
        sim.effective_mass()
    }

    fn sound_speed(&self, dim: &ScalarPerturbationSimulator<'_, Self>) -> f64 where Self: Sized {
        10.
    }
}

pub struct ScalarPerturbationSimulator<'a, S> {
    pub input: &'a BackgroundOutput,
    pub k: f64,
    pub u: Complex64,
    pub mom_u: Complex64,
    mode: S,
    time_cursor: usize,
    local_time: f64,
    time_accumulator: f64,

    last_u: Complex64,
    last_u2: Complex64,
    last_potential: f64,
    last_dt: f64,
    last_dt2: f64,
    last_error: f64,
}

#[derive(Default)]
pub struct ScalarPerturbationOutput {
    pub k: f64,
    pub scale_factor: Vec<f64>,
    pub dt: Vec<f64>,
    pub time: Vec<f64>,
    pub u: Vec<Complex64>,
    pub mom_u: Vec<Complex64>,
    pub perturbation: Vec<Complex64>,
    pub potential: Vec<f64>,
    pub effective_mass: Vec<f64>,
    pub k2_mass: Vec<f64>,
    pub z: Vec<f64>,
    pub residual: Vec<f64>,
    pub calculated_potential: Vec<Complex64>,
    pub error: Vec<f64>,
}

#[derive(Debug, Clone, Copy)]
pub struct OutputSelector {
    pub dn: Option<f64>,
    pub start_n: Option<f64>,
    pub end_n: Option<f64>,
}

impl Default for OutputSelector {
    fn default() -> Self {
        Self { dn: Some(0.00001), start_n: Default::default(), end_n: Default::default() }
    }
}

impl OutputSelector {
    pub fn test(&self, last_n: f64, n: f64) -> bool {
        self.dn.map(|dn|n >= last_n + dn).unwrap_or(true) && self.start_n.map(|f|n >= f).unwrap_or(true) && self.end_n.map(|f|n <= f).unwrap_or(true)
    }
}

struct ScalarPerturbationUpdator<'a, 'b, S> {
    simulator: &'a ScalarPerturbationSimulator<'b, S>,
    u: Complex64,
    mom_u: Complex64,
}

impl<'a, 'b, S> ScalarPerturbationUpdator<'a, 'b, S> where
    S: PerturbationParameters,
{
    pub fn new(simulator: &'a ScalarPerturbationSimulator<'b, S>) -> Self {
        Self {
            simulator,
            u: simulator.u,
            mom_u: simulator.mom_u,
        }
    }
    pub fn apply_k1(&mut self, dt: f64) {
        self.mom_u += self.u.conj() * (-(self.simulator.k * self.simulator.k + self.simulator.effective_mass()) * dt);
        // self.mom_u += self.u.conj() * (-self.simulator.mode.potential(&self.simulator) * dt);
    }
    pub fn apply_k2(&mut self, dt: f64) {
        self.u += self.mom_u.conj() * dt;
    }
    pub fn apply_full_k_order2(&mut self, dt: f64) {
        self.apply_k1(dt / 2.0);
        self.apply_k2(dt);
        self.apply_k1(dt / 2.0);
    }
    pub fn apply_full_k_order_n(&mut self, dt: f64, order: usize) {
        if order == 1 {
            self.apply_k1(dt);
            self.apply_k2(dt);
        } else if order == 2 {
            self.apply_full_k_order2(dt);
        } else {
            let beta = 2.0 - pow(2.0, 1.0 / ((order - 1) as f64));
            self.apply_full_k_order_n(dt / beta, order - 2);
            self.apply_full_k_order_n(dt * (1.0 - 2.0 / beta), order - 2);
            self.apply_full_k_order_n(dt / beta, order - 2);
        }
    }
}

impl<'a, S> ScalarPerturbationSimulator<'a, S> where
    S: PerturbationParameters,
{
    pub fn new(input: &'a BackgroundOutput, mode: S, k: f64) -> Self {
        Self {
            mode,
            input,
            k,
            u: Complex64::new(1.0 / sqrt(2.0 * k), 0.0),
            mom_u: Complex64::new(0.0, sqrt(k / 2.0)),
            local_time: 0.0,
            time_cursor: 0,
            time_accumulator: 0.0,

            last_dt: 0.0,
            last_dt2: 0.0,
            last_potential: 0.0,
            last_u: 0.0.into(),
            last_u2: 0.0.into(),
            last_error: 0.0,
        }
    }
    pub fn scan_spectrum<F>(creator: F, k_range: (f64, f64), n_range: (f64, f64), count: usize, order: usize, du: f64) -> Vec<(f64, f64)> where
        F: Fn(f64) -> Self + Sync,
    {
        let done_count = AtomicUsize::new(0);
        (0..count).into_par_iter().map(|i|{
            let l = (i as f64) / ((count - 1) as f64);
            let k = pow(k_range.0, 1.0 - l) * pow(k_range.1, l);
            let mut sim = creator(k);
            let output = sim.run::<true>(Some(n_range.0), Some(n_range.1), order, du, OutputSelector::default());
            println!("[spectrum] ({}/{}) n = {}, k = {}, done", done_count.fetch_add(1, Ordering::Acquire) + 1, count, i, k);
            let r = output.perturbation.last().unwrap().abs();
            (k, k * k * k * r * r / 2.0 / PI / PI)
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
    pub fn potential(&self) -> f64 {
        let k = self.k;
        k * k * self.mode.sound_speed(self) + self.mode.effective_mass(self)
    }
    pub fn effective_mass(&self) -> f64 {
        self.time_interpolate(&self.input.effective_mass)
    }
    pub fn scale_factor(&self) -> f64 {
        self.time_interpolate(&self.input.scale_factor)
    }
    pub fn tensor_effective_mass(&self) -> f64 {
        self.time_interpolate(&self.input.tensor_effective_mass)
    }
    pub fn time_accumulator(&self) -> f64 {
        self.time_accumulator
    }
    pub fn input_z(&self) -> f64 {
        self.time_interpolate(&self.input.z)
    }
    fn move_to_start_time(&mut self, min_n: Option<f64>) {
        let k2 = self.k * self.k * self.mode.sound_speed(self);
        let mut i = 0;
        while min_n.map(|n|self.input.scale_factor[i].ln() <= n).unwrap_or(true) && k2 / self.input.effective_mass[i].abs() >= 1e3 && i < self.input.effective_mass.len() {
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
        // m1 + (m2 - m1) * t
        m1
    }
    pub fn run<const SILENCE: bool>(&mut self, start_n: Option<f64>, end_n: Option<f64>, order: usize, du: f64, output_selector: OutputSelector) -> ScalarPerturbationOutput {
        let k2 = self.k * self.k;
        self.move_to_start_time(start_n);
        if !SILENCE {
            println!("start time i = {}, k^2 / m^2 = {}, N = {}", self.time_cursor, k2 / self.effective_mass(), self.scale_factor().ln());
        }
        let mut output: ScalarPerturbationOutput = ScalarPerturbationOutput {
            k: self.k,
            ..Default::default()
        };
        let mut steps = 1;
        let mut last_display_time = SystemTime::now();
        let mut last_efolding = self.scale_factor().ln();
        while self.time_cursor < self.input.dt.len() - 1 && end_n.map(|e|self.scale_factor().ln() <= e).unwrap_or(self.input.effective_mass[self.time_cursor].abs() >= 1e-3) {
            let dt = fmin(du / (self.k * self.k + self.effective_mass()).abs().sqrt(), self.input.dt[self.time_cursor]);
            // let dt = self.input.dt[self.time_cursor];
            let efolding = log(self.scale_factor());
            let m2 = self.effective_mass();
            let potential = k2 + m2;
            if output_selector.dn.map(|d|efolding >= last_efolding + d).unwrap_or(true) && output_selector.start_n.map(|n|efolding >= n).unwrap_or(true) && output_selector.end_n.map(|n|efolding <= n).unwrap_or(true) {
                last_efolding = efolding;
                let z = self.time_interpolate(&self.input.z);
                output.u.push(self.u);
                output.scale_factor.push(self.scale_factor());
                output.dt.push(dt);
                output.time.push(self.time_accumulator);
                output.potential.push(potential);
                output.effective_mass.push(m2);
                output.mom_u.push(self.mom_u);
                output.z.push(z);
                output.k2_mass.push(k2 / m2.abs());
                output.perturbation.push(self.mode.perturbation(self.u, self));
                let u_dd = (self.last_u2 / self.last_dt2 + self.u / self.last_dt - (1.0 / self.last_dt + 1.0 / self.last_dt2) * self.last_u) / (self.last_dt2 + self.last_dt) * 2.0;
                let residual = u_dd + self.last_potential * self.last_u;
                output.residual.push(residual.abs() / self.last_u.abs() / k2);
                output.error.push(self.last_error);
                output.calculated_potential.push(-u_dd / self.last_u);
                if !SILENCE && last_display_time.elapsed().map(|f|f.as_millis() >= 100).unwrap_or(true) {
                    println!("step = {}, |u| = {}, mass = {}, k^2 / mass = {}, N = {}, dt = {}", steps, self.u.abs(), self.effective_mass(), self.k * self.k / self.effective_mass(), log(self.scale_factor()), dt);
                    last_display_time = SystemTime::now();
                }
            }
            self.last_dt2 = self.last_dt;
            self.last_dt = dt;
            self.last_u2 = self.last_u;
            self.last_u = self.u;
            self.last_potential = potential;
            self.last_error = self.step(dt, order);
            self.advance_time(dt);
            steps += 1;
        }
        output
    }
    pub fn step(&mut self, dt: f64, order: usize) -> f64 {
        let mut updater = ScalarPerturbationUpdator::new(self);
        updater.apply_full_k_order_n(dt, order);
        let u = updater.u;
        let mom_u = updater.mom_u;
        let mut higher_updater = ScalarPerturbationUpdator::new(self);
        higher_updater.apply_full_k_order_n(dt, order + 2);
        let err = higher_updater.u - u;
        self.u = u;
        self.mom_u = mom_u;
        err.abs()
    }
}
