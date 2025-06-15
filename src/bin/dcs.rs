use std::time::SystemTime;

use inflat::{
    background::{
        BINCODE_CONFIG, BackgroundFn, DefaultPerturbationInitializer, HamitonianSimulator,
        HorizonSelector, Kappa, PhiD, TwoFieldBackgroundInput, TwoFieldBackgroundState,
    },
    c2fn::{C2Fn, C2Fn2},
    models::StarobinskyPotential,
    util::{derivative_2, lazy_file, limit_length},
};
use libm::{cosh, fmin, sqrt, tanh};
use num_complex::ComplexFloat;
use plotly::{
    Layout, Plot, Scatter,
    common::{ExponentFormat, Pattern},
    layout::{Axis, AxisType, GridPattern, LayoutGrid},
};

struct FuncB {
    pub b1: f64,
    pub gamma: f64,
    pub phi_c: f64,
}

impl C2Fn<f64> for FuncB {
    type Output = f64;
    fn value(&self, phi: f64) -> f64 {
        self.b1 / 2.0 * (1.0 + tanh((phi - self.phi_c) / self.gamma))
    }

    fn value_d(&self, phi: f64) -> f64 {
        let s = 1.0 / cosh((phi - self.phi_c) / self.gamma);
        self.b1 / 2.0 / self.gamma * s * s
    }

    fn value_dd(&self, phi: f64) -> f64 {
        let s = 1.0 / cosh((phi - self.phi_c) / self.gamma);
        -self.b1 / self.gamma / self.gamma * s * s * tanh((phi - self.phi_c) / self.gamma)
    }
}

struct PotentialV<F> {
    pub m_chi: f64,
    pub u: F,
}

impl<F> C2Fn2<f64, f64> for PotentialV<F>
where
    F: C2Fn<f64, Output = f64>,
{
    type Ret = f64;

    fn value_00(&self, phi: f64, chi: f64) -> f64 {
        self.u.value(phi) + 0.5 * self.m_chi * self.m_chi * chi * chi
    }

    fn value_10(&self, phi: f64, _chi: f64) -> f64 {
        self.u.value_d(phi)
    }

    fn value_01(&self, _phi: f64, chi: f64) -> f64 {
        self.m_chi * self.m_chi * chi
    }

    fn value_11(&self, _phi: f64, _chi: f64) -> Self::Ret {
        0.0
    }

    fn value_20(&self, phi: f64, _chi: f64) -> Self::Ret {
        self.u.value_dd(phi)
    }

    fn value_02(&self, _phi: f64, _chi: f64) -> Self::Ret {
        self.m_chi * self.m_chi
    }
}

struct Params<F1, F2> {
    pub a: f64,
    pub phi: f64,
    pub chi: f64,
    pub v_chi: f64,
    pub alpha: f64,
    pub input: TwoFieldBackgroundInput<F1, F2>,
}

impl<F1, F2> Kappa for Params<F1, F2> {
    fn kappa(&self) -> f64 {
        self.input.kappa
    }
}

struct KineticCoef {
    pub lambda: f64,
}

impl<F1, F2> BackgroundFn<Params<F1, F2>, TwoFieldBackgroundState> for KineticCoef
where
    F1: C2Fn<f64, Output = f64>,
    F2: C2Fn2<f64, f64, Ret = f64>,
{
    type Output = f64;
    fn apply(
        &self,
        context: &Params<F1, F2>,
        state: &TwoFieldBackgroundState,
        k: f64,
    ) -> Self::Output {
        let a = state.a();
        let kappa = context.input.kappa;
        a * a * (k * context.alpha * kappa * self.lambda * state.v_chi(&context.input) + a)
            / 8.0
            / kappa
    }
}

struct PotentialCoef {
    pub lambda: f64,
}

impl<F1, F2> BackgroundFn<Params<F1, F2>, TwoFieldBackgroundState> for PotentialCoef
where
    F1: C2Fn<f64, Output = f64>,
    F2: C2Fn2<f64, f64, Ret = f64>,
{
    type Output = f64;
    fn apply(
        &self,
        context: &Params<F1, F2>,
        state: &TwoFieldBackgroundState,
        k: f64,
    ) -> Self::Output {
        let a = state.a();
        k * k / a / a - state.dcs_fa_potential(&context.input, k, context.alpha, self.lambda)
    }
}

struct Horizon {
    pub lambda: f64,
}
impl<F1, F2> BackgroundFn<Params<F1, F2>, TwoFieldBackgroundState> for Horizon
where
    F1: C2Fn<f64, Output = f64>,
    F2: C2Fn2<f64, f64, Ret = f64>,
{
    type Output = f64;
    fn apply(
        &self,
        context: &Params<F1, F2>,
        state: &TwoFieldBackgroundState,
        k: f64,
    ) -> Self::Output {
        state.dcs_horizon(&context.input, k, context.alpha, self.lambda)
    }
}

struct FaCoef {
    pub lambda: f64,
}

impl<F1, F2> BackgroundFn<Params<F1, F2>, TwoFieldBackgroundState> for FaCoef
where
    F1: C2Fn<f64, Output = f64>,
    F2: C2Fn2<f64, f64, Ret = f64>,
{
    type Output = f64;

    fn apply(
        &self,
        context: &Params<F1, F2>,
        state: &TwoFieldBackgroundState,
        k: f64,
    ) -> Self::Output {
        1.0 / state.dcs_fa(&context.input, k, context.alpha, self.lambda)
    }
}

impl<F1, F2> Params<F1, F2>
where
    F1: C2Fn<f64, Output = f64> + Send + Sync,
    F2: C2Fn2<f64, f64, Ret = f64> + Send + Sync,
{
    pub fn run(&self, out_dir: &str) -> anyhow::Result<()> {
        let max_length = 500000usize;
        let background = lazy_file(
            &format!("{}/background.bincode", out_dir),
            BINCODE_CONFIG,
            || {
                let initial = TwoFieldBackgroundState::init_slowroll(
                    self.a,
                    self.phi,
                    self.chi,
                    self.v_chi,
                    &self.input,
                );
                let mut last_log_time = SystemTime::now();
                initial.simulate(
                    &self.input,
                    0.1,
                    10.0,
                    |s| s.phi < 0.0,
                    |s, time| {
                        if last_log_time
                            .elapsed()
                            .map(|t| t.as_millis() >= 100)
                            .unwrap_or(false)
                        {
                            println!("[background]t = {}, {:?}", time, s);
                            last_log_time = SystemTime::now();
                        }
                    },
                )
            },
        )?;
        {
            let k = 1e1;
            let mut time = vec![];
            let mut t = 0.0;
            for state in &background {
                time.push(t * sqrt(1.048e-10));
                t += state.dt;
            }
            println!(
                "v_phi[0] = {}, v_a[0] = {}, t[-1] = {}",
                background[0].v_phi(),
                background[0].v_a(&self.input),
                t
            );
            time = limit_length(time, max_length);
            let mut efoldings = vec![];
            let mut phi = vec![];
            let mut chi = vec![];
            let mut v_chi = vec![];
            let mut vv_chi = vec![];
            let mut vvv_chi = vec![];
            let mut epsilon = vec![];
            let mut hubble_constraint = vec![];
            let mut hubble = vec![];
            let mut test_plot1 = vec![];
            for state in limit_length(background.clone(), max_length) {
                efoldings.push(state.a().ln());
                phi.push(state.phi);
                chi.push(state.chi);
                v_chi.push(state.v_chi(&self.input));
                vv_chi.push(state.vv_chi(&self.input));
                vvv_chi.push(state.vvv_chi(&self.input));
                epsilon.push(state.epsilon(&self.input).abs());
                hubble.push(state.v_a(&self.input) / state.a());
                hubble_constraint.push(state.hubble_constraint(&self.input).abs());
                test_plot1.push(state.dcs_fa_potential(&self.input, k, self.alpha, -1.0));
            }
            let mut plot = Plot::new();
            plot.add_trace(Scatter::new(efoldings.clone(), phi).name("phi"));
            plot.add_trace(
                Scatter::new(efoldings.clone(), chi)
                    .name("chi")
                    .y_axis("y2"),
            );
            plot.add_trace(
                Scatter::new(efoldings.clone(), v_chi)
                    .name("v_chi")
                    .y_axis("y3"),
            );
            plot.add_trace(
                Scatter::new(efoldings.clone(), vv_chi)
                    .name("v_chi")
                    .y_axis("y3"),
            );
            plot.add_trace(
                Scatter::new(efoldings.clone(), vvv_chi)
                    .name("v_chi")
                    .y_axis("y3"),
            );
            plot.add_trace(
                Scatter::new(efoldings.clone(), epsilon)
                    .name("epsilon")
                    .y_axis("y4"),
            );
            plot.add_trace(
                Scatter::new(efoldings.clone(), hubble_constraint)
                    .name("hubble_constraint")
                    .y_axis("y5"),
            );
            plot.add_trace(
                Scatter::new(efoldings.clone(), hubble)
                    .name("H")
                    .y_axis("y6"),
            );
            plot.add_trace(
                Scatter::new(efoldings.clone(), test_plot1)
                    .name("test1")
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
                    .x_axis(Axis::new().exponent_format(ExponentFormat::Power))
                    .y_axis(Axis::new().exponent_format(ExponentFormat::Power))
                    .y_axis2(Axis::new().exponent_format(ExponentFormat::Power))
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
                    .y_axis6(Axis::new().exponent_format(ExponentFormat::Power))
                    .y_axis7(
                        Axis::new()
                            .type_(AxisType::Log)
                            .exponent_format(ExponentFormat::Power),
                    )
                    .height(1200),
            );
            plot.write_html(&format!("{}/background.plot.html", out_dir));
        }
        let pert = HamitonianSimulator::new(
            self,
            background.len(),
            background.as_slice(),
            DefaultPerturbationInitializer,
            PotentialCoef { lambda: -1.0 },
            HorizonSelector::new(1e3),
            FaCoef { lambda: -1.0 },
        );
        {
            let mut efoldings = vec![];
            let mut phi = vec![];
            let mut phi_im = vec![];
            let mut em = vec![];
            let mut last_log_time = SystemTime::now();
            pert.run(1e9, 0.5, |pert, b, s, h, potential, dt| {
                efoldings.push(b.a().ln());
                phi.push(h.abs());
                phi_im.push(s.x.im);
                em.push(potential);
                if last_log_time
                    .elapsed()
                    .map(|f| f.as_millis() >= 100)
                    .unwrap_or(false)
                {
                    last_log_time = SystemTime::now();
                    println!("[pert]N = {}, dt = {}, {:?}", b.a().ln(), dt, s);
                }
            });
            efoldings = limit_length(efoldings, max_length);
            phi = limit_length(phi, max_length);
            em = limit_length(em, max_length);
            let mut plot = Plot::new();
            plot.add_trace(Scatter::new(efoldings.clone(), phi).name("re(phi)"));
            // plot.add_trace(Scatter::new(efoldings.clone(), phi_im).name("im(phi)"));
            plot.add_trace(Scatter::new(efoldings.clone(), em).y_axis("y2"));
            plot.set_layout(
                Layout::new()
                    .grid(
                        LayoutGrid::new()
                            .rows(2)
                            .columns(1)
                            .pattern(GridPattern::Coupled),
                    )
                    .y_axis(
                        Axis::new()
                            .type_(AxisType::Log)
                            .exponent_format(ExponentFormat::Power),
                    )
                    .y_axis2(Axis::new().exponent_format(ExponentFormat::Power))
                    .height(800),
            );
            plot.write_html(&format!("{}/perturbation.html", out_dir));
        }
        {
            let spectrum = lazy_file(
                &format!("{}/spectrum.bincode", out_dir),
                BINCODE_CONFIG,
                || pert.spectrum((1.0, 1e11), 100, 0.5),
            )?;
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
                    ),
            );
            plot.write_html(&format!("{}/spectrum.html", out_dir));
        }
        Ok(())
    }
}

pub fn main() {
    let set1 = Params {
        a: 1.0,
        phi: 5.6,
        chi: 1e-5,
        v_chi: 0.0,
        alpha: 2.9e3,
        input: TwoFieldBackgroundInput {
            kappa: 1.0,
            b: FuncB {
                b1: 12.0,
                gamma: 1e-2,
                phi_c: 4.94,
            },
            v: {
                let v0 = 1.048e-10;
                PotentialV {
                    m_chi: sqrt(4e6 * v0),
                    u: StarobinskyPotential {
                        v0,
                        phi0: sqrt(2.0 / 3.0),
                    },
                }
            },
        },
    };
    set1.run("out/dcs.set1").unwrap();
}
