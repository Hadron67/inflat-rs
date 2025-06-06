use std::time::SystemTime;

use inflat::{
    background::{
        BackgroundFn, HamitonianSimulator, TwoFieldBackgroundInput, TwoFieldBackgroundState, BINCODE_CONFIG
    },
    c2fn::{C2Fn, C2Fn2},
    models::StarobinskyPotential,
    util::{derivative_2, lazy_file, limit_length},
};
use libm::{cosh, sqrt, tanh};
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
    F: C2Fn<f64>,
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

struct PertMass {
    pub lambda: f64,
    pub alpha: f64,
}

impl<F1, F2> BackgroundFn<Params<F1, F2>, TwoFieldBackgroundState> for PertMass
where
    F1: C2Fn<f64>,
    F2: C2Fn2<f64, f64, Ret = f64>,
{
    fn apply(&self, context: &Params<F1, F2>, state: &TwoFieldBackgroundState, k: f64) -> f64 {
        1.0 + k * self.alpha * context.input.kappa * self.lambda * state.v_chi(&context.input)
            / state.a
            / state.a
    }
}

struct SubHorizon {
    pub alpha: f64,
    pub lambda: f64,
}

impl<F1, F2> BackgroundFn<Params<F1, F2>, TwoFieldBackgroundState> for SubHorizon
where
    F1: C2Fn<f64>,
    F2: C2Fn2<f64, f64, Ret = f64>,
{
    fn apply(&self, context: &Params<F1, F2>, state: &TwoFieldBackgroundState, k: f64) -> f64 {
        state.intermediate_potential(&context.input, k, self.alpha, self.lambda)
    }
}

struct SuperHorizon;
impl<F1, F2> BackgroundFn<Params<F1, F2>, TwoFieldBackgroundState> for SuperHorizon
where
    F1: C2Fn<f64>,
    F2: C2Fn2<f64, f64, Ret = f64>,
{
    fn apply(&self, context: &Params<F1, F2>, state: &TwoFieldBackgroundState, k: f64) -> f64 {
        -state.vv_a(&context.input) / state.a
    }
}

impl<F1, F2> Params<F1, F2>
where
    F1: C2Fn<f64>,
    F2: C2Fn2<f64, f64, Ret = f64>,
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
                    0.00001,
                    0.1,
                    |s| s.phi < 0.0,
                    |s| {
                        if last_log_time
                            .elapsed()
                            .map(|t| t.as_millis() >= 100)
                            .unwrap_or(false)
                        {
                            println!("[background] {:?}", s);
                            last_log_time = SystemTime::now();
                        }
                    },
                )
            },
        )?;
        {
            let k = 1e11;
            let mut plot = Plot::new();
            let mut efoldings = vec![];
            let mut phi = vec![];
            let mut chi = vec![];
            let mut epsilon = vec![];
            let mut hubble_constraint = vec![];
            let mut test_plot1 = vec![];
            let mut test_plot2 = vec![];
            let mut vv_chi = vec![];
            vv_chi.push(0.0);
            for i in 1..background.len() - 1 {
                vv_chi.push(derivative_2(background[i - 1].dt, background[i].dt, background[i - 1].chi, background[i].chi, background[i + 1].chi));
            }
            vv_chi.push(0.0);
            vv_chi = limit_length(vv_chi, max_length);
            for state in limit_length(background.clone(), max_length) {
                efoldings.push(state.a.ln());
                phi.push(state.phi);
                chi.push(state.chi);
                epsilon.push(state.epsilon(&self.input).abs());
                hubble_constraint.push(state.hubble_constraint(&self.input).abs());
                test_plot1.push(
                    state.normalized_potential(&self.input, k, self.alpha, 1.0)
                );
                test_plot2.push(state.vv_a(&self.input) / state.a);
            }
            println!("m^2[-1] = {}", test_plot1.last().unwrap());
            plot.add_trace(Scatter::new(efoldings.clone(), phi).name("phi"));
            plot.add_trace(
                Scatter::new(efoldings.clone(), chi)
                    .name("chi")
                    .y_axis("y2"),
            );
            plot.add_trace(
                Scatter::new(efoldings.clone(), epsilon)
                    .name("epsilon")
                    .y_axis("y3"),
            );
            plot.add_trace(
                Scatter::new(efoldings.clone(), hubble_constraint)
                    .name("hubble_constraint")
                    .y_axis("y4"),
            );
            plot.add_trace(
                Scatter::new(efoldings.clone(), test_plot1)
                    .name("test1")
                    .y_axis("y5"),
            );
            plot.add_trace(
                Scatter::new(efoldings.clone(), test_plot2)
                    .name("test2")
                    .y_axis("y5"),
            );
            plot.set_layout(
                Layout::new()
                    .grid(
                        LayoutGrid::new()
                            .rows(6)
                            .columns(1)
                            .pattern(GridPattern::Coupled),
                    )
                    .x_axis(Axis::new().exponent_format(ExponentFormat::Power))
                    .y_axis(Axis::new().exponent_format(ExponentFormat::Power))
                    .y_axis2(Axis::new().exponent_format(ExponentFormat::Power))
                    .y_axis3(
                        Axis::new()
                            .type_(AxisType::Log)
                            .exponent_format(ExponentFormat::Power),
                    )
                    .y_axis4(
                        Axis::new()
                            .type_(AxisType::Log)
                            .exponent_format(ExponentFormat::Power),
                    )
                    .y_axis5(Axis::new().type_(AxisType::Log).exponent_format(ExponentFormat::Power))
                    .y_axis6(Axis::new().exponent_format(ExponentFormat::Power))
                    .height(1200),
            );
            plot.write_html(&format!("{}/background.plot.html", out_dir));
        }
        let mut pert = HamitonianSimulator::new(
            self,
            background.len(),
            background.as_slice(),
            PertMass {
                lambda: -1.0,
                alpha: self.alpha,
            },
            SubHorizon {
                lambda: -1.0,
                alpha: self.alpha,
            },
            SuperHorizon,
        );
        {
            let mut efoldings = vec![];
            let mut phi = vec![];
            let mut em = vec![];
            let mut last_log_time = SystemTime::now();
            pert.run(1e12, 0.01, 1e3, |pert, b, s, potential, dt| {
                efoldings.push(b.a.ln());
                phi.push(s.x.re);
                em.push(potential);
                if last_log_time
                    .elapsed()
                    .map(|f| f.as_millis() >= 100)
                    .unwrap_or(false)
                {
                    last_log_time = SystemTime::now();
                    println!("[pert]N = {}, dt = {}, {:?}", b.a.ln(), dt, s);
                }
            });
            efoldings = limit_length(efoldings, max_length);
            phi = limit_length(phi, max_length);
            em = limit_length(em, max_length);
            let mut plot = Plot::new();
            plot.add_trace(Scatter::new(efoldings.clone(), phi));
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
                            .exponent_format(ExponentFormat::Power),
                    )
                    .y_axis2(Axis::new().exponent_format(ExponentFormat::Power))
                    .height(800),
            );
            plot.write_html(&format!("{}/perturbation.html", out_dir));
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
        alpha: 0.029,
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
                        phi0: sqrt(3.0 / 2.0),
                    },
                }
            },
        },
    };
    set1.run("out/dcs.set1").unwrap();
}
