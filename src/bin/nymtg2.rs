use std::time::Duration;

use inflat::{
    background::{
        BINCODE_CONFIG, CubicScaleFactor, DefaultPerturbationInitializer, HamitonianSimulator,
        HorizonSelector, Kappa, NymtgTensorPerturbationPotential, PhiD, ScaleFactorD,
        TwoFieldBackgroundInput, TwoFieldBackgroundInputProvider, TwoFieldBackgroundState,
    },
    c2fn::{C2Fn, Plus2},
    models::{LinearSinePotential, QuadraticPotential, StarobinskyPotential, ZeroFn},
    util::{ENERGY_SPECTRUM_EVAL_FACTOR, RateLimiter, lazy_file, limit_length},
};
use libm::sqrt;
use plotly::{
    Layout, Plot, Scatter,
    common::ExponentFormat,
    layout::{Axis, AxisType, LayoutGrid},
};

struct Params<V, U> {
    pub a0: f64,
    pub phi0: f64,
    pub varphi0: f64,
    pub v_varphi0: f64,
    pub v0: f64,
    pub alpha: f64,
    pub input: TwoFieldBackgroundInput<ZeroFn<f64>, Plus2<V, U, f64>>,
}

impl<V, U> Params<V, U> {
    pub fn run(&self, out_dir: &str) -> anyhow::Result<()>
    where
        V: C2Fn<f64, Output = f64> + Send + Sync,
        U: C2Fn<f64, Output = f64> + Send + Sync,
    {
        let max_length = 5000000;
        let mut rate_limiter = RateLimiter::new(Duration::from_millis(100));
        let background = lazy_file(
            &format!("{}/background.bincode", out_dir),
            BINCODE_CONFIG,
            || {
                let state = TwoFieldBackgroundState::init_slowroll(
                    self.a0,
                    self.phi0,
                    self.varphi0,
                    self.v_varphi0,
                    &self.input,
                );
                state.simulate(
                    &self.input,
                    0.1,
                    10.0,
                    |s| s.epsilon(&self.input) > 1.0,
                    |state, _time| {
                        rate_limiter.run(|| {
                            println!("[background] N = {}, state = {:?}", state.a().ln(), state);
                        });
                    },
                )
            },
        )?;
        {
            let mut efolding = vec![];
            let mut phi = vec![];
            let mut v_phi = vec![];
            let mut chi = vec![];
            let mut epsilon = vec![];
            for state in limit_length(background.clone(), max_length) {
                efolding.push(state.a().ln());
                phi.push(state.phi);
                v_phi.push(state.v_phi().abs() / self.v0.sqrt());
                chi.push(state.chi);
                epsilon.push(state.epsilon(&self.input));
            }
            let mut plot = Plot::new();
            plot.add_trace(Scatter::new(efolding.clone(), phi).name("phi"));
            plot.add_trace(
                Scatter::new(efolding.clone(), v_phi)
                    .name("v_phi")
                    .y_axis("y2"),
            );
            plot.add_trace(Scatter::new(efolding.clone(), chi).name("chi").y_axis("y3"));
            plot.add_trace(
                Scatter::new(efolding.clone(), epsilon)
                    .name("epsilon")
                    .y_axis("y4"),
            );
            plot.set_layout(
                Layout::new()
                    .grid(LayoutGrid::new().rows(4).columns(1))
                    .x_axis(Axis::new().exponent_format(ExponentFormat::Power))
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
                    .height(1000),
            );
            plot.write_html(&format!("{}/background.html", out_dir));
        }
        let pert = HamitonianSimulator::new(
            self,
            background.len(),
            &background,
            DefaultPerturbationInitializer,
            NymtgTensorPerturbationPotential {
                lambda: 1.0,
                alpha: self.alpha,
            },
            HorizonSelector::new(1e3),
            CubicScaleFactor,
        );
        {
            let k_coef = background[0].mom_unit_coef_hz(self.input.kappa, 0.05);
            let spectrum = lazy_file(
                &format!("{}/spectrum.bincode", out_dir),
                BINCODE_CONFIG,
                || pert.spectrum((1.0, 1e4), 1000, 0.1),
            )?;
            let mut plot = Plot::new();
            plot.add_trace(Scatter::new(
                spectrum.iter().map(|f| f.0 * k_coef).collect(),
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

impl<V, U> Kappa for Params<V, U> {
    fn kappa(&self) -> f64 {
        self.input.kappa
    }
}

impl<V, U> TwoFieldBackgroundInputProvider for Params<V, U> {
    type F1 = ZeroFn<f64>;

    type F2 = Plus2<V, U, f64>;

    fn input(&self) -> &TwoFieldBackgroundInput<Self::F1, Self::F2> {
        &self.input
    }
}

pub fn main() {
    let v0 = 9.75e-11;
    let sigma = 0.0002;
    let m = sqrt(0.16 * v0);
    let beta = 0.93;
    let lambda4 = beta * sigma * sigma * m * m;
    let potential_phi = QuadraticPotential {
        mass: sqrt(0.16 * v0),
    }
    .plus(LinearSinePotential {
        coef: lambda4 / sigma,
        omega: 1.0 / sigma,
    });
    let potential_varphi = StarobinskyPotential {
        v0,
        phi0: sqrt(2.0 / 3.0),
    };
    let params = Params {
        a0: 1.0,
        v0,
        phi0: 6.612e-4,
        varphi0: 5.42,
        v_varphi0: 0.0,
        alpha: 1.33e5,
        input: TwoFieldBackgroundInput {
            kappa: 1.0,
            b: ZeroFn::default(),
            v: potential_phi.plus2(potential_varphi),
        },
    };
    params.run("out/nymtg2.set1/").unwrap();
}
