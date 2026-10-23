use std::{fs::create_dir_all, time::Duration};

use inflat::{
    background::{
        BINCODE_CONFIG, BackgroundState, BackgroundStateInput, BackgroundStateInputProvider,
        DefaultPerturbationInitializer, HamitonianSimulator, HorizonSelectorWithExlusion, Kappa,
        ScalarPerturbationFactor, ScalarPerturbationPotential, ScaleFactor, ZPotential,
    },
    c2fn::{C2Fn, Plus},
    models::{ParametricResonanceParams, StarobinskyPotential, TruncSinePotential},
    util::{ParamRange, RateLimiter, lazy_file, limit_length},
};
use libm::sqrt;
use num_complex::ComplexFloat;
use plotly::{
    Layout, Plot, Scatter,
    common::ExponentFormat,
    layout::{Axis, AxisType, LayoutGrid},
};

struct Params<F> {
    pub a0: f64,
    pub phi0: f64,
    pub input: BackgroundStateInput<F>,
    pub max_dt: f64,
    pub resonance_range: (f64, f64),
    pub spectrum_range: ParamRange<f64>,
}

impl<F> BackgroundStateInputProvider for Params<F> {
    type F = F;

    fn input(&self) -> &BackgroundStateInput<Self::F> {
        &self.input
    }
}

impl<F> Kappa for Params<F> {
    fn kappa(&self) -> f64 {
        self.input.kappa
    }
}

impl<F> Params<F>
where
    F: C2Fn<f64, Output = f64> + Send + Sync,
{
    pub fn run(&self, out_dir: &str) -> anyhow::Result<()> {
        create_dir_all(out_dir)?;
        let background = lazy_file(
            &format!("{}/background.bincode", out_dir),
            BINCODE_CONFIG,
            || {
                let state = BackgroundState::init_slowroll(self.a0, self.phi0, &self.input);
                let mut rate_limiter = RateLimiter::new(Duration::from_millis(100));
                state.simulate(
                    &self.input,
                    0.1,
                    self.max_dt,
                    4,
                    |s| s.epsilon(&self.input) > 1.0,
                    |s| {
                        rate_limiter.run(|| {
                            println!("[background] {:?}", &s);
                        });
                    },
                )
            },
        )?;
        let max_length = 50000usize;
        {
            let mut efoldings = vec![];
            let mut phi = vec![];
            let mut zdd_z = vec![];
            let mut epsilon = vec![];
            let mut hubble = vec![];
            for state in limit_length(&background, max_length) {
                efoldings.push(state.scale_factor(&self.input).ln());
                phi.push(state.phi);
                epsilon.push(state.epsilon(&self.input));
                zdd_z.push(state.z_potential(&self.input));
                hubble.push(state.v_a(&self.input) / state.scale_factor(&self.input));
            }
            let mut plot = Plot::new();
            plot.add_trace(Scatter::new(efoldings.clone(), phi).name("phi"));
            plot.add_trace(
                Scatter::new(efoldings.clone(), zdd_z)
                    .name("phi")
                    .y_axis("y2"),
            );
            plot.add_trace(
                Scatter::new(efoldings.clone(), hubble)
                    .name("hubble")
                    .y_axis("y3"),
            );
            plot.add_trace(
                Scatter::new(efoldings.clone(), epsilon)
                    .name("epsilon")
                    .y_axis("y4"),
            );
            plot.set_layout(
                Layout::new()
                    .grid(LayoutGrid::new().rows(4).columns(1))
                    .x_axis(Axis::new().exponent_format(ExponentFormat::Power))
                    .y_axis2(Axis::new().exponent_format(ExponentFormat::Power))
                    .y_axis3(Axis::new().exponent_format(ExponentFormat::Power))
                    .y_axis4(
                        Axis::new()
                            .type_(AxisType::Log)
                            .exponent_format(ExponentFormat::Power),
                    )
                    .height(800),
            );
            plot.write_html(&format!("{}/background.html", out_dir));
        }
        let pert = HamitonianSimulator::new(
            self,
            background.len(),
            &background,
            DefaultPerturbationInitializer,
            ScalarPerturbationPotential,
            HorizonSelectorWithExlusion::new(1e3, self.resonance_range),
            ScalarPerturbationFactor,
        );
        {
            let mut efolding = vec![];
            let mut phi = vec![];
            let _ = pert.run(1e6, 0.01, |_, b, _s, phi0, _, _| {
                phi.push(phi0.abs());
                efolding.push(b.scale_factor(&self.input).ln());
            });
            let mut plot = Plot::new();
            plot.add_trace(Scatter::new(efolding, phi));
            plot.set_layout(
                Layout::new()
                    .x_axis(Axis::new().exponent_format(ExponentFormat::Power))
                    .y_axis(
                        Axis::new()
                            .type_(AxisType::Log)
                            .exponent_format(ExponentFormat::Power),
                    ),
            );
            plot.write_html(&format!("{}/perturbation.html", out_dir));
        }
        {
            let spectrum = pert.spectrum_with_cache(
                &format!("{}/spectrum.bincode", out_dir),
                self.spectrum_range,
                0.01,
                false,
            )?;
            let mut plot = Plot::new();
            plot.add_trace(Scatter::new(
                self.spectrum_range.as_logspace().collect(),
                spectrum,
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

fn make_parametric_resonance_potential(
    lambda: f64,
    phi_s: f64,
    phi_e: f64,
    phi_star: f64,
    xi: f64,
) -> Plus<StarobinskyPotential, TruncSinePotential, f64> {
    let v1 = StarobinskyPotential {
        v0: lambda * lambda * lambda * lambda,
        phi0: sqrt(2.0 / 3.0),
    };
    let v2 = TruncSinePotential {
        amp: xi,
        omega: 1.0 / phi_star,
        begin: phi_e,
        end: phi_s,
    };
    v1.plus(v2)
}

fn main() {
    let set1 = Params {
        a0: 1.0,
        phi0: 5.45,
        max_dt: 10.0,
        input: BackgroundStateInput {
            kappa: 1.0,
            potential: make_parametric_resonance_potential(0.0032, 4.9878, 4.9731, 8e-6, 1.7e-15),
        },
        resonance_range: (20.01, 20.54),
        spectrum_range: ParamRange::new(1e2, 1e11, 1000),
    };
    set1.run("out/parametric-resonance.set1").unwrap();
    let _sets = [
        (
            "parameter-resonance.set1",
            ParametricResonanceParams {
                lambda: 0.0032,
                phi0: 5.45,
                phi_s: 4.9878,
                phi_e: 4.9731,
                phi_star: 8e-6,
                xi: 1.7e-15,
            },
        ),
        (
            "parameter-resonance.set1.no-pert",
            ParametricResonanceParams {
                lambda: 0.0032,
                phi0: 5.1,
                phi_s: 4.9878,
                phi_e: 4.9731,
                phi_star: 8e-6,
                xi: 0.0,
            },
        ),
        (
            "parameter-resonance.set2",
            ParametricResonanceParams {
                lambda: 0.0032,
                phi0: 5.5,
                phi_s: 5.2118,
                phi_e: 5.2088,
                phi_star: 6.64e-6,
                xi: 1.23e-15,
            },
        ),
        (
            "parameter-resonance.set2.no-pert",
            ParametricResonanceParams {
                lambda: 0.0032,
                phi0: 5.5,
                phi_s: 5.2118,
                phi_e: 5.2088,
                phi_star: 6.64e-6,
                xi: 0.0,
            },
        ),
    ];
}
