use std::{f64::consts::PI, fs::create_dir_all, time::Duration};

use anyhow::Ok;
use bincode::{Decode, Encode};
use inflat::{
    background::{
        BINCODE_CONFIG, DefaultPerturbationInitializer, HamitonianSimulator, HorizonSelector,
    },
    c2fn::C2Fn,
    gauss_bonnet::{
        GaussBonnetBInput, GaussBonnetBackgroundState, GaussBonnetField, GaussBonnetFieldSimulator,
        GaussBonnetScalarPerturbationCoef, GaussBonnetScalarPerturbationPotential,
    },
    lat::{BoxLattice, Lattice, LatticeParam},
    models::TanhPotential,
    scalar::spectrum_with_scratch,
    util::{Hms, ParamRange, RateLimiter, TimeEstimator, VecN, lazy_file, limit_length},
};
use libm::{cos, sin, sqrt};
use num_complex::ComplexFloat;
use plotly::{
    Layout, Plot, Scatter,
    common::ExponentFormat,
    layout::{Axis, AxisType, LayoutGrid},
};

struct NaturalInflationPotential {
    pub lambda4: f64,
    pub f: f64,
}

impl C2Fn<f64> for NaturalInflationPotential {
    type Output = f64;

    fn value(&self, phi: f64) -> Self::Output {
        self.lambda4 * (1.0 + cos(phi / self.f))
    }

    fn value_d(&self, phi: f64) -> Self::Output {
        -self.lambda4 / self.f * sin(phi / self.f)
    }

    fn value_dd(&self, phi: f64) -> Self::Output {
        -self.lambda4 / self.f / self.f * cos(phi / self.f)
    }
}

struct LatticeSetting {
    pub name: String,
    pub starting_k: f64,
    pub dt: f64,
    pub lattice_size: usize,
    pub horizon_tolerance: f64,
    pub spectrum_count: usize,
}

#[derive(Debug, Clone, Copy, Encode, Decode)]
struct LatticeMeasurables {
    pub a: f64,
    pub v_a: f64,
    pub phi: f64,
    pub v_phi: f64,
    pub metric_perts: (f64, f64),
}

#[derive(Encode, Decode)]
struct LatticeOutputData {
    pub measurables: Vec<LatticeMeasurables>,
    pub spectrum_k: Vec<f64>,
    pub spectrums: Vec<(f64, Vec<f64>)>,
}

impl LatticeSetting {
    pub fn run<V, Xi>(
        &self,
        out_dir: &str,
        background: &[GaussBonnetBackgroundState],
        input: &GaussBonnetBInput<V, Xi>,
    ) -> anyhow::Result<()>
    where
        V: C2Fn<f64, Output = f64> + Sync + Send,
        Xi: C2Fn<f64, Output = f64> + Sync + Send,
    {
        let data = lazy_file(
            &format!("{}/lattice.{}.mearables.bincode", out_dir, &self.name),
            BINCODE_CONFIG,
            || {
                let starting_horizon = self.starting_k / self.horizon_tolerance;
                let end_k = self.starting_k * sqrt(3.0) / PI * (self.lattice_size as f64);
                let end_horizon = end_k * self.horizon_tolerance;
                let dx = 2.0 * PI / self.starting_k;
                let lattice = LatticeParam {
                    size: VecN::new([self.lattice_size; 3]),
                    spacing: VecN::new([dx; 3]),
                };
                println!("[lattice] dx = {}", dx);
                let start_state = background
                    .iter()
                    .find(|state| state.v_a >= starting_horizon)
                    .unwrap();
                let end_state = background
                    .iter()
                    .find(|state| state.v_a >= end_horizon)
                    .unwrap();
                let n_range = start_state.a.ln()..end_state.a.ln();
                println!("[lattice] N = {:?}", n_range);
                let mut simulator = GaussBonnetFieldSimulator::new(&lattice, input, {
                    let mut lattice_state = GaussBonnetField::zero(lattice.size);
                    lattice_state.init(start_state.a, start_state.phi, start_state.v_phi, input);
                    lattice_state.populate_noise(&mut random::default(1), input, &lattice);
                    lattice_state
                });
                let mut spectrum_scratch = BoxLattice::zeros(lattice.size);
                let initial_spectrum =
                    spectrum_with_scratch(&simulator.field.phi, &lattice, &mut spectrum_scratch);
                let spectrum_k = initial_spectrum.iter().map(|f| f.0).collect();
                let mut spectrums = vec![(
                    start_state.a.ln(),
                    initial_spectrum.iter().map(|f| f.1).collect(),
                )];
                let spectrum_delta_n = (n_range.end - n_range.start) / (self.spectrum_count as f64);
                let mut next_spectrum_n = start_state.a.ln() + spectrum_delta_n;
                let mut measurables = vec![];
                let mut rate_limiter = RateLimiter::new(Duration::from_millis(100));
                let mut time_estimator = TimeEstimator::new(n_range.clone(), 100);
                while simulator.field.a.ln() < n_range.end {
                    simulator.update(self.dt);
                    time_estimator.update(simulator.field.a.ln());
                    let state = LatticeMeasurables {
                        a: simulator.field.a,
                        v_a: simulator.field.v_a,
                        phi: simulator.field.phi.average(),
                        v_phi: simulator.field.v_phi.average(),
                        metric_perts: simulator.field.metric_perturbations(input, &lattice),
                    };
                    rate_limiter.run(|| {
                        println!(
                            "[lattice] eta remaining = {}, step = {}, measurables = {:?}",
                            Hms::from_secs(time_estimator.remaining_secs()),
                            measurables.len(),
                            &state
                        )
                    });
                    measurables.push(state);
                    if simulator.field.a.ln() >= next_spectrum_n {
                        next_spectrum_n += spectrum_delta_n;
                        let spec = spectrum_with_scratch(
                            &simulator.field.phi,
                            &lattice,
                            &mut spectrum_scratch,
                        );
                        spectrums
                            .push((simulator.field.a.ln(), spec.iter().map(|f| f.1).collect()));
                    }
                }
                LatticeOutputData {
                    measurables,
                    spectrum_k,
                    spectrums,
                }
            },
        )?;
        {
            let mut efolding = vec![];
            let mut phi = vec![];
            let mut v_phi = vec![];
            let mut hubble = vec![];
            let mut pert_a = vec![];
            let mut pert_b = vec![];
            for state in limit_length(data.measurables, 500000) {
                efolding.push(state.a.ln());
                phi.push(state.phi);
                v_phi.push(state.v_phi);
                hubble.push(state.v_a / state.a);
                pert_a.push(state.metric_perts.0.abs());
                pert_b.push(state.metric_perts.1.abs());
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
            plot.add_trace(
                Scatter::new(efolding.clone(), pert_a)
                    .name("|A|")
                    .y_axis("y4"),
            );
            plot.add_trace(
                Scatter::new(efolding.clone(), pert_b)
                    .name("|\\Box B / a^2|")
                    .y_axis("y5"),
            );
            plot.set_layout(
                Layout::new()
                    .grid(LayoutGrid::new().rows(5).columns(1))
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
                    .height(1200),
            );
            plot.write_html(&format!(
                "{}/lattice.{}.background.html",
                out_dir, &self.name,
            ));
        }
        {
            let mut plot = Plot::new();
            for (n, spec) in &data.spectrums {
                plot.add_trace(
                    Scatter::new(data.spectrum_k.clone(), spec.clone()).name(&format!("N = {}", n)),
                );
            }
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
            plot.write_html(&format!(
                "{}/lattice.{}.spectrums.html",
                out_dir, &self.name,
            ));
        }
        Ok(())
    }
}

struct ScalarPerturbationSetting {
    pub name: String,
    pub k: f64,
    pub da: f64,
}

impl ScalarPerturbationSetting {
    pub fn run<V, Xi>(
        &self,
        out_dir: &str,
        background: &[GaussBonnetBackgroundState],
        input: &GaussBonnetBInput<V, Xi>,
    ) {
        let pert = HamitonianSimulator::new(
            input,
            background.len(),
            background,
            DefaultPerturbationInitializer,
            GaussBonnetScalarPerturbationPotential,
            HorizonSelector::new(1e3),
            GaussBonnetScalarPerturbationCoef,
        );
        let mut efolding = vec![];
        let mut zeta = vec![];
        pert.run(self.k, self.da, |_, background, _, field, _, _| {
            efolding.push(background.a.ln());
            zeta.push(field.abs());
        });
        let mut plot = Plot::new();
        plot.add_trace(Scatter::new(efolding, zeta));
        plot.set_layout(
            Layout::new()
                .x_axis(Axis::new().exponent_format(ExponentFormat::Power))
                .y_axis(
                    Axis::new()
                        .type_(AxisType::Log)
                        .exponent_format(ExponentFormat::Power),
                )
                .height(800),
        );
        plot.write_html(&format!("{}/perturbation.{}.html", out_dir, &self.name));
    }
}

struct SpectrumSetting {
    pub name: String,
    pub k_range: ParamRange<f64>,
    pub da: f64,
}

impl SpectrumSetting {
    pub fn run<V, Xi>(
        &self,
        out_dir: &str,
        background: &[GaussBonnetBackgroundState],
        input: &GaussBonnetBInput<V, Xi>,
    ) -> anyhow::Result<()>
    where
        V: C2Fn<f64, Output = f64> + Send + Sync,
        Xi: C2Fn<f64, Output = f64> + Send + Sync,
    {
        let pert = HamitonianSimulator::new(
            input,
            background.len(),
            background,
            DefaultPerturbationInitializer,
            GaussBonnetScalarPerturbationPotential,
            HorizonSelector::new(1e3),
            GaussBonnetScalarPerturbationCoef,
        );
        let spectrum = pert.spectrum_with_cache(
            &format!("{}/spectrum.{}.bincode", out_dir, &self.name),
            self.k_range,
            self.da,
        )?;
        let mut plot = Plot::new();
        plot.add_trace(Scatter::new(self.k_range.as_logspace().collect(), spectrum));
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
        plot.write_html(&format!("{}/spectrum.{}.html", out_dir, &self.name));
        Ok(())
    }
}

struct Params<V, Xi> {
    pub input: GaussBonnetBInput<V, Xi>,
    pub phi0: f64,
    pub a0: f64,
    pub dt: f64,
    pub lattice: Vec<LatticeSetting>,
    pub perts: Vec<ScalarPerturbationSetting>,
    pub spectrums: Vec<SpectrumSetting>,
}

impl<V, Xi> Params<V, Xi> {
    pub fn run(&self, out_dir: &str) -> anyhow::Result<()>
    where
        V: C2Fn<f64, Output = f64> + Send + Sync,
        Xi: C2Fn<f64, Output = f64> + Send + Sync,
    {
        create_dir_all(out_dir)?;
        let mut rate_limiter = RateLimiter::new(Duration::from_millis(100));
        let background = lazy_file(
            &format!("{}/background.bincode", out_dir),
            BINCODE_CONFIG,
            || {
                let mut state =
                    GaussBonnetBackgroundState::init_slowroll3d(&self.input, self.a0, self.phi0);
                state.dt = self.dt;
                let mut ret = vec![state];
                while state.epsilon(&self.input) < 1.0 {
                    rate_limiter.run(|| {
                        println!(
                            "[background] (epsilon = {}) {:?}",
                            state.epsilon(&self.input),
                            &state
                        )
                    });
                    state = state.update(&self.input, self.dt);
                    state.dt = self.dt;
                    ret.push(state);
                }
                GaussBonnetBackgroundState::calculate_pert_coefs(&mut ret, &self.input);
                ret
            },
        )?;
        let max_length = 500000usize;
        {
            let mut efolding = vec![];
            let mut phi = vec![];
            let mut v_phi = vec![];
            let mut hubble = vec![];
            let mut epsilon = vec![];
            let mut hubble_constraint = vec![];
            let mut mom_coef = vec![];
            let mut horizon = vec![];
            for state in limit_length(background.clone(), max_length) {
                efolding.push(state.a.ln());
                phi.push(state.phi);
                v_phi.push(state.v_phi);
                hubble.push(state.v_a / state.a);
                epsilon.push(state.epsilon(&self.input));
                hubble_constraint.push(state.hubble_constraint(&self.input).abs());
                mom_coef.push(state.pert_c / state.pert_a);
                horizon.push(state.horizon.sqrt());
            }
            let mut plot = Plot::new();
            plot.add_trace(Scatter::new(efolding.clone(), phi).name("phi"));
            plot.add_trace(
                Scatter::new(efolding.clone(), v_phi)
                    .name("v_phi")
                    .y_axis("y2"),
            );
            plot.add_trace(
                Scatter::new(efolding.clone(), epsilon)
                    .name("epsilon")
                    .y_axis("y3"),
            );
            plot.add_trace(
                Scatter::new(efolding.clone(), hubble_constraint)
                    .name("hubble_constraint")
                    .y_axis("y4"),
            );
            plot.add_trace(
                Scatter::new(efolding.clone(), hubble)
                    .name("H")
                    .y_axis("y5"),
            );
            plot.add_trace(
                Scatter::new(efolding.clone(), mom_coef)
                    .name("mom_coef")
                    .y_axis("y6"),
            );
            plot.add_trace(
                Scatter::new(efolding.clone(), horizon)
                    .name("horizon")
                    .y_axis("y7"),
            );
            plot.set_layout(
                Layout::new()
                    .grid(LayoutGrid::new().rows(7).columns(1))
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
            plot.write_html(&format!("{}/background.html", out_dir));
        }
        for pert in &self.perts {
            pert.run(out_dir, &background, &self.input);
        }
        for spec in &self.spectrums {
            spec.run(out_dir, &background, &self.input)?;
        }
        for lat in &self.lattice {
            lat.run(out_dir, &background, &self.input)?;
        }
        Ok(())
    }
}

pub fn main() {
    let params1 = {
        let lambda = 0.0065;
        let lambda4 = lambda * lambda * lambda * lambda;
        let f = 7.0;
        let phi_c = 13.0;
        let xi0 = 6.044e7;
        let xi1 = 14.901;
        // let xi1 = 6.0 * sin(phi_c / f) / f / lambda4 / xi0 / {
        //     let cc = 1.0 + cos(phi_c / f);
        //     cc * cc
        // };
        Params {
            input: GaussBonnetBInput {
                dim: 3,
                kappa: 1.0,
                v: NaturalInflationPotential { lambda4, f },
                xi: TanhPotential {
                    coef: xi0,
                    omega: xi1,
                    shift: -xi1 * phi_c,
                },
            },
            phi0: 7.0,
            a0: 1.0,
            dt: 1.0,
            lattice: vec![LatticeSetting {
                name: "0".to_string(),
                starting_k: 3e17,
                dt: 1.0,
                lattice_size: 16,
                horizon_tolerance: 4.0,
                spectrum_count: 10,
            }],
            perts: vec![ScalarPerturbationSetting {
                name: "0".to_string(),
                k: 1e18,
                da: 0.1,
            }],
            spectrums: vec![SpectrumSetting {
                name: "0".to_string(),
                k_range: ParamRange::new(1.0, 1e25, 1000),
                da: 0.01,
            }],
        }
    };
    params1.run("out/gb.set1").unwrap();
}
