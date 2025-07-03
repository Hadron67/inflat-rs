use std::{fs::create_dir_all, time::Duration};

use anyhow::Ok;
use bincode::{Decode, Encode};
use inflat::{
    background::BINCODE_CONFIG,
    c2fn::C2Fn,
    lat::{BoxLattice, Lattice, LatticeParam},
    models::QuadraticPotential,
    scalar::{ScalarFieldParams, ScalarFieldState, spectrum, spectrum_with_scratch},
    util::{RateLimiter, VecN, lazy_file, limit_length},
};
use libm::{cosh, tanh};
use plotly::{
    Layout, Plot, Scatter,
    common::ExponentFormat,
    layout::{Axis, AxisType, LayoutGrid},
};

struct PhiPotential2 {
    pub mass: f64,
    pub s: f64,
    pub d: f64,
    pub phi_step: f64,
}

impl C2Fn<f64> for PhiPotential2 {
    type Output = f64;

    fn value(&self, phi: f64) -> Self::Output {
        0.5 * self.mass
            * self.mass
            * phi
            * phi
            * (1.0 + self.s * tanh((phi - self.phi_step) / self.d))
    }

    fn value_d(&self, phi: f64) -> Self::Output {
        let sech = 1.0 / cosh((phi - self.phi_step) / self.d);
        self.mass * self.mass / 2.0 / self.d
            * phi
            * (self.s * phi * sech * sech
                + 2.0 * self.d * (1.0 + self.s * tanh((phi - self.phi_step) / self.d)))
    }

    fn value_dd(&self, phi: f64) -> Self::Output {
        let sech = 1.0 / cosh((phi - self.phi_step) / self.d);
        let tanh = tanh((phi - self.phi_step) / self.d);
        self.mass
            * self.mass
            * (1.0
                + self.s * tanh
                + self.s / self.d / self.d * phi * sech * sech * (2.0 * self.d - phi * tanh))
    }
}

struct Params<F> {
    pub scalar_params: ScalarFieldParams<3, F>,
    pub a: f64,
    pub phi: f64,
    pub v_phi: f64,
    pub dt: f64,
    pub end_n: f64,
    pub mass: f64,
    pub spectrum_count: usize,
}

#[derive(Encode, Decode, Debug, Clone, Copy)]
struct Measurables {
    pub a: f64,
    pub v_a: f64,
    pub phi: f64,
    pub v_phi: f64,
}

#[derive(Encode, Decode)]
struct IntermediateData {
    pub evaluation_measurables: Vec<Measurables>,
    pub final_state: ScalarFieldState<3>,
    pub spectrum_data: Vec<(f64, Vec<f64>)>,
    pub spectrum_mom: Vec<f64>,
}

impl<F> Params<F>
where
    F: C2Fn<f64, Output = f64> + Send + Sync,
{
    pub fn run(&self, out_dir: &str) -> anyhow::Result<()> {
        create_dir_all(out_dir)?;
        let int_data = lazy_file(
            &format!("{}/int_data.bincode", out_dir),
            BINCODE_CONFIG,
            || {
                let mut steps = 0usize;
                let mut field = ScalarFieldState::zeros(self.scalar_params.lattice.size);
                let mut rand = random::Default::new([10, 100]);
                self.scalar_params
                    .init(&mut field, self.a, self.phi, self.v_phi);
                self.scalar_params.populate_noise(&mut rand, &mut field);
                let mut rate_limiter = RateLimiter::new(Duration::from_millis(100));
                let mut measurables = vec![Measurables {
                    a: self.a,
                    v_a: self.scalar_params.v_a(&field),
                    phi: self.phi,
                    v_phi: self.v_phi,
                }];
                let mut spectrum_n_cursor = self.a.ln();
                let spectrum_n_step = (self.end_n - self.a.ln()) / (self.spectrum_count as f64);
                let mut spectrum_scratch = BoxLattice::zeros(self.scalar_params.lattice.size);
                let initial_spectrum = spectrum_with_scratch(
                    &field.phi,
                    &self.scalar_params.lattice,
                    &mut spectrum_scratch,
                );
                let mut spectrum_data =
                    vec![(self.a.ln(), initial_spectrum.iter().map(|f| f.1).collect())];
                let spectrum_mom = initial_spectrum.iter().map(|f| f.0).collect();
                println!("initial: {:?}", &measurables);
                while self.scalar_params.scale_factor(&field).ln() < self.end_n {
                    self.scalar_params.apply_full_k_order2(&mut field, self.dt);
                    steps += 1;
                    let m = Measurables {
                        a: self.scalar_params.scale_factor(&field),
                        v_a: self.scalar_params.v_a(&field),
                        phi: field.phi.average(),
                        v_phi: self.scalar_params.v_phi(&field),
                    };
                    rate_limiter.run(|| {
                        println!(
                            "t = {}, n = {}, {:?}",
                            (steps as f64) * self.dt,
                            m.a.ln(),
                            &m
                        )
                    });
                    if m.a.ln() > spectrum_n_cursor {
                        spectrum_n_cursor += spectrum_n_step;
                        let spectrum = spectrum_with_scratch(
                            &field.phi,
                            &self.scalar_params.lattice,
                            &mut spectrum_scratch,
                        );
                        spectrum_data.push((m.a.ln(), spectrum.iter().map(|f| f.1).collect()));
                    }
                    measurables.push(m);
                }
                IntermediateData {
                    evaluation_measurables: measurables,
                    final_state: field,
                    spectrum_data,
                    spectrum_mom,
                }
            },
        )?;
        let max_length = 500000;
        {
            let mut plot = Plot::new();
            let mut efoldings = vec![];
            let mut phi = vec![];
            let mut v_phi = vec![];
            let mut hubble = vec![];
            for state in limit_length(int_data.evaluation_measurables, max_length) {
                efoldings.push(state.a.ln());
                phi.push(state.phi);
                v_phi.push(state.v_phi / self.mass);
                hubble.push(state.v_a / state.a / self.mass);
            }
            plot.add_trace(Scatter::new(efoldings.clone(), phi).name("phi"));
            plot.add_trace(
                Scatter::new(efoldings.clone(), v_phi)
                    .name("v_phi")
                    .y_axis("y2"),
            );
            plot.add_trace(
                Scatter::new(efoldings.clone(), hubble)
                    .name("H / m")
                    .y_axis("y3"),
            );
            plot.set_layout(
                Layout::new()
                    .grid(LayoutGrid::new().rows(3).columns(1))
                    .x_axis(Axis::new().exponent_format(ExponentFormat::Power))
                    .y_axis(Axis::new().exponent_format(ExponentFormat::Power))
                    .y_axis2(Axis::new().exponent_format(ExponentFormat::Power))
                    .y_axis3(Axis::new().exponent_format(ExponentFormat::Power))
                    .height(1000),
            );
            plot.write_html(&format!("{}/background.html", out_dir));
        }
        {
            let mut plot = Plot::new();
            let final_spectrum = spectrum(&int_data.final_state.phi, &self.scalar_params.lattice);
            for (n, data) in &int_data.spectrum_data {
                plot.add_trace(
                    Scatter::new(
                        limit_length(int_data.spectrum_mom.clone(), max_length),
                        limit_length(data.clone(), max_length),
                    )
                    .name(&format!("N = {}", n)),
                );
            }
            plot.add_trace(
                Scatter::new(
                    int_data.spectrum_mom.clone(),
                    final_spectrum.iter().map(|f| f.1).collect(),
                )
                .y_axis("y2"),
            );
            plot.set_layout(
                Layout::new()
                    .grid(LayoutGrid::new().rows(2).columns(1))
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
                    .y_axis2(
                        Axis::new()
                            .type_(AxisType::Log)
                            .exponent_format(ExponentFormat::Power),
                    )
                    .height(1000),
            );
            plot.write_html(&format!("{}/spectrum.html", out_dir));
        }
        Ok(())
    }
}

pub fn main() {
    let mass = 0.51e-5;
    let params_set1 = {
        let size = 16usize;
        let l = 1.4 / mass;
        Params {
            scalar_params: ScalarFieldParams {
                kappa: 1.0,
                potential: QuadraticPotential::new(mass),
                lattice: LatticeParam {
                    spacing: VecN::new([l / (size as f64); 3]),
                    size: VecN::new([size; 3]),
                },
            },
            a: 1.0,
            phi: 14.5,
            v_phi: -0.8152 * mass,
            dt: 1.0,
            end_n: 7.0,
            mass,
            spectrum_count: 10,
        }
    };
    params_set1.run("out/scalar_lat.set1").unwrap();

    let params_set2 = {
        let size = 256usize;
        let l = 0.6 / mass;
        let dx = l / (size as f64);
        Params {
            scalar_params: ScalarFieldParams {
                kappa: 1.0,
                potential: PhiPotential2 {
                    mass,
                    s: 0.01,
                    d: 0.005,
                    phi_step: 14.35,
                },
                lattice: LatticeParam {
                    size: VecN::new([size; 3]),
                    spacing: VecN::new([dx; 3]),
                },
            },
            a: 1.0,
            phi: 14.5,
            v_phi: -0.8152 * mass,
            dt: 1.0,
            end_n: 7.0,
            mass,
            spectrum_count: 10,
        }
    };
    params_set2.run("out/scalar_lat.set2").unwrap();

    let params_set3 = {
        let size = 64usize;
        let l = 0.6 / mass;
        let dx = l / (size as f64);
        Params {
            scalar_params: ScalarFieldParams {
                kappa: 1.0,
                potential: PhiPotential2 {
                    mass,
                    s: 0.01,
                    d: 0.005,
                    phi_step: 14.35,
                },
                lattice: LatticeParam {
                    size: VecN::new([size; 3]),
                    spacing: VecN::new([dx; 3]),
                },
            },
            a: 1.0,
            phi: 14.5,
            v_phi: -0.8152 * mass,
            dt: 1.0,
            end_n: 7.0,
            mass,
            spectrum_count: 10,
        }
    };
    params_set3.run("out/scalar_lat.set3").unwrap();
}
