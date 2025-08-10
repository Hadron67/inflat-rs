use std::{fs::create_dir_all, time::Duration};

use anyhow::Ok;
use bincode::{Decode, Encode};
use inflat::{
    background::{
        BackgroundState, BackgroundStateInput, BackgroundStateInputProvider, DefaultPerturbationInitializer, HamitonianSimulator, HorizonSelector, Kappa, PhiD, ScalarPerturbationFactor, ScalarPerturbationPotential, ScaleFactor, ScaleFactorD, BINCODE_CONFIG
    },
    c2fn::C2Fn,
    lat::{BoxLattice, Lattice, LatticeParam},
    models::QuadraticPotential,
    scalar::{
        spectrum, spectrum_with_scratch, ScalarFieldParams, ScalarFieldState
    },
    util::{lazy_file, limit_length, Hms, ParamRange, RateLimiter, TimeEstimator, VecN},
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
    pub input: BackgroundStateInput<F>,
    pub lattice_param: LatticeParam<3>,
    pub linear_phi: f64,
    pub linear_spectrum_k: Option<ParamRange<f64>>,
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

impl<F> Kappa for Params<F> {
    fn kappa(&self) -> f64 {
        self.input.kappa
    }
}

impl<F> BackgroundStateInputProvider for Params<F> {
    type F = F;

    fn input(&self) -> &BackgroundStateInput<Self::F> {
        &self.input
    }
}

impl<F> Params<F>
where
    F: C2Fn<f64, Output = f64> + Send + Sync,
{
    pub fn run(&self, out_dir: &str) -> anyhow::Result<()> {
        println!("working on {}", out_dir);
        create_dir_all(out_dir)?;
        let background = {
            let state = BackgroundState::init_slowroll(self.a, self.linear_phi, &self.input);
            state.simulate(
                &self.input,
                0.1,
                1.0,
                4,
                |state| state.epsilon(&self.input) >= 1.0,
                |_| {},
            )
        };
        {
            let mut plot = Plot::new();
            let mut efoldings = vec![];
            let mut phi = vec![];
            let mut v_phi = vec![];
            let mut hubble = vec![];
            for state in limit_length(&background, 100000) {
                efoldings.push(state.scale_factor().ln());
                phi.push(state.phi);
                v_phi.push(state.v_phi());
                hubble.push(state.v_scale_factor(self.input.kappa) / state.scale_factor());
            }
            plot.add_trace(Scatter::new(efoldings.clone(), phi).name("phi"));
            plot.add_trace(
                Scatter::new(efoldings.clone(), v_phi)
                    .name("v_phi")
                    .y_axis("y2"),
            );
            plot.add_trace(
                Scatter::new(efoldings.clone(), hubble)
                    .name("H")
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
        let ht = 1e3;
        let linear_spectrum_k_range = self.linear_spectrum_k.unwrap_or_else(|| {
            ParamRange::new(
                background[0].v_a(&self.input) * ht,
                background.last().unwrap().v_a(&self.input),
                1000,
            )
        });
        let linear_spectrum = {
            let pert = HamitonianSimulator::new(
                self,
                background.len(),
                &background,
                DefaultPerturbationInitializer,
                ScalarPerturbationPotential,
                HorizonSelector::new(ht),
                ScalarPerturbationFactor,
            );
            pert.spectrum_with_cache(
                &format!("{}/linear_spectrum.bincode", out_dir),
                linear_spectrum_k_range,
                0.1,
                false,
            )?
        };
        {
            let mut plot = Plot::new();
            plot.add_trace(Scatter::new(
                linear_spectrum_k_range.as_logspace().into_iter().collect(),
                linear_spectrum.clone(),
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
            plot.write_html(&format!("{}/linear_spectrum.html", out_dir));
        }
        let lat_params = ScalarFieldParams {
            kappa: self.input.kappa,
            potential: &self.input.potential,
            lattice: self.lattice_param,
        };
        let int_data = lazy_file(
            &format!("{}/int_data.bincode", out_dir),
            BINCODE_CONFIG,
            || {
                let mut steps = 0usize;
                let mut field = ScalarFieldState::zeros(self.lattice_param.size);
                lat_params.init(&mut field, self.a, self.phi, self.v_phi);
                lat_params.populate_noise(&mut random::default(10), &mut field);
                let mut rate_limiter = RateLimiter::new(Duration::from_millis(100));
                let mut measurables = vec![Measurables {
                    a: self.a,
                    v_a: lat_params.v_a(&field),
                    phi: self.phi,
                    v_phi: self.v_phi,
                }];
                let mut spectrum_n_cursor = self.a.ln();
                let spectrum_n_step = (self.end_n - self.a.ln()) / (self.spectrum_count as f64);
                let mut spectrum_scratch = BoxLattice::zeros(self.lattice_param.size);
                let initial_spectrum =
                    spectrum_with_scratch(&field.phi, &self.lattice_param, &mut spectrum_scratch);
                let mut spectrum_data =
                    vec![(self.a.ln(), initial_spectrum.iter().map(|f| f.1).collect())];
                let spectrum_mom = initial_spectrum.iter().map(|f| f.0).collect();
                println!("initial: {:?}", &measurables);
                let mut time_estimator = TimeEstimator::new(0.0..self.end_n, 1000);
                while lat_params.scale_factor(&field).ln() < self.end_n {
                    lat_params.apply_full_k_order2(&mut field, self.dt);
                    steps += 1;
                    time_estimator.update(lat_params.scale_factor(&field).ln());
                    let m = Measurables {
                        a: lat_params.scale_factor(&field),
                        v_a: lat_params.v_a(&field),
                        phi: field.phi.average(),
                        v_phi: lat_params.v_phi_average(&field),
                    };
                    rate_limiter.run(|| {
                        println!(
                            "eta time = {}, t = {}, n = {}, measurables = {:?}",
                            Hms::from_secs(time_estimator.remaining_secs()),
                            (steps as f64) * self.dt,
                            m.a.ln(),
                            &m
                        )
                    });
                    if m.a.ln() > spectrum_n_cursor {
                        spectrum_n_cursor += spectrum_n_step;
                        let spectrum = spectrum_with_scratch(
                            &field.phi,
                            &self.lattice_param,
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
        // let zeta_and_spectrum = lazy_file(
        //     &format!("{}/zeta_field.bincode", out_dir),
        //     BINCODE_CONFIG,
        //     || {
        //         let reference_phi = int_data
        //             .final_state
        //             .phi
        //             .as_ref()
        //             .map(|phi| self.input.potential.value(phi))
        //             .min()
        //             .1;
        //         let zeta = construct_zeta(
        //             lat_params.scale_factor(&int_data.final_state),
        //             lat_params.v_a(&int_data.final_state),
        //             &int_data.final_state.phi,
        //             &int_data.final_state.mom_phi.as_ref().map(|mom_phi| {
        //                 lat_params.v_phi_from_mom_phi(int_data.final_state.b, mom_phi)
        //             }),
        //             reference_phi,
        //             100,
        //             1.0,
        //             &self.input,
        //             |done, total| println!("[zeta] ({}/{})", done, total),
        //         );
        //         let zeta_spectrum = spectrum(&zeta, &self.lattice_param);
        //         (zeta, zeta_spectrum)
        //     },
        // )?;
        let max_length = 500000;
        {
            let mut plot = Plot::new();
            let mut efoldings = vec![];
            let mut phi = vec![];
            let mut v_phi = vec![];
            let mut hubble = vec![];
            for state in limit_length(&int_data.evaluation_measurables, max_length) {
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
            plot.write_html(&format!("{}/background.lattice.html", out_dir));
        }
        {
            let mut plot = Plot::new();
            let linear_zeta = {
                let field = &int_data.final_state;
                let final_hubble = lat_params.hubble(&field);
                let phi_avg = field.phi.average();
                field.phi.as_ref().zip(field.mom_phi.as_ref()).map(move |(phi, mom_phi)| {
                    let v_phi = lat_params.v_phi_from_mom_phi(int_data.final_state.b, mom_phi);
                    (phi - phi_avg) / v_phi * final_hubble
                })
            };
            let final_spectrum = spectrum(&linear_zeta, &self.lattice_param);
            let lattice_start_index = background.partition_point(|state| state.phi > self.phi);
            for (n, data) in &int_data.spectrum_data {
                plot.add_trace(
                    Scatter::new(
                        limit_length(&int_data.spectrum_mom, max_length)
                            .cloned()
                            .collect(),
                        limit_length(&data, max_length).cloned().collect(),
                    )
                    .name(&format!("N = {}", n)),
                );
            }
            plot.add_trace(
                Scatter::new(
                    int_data.spectrum_mom.clone(),
                    final_spectrum.iter().map(|f| f.1).collect(),
                ).name("final")
                .y_axis("y2"),
            );
            plot.add_trace(Scatter::new((linear_spectrum_k_range / background[lattice_start_index].scale_factor()).as_logspace().collect(), linear_spectrum.clone()).name("linear").y_axis("y2"));
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
            input: BackgroundStateInput {
                kappa: 1.0,
                potential: QuadraticPotential::new(mass),
            },
            lattice_param: LatticeParam {
                spacing: VecN::new([l / (size as f64); 3]),
                size: VecN::new([size; 3]),
            },
            linear_spectrum_k: None,
            a: 1.0,
            phi: 14.5,
            linear_phi: 16.0,
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
            input: BackgroundStateInput {
                kappa: 1.0,
                potential: PhiPotential2 {
                    mass,
                    s: 0.01,
                    d: 0.005,
                    phi_step: 14.35,
                },
            },
            lattice_param: LatticeParam {
                size: VecN::new([size; 3]),
                spacing: VecN::new([dx; 3]),
            },
            linear_spectrum_k: Some(ParamRange::new(1.0, 1e4, 1000)),
            a: 1.0,
            phi: 14.5,
            linear_phi: 16.0,
            v_phi: -0.8152 * mass,
            dt: 1.0,
            end_n: 7.0,
            mass,
            spectrum_count: 10,
        }
    };
    params_set2.run("out/scalar_lat.set2").unwrap();

    // let params_set3 = {
    //     let size = 64usize;
    //     let l = 0.6 / mass;
    //     let dx = l / (size as f64);
    //     Params {
    //         scalar_params: ScalarFieldParams {
    //             kappa: 1.0,
    //             potential: PhiPotential2 {
    //                 mass,
    //                 s: 0.01,
    //                 d: 0.005,
    //                 phi_step: 14.35,
    //             },
    //             lattice: LatticeParam {
    //                 size: VecN::new([size; 3]),
    //                 spacing: VecN::new([dx; 3]),
    //             },
    //         },
    //         a: 1.0,
    //         phi: 14.5,
    //         v_phi: -0.8152 * mass,
    //         dt: 1.0,
    //         end_n: 7.0,
    //         mass,
    //         spectrum_count: 10,
    //     }
    // };
    // params_set3.run("out/scalar_lat.set3").unwrap();
}
