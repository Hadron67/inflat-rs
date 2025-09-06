use std::{f64::consts::PI, fmt::Display, fs::create_dir_all, time::Duration};

use bincode::{Decode, Encode};
use inflat::{
    background::{
        spectrum_k_range_from_background, BackgroundSolver, BackgroundState, BackgroundStateInput, BackgroundStateInputProvider, DefaultPerturbationInitializer, HamitonianSimulator, HorizonSelector, HubbleConstraint, Kappa, PhiD, ScalarPerturbationFactor, ScalarPerturbationPotential, ScaleFactor, ScaleFactorD, SlowrollEpsilon, SlowrollKappa, BINCODE_CONFIG, MPC_HZ
    },
    c2fn::C2Fn,
    lat::{BoxLattice, Lattice, LatticeParam},
    models::QuadraticPotential,
    scalar::{
        spectrum, spectrum_with_scratch, LatticeInput, LatticeNoiseGenerator as _, LatticeSimulator, LatticeSimulatorCreator, LatticeState, ScalarFieldSimulatorCreator, ScalarFieldState
    },
    util::{
        self, lazy_file_opt, limit_length, log_interp, plot_spectrum, Hms, ParamRange, RateLimiter, TimeEstimator, VecN
    },
};
use libm::{cosh, tanh};
use num_traits::Pow;
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
    pub linear_spectrum_k: ParamRange<f64>,
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
    pub hubble_constraint: f64,
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
    pub fn run(&self, out_dir: &str, skip_lattice: bool) -> anyhow::Result<()> {
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
                efoldings.push(state.scale_factor(&self.input).ln());
                phi.push(state.phi);
                v_phi.push(state.v_phi(&self.input));
                hubble.push(state.v_scale_factor(&self.input) / state.scale_factor(&self.input));
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
        let linear_spectrum_k_range = {
            let min_k = background[0].v_a(&self.input) * ht;
            let max_k = background.last().unwrap().v_a(&self.input) / ht;
            let k_range = &self.linear_spectrum_k;
            ParamRange::new(
                log_interp(min_k, max_k, k_range.start),
                log_interp(min_k, max_k, k_range.end),
                k_range.count,
            )
        };
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
        let pivot_k_index = linear_spectrum.partition_point(|x| *x > 2.1e-9);
        let pivot_k_background_index = {
            let k0 = linear_spectrum_k_range.log_interp(pivot_k_index);
            background.partition_point(|s| s.v_scale_factor(&self.input) <= k0)
        };
        let k_star = if pivot_k_index < linear_spectrum.len() {
            linear_spectrum_k_range.log_interp(pivot_k_index) / (0.05 * MPC_HZ)
        } else {
            1.0
        };
        {
            let mut plot = Plot::new();
            let amp = self
                .input
                .scalar_spectral_amplitude(&background[pivot_k_background_index]);
            let p = self
                .input
                .scalar_spectral_power(&background[pivot_k_background_index]);
            let k_data: Vec<_> = (linear_spectrum_k_range / k_star)
                .as_logspace()
                .into_iter()
                .collect();
            plot.add_trace(Scatter::new(k_data.clone(), linear_spectrum.clone()).name("numeric"));
            plot.add_trace(
                Scatter::new(
                    k_data.clone(),
                    k_data.iter().map(|k| amp * k.pow(p)).collect(),
                )
                .name("slow roll"),
            );
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
        // let lat_params = ScalarFieldParams {
        //     kappa: self.input.kappa,
        //     potential: &self.input.potential,
        //     lattice: self.lattice_param,
        // };
        let simulator = ScalarFieldSimulatorCreator.create(self.lattice_param, &self.input);
        if let Ok(int_data) = lazy_file_opt(
            &format!("{}/int_data.bincode", out_dir),
            BINCODE_CONFIG,
            !skip_lattice,
            || {
                let mut steps = 0usize;
                let mut field = ScalarFieldState::zero(self.lattice_param.size);
                simulator.init(&mut field, self.a, self.phi, self.v_phi);
                {
                    let l = (self.lattice_param.size[0] as f64) * self.lattice_param.spacing[0];
                    let k_min = 2.0 * PI / l;
                    let hubble = field.hubble(&self.input);
                    println!("k_min = {:e} * H_in", k_min / hubble);
                }
                simulator.populate_noise(&mut field, &mut random::default(10));
                let mut rate_limiter = RateLimiter::new(Duration::from_millis(100));
                let mut measurables = vec![Measurables {
                    a: self.a,
                    v_a: simulator.v_a(&field),
                    phi: self.phi,
                    v_phi: self.v_phi,
                    hubble_constraint: simulator.hubble_constraint(&field),
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
                while simulator.scale_factor(&field).ln() < self.end_n {
                    simulator.apply_full_k_order2(&mut field, self.dt);
                    steps += 1;
                    time_estimator.update(simulator.scale_factor(&field).ln());
                    let m = Measurables {
                        a: simulator.scale_factor(&field),
                        v_a: simulator.v_a(&field),
                        phi: field.phi.average(),
                        v_phi: simulator.v_phi_average(&field),
                        hubble_constraint: simulator.hubble_constraint(&field),
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
        ) {
            let max_length = 500000;
            {
                let mut plot = Plot::new();
                let mut efoldings = vec![];
                let mut phi = vec![];
                let mut v_phi = vec![];
                let mut hubble = vec![];
                let mut hubble_constraint = vec![];
                for state in limit_length(&int_data.evaluation_measurables, max_length) {
                    efoldings.push(state.a.ln());
                    phi.push(state.phi);
                    v_phi.push(state.v_phi);
                    hubble.push(state.v_a / state.a);
                    hubble_constraint.push(state.hubble_constraint);
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
                plot.add_trace(
                    Scatter::new(efoldings.clone(), hubble_constraint)
                        .name("hubble constraint")
                        .y_axis("y4"),
                );
                plot.set_layout(
                    Layout::new()
                        .grid(LayoutGrid::new().rows(4).columns(1))
                        .x_axis(Axis::new().exponent_format(ExponentFormat::Power))
                        .y_axis(Axis::new().exponent_format(ExponentFormat::Power))
                        .y_axis2(Axis::new().exponent_format(ExponentFormat::Power))
                        .y_axis3(Axis::new().exponent_format(ExponentFormat::Power))
                        .y_axis4(Axis::new().type_(AxisType::Log).exponent_format(ExponentFormat::Power))
                        .height(1200),
                );
                plot.write_html(&format!("{}/background.lattice.html", out_dir));
            }
            {
                let mut plot = Plot::new();
                let linear_zeta = {
                    let field = &int_data.final_state;
                    let final_hubble = simulator.hubble(&field);
                    let phi_avg = field.phi.average();
                    field
                        .phi
                        .as_ref()
                        .zip(field.mom_phi.as_ref())
                        .map(move |(phi, mom_phi)| {
                            let v_phi =
                                simulator.v_phi_from_mom_phi(int_data.final_state.b, mom_phi);
                            (phi - phi_avg) / v_phi * final_hubble
                        })
                };
                let final_spectrum = spectrum(&linear_zeta, &self.lattice_param);
                let lattice_start_index = background.partition_point(|state| state.phi > self.phi);
                plot.add_trace(
                    Scatter::new(
                        int_data.spectrum_mom.clone(),
                        int_data
                            .spectrum_mom
                            .iter()
                            .map(|k| k * k / 4.0 / PI / PI)
                            .collect(),
                    )
                    .name("initial linear"),
                );
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
                    )
                    .name("final")
                    .y_axis("y2"),
                );
                plot.add_trace(
                    Scatter::new(
                        (linear_spectrum_k_range
                            / background[lattice_start_index].scale_factor(&self.input))
                        .as_logspace()
                        .collect(),
                        linear_spectrum.clone(),
                    )
                    .name("linear")
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
        }
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
        Ok(())
    }
}

struct LatticeIn {
    pub size: usize,
    pub k: f64,
    pub subhorizon_tolerance: f64,
    pub superhorizon_tolerance: f64,
}

fn run_common<F>(
    dir: &str,
    name: &str,
    input: &BackgroundStateInput<F>,
    phi0: f64,
    dt: f64,
    linear_spectrum_k_range: ParamRange<f64>,
    lattice_in: Option<LatticeIn>,
) -> util::Result<()>
where
    F: C2Fn<f64, Output = f64> + Sync,
{
    println!("[run_common] working on {}", name);
    create_dir_all(dir)?;
    let background = input.simulate(
        BackgroundState::init_slowroll(1.0, phi0, input),
        dt,
        |s| s.epsilon(input) > 1.0,
        |_| {},
    );
    plot_background(
        &format!("{}/{}.background.html", dir, name),
        &background,
        input,
    );
    let ht = 1e3;
    let k_range = spectrum_k_range_from_background(&background, input, linear_spectrum_k_range, ht);
    let linear_spectrum = {
        let pert = HamitonianSimulator::new(
            input,
            background.len(),
            &background,
            DefaultPerturbationInitializer,
            ScalarPerturbationPotential,
            HorizonSelector::new(ht),
            ScalarPerturbationFactor,
        );
        pert.spectrum_with_cache(
            &format!("{}/{}.linear_spectrum.bincode", dir, name),
            k_range,
            0.1,
            false,
        )?
    };
    let (k_star, sr_amp, sr_power) = {
        let k_star_index = {
            let i = linear_spectrum.partition_point(|x| *x > 2.1e-9);
            if i < linear_spectrum.len() { i } else { 0 }
        };
        let k0 = k_range.log_interp(k_star_index);
        let k_star_background_index = background.partition_point(|s| s.v_scale_factor(input) <= k0);
        (
            k0,
            input.scalar_spectral_amplitude(&background[k_star_background_index]),
            input.scalar_spectral_power(&background[k_star_background_index]),
        )
    };
    let plot_k_range = k_range / k_star;
    plot_spectrum(
        &format!("{}/{}.linear_spectrum.html", dir, name),
        &[
            (
                "linear",
                &plot_k_range.as_logspace().collect::<Vec<_>>(),
                &linear_spectrum,
            ),
            (
                "slowroll",
                &plot_k_range.as_logspace().collect::<Vec<_>>(),
                &plot_k_range
                    .as_logspace()
                    .map(|k| sr_amp * k.pow(sr_power))
                    .collect::<Vec<_>>(),
            ),
        ],
        0.05 * MPC_HZ,
    );
    if let Some(lat_param) = &lattice_in {
        let lat_in = LatticeInput::<3, BackgroundState>::from_background_and_k_normalized(
            &background,
            input,
            log_interp(
                background[0].v_a(input),
                background.last().unwrap().v_a(input),
                lat_param.k,
            ),
            lat_param.subhorizon_tolerance,
            lat_param.superhorizon_tolerance,
            lat_param.size,
            1.0,
        );
        let output = lat_in.run(
            &ScalarFieldSimulatorCreator,
            input,
            &format!("{}/{}.lattice", dir, name),
            true,
            10,
        )?;
        output.plot_all(
            dir,
            &format!("{}.lattice", name),
            0.05 * MPC_HZ / k_star,
            k_range,
            &linear_spectrum,
        );
    }
    Ok(())
}

fn plot_background<F: C2Fn<f64, Output = f64>>(
    out_file: &str,
    background: &[BackgroundState],
    input: &BackgroundStateInput<F>,
) {
    let mut plot = Plot::new();
    let mut efoldings = vec![];
    let mut phi = vec![];
    let mut v_phi = vec![];
    let mut hubble = vec![];
    let mut slowroll_kappa = vec![];
    for state in limit_length(&background, 100000) {
        efoldings.push(state.scale_factor(input).ln());
        phi.push(state.phi);
        v_phi.push(state.v_phi(input));
        hubble.push(state.hubble(input));
        slowroll_kappa.push(input.slowroll_kappa(state));
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
    plot.add_trace(
        Scatter::new(efoldings.clone(), slowroll_kappa)
            .name("kappa")
            .y_axis("y4"),
    );
    plot.set_layout(
        Layout::new()
            .grid(LayoutGrid::new().rows(4).columns(1))
            .x_axis(Axis::new().exponent_format(ExponentFormat::Power))
            .y_axis(Axis::new().exponent_format(ExponentFormat::Power))
            .y_axis2(Axis::new().exponent_format(ExponentFormat::Power))
            .y_axis3(Axis::new().exponent_format(ExponentFormat::Power))
            .y_axis4(Axis::new().exponent_format(ExponentFormat::Power))
            .height(1200),
    );
    plot.write_html(out_file);
}

fn report_err<T, E: Display>(res: Result<T, E>) {
    let _ = res.inspect_err(|e| println!("error: {}", e));
}

pub fn main() {
    let mass = 0.51e-5;
    let params_set1 = {
        let size = 32usize;
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
            linear_spectrum_k: ParamRange::new(0.0, 1.0, 1000),
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
    report_err(params_set1.run("out/scalar_lat.set1", false));

    // let params_set2 = {
    //     let size = 256usize;
    //     let l = 0.6 / mass;
    //     let dx = l / (size as f64);
    //     Params {
    //         input: BackgroundStateInput {
    //             kappa: 1.0,
    //             potential: PhiPotential2 {
    //                 mass,
    //                 s: 0.01,
    //                 d: 0.005,
    //                 phi_step: 14.35,
    //             },
    //         },
    //         lattice_param: LatticeParam {
    //             size: VecN::new([size; 3]),
    //             spacing: VecN::new([dx; 3]),
    //         },
    //         linear_spectrum_k: ParamRange::new(0.0, 1.0, 1000),
    //         a: 1.0,
    //         phi: 14.5,
    //         linear_phi: 16.0,
    //         v_phi: -0.8152 * mass,
    //         dt: 1.0,
    //         end_n: 7.0,
    //         mass,
    //         spectrum_count: 10,
    //     }
    // };
    // report_err(params_set2.run("out/scalar_lat.set2", false));

    {
        // let params_set2 = {
        //     let size = 64usize;
        //     let l = 0.6 / mass;
        //     let dx = l / (size as f64);
        //     Params {
        //         input: BackgroundStateInput {
        //             kappa: 1.0,
        //             potential: PhiPotential2 {
        //                 mass,
        //                 s: 0.01,
        //                 d: 0.005,
        //                 phi_step: 14.35,
        //             },
        //         },
        //         lattice_param: LatticeParam {
        //             size: VecN::new([size; 3]),
        //             spacing: VecN::new([dx; 3]),
        //         },
        //         linear_spectrum_k: ParamRange::new(0.0, 1.0, 1000),
        //         a: 1.0,
        //         phi: 14.6,
        //         linear_phi: 16.0,
        //         v_phi: -0.8152 * mass,
        //         // v_phi: -4.178e-6,
        //         dt: 1.0,
        //         end_n: 7.0,
        //         mass,
        //         spectrum_count: 10,
        //     }
        // };
        // report_err(params_set2.run("out/scalar_lat.set2_alt", false));
    }

    // let params_set4 = {
    //     let size = 64usize;
    //     let l = 0.6 / mass;
    //     let dx = l / (size as f64);
    //     Params {
    //         input: BackgroundStateInput {
    //             kappa: 1.0,
    //             potential: PhiPotential2 {
    //                 mass,
    //                 s: -0.0020365,
    //                 d: 0.003,
    //                 phi_step: 14.1,
    //             },
    //         },
    //         lattice_param: LatticeParam {
    //             size: VecN::new([size; 3]),
    //             spacing: VecN::new([dx; 3]),
    //         },
    //         linear_spectrum_k: ParamRange::new(0.0, 0.5, 2000),
    //         a: 1.0,
    //         phi: 14.5,
    //         linear_phi: 17.5,
    //         v_phi: -0.8152 * mass,
    //         dt: 1.0,
    //         end_n: 7.0,
    //         mass,
    //         spectrum_count: 10,
    //     }
    // };
    // report_err(params_set4.run("out/scalar_lat.set4", true));

    {
        let set4_input = BackgroundStateInput {
            kappa: 1.0,
            potential: PhiPotential2 {
                mass,
                s: -0.0020365,
                d: 0.003,
                phi_step: 14.1,
            },
        };
        report_err(run_common(
            "out/scalar_lat",
            "set4",
            &set4_input,
            17.5,
            1.0,
            ParamRange::new(0.0, 1.0, 1000),
            Some(LatticeIn {
                size: 64,
                k: 0.32,
                subhorizon_tolerance: 10.0,
                superhorizon_tolerance: 40.0,
            }),
        ));
    }
    {
        let set4_input = BackgroundStateInput {
            kappa: 1.0,
            potential: PhiPotential2 {
                mass,
                s: -0.00201,
                d: 0.003,
                phi_step: 14.1,
            },
        };
        report_err(run_common(
            "out/scalar_lat",
            "set4_alt",
            &set4_input,
            17.5,
            1.0,
            ParamRange::new(0.0, 1.0, 1000),
            Some(LatticeIn {
                size: 32,
                k: 0.2,
                subhorizon_tolerance: 5.0,
                superhorizon_tolerance: 40.0,
            }),
        ));
    }
    {
        let set4_input = BackgroundStateInput {
            kappa: 1.0,
            potential: PhiPotential2 {
                mass,
                s: -0.00201,
                d: 0.003,
                phi_step: 14.1,
            },
        };
        report_err(run_common(
            "out/scalar_lat",
            "set4_alt2",
            &set4_input,
            17.5,
            1.0,
            ParamRange::new(0.0, 1.0, 1000),
            Some(LatticeIn {
                size: 32,
                k: 0.31,
                subhorizon_tolerance: 50.0,
                superhorizon_tolerance: 100.0,
            }),
        ));
    }
    {
        let set5_input = BackgroundStateInput {
            kappa: 1.0,
            potential: PhiPotential2 {
                mass,
                s: 0.01,
                d: 0.005,
                phi_step: 14.35,
            },
        };
        report_err(run_common(
            "out/scalar_lat",
            "set5",
            &set5_input,
            17.5,
            1.0,
            ParamRange::new(0.25, 0.4, 1000),
            Some(LatticeIn {
                size: 32,
                k: 0.32,
                subhorizon_tolerance: 40.0,
                superhorizon_tolerance: 40.0,
            }),
        ));
    }
    {
        let qp_set1_input = BackgroundStateInput {
            kappa: 1.0,
            potential: QuadraticPotential::new(mass),
        };
        report_err(run_common(
            "out/scalar_lat",
            "qp_set1_alt",
            &qp_set1_input,
            17.5,
            1.0,
            ParamRange::new(0.0, 1.0, 1000),
            Some(LatticeIn {
                size: 32,
                k: 0.37,
                subhorizon_tolerance: 1.3,
                superhorizon_tolerance: 40.0,
            }),
        ));
    }
}
