use std::{
    env,
    f64::consts::PI,
    fs::create_dir_all,
    sync::atomic::AtomicUsize,
    time::{Duration, SystemTime},
};

use bincode::{Decode, Encode};
use inflat::{
    background::{
        BINCODE_CONFIG, DefaultPerturbationInitializer, HamitonianSimulator, HorizonSelector,
        MPC_HZ,
    },
    c2fn::C2Fn,
    gauss_bonnet::{
        GaussBonnetBInput, GaussBonnetBackgroundState, GaussBonnetField, GaussBonnetFieldSimulator,
        GaussBonnetScalarPerturbationCoef, GaussBonnetScalarPerturbationPotential,
    },
    lat::{BoxLattice, Lattice, LatticeParam},
    models::TanhPotential,
    scalar::{construct_zeta_inplace, spectrum_with_scratch},
    util::{
        self, Hms, ParamRange, RateLimiter, TimeEstimator, VecN, lazy_file_opt, limit_length,
        linear_interp,
    },
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

#[derive(Debug, Clone, Copy, Encode, Decode)]
struct LatticeMeasurables {
    pub a: f64,
    pub v_a: f64,
    pub phi: f64,
    pub v_phi: f64,
    pub hubble_constraint: f64,
    pub metric_perts: (f64, f64),
}

#[derive(Encode, Decode)]
struct LatticeOutputData<const D: usize> {
    pub measurables: Vec<LatticeMeasurables>,
    pub final_state: GaussBonnetField<D>,
    pub spectrum_k: Vec<f64>,
    pub spectrums: Vec<(f64, Vec<f64>)>,
    pub zeta_spectrum: Vec<f64>,
    pub final_zeta: BoxLattice<D, f64>,
}

#[derive(Debug)]
struct LatticeInput<const D: usize> {
    pub lattice: LatticeParam<D>,
    pub a: f64,
    pub v_a: f64,
    pub phi: f64,
    pub v_phi: f64,
    pub dt: f64,
    pub end_n: f64,
    pub k_star: f64,
}

impl<const D: usize> LatticeInput<D> {
    pub fn from_background_and_k(
        background: &[GaussBonnetBackgroundState],
        start_k: f64,
        horizon_tolerance: f64,
        lattice_size: usize,
        dt: f64,
    ) -> Self {
        let starting_horizon = start_k / horizon_tolerance;
        let end_k = start_k * sqrt(D as f64) / PI * (lattice_size as f64);
        let end_horizon = end_k * horizon_tolerance;
        let dx = 2.0 * PI / start_k / (lattice_size as f64);
        let lattice = LatticeParam {
            size: VecN::new([lattice_size; D]),
            spacing: VecN::new([dx; D]),
        };
        let start_state = background
            .iter()
            .find(|state| state.v_a >= starting_horizon)
            .unwrap();
        let end_state = background
            .iter()
            .find(|state| state.v_a >= end_horizon)
            .unwrap();
        Self {
            lattice,
            a: start_state.a,
            v_a: start_state.v_a,
            phi: start_state.phi,
            v_phi: start_state.v_phi,
            dt,
            end_n: end_state.a.ln(),
            k_star: 1.0,
        }
    }
    pub fn from_background_and_k_normalized(
        background: &[GaussBonnetBackgroundState],
        start_k: f64,
        horizon_tolerance: f64,
        lattice_size: usize,
        dt: f64,
    ) -> Self {
        let starting_horizon = start_k / horizon_tolerance;
        let end_k = start_k * sqrt(D as f64) / PI * (lattice_size as f64);
        let end_horizon = end_k * horizon_tolerance;
        let start_state = background
            .iter()
            .find(|state| state.v_a >= starting_horizon)
            .unwrap();
        let start_k_state = background
            .iter()
            .find(|state| state.v_a >= start_k)
            .unwrap();
        let end_state = background
            .iter()
            .find(|state| state.v_a >= end_horizon)
            .unwrap();
        let start_hubble = start_state.v_a / start_state.a;
        let normalized_start_k = start_k_state.v_a / start_state.a;
        let dx = 2.0 * PI / normalized_start_k / (lattice_size as f64);
        let lattice = LatticeParam {
            size: VecN::new([lattice_size; D]),
            spacing: VecN::new([dx; D]),
        };
        Self {
            lattice,
            a: 1.0,
            v_a: start_hubble,
            phi: start_state.phi,
            v_phi: start_state.v_phi,
            dt,
            end_n: (end_state.a / start_state.a).ln(),
            k_star: start_k / normalized_start_k,
        }
    }
    pub fn run<V, Xi>(
        &self,
        out_file: &str,
        create: bool,
        input: &GaussBonnetBInput<V, Xi>,
        spectrum_count: usize,
    ) -> util::Result<LatticeOutputData<D>>
    where
        V: C2Fn<f64, Output = f64> + Sync,
        Xi: C2Fn<f64, Output = f64> + Sync,
    {
        lazy_file_opt(out_file, BINCODE_CONFIG, create, || {
            println!("[lattice] input = {:?}", self);
            let checkpoint_file_name = format!("{}.checkpoint.bincode", out_file);
            let mut simulator = GaussBonnetFieldSimulator::new(&self.lattice, input, {
                GaussBonnetField::from_file(&checkpoint_file_name)
                    .inspect(|_| {
                        println!(
                            "[lattice] read from previous saved state {}",
                            &checkpoint_file_name
                        );
                    })
                    .unwrap_or_else(|_| {
                        let mut lattice_state = GaussBonnetField::zero(self.lattice.size);
                        lattice_state.init(self.a, self.phi, self.v_phi, input);
                        lattice_state.populate_noise(
                            &mut random::default(1),
                            input,
                            &self.lattice,
                            Some(self.v_a / self.a),
                        );
                        lattice_state
                    })
            });
            println!(
                "[lattice] background H = {}, initial H = {}, dx = {}",
                self.v_a / self.a,
                simulator.field.v_a / simulator.field.a,
                self.lattice.spacing[0],
            );
            let mut spectrum_scratch = BoxLattice::zeros(self.lattice.size);
            let initial_spectrum = spectrum_with_scratch(
                &simulator.field.phi.view().map(|f| f[0]),
                &self.lattice,
                &mut spectrum_scratch,
            );
            let spectrum_k = initial_spectrum.iter().map(|f| f.0 * self.k_star).collect();
            let mut spectrums = vec![(self.a.ln(), {
                let zeta_factor = simulator.field.zeta_factor();
                initial_spectrum
                    .iter()
                    .map(|f| f.1 * zeta_factor * zeta_factor)
                    .collect()
            })];
            let n_range = self.a.ln()..self.end_n;
            let spectrum_delta_n = (n_range.end - n_range.start) / (spectrum_count as f64);
            let mut next_spectrum_n = n_range.start + spectrum_delta_n;
            let mut measurables = vec![];
            let mut rate_limiter = RateLimiter::new(Duration::from_millis(2000));
            let mut time_estimator = TimeEstimator::new(n_range.clone(), 100);
            let mut last_checkpoint_time = SystemTime::now();
            while simulator.field.a.ln() < n_range.end {
                simulator.update(self.dt);
                time_estimator.update(simulator.field.a.ln());
                if let Ok(elapsed) = last_checkpoint_time.elapsed() {
                    if elapsed.as_secs() >= 600 {
                        match simulator.field.to_file(&checkpoint_file_name) {
                            Ok(_) => {
                                println!("[lattice] saved checkpoint {}", &checkpoint_file_name)
                            }
                            Err(err) => println!(
                                "[lattice] failed to save checkpoint {}: {}",
                                &checkpoint_file_name, &err
                            ),
                        }
                        last_checkpoint_time = SystemTime::now();
                    }
                } else {
                    last_checkpoint_time = SystemTime::now();
                }
                let state = LatticeMeasurables {
                    a: simulator.field.a,
                    v_a: simulator.field.v_a,
                    phi: simulator.field.phi.view().map(|f| f[0]).average(),
                    v_phi: simulator.field.phi.view().map(|f| f[1]).average(),
                    metric_perts: simulator.field.metric_perturbations(input, &self.lattice),
                    hubble_constraint: simulator.field.hubble_constraint(input, &self.lattice),
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
                    let zeta_factor = simulator.field.zeta_factor();
                    let spec = spectrum_with_scratch(
                        &simulator.field.phi.as_ref().map(|f| f[0]),
                        &self.lattice,
                        &mut spectrum_scratch,
                    );
                    spectrums.push((
                        simulator.field.a.ln(),
                        spec.iter()
                            .map(|f| f.1 * zeta_factor * zeta_factor)
                            .collect(),
                    ));
                }
            }
            let mut zeta = BoxLattice::zeros(self.lattice.size);
            let zeta_spectrum = {
                let percentage = AtomicUsize::new(0);
                let denom = 100usize;
                let reference_phi_old = simulator.field.phi.as_ref().map(|f| f[0]).max().1;
                let reference_phi = {
                    let a = simulator.field.a;
                    let v_a = simulator.field.v_a;
                    let coord = simulator
                        .field
                        .phi
                        .as_ref()
                        .map(move |f| input.scalar_eff_potential(a, v_a, f[0], f[1]))
                        .min()
                        .0;
                    simulator.field.phi.get_by_coord(&coord)[0]
                };
                println!(
                    "[lattice] calculating spectrum, reference_phi = {}, reference_phi_old = {}",
                    reference_phi, reference_phi_old
                );
                construct_zeta_inplace(
                    &mut zeta,
                    simulator.field.a,
                    simulator.field.v_a,
                    &simulator.field.phi.as_ref().map(|f| f[0]),
                    &simulator.field.phi.as_ref().map(|f| f[1]),
                    reference_phi,
                    100,
                    10.0,
                    input,
                    |count, total| {
                        let p = ((count as f64) / (total as f64) * (denom as f64)).floor() as usize;
                        let c = percentage.fetch_max(p, std::sync::atomic::Ordering::SeqCst);
                        if p != c {
                            println!("[zeta] {:.2}%", (p as f64) / ((denom / 100) as f64));
                        }
                    },
                );
                spectrum_with_scratch(&zeta, &self.lattice, &mut spectrum_scratch)
                    .iter()
                    .map(|f| f.1)
                    .collect()
            };
            LatticeOutputData {
                measurables,
                spectrum_k,
                spectrums,
                final_state: simulator.field,
                final_zeta: zeta,
                zeta_spectrum,
            }
        })
    }
}

impl<const D: usize> LatticeOutputData<D> {
    pub fn plot_background(&self, out_file: &str) {
        let mut efolding = vec![];
        let mut phi = vec![];
        let mut v_phi = vec![];
        let mut hubble = vec![];
        let mut pert_a = vec![];
        let mut pert_b = vec![];
        let mut hubble_constraint = vec![];
        for state in limit_length(&self.measurables, 500000) {
            efolding.push(state.a.ln());
            phi.push(state.phi);
            v_phi.push(state.v_phi);
            hubble.push(state.v_a / state.a);
            pert_a.push(state.metric_perts.0.abs());
            pert_b.push(state.metric_perts.1.abs());
            hubble_constraint.push(state.hubble_constraint.abs());
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
        plot.add_trace(
            Scatter::new(efolding.clone(), hubble_constraint)
                .name("hubble_constraint")
                .y_axis("y6"),
        );
        plot.set_layout(
            Layout::new()
                .grid(LayoutGrid::new().rows(6).columns(1))
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
                .y_axis6(
                    Axis::new()
                        .type_(AxisType::Log)
                        .exponent_format(ExponentFormat::Power),
                )
                .height(1200),
        );
        plot.write_html(out_file);
    }
    pub fn plot_spectrums(&self, out_file: &str, k_star: f64) {
        let mut plot = Plot::new();
        let ks = self
            .spectrum_k
            .iter()
            .map(|k| k * k_star)
            .collect::<Vec<_>>();
        for (n, spec) in &self.spectrums {
            plot.add_trace(Scatter::new(ks.clone(), spec.clone()).name(&format!("N = {}", n)));
        }
        plot.add_trace(
            Scatter::new(ks, self.zeta_spectrum.clone())
                .name("final")
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
                .height(1600),
        );
        plot.write_html(out_file);
    }
    pub fn plot_all(
        &self,
        out_dir: &str,
        name: &str,
        k_star: f64,
        spectrum_k_range: ParamRange<f64>,
        pert_spectrum: &[f64],
    ) {
        self.plot_background(&format!("{}/{}.lattice.background.html", out_dir, name));
        self.plot_spectrums(
            &format!("{}/{}.lattice.spectrums.html", out_dir, name),
            k_star,
        );
        plot_spectrum(
            &format!("{}/{}.lattice.combined_spectrum.html", out_dir, name),
            &[
                (
                    "tree",
                    &spectrum_k_range.as_logspace().collect::<Vec<_>>(),
                    pert_spectrum,
                ),
                (
                    "lattice",
                    remove_first_end_last(&self.spectrum_k),
                    remove_first_end_last(&self.zeta_spectrum),
                ),
            ],
            k_star,
        );
    }
}

pub fn plot_background<V, Xi>(
    out_file: &str,
    data: &[GaussBonnetBackgroundState],
    input: &GaussBonnetBInput<V, Xi>,
) where
    V: C2Fn<f64, Output = f64>,
    Xi: C2Fn<f64, Output = f64>,
{
    let max_length = 500000usize;
    let mut efolding = vec![];
    let mut phi = vec![];
    let mut v_phi = vec![];
    let mut hubble = vec![];
    let mut epsilon = vec![];
    let mut hubble_constraint = vec![];
    let mut mom_coef = vec![];
    let mut horizon = vec![];
    let mut amp_as = vec![];
    for state in limit_length(data, max_length) {
        let hubble0 = state.v_a / state.a;
        efolding.push(state.a.ln());
        phi.push(state.phi);
        v_phi.push(state.v_phi);
        hubble.push(hubble0);
        epsilon.push(state.epsilon(input));
        hubble_constraint.push(state.hubble_constraint(input).abs());
        mom_coef.push(state.pert_c / state.pert_a);
        horizon.push(state.horizon.sqrt());
        amp_as.push(1.0 / 8.0 / PI / PI / state.epsilon(input) / input.kappa * hubble0 * hubble0);
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
    plot.add_trace(
        Scatter::new(efolding.clone(), amp_as)
            .name("A_s")
            .y_axis("y8"),
    );
    plot.set_layout(
        Layout::new()
            .grid(LayoutGrid::new().rows(8).columns(1))
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
            .y_axis8(Axis::new().exponent_format(ExponentFormat::Power))
            .height(1600),
    );
    plot.write_html(out_file);
}

fn plot_spectrum(out_file: &str, spectrums: &[(&str, &[f64], &[f64])], k_star: f64) {
    let mut plot = Plot::new();
    for (name, k, spec) in spectrums.iter().cloned() {
        plot.add_trace(
            Scatter::new(k.iter().map(|k| k * k_star).collect(), spec.into())
                .name(name.to_string()),
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
            )
            .height(800),
    );
    plot.write_html(out_file);
}

fn calculate_background<V, Xi>(
    input: &GaussBonnetBInput<V, Xi>,
    a0: f64,
    phi0: f64,
    dt: f64,
    quiet: bool,
) -> Vec<GaussBonnetBackgroundState>
where
    V: C2Fn<f64, Output = f64>,
    Xi: C2Fn<f64, Output = f64>,
{
    let mut rate_limiter = RateLimiter::new(Duration::from_millis(100));
    let mut state = GaussBonnetBackgroundState::init_slowroll3d(input, a0, phi0);
    state.dt = dt;
    let mut ret = vec![state];
    while state.epsilon(input) < 1.0 {
        if !quiet {
            rate_limiter.run(|| {
                println!(
                    "[background] (epsilon = {}) {:?}",
                    state.epsilon(input),
                    &state
                )
            });
        }
        state = state.update(input, dt);
        state.dt = dt;
        ret.push(state);
    }
    GaussBonnetBackgroundState::calculate_pert_coefs(&mut ret, input);
    ret
}

fn run_spectrum<V, Xi>(
    out_file: &str,
    background: &[GaussBonnetBackgroundState],
    input: &GaussBonnetBInput<V, Xi>,
    k_range: ParamRange<f64>,
    da: f64,
    horizon_tolerance: f64,
    quiet: bool,
) -> util::Result<Vec<f64>>
where
    V: C2Fn<f64, Output = f64> + Sync,
    Xi: C2Fn<f64, Output = f64> + Sync,
{
    let pert = HamitonianSimulator::new(
        input,
        background.len(),
        background,
        DefaultPerturbationInitializer,
        GaussBonnetScalarPerturbationPotential,
        HorizonSelector::new(horizon_tolerance),
        GaussBonnetScalarPerturbationCoef,
    );
    pert.spectrum_with_cache(out_file, k_range, da, quiet)
}

#[allow(unused)]
fn run_perturbation<V, Xi>(
    out_file: &str,
    background: &[GaussBonnetBackgroundState],
    input: &GaussBonnetBInput<V, Xi>,
    ks: &[f64],
    k_unit: f64,
    da: f64,
    horizon_tolerance: f64,
) {
    let pert = HamitonianSimulator::new(
        input,
        background.len(),
        background,
        DefaultPerturbationInitializer,
        GaussBonnetScalarPerturbationPotential,
        HorizonSelector::new(horizon_tolerance),
        GaussBonnetScalarPerturbationCoef,
    );
    let max_length = 500000usize;
    let mut plot = Plot::new();
    for k0 in ks {
        let mut efolding = vec![];
        let mut phi = vec![];
        let mut potential = vec![];
        pert.run(k0 * k_unit, da, |_, b, _, s, pot, _| {
            efolding.push(b.a.ln());
            phi.push(s.abs());
            potential.push(pot);
        });
        let efolding = limit_length(&efolding, max_length)
            .cloned()
            .collect::<Vec<_>>();
        plot.add_trace(
            Scatter::new(
                efolding.clone(),
                limit_length(&phi, max_length).cloned().collect(),
            )
            .name(&format!("k = {:e}", *k0)),
        );
        plot.add_trace(
            Scatter::new(
                efolding.clone(),
                limit_length(&potential, max_length).cloned().collect(),
            )
            .name(&format!("k = {:e}", *k0))
            .y_axis("y2"),
        );
    }
    plot.set_layout(
        Layout::new()
            .grid(LayoutGrid::new().rows(2).columns(1))
            .x_axis(Axis::new().exponent_format(ExponentFormat::Power))
            .y_axis(
                Axis::new()
                    .type_(AxisType::Log)
                    .exponent_format(ExponentFormat::Power),
            )
            .y_axis2(Axis::new().exponent_format(ExponentFormat::Power))
            .height(1000),
    );
    plot.write_html(out_file);
}

fn param_set1(
    phi_c: f64,
    xi0: f64,
    xi1: f64,
) -> GaussBonnetBInput<NaturalInflationPotential, TanhPotential> {
    let lambda = 0.0065;
    let lambda4 = lambda * lambda * lambda * lambda;
    let f = 7.0;
    // let phi_c = 13.0;
    // let xi0 = 6.044e7;
    GaussBonnetBInput {
        dim: 3,
        kappa: 1.0,
        v: NaturalInflationPotential { lambda4, f },
        xi: TanhPotential {
            coef: xi0,
            omega: xi1,
            shift: -xi1 * phi_c,
        },
    }
}

fn run_common<V, Xi>(
    out_dir: &str,
    name: &str,
    input: &GaussBonnetBInput<V, Xi>,
    background_quiet: bool,
    remote: bool,
    misc_tests: bool,
    skip_lattice: bool,
    lattice_k0: f64,
) -> anyhow::Result<()>
where
    V: C2Fn<f64, Output = f64> + Sync,
    Xi: C2Fn<f64, Output = f64> + Sync,
{
    create_dir_all(out_dir)?;
    println!("[run_common] working with {}/{}", out_dir, name);
    let background = calculate_background(input, 1.0, 7.0, 1.0, background_quiet);
    plot_background(
        &format!("{}/{}.background.html", out_dir, name),
        &background,
        input,
    );
    let spectrum_k_range = ParamRange::new(1e-2, 1e25, 1000);
    let pert_spectrum = run_spectrum(
        &format!("{}/{}.spectrum.bincode", out_dir, name),
        &background,
        input,
        spectrum_k_range,
        0.01,
        1e3,
        false,
    )?;
    let pivot_k_index = pert_spectrum
        .iter()
        .cloned()
        .zip(0usize..)
        .find(|(f, _)| *f <= 2.9e-9)
        .unwrap()
        .1;
    let k_star = 0.05 * MPC_HZ / spectrum_k_range.log_interp(pivot_k_index);
    plot_spectrum(
        &format!("{}/{}.spectrum.html", out_dir, name),
        &[(
            "tree",
            &spectrum_k_range.as_logspace().collect::<Vec<_>>(),
            &pert_spectrum,
        )],
        k_star,
    );

    if misc_tests {
        run_perturbation(
            &format!("{}/{}.misc_test.perturbation.html", out_dir, name),
            &background,
            input,
            &[1e-7, 1e-8, 1e-9],
            1.0 / k_star,
            0.1,
            1e3,
        );
    }

    if !skip_lattice {
        if misc_tests {
            let lat_test = LatticeInput::<3>::from_background_and_k_normalized(
                &background,
                1e-9 / k_star,
                40.0,
                128,
                1.0,
            )
            .run(
                &format!("{}/{}.lattice.small_mom_test.bincode", out_dir, name),
                true,
                input,
                10,
            )?;
            lat_test.plot_spectrums(
                &format!("{}/{}.lattice.small_mom_test.spectrums.html", out_dir, name),
                k_star,
            );
            plot_spectrum(
                &format!(
                    "{}/{}.lattice.small_mom_test.combined_spectrum.html",
                    out_dir, name
                ),
                &[
                    (
                        "tree",
                        &spectrum_k_range.as_logspace().collect::<Vec<_>>(),
                        &pert_spectrum,
                    ),
                    (
                        "lattice",
                        remove_first_end_last(&lat_test.spectrum_k),
                        remove_first_end_last(&lat_test.zeta_spectrum),
                    ),
                ],
                k_star,
            );

            let lat_test = LatticeInput::<3>::from_background_and_k_normalized(
                &background,
                1e-8 / k_star,
                40.0,
                128,
                1.0,
            )
            .run(
                &format!("{}/{}.lattice.small_mom_test.2.bincode", out_dir, name),
                true,
                input,
                10,
            )?;
            lat_test.plot_spectrums(
                &format!(
                    "{}/{}.lattice.small_mom_test.2.spectrums.html",
                    out_dir, name
                ),
                k_star,
            );
            plot_spectrum(
                &format!(
                    "{}/{}.lattice.small_mom_test.2.combined_spectrum.html",
                    out_dir, name
                ),
                &[
                    (
                        "tree",
                        &spectrum_k_range.as_logspace().collect::<Vec<_>>(),
                        &pert_spectrum,
                    ),
                    (
                        "lattice",
                        remove_first_end_last(&lat_test.spectrum_k),
                        remove_first_end_last(&lat_test.zeta_spectrum),
                    ),
                ],
                k_star,
            );

            let lat_test = LatticeInput::<3>::from_background_and_k_normalized(
                &background,
                1e-7 / k_star,
                40.0,
                128,
                1.0,
            )
            .run(
                &format!("{}/{}.lattice.small_mom_test.5.bincode", out_dir, name),
                true,
                input,
                10,
            )?;
            lat_test.plot_spectrums(
                &format!(
                    "{}/{}.lattice.small_mom_test.5.spectrums.html",
                    out_dir, name
                ),
                k_star,
            );
            plot_spectrum(
                &format!(
                    "{}/{}.lattice.small_mom_test.5.combined_spectrum.html",
                    out_dir, name
                ),
                &[
                    (
                        "tree",
                        &spectrum_k_range.as_logspace().collect::<Vec<_>>(),
                        &pert_spectrum,
                    ),
                    (
                        "lattice",
                        remove_first_end_last(&lat_test.spectrum_k),
                        remove_first_end_last(&lat_test.zeta_spectrum),
                    ),
                ],
                k_star,
            );

            let lat_test = LatticeInput::<3>::from_background_and_k_normalized(
                &background,
                1e-3 / k_star,
                40.0,
                128,
                1.0,
            )
            .run(
                &format!("{}/{}.lattice.small_mom_test.8.bincode", out_dir, name),
                true,
                input,
                10,
            )?;
            lat_test.plot_spectrums(
                &format!(
                    "{}/{}.lattice.small_mom_test.8.spectrums.html",
                    out_dir, name
                ),
                k_star,
            );
            plot_spectrum(
                &format!(
                    "{}/{}.lattice.small_mom_test.8.combined_spectrum.html",
                    out_dir, name
                ),
                &[
                    (
                        "tree",
                        &spectrum_k_range.as_logspace().collect::<Vec<_>>(),
                        &pert_spectrum,
                    ),
                    (
                        "lattice",
                        remove_first_end_last(&lat_test.spectrum_k),
                        remove_first_end_last(&lat_test.zeta_spectrum),
                    ),
                ],
                k_star,
            );
        }
        if let Ok(lat_remote) = LatticeInput::<3>::from_background_and_k_normalized(
            &background,
            lattice_k0 / k_star,
            10.0,
            256,
            2.0,
        )
        .run(
            &format!("{}/{}.lattice.remote.bincode", out_dir, name),
            remote,
            input,
            10,
        ) {
            lat_remote.plot_background(&format!(
                "{}/{}.lattice.remote.background.html",
                out_dir, name
            ));
            lat_remote.plot_spectrums(
                &format!("{}/{}.lattice.remote.spectrums.html", out_dir, name),
                k_star,
            );
            plot_spectrum(
                &format!("{}/{}.lattice.remote.combined_spectrum.html", out_dir, name),
                &[
                    (
                        "tree",
                        &spectrum_k_range.as_logspace().collect::<Vec<_>>(),
                        &pert_spectrum,
                    ),
                    (
                        "lattice",
                        remove_first_end_last(&lat_remote.spectrum_k),
                        remove_first_end_last(&lat_remote.zeta_spectrum),
                    ),
                ],
                k_star,
            );
        }
    }

    Ok(())
}

struct ScanLatticeInput {
    pub k0: f64,
    pub dt: f64,
    pub size: usize,
}

fn run_scan<F, V, Xi>(
    out_dir: &str,
    name: &str,
    linear_spectrum_k_range: ParamRange<f64>,
    lat: Option<ScanLatticeInput>,
    mut provider: F,
    count: usize,
) -> util::Result<()>
where
    F: FnMut(f64) -> GaussBonnetBInput<V, Xi>,
    V: C2Fn<f64, Output = f64> + Sync,
    Xi: C2Fn<f64, Output = f64> + Sync,
{
    println!("[run_scan] working with {}/{}", out_dir, name);
    create_dir_all(out_dir)?;
    let mut linear_spectrums = vec![];
    let mut lattice_spectrums = vec![];
    for i in 0..count {
        let param = provider((i as f64) / ((count - 1) as f64));
        let background = calculate_background(&param, 1.0, 7.0, 1.0, true);
        let linear_spectrum = run_spectrum(
            &format!("{}/{}.scan.linear_spectrum.{}.bincode", out_dir, name, i),
            &background,
            &param,
            linear_spectrum_k_range,
            0.1,
            1e3,
            true,
        )?;
        let pivot_k_index = linear_spectrum
            .iter()
            .cloned()
            .zip(0usize..)
            .find(|(f, _)| *f <= 2.9e-9)
            .unwrap()
            .1;
        let k_star = 0.05 * MPC_HZ / linear_spectrum_k_range.log_interp(pivot_k_index);
        linear_spectrums.push((
            (linear_spectrum_k_range * k_star)
                .as_logspace()
                .collect::<Vec<_>>(),
            linear_spectrum.clone(),
        ));
        println!("[scan/linear_spectrum] {}/{}", i + 1, count);
        if let Some(lat_params) = &lat {
            let lat = LatticeInput::<3>::from_background_and_k_normalized(
                &background,
                lat_params.k0 / k_star,
                40.0,
                lat_params.size,
                lat_params.dt,
            )
            .run(
                &format!("{}/{}.scan.lattice.{}.bincode", out_dir, name, i),
                true,
                &param,
                10,
            )?;
            lattice_spectrums.push((
                remove_first_end_last(&lat.spectrum_k).to_vec(),
                remove_first_end_last(&lat.zeta_spectrum).to_vec(),
            ));
            plot_spectrum(
                &format!("{}/{}.scan.lattice.spectrum.{}.html", out_dir, name, i),
                &[
                    (
                        "tree",
                        &linear_spectrum_k_range.as_logspace().collect::<Vec<_>>(),
                        &linear_spectrum,
                    ),
                    (
                        "lattice",
                        remove_first_end_last(&lat.spectrum_k),
                        remove_first_end_last(&lat.zeta_spectrum),
                    ),
                ],
                k_star,
            );
        }
        println!("[scan] ({}/{}) done", i + 1, count);
    }
    {
        let mut plot = Plot::new();
        for i in 0..count {
            let (k, spectrum) = &linear_spectrums[i];
            plot.add_trace(
                Scatter::new(k.clone(), spectrum.clone()).name(&format!("linear {}", i)),
            );
            if lat.is_some() {
                let (lat_k, lat_spectrum) = &lattice_spectrums[i];
                plot.add_trace(
                    Scatter::new(lat_k.clone(), lat_spectrum.clone())
                        .name(&format!("lattice {}", i)),
                );
            }
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
                )
                .height(800),
        );
        plot.write_html(&format!("{}/{}.scans.linear_spectrum.html", out_dir, name));
    }
    Ok(())
}

fn remove_first_end_last<T>(arr: &[T]) -> &[T] {
    &arr[..arr.len() - 1]
}

fn piecewise_linear_interp(a: f64, b: f64, c: f64, l: f64, l1: f64) -> f64 {
    if l <= l1 {
        linear_interp(a, b, l / l1)
    } else {
        linear_interp(b, c, (l - l1) / (1.0 - l1))
    }
}

pub fn main() {
    let is_remote = env::var("REMOTE").map(|v| v != "false").unwrap_or(false);
    // run_common(
    //     "out/gauss_bonnet",
    //     "set1.xi01.000",
    //     &param_set1(9.6, 1.000e7, 30.0),
    //     true,
    //     is_remote,
    //     false,
    //     false,
    //     1e-9,
    // )
    // .unwrap();
    // run_common(
    //     "out/gauss_bonnet",
    //     "set1.xi00.800",
    //     &param_set1(9.6, 0.800e7, 30.0),
    //     true,
    //     is_remote,
    //     false,
    //     true,
    //     1e-9,
    // )
    // .unwrap();
    // run_common(
    //     "out/gauss_bonnet",
    //     "set1.xi00.200",
    //     &param_set1(9.6, 0.200e7, 30.0),
    //     true,
    //     is_remote,
    //     true,
    //     false,
    //     1e-9,
    // )
    // .unwrap();
    // run_common(
    //     "out/gauss_bonnet",
    //     "set1.xi00.200.small_k",
    //     &param_set1(9.6, 0.200e7, 30.0),
    //     true,
    //     is_remote,
    //     false,
    //     false,
    //     1e-12,
    // )
    // .unwrap();
    // run_common(
    //     "out/gauss_bonnet",
    //     "set1.xi01.133",
    //     &param_set1(9.6, 1.133e7, 30.0),
    //     true,
    //     is_remote,
    //     false,
    //     false,
    //     1e-9,
    // )
    // .unwrap();
    // run_common(
    //     "out/gauss_bonnet",
    //     "set1.xi01.134",
    //     &param_set1(9.6, 1.134e7, 30.0),
    //     true,
    //     is_remote,
    //     true,
    //     false,
    //     1e-9,
    // )
    // .unwrap();
    // run_common(
    //     "out/gauss_bonnet",
    //     "set1.xi01.13417",
    //     &param_set1(9.6, 1.13417e7, 30.0),
    //     true,
    //     is_remote,
    //     false,
    //     true,
    //     1e-9,
    // )
    // .unwrap();

    // run_scan(
    //     "out/gauss_bonnet",
    //     "scan_set1",
    //     ParamRange::new(1e-2, 1e25, 1000),
    //     Some(ScanLatticeInput { k0: 1e-9, dt: 5.0, size: 256 }),
    //     |l| param_set1(9.6, piecewise_linear_interp(1e7, 11294758.620689655, 1.1341e7, l, 0.5), 30.0),
    //     30,
    // )
    // .unwrap();
    run_scan(
        "out/gauss_bonnet",
        "scan_set1_dt2",
        ParamRange::new(1e-2, 1e25, 1000),
        Some(ScanLatticeInput {
            k0: 1e-9,
            dt: 2.0,
            size: 256,
        }),
        |l| {
            param_set1(
                9.6,
                piecewise_linear_interp(1e7, 11294758.620689655, 1.134e7, l, 0.5),
                30.0,
            )
        },
        30,
    )
    .unwrap();
    // run_scan(
    //     "out/gauss_bonnet",
    //     "scan_set1_dt2",
    //     ParamRange::new(1e-2, 1e25, 1000),
    //     Some(ScanLatticeInput {
    //         k0: 1e-9,
    //         dt: 2.0,
    //         size: 256,
    //     }),
    //     |l| {
    //         param_set1(
    //             9.6,
    //             piecewise_linear_interp(1e7, 11294758.620689655, 1.134e7, l, 0.5),
    //             30.0,
    //         )
    //     },
    //     30,
    // )
    // .unwrap();
}
