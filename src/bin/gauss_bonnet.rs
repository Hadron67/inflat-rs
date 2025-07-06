use std::{f64::consts::PI, fs::create_dir_all, iter::zip, time::Duration};

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
    scalar::{construct_zeta_inplace, spectrum_with_scratch},
    util::{self, Hms, ParamRange, RateLimiter, TimeEstimator, VecN, lazy_file, limit_length},
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
struct LatticeOutputData {
    pub measurables: Vec<LatticeMeasurables>,
    pub final_state: GaussBonnetField<3>,
    pub spectrum_k: Vec<f64>,
    pub spectrums: Vec<(f64, Vec<f64>)>,
    pub final_zeta: BoxLattice<3, f64>,
}

impl LatticeOutputData {
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
    pub fn plot_spectrums(&self, out_file: &str) {
        let mut names = vec![];
        let mut specs = vec![];
        for (n, _) in &self.spectrums {
            names.push(format!("N = {}", n));
        }
        for ((_, spec), i) in zip(&self.spectrums, 0usize..) {
            specs.push((
                names[i].as_str(),
                self.spectrum_k.as_slice(),
                spec.as_slice(),
            ));
        }
        plot_spectrum(out_file, &specs);
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

fn plot_spectrum(out_file: &str, spectrums: &[(&str, &[f64], &[f64])]) {
    let mut plot = Plot::new();
    for (name, k, spec) in spectrums.iter().cloned() {
        plot.add_trace(Scatter::new(k.into(), spec.into()).name(name.to_string()));
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
    out_file: &str,
    input: &GaussBonnetBInput<V, Xi>,
    a0: f64,
    phi0: f64,
    dt: f64,
) -> util::Result<Vec<GaussBonnetBackgroundState>>
where
    V: C2Fn<f64, Output = f64>,
    Xi: C2Fn<f64, Output = f64>,
{
    let mut rate_limiter = RateLimiter::new(Duration::from_millis(100));
    lazy_file(out_file, BINCODE_CONFIG, || {
        let mut state = GaussBonnetBackgroundState::init_slowroll3d(input, a0, phi0);
        state.dt = dt;
        let mut ret = vec![state];
        while state.epsilon(input) < 1.0 {
            rate_limiter.run(|| {
                println!(
                    "[background] (epsilon = {}) {:?}",
                    state.epsilon(input),
                    &state
                )
            });
            state = state.update(input, dt);
            state.dt = dt;
            ret.push(state);
        }
        GaussBonnetBackgroundState::calculate_pert_coefs(&mut ret, input);
        ret
    })
}

fn run_spectrum<V, Xi>(
    out_file: &str,
    background: &[GaussBonnetBackgroundState],
    input: &GaussBonnetBInput<V, Xi>,
    k_range: ParamRange<f64>,
    da: f64,
    horizon_tolerance: f64,
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
    pert.spectrum_with_cache(out_file, k_range, da)
}

#[allow(unused)]
fn run_perturbation<V, Xi>(
    out_file: &str,
    background: &[GaussBonnetBackgroundState],
    input: &GaussBonnetBInput<V, Xi>,
    ks: &[f64],
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
    for k in ks {
        let mut efolding = vec![];
        let mut phi = vec![];
        let mut potential = vec![];
        pert.run(*k, da, |_, b, _, s, pot, _| {
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
            .name(&format!("k = {}", *k)),
        );
        plot.add_trace(
            Scatter::new(
                efolding.clone(),
                limit_length(&potential, max_length).cloned().collect(),
            )
            .name(&format!("k = {}", *k))
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

fn run_lattice<V, Xi>(
    out_file: &str,
    background: &[GaussBonnetBackgroundState],
    input: &GaussBonnetBInput<V, Xi>,
    start_k: f64,
    horizon_tolerance: f64,
    lattice_size: usize,
    spectrum_count: usize,
    dt: f64,
) -> util::Result<LatticeOutputData>
where
    V: C2Fn<f64, Output = f64> + Sync,
    Xi: C2Fn<f64, Output = f64> + Sync,
{
    let starting_horizon = start_k / horizon_tolerance;
    let end_k = start_k * sqrt(3.0) / PI * (lattice_size as f64);
    let end_horizon = end_k * horizon_tolerance;
    let dx = 2.0 * PI / start_k / (lattice_size as f64);
    let lattice = LatticeParam {
        size: VecN::new([lattice_size; 3]),
        spacing: VecN::new([dx; 3]),
    };
    lazy_file(out_file, BINCODE_CONFIG, || {
        println!("[lattice] dx = {}, k_range = [{}, {}]", dx, start_k, end_k);
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
        println!(
            "[lattice] initial H = {}",
            simulator.field.v_a / simulator.field.a
        );
        let mut spectrum_scratch = BoxLattice::zeros(lattice.size);
        let mut zeta = BoxLattice::zeros(lattice.size);
        let initial_spectrum = spectrum_with_scratch(
            &simulator.field.phi.view().map(|f| f[0]),
            &lattice,
            &mut spectrum_scratch,
        );
        let spectrum_k = initial_spectrum.iter().map(|f| f.0).collect();
        let mut spectrums = vec![(
            start_state.a.ln(),
            initial_spectrum.iter().map(|f| f.1).collect(),
        )];
        let spectrum_delta_n = (n_range.end - n_range.start) / (spectrum_count as f64);
        let mut next_spectrum_n = start_state.a.ln() + spectrum_delta_n;
        let mut measurables = vec![];
        let mut rate_limiter = RateLimiter::new(Duration::from_millis(100));
        let mut time_estimator = TimeEstimator::new(n_range.clone(), 100);
        while simulator.field.a.ln() < n_range.end {
            simulator.update(dt);
            time_estimator.update(simulator.field.a.ln());
            let state = LatticeMeasurables {
                a: simulator.field.a,
                v_a: simulator.field.v_a,
                phi: simulator.field.phi.view().map(|f| f[0]).average(),
                v_phi: simulator.field.phi.view().map(|f| f[1]).average(),
                metric_perts: simulator.field.metric_perturbations(input, &lattice),
                hubble_constraint: simulator.field.hubble_constraint(input, &lattice),
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
                println!("[lattice] calculating spectrum");
                construct_zeta_inplace(
                    &mut zeta,
                    simulator.field.a,
                    simulator.field.v_a,
                    &simulator.field.phi.as_ref().map(|f| f[0]),
                    &simulator.field.phi.as_ref().map(|f| f[1]),
                    simulator.field.phi.as_ref().map(|f| f[0]).max().1,
                    100,
                    10.0,
                    input,
                    |_, _| {},
                );
                let spec = spectrum_with_scratch(&zeta, &lattice, &mut spectrum_scratch);
                spectrums.push((simulator.field.a.ln(), spec.iter().map(|f| f.1).collect()));
            }
        }
        LatticeOutputData {
            measurables,
            spectrum_k,
            spectrums,
            final_state: simulator.field,
            final_zeta: zeta,
        }
    })
}

fn param_set1(xi1: f64) -> GaussBonnetBInput<NaturalInflationPotential, TanhPotential> {
    let lambda = 0.0065;
    let lambda4 = lambda * lambda * lambda * lambda;
    let f = 7.0;
    let phi_c = 13.0;
    let xi0 = 6.044e7;
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
) -> anyhow::Result<()>
where
    V: C2Fn<f64, Output = f64> + Sync,
    Xi: C2Fn<f64, Output = f64> + Sync,
{
    create_dir_all(out_dir)?;
    let background = calculate_background(
        &format!("{}/background.{}.bincode", out_dir, name),
        input,
        1.0,
        7.0,
        1.0,
    )?;
    plot_background(
        &format!("{}/background.{}.html", out_dir, name),
        &background,
        input,
    );
    let spectrum_k_range = ParamRange::new(1.0, 1e25, 1000);
    let pert_spectrum = run_spectrum(
        &format!("{}/spectrum.{}.bincode", out_dir, name),
        &background,
        input,
        spectrum_k_range,
        0.01,
        1e3,
    )?;
    plot_spectrum(
        &format!("{}/spectrum.{}.html", out_dir, name),
        &[(
            "tree",
            &spectrum_k_range.as_logspace().collect::<Vec<_>>(),
            &pert_spectrum,
        )],
    );
    let lat_remote = run_lattice(
        &format!("{}/remote.lattice.{}.bincode", out_dir, name),
        &background,
        input,
        1e17,
        10.0,
        512,
        10,
        10.0,
    )?;
    lat_remote.plot_background(&format!("{}/lattice.{}.background.html", out_dir, name));
    lat_remote.plot_spectrums(&format!("{}/lattice.{}.spectrums.html", out_dir, name));
    plot_spectrum(
        &format!("{}/lattice.{}.combined_spectrum.html", out_dir, name),
        &[(
            "tree",
            &spectrum_k_range.as_logspace().collect::<Vec<_>>(),
            &pert_spectrum,
        ), (
            "lattice",
            remove_first_end_last(&lat_remote.spectrum_k),
            remove_first_end_last(&lat_remote.spectrums.last().unwrap().1),
        )],
    );
    Ok(())
}

fn remove_first_end_last<T>(arr: &[T]) -> &[T] {
    &arr[1..arr.len() - 1]
}

pub fn main() {
    let param1 = param_set1(14.903);
    run_common("out/gauss_bonnet", "set1", &param1).unwrap();
}
