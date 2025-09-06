use std::{env, f64::consts::PI, fs::create_dir_all, time::Duration};

use bincode::{Decode, Encode};
use inflat::{
    background::{DefaultPerturbationInitializer, HamitonianSimulator, HorizonSelector, MPC_HZ},
    c2fn::C2Fn,
    gauss_bonnet::{
        GaussBonnetBInput, GaussBonnetBackgroundState, GaussBonnetField,
        GaussBonnetFieldSimulatorCreator, GaussBonnetScalarPerturbationCoef,
        GaussBonnetScalarPerturbationPotential,
    },
    lat::BoxLattice,
    models::TanhPotential,
    scalar::LatticeInput,
    util::{
        self, ParamRange, RateLimiter, limit_length, linear_interp, plot_spectrum,
        remove_first_and_last,
    },
};
use libm::{cos, sin};
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

type LI = LatticeInput<3, GaussBonnetBackgroundState>;

fn run_common<'a, V, Xi>(
    out_dir: &str,
    name: &str,
    input: &'a GaussBonnetBInput<V, Xi>,
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
            let lat_test = LI::from_background_and_k_normalized(
                &background,
                input,
                1e-9 / k_star,
                1.3,
                20.0,
                32,
                2.0,
            )
            .run(
                &GaussBonnetFieldSimulatorCreator,
                input,
                &format!("{}/{}.lattice.test1.bincode", out_dir, name),
                true,
                10,
            )?;
            lat_test.plot_spectrums(
                &format!("{}/{}.lattice.test1.spectrums.html", out_dir, name),
                k_star,
            );
            lat_test.plot_background(&format!("{}/{}.lattice.test1.background.html", out_dir, name));
            plot_spectrum(
                &format!(
                    "{}/{}.lattice.test1.combined_spectrum.html",
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
                        remove_first_and_last(&lat_test.spectrum_k),
                        remove_first_and_last(&lat_test.zeta_spectrum),
                    ),
                ],
                k_star,
            );
        }
        if let Ok(lat_remote) = LI::from_background_and_k_normalized(
            &background,
            input,
            lattice_k0 / k_star,
            1.3,
            20.0,
            128,
            2.0,
        )
        .run(
            &GaussBonnetFieldSimulatorCreator,
            input,
            &format!("{}/{}.lattice.remote.bincode", out_dir, name),
            remote,
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
                        remove_first_and_last(&lat_remote.spectrum_k),
                        remove_first_and_last(&lat_remote.zeta_spectrum),
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
            let lat = LI::from_background_and_k_normalized(
                &background,
                &param,
                lat_params.k0 / k_star,
                40.0,
                40.0,
                lat_params.size,
                lat_params.dt,
            )
            .run(
                &GaussBonnetFieldSimulatorCreator,
                &param,
                &format!("{}/{}.scan.lattice.{}.bincode", out_dir, name, i),
                true,
                10,
            )?;
            lattice_spectrums.push((
                remove_first_and_last(&lat.spectrum_k).to_vec(),
                remove_first_and_last(&lat.zeta_spectrum).to_vec(),
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
                        remove_first_and_last(&lat.spectrum_k),
                        remove_first_and_last(&lat.zeta_spectrum),
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
    run_common(
        "out/gauss_bonnet",
        "set1.xi01.13417",
        &param_set1(9.6, 1.13417e7, 30.0),
        true,
        is_remote,
        true,
        false,
        1e-9,
    )
    .unwrap();

    // run_scan(
    //     "out/gauss_bonnet",
    //     "scan_set1",
    //     ParamRange::new(1e-2, 1e25, 1000),
    //     Some(ScanLatticeInput { k0: 1e-9, dt: 5.0, size: 256 }),
    //     |l| param_set1(9.6, piecewise_linear_interp(1e7, 11294758.620689655, 1.1341e7, l, 0.5), 30.0),
    //     30,
    // )
    // .unwrap();
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
