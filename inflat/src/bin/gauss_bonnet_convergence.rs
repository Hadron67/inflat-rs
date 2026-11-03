//! Lattice convergence tests for the scalar Gauss-Bonnet model.
//!
//! This binary is intentionally independent of `gauss_bonnet.rs` so that the
//! production outputs are never touched.  By default it only prints the plan;
//! pass `--run` to compute missing configurations.  Existing results are read
//! back instead of being overwritten.
//!
//! The initial recommended step is:
//!
//! ```text
//! cargo run --release --bin gauss_bonnet_convergence -- --run
//! ```
//!
//! which runs the two resolution cases N=64 and N=128.  Other groups can be
//! selected with `--group box|timestep|times|seed|all`, and individual cases
//! with `--config <name>`.

use std::{
    env,
    f64::consts::PI,
    fs::{File, create_dir_all},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use inflat::{
    background::{
        BINCODE_CONFIG, DefaultPerturbationInitializer, HamitonianSimulator, HorizonSelector,
        MPC_HZ,
    },
    c2fn::C2Fn,
    gauss_bonnet::{
        GaussBonnetBInput, GaussBonnetBackgroundState, GaussBonnetField, GaussBonnetFieldSimulatorCreator,
        GaussBonnetScalarPerturbationCoef, GaussBonnetScalarPerturbationPotential,
    },
    igw::sigw_2_spectrum,
    lat::Lattice,
    models::TanhPotential,
    scalar::{LatticeInput, LatticeOutputData},
    util::{self, ParamRange, VecN, lazy_file},
};
use libm::{cos, sin};
use ndarray::{Array1, Array2, Array3};
use ndarray_npy::{NpzReader, NpzWriter};
use plotly::{
    Layout, Plot, Scatter,
    common::{DashType, ExponentFormat, Line},
    layout::{Axis, AxisType, LayoutGrid},
};

const XI0_DEFAULT: f64 = 1.13412e7;
const BASELINE_N: usize = 200;
const BASELINE_K0: f64 = 2e-9;
const BASELINE_DT: f64 = 2.0;
const BASELINE_SUB: f64 = 1.3;
const BASELINE_SUPER: f64 = 200.0;
const BASELINE_SEED: u64 = 10;
const SPECTRUM_COUNT: usize = 10;
const MAX_SUMMARY_MEASURABLES: usize = 10_000;

// Well-resolved comparison window for the relative-difference diagnostics.
// The highest-k bins are affected by the finite UV cutoff, especially at N=64.
const COMPARISON_K_MIN: f64 = 1e-8;
const COMPARISON_K_MAX_FRACTION: f64 = 0.7;

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

fn param_set1(
    phi_c: f64,
    xi0: f64,
    xi1: f64,
) -> GaussBonnetBInput<NaturalInflationPotential, TanhPotential> {
    let lambda = 0.0065;
    let lambda4 = lambda * lambda * lambda * lambda;
    let f = 7.0;
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

fn calculate_background<V, Xi>(
    input: &GaussBonnetBInput<V, Xi>,
    a0: f64,
    phi0: f64,
    dt: f64,
) -> Vec<GaussBonnetBackgroundState>
where
    V: C2Fn<f64, Output = f64>,
    Xi: C2Fn<f64, Output = f64>,
{
    let mut state = GaussBonnetBackgroundState::init_slowroll3d(input, a0, phi0);
    state.dt = dt;
    let mut ret = vec![state];
    while state.epsilon(input) < 1.0 {
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

type LI = LatticeInput<3, GaussBonnetBackgroundState>;

#[derive(Clone, Debug)]
struct Config {
    name: &'static str,
    group: &'static str,
    n: usize,
    /// Physical k_min in Hz.
    k0: f64,
    dt: f64,
    sub: f64,
    sup: f64,
    seed: u64,
}

/// The first run only uses N=64 and N=128.  The existing N=200 production
/// result is used as the baseline.
fn configs() -> Vec<Config> {
    vec![
        Config {
            name: "n64",
            group: "resolution",
            n: 64,
            k0: BASELINE_K0,
            dt: BASELINE_DT,
            sub: BASELINE_SUB,
            sup: BASELINE_SUPER,
            seed: BASELINE_SEED,
        },
        Config {
            name: "n128",
            group: "resolution",
            n: 128,
            k0: BASELINE_K0,
            dt: BASELINE_DT,
            sub: BASELINE_SUB,
            sup: BASELINE_SUPER,
            seed: BASELINE_SEED,
        },
        Config {
            name: "n300",
            group: "resolution",
            n: 300,
            k0: BASELINE_K0,
            dt: BASELINE_DT,
            sub: BASELINE_SUB,
            sup: BASELINE_SUPER,
            seed: BASELINE_SEED,
        },
        Config {
            name: "n400",
            group: "resolution",
            n: 400,
            k0: BASELINE_K0,
            dt: BASELINE_DT,
            sub: BASELINE_SUB,
            sup: BASELINE_SUPER,
            seed: BASELINE_SEED,
        },
        // Fixed Delta x, smaller boxes: N * k0 = 200 * 2e-9.
        Config {
            name: "n512",
            group: "resolution",
            n: 512,
            k0: BASELINE_K0,
            dt: BASELINE_DT,
            sub: BASELINE_SUB,
            sup: BASELINE_SUPER,
            seed: BASELINE_SEED,
        },
        // Fixed Delta x, smaller boxes: N * k0 = 200 * 2e-9.
        Config {
            name: "box_n128_k0_3p125e-9",
            group: "box",
            n: 128,
            k0: 3.125e-9,
            dt: BASELINE_DT,
            sub: BASELINE_SUB,
            sup: BASELINE_SUPER,
            seed: BASELINE_SEED,
        },
        Config {
            name: "box_n64_k0_6p25e-9",
            group: "box",
            n: 64,
            k0: 6.25e-9,
            dt: BASELINE_DT,
            sub: BASELINE_SUB,
            sup: BASELINE_SUPER,
            seed: BASELINE_SEED,
        },
        Config {
            name: "dt1_n64",
            group: "timestep",
            n: 64,
            k0: BASELINE_K0,
            dt: 1.0,
            sub: BASELINE_SUB,
            sup: BASELINE_SUPER,
            seed: BASELINE_SEED,
        },
        Config {
            name: "dt4_n128",
            group: "timestep",
            n: 128,
            k0: BASELINE_K0,
            dt: 4.0,
            sub: BASELINE_SUB,
            sup: BASELINE_SUPER,
            seed: BASELINE_SEED,
        },
        Config {
            name: "dt8_n128",
            group: "timestep",
            n: 128,
            k0: BASELINE_K0,
            dt: 8.0,
            sub: BASELINE_SUB,
            sup: BASELINE_SUPER,
            seed: BASELINE_SEED,
        },
        Config {
            name: "sub2_n128",
            group: "times",
            n: 128,
            k0: BASELINE_K0,
            dt: BASELINE_DT,
            sub: 2.0,
            sup: BASELINE_SUPER,
            seed: BASELINE_SEED,
        },
        Config {
            name: "super500_n128",
            group: "times",
            n: 128,
            k0: BASELINE_K0,
            dt: BASELINE_DT,
            sub: BASELINE_SUB,
            sup: 500.0,
            seed: BASELINE_SEED,
        },
        Config {
            name: "n400_super500",
            group: "times",
            n: 400,
            k0: BASELINE_K0,
            dt: BASELINE_DT,
            sub: BASELINE_SUB,
            sup: 500.0,
            seed: BASELINE_SEED,
        },
        Config {
            name: "seed11_n128",
            group: "seed",
            n: 128,
            k0: BASELINE_K0,
            dt: BASELINE_DT,
            sub: BASELINE_SUB,
            sup: BASELINE_SUPER,
            seed: 11,
        },
        Config {
            name: "seed12_n128",
            group: "seed",
            n: 128,
            k0: BASELINE_K0,
            dt: BASELINE_DT,
            sub: BASELINE_SUB,
            sup: BASELINE_SUPER,
            seed: 12,
        },
    ]
}

struct Baseline {
    k_star: f64,
    linear_k: Vec<f64>,
    linear_p: Vec<f64>,
    lat_k: Vec<f64>,
    lat_p: Vec<f64>,
    lat_sigw: Vec<f64>,
}

struct Loaded {
    cfg: Config,
    /// Physical k in Hz.
    k: Vec<f64>,
    p: Vec<f64>,
    p_lin: Vec<f64>,
    sigw: Vec<f64>,
    l_mpc: f64,
    dx_mpc: f64,
    k_min: f64,
    k_max: f64,
    h_dt: f64,
    k_peak: f64,
    p_peak: f64,
    p_peak_over_lin: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct RelStats {
    max: f64,
    median: f64,
    rms: f64,
}

struct Args {
    run: bool,
    plot_only: bool,
    list: bool,
    groups: Vec<String>,
    configs: Vec<String>,
    xi0: f64,
    baseline: Option<PathBuf>,
    dump_background: Option<PathBuf>,
}

fn parse_args() -> Result<Args> {
    let mut args = Args {
        run: false,
        plot_only: false,
        list: false,
        groups: Vec::new(),
        configs: Vec::new(),
        xi0: XI0_DEFAULT,
        baseline: None,
        dump_background: None,
    };

    let mut it = env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--run" => args.run = true,
            "--plot-only" => args.plot_only = true,
            "--list" => args.list = true,
            "--group" => args
                .groups
                .push(it.next().context("--group requires a value")?),
            "--config" => args
                .configs
                .push(it.next().context("--config requires a value")?),
            "--xi0" => {
                args.xi0 = it
                    .next()
                    .context("--xi0 requires a value")?
                    .parse()
                    .context("invalid --xi0 value")?
            }
            "--baseline" => {
                args.baseline = Some(PathBuf::from(
                    it.next().context("--baseline requires a path")?,
                ))
            }
            "--dump-background" => {
                args.dump_background = Some(PathBuf::from(
                    it.next().context("--dump-background requires a path")?,
                ))
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            _ => bail!("unknown argument `{arg}`; use --help"),
        }
    }

    if args.run && args.plot_only {
        bail!("--run and --plot-only are mutually exclusive");
    }

    Ok(args)
}

fn print_help() {
    println!(
        "gauss_bonnet_convergence -- lattice convergence tests for the SGB model\n\
         \n\
         Usage:\n\
           cargo run --release --bin gauss_bonnet_convergence -- --run [options]\n\
         \n\
         Options:\n\
           --run                 compute missing configurations (existing files are read, not overwritten)\n\
           --plot-only           load existing bincode files and regenerate plots/tables\n\
           --group <name>        resolution (default) | box | timestep | times | seed | all\n\
           --config <name>       run/load a single named configuration (repeatable)\n\
           --xi0 <value>         alpha value (default {XI0_DEFAULT:e})\n\
           --baseline <path>     existing baseline npz (default inflat/out/gauss_bonnet.bak/case1-final.output.npz)\n\
           --dump-background <path>  write downsampled background ln a, H, phi, v_phi and exit\n\
           --list                print the test plan and exit\n\
           --help                print this help"
    );
}

fn print_plan() {
    println!(
        "baseline (existing): N={BASELINE_N}, k0={BASELINE_K0:e} Hz, dt={BASELINE_DT}, \
         sub={BASELINE_SUB}, super={BASELINE_SUPER}, seed={BASELINE_SEED}"
    );
    println!(
        "{:<26} {:<12} {:>4} {:>12} {:>6} {:>6} {:>7} {:>5} {:>12} {:>12}",
        "config", "group", "N", "k0 [Hz]", "dt", "sub", "super", "seed", "L [pc]", "dx [pc]"
    );
    for c in configs() {
        let l_mpc = 2.0 * PI * MPC_HZ / c.k0;
        let l_pc = l_mpc * 1e6;
        let dx_pc = l_pc / c.n as f64;
        println!(
            "{:<26} {:<12} {:>4} {:>12.3e} {:>6.1} {:>6.1} {:>7.1} {:>5} {:>12.4} {:>12.5}",
            c.name, c.group, c.n, c.k0, c.dt, c.sub, c.sup, c.seed, l_pc, dx_pc
        );
    }
}

fn select_configs(all: &[Config], args: &Args) -> Vec<Config> {
    if !args.configs.is_empty() {
        return all
            .iter()
            .filter(|c| args.configs.iter().any(|name| name == c.name))
            .cloned()
            .collect();
    }

    let groups = if args.groups.is_empty() {
        vec!["resolution".to_string()]
    } else {
        args.groups.clone()
    };

    all.iter()
        .filter(|c| groups.iter().any(|g| g == "all" || g == c.group))
        .cloned()
        .collect()
}

fn load_baseline(path: &Path, xi0: f64) -> Result<Option<Baseline>> {
    if !path.exists() {
        return Ok(None);
    }

    let mut npz = NpzReader::new(File::open(path)?)?;
    let params: Array1<f64> = npz.by_name("params")?;
    let Some(index) = params.iter().position(|p| (p - xi0).abs() < 1.0) else {
        return Ok(None);
    };

    let linear_k: Array1<f64> = npz.by_name("linear_k")?;
    let linear_p: Array2<f64> = npz.by_name("linear_spectrums")?;
    let lat_k: Array1<f64> = npz.by_name("lat_k")?;
    let lat_p: Array2<f64> = npz.by_name("lat_spectrums")?;
    let lat_sigw: Array2<f64> = npz.by_name("lat_sigw_omega")?;

    if linear_p.ncols() != linear_k.len()
        || lat_p.ncols() != lat_k.len()
        || lat_sigw.ncols() != lat_k.len()
    {
        bail!("baseline npz has inconsistent array shapes");
    }

    // `linear_k` is built from a ParamRange starting at 1e-2 multiplied by k_star.
    let k_star = linear_k[0] / 1e-2;
    if !(k_star > 0.0) {
        bail!("invalid k_star derived from baseline npz");
    }

    Ok(Some(Baseline {
        k_star,
        linear_k: linear_k.to_vec(),
        linear_p: linear_p.row(index).to_vec(),
        lat_k: lat_k.to_vec(),
        lat_p: lat_p.row(index).to_vec(),
        lat_sigw: lat_sigw.row(index).to_vec(),
    }))
}

fn compute_linear_baseline(
    input: &GaussBonnetBInput<NaturalInflationPotential, TanhPotential>,
    background: &[GaussBonnetBackgroundState],
    out_dir: &Path,
) -> Result<Baseline> {
    let range = ParamRange::new(1e-2, 1e25, 1000);
    let p = run_spectrum(
        &format!("{}/fallback.spectrum", out_dir.display()),
        background,
        input,
        range,
        0.01,
        1e3,
        true,
    )?;
    let pivot = p
        .iter()
        .position(|&v| v <= 2.9e-9)
        .context("failed to locate the CMB pivot in the linear spectrum")?;
    let k_star = 0.05 * MPC_HZ / range.log_interp(pivot);
    let k = (range * k_star).as_logspace().collect::<Vec<_>>();

    Ok(Baseline {
        k_star,
        linear_k: k,
        linear_p: p,
        lat_k: Vec::new(),
        lat_p: Vec::new(),
        lat_sigw: Vec::new(),
    })
}

fn interp_loglog(x: f64, xs: &[f64], ys: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    if xs.len() == 1 {
        return ys[0];
    }
    if x <= xs[0] {
        return ys[0];
    }
    if x >= xs[xs.len() - 1] {
        return ys[ys.len() - 1];
    }

    let i = xs.partition_point(|&v| v < x).clamp(1, xs.len() - 1);
    let x0 = xs[i - 1];
    let x1 = xs[i];
    let y0 = ys[i - 1];
    let y1 = ys[i];
    let t = (x.ln() - x0.ln()) / (x1.ln() - x0.ln());
    if y0 > 0.0 && y1 > 0.0 {
        (y0.ln() + t * (y1.ln() - y0.ln())).exp()
    } else {
        y0 + t * (y1 - y0)
    }
}

fn trim<'a>(k: &'a [f64], p: &'a [f64]) -> (&'a [f64], &'a [f64]) {
    if k.len() < 4 {
        (k, p)
    } else {
        (&k[1..k.len() - 1], &p[1..p.len() - 1])
    }
}

fn comparison_grid(ka: &[f64], kb: &[f64]) -> Vec<f64> {
    if ka.len() < 2 || kb.len() < 2 {
        return Vec::new();
    }
    let lo = ka[0].max(kb[0]).max(COMPARISON_K_MIN);
    let hi = COMPARISON_K_MAX_FRACTION * ka[ka.len() - 1].min(kb[kb.len() - 1]);
    if !(hi > lo) {
        return Vec::new();
    }
    ParamRange::new(lo, hi, 300).as_logspace().collect()
}

fn rel_curve(ka: &[f64], pa: &[f64], kb: &[f64], pb: &[f64]) -> Option<(Vec<f64>, Vec<f64>)> {
    // Drop the first and last bins: the lowest shell is coarse and the highest
    // bin is affected by the lattice UV cutoff.  These are also removed from
    // the paper figures.
    let (ka, pa) = trim(ka, pa);
    let (kb, pb) = trim(kb, pb);
    let grid = comparison_grid(ka, kb);
    if grid.len() < 2 {
        return None;
    }

    let a = grid
        .iter()
        .map(|&x| interp_loglog(x, ka, pa))
        .collect::<Vec<_>>();
    let b = grid
        .iter()
        .map(|&x| interp_loglog(x, kb, pb))
        .collect::<Vec<_>>();
    let rel = a
        .iter()
        .zip(&b)
        .map(|(a, b)| a / b - 1.0)
        .collect::<Vec<_>>();
    Some((grid, rel))
}

fn stats(v: &[f64]) -> RelStats {
    if v.is_empty() {
        return RelStats::default();
    }
    let max = v.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
    let mut abs = v.iter().map(|x| x.abs()).collect::<Vec<_>>();
    abs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    RelStats {
        max,
        median: abs[abs.len() / 2],
        rms: (v.iter().map(|x| x * x).sum::<f64>() / (v.len() as f64)).sqrt(),
    }
}

fn rel_stats(ka: &[f64], pa: &[f64], kb: &[f64], pb: &[f64]) -> Option<RelStats> {
    rel_curve(ka, pa, kb, pb).map(|(_, rel)| stats(&rel))
}

fn corr_stats(l: &Loaded) -> RelStats {
    let (k, p) = trim(&l.k, &l.p);
    let (_, p_lin) = trim(&l.k, &l.p_lin);
    let k_max = COMPARISON_K_MAX_FRACTION * k[k.len() - 1];
    let mut corr = Vec::new();
    for i in 0..k.len() {
        if k[i] >= COMPARISON_K_MIN && k[i] <= k_max {
            corr.push(p[i] / p_lin[i] - 1.0);
        }
    }
    stats(&corr)
}

fn collect_lattice<const D: usize, T: Clone>(lat: &impl Lattice<D, T>) -> Vec<T> {
    let zero = VecN::new([0usize; D]);
    (0..lat.dim().product())
        .map(|index| lat.get(index, &zero))
        .collect()
}

fn save_final_field(
    path: &Path,
    data: &LatticeOutputData<3, GaussBonnetField<3>>,
) -> Result<()> {
    let n = data.final_state.phi.dim()[0];
    let raw = collect_lattice(&data.final_state.phi);
    let phi = raw.iter().map(|v| v[0]).collect::<Vec<_>>();
    let v_phi = raw.iter().map(|v| v[1]).collect::<Vec<_>>();
    let zeta = collect_lattice(&data.final_zeta);

    let mut npz = NpzWriter::new_compressed(BufWriter::new(File::create(path)?));
    npz.add_array(
        "phi",
        &Array3::from_shape_vec((n, n, n), phi).context("invalid phi shape")?,
    )?;
    npz.add_array(
        "v_phi",
        &Array3::from_shape_vec((n, n, n), v_phi).context("invalid v_phi shape")?,
    )?;
    npz.add_array(
        "zeta",
        &Array3::from_shape_vec((n, n, n), zeta).context("invalid zeta shape")?,
    )?;
    npz.add_array(
        "a",
        &Array1::from_vec(vec![data.final_state.a]),
    )?;
    npz.add_array(
        "v_a",
        &Array1::from_vec(vec![data.final_state.v_a]),
    )?;
    npz.add_array("N", &Array1::from_vec(vec![n as f64]))?;
    npz.finish()?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn save_summary(
    path: &Path,
    data: &LatticeOutputData<3, GaussBonnetField<3>>,
    cfg: &Config,
    k: &[f64],
    p: &[f64],
    p_lin: &[f64],
    sigw: &[f64],
    k_star: f64,
) -> Result<()> {
    let mut npz = NpzWriter::new(BufWriter::new(File::create(path)?));
    npz.add_array("k", &Array1::from_vec(k.to_vec()))?;
    npz.add_array("p", &Array1::from_vec(p.to_vec()))?;
    npz.add_array("p_lin", &Array1::from_vec(p_lin.to_vec()))?;
    npz.add_array("sigw", &Array1::from_vec(sigw.to_vec()))?;
    npz.add_array(
        "spectrum_k",
        &Array1::from_vec(
            data.spectrum_k
                .iter()
                .map(|k| k * k_star)
                .collect::<Vec<_>>(),
        ),
    )?;

    if let Some((_, first)) = data.spectrums.first() {
        let n_k = first.len();
        let n_spectrum = data.spectrums.len();
        let mut ln_a = Vec::with_capacity(n_spectrum);
        let mut spectrums = Vec::with_capacity(n_k * n_spectrum);
        for (ln_a_value, spectrum) in &data.spectrums {
            ln_a.push(*ln_a_value);
            spectrums.extend_from_slice(spectrum);
        }
        npz.add_array("spectrum_ln_a", &Array1::from_vec(ln_a))?;
        npz.add_array(
            "spectrums",
            &Array2::from_shape_vec((n_spectrum, n_k), spectrums)
                .context("invalid mid-run spectrum shape")?,
        )?;
    }

    // Save a, a-dot and the other background quantities at a bounded number of
    // points.  The full time series remains in the lattice bincode file.
    let stride = (data.measurables.len() / MAX_SUMMARY_MEASURABLES).max(1);
    let mut a = Vec::new();
    let mut v_a = Vec::new();
    let mut phi = Vec::new();
    let mut v_phi = Vec::new();
    let mut metric_pert_a = Vec::new();
    let mut metric_pert_b = Vec::new();
    let mut hubble_constraint = Vec::new();
    for m in data.measurables.iter().step_by(stride) {
        a.push(m.a);
        v_a.push(m.v_a);
        phi.push(m.phi);
        v_phi.push(m.v_phi);
        metric_pert_a.push(m.metric_perts.0);
        metric_pert_b.push(m.metric_perts.1);
        hubble_constraint.push(m.hubble_constraint);
    }
    npz.add_array("measurable_a", &Array1::from_vec(a))?;
    npz.add_array("measurable_v_a", &Array1::from_vec(v_a))?;
    npz.add_array("measurable_phi", &Array1::from_vec(phi))?;
    npz.add_array("measurable_v_phi", &Array1::from_vec(v_phi))?;
    npz.add_array("measurable_metric_pert_a", &Array1::from_vec(metric_pert_a))?;
    npz.add_array("measurable_metric_pert_b", &Array1::from_vec(metric_pert_b))?;
    npz.add_array(
        "measurable_hubble_constraint",
        &Array1::from_vec(hubble_constraint),
    )?;

    npz.add_array("final_a", &Array1::from_vec(vec![data.final_state.a]))?;
    npz.add_array("final_v_a", &Array1::from_vec(vec![data.final_state.v_a]))?;
    npz.add_array("N", &Array1::from_vec(vec![cfg.n as f64]))?;
    npz.add_array("k0", &Array1::from_vec(vec![cfg.k0]))?;
    npz.add_array("dt", &Array1::from_vec(vec![cfg.dt]))?;
    npz.add_array("subhorizon_tolerance", &Array1::from_vec(vec![cfg.sub]))?;
    npz.add_array("superhorizon_tolerance", &Array1::from_vec(vec![cfg.sup]))?;
    npz.add_array("seed", &Array1::from_vec(vec![cfg.seed as f64]))?;
    npz.finish()?;
    Ok(())
}

fn process_config(
    cfg: &Config,
    input: &GaussBonnetBInput<NaturalInflationPotential, TanhPotential>,
    background: &[GaussBonnetBackgroundState],
    k_star: f64,
    baseline: &Baseline,
    out_dir: &Path,
    create: bool,
) -> Result<Option<Loaded>> {
    let mut lat_in = LI::from_background_and_k_normalized(
        background,
        input,
        cfg.k0 / k_star,
        cfg.sub,
        cfg.sup,
        cfg.n,
        cfg.dt,
    );
    lat_in.seed = cfg.seed;

    let k_min = lat_in.k_min() * k_star;
    let k_max = lat_in.k_max() * k_star;
    let l_mpc = 2.0 * PI * MPC_HZ / k_min;
    let dx_mpc = l_mpc / (cfg.n as f64);
    let h_dt = (lat_in.state.v_a / lat_in.state.a) * cfg.dt;

    let out_base = format!("{}/{}.lattice", out_dir.display(), cfg.name);
    let data = match lat_in.run(
        &GaussBonnetFieldSimulatorCreator,
        input,
        &out_base,
        create,
        SPECTRUM_COUNT,
    ) {
        Ok(data) => data,
        Err(err) => {
            println!("[skip] {}: {}", cfg.name, err);
            return Ok(None);
        }
    };

    let k = data
        .spectrum_k
        .iter()
        .map(|k| k * k_star)
        .collect::<Vec<_>>();
    let p = data.zeta_spectrum.clone();
    let p_lin = k
        .iter()
        .map(|&k| interp_loglog(k, &baseline.linear_k, &baseline.linear_p))
        .collect::<Vec<_>>();
    let sigw = lazy_file(
        &format!("{}/{}.sigw.bincode", out_dir.display(), cfg.name),
        BINCODE_CONFIG,
        || sigw_2_spectrum(&k, &p, 100.0, 0.05, 0.05, |_, _| {}),
    )?;

    save_final_field(
        &out_dir.join(format!("{}.final_field.npz", cfg.name)),
        &data,
    )?;
    save_summary(
        &out_dir.join(format!("{}.summary.npz", cfg.name)),
        &data,
        cfg,
        &k,
        &p,
        &p_lin,
        &sigw,
        k_star,
    )?;

    let (k_trimmed, p_trimmed) = trim(&k, &p);
    let peak_index = p_trimmed
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(index, _)| index)
        .unwrap_or(0);
    let k_peak = k_trimmed[peak_index];
    let p_peak = p_trimmed[peak_index];
    let p_peak_over_lin =
        p_peak / interp_loglog(k_peak, &baseline.linear_k, &baseline.linear_p);

    println!(
        "[loaded] {}: N={}, k=[{:.3e}, {:.3e}] Hz, peak k={:.3e} Hz, P={:.3e}",
        cfg.name,
        cfg.n,
        k[0],
        k[k.len() - 1],
        k_peak,
        p_peak
    );

    Ok(Some(Loaded {
        cfg: cfg.clone(),
        k,
        p,
        p_lin,
        sigw,
        l_mpc,
        dx_mpc,
        k_min,
        k_max,
        h_dt,
        k_peak,
        p_peak,
        p_peak_over_lin,
    }))
}

fn color_for(name: &str) -> &'static str {
    match name {
        "n64" => "#1f77b4",
        "n128" => "#d62728",
        "n300" => "#ff7f0e",
        "n400" => "#9467bd",
        "n512" => "#2ca02c",
        "box_n64_k0_6p25e-9" => "#9467bd",
        "box_n128_k0_3p125e-9" => "#8c564b",
        "dt1_n64" => "#2ca02c",
        "dt4_n128" => "#98df8a",
        "dt8_n128" => "#bcbd22",
        "sub2_n128" => "#ff7f0e",
        "super500_n128" => "#e377c2",
        "n400_super500" => "#8c564b",
        "seed11_n128" => "#17becf",
        "seed12_n128" => "#aec7e8",
        _ => "#7f7f7f",
    }
}

fn new_two_panel_plot() -> Plot {
    let mut plot = Plot::new();
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
            .y_axis2(Axis::new().exponent_format(ExponentFormat::Power))
            .height(900),
    );
    plot
}

fn new_log_plot() -> Plot {
    let mut plot = Plot::new();
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
            .height(650),
    );
    plot
}

fn plot_range(baseline: &Baseline, include: &[&Loaded]) -> Option<(f64, f64)> {
    let mut lo = f64::INFINITY;
    let mut hi = 0.0_f64;

    if !baseline.lat_k.is_empty() {
        let (k, _) = trim(&baseline.lat_k, &baseline.lat_p);
        lo = lo.min(k[0]);
        hi = hi.max(k[k.len() - 1]);
    }
    for l in include {
        let (k, _) = trim(&l.k, &l.p);
        lo = lo.min(k[0]);
        hi = hi.max(k[k.len() - 1]);
    }

    if lo.is_finite() && hi > lo {
        Some((lo * 0.9, hi * 1.1))
    } else {
        None
    }
}

fn add_linear_trace(plot: &mut Plot, baseline: &Baseline, range: (f64, f64)) {
    if baseline.linear_k.is_empty() {
        return;
    }
    let grid = ParamRange::new(range.0, range.1, 500)
        .as_logspace()
        .collect::<Vec<_>>();
    let p = grid
        .iter()
        .map(|&k| interp_loglog(k, &baseline.linear_k, &baseline.linear_p))
        .collect::<Vec<_>>();
    plot.add_trace(
        Scatter::new(grid, p)
            .name("linear")
            .line(Line::new().dash(DashType::Dash).color("#888888")),
    );
}

fn add_lattice_traces(plot: &mut Plot, baseline: &Baseline, include: &[&Loaded]) {
    if !baseline.lat_k.is_empty() {
        let (k, p) = trim(&baseline.lat_k, &baseline.lat_p);
        plot.add_trace(
            Scatter::new(k.to_vec(), p.to_vec())
                .name("baseline N=200")
                .line(Line::new().color("#000000")),
        );
    }
    for l in include {
        let (k, p) = trim(&l.k, &l.p);
        plot.add_trace(
            Scatter::new(k.to_vec(), p.to_vec())
                .name(&format!("{} (N={})", l.cfg.name, l.cfg.n))
                .line(Line::new().color(color_for(l.cfg.name))),
        );
    }
}

fn write_spectrum_plot(
    path: &Path,
    baseline: &Baseline,
    include: &[&Loaded],
) -> Result<()> {
    let Some(range) = plot_range(baseline, include) else {
        return Ok(());
    };
    let mut plot = new_two_panel_plot();
    add_linear_trace(&mut plot, baseline, range);
    add_lattice_traces(&mut plot, baseline, include);

    if !baseline.lat_k.is_empty() {
        plot.add_trace(
            Scatter::new(vec![range.0, range.1], vec![0.0, 0.0])
                .name("baseline")
                .y_axis("y2")
                .line(Line::new().dash(DashType::Dot).color("#000000")),
        );
        for l in include {
            if let Some((k, rel)) = rel_curve(&l.k, &l.p, &baseline.lat_k, &baseline.lat_p) {
                plot.add_trace(
                    Scatter::new(k, rel)
                        .name(&format!("{} vs baseline", l.cfg.name))
                        .y_axis("y2")
                        .line(Line::new().color(color_for(l.cfg.name))),
                );
            }
        }
    }

    plot.write_html(path);
    Ok(())
}

fn write_correction_plot(
    path: &Path,
    baseline: &Baseline,
    include: &[&Loaded],
) -> Result<()> {
    let Some(range) = plot_range(baseline, include) else {
        return Ok(());
    };
    let mut plot = new_two_panel_plot();
    add_linear_trace(&mut plot, baseline, range);
    add_lattice_traces(&mut plot, baseline, include);

    plot.add_trace(
        Scatter::new(vec![range.0, range.1], vec![0.0, 0.0])
            .name("zero")
            .y_axis("y2")
            .line(Line::new().dash(DashType::Dot).color("#000000")),
    );

    if !baseline.lat_k.is_empty() {
        let (k, p) = trim(&baseline.lat_k, &baseline.lat_p);
        let corr = k
            .iter()
            .zip(p)
            .map(|(&k, &p)| {
                let p_lin = interp_loglog(k, &baseline.linear_k, &baseline.linear_p);
                p / p_lin - 1.0
            })
            .collect::<Vec<_>>();
        plot.add_trace(
            Scatter::new(k.to_vec(), corr)
                .name("baseline N=200")
                .y_axis("y2")
                .line(Line::new().color("#000000")),
        );
    }

    for l in include {
        let (k, p) = trim(&l.k, &l.p);
        let (_, p_lin) = trim(&l.k, &l.p_lin);
        let corr = p
            .iter()
            .zip(p_lin)
            .map(|(p, p_lin)| p / p_lin - 1.0)
            .collect::<Vec<_>>();
        plot.add_trace(
            Scatter::new(k.to_vec(), corr)
                .name(&format!("{} (N={})", l.cfg.name, l.cfg.n))
                .y_axis("y2")
                .line(Line::new().color(color_for(l.cfg.name))),
        );
    }

    plot.write_html(path);
    Ok(())
}

fn write_sigw_plot(path: &Path, baseline: &Baseline, include: &[&Loaded]) -> Result<()> {
    if plot_range(baseline, include).is_none() {
        return Ok(());
    }
    let mut plot = new_log_plot();

    if !baseline.lat_sigw.is_empty() {
        let (k, p) = trim(&baseline.lat_k, &baseline.lat_sigw);
        plot.add_trace(
            Scatter::new(k.to_vec(), p.to_vec())
                .name("baseline N=200")
                .line(Line::new().color("#000000")),
        );
    }
    for l in include {
        let (k, p) = trim(&l.k, &l.sigw);
        plot.add_trace(
            Scatter::new(k.to_vec(), p.to_vec())
                .name(&format!("{} (N={})", l.cfg.name, l.cfg.n))
                .line(Line::new().color(color_for(l.cfg.name))),
        );
    }

    plot.write_html(path);
    Ok(())
}

fn opt(value: Option<f64>) -> f64 {
    value.unwrap_or(f64::NAN)
}

fn write_csv(path: &Path, baseline: &Baseline, loaded: &[Loaded]) -> Result<()> {
    let mut file = File::create(path)?;
    writeln!(
        file,
        "config,group,N,k0_Hz,dt,sub,super,seed,L_Mpc,dx_Mpc,L_pc,dx_pc,k_min_Hz,k_max_Hz,H_start_dt,peak_k_Hz,peak_P,peak_P_over_lin,corr_max,corr_median,rel_baseline_max,rel_baseline_median,rel_baseline_rms,rel_sameN_max,rel_sameN_median,rel_sameN_rms,sigw_rel_baseline_max,sigw_rel_baseline_median"
    )?;

    for l in loaded {
        let corr = corr_stats(l);
        let rel_baseline = if baseline.lat_k.is_empty() {
            None
        } else {
            rel_stats(&l.k, &l.p, &baseline.lat_k, &baseline.lat_p)
        };
        let same_n = if matches!(l.cfg.group, "timestep" | "times" | "seed") {
            let reference_name = match l.cfg.n {
                64 => "n64",
                128 => "n128",
                _ => "",
            };
            if l.cfg.name == reference_name {
                None
            } else {
                loaded
                    .iter()
                    .find(|r| r.cfg.name == reference_name)
            }
        } else {
            None
        };
        let rel_same_n = same_n.and_then(|r| rel_stats(&l.k, &l.p, &r.k, &r.p));
        let sigw_rel = if baseline.lat_sigw.is_empty() {
            None
        } else {
            rel_stats(&l.k, &l.sigw, &baseline.lat_k, &baseline.lat_sigw)
        };

        writeln!(
            file,
            "{},{},{},{:.6e},{:.6e},{},{},{},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e}",
            l.cfg.name,
            l.cfg.group,
            l.cfg.n,
            l.cfg.k0,
            l.cfg.dt,
            l.cfg.sub,
            l.cfg.sup,
            l.cfg.seed,
            l.l_mpc,
            l.dx_mpc,
            l.l_mpc * 1e6,
            l.dx_mpc * 1e6,
            l.k_min,
            l.k_max,
            l.h_dt,
            l.k_peak,
            l.p_peak,
            l.p_peak_over_lin,
            corr.max,
            corr.median,
            opt(rel_baseline.map(|s| s.max)),
            opt(rel_baseline.map(|s| s.median)),
            opt(rel_baseline.map(|s| s.rms)),
            opt(rel_same_n.map(|s| s.max)),
            opt(rel_same_n.map(|s| s.median)),
            opt(rel_same_n.map(|s| s.rms)),
            opt(sigw_rel.map(|s| s.max)),
            opt(sigw_rel.map(|s| s.median)),
        )?;
    }

    Ok(())
}

fn write_comparison_npz(path: &Path, baseline: &Baseline, loaded: &[Loaded]) -> Result<()> {
    let mut npz = NpzWriter::new(BufWriter::new(File::create(path)?));

    if !baseline.lat_k.is_empty() {
        npz.add_array("baseline_k", &Array1::from_vec(baseline.lat_k.clone()))?;
        npz.add_array("baseline_p", &Array1::from_vec(baseline.lat_p.clone()))?;
        npz.add_array(
            "baseline_sigw",
            &Array1::from_vec(baseline.lat_sigw.clone()),
        )?;
    }

    for l in loaded {
        npz.add_array(
            format!("{}_k", l.cfg.name),
            &Array1::from_vec(l.k.clone()),
        )?;
        npz.add_array(
            format!("{}_p", l.cfg.name),
            &Array1::from_vec(l.p.clone()),
        )?;
        npz.add_array(
            format!("{}_p_lin", l.cfg.name),
            &Array1::from_vec(l.p_lin.clone()),
        )?;
        npz.add_array(
            format!("{}_sigw", l.cfg.name),
            &Array1::from_vec(l.sigw.clone()),
        )?;

        if !baseline.lat_k.is_empty() {
            if let Some((k, rel)) = rel_curve(&l.k, &l.p, &baseline.lat_k, &baseline.lat_p) {
                npz.add_array(
                    format!("{}_reldiff_k", l.cfg.name),
                    &Array1::from_vec(k),
                )?;
                npz.add_array(
                    format!("{}_reldiff", l.cfg.name),
                    &Array1::from_vec(rel),
                )?;
            }
        }
    }

    npz.finish()?;
    Ok(())
}

fn write_outputs(case_dir: &Path, baseline: &Baseline, loaded: &[Loaded]) -> Result<()> {
    write_csv(&case_dir.join("summary.csv"), baseline, loaded)?;
    write_comparison_npz(&case_dir.join("comparison.npz"), baseline, loaded)?;

    for group in ["resolution", "box", "timestep", "times", "seed"] {
        let mut include = loaded
            .iter()
            .filter(|l| l.cfg.group == group)
            .collect::<Vec<_>>();
        if include.is_empty() {
            continue;
        }

        // For time-step and tolerance tests the meaningful reference is the
        // same-N run (n64 or n128), not the production N=200 baseline, and not
        // a run with a different N.
        if matches!(group, "timestep" | "times" | "seed") {
            let mut references: Vec<&Loaded> = Vec::new();
            for l in &include {
                let reference_name = match l.cfg.n {
                    64 => "n64",
                    128 => "n128",
                    _ => continue,
                };
                if let Some(reference) = loaded.iter().find(|r| r.cfg.name == reference_name)
                {
                    if !include.iter().any(|x| x.cfg.name == reference.cfg.name)
                        && !references
                            .iter()
                            .any(|x| x.cfg.name == reference.cfg.name)
                    {
                        references.push(reference);
                    }
                }
            }
            include.extend(references);
        }

        write_spectrum_plot(
            &case_dir.join(format!("{group}.spectra.html")),
            baseline,
            &include,
        )?;
        write_correction_plot(
            &case_dir.join(format!("{group}.vs_linear.html")),
            baseline,
            &include,
        )?;
        write_sigw_plot(
            &case_dir.join(format!("{group}.sigw.html")),
            baseline,
            &include,
        )?;
    }

    let all = loaded.iter().collect::<Vec<_>>();
    write_spectrum_plot(&case_dir.join("all.spectra.html"), baseline, &all)?;
    write_correction_plot(&case_dir.join("all.vs_linear.html"), baseline, &all)?;
    write_sigw_plot(&case_dir.join("all.sigw.html"), baseline, &all)?;

    Ok(())
}

fn dump_background(
    path: &Path,
    background: &[GaussBonnetBackgroundState],
    k_star: f64,
) -> Result<()> {
    // Same start condition as LatticeInput::from_background_and_k_normalized
    // for the resolution runs (k0 = BASELINE_K0, sub = BASELINE_SUB).
    let starting_horizon = BASELINE_K0 / k_star / BASELINE_SUB;
    let start = background
        .iter()
        .position(|state| state.v_a >= starting_horizon)
        .context("no background state reaches the lattice starting horizon")?;
    let a_start = background[start].a;

    let stride = (background.len() / MAX_SUMMARY_MEASURABLES).max(1);
    let mut ln_a = Vec::new();
    let mut h = Vec::new();
    let mut phi = Vec::new();
    let mut v_phi = Vec::new();
    for state in background[start..].iter().step_by(stride) {
        ln_a.push((state.a / a_start).ln());
        h.push(state.v_a / state.a);
        phi.push(state.phi);
        v_phi.push(state.v_phi);
    }

    let mut npz = NpzWriter::new(BufWriter::new(File::create(path)?));
    npz.add_array("ln_a", &Array1::from_vec(ln_a))?;
    npz.add_array("H", &Array1::from_vec(h))?;
    npz.add_array("phi", &Array1::from_vec(phi))?;
    npz.add_array("v_phi", &Array1::from_vec(v_phi))?;
    npz.finish()?;
    Ok(())
}

fn main() -> Result<()> {
    let args = parse_args()?;

    if args.list || (args.dump_background.is_none() && !args.run && !args.plot_only) {
        print_plan();
        return Ok(());
    }

    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let out_root = root.join("out/gauss_bonnet_convergence");
    let case_dir = out_root.join(format!("alpha_{:.5e}", args.xi0));
    create_dir_all(&case_dir)?;

    let baseline_path = args
        .baseline
        .clone()
        .unwrap_or_else(|| root.join("out/gauss_bonnet.bak/case1-final.output.npz"));

    println!("[setup] output: {}", case_dir.display());
    println!("[setup] baseline: {}", baseline_path.display());

    let input = param_set1(9.6, args.xi0, 30.0);
    println!("[background] computing ...");
    let background = calculate_background(&input, 1.0, 7.0, 1.0);
    println!("[background] {} states", background.len());

    let baseline = match load_baseline(&baseline_path, args.xi0)? {
        Some(baseline) => {
            println!(
                "[baseline] loaded xi0={:.5e}, k_star={:.4e}, lattice bins={}",
                args.xi0,
                baseline.k_star,
                baseline.lat_k.len()
            );
            baseline
        }
        None => {
            println!(
                "[baseline] no matching baseline for xi0={:.5e}; computing only the linear spectrum",
                args.xi0
            );
            compute_linear_baseline(&input, &background, &case_dir)?
        }
    };

    let all = configs();
    let selected = select_configs(&all, &args);

    if let Some(path) = &args.dump_background {
        dump_background(path, &background, baseline.k_star)?;
        println!("[background] wrote {}", path.display());
        return Ok(());
    }

    if selected.is_empty() {
        bail!("no configuration selected");
    }

    println!(
        "[plan] running/loading: {}",
        selected
            .iter()
            .map(|c| c.name)
            .collect::<Vec<_>>()
            .join(", ")
    );

    let create = args.run;
    let mut loaded = Vec::new();
    for cfg in &selected {
        if let Some(result) = process_config(
            cfg,
            &input,
            &background,
            baseline.k_star,
            &baseline,
            &case_dir,
            create,
        )? {
            loaded.push(result);
        }
    }

    // If the N=128 point is not part of the selection, try to load it from disk
    // as the same-N reference for dt/tolerance/seed comparisons.
    if !loaded.iter().any(|l| l.cfg.name == "n128") {
        if let Some(cfg) = all.iter().find(|c| c.name == "n128") {
            if let Some(result) = process_config(
                cfg,
                &input,
                &background,
                baseline.k_star,
                &baseline,
                &case_dir,
                false,
            )? {
                loaded.push(result);
            }
        }
    }

    if loaded.is_empty() {
        println!("[done] no configurations were run or found");
        return Ok(());
    }

    write_outputs(&case_dir, &baseline, &loaded)?;

    println!("[done] wrote outputs to {}", case_dir.display());
    Ok(())
}
