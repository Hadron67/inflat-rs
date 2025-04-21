use std::{env::args, fs::File, io::{BufReader, BufWriter}, iter::zip, ops::{Add, DerefMut, Div, Mul, Sub}, panic::catch_unwind, process::Output, sync::Mutex};

use bincode::{config::{standard, Configuration}, decode_from_std_read, encode_into_std_write};
use inflat::background::{self, BackgroundInput, BackgroundOutput, BackgroundSimulator, ScalarPerturbationOutput, ScalarPerturbationSimulator};
use libm::{cos, exp, log, sin, sqrt};
use num_complex::{Complex64, ComplexFloat};
use num_traits::FromPrimitive;
use plotly::{layout::{Axis, AxisType, GridPattern, LayoutGrid}, Layout, Plot, Scatter};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::Serialize;

fn resournance_input() {

}

fn limit_length<T: Clone>(arr: Vec<T>, max_length: usize) -> Vec<T> {
    if arr.len() > max_length {
        let mut arr2 = vec![];
        arr2.reserve(max_length);
        for i in 0..max_length {
            arr2.push(arr[((i as f64) / ((max_length - 1) as f64) * ((arr.len() - 1) as f64)) as usize].clone());
        }
        arr2
    } else {
        arr
    }
}

fn test_mass_eff(output: &BackgroundOutput) {
    let mut mass_eff2 = vec![];
    mass_eff2.reserve(output.dt.len());
    mass_eff2.push(output.effective_mass[0]);
    let z_fn = |output: &BackgroundOutput, i: usize|output.scale_factor[i] * output.scale_factor[i] * output.phi_d[i] / output.d_scale_factor[i];
    for i in 1..output.dt.len() - 1 {
        let z1 = z_fn(&output, i - 1);
        let z2 = z_fn(&output, i + 1);
        let z = z_fn(&output, i);
        let s = output.dt[i - 1];
        let t = output.dt[i];
        let z_dd = (z1 / s + z2 / t - (1.0 / s + 1.0 / t) * z) / (s + t) * 2.0;
        mass_eff2.push(z_dd / z);
    }
    mass_eff2.push(*output.effective_mass.last().unwrap());

    {
        let mut plot = Plot::new();
        const MAX_LENGTH: Option<usize> = Some(10000usize);
        let efoldings = output.scale_factor.iter().cloned().map(&log).collect::<Vec<_>>();
        plot.add_trace(new_scatter(efoldings.clone().to_vec(), mass_eff2, MAX_LENGTH).name("z'' / z"));
        plot.add_trace(new_scatter(efoldings.clone().to_vec(), output.effective_mass.iter().map(|f|-f).collect(), MAX_LENGTH).name("mass").x_axis("x1").y_axis("y1"));
        plot.add_trace(
            new_scatter(
                efoldings.clone(),
                zip(&output.phi, &output.phi_d)
                    .map(|(phi, phi_d)|{
                        let h = *phi_d / *phi;
                        2.0 * h * h
                    })
                    .collect::<Vec<_>>(),
                MAX_LENGTH,
            ).x_axis("x1")
            .y_axis("y1")
            .name("cH"),
        );
        plot.set_layout(
            Layout::new()
                .grid(
                    LayoutGrid::new()
                        .rows(2)
                        .columns(1)
                        .pattern(GridPattern::Coupled)
                )
                .y_axis(Axis::new().type_(AxisType::Log))
                .height(1000)
        );
        plot.write_html("out/test.html");
    }
}

fn derivative_2<T, T2>(dx: &[T], y: &[T2], i: usize) -> T2 where
    T: Copy,
    T2: Copy + Add<T2, Output = T2> + Div<T2, Output = T2> + Mul<T2, Output = T2> + Sub<T2, Output = T2> + FromPrimitive + From<T>,
{
    let s: T2 = dx[i - 1].into();
    let t: T2 = dx[i].into();
    let one = T2::from_usize(1).unwrap();
    (y[i - 1] / s + y[i + 1] / t - (one / s + one / t) * y[i]) / (s + t) * T2::from_usize(2).unwrap()
}

fn make_test_background() -> BackgroundOutput {
    let mut output = BackgroundOutput {
        ..Default::default()
    };
    for i in 0..10000 {
        output.dt.push(0.001);
        output.effective_mass.push(0.0);
        output.phi.push(0.0);
        output.phi_d.push(0.0);
        output.scale_factor.push(exp((i as f64) / 100.0));
        output.d_scale_factor.push(0.0);
    }
    output
}

fn make_desitter_background(h: f64, start_n: f64, end_n: f64, count: usize) -> BackgroundOutput {
    let mut output = BackgroundOutput {
        ..Default::default()
    };
    let dn = (end_n - start_n) / ((count - 1) as f64);
    for i in 0..count {
        let n = start_n + (i as f64) * dn;
        output.scale_factor.push(exp(n));
        output.effective_mass.push(-2.0 * h * h * exp(2.0 * n));
        output.d_scale_factor.push(h * exp(2.0 * n));
        output.phi.push(0.0);
        output.phi_d.push(0.0);
        output.dt.push(1.0 / h * exp(-n) * (1.0 - exp(-dn)));
    }
    println!("T = {}", output.dt.iter().sum::<f64>());
    output
}

fn check_ms_eqn(output: &ScalarPerturbationOutput) {
    let k = output.k;
    let mut eqn = vec![];
    eqn.reserve(output.dt.len());
    eqn.push(0.0);
    for i in 1..output.dt.len() - 1 {
        let u_dd = derivative_2(&output.dt, &output.u, i);
        let diff = u_dd + output.potential[i] * output.u[i];
        // println!("i = {} =============================================", i);
        // println!("u[i + 1] = {}", output.u[i + 1]);
        // println!("u[i - 1] = {}", output.u[i - 1]);
        // println!("u[i] = {}", output.u[i]);
        // println!("dt[i - 1] = {}", output.dt[i - 1]);
        // println!("dt[i] = {}", output.dt[i]);
        // println!("u_dd = {}", u_dd);
        // println!("diff = {}", diff);
        eqn.push(diff.re.abs() / output.u[i].abs());
    }
    eqn.push(0.0);
    let r = zip(&output.u, &output.scale_factor).map(|(u, a)|u.abs() / *a).collect();

    let mut plot = Plot::new();
    let max_length = Some(100000usize);
    let efoldings = output.scale_factor.iter().cloned().map(&log).collect::<Vec<_>>();
    plot.add_trace(new_scatter(efoldings.clone(), eqn, max_length).name("eqn"));
    plot.add_trace(new_scatter(efoldings.clone(), r, max_length).x_axis("x1").y_axis("y2").name("|R|"));
    plot.add_trace(new_scatter(efoldings.clone(), output.u.iter().map(|f|f.re).collect(), max_length).x_axis("x1").y_axis("y3").name("re u"));
    plot.add_trace(new_scatter(efoldings.clone(), output.u.iter().map(|f|f.im).collect(), max_length).x_axis("x1").y_axis("y3").name("im u"));
    plot.add_trace(new_scatter(efoldings.clone(), output.potential.clone(), max_length).x_axis("x1").y_axis("y4").name("k^2 + m^2"));
    plot.add_trace(new_scatter(efoldings.clone(), output.time.clone(), max_length).x_axis("x1").y_axis("y5").name("dt"));
    plot.set_layout(
        Layout::new()
            .grid(LayoutGrid::new().rows(5).columns(1).pattern(GridPattern::Coupled))
            .y_axis(Axis::new().type_(AxisType::Log))
            .y_axis2(Axis::new().type_(AxisType::Log))
            .height(1000)
    );
    plot.write_html("out/test.html");
    println!("writtten test plot");
}

const BINCODE_CONFIG: Configuration = standard();

fn new_scatter<X, Y>(x: Vec<X>, y: Vec<Y>, length_limit: Option<usize>) -> Box<Scatter<X, Y>> where
    X: Serialize + Clone + 'static,
    Y: Serialize + Clone + 'static,
{
    if let Some(len) = length_limit {
        Scatter::new(limit_length(x, len), limit_length(y, len))
    } else {
        Scatter::new(x, y)
    }
}

fn main() {
    let lambda = 0.0032;
    let lambda4 = { let a = lambda * lambda; a * a};
    let sqrt_2_3 = sqrt(2.0 / 3.0);
    let phi_s = 4.9878;
    let phi_e = 4.9731;
    let phi_star = 8e-6;
    let xi = 1.7e-15;
    let params = BackgroundInput {
        kappa: 1.0,
        scale_factor: 0.1,
        phi: 5.5,
        phi_d: 0.0,
        potential: |phi: f64|lambda4 * {let a = 1.0 - exp(-sqrt_2_3 * phi); a * a} + (if phi >= phi_e && phi <= phi_s { xi * cos(phi / phi_star) } else { 0.0 }),
        potential_d: |phi: f64|lambda4 * 2.0 * (1.0 - exp(-sqrt_2_3 * phi)) * sqrt_2_3 * exp(-sqrt_2_3 * phi) + (if phi >= phi_e && phi <= phi_s { -xi / phi_star * sin(phi / phi_star) } else { 0.0 }),
        potential_dd: |phi: f64|lambda4 * (-4.0 / 3.0 * (2.0 * exp(-2.0 * sqrt_2_3 * phi) - exp(-sqrt_2_3 * phi))) + (if phi >= phi_e && phi <= phi_s { -xi / phi_star / phi_star * cos(phi / phi_star) } else { 0.0 }),
    };
    let k_star = 163.1373e-9 / 2.0 / phi_star;
    let k_start = k_star * exp(20.7);
    let k_end = k_star * exp(21.23);

    let mut background = true;
    if args().any(|f|f == "--perturbation") {
        background = false;
    }
    let max_length = Some(100000usize);
    if background {
        let mut sim = BackgroundSimulator::new(params);
        let (measurables, output) = sim.run_with_dn(0.00005, 0.1, |s|s.phi < 0.0);
        let efoldings = measurables.scale_factor.iter().cloned().map(&log).collect::<Vec<f64>>();

        {
            let mut file = File::create("out/parameter-resonance.background.bincode").unwrap();
            encode_into_std_write(&output, &mut BufWriter::new(&mut file), BINCODE_CONFIG).unwrap();
        }

        let mut plot = Plot::new();
        plot.add_trace(Scatter::new(efoldings.clone(), measurables.phi).name("phi"));
        plot.add_trace(Scatter::new(efoldings.clone(), measurables.d_phi).name("d_phi").x_axis("x").y_axis("y2"));
        plot.add_trace(Scatter::new(efoldings.clone(), measurables.hubble).name("H").x_axis("x").y_axis("y3"));
        plot.add_trace(Scatter::new(efoldings.clone(), measurables.hubble_constraint.iter().map(|f|f.abs()).collect()).name("hubble").x_axis("x").y_axis("y4"));
        plot.add_trace(Scatter::new(efoldings.clone(), measurables.epsilon).name("epsilon").x_axis("x").y_axis("y5"));
        plot.set_layout(
            Layout::new()
                .grid(
                    LayoutGrid::new()
                        .rows(5)
                        .columns(1)
                        .pattern(GridPattern::Coupled)
                )
                .y_axis4(Axis::new().type_(AxisType::Log))
                .y_axis5(Axis::new().type_(AxisType::Log))
                .height(1000)
        );
        plot.write_html("out/parameter-resonance-background.html");
    } else {
        let output: BackgroundOutput = {
            let mut file = File::open("out/parameter-resonance.background.bincode").unwrap();
            decode_from_std_read(&mut BufReader::new(&mut file), BINCODE_CONFIG).unwrap()
        };

        let mut perturbation_sim = ScalarPerturbationSimulator::new(&output, sqrt(k_start * k_end));
        let perturbation_output = perturbation_sim.run::<false>(20.0, Some(50.0), 8);
        let efoldings = perturbation_output.scale_factor.iter().cloned().map(&log).collect::<Vec<_>>();
        let mut plot = Plot::new();

        let n_start = output.scale_factor.iter().cloned().zip(0usize..).find(|(v, i)|{
            log(*v) >= 20.6
        }).unwrap().1;
        let n_end = output.scale_factor.iter().cloned().zip(0usize..).find(|(v, i)|{
            log(*v) >= 21.3
        }).unwrap().1;

        plot.add_trace(new_scatter(efoldings.clone(), perturbation_output.u.iter().cloned().map(|f|f.abs()).collect(), max_length).name("|u|").x_axis("x1").y_axis("y1"));
        plot.add_trace(new_scatter(efoldings.clone(), perturbation_output.perturbation.iter().cloned().map(|f|f.abs()).collect(), max_length).name("|R|").x_axis("x1").y_axis("y2"));
        plot.add_trace(new_scatter(output.scale_factor[n_start..n_end].iter().cloned().map(&log).collect(), output.effective_mass[n_start..n_end].to_vec(), max_length).name("mass").x_axis("x1").y_axis("y3"));
        // plot.add_trace(Scatter::new(spectrum.iter().map(|a|a.0).collect(), spectrum.iter().map(|a|a.1).collect()).name("spectrum").x_axis("x2").y_axis("y3"));
        plot.set_layout(
            Layout::new()
                .grid(
                    LayoutGrid::new()
                        .rows(3)
                        .columns(1)
                        .pattern(GridPattern::Coupled)
                )
                .y_axis(Axis::new().type_(AxisType::Log))
                .y_axis2(Axis::new().type_(AxisType::Log))
                .height(1200)
        );
        plot.write_html("out/parameter-resonance-perturbation.html");
    }
}
