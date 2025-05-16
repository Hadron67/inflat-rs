use std::{env::args, f64::consts::PI, fmt, fs::{write, File}, io::{self, BufReader, BufWriter, Write}, iter::zip, mem::offset_of, ops::{Add, DerefMut, Div, Mul, Range, Sub}, panic::catch_unwind, process::Output, sync::{atomic::{AtomicUsize, Ordering}, Mutex}, time::SystemTime};

use bincode::{config::{standard, Configuration}, decode_from_std_read, encode_into_std_write, Encode};
use inflat::{background::{self, BackgroundInput, BackgroundOutput, BackgroundSimulator, BackgroundState, C2Fn, Context, InputData, OutputSelector, PerturbationParameters, PerturbationSimulator, ScalarPerturbation, ScalarPerturbation2, ScalarPerturbationOutput, ScalarPerturbationSimulator}, util::{derivative_2, linear_interp}, wl::WlEncode};
use libm::{cos, exp, log, sin, sqrt};
use ndarray::Order;
use ndarray_npy::WritableElement;
use num_complex::{Complex, Complex64, ComplexFloat};
use num_traits::{Float, FromPrimitive};
use plotly::{layout::{Axis, AxisType, GridPattern, LayoutGrid}, Layout, Plot, Scatter};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use serde::Serialize;

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

// fn test_mass_eff(output: &BackgroundOutput) {
//     let mut mass_eff2 = vec![];
//     mass_eff2.reserve(output.dt.len());
//     mass_eff2.push(output.effective_mass[0]);
//     for i in 1..output.dt.len() - 1 {
//         let z_dd = derivative_2(&output.dt, &output.z, i);
//         mass_eff2.push(z_dd / output.z[i]);
//     }
//     mass_eff2.push(*output.effective_mass.last().unwrap());

//     {
//         let mut plot = Plot::new();
//         const MAX_LENGTH: Option<usize> = Some(10000usize);
//         let efoldings = output.scale_factor.iter().cloned().map(&log).collect::<Vec<_>>();
//         plot.add_trace(new_scatter(efoldings.clone(), zip(&mass_eff2, &output.effective_mass).map(|(a, b)|-*a / *b).collect(), MAX_LENGTH).name("z'' / z / m^2"));
//         plot.add_trace(new_scatter(efoldings.clone(), mass_eff2, MAX_LENGTH).name("z'' / z").x_axis("x1").y_axis("y2"));
//         plot.add_trace(new_scatter(efoldings.clone(), output.effective_mass.iter().map(|f|-f).collect(), MAX_LENGTH).name("mass").x_axis("x1").y_axis("y2"));
//         plot.set_layout(
//             Layout::new()
//                 .grid(
//                     LayoutGrid::new()
//                         .rows(2)
//                         .columns(1)
//                         .pattern(GridPattern::Coupled)
//                 )
//                 .y_axis(Axis::new().type_(AxisType::Log))
//                 .y_axis2(Axis::new().type_(AxisType::Log))
//                 .height(1000)
//         );
//         plot.write_html("out/test.html");
//     }
// }

// fn make_test_background() -> BackgroundOutput {
//     let mut output = BackgroundOutput {
//         ..Default::default()
//     };
//     for i in 0..10000 {
//         output.dt.push(0.001);
//         output.effective_mass.push(0.0);
//         output.scale_factor.push(exp((i as f64) / 100.0));
//     }
//     output
// }

// fn make_desitter_background(h: f64, start_n: f64, end_n: f64, count: usize) -> BackgroundOutput {
//     let mut output = BackgroundOutput {
//         ..Default::default()
//     };
//     let dn = (end_n - start_n) / ((count - 1) as f64);
//     for i in 0..count {
//         let n = start_n + (i as f64) * dn;
//         output.scale_factor.push(exp(n));
//         output.z.push(-exp(n));
//         output.effective_mass.push(-2.0 * h * h * exp(2.0 * n));
//         output.dt.push(1.0 / h * exp(-n) * (1.0 - exp(-dn)));
//     }
//     println!("T = {}", output.dt.iter().sum::<f64>());
//     output
// }

// fn check_ms_eqn(output: &ScalarPerturbationOutput) {
//     let mut eqn = vec![];
//     eqn.reserve(output.dt.len());
//     eqn.push(0.0);
//     for i in 1..output.dt.len() - 1 {
//         let u_dd = derivative_2(&output.dt, &output.u, i);
//         let diff = u_dd + output.potential[i] * output.u[i];
//         eqn.push(diff.abs() / u_dd.abs());
//     }
//     eqn.push(0.0);

//     let mut plot = Plot::new();
//     let max_length = Some(100000usize);
//     let efoldings = output.scale_factor.iter().cloned().map(&log).collect::<Vec<_>>();
//     plot.add_trace(new_scatter(efoldings.clone(), eqn, max_length).name("eqn"));
//     plot.add_trace(new_scatter(efoldings.clone(), output.u.iter().map(|f|f.re).collect(), max_length).x_axis("x1").y_axis("y2").name("re u"));
//     plot.add_trace(new_scatter(efoldings.clone(), output.u.iter().map(|f|f.im).collect(), max_length).x_axis("x1").y_axis("y2").name("im u"));
//     plot.add_trace(new_scatter(efoldings.clone(), output.potential.clone(), max_length).x_axis("x1").y_axis("y3").name("k^2 + m^2"));
//     plot.add_trace(new_scatter(efoldings.clone(), output.time.clone(), max_length).x_axis("x1").y_axis("y4").name("eta"));
//     plot.set_layout(
//         Layout::new()
//             .grid(LayoutGrid::new().rows(5).columns(1).pattern(GridPattern::Coupled))
//             .y_axis(Axis::new().type_(AxisType::Log))
//             .height(1000)
//     );
//     plot.write_html("out/test.html");
//     println!("writtten test plot");
// }

const BINCODE_CONFIG: Configuration = standard();

// fn new_scatter<X, Y>(x: Vec<X>, y: Vec<Y>, length_limit: Option<usize>) -> Box<Scatter<X, Y>> where
//     X: Serialize + Clone + 'static,
//     Y: Serialize + Clone + 'static,
// {
//     if let Some(len) = length_limit {
//         Scatter::new(limit_length(x, len), limit_length(y, len))
//     } else {
//         Scatter::new(x, y)
//     }
// }

// #[derive(Clone, Copy)]
// struct ParamSet {
//     pub lambda: f64,
//     pub phi0: f64,
//     pub phi_s: f64,
//     pub phi_e: f64,
//     pub phi_star: f64,
//     pub xi: f64,
//     pub tensor_params: TensorSoundSpeedResonancePerturbation,
// }

// struct Runner<'a> {
//     is_test: bool,
//     base_name: &'a str,
//     params: ParamSet,
//     plot_max_length: Option<usize>,
//     background_output: Option<BackgroundOutput>,
// }

// impl ParamSet {
//     pub fn background_params(&self) -> BackgroundInput<impl Fn(f64) -> f64, impl Fn(f64) -> f64, impl Fn(f64) -> f64> {
//         let lambda = self.lambda;
//         let lambda4 = { let a = lambda * lambda; a * a};
//         let sqrt_2_3 = sqrt(2.0 / 3.0);
//         let phi_s = self.phi_s;
//         let phi_e = self.phi_e;
//         let phi_star = self.phi_star;
//         let xi = self.xi;
//         BackgroundInput {
//             kappa: 1.0,
//             scale_factor: 0.1,
//             phi: self.phi0,
//             phi_d: 0.0,
//             potential: move |phi: f64|lambda4 * {let a = 1.0 - exp(-sqrt_2_3 * phi); a * a} + (if phi >= phi_e && phi <= phi_s { xi * sin(phi / phi_star) } else { 0.0 }),
//             potential_d: move |phi: f64|lambda4 * 2.0 * (1.0 - exp(-sqrt_2_3 * phi)) * sqrt_2_3 * exp(-sqrt_2_3 * phi) + (if phi >= phi_e && phi <= phi_s { xi / phi_star * cos(phi / phi_star) } else { 0.0 }),
//             potential_dd: move |phi: f64|lambda4 * 4.0 / 3.0 * (2.0 * exp(-2.0 * sqrt_2_3 * phi) - exp(-sqrt_2_3 * phi)) + (if phi >= phi_e && phi <= phi_s { -xi / phi_star / phi_star * sin(phi / phi_star) } else { 0.0 }),
//         }
//     }
// }

// impl<'a> Runner<'a> {
//     pub fn new(name_and_params: (&'a str, ParamSet), max_length: Option<usize>, is_test: bool) -> Self {
//         Self {
//             is_test,
//             base_name: name_and_params.0,
//             params: name_and_params.1,
//             plot_max_length: max_length,
//             background_output: None,
//         }
//     }
//     pub fn background_data_file_name(&self) -> String {
//         "out/".to_owned() + self.base_name + ".background.bincode"
//     }
//     pub fn run_background(&mut self) {
//         let mut sim = BackgroundSimulator::new(self.params.background_params());
//         let (measurables, output) = sim.run_with_dn(0.00005, 0.1, |s|s.phi < 0.1);
//         let efoldings = measurables.scale_factor.iter().cloned().map(&log).collect::<Vec<f64>>();

//         println!("len = {}", measurables.phi.len());
//         {
//             let mut file = File::create(self.background_data_file_name()).unwrap();
//             encode_into_std_write(&output, &mut BufWriter::new(&mut file), BINCODE_CONFIG).unwrap();
//         }
//         println!("written output file");

//         let mut computed_dda_a = vec![];
//         computed_dda_a.push(0.0);
//         for i in 1..output.dt.len() - 1 {
//             let dda = derivative_2(&output.dt, &output.scale_factor, i);
//             computed_dda_a.push(dda / output.scale_factor[i]);
//         }
//         computed_dda_a.push(0.0);

//         let max_length = self.plot_max_length;
//         let mut plot = Plot::new();
//         plot.add_trace(new_scatter(efoldings.clone(), measurables.phi, max_length).name("phi"));
//         plot.add_trace(new_scatter(output.scale_factor.iter().cloned().map(&log).collect(), output.effective_mass.iter().map(|m2|sqrt(-*m2)).collect(), max_length).x_axis("x").y_axis("y2").name("k"));
//         plot.add_trace(new_scatter(output.scale_factor.iter().cloned().map(&log).collect(), output.tensor_effective_mass.iter().map(|f|sqrt(-f)).collect(), max_length).x_axis("x").y_axis("y2").name("a'' / a"));
//         plot.add_trace(new_scatter(efoldings.clone(), measurables.dphi_dt, max_length).name("d_phi").x_axis("x").y_axis("y3"));
//         plot.add_trace(new_scatter(efoldings.clone(), measurables.hubble, max_length).name("H").x_axis("x").y_axis("y4"));
//         plot.add_trace(new_scatter(efoldings.clone(), measurables.hubble_constraint.iter().map(|f|f.abs()).collect(), max_length).name("hubble").x_axis("x").y_axis("y5"));
//         plot.add_trace(new_scatter(efoldings.clone(), measurables.epsilon, max_length).name("epsilon").x_axis("x").y_axis("y6"));
//         plot.set_layout(
//             Layout::new()
//                 .grid(
//                     LayoutGrid::new()
//                         .rows(6)
//                         .columns(1)
//                         .pattern(GridPattern::Coupled)
//                 )
//                 .y_axis(Axis::new().type_(AxisType::Log))
//                 .y_axis2(Axis::new().type_(AxisType::Log))
//                 .y_axis5(Axis::new().type_(AxisType::Log))
//                 .y_axis6(Axis::new().type_(AxisType::Log))
//                 .height(1200)
//         );
//         plot.write_html("out/".to_owned() + self.base_name + ".background.html");

//         self.background_output = Some(output);
//         self.export_background_to_wl(("out/".to_owned() + self.base_name + ".background.wl").as_str(), OutputSelector { dn: Some(0.001), ..Default::default() }).unwrap();
//     }

//     fn load_background_data(&mut self) {
//         if self.background_output.is_none() {
//             self.background_output = Some({
//                 let file = File::open(self.background_data_file_name()).unwrap();
//                 let mut output: BackgroundOutput = decode_from_std_read(&mut BufReader::new(file), BINCODE_CONFIG).unwrap();
//                 if self.is_test {
//                     const OLD_AMP: f64 = -0.00223479;
//                     make_test_potential_background(&mut output, (2.2132, 2.7), -0.00223479, 3658.52, -1146.98, 2.07, (0.0, 10.0));
//                 }
//                 output
//             });
//             let mut file_name = "out/".to_owned() + self.base_name + ".background2";
//             if self.is_test {
//                 file_name.push_str(".test");
//             }
//             file_name.push_str(".wl");
//             self.export_background_to_wl(&file_name, OutputSelector { dn: None, start_n: Some(2.2), end_n: Some(2.4) }).unwrap();
//         }
//     }

//     pub fn resonance_index_range(&self) -> (usize, usize) {
//         let output = self.background_output.as_ref().unwrap();
//         let i_start = output.phi.iter().cloned().zip(0usize..).find(|(phi, _)|*phi <= self.params.phi_s).unwrap().1;
//         let i_end = output.phi.iter().cloned().zip(0usize..).find(|(phi, _)|*phi <= self.params.phi_e).unwrap().1;
//         (i_start, i_end)
//     }

//     pub fn resonance_mode_range(&self) -> (f64, f64) {
//         let (i_start, i_end) = self.resonance_index_range();
//         let output = self.background_output.as_ref().unwrap();
//         let k_star = output.d_phi[i_start].abs() / output.scale_factor[i_start] / 2.0 / self.params.phi_star;
//         (k_star * output.scale_factor[i_start], k_star * output.scale_factor[i_end])
//     }

//     pub fn resonance_n_range(&self) -> (f64, f64) {
//         let (i_start, i_end) = self.resonance_index_range();
//         let output = self.background_output.as_ref().unwrap();
//         (output.scale_factor[i_start].ln(), output.scale_factor[i_end].ln())
//     }

//     pub fn export_background_to_wl(&self, file_name: &str, selector: OutputSelector) -> io::Result<()> {
//         let mut file = BufWriter::new(File::create(file_name)?);
//         let output = self.background_output.as_ref().unwrap();
//         let mut scale_factor = vec![];
//         let mut phi = vec![];
//         let mut d_phi = vec![];
//         let mut effective_mass = vec![];
//         let mut eta = 0.0;
//         let mut last_efolding = output.scale_factor[0].ln();
//         let mut first = true;
//         file.write_all(b"<|\n    \"Time\" -> Developer`ToPackedArray@{")?;
//         let mut last_dt = 0.0;
//         for i in 0..output.dt.len() {
//             let efolding = output.scale_factor[i].ln();
//             if selector.test(last_efolding, efolding) {
//                 last_efolding = efolding;
//                 scale_factor.push(output.scale_factor[i]);
//                 phi.push(output.phi[i]);
//                 d_phi.push(output.d_phi[i]);
//                 effective_mass.push(output.effective_mass[i]);
//                 if !first {
//                     file.write_all(b",")?;
//                 } else {
//                     first = false;
//                 }
//                 eta.encode(&mut file)?;
//             }
//             eta += output.dt[i];
//         }
//         file.write_all(b"},\n    \"ScaleFactor\" -> ")?;
//         scale_factor.encode(&mut file)?;
//         file.write_all(b",\n    \"Phi\" -> ")?;
//         phi.encode(&mut file)?;
//         file.write_all(b",\n    \"DPhi\" -> ")?;
//         d_phi.encode(&mut file)?;
//         file.write_all(b",\n    \"EffectiveMass\" -> ")?;
//         effective_mass.encode(&mut file)?;
//         file.write_all(b"\n|>\n")?;
//         Ok(())
//     }

//     pub fn export_perturbation_to_wl(&self, output: &ScalarPerturbationOutput, file_name: &str) -> io::Result<()> {
//         let mut file = BufWriter::new(File::create(file_name)?);
//         file.write_all(b"<|\n    \"Time\" -> Developer`ToPackedArray@{")?;
//         let mut eta = 0.0;
//         for i in 0..output.dt.len() {
//             if i > 0 {
//                 file.write_all(b",")?;
//             }
//             eta.encode(&mut file)?;
//             eta += output.dt[i];
//         }
//         file.write_all(b"},\n    \"ScaleFactor\" -> ")?;
//         output.scale_factor.encode(&mut file)?;
//         file.write_all(b",\n    \"u\" -> ")?;
//         output.u.encode(&mut file)?;
//         file.write_all(b",\n    \"du\" -> ")?;
//         output.mom_u.encode(&mut file)?;
//         file.write_all(b",\n    \"Potential\" -> ")?;
//         output.potential.encode(&mut file)?;
//         file.write_all(b",\n    \"EffectiveMass\" -> ")?;
//         output.effective_mass.encode(&mut file)?;
//         file.write_all(b",\n    \"CalculatedPotential\" -> ")?;
//         output.calculated_potential.encode(&mut file)?;
//         file.write_all(b"\n|>\n")?;
//         Ok(())
//     }

//     pub fn run_scalar_perturbation_one(&self, n_range: (f64, f64), k: f64, output_selector: OutputSelector) {
//         let perturbation_output = ScalarPerturbationSimulator::new(self.background_output.as_ref().unwrap(), ScalarPerturbation, k)
//             .run::<false>(Some(n_range.0), Some(n_range.1), 1, 0.001, output_selector);
//         let efoldings = perturbation_output.scale_factor.iter().map(|f|f.ln()).collect::<Vec<_>>();

//         let perturbation_r = zip(perturbation_output.u.iter(), perturbation_output.z.iter()).map(|(u, z)|(-u / z).abs()).collect();

//         let mut plot = Plot::new();
//         let max_length = self.plot_max_length;
//         plot.add_trace(new_scatter(efoldings.clone(), perturbation_output.u.iter().cloned().map(|f|f.abs()).collect(), max_length).name("|u|").x_axis("x1").y_axis("y1"));
//         plot.add_trace(new_scatter(efoldings.clone(), perturbation_r, max_length).name("|R|").x_axis("x1").y_axis("y2"));
//         plot.add_trace(new_scatter(efoldings.clone(), perturbation_output.k2_mass.clone(), max_length).x_axis("x1").y_axis("y3").name("k^2 / m^2"));
//         plot.add_trace(new_scatter(efoldings.clone(), perturbation_output.potential.clone(), max_length).name("k^2 + m^2").x_axis("x1").y_axis("y4"));
//         plot.add_trace(new_scatter(efoldings.clone(), perturbation_output.calculated_potential.iter().map(|f|f.re).collect(), max_length).x_axis("x1").y_axis("y4").name("u'' / u"));
//         plot.add_trace(new_scatter(efoldings.clone(), perturbation_output.u.iter().cloned().map(|f|f.re).collect(), max_length).name("re u").x_axis("x1").y_axis("y5"));
//         plot.add_trace(new_scatter(efoldings.clone(), perturbation_output.u.iter().cloned().map(|f|f.im).collect(), max_length).name("im u").x_axis("x1").y_axis("y5"));
//         plot.add_trace(new_scatter(efoldings.clone(), perturbation_output.mom_u.iter().cloned().map(|f|f.re).collect(), max_length).name("re mom_u").x_axis("x1").y_axis("y6"));
//         plot.add_trace(new_scatter(efoldings.clone(), perturbation_output.mom_u.iter().cloned().map(|f|f.im).collect(), max_length).name("im mom_u").x_axis("x1").y_axis("y6"));
//         plot.add_trace(new_scatter(efoldings.clone(), perturbation_output.residual.clone(), max_length).x_axis("x1").y_axis("y7").name("residual"));
//         plot.set_layout(
//             Layout::new()
//                 .grid(
//                     LayoutGrid::new()
//                         .rows(7)
//                         .columns(1)
//                         .pattern(GridPattern::Coupled)
//                 )
//                 // .y_axis(Axis::new().type_(AxisType::Log))
//                 .y_axis2(Axis::new().type_(AxisType::Log))
//                 .y_axis3(Axis::new().type_(AxisType::Log))
//                 // .y_axis7(Axis::new().type_(AxisType::Log))
//                 .height(1400)
//         );
//         plot.write_html("out/".to_owned() + self.base_name + ".perturbation.html");

//         let mut file_name = "out/".to_owned() + self.base_name + ".perturbation";
//         if self.is_test {
//             file_name.push_str(".test");
//         }
//         file_name.push_str(".wl");
//         self.export_perturbation_to_wl(&perturbation_output, &file_name).unwrap();
//     }
//     pub fn run_tensor_perturbation_one(&self, n_range: (f64, f64), k: f64, output_selector: OutputSelector) {
//         let perturbation_output = ScalarPerturbationSimulator::new(self.background_output.as_ref().unwrap(), self.params.tensor_params, k)
//             .run::<false>(Some(n_range.0), Some(n_range.1), 1, 0.001, output_selector);
//         let efoldings = perturbation_output.scale_factor.iter().map(|f|f.ln()).collect::<Vec<_>>();

//         let h = zip(perturbation_output.u.iter(), perturbation_output.scale_factor.iter()).map(|(u, a)|(u / a).abs()).collect();

//         let mut plot = Plot::new();
//         let max_length = self.plot_max_length;
//         plot.add_trace(new_scatter(efoldings.clone(), perturbation_output.u.iter().cloned().map(|f|f.abs()).collect(), max_length).name("|u|").x_axis("x1").y_axis("y1"));
//         plot.add_trace(new_scatter(efoldings.clone(), h, max_length).name("|h|").x_axis("x1").y_axis("y2"));
//         plot.set_layout(
//             Layout::new()
//                 .grid(
//                     LayoutGrid::new()
//                         .rows(2)
//                         .columns(1)
//                         .pattern(GridPattern::Coupled)
//                 )
//                 .y_axis(Axis::new().type_(AxisType::Log))
//                 .y_axis2(Axis::new().type_(AxisType::Log))
//                 // .y_axis7(Axis::new().type_(AxisType::Log))
//                 .height(800)
//         );
//         plot.write_html("out/".to_owned() + self.base_name + ".tensor_perturbation.html");
//     }
//     fn run_spectrum_with_mode<S: PerturbationParameters + Copy + Send + Sync>(&self, mode: S, name: &str, k_range: (f64, f64), n_range: (f64, f64), count: usize) {
//         let spectrum = ScalarPerturbationSimulator::scan_spectrum(
//             |k|ScalarPerturbationSimulator::new(self.background_output.as_ref().unwrap(), mode, k),
//             k_range,
//             n_range,
//             count,
//             8,
//             0.1,
//         );
//         let mut plot = Plot::new();
//         plot.add_trace(Scatter::new(spectrum.iter().map(|a|a.0).collect(), spectrum.iter().map(|a|a.1).collect()).name("spectrum"));
//         plot.set_layout(
//             Layout::new()
//                 .x_axis(Axis::new().type_(AxisType::Log))
//                 .y_axis(Axis::new().type_(AxisType::Log)),
//         );
//         plot.write_html("out/".to_owned() + self.base_name + ".spectrum." + name + ".html");
//     }
//     pub fn run_scalar_spectrum(&self, k_range: (f64, f64), n_range: (f64, f64), count: usize) {
//         self.run_spectrum_with_mode(ScalarPerturbation, "scalar", k_range, n_range, count);
//     }
//     pub fn run_tensor_spectrum(&self, k_range: (f64, f64), n_range: (f64, f64), count: usize) {
//         self.run_spectrum_with_mode(self.params.tensor_params, "tensor", k_range, n_range, count);
//     }
//     pub fn run_initial_value_test(&self, k: f64, start_n_range: (f64, f64), end_n: f64, count: usize) {
//         let done = AtomicUsize::new(0);
//         let result = (0..count).into_par_iter().map(|i|{
//             let n = linear_interp(start_n_range.0, start_n_range.1, (i as f64) / ((count - 1) as f64));
//             let mut sim = ScalarPerturbationSimulator::new(self.background_output.as_ref().unwrap(), ScalarPerturbation, k);
//             let output = sim.run::<true>(Some(n), Some(end_n), 2, 0.001, OutputSelector::default());
//             let done0 = done.fetch_add(1, Ordering::AcqRel);
//             println!("done: ({}/ {})", done0, count);
//             (n, *output.perturbation.last().unwrap())
//         }).collect::<Vec<_>>();
//         let mut plot = Plot::new();
//         plot.add_trace(Scatter::new(result.iter().map(|f|f.0).collect(), result.iter().map(|f|f.1.abs()).collect()));
//         plot.write_html("out/".to_owned() + self.base_name + ".init_value_test.html");
//     }
// }

// fn make_test_potential_background(output: &mut BackgroundOutput, n_range: (f64, f64), amp: f64, omega: f64, phase: f64, exp_coeff: f64, apply_n_range: (f64, f64)) {
//     for i in 0..output.dt.len() {
//         let n = output.scale_factor[i].ln();
//         if n >= apply_n_range.0 && n <= apply_n_range.1 {
//             let bkgrnd = -6.99194e-11 * exp(2.0 * n);
//             let osc = if n >= n_range.0 && n <= n_range.1 { amp * exp(exp_coeff * (n - n_range.0)) * cos(n * omega + phase) } else { 0.0 };
//             output.effective_mass[i] = bkgrnd + osc;
//         }
//     }
// }

// #[derive(Clone, Copy)]
// struct TensorSoundSpeedResonancePerturbation {
//     pub alpha: f64,
//     pub tau0: f64,
//     pub k_star: f64,
// }

// impl PerturbationParameters for TensorSoundSpeedResonancePerturbation {
//     fn potential(&self, sim: &ScalarPerturbationSimulator<'_, Self>) -> f64 where Self: Sized {
//         let k = sim.k;
//         let tau = sim.time_accumulator();
//         let d = 1.0 + tau / self.tau0;
//         let c = cos(self.k_star * tau);
//         let cs2 = 1.0 - self.alpha / d / d * c * c;
//         cs2 * k * k + sim.tensor_effective_mass()
//     }

//     fn perturbation(&self, u: Complex64, sim: &ScalarPerturbationSimulator<'_, Self>) -> Complex64 where Self: Sized {
//         u / sim.scale_factor()
//     }
// }
pub struct ParametricResonanceParams {
    pub lambda: f64,
    pub phi0: f64,
    pub phi_s: f64,
    pub phi_e: f64,
    pub phi_star: f64,
    pub xi: f64,
}

impl C2Fn for ParametricResonanceParams {
    fn value(&self, phi: f64) -> f64 {
        let lambda2 = self.lambda * self.lambda;
        let lambda4 = lambda2 * lambda2;
        let a = 1.0 - exp(-sqrt(2.0 / 3.0) * phi);
        lambda4 * a * a + (if phi >= self.phi_e && phi <= self.phi_s { self.xi * sin(phi / self.phi_star) } else { 0.0 })
    }

    fn value_d(&self, phi: f64) -> f64 {
        let lambda2 = self.lambda * self.lambda;
        let lambda4 = lambda2 * lambda2;
        let sqrt_2_3 = sqrt(2.0 / 3.0);
        lambda4 * 2.0 * (1.0 - exp(-sqrt_2_3 * phi)) * sqrt_2_3 * exp(-sqrt_2_3 * phi) + (if phi >= self.phi_e && phi <= self.phi_s { self.xi / self.phi_star * cos(phi / self.phi_star) } else { 0.0 })
    }

    fn value_dd(&self, phi: f64) -> f64 {
        let lambda2 = self.lambda * self.lambda;
        let lambda4 = lambda2 * lambda2;
        let sqrt_2_3 = sqrt(2.0 / 3.0);
        lambda4 * 4.0 / 3.0 * (2.0 * exp(-2.0 * sqrt_2_3 * phi) - exp(-sqrt_2_3 * phi)) + (if phi >= self.phi_e && phi <= self.phi_s { -self.xi / self.phi_star / self.phi_star * sin(phi / self.phi_star) } else { 0.0 })
    }
}

fn run_background(out_dir: &str, param: &(&str, ParametricResonanceParams), max_length: usize) {
    let kappa = 1.0;
    let initial = BackgroundState::init(kappa, param.1.phi0, 0.0, 0.1, &param.1);
    let mut last_log_time = SystemTime::now();
    let result = initial.simulate(kappa, &param.1, 0.00005, 0.1, |s|s.phi < 0.1, |s| {
        let now = SystemTime::now();
        if last_log_time.elapsed().unwrap().as_millis() > 100 {
            last_log_time = now;
            println!("{:?}", s);
        }
    });
    {
        let out_file_name = out_dir.to_owned() + "/" + param.0 + ".background.bincode";
        encode_into_std_write(&result, &mut BufWriter::new(File::create(out_file_name).unwrap()), BINCODE_CONFIG).unwrap();
    }
    let result = limit_length(result, max_length);
    {
        let mut plot = Plot::new();
        let mut efoldings = vec![];
        let mut phi = vec![];
        let mut epsilon = vec![];
        let mut hubble_constraint = vec![];
        let mut effective_mass = vec![];
        for elem in result {
            efoldings.push(elem.a.ln());
            phi.push(elem.phi);
            epsilon.push(elem.epsilon(kappa));
            hubble_constraint.push(elem.hubble_constraint(&param.1));
            effective_mass.push(-elem.scalar_effective_mass(kappa, &param.1));
        }
        plot.add_trace(Scatter::new(efoldings.clone(), phi).name("phi"));
        plot.add_trace(Scatter::new(efoldings.clone(), epsilon).name("epsilon").x_axis("x1").y_axis("y2"));
        plot.add_trace(Scatter::new(efoldings.clone(), effective_mass).name("m^2").x_axis("x1").y_axis("y3"));
        plot.add_trace(Scatter::new(efoldings.clone(), hubble_constraint).name("hubble constraint").x_axis("x1").y_axis("y4"));
        plot.set_layout(
            Layout::new()
                .grid(
                    LayoutGrid::new()
                        .rows(4)
                        .columns(1)
                        .pattern(GridPattern::Coupled)
                )
                .y_axis2(Axis::new().type_(AxisType::Log))
                .y_axis3(Axis::new().type_(AxisType::Log))
                .y_axis4(Axis::new().type_(AxisType::Log))
                .height(1400)
        );
        plot.write_html(out_dir.to_owned() + "/" + param.0 + ".background.html");
    }
}

fn run_perturbation(out_dir: &str, param: &(&str, ParametricResonanceParams), k: f64, n_range: (Option<f64>, Option<f64>)) {
    let background: Vec<BackgroundState> = {
        let out_file_name = out_dir.to_owned() + "/" + param.0 + ".background.bincode";
        decode_from_std_read(&mut BufReader::new(File::open(out_file_name).unwrap()), BINCODE_CONFIG).unwrap()
    };
    let pert_param = ScalarPerturbation2 {
        kappa: 1.0,
        potential: &param.1,
    };
    let mut sim = PerturbationSimulator::new(k, &background, &pert_param);
    let output_selector = OutputSelector::default();
    let mut efolding = vec![];
    let mut u = vec![];
    let mut r = vec![];

    let mut last_n = 0.0;
    let mut last_log_time = SystemTime::now();
    sim.run(n_range, 0.01, 2, |b, u1| {
        let n = b.a.ln();
        if output_selector.test(last_n, n) {
            last_n = n;
            efolding.push(n);
            u.push(u1.u.abs());
            r.push(-u1.u.abs() / b.z(1.0));
        }
        if last_log_time.elapsed().unwrap().as_millis() > 100 {
            last_log_time = SystemTime::now();
            println!("N = {}, pert = {:?}", n, u1);
        }
    });
    {
        let mut plot = Plot::new();
        plot.add_trace(Scatter::new(efolding.clone(), u).name("u"));
        plot.add_trace(Scatter::new(efolding.clone(), r).name("r").x_axis("x1").y_axis("y2"));
        plot.set_layout(
            Layout::new()
                .grid(
                    LayoutGrid::new()
                        .rows(2)
                        .columns(1)
                        .pattern(GridPattern::Coupled)
                )
                .y_axis(Axis::new().type_(AxisType::Log))
                .y_axis2(Axis::new().type_(AxisType::Log))
                .height(800)
        );
        plot.write_html(out_dir.to_owned() + "/" + param.0 + ".perturbation.scalar.html");
    }
}

fn run(background: bool, set: &(&str, ParametricResonanceParams)) {
    let kappa = 1.0;
    let input = InputData {
        name: set.0,
        kappa: 1.0,
        phi0: set.1.phi0,
        a0: 0.1,
        potential: &set.1,
        pert_param: ScalarPerturbation2 {
            kappa,
            potential: &set.1,
        },
    };
    let mut ctx = Context::new("out", 500000, &input);
    if background {
        ctx.run_background(0.00005, 0.1);
    } else {
        ctx.run_perturbation(0.1, (Some(2.0), Some(40.0)));
    }
}

fn main() {
    let sets = [
        ("parameter-resonance.set1", ParametricResonanceParams {
            lambda: 0.0032,
            phi0: 5.1,
            phi_s: 4.9878,
            phi_e: 4.9731,
            phi_star: 8e-6,
            xi: 1.7e-15,
        }),
        ("parameter-resonance.set1.no-pert", ParametricResonanceParams {
            lambda: 0.0032,
            phi0: 5.1,
            phi_s: 4.9878,
            phi_e: 4.9731,
            phi_star: 8e-6,
            xi: 0.0,
        }),
        ("parameter-resonance.set2", ParametricResonanceParams {
            lambda: 0.0032,
            phi0: 5.5,
            phi_s: 5.2118,
            phi_e: 5.2088,
            phi_star: 6.64e-6,
            xi: 1.23e-15,
        }),
        ("parameter-resonance.set2.no-pert", ParametricResonanceParams {
            lambda: 0.0032,
            phi0: 5.5,
            phi_s: 5.2118,
            phi_e: 5.2088,
            phi_star: 6.64e-6,
            xi: 0.0,
        }),
    ];
    let background = !args().any(|f|f == "--perturbation");
    run(background, &sets[0]);
    // let mut runner = Runner::new(sets[0], Some(500000), false);
    // // let mut test_runner = Runner::new(sets[0], Some(500000), true);
    // // let mut runner2 = Runner::new(sets[1], Some(100000));
    // let background = !args().any(|f|f == "--perturbation");
    // if background {
    //     runner.run_background();
    // } else {
    //     runner.load_background_data();
    //     // test_runner.load_background_data();
    //     let (k_start, k_end) = runner.resonance_mode_range();
    //     println!("k_start = {}, k_end = {}", k_start, k_end);
    //     // runner.run_spectrum((0.1 * k_start, 1.5 * k_end), (2.15, 40.0), 1000);
    //     runner.run_scalar_perturbation_one((10.0, 40.0), 0.001, OutputSelector::default());
    //     // runner.run_tensor_perturbation_one((10.0, 40.0), 100.0, OutputSelector::default());
    //     // runner.run_tensor_spectrum((1.0, 10000.0), (10.0, 40.0), 1000);
    //     runner.run_initial_value_test(0.001, (10.0, 20.0), 40.0, 100);
    //     // test_runner.run_perturbation_one((-3.0, 3.0), 0.001, OutputSelector { dn: None, start_n: Some(2.21), end_n: Some(2.24) });
    // }
}
