use std::env::args;

use inflat::background::{C2Fn, Context, InputData, ScalarPerturbation2};
use libm::{cos, exp, sin, sqrt};

// fn limit_length<T: Clone>(arr: Vec<T>, max_length: usize) -> Vec<T> {
//     if arr.len() > max_length {
//         let mut arr2 = vec![];
//         arr2.reserve(max_length);
//         for i in 0..max_length {
//             arr2.push(arr[((i as f64) / ((max_length - 1) as f64) * ((arr.len() - 1) as f64)) as usize].clone());
//         }
//         arr2
//     } else {
//         arr
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

// fn run_background(out_dir: &str, param: &(&str, ParametricResonanceParams), max_length: usize) {
//     let kappa = 1.0;
//     let initial = BackgroundState::init(kappa, param.1.phi0, 0.0, 0.1, &param.1);
//     let mut last_log_time = SystemTime::now();
//     let result = initial.simulate(kappa, &param.1, 0.00005, 0.1, |s|s.phi < 0.1, |s| {
//         let now = SystemTime::now();
//         if last_log_time.elapsed().unwrap().as_millis() > 100 {
//             last_log_time = now;
//             println!("{:?}", s);
//         }
//     });
//     {
//         let out_file_name = out_dir.to_owned() + "/" + param.0 + ".background.bincode";
//         encode_into_std_write(&result, &mut BufWriter::new(File::create(out_file_name).unwrap()), BINCODE_CONFIG).unwrap();
//     }
//     let result = limit_length(result, max_length);
//     {
//         let mut plot = Plot::new();
//         let mut efoldings = vec![];
//         let mut phi = vec![];
//         let mut epsilon = vec![];
//         let mut hubble_constraint = vec![];
//         let mut effective_mass = vec![];
//         for elem in result {
//             efoldings.push(elem.a.ln());
//             phi.push(elem.phi);
//             epsilon.push(elem.epsilon(kappa));
//             hubble_constraint.push(elem.hubble_constraint(&param.1));
//             effective_mass.push(-elem.scalar_effective_mass(kappa, &param.1));
//         }
//         plot.add_trace(Scatter::new(efoldings.clone(), phi).name("phi"));
//         plot.add_trace(Scatter::new(efoldings.clone(), epsilon).name("epsilon").x_axis("x1").y_axis("y2"));
//         plot.add_trace(Scatter::new(efoldings.clone(), effective_mass).name("m^2").x_axis("x1").y_axis("y3"));
//         plot.add_trace(Scatter::new(efoldings.clone(), hubble_constraint).name("hubble constraint").x_axis("x1").y_axis("y4"));
//         plot.set_layout(
//             Layout::new()
//                 .grid(
//                     LayoutGrid::new()
//                         .rows(4)
//                         .columns(1)
//                         .pattern(GridPattern::Coupled)
//                 )
//                 .y_axis2(Axis::new().type_(AxisType::Log))
//                 .y_axis3(Axis::new().type_(AxisType::Log))
//                 .y_axis4(Axis::new().type_(AxisType::Log))
//                 .height(1400)
//         );
//         plot.write_html(out_dir.to_owned() + "/" + param.0 + ".background.html");
//     }
// }

// fn run_perturbation(out_dir: &str, param: &(&str, ParametricResonanceParams), k: f64, n_range: (Option<f64>, Option<f64>)) {
//     let background: Vec<BackgroundState> = {
//         let out_file_name = out_dir.to_owned() + "/" + param.0 + ".background.bincode";
//         decode_from_std_read(&mut BufReader::new(File::open(out_file_name).unwrap()), BINCODE_CONFIG).unwrap()
//     };
//     let pert_param = ScalarPerturbation2 {
//         kappa: 1.0,
//         potential: &param.1,
//     };
//     let mut sim = PerturbationSimulator::new(k, &background, &pert_param);
//     let output_selector = OutputSelector::default();
//     let mut efolding = vec![];
//     let mut u = vec![];
//     let mut r = vec![];

//     let mut last_n = 0.0;
//     let mut last_log_time = SystemTime::now();
//     sim.run(n_range, 0.01, 2, |b, u1| {
//         let n = b.a.ln();
//         if output_selector.test(last_n, n) {
//             last_n = n;
//             efolding.push(n);
//             u.push(u1.u.abs());
//             r.push(-u1.u.abs() / b.z(1.0));
//         }
//         if last_log_time.elapsed().unwrap().as_millis() > 100 {
//             last_log_time = SystemTime::now();
//             println!("N = {}, pert = {:?}", n, u1);
//         }
//     });
//     {
//         let mut plot = Plot::new();
//         plot.add_trace(Scatter::new(efolding.clone(), u).name("u"));
//         plot.add_trace(Scatter::new(efolding.clone(), r).name("r").x_axis("x1").y_axis("y2"));
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
//                 .height(800)
//         );
//         plot.write_html(out_dir.to_owned() + "/" + param.0 + ".perturbation.scalar.html");
//     }
// }

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
        ctx.run_spectrum((0.01, 0.2), (Some(2.0), Some(40.0)), 1000);
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
}
