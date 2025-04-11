use std::{path::Path, time::SystemTime};
use inflat::{Dim, FullSimulateParams, SimulateParams, Simulator};
use libm::round;
use ndarray::{s, Array2, ArrayView1};
use ndarray_npy::{read_npy, write_npy};
use num_traits::Zero;
use plotly::{layout::{GridPattern, LayoutGrid}, Configuration, Layout, Plot, Scatter};

const OUT_DIR: &'static str = "out";

fn limit_plot_array_length<A: Clone + Zero>(arr: Array2<A>, max_length: usize) -> Array2<A> {
    if arr.dim().1 <= max_length {
        arr
    } else {
        let mut ret = Array2::zeros((arr.dim().0, max_length));
        for i in 0..max_length {
            let j = round((i as f64) / ((max_length - 1) as f64) * ((arr.dim().1 - 1) as f64)) as usize;
            ret.slice_mut(s![.., i]).assign(&arr.slice(s![.., j]));
        }
        ret
    }
}

fn run_simulation<T1: (Fn(f64) -> f64) + Send + Sync, T2: (Fn(f64) -> f64) + Send + Sync>(name: &str, params: FullSimulateParams<T1, T2>, mass_scale: f64) {
    let npy_file_name = OUT_DIR.to_owned() + "/" + name + ".measurables.npy";
    let npy_path = Path::new(&npy_file_name);

    if !npy_path.exists() {
        println!("File not found for {}, running simulation", name);
        let mut simulator = Simulator::new(params);

        let mut out_arr = Vec::<f64>::new();

        let mut step = 0;
        let start_time = SystemTime::now();
        while simulator.scale_factor() <= 1000.0 {
            simulator.step();
            let measurable = simulator.measure();
            out_arr.push(measurable.efolding);
            out_arr.push(measurable.hubble / mass_scale);
            out_arr.push(measurable.phi);
            out_arr.push(measurable.phi_d / mass_scale);
            out_arr.push(measurable.slowroll_epsilon);
            println!("step = {}, a = {}", step, simulator.scale_factor());
            step += 1;
        }
        println!("Done in {} s", start_time.elapsed().unwrap().as_secs());
        let out_arr = Array2::from_shape_vec((out_arr.len() / 5, 5), out_arr).unwrap().reversed_axes();
        write_npy(npy_file_name, &out_arr).unwrap();
        let field = simulator.phi.as_ref();
        write_npy(OUT_DIR.to_owned() + "/" + name + ".field.npy", &ArrayView1::from_shape(field.len(), field).unwrap()).unwrap();
    } else {
        println!("File exists, proceeding");
        let measurables: Array2<f64> = read_npy(npy_path).unwrap();
        let measurables = limit_plot_array_length(measurables, 100);
        let efoldings = measurables.slice(s![0, ..]);
        let hubble_data = measurables.slice(s![1, ..]);
        let phi_data = measurables.slice(s![2, ..]);
        let phi_d_data = measurables.slice(s![3, ..]);
        let slowroll_epsilon_data = measurables.slice(s![4, ..]);

        let mut plot = Plot::new();
        let hubble_plot = Scatter::new(efoldings.to_vec(), hubble_data.to_vec()).name("H / m");
        let phi_plot = Scatter::new(efoldings.to_vec(), phi_data.to_vec()).x_axis("x2").y_axis("y2").name("\\phi");
        let phi_d_plot = Scatter::new(efoldings.to_vec(), phi_d_data.to_vec()).x_axis("x3").y_axis("y3").name("\\dot\\phi / m");
        let slowroll_epsilon_plot = Scatter::new(efoldings.to_vec(), slowroll_epsilon_data.to_vec()).x_axis("x4").y_axis("y4").name("\\epsilon");
        plot.add_trace(hubble_plot);
        plot.add_trace(phi_plot);
        plot.add_trace(phi_d_plot);
        plot.add_trace(slowroll_epsilon_plot);
        plot.set_layout(Layout::new().grid(LayoutGrid::new().rows(2).columns(2).pattern(GridPattern::Independent)));
        plot.set_configuration(Configuration::new().typeset_math(true));
        plot.write_html(OUT_DIR.to_owned() + "/" + name + ".plot.html");
        plot.show();
    }
}

fn main() {
    let mass: f64 = 0.51e-5;
    let l = 1.4 / mass;
    let size: usize = 128;
    let params = FullSimulateParams {
        params: SimulateParams {
            kappa: 1.0,
            time_step: 0.001,
            dim: Dim { x: size, y: size, z: size },
            lattice_spacing: (size as f64) / l,
            initial_scale_factor: 1.0,
            initial_phi: 14.5,
            initial_d_phi: -0.8152 * mass,
        },
        potential: |phi| mass * mass / 2.0 * phi * phi,
        potential_d: |phi|mass * mass * phi,
    };

    run_simulation("massive-free-scalar", params, mass);
}
