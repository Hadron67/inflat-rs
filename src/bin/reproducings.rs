use inflat::{
    field::Dim,
    scalar::{FullSimulateParams, SimulateParams, Simulator},
};
use libm::{cosh, round, tanh};
use ndarray::{Array2, Array3, ArrayView1, ArrayView2, ArrayViewMut2, Axis, Slice, s};
use ndarray_npy::{read_npy, write_npy};
use num_traits::Zero;
use plotly::{
    Configuration, Layout, Plot, Scatter,
    layout::{GridPattern, LayoutGrid},
};
use rayon::iter::IntoParallelRefMutIterator;
use std::{env::args, path::Path, time::SystemTime};

const OUT_DIR: &'static str = "out";

fn limit_plot_array_length<A: Clone + Zero>(arr: Array2<A>, max_length: usize) -> Array2<A> {
    if arr.dim().1 <= max_length {
        arr
    } else {
        let mut ret = Array2::zeros((arr.dim().0, max_length));
        for i in 0..max_length {
            let j =
                round((i as f64) / ((max_length - 1) as f64) * ((arr.dim().1 - 1) as f64)) as usize;
            ret.slice_mut(s![.., i]).assign(&arr.slice(s![.., j]));
        }
        ret
    }
}

fn run_simulation<T1: Fn(f64) -> f64 + Send + Sync, T2: Fn(f64) -> f64 + Send + Sync>(
    name: &str,
    params: FullSimulateParams<T1, T2>,
    mass_scale: f64,
    force_rerun: bool,
) {
    let npy_file_name = OUT_DIR.to_owned() + "/" + name + ".measurables.npy";
    let npy_path = Path::new(&npy_file_name);

    if force_rerun || !npy_path.exists() {
        println!("File not found for {}, running simulation", name);
        let mut simulator = Simulator::new(params);

        let mut out_arr = Vec::<f64>::new();

        let mut step = 0;
        let start_time = SystemTime::now();
        while simulator.scale_factor() <= 1000.0 {
            simulator.step();
            let measurable = simulator.measure();
            out_arr.push(measurable.efolding);
            out_arr.push(measurable.hamitonian);
            out_arr.push(measurable.hubble_constraint);
            out_arr.push(measurable.hubble / mass_scale);
            out_arr.push(measurable.phi);
            out_arr.push(measurable.phi_d / mass_scale);
            out_arr.push(measurable.slowroll_epsilon);
            println!("step = {}, a = {}", step, simulator.scale_factor());
            step += 1;
        }
        println!("Done in {} s", start_time.elapsed().unwrap().as_secs());
        let out_arr = Array2::from_shape_vec((out_arr.len() / 7, 7), out_arr)
            .unwrap()
            .reversed_axes();
        write_npy(npy_path, &out_arr).unwrap();
        let field = simulator.phi.data();
        write_npy(
            OUT_DIR.to_owned() + "/" + name + ".field.npy",
            &ArrayView1::from_shape(field.len(), field).unwrap(),
        )
        .unwrap();
    } else {
        println!("File exists, proceeding");
    }
    let mut measurables: Array2<f64> = read_npy(npy_path).unwrap();
    let efoldings = measurables.slice(s![0, ..]);
    let hamitonian_data = measurables.slice(s![1, ..]);
    let hubble_constraint_data = measurables.slice(s![2, ..]);
    let hubble_data = measurables.slice(s![3, ..]);
    let phi_data = measurables.slice(s![4, ..]);
    let phi_d_data = measurables.slice(s![5, ..]);
    let slowroll_epsilon_data = measurables.slice(s![6, ..]);

    let mut plot = Plot::new();
    plot.add_trace(Scatter::new(efoldings.to_vec(), hubble_data.to_vec()).name("H / m"));
    plot.add_trace(
        Scatter::new(efoldings.to_vec(), phi_data.to_vec())
            .x_axis("x2")
            .y_axis("y2")
            .name("\\phi"),
    );
    plot.add_trace(
        Scatter::new(efoldings.to_vec(), phi_d_data.to_vec())
            .x_axis("x3")
            .y_axis("y3")
            .name("\\dot\\phi / m"),
    );
    plot.add_trace(
        Scatter::new(efoldings.to_vec(), slowroll_epsilon_data.to_vec())
            .x_axis("x4")
            .y_axis("y4")
            .name("\\epsilon"),
    );
    plot.add_trace(
        Scatter::new(efoldings.to_vec(), hamitonian_data.to_vec())
            .x_axis("x5")
            .y_axis("y5")
            .name("Hamitonian"),
    );
    plot.add_trace(
        Scatter::new(efoldings.to_vec(), hubble_constraint_data.to_vec())
            .x_axis("x6")
            .y_axis("y6")
            .name("Hubble constraint"),
    );
    plot.set_layout(
        Layout::new().grid(
            LayoutGrid::new()
                .rows(2)
                .columns(3)
                .pattern(GridPattern::Independent),
        ),
    );
    plot.set_configuration(Configuration::new().typeset_math(true));
    plot.write_html(OUT_DIR.to_owned() + "/" + name + ".plot.html");
}

fn run_massive_free_scalar_sim(force_rerun: bool) {
    let mass: f64 = 0.51e-5;
    let l = 1.4 / mass;
    let size: usize = 128;
    let params = FullSimulateParams {
        params: SimulateParams {
            kappa: 1.0,
            time_step: 1.0,
            dim: Dim::new_equal(size),
            lattice_spacing: l / (size as f64),
            initial_scale_factor: 1.0,
            initial_phi: 14.5,
            initial_d_phi: -0.8152 * mass,
            random_seed: 0,
        },
        potential: |phi| mass * mass / 2.0 * phi * phi,
        potential_d: |phi| mass * mass * phi,
    };

    run_simulation("massive-free-scalar", params, mass, force_rerun);
}

fn run_stepped_potential(force_rerun: bool) {
    let mass = 0.51e-5;
    let s = 0.01;
    let d = 0.005;
    let l = 0.6 / mass;
    let size: usize = 16;
    let phi_step = 14.35;
    let params = FullSimulateParams {
        params: SimulateParams {
            time_step: 0.1,
            kappa: 1.0,
            dim: Dim {
                x: size,
                y: size,
                z: size,
            },
            lattice_spacing: l / (size as f64),
            initial_scale_factor: 1.0,
            initial_phi: 14.5,
            initial_d_phi: -0.8152 * mass,
            random_seed: 0,
        },
        potential: |phi| 0.5 * mass * mass * phi * phi * (1.0 + s * tanh((phi - phi_step) / d)),
        potential_d: |phi| {
            mass * mass * phi * (1.0 + s * tanh((phi - phi_step) / d))
                + 0.5 * mass * mass * phi * phi * s / d * {
                    let h = cosh((phi - phi_step) / d);
                    1.0 / h / h
                }
        },
    };

    run_simulation("stepped-potential", params, mass, force_rerun);
}

fn run_oscillator() {
    let mass: f64 = 0.5;
    let mut simulator = Simulator::new(FullSimulateParams {
        params: SimulateParams {
            time_step: 0.1,
            kappa: 0.01,
            dim: Dim {
                x: 16,
                y: 16,
                z: 16,
            },
            lattice_spacing: 0.1,
            initial_scale_factor: 1.0,
            initial_phi: 1.0,
            initial_d_phi: 0.0,
            random_seed: 0,
        },
        potential: |phi| mass * mass * phi * phi / 2.0,
        potential_d: |phi| mass * mass * phi,
    });
    let mut x_data = vec![];
    let mut phi_data = vec![];
    let mut hamitonian_data = vec![];
    let mut scale_factor_data = vec![];
    let mut hubble_constraint_data = vec![];
    for step in 0..1000 {
        simulator.step();
        let m = simulator.measure();
        x_data.push(step as usize);
        phi_data.push(m.phi);
        hamitonian_data.push(m.hamitonian);
        scale_factor_data.push(simulator.scale_factor());
        hubble_constraint_data.push(m.hubble_constraint);
        println!("step = {}, phi = {}", step, m.phi);
    }
    let mut plot = Plot::new();
    plot.add_trace(Scatter::new(x_data.clone(), phi_data).name("phi"));
    plot.add_trace(
        Scatter::new(x_data.clone(), hamitonian_data)
            .name("H")
            .x_axis("x2")
            .y_axis("y2"),
    );
    plot.add_trace(
        Scatter::new(x_data.clone(), scale_factor_data)
            .name("a")
            .x_axis("x3")
            .y_axis("y3"),
    );
    plot.add_trace(
        Scatter::new(x_data, hubble_constraint_data)
            .name("hc")
            .x_axis("x4")
            .y_axis("y4"),
    );
    plot.set_layout(
        Layout::new()
            .grid(
                LayoutGrid::new()
                    .rows(2)
                    .columns(2)
                    .pattern(GridPattern::Independent),
            )
            .height(900),
    );
    plot.write_html(OUT_DIR.to_owned() + "/oscillator.html");
}

fn main() {
    let mut force_rerun = false;
    for arg in args() {
        if arg == "-f" || arg == "--force-rerun" {
            force_rerun = true;
        }
    }
    run_massive_free_scalar_sim(force_rerun);
    // run_stepped_potential(force_rerun);
    // run_oscillator();
}
