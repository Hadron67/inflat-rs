use core::sync;
use std::{
    env::args,
    fs::File,
    io::{self, BufRead, BufReader, BufWriter, Write},
    iter::zip,
    time::SystemTime,
};

use bincode::decode_from_std_read;
use inflat::{
    background::{
        self, BINCODE_CONFIG, BackgroundState, Context, InputData, PerturbationParams,
        ScalarPerturbation2, SpectrumSetting, scan_spectrum,
    },
    c2fn::{self, C2Fn},
    models::{
        LinearSinePotential, ParametricResonanceParams, QuadraticPotential,
        SmearedStarobinskyLinearPotential, StarobinskyLinearPotential,
    },
    util::{self, lazy_file, limit_length, linear_interp},
    wl::WlEncode,
};
use libm::sqrt;
use ndarray::{Array2, ArrayView};
use ndarray_npy::{NpzWriter, ReadNpzError};
use num_traits::abs;
use plotly::{
    Layout, Plot, Scatter,
    common::ExponentFormat,
    layout::{Axis, AxisType, GridPattern, LayoutGrid},
};

fn run_parametric_resonance(background: bool, set: &(&str, ParametricResonanceParams)) {
    let kappa = 1.0;
    let input = InputData {
        name: set.0,
        kappa: 1.0,
        phi0: set.1.phi0,
        a0: 1.0,
        potential: &set.1,
        pert_param: ScalarPerturbation2 {
            kappa,
            potential: &set.1,
        },
    };
    let mut ctx = Context::new("out", 500000, 4, &input);
    if background {
        ctx.run_background(0.00005, 0.1);
    } else {
        // ctx.run_perturbation(1e6, (None, Some(50.0)));
        let spectrum_settings = SpectrumSetting {
            k_range: (1e5, 1e9),
            n_range: (None, None),
            count: 1000,
        };
        let spec = ctx.run_spectrum("scalar", &spectrum_settings);
        ctx.plot_spectrum(&spec, "scalar");
    }
}

fn main() {
    let sets = [
        (
            "parameter-resonance.set1",
            ParametricResonanceParams {
                lambda: 0.0032,
                phi0: 5.45,
                phi_s: 4.9878,
                phi_e: 4.9731,
                phi_star: 8e-6,
                xi: 1.7e-15,
            },
        ),
        (
            "parameter-resonance.set1.no-pert",
            ParametricResonanceParams {
                lambda: 0.0032,
                phi0: 5.1,
                phi_s: 4.9878,
                phi_e: 4.9731,
                phi_star: 8e-6,
                xi: 0.0,
            },
        ),
        (
            "parameter-resonance.set2",
            ParametricResonanceParams {
                lambda: 0.0032,
                phi0: 5.5,
                phi_s: 5.2118,
                phi_e: 5.2088,
                phi_star: 6.64e-6,
                xi: 1.23e-15,
            },
        ),
        (
            "parameter-resonance.set2.no-pert",
            ParametricResonanceParams {
                lambda: 0.0032,
                phi0: 5.5,
                phi_s: 5.2118,
                phi_e: 5.2088,
                phi_star: 6.64e-6,
                xi: 0.0,
            },
        ),
    ];
}
