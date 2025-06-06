use std::{fs::File, io::BufWriter, iter::zip, time::SystemTime};

use inflat::{
    background::{
        BINCODE_CONFIG, BackgroundState, PerturbationParams, SpectrumSetting, scan_spectrum,
    },
    c2fn::C2Fn,
    models::{LinearSinePotential, QuadraticPotential, StarobinskyLinearPotential},
    util::{lazy_file, linear_interp},
};
use ndarray::{Array2, ArrayView};
use ndarray_npy::NpzWriter;
use plotly::{
    Layout, Plot, Scatter,
    common::ExponentFormat,
    layout::{Axis, AxisType},
};

#[derive(Clone, Copy)]
struct NymtgInputParams<F> {
    pub potential: F,
    pub phi0: f64,
    pub alpha: f64,
    pub spectrum_setting: SpectrumSetting,
}

struct NymtgPerturbationParams<'a, F> {
    pub input: &'a NymtgInputParams<F>,
    pub lambda: f64,
}

impl<'a, F> PerturbationParams for NymtgPerturbationParams<'a, F>
where
    F: C2Fn<f64>,
{
    fn perturbation(
        &self,
        u: num_complex::Complex64,
        background: &inflat::background::BackgroundState,
    ) -> num_complex::Complex64 {
        -u / background.a * 2.0
    }

    fn constant_term(&self, _background: &inflat::background::BackgroundState, k: f64) -> f64 {
        k * k
    }

    fn intermediate_term(&self, background: &inflat::background::BackgroundState, k: f64) -> f64 {
        k * self.lambda * self.input.alpha * background.v_phi()
    }

    fn horizon_term(&self, background: &inflat::background::BackgroundState, _k: f64) -> f64 {
        -background.vv_a(1.0, &self.input.potential) / background.a
    }
}

fn run_nymtg_scan<F, P>(
    out_dir: &str,
    param_provider: P,
    alpha_range: (f64, f64),
    count: usize,
) -> anyhow::Result<()>
where
    F: C2Fn<f64> + Copy + Send + Sync,
    P: Fn(f64) -> NymtgInputParams<F>,
{
    let kappa = 1.0;
    let background = lazy_file(
        &format!("{}/background.bincode", out_dir),
        BINCODE_CONFIG,
        || {
            let param = param_provider(0.0);
            let initial = BackgroundState::init_slowroll(kappa, param.phi0, 1.0, &param.potential);
            let mut last_log_time = SystemTime::now();
            initial.simulate(
                kappa,
                &param.potential,
                0.001,
                0.1,
                4,
                |s| s.epsilon(kappa) > 1.0,
                |s| {
                    if last_log_time
                        .elapsed()
                        .map(|a| a.as_millis() >= 100)
                        .unwrap_or(false)
                    {
                        println!("[background] {:?}", s);
                        last_log_time = SystemTime::now();
                    }
                },
            )
        },
    )?;
    let k_coef = background[0].spectrum_k_scale_hz(kappa);
    let mut alpha_values = vec![];
    let mut k_values = vec![];
    let mut spectrums = vec![];
    for i in 0..count {
        println!("[outer]({}/{})", i + 1, count);
        let alpha = linear_interp(
            alpha_range.0,
            alpha_range.1,
            (i as f64) / ((count - 1) as f64),
        );
        alpha_values.push(alpha);
        let params = param_provider(alpha);
        let params_pos = NymtgPerturbationParams {
            input: &params,
            lambda: 1.0,
        };
        let params_neg = NymtgPerturbationParams {
            input: &params,
            lambda: -1.0,
        };
        let file_name_pos = format!("{}/spectrum.{}.+.bincode", out_dir, i);
        let file_name_neg = format!("{}/spectrum.{}.-.bincode", out_dir, i);
        let spectrum_pos = lazy_file(&file_name_pos, BINCODE_CONFIG, || {
            scan_spectrum(
                &background,
                &params_pos,
                &params_pos.input.spectrum_setting,
                1.0,
            )
        })?;
        let spectrum_neg = lazy_file(&file_name_neg, BINCODE_CONFIG, || {
            scan_spectrum(
                &background,
                &params_neg,
                &params_neg.input.spectrum_setting,
                1.0,
            )
        })?;
        let mut spectrum = vec![];
        for ((k, s1), (_, s2)) in zip(&spectrum_pos, &spectrum_neg) {
            if i == 0 {
                k_values.push(*k * k_coef);
            }
            spectrum.push(*s1 + *s2);
        }
        spectrums.push(spectrum);
    }
    {
        let mut plot = Plot::new();
        for (spectrum, alpha) in zip(&spectrums, &alpha_values) {
            plot.add_trace(
                Scatter::new(k_values.clone(), spectrum.clone())
                    .name(format!("a = {}", *alpha))
                    .y_axis("y1")
                    .x_axis("x1"),
            );
        }
        plot.set_layout(
            Layout::new()
                .y_axis(
                    Axis::new()
                        .type_(AxisType::Log)
                        .exponent_format(ExponentFormat::Power),
                )
                .x_axis(
                    Axis::new()
                        .type_(AxisType::Log)
                        .exponent_format(ExponentFormat::Power),
                )
                .height(800),
        );
        plot.write_html(format!("{}/spectrum.plot.html", out_dir));
    }
    {
        let file = BufWriter::new(File::create(&format!("{}/spectrum-data.npz", out_dir))?);
        let mut npz = NpzWriter::new(file);
        let spectrum_output = Array2::from_shape_vec(
            (spectrums.len(), spectrums[0].len()),
            spectrums.iter().flatten().cloned().collect(),
        )?;
        npz.add_array("spectrum", &spectrum_output)?;
        npz.add_array(
            "k",
            &ArrayView::from_shape((k_values.len(),), &k_values).unwrap(),
        )?;
        npz.add_array(
            "alpha",
            &ArrayView::from_shape((alpha_values.len(),), &alpha_values).unwrap(),
        )?;
        npz.finish()?;
    }
    Ok(())
}

fn run_nymtg_scan_set1() {
    run_nymtg_scan(
        "out/nymtg.set1",
        |alpha| NymtgInputParams {
            potential: StarobinskyLinearPotential {
                v0: 1e-14,
                ap: 1e-14,
                am: 1e-15,
                phi0: 6.0,
            },
            phi0: 11.32,
            alpha,
            spectrum_setting: SpectrumSetting {
                k_range: (1e-1, 1e4),
                n_range: (None, None),
                count: 1000,
            },
        },
        (24.0, 30.0),
        30,
    )
    .unwrap();
}

fn main() {
    let params = NymtgInputParams {
        potential: {
            let alpha = 3.295;
            let beta = 0.996;
            let mass = 1e-6;
            QuadraticPotential::new(mass)
                .plus(LinearSinePotential::new(beta / alpha * mass * mass, alpha))
        },
        phi0: 4.779,
        alpha: 20.0,
        spectrum_setting: SpectrumSetting {
            k_range: (1e2, 1e11),
            n_range: (None, None),
            count: 1000,
        },
    };
    run_nymtg_scan_set1();
}
