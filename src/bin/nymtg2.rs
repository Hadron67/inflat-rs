use std::{
    f64::consts::PI,
    fs::{File, create_dir_all},
    io::BufWriter,
    iter::zip,
    ops::Index,
    time::Duration,
};

use inflat::{
    background::{
        BINCODE_CONFIG, CubicScaleFactor, DefaultPerturbationInitializer, HamitonianSimulator,
        HorizonSelector, Kappa, NymtgTensorPerturbationPotential, PhiD, ScaleFactorD,
        TwoFieldBackgroundInput, TwoFieldBackgroundInputProvider, TwoFieldBackgroundState,
    },
    c2fn::{C2Fn, Plus2},
    igw::tigw_2_spectrum,
    models::{LinearSinePotential, QuadraticPotential, StarobinskyPotential, ZeroFn},
    util::{ParamRange, RateLimiter, lazy_file, limit_length},
};
use libm::sqrt;
use ndarray::{Array, Array2};
use ndarray_npy::NpzWriter;
use plotly::{
    Layout, Plot, Scatter,
    common::ExponentFormat,
    layout::{Axis, AxisType, LayoutGrid},
};

struct Params<V, U> {
    pub a0: f64,
    pub phi0: f64,
    pub varphi0: f64,
    pub v_varphi0: f64,
    pub v0: f64,
    pub input: TwoFieldBackgroundInput<ZeroFn<f64>, Plus2<V, U, f64>>,
    pub alpha: f64,
    pub spectrum_range: ParamRange<f64>,
    pub alpha_scan_range: Option<ParamRange<f64>>,
    pub tigw2: bool,
}

impl<V, U> Params<V, U> {
    pub fn run(&self, out_dir: &str) -> anyhow::Result<()>
    where
        V: C2Fn<f64, Output = f64> + Send + Sync,
        U: C2Fn<f64, Output = f64> + Send + Sync,
    {
        create_dir_all(out_dir)?;
        let max_length = 5000000;
        let mut rate_limiter = RateLimiter::new(Duration::from_millis(100));
        let background = lazy_file(
            &format!("{}/background.bincode", out_dir),
            BINCODE_CONFIG,
            || {
                let state = TwoFieldBackgroundState::init_slowroll(
                    self.a0,
                    self.phi0,
                    self.varphi0,
                    self.v_varphi0,
                    &self.input,
                );
                state.simulate(
                    &self.input,
                    0.1,
                    10.0,
                    |s| s.epsilon(&self.input) > 1.0,
                    |state, _time| {
                        rate_limiter.run(|| {
                            println!("[background] N = {}, state = {:?}", state.a().ln(), state);
                        });
                    },
                )
            },
        )?;
        {
            let mut efolding = vec![];
            let mut phi = vec![];
            let mut v_phi = vec![];
            let mut chi = vec![];
            let mut epsilon = vec![];
            for state in limit_length(&background, max_length) {
                efolding.push(state.a().ln());
                phi.push(state.phi);
                v_phi.push(state.v_phi().abs() / self.v0.sqrt());
                chi.push(state.chi);
                epsilon.push(state.epsilon(&self.input));
            }
            let mut plot = Plot::new();
            plot.add_trace(Scatter::new(efolding.clone(), phi).name("phi"));
            plot.add_trace(
                Scatter::new(efolding.clone(), v_phi)
                    .name("v_phi")
                    .y_axis("y2"),
            );
            plot.add_trace(Scatter::new(efolding.clone(), chi).name("chi").y_axis("y3"));
            plot.add_trace(
                Scatter::new(efolding.clone(), epsilon)
                    .name("epsilon")
                    .y_axis("y4"),
            );
            plot.set_layout(
                Layout::new()
                    .grid(LayoutGrid::new().rows(4).columns(1))
                    .x_axis(Axis::new().exponent_format(ExponentFormat::Power))
                    .y_axis(Axis::new().exponent_format(ExponentFormat::Power))
                    .y_axis2(
                        Axis::new()
                            .type_(AxisType::Log)
                            .exponent_format(ExponentFormat::Power),
                    )
                    .y_axis3(Axis::new().exponent_format(ExponentFormat::Power))
                    .y_axis4(
                        Axis::new()
                            .type_(AxisType::Log)
                            .exponent_format(ExponentFormat::Power),
                    )
                    .height(1000),
            );
            plot.write_html(&format!("{}/background.html", out_dir));
        }
        let k_coef = background[0].mom_unit_coef_hz(self.input.kappa, 0.05);
        let k_range = self.spectrum_range / k_coef;
        {
            let spectrum_pos = self
                .pert(background.len(), &background, 1.0, self.alpha)
                .spectrum_with_cache(
                    &format!("{}/spectrum.+.bincode", out_dir),
                    k_range,
                    0.1,
                    false,
                )?;
            let spectrum_neg = self
                .pert(background.len(), &background, -1.0, self.alpha)
                .spectrum_with_cache(
                    &format!("{}/spectrum.-.bincode", out_dir),
                    k_range,
                    0.1,
                    false,
                )?;
            let mut plot = Plot::new();
            plot.add_trace(
                Scatter::new(self.spectrum_range.as_logspace().collect(), spectrum_pos).name("+"),
            );
            plot.add_trace(
                Scatter::new(self.spectrum_range.as_logspace().collect(), spectrum_neg).name("-"),
            );
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
                    ),
            );
            plot.write_html(&format!("{}/spectrum.html", out_dir));
        }
        if let Some(scan) = &self.alpha_scan_range {
            let mut spectrum_arr = Array2::zeros((scan.count, self.spectrum_range.count));
            let mut spectrum_scratch = vec![0.0; self.spectrum_range.count];
            let k_data = self.spectrum_range.as_logspace().collect::<Vec<_>>();
            let mut spectrum_plot = Plot::new();
            let mut tigw2_plot = Plot::new();
            for (i, alpha) in zip(0.., scan.as_linspace()) {
                let spectrum_pos = self
                    .pert(background.len(), &background, 1.0, alpha)
                    .spectrum_with_cache(
                        &format!("{}/spectrum.scan.{}.+.bincode", out_dir, i),
                        k_range,
                        0.1,
                        false,
                    )?;
                let spectrum_neg = self
                    .pert(background.len(), &background, -1.0, alpha)
                    .spectrum_with_cache(
                        &format!("{}/spectrum.scan.{}.-.bincode", out_dir, i),
                        k_range,
                        0.1,
                        false,
                    )?;
                for j in 0..self.spectrum_range.count {
                    let val = (spectrum_pos[j] + spectrum_neg[j]) / PI; // XXX: fix an error in previously saved data
                    spectrum_scratch[j] = val;
                    spectrum_arr[[i, j]] = val;
                }
                spectrum_plot.add_trace(
                    Scatter::new(k_data.clone(), spectrum_scratch.clone())
                        .name(&format!("alpha = {}", alpha)),
                );
                if self.tigw2 {
                    let tigw2_data = lazy_file(
                        &format!("{}/spectrum.scan.{}.tigw2.bincode", out_dir, i),
                        BINCODE_CONFIG,
                        || tigw_2_spectrum(&k_data, &spectrum_scratch, 100.0, 0.1, 0.1, |_, _| {}),
                    )?;
                    tigw2_plot.add_trace(
                        Scatter::new(
                            k_data.clone(),
                            spectrum_scratch.iter().map(|f| f / 12.0).collect(),
                        )
                        .name(&format!("alpha = {}", alpha)),
                    );
                    tigw2_plot.add_trace(
                        Scatter::new(k_data.clone(), tigw2_data)
                            .name(&format!("tigw2 alpha = {}", alpha)),
                    );
                }
                println!("[scan]({}/{})", i + 1, scan.count);
            }
            spectrum_plot.set_layout(
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
                    .height(1000),
            );
            spectrum_plot.write_html(&format!("{}/spectrums.scan.html", out_dir));
            if self.tigw2 {
                tigw2_plot.set_layout(
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
                        .height(1000),
                );
                tigw2_plot.write_html(&format!("{}/spectrums.scan.tigw2.html", out_dir));
            }
            {
                let mut npz = NpzWriter::new(BufWriter::new(File::create(&format!(
                    "{}/spectrums.scan.npz",
                    out_dir
                ))?));
                npz.add_array("spectrum", &spectrum_arr)?;
                npz.add_array("k", &Array::from_vec(k_data))?;
                npz.add_array("alpha", &Array::from_iter(scan.as_linspace()))?;
                npz.finish()?;
            }
        }
        Ok(())
    }
    pub fn pert<'a, 'b, I>(
        &'a self,
        length: usize,
        background: &'b I,
        lambda: f64,
        alpha: f64,
    ) -> HamitonianSimulator<
        'a,
        'b,
        Self,
        I,
        I::Output,
        DefaultPerturbationInitializer,
        NymtgTensorPerturbationPotential,
        HorizonSelector,
        CubicScaleFactor,
    >
    where
        I: Index<usize>,
        I::Output: Sized,
    {
        HamitonianSimulator::new(
            self,
            length,
            background,
            DefaultPerturbationInitializer,
            NymtgTensorPerturbationPotential { lambda, alpha },
            HorizonSelector::new(1e3),
            CubicScaleFactor,
        )
    }
}

impl<V, U> Kappa for Params<V, U> {
    fn kappa(&self) -> f64 {
        self.input.kappa
    }
}

impl<V, U> TwoFieldBackgroundInputProvider for Params<V, U> {
    type F1 = ZeroFn<f64>;

    type F2 = Plus2<V, U, f64>;

    fn input(&self) -> &TwoFieldBackgroundInput<Self::F1, Self::F2> {
        &self.input
    }
}

pub fn main() {
    let v0 = 9.75e-11;
    let sigma = 0.0002;
    let m = sqrt(0.16 * v0);
    let beta = 0.93;
    let lambda4 = beta * sigma * sigma * m * m;
    let potential_phi = QuadraticPotential {
        mass: sqrt(0.16 * v0),
    }
    .plus(LinearSinePotential {
        coef: lambda4 / sigma,
        omega: 1.0 / sigma,
    });
    let potential_varphi = StarobinskyPotential {
        v0,
        phi0: sqrt(2.0 / 3.0),
    };
    let params = Params {
        a0: 1.0,
        v0,
        phi0: 6.612e-4,
        varphi0: 5.42,
        v_varphi0: 0.0,
        alpha: 1.33e5,
        input: TwoFieldBackgroundInput {
            kappa: 1.0,
            b: ZeroFn::default(),
            v: potential_phi.plus2(potential_varphi),
        },
        spectrum_range: ParamRange::new(1e-10, 1e-6, 100),
        alpha_scan_range: Some(ParamRange::new(0.0, 1.56e5, 40)),
        tigw2: true,
        // alpha_scan_range: None,
    };
    params.run("out/nymtg2.set1/").unwrap();
}
