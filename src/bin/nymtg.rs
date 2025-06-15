use std::{iter::zip, ops::Index, time::Duration};

use inflat::{
    background::{
        BINCODE_CONFIG, BackgroundState, BackgroundStateInput, BackgroundStateInputProvider,
        CubicScaleFactor, DefaultPerturbationInitializer, HamitonianSimulator, HorizonSelector,
        Kappa, NymtgTensorPerturbationPotential, PhiD, ScaleFactor, ScaleFactorD,
    },
    c2fn::C2Fn,
    models::{LinearSinePotential, QuadraticPotential, StarobinskyLinearPotential},
    util::{RateLimiter, lazy_file, limit_length, linear_interp},
};
use plotly::{
    Layout, Plot, Scatter,
    common::ExponentFormat,
    layout::{Axis, AxisType, LayoutGrid},
};

struct NymtgInputParams<F> {
    pub input: BackgroundStateInput<F>,
    pub a0: f64,
    pub phi0: f64,
    pub spectrum_range: (f64, f64),
    pub spectrum_count: usize,
    pub alpha: f64,
    pub max_dt: f64,
    pub alpha_scan: Option<ParamRange>,
}

struct ParamRange {
    pub range: (f64, f64),
    pub count: usize,
}

impl<F> Kappa for NymtgInputParams<F> {
    fn kappa(&self) -> f64 {
        self.input.kappa
    }
}

impl<F> BackgroundStateInputProvider for NymtgInputParams<F> {
    type F = F;

    fn input(&self) -> &BackgroundStateInput<Self::F> {
        &self.input
    }
}

impl<F> NymtgInputParams<F>
where
    F: C2Fn<f64, Output = f64> + Send + Sync,
{
    pub fn run(&self, out_dir: &str) -> anyhow::Result<()> {
        let max_length = 500000usize;
        let background = lazy_file(
            &format!("{}/background.bincode", out_dir),
            BINCODE_CONFIG,
            || {
                let initial = BackgroundState::init_slowroll(self.a0, self.phi0, &self.input);
                let mut rate_limiter = RateLimiter::new(Duration::from_millis(100));
                initial.simulate(
                    &self.input,
                    0.01,
                    self.max_dt,
                    4,
                    |s| s.epsilon(&self.input) > 1.0,
                    |s| {
                        rate_limiter.run(|| {
                            println!("[background] {:?}", s);
                        });
                    },
                )
            },
        )?;
        {
            let mut efolding = vec![];
            let mut phi = vec![];
            let mut v_phi = vec![];
            let mut epsilon = vec![];
            for state in limit_length(background.clone(), max_length) {
                efolding.push(state.scale_factor().ln());
                phi.push(state.phi);
                v_phi.push(state.v_phi().abs());
                epsilon.push(state.epsilon(&self.input));
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
            plot.set_layout(
                Layout::new()
                    .grid(LayoutGrid::new().rows(3).columns(1))
                    .x_axis(Axis::new().exponent_format(ExponentFormat::Power))
                    .y_axis(Axis::new().exponent_format(ExponentFormat::Power))
                    .y_axis2(
                        Axis::new()
                            .type_(AxisType::Log)
                            .exponent_format(ExponentFormat::Power),
                    )
                    .y_axis3(
                        Axis::new()
                            .type_(AxisType::Log)
                            .exponent_format(ExponentFormat::Power),
                    )
                    .height(1000),
            );
            plot.write_html(&format!("{}/background.html", out_dir));
        }
        let k_coef = background[0].mom_unit_coef_hz(self.input.kappa, 0.05);
        let k_range = (
            self.spectrum_range.0 / k_coef,
            self.spectrum_range.1 / k_coef,
        );
        {
            let pert_pos = self.pert(background.len(), &background, 1.0, self.alpha);
            let spectrum_pos = pert_pos.spectrum_with_cache(
                &format!("{}/spectrum.+.bincode", out_dir),
                k_range,
                self.spectrum_count,
                0.1,
            )?;
            let pert_neg = self.pert(background.len(), &background, -1.0, self.alpha);
            let spectrum_neg = pert_neg.spectrum_with_cache(
                &format!("{}/spectrum.+.bincode", out_dir),
                k_range,
                self.spectrum_count,
                0.1,
            )?;
            let mut plot = Plot::new();
            plot.add_trace(
                Scatter::new(
                    spectrum_pos.iter().map(|f| f.0 * k_coef).collect(),
                    spectrum_pos.iter().map(|f| f.1).collect(),
                )
                .name("+"),
            );
            plot.add_trace(
                Scatter::new(
                    spectrum_neg.iter().map(|f| f.0 * k_coef).collect(),
                    spectrum_neg.iter().map(|f| f.1).collect(),
                )
                .name("-"),
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
                    )
                    .height(800),
            );
            plot.write_html(&format!("{}/spectrum.html", out_dir));
        }
        if let Some(scan) = &self.alpha_scan {
            let mut plot = Plot::new();
            for i in 0..scan.count {
                println!("[scan]({}/{})", i + 1, scan.count);
                let alpha = linear_interp(
                    scan.range.0,
                    scan.range.1,
                    (i as f64) / ((scan.count - 1) as f64),
                );
                let pert_pos = self.pert(background.len(), &background, 1.0, alpha);
                let pert_neg = self.pert(background.len(), &background, -1.0, alpha);
                let spectrum_pos = pert_pos.spectrum_with_cache(
                    &format!("{}/spectrum.scan1.{}.+.bincode", out_dir, i),
                    k_range,
                    self.spectrum_count,
                    0.1,
                )?;
                let spectrum_neg = pert_neg.spectrum_with_cache(
                    &format!("{}/spectrum.scan1.{}.-.bincode", out_dir, i),
                    k_range,
                    self.spectrum_count,
                    0.1,
                )?;
                plot.add_trace(Scatter::new(spectrum_pos.iter().map(|f|f.0).collect(), zip(&spectrum_pos, &spectrum_neg).map(|f|f.0.1 + f.1.1).collect()).name(&format!("alpha = {}", alpha)));
            }
            plot.write_html(&format!("{}/spectrum.scan.html", out_dir));
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
        'b,
        'a,
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

fn main() {
    let params_2112_04794 = NymtgInputParams {
        input: BackgroundStateInput {
            potential: {
                let alpha = 3.295;
                let beta = 0.996;
                let mass = 1e-6;
                QuadraticPotential::new(mass)
                    .plus(LinearSinePotential::new(beta / alpha * mass * mass, alpha))
            },
            kappa: 1.0,
        },
        phi0: 4.779,
        a0: 1.0,
        max_dt: 10.0,
        spectrum_count: 100,
        spectrum_range: (1e-6, 1.0),
        alpha: 20.0,
        alpha_scan: None,
    };
    params_2112_04794.run("out/nymtg-2112.04794.set1").unwrap();
    let params_2308_15329 = NymtgInputParams {
        phi0: 11.32,
        input: BackgroundStateInput {
            potential: StarobinskyLinearPotential {
                v0: 1e-14,
                ap: 1e-14,
                am: 1e-15,
                phi0: 6.0,
            },
            kappa: 1.0,
        },
        a0: 1.0,
        max_dt: 100.0,
        spectrum_range: (1e-10, 1e-5),
        spectrum_count: 100,
        alpha: 30.0,
        alpha_scan: None,
    };
    params_2308_15329.run("out/nymtg-2308.15329.set1").unwrap();

    let params_2112_04794_mod = NymtgInputParams {
        input: BackgroundStateInput {
            potential: {
                let alpha = 3.285;
                let beta = 0.9959;
                let mass = 1e-6;
                QuadraticPotential::new(mass)
                    .plus(LinearSinePotential::new(beta / alpha * mass * mass, alpha))
            },
            kappa: 1.0,
        },
        phi0: 4.779,
        a0: 1.0,
        max_dt: 10.0,
        spectrum_count: 1000,
        spectrum_range: (1e-10, 1e-5),
        alpha: 20.0,
        alpha_scan: Some(ParamRange {
            range: (0.0, 30.0),
            count: 40,
        }),
    };
    params_2112_04794_mod
        .run("out/nymtg-2308.15329-mod.set1")
        .unwrap();
}
