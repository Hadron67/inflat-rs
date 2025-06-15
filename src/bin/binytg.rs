use std::{iter::zip, ops::Index, process::Output, time::Duration};

use inflat::{
    background::{
        BINCODE_CONFIG, BiNymtgBackgroundState, BiNymtgBackgroundStateInput,
        BiNymtgBackgroundStateInputProvider, CubicScaleFactor, DefaultPerturbationInitializer,
        HamitonianSimulator, HorizonSelector, Kappa, NymtgTensorPerturbationPotential,
        ScaleFactorD,
    },
    c2fn::{C2Fn, C2Fn2, Plus2},
    models::{LinearSinePotential, QuadraticPotential},
    util::{RateLimiter, derivative_2, lazy_file, limit_length},
};
use libm::{cosh, exp, sqrt, tanh};
use ndarray::{Linspace, linspace};
use num_complex::ComplexFloat;
use plotly::{
    Layout, Plot, Scatter,
    common::{DashType, ExponentFormat, Line},
    layout::{Axis, AxisType, LayoutGrid},
};

struct PhiPotential {
    pub lambda2: f64,
    pub lambda3: f64,
    pub v0: f64,
    pub lambda: f64,
    pub q: f64,
}

impl C2Fn<f64> for PhiPotential {
    type Output = f64;

    #[rustfmt::skip]
    fn value(&self, phi: f64) -> Self::Output {
        (-1.0 / 2.0) * (exp((sqrt(2.0)) * (sqrt(1.0 / (self.q))) * (phi))) * (self.v0) * (1.0 + (-1.0) * (tanh((1.0 / (self.lambda2)) * (phi)))) + (1.0 / 2.0) * (self.lambda) * ((1.0 + (-1.0) * (1.0 / (self.lambda3) / (self.lambda3)) * ((phi) * (phi))) * (1.0 + (-1.0) * (1.0 / (self.lambda3) / (self.lambda3)) * ((phi) * (phi)))) * (1.0 + tanh((1.0 / (self.lambda2)) * (phi)))
    }

    #[rustfmt::skip]
    fn value_d(&self, phi: f64) -> Self::Output {
        (1.0 / 2.0) * ((exp((sqrt(2.0)) * (sqrt(1.0 / (self.q))) * (phi))) * (self.v0) * (1.0 / (self.lambda2)) * ((1.0 / cosh((1.0 / (self.lambda2)) * (phi))) * (1.0 / cosh((1.0 / (self.lambda2)) * (phi)))) + (self.lambda) * (1.0 / (self.lambda2)) * ((-1.0 + (1.0 / (self.lambda3) / (self.lambda3)) * ((phi) * (phi))) * (-1.0 + (1.0 / (self.lambda3) / (self.lambda3)) * ((phi) * (phi)))) * ((1.0 / cosh((1.0 / (self.lambda2)) * (phi))) * (1.0 / cosh((1.0 / (self.lambda2)) * (phi)))) + (sqrt(2.0)) * (exp((sqrt(2.0)) * (sqrt(1.0 / (self.q))) * (phi))) * (sqrt(1.0 / (self.q))) * (self.v0) * (-1.0 + tanh((1.0 / (self.lambda2)) * (phi))) + (4.0) * (self.lambda) * (1.0 / (self.lambda3) / (self.lambda3) / (self.lambda3) / (self.lambda3)) * (phi) * ((-1.0) * ((self.lambda3) * (self.lambda3)) + (phi) * (phi)) * (1.0 + tanh((1.0 / (self.lambda2)) * (phi))))
    }

    #[rustfmt::skip]
    fn value_dd(&self, phi: f64) -> Self::Output {
        (1.0 / (self.lambda3) / (self.lambda3) / (self.lambda3) / (self.lambda3)) * ((1.0 / (self.q)) * ((-2.0) * (self.q) * (self.lambda) * ((self.lambda3) * (self.lambda3)) + (-1.0) * (exp((sqrt(2.0)) * (sqrt(1.0 / (self.q))) * (phi))) * (self.v0) * ((self.lambda3) * (self.lambda3) * (self.lambda3) * (self.lambda3)) + (6.0) * (self.q) * (self.lambda) * ((phi) * (phi)) + ((-2.0) * (self.q) * (self.lambda) * ((self.lambda3) * (self.lambda3)) + (exp((sqrt(2.0)) * (sqrt(1.0 / (self.q))) * (phi))) * (self.v0) * ((self.lambda3) * (self.lambda3) * (self.lambda3) * (self.lambda3)) + (6.0) * (self.q) * (self.lambda) * ((phi) * (phi))) * (tanh((1.0 / (self.lambda2)) * (phi)))) + (1.0 / (self.lambda2) / (self.lambda2)) * ((1.0 / cosh((1.0 / (self.lambda2)) * (phi))) * (1.0 / cosh((1.0 / (self.lambda2)) * (phi)))) * ((sqrt(2.0)) * (exp((sqrt(2.0)) * (sqrt(1.0 / (self.q))) * (phi))) * (sqrt(1.0 / (self.q))) * (self.v0) * (self.lambda2) * ((self.lambda3) * (self.lambda3) * (self.lambda3) * (self.lambda3)) + (4.0) * (self.lambda) * (self.lambda2) * (phi) * ((-1.0) * ((self.lambda3) * (self.lambda3)) + (phi) * (phi)) + (-1.0) * ((exp((sqrt(2.0)) * (sqrt(1.0 / (self.q))) * (phi))) * (self.v0) * ((self.lambda3) * (self.lambda3) * (self.lambda3) * (self.lambda3)) + (self.lambda) * (((self.lambda3) * (self.lambda3) + (-1.0) * ((phi) * (phi))) * ((self.lambda3) * (self.lambda3) + (-1.0) * ((phi) * (phi))))) * (tanh((1.0 / (self.lambda2)) * (phi)))))
    }
}

struct PhiPotential2 {
    pub mass: f64,
    pub q2: f64,
    pub q4: f64,
    pub phi1: f64,
    pub phi2: f64,
    pub sigma: f64,
    pub lambda: f64,
}

impl C2Fn<f64> for PhiPotential2 {
    type Output = f64;

    #[rustfmt::skip]
    fn value(&self, phi: f64) -> Self::Output {
        (1.0 / 2.0) * (1.0 / (1.0 + exp((self.q2) * ((-1.0) * (self.phi2) + phi)))) * ((self.mass) * (self.mass)) * ((phi) * (phi)) + (1.0 / (1.0 + exp((-1.0) * (self.q4) * ((-1.0) * (self.phi1) + phi)))) * (self.lambda) * ((1.0 + (-1.0) * (1.0 / (self.sigma) / (self.sigma)) * (((-1.0) * (self.phi1) + phi) * ((-1.0) * (self.phi1) + phi))) * (1.0 + (-1.0) * (1.0 / (self.sigma) / (self.sigma)) * (((-1.0) * (self.phi1) + phi) * ((-1.0) * (self.phi1) + phi))))
    }

    #[rustfmt::skip]
    fn value_d(&self, phi: f64) -> Self::Output {
        (1.0 / (1.0 + exp((self.q2) * ((-1.0) * (self.phi2) + phi)))) * ((self.mass) * (self.mass)) * (phi) + (-1.0 / 2.0) * (exp((self.q2) * ((-1.0) * (self.phi2) + phi))) * (1.0 / (1.0 + exp((self.q2) * ((-1.0) * (self.phi2) + phi))) / (1.0 + exp((self.q2) * ((-1.0) * (self.phi2) + phi)))) * ((self.mass) * (self.mass)) * (self.q2) * ((phi) * (phi)) + (-4.0) * (1.0 / (1.0 + exp((-1.0) * (self.q4) * ((-1.0) * (self.phi1) + phi)))) * (self.lambda) * (1.0 / (self.sigma) / (self.sigma)) * ((-1.0) * (self.phi1) + phi) * (1.0 + (-1.0) * (1.0 / (self.sigma) / (self.sigma)) * (((-1.0) * (self.phi1) + phi) * ((-1.0) * (self.phi1) + phi))) + (exp((-1.0) * (self.q4) * ((-1.0) * (self.phi1) + phi))) * (1.0 / (1.0 + exp((-1.0) * (self.q4) * ((-1.0) * (self.phi1) + phi))) / (1.0 + exp((-1.0) * (self.q4) * ((-1.0) * (self.phi1) + phi)))) * (self.q4) * (self.lambda) * ((1.0 + (-1.0) * (1.0 / (self.sigma) / (self.sigma)) * (((-1.0) * (self.phi1) + phi) * ((-1.0) * (self.phi1) + phi))) * (1.0 + (-1.0) * (1.0 / (self.sigma) / (self.sigma)) * (((-1.0) * (self.phi1) + phi) * ((-1.0) * (self.phi1) + phi))))
    }

    #[rustfmt::skip]
    fn value_dd(&self, phi: f64) -> Self::Output {
        (1.0 / (1.0 + exp((self.q2) * ((-1.0) * (self.phi2) + phi)))) * ((self.mass) * (self.mass)) + (-2.0) * (exp((self.q2) * ((-1.0) * (self.phi2) + phi))) * (1.0 / (1.0 + exp((self.q2) * ((-1.0) * (self.phi2) + phi))) / (1.0 + exp((self.q2) * ((-1.0) * (self.phi2) + phi)))) * ((self.mass) * (self.mass)) * (self.q2) * (phi) + (exp((2.0) * (self.q2) * ((-1.0) * (self.phi2) + phi))) * (1.0 / (1.0 + exp((self.q2) * ((-1.0) * (self.phi2) + phi))) / (1.0 + exp((self.q2) * ((-1.0) * (self.phi2) + phi))) / (1.0 + exp((self.q2) * ((-1.0) * (self.phi2) + phi)))) * ((self.mass) * (self.mass)) * ((self.q2) * (self.q2)) * ((phi) * (phi)) + (-1.0 / 2.0) * (exp((self.q2) * ((-1.0) * (self.phi2) + phi))) * (1.0 / (1.0 + exp((self.q2) * ((-1.0) * (self.phi2) + phi))) / (1.0 + exp((self.q2) * ((-1.0) * (self.phi2) + phi)))) * ((self.mass) * (self.mass)) * ((self.q2) * (self.q2)) * ((phi) * (phi)) + (8.0) * (1.0 / (1.0 + exp((-1.0) * (self.q4) * ((-1.0) * (self.phi1) + phi)))) * (self.lambda) * (1.0 / (self.sigma) / (self.sigma) / (self.sigma) / (self.sigma)) * (((-1.0) * (self.phi1) + phi) * ((-1.0) * (self.phi1) + phi)) + (-4.0) * (1.0 / (1.0 + exp((-1.0) * (self.q4) * ((-1.0) * (self.phi1) + phi)))) * (self.lambda) * (1.0 / (self.sigma) / (self.sigma)) * (1.0 + (-1.0) * (1.0 / (self.sigma) / (self.sigma)) * (((-1.0) * (self.phi1) + phi) * ((-1.0) * (self.phi1) + phi))) + (-8.0) * (exp((-1.0) * (self.q4) * ((-1.0) * (self.phi1) + phi))) * (1.0 / (1.0 + exp((-1.0) * (self.q4) * ((-1.0) * (self.phi1) + phi))) / (1.0 + exp((-1.0) * (self.q4) * ((-1.0) * (self.phi1) + phi)))) * (self.q4) * (self.lambda) * (1.0 / (self.sigma) / (self.sigma)) * ((-1.0) * (self.phi1) + phi) * (1.0 + (-1.0) * (1.0 / (self.sigma) / (self.sigma)) * (((-1.0) * (self.phi1) + phi) * ((-1.0) * (self.phi1) + phi))) + (2.0) * (exp((-2.0) * (self.q4) * ((-1.0) * (self.phi1) + phi))) * (1.0 / (1.0 + exp((-1.0) * (self.q4) * ((-1.0) * (self.phi1) + phi))) / (1.0 + exp((-1.0) * (self.q4) * ((-1.0) * (self.phi1) + phi))) / (1.0 + exp((-1.0) * (self.q4) * ((-1.0) * (self.phi1) + phi)))) * ((self.q4) * (self.q4)) * (self.lambda) * ((1.0 + (-1.0) * (1.0 / (self.sigma) / (self.sigma)) * (((-1.0) * (self.phi1) + phi) * ((-1.0) * (self.phi1) + phi))) * (1.0 + (-1.0) * (1.0 / (self.sigma) / (self.sigma)) * (((-1.0) * (self.phi1) + phi) * ((-1.0) * (self.phi1) + phi)))) + (-1.0) * (exp((-1.0) * (self.q4) * ((-1.0) * (self.phi1) + phi))) * (1.0 / (1.0 + exp((-1.0) * (self.q4) * ((-1.0) * (self.phi1) + phi))) / (1.0 + exp((-1.0) * (self.q4) * ((-1.0) * (self.phi1) + phi)))) * ((self.q4) * (self.q4)) * (self.lambda) * ((1.0 + (-1.0) * (1.0 / (self.sigma) / (self.sigma)) * (((-1.0) * (self.phi1) + phi) * ((-1.0) * (self.phi1) + phi))) * (1.0 + (-1.0) * (1.0 / (self.sigma) / (self.sigma)) * (((-1.0) * (self.phi1) + phi) * ((-1.0) * (self.phi1) + phi))))
    }
}

struct PotentialG1 {
    pub q1: f64,
    pub q2: f64,
    pub f1: f64,
    pub phi0: f64,
    pub phi3: f64,
}

impl C2Fn<f64> for PotentialG1 {
    type Output = f64;

    #[rustfmt::skip]
    fn value(&self, phi: f64) -> Self::Output {
        (2.0) * (1.0 / (1.0 + exp((self.q1) * (self.phi0 + (-1.0) * (phi))))) + 1.0 / (1.0 + exp((self.q2) * ((-1.0) * (self.phi3) + phi))) + (-1.0) * (exp((2.0) * (phi))) * (self.f1) * (1.0 / (1.0 + (exp((2.0) * (phi))) * (self.f1)))
    }

    #[rustfmt::skip]
    fn value_d(&self, phi: f64) -> Self::Output {
        (2.0) * (exp((4.0) * (phi))) * ((self.f1) * (self.f1)) * (1.0 / (1.0 + (exp((2.0) * (phi))) * (self.f1)) / (1.0 + (exp((2.0) * (phi))) * (self.f1))) + (-2.0) * (exp((2.0) * (phi))) * (self.f1) * (1.0 / (1.0 + (exp((2.0) * (phi))) * (self.f1))) + (2.0) * (exp((self.q1) * (self.phi0 + phi))) * (1.0 / (exp((self.q1) * (self.phi0)) + exp((self.q1) * (phi))) / (exp((self.q1) * (self.phi0)) + exp((self.q1) * (phi)))) * (self.q1) + (-1.0) * (exp((self.q2) * (self.phi3 + phi))) * (1.0 / (exp((self.q2) * (self.phi3)) + exp((self.q2) * (phi))) / (exp((self.q2) * (self.phi3)) + exp((self.q2) * (phi)))) * (self.q2)
    }

    #[rustfmt::skip]
    fn value_dd(&self, phi: f64) -> Self::Output {
        (-8.0) * (exp((6.0) * (phi))) * ((self.f1) * (self.f1) * (self.f1)) * (1.0 / (1.0 + (exp((2.0) * (phi))) * (self.f1)) / (1.0 + (exp((2.0) * (phi))) * (self.f1)) / (1.0 + (exp((2.0) * (phi))) * (self.f1))) + (12.0) * (exp((4.0) * (phi))) * ((self.f1) * (self.f1)) * (1.0 / (1.0 + (exp((2.0) * (phi))) * (self.f1)) / (1.0 + (exp((2.0) * (phi))) * (self.f1))) + (-4.0) * (exp((2.0) * (phi))) * (self.f1) * (1.0 / (1.0 + (exp((2.0) * (phi))) * (self.f1))) + (4.0) * (exp((self.q1) * ((2.0) * (self.phi0) + phi))) * (1.0 / (exp((self.q1) * (self.phi0)) + exp((self.q1) * (phi))) / (exp((self.q1) * (self.phi0)) + exp((self.q1) * (phi))) / (exp((self.q1) * (self.phi0)) + exp((self.q1) * (phi)))) * ((self.q1) * (self.q1)) + (-2.0) * (exp((self.q1) * (self.phi0 + phi))) * (1.0 / (exp((self.q1) * (self.phi0)) + exp((self.q1) * (phi))) / (exp((self.q1) * (self.phi0)) + exp((self.q1) * (phi)))) * ((self.q1) * (self.q1)) + (2.0) * (exp((self.q2) * (self.phi3 + (2.0) * (phi)))) * (1.0 / (exp((self.q2) * (self.phi3)) + exp((self.q2) * (phi))) / (exp((self.q2) * (self.phi3)) + exp((self.q2) * (phi))) / (exp((self.q2) * (self.phi3)) + exp((self.q2) * (phi)))) * ((self.q2) * (self.q2)) + (-1.0) * (exp((self.q2) * (self.phi3 + phi))) * (1.0 / (exp((self.q2) * (self.phi3)) + exp((self.q2) * (phi))) / (exp((self.q2) * (self.phi3)) + exp((self.q2) * (phi)))) * ((self.q2) * (self.q2))
    }
}

struct PotentialG2 {
    pub f2: f64,
    pub phi0: f64,
    pub phi3: f64,
    pub q2: f64,
    pub q3: f64,
}

impl C2Fn<f64> for PotentialG2 {
    type Output = f64;

    fn value(&self, phi: f64) -> Self::Output {
        (1.0 / (1.0 + exp((self.q2) * (self.phi3 + (-1.0) * (phi)))))
            * (1.0 / (1.0 + exp((self.q3) * ((-1.0) * (self.phi0) + phi))))
            * (self.f2)
    }

    fn value_d(&self, phi: f64) -> Self::Output {
        (-1.0)
            * (exp((self.q2) * ((-1.0) * (self.phi3) + phi)))
            * (1.0
                / (1.0 + exp((self.q3) * ((-1.0) * (self.phi0) + phi)))
                / (1.0 + exp((self.q3) * ((-1.0) * (self.phi0) + phi))))
            * (1.0
                / (1.0 + exp((self.q2) * ((-1.0) * (self.phi3) + phi)))
                / (1.0 + exp((self.q2) * ((-1.0) * (self.phi3) + phi))))
            * (self.f2)
            * ((-1.0) * (1.0 + exp((self.q3) * ((-1.0) * (self.phi0) + phi))) * (self.q2)
                + (exp((self.q3) * ((-1.0) * (self.phi0) + phi)))
                    * (1.0 + exp((self.q2) * ((-1.0) * (self.phi3) + phi)))
                    * (self.q3))
    }

    fn value_dd(&self, phi: f64) -> Self::Output {
        (exp((self.q2) * ((-1.0) * (self.phi3) + phi)))
            * (1.0
                / (1.0 + exp((self.q3) * ((-1.0) * (self.phi0) + phi)))
                / (1.0 + exp((self.q3) * ((-1.0) * (self.phi0) + phi)))
                / (1.0 + exp((self.q3) * ((-1.0) * (self.phi0) + phi))))
            * (1.0
                / (1.0 + exp((self.q2) * ((-1.0) * (self.phi3) + phi)))
                / (1.0 + exp((self.q2) * ((-1.0) * (self.phi3) + phi)))
                / (1.0 + exp((self.q2) * ((-1.0) * (self.phi3) + phi))))
            * (self.f2)
            * ((-1.0)
                * ((1.0 + exp((self.q3) * ((-1.0) * (self.phi0) + phi)))
                    * (1.0 + exp((self.q3) * ((-1.0) * (self.phi0) + phi))))
                * (-1.0 + exp((self.q2) * ((-1.0) * (self.phi3) + phi)))
                * ((self.q2) * (self.q2))
                + (-2.0)
                    * (exp((self.q3) * ((-1.0) * (self.phi0) + phi)))
                    * (1.0 + exp((self.q3) * ((-1.0) * (self.phi0) + phi)))
                    * (1.0 + exp((self.q2) * ((-1.0) * (self.phi3) + phi)))
                    * (self.q2)
                    * (self.q3)
                + (exp((self.q3) * ((-1.0) * (self.phi0) + phi)))
                    * (-1.0 + exp((self.q3) * ((-1.0) * (self.phi0) + phi)))
                    * ((1.0 + exp((self.q2) * ((-1.0) * (self.phi3) + phi)))
                        * (1.0 + exp((self.q2) * ((-1.0) * (self.phi3) + phi))))
                    * ((self.q3) * (self.q3)))
    }
}

struct AlphaPotential {
    pub lambda1: f64,
    pub alpha0: f64,
    pub coef: f64,
}

impl C2Fn<f64> for AlphaPotential {
    type Output = f64;

    fn value(&self, phi: f64) -> Self::Output {
        let value = (self.alpha0)
            * ((self.lambda1) * (self.lambda1) * (self.lambda1) * (self.lambda1))
            * (1.0
                / ((self.lambda1) * (self.lambda1) + (phi) * (phi))
                / ((self.lambda1) * (self.lambda1) + (phi) * (phi)));
        value * self.coef
    }

    fn value_d(&self, phi: f64) -> Self::Output {
        let value = (-4.0)
            * (self.alpha0)
            * ((self.lambda1) * (self.lambda1) * (self.lambda1) * (self.lambda1))
            * (phi)
            * (1.0
                / ((self.lambda1) * (self.lambda1) + (phi) * (phi))
                / ((self.lambda1) * (self.lambda1) + (phi) * (phi))
                / ((self.lambda1) * (self.lambda1) + (phi) * (phi)));
        value * self.coef
    }

    fn value_dd(&self, phi: f64) -> Self::Output {
        let value = (-4.0)
            * (self.alpha0)
            * ((self.lambda1) * (self.lambda1) * (self.lambda1) * (self.lambda1))
            * ((self.lambda1) * (self.lambda1) + (-5.0) * ((phi) * (phi)))
            * (1.0
                / ((self.lambda1) * (self.lambda1) + (phi) * (phi))
                / ((self.lambda1) * (self.lambda1) + (phi) * (phi))
                / ((self.lambda1) * (self.lambda1) + (phi) * (phi))
                / ((self.lambda1) * (self.lambda1) + (phi) * (phi)));
        value * self.coef
    }
}

struct Params<A, B, V> {
    pub a0: f64,
    pub phi: f64,
    pub v_phi: f64,
    pub chi: f64,
    pub v_chi: f64,
    pub input: BiNymtgBackgroundStateInput<A, B, V>,
    pub k_range: (f64, f64),
    pub spectrum_count: usize,
}

impl<A, B, V> Params<A, B, V>
where
    A: C2Fn<f64, Output = f64> + Send + Sync,
    B: C2Fn<f64, Output = f64> + Send + Sync,
    V: C2Fn2<f64, f64, Ret = f64> + Send + Sync,
{
    pub fn run(&self, out_dir: &str) -> anyhow::Result<()> {
        let max_length = 500000;
        let mut rate_limiter = RateLimiter::new(Duration::from_millis(100));
        let background = lazy_file(
            &format!("{}/background.bincode", out_dir),
            BINCODE_CONFIG,
            || {
                let state = BiNymtgBackgroundState::init(
                    self.a0,
                    self.phi,
                    self.v_phi,
                    self.chi,
                    self.v_chi,
                    &self.input,
                );
                println!("initial: {:?}", &state);
                state.simulate(
                    &self.input,
                    0.1,
                    |s| s.a > 5e141,
                    |s| {
                        rate_limiter.run(|| println!("[background]{:?}", s));
                    },
                )
            },
        )?;
        let mut times = vec![];
        {
            let mut time = 0.0;
            for state in &background {
                times.push(time);
                time += state.dt;
            }
        }
        times = limit_length(times, max_length);
        {
            let mut phi = vec![];
            let mut scale_factor = vec![];
            let mut hubble = vec![];
            let mut epsilon = vec![];
            let mut v_phi = vec![];
            let mut v_a = vec![];
            let mut dphi_deta = vec![];
            let mut hubble_constraint = vec![];
            let mut v_chi = vec![];
            for state in &limit_length(background.clone(), max_length) {
                phi.push(state.phi);
                scale_factor.push(state.a);
                hubble.push(state.v_a / state.a);
                v_phi.push(state.v_phi);
                v_a.push(state.v_a);
                dphi_deta.push(state.v_phi * state.a);
                epsilon.push(state.epsilon(&self.input));
                hubble_constraint.push(state.hubble_constraint(&self.input).abs());
                v_chi.push(state.v_chi * 1e10);
            }
            let mut plot = Plot::new();
            plot.add_trace(Scatter::new(times.clone(), phi).name("phi"));
            plot.add_trace(
                Scatter::new(times.clone(), scale_factor)
                    .name("a")
                    .y_axis("y2"),
            );
            plot.add_trace(
                Scatter::new(times.clone(), hubble.clone())
                    .name("H")
                    .y_axis("y3"),
            );
            plot.add_trace(
                Scatter::new(times.clone(), epsilon)
                    .name("epsilon")
                    .y_axis("y4"),
            );
            plot.add_trace(
                Scatter::new(times.clone(), v_phi)
                    .name("v_phi")
                    .y_axis("y5"),
            );
            plot.add_trace(
                Scatter::new(times.clone(), hubble_constraint)
                    .name("hubble constraint")
                    .y_axis("y6"),
            );
            plot.add_trace(Scatter::new(times.clone(), v_chi).name("chi").y_axis("y7"));
            plot.add_trace(Scatter::new(times.clone(), hubble).name("chi").y_axis("y7"));
            plot.add_trace(Scatter::new(times.clone(), v_a).name("v_a").y_axis("y8"));
            plot.set_layout(
                Layout::new()
                    .grid(LayoutGrid::new().rows(8).columns(1))
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
                    .y_axis4(
                        Axis::new()
                            .type_(AxisType::Log)
                            .exponent_format(ExponentFormat::Power),
                    )
                    .y_axis5(Axis::new().exponent_format(ExponentFormat::Power))
                    .y_axis6(
                        Axis::new()
                            .type_(AxisType::Log)
                            .exponent_format(ExponentFormat::Power),
                    )
                    .y_axis7(
                        Axis::new()
                            .type_(AxisType::Log)
                            .exponent_format(ExponentFormat::Power),
                    )
                    .y_axis8(
                        Axis::new()
                            .type_(AxisType::Log)
                            .exponent_format(ExponentFormat::Power),
                    )
                    .height(1800),
            );
            plot.write_html(&format!("{}/background.html", out_dir));
        }
        {
            let pert_no_alpha = self.pert(background.len(), &background, 0.0, 0.0);
            let pert_pos = self.pert(background.len(), &background, 1.0, 1.3);
            let pert_neg = self.pert(background.len(), &background, -1.0, 1.3);
            let spectrum_no_alpha = lazy_file(
                &format!("{}/spectrum.no-alpha.bincode", out_dir),
                BINCODE_CONFIG,
                || pert_no_alpha.spectrum(self.k_range, self.spectrum_count, 0.01),
            )?;
            let spectrum_pos = lazy_file(
                &format!("{}/spectrum.+.bincode", out_dir),
                BINCODE_CONFIG,
                || pert_pos.spectrum(self.k_range, self.spectrum_count, 0.01),
            )?;
            let spectrum_neg = lazy_file(
                &format!("{}/spectrum.-.bincode", out_dir),
                BINCODE_CONFIG,
                || pert_neg.spectrum(self.k_range, self.spectrum_count, 0.01),
            )?;
            let mut plot = Plot::new();
            let k_coef = background[0].mom_unit_coef_mpc(self.input.kappa, 0.05);
            plot.add_trace(
                Scatter::new(
                    spectrum_no_alpha.iter().map(|f| f.0 * k_coef).collect(),
                    spectrum_no_alpha.iter().map(|f| f.1).collect(),
                )
                .line(Line::new().dash(DashType::Dash))
                .name("no alpha"),
            );
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
                    ),
            );
            plot.write_html(&format!("{}/spectrum.html", out_dir));
        }
        Ok(())
    }
    fn pert<'s, 'a, I>(
        &'s self,
        length: usize,
        background: &'a I,
        lambda: f64,
        alpha: f64,
    ) -> HamitonianSimulator<
        'a,
        's,
        Self,
        I,
        BiNymtgBackgroundState,
        DefaultPerturbationInitializer,
        NymtgTensorPerturbationPotential,
        HorizonSelector,
        CubicScaleFactor,
    >
    where
        I: Index<usize, Output = BiNymtgBackgroundState>,
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

impl<A, B, V> Kappa for Params<A, B, V> {
    fn kappa(&self) -> f64 {
        self.input.kappa
    }
}

impl<A, B, V> BiNymtgBackgroundStateInputProvider for Params<A, B, V> {
    type Alpha = A;

    type Beta = B;

    type V = V;

    fn input(&self) -> &BiNymtgBackgroundStateInput<Self::Alpha, Self::Beta, Self::V> {
        &self.input
    }
}

pub fn main() {
    // let params = Params {
    //     t0: -1e5,
    //     a0: 1.0,
    //     chi: -1e-4,
    //     v_chi: 3.1e-12,
    //     input: {
    //         let lambda1 = 0.224;
    //         let alpha0 = 20.0;
    //         let beta0 = 4.5e9;
    //         let ma = 3.32e-6;
    //         let fa = 1e-5;
    //         let lambdaa4 = ma * ma * fa * fa;
    //         BiNymtgBackgroundStateInput {
    //             kappa: 1.0,
    //             dim: 3.0,
    //             alpha: AlphaPotential {
    //                 lambda1,
    //                 alpha0,
    //                 coef: 1.0,
    //             },
    //             beta: AlphaPotential {
    //                 lambda1,
    //                 alpha0,
    //                 coef: beta0 / alpha0,
    //             },
    //             potential_v: PhiPotential {
    //                 lambda2: 0.067,
    //                 lambda3: 12.0,
    //                 v0: 5e-9,
    //                 lambda: 2.5e-9,
    //                 q: 0.1,
    //             }
    //             .plus2(QuadraticPotential { mass: ma }.plus(LinearSinePotential {
    //                 coef: lambdaa4 / fa,
    //                 omega: 1.0 / fa,
    //             })),
    //         }
    //     },
    // };
    let params2 = Params {
        a0: 1.0,
        phi: -7.0,
        v_phi: 0.0,
        chi: -1e-4,
        v_chi: 3.1e-12,
        input: {
            let phi0 = 3.2;
            let phi1 = 2.0;
            let phi2 = -4.0;
            let phi3 = -4.38;
            let q1 = 10.0;
            let q2 = 6.0;
            let q3 = 10.0;
            let q4 = 4.0;
            let f1 = 1.0;
            let f2 = 40.0;
            let lambda = 0.01;
            let sigma = 23.0;
            let mass = -4.5e-6;
            let ma = 3.32e-6;
            let fa = 1e-5;
            let lambda4 = ma * ma * fa * fa;
            BiNymtgBackgroundStateInput {
                kappa: 1.0,
                dim: 3.0,
                alpha: PotentialG1 {
                    q1,
                    q2,
                    f1,
                    phi0,
                    phi3,
                },
                beta: PotentialG2 {
                    f2,
                    phi0,
                    phi3,
                    q2,
                    q3,
                },
                potential_v: PhiPotential2 {
                    mass,
                    q2,
                    q4,
                    phi1,
                    phi2,
                    sigma,
                    lambda,
                }
                .plus2(QuadraticPotential { mass: ma }.plus(LinearSinePotential {
                    coef: lambda4 / fa,
                    omega: 1.0 / fa,
                })),
            }
        },
        k_range: (1e-1, 1e6),
        spectrum_count: 1000,
    };
    params2.run("out/binytg.set2").unwrap();
}
