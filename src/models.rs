use std::marker::PhantomData;

use libm::{cos, cosh, exp, sin, sqrt, tanh};
use num_traits::Zero;

use crate::{c2fn::C2Fn, util::sigmoid};

pub struct ZeroFn<T>(PhantomData<T>);

impl<T> Default for ZeroFn<T> {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<T> C2Fn<T> for ZeroFn<T>
where
    T: Zero,
{
    type Output = T;
    fn value(&self, _phi: T) -> T {
        T::zero()
    }

    fn value_d(&self, _phi: T) -> T {
        T::zero()
    }

    fn value_dd(&self, _phi: T) -> T {
        T::zero()
    }
}

pub struct ParametricResonanceParams {
    pub lambda: f64,
    pub phi0: f64,
    pub phi_s: f64,
    pub phi_e: f64,
    pub phi_star: f64,
    pub xi: f64,
}

impl C2Fn<f64> for ParametricResonanceParams {
    type Output = f64;
    fn value(&self, phi: f64) -> f64 {
        let lambda2 = self.lambda * self.lambda;
        let lambda4 = lambda2 * lambda2;
        let a = 1.0 - exp(-sqrt(2.0 / 3.0) * phi);
        lambda4 * a * a
            + (if phi >= self.phi_e && phi <= self.phi_s {
                self.xi * sin(phi / self.phi_star)
            } else {
                0.0
            })
    }

    fn value_d(&self, phi: f64) -> f64 {
        let lambda2 = self.lambda * self.lambda;
        let lambda4 = lambda2 * lambda2;
        let sqrt_2_3 = sqrt(2.0 / 3.0);
        lambda4 * 2.0 * (1.0 - exp(-sqrt_2_3 * phi)) * sqrt_2_3 * exp(-sqrt_2_3 * phi)
            + (if phi >= self.phi_e && phi <= self.phi_s {
                self.xi / self.phi_star * cos(phi / self.phi_star)
            } else {
                0.0
            })
    }

    fn value_dd(&self, phi: f64) -> f64 {
        let lambda2 = self.lambda * self.lambda;
        let lambda4 = lambda2 * lambda2;
        let sqrt_2_3 = sqrt(2.0 / 3.0);
        lambda4 * 4.0 / 3.0 * (2.0 * exp(-2.0 * sqrt_2_3 * phi) - exp(-sqrt_2_3 * phi))
            + (if phi >= self.phi_e && phi <= self.phi_s {
                -self.xi / self.phi_star / self.phi_star * sin(phi / self.phi_star)
            } else {
                0.0
            })
    }
}

#[derive(Clone, Copy)]
pub struct StarobinskyLinearPotential {
    pub v0: f64,
    pub ap: f64,
    pub am: f64,
    pub phi0: f64,
}

impl C2Fn<f64> for StarobinskyLinearPotential {
    type Output = f64;
    fn value(&self, phi: f64) -> f64 {
        self.v0 + (if phi > self.phi0 { self.ap } else { self.am }) * (phi - self.phi0)
    }

    fn value_d(&self, phi: f64) -> f64 {
        if phi > self.phi0 { self.ap } else { self.am }
    }

    fn value_dd(&self, _phi: f64) -> f64 {
        0.0
    }
}

pub struct SmearedStarobinskyLinearPotential {
    pub v0: f64,
    pub ap: f64,
    pub am: f64,
    pub phi0: f64,
    pub phi1: f64,
}

impl C2Fn<f64> for SmearedStarobinskyLinearPotential {
    type Output = f64;
    fn value(&self, phi: f64) -> f64 {
        self.v0
            + (phi - self.phi0)
                * (self.am + (self.ap - self.am) * sigmoid((phi - self.phi0) / self.phi1))
    }

    fn value_d(&self, phi: f64) -> f64 {
        let ls = sigmoid((phi - self.phi0) / self.phi1);
        self.am
            + (self.am - self.ap) / self.phi1
                * ls
                * (-phi + self.phi0 - self.phi1 + (phi - self.phi0) * ls)
    }

    fn value_dd(&self, phi: f64) -> f64 {
        let ls = sigmoid((phi - self.phi0) / self.phi1);
        (self.am - self.ap) / self.phi1 / self.phi1
            * (-1.0 + ls)
            * ls
            * (phi - self.phi0 + 2.0 * self.phi1 + 2.0 * (-phi + self.phi0) * ls)
    }
}

pub struct QuadraticPotential {
    pub mass: f64,
}

impl QuadraticPotential {
    pub const fn new(mass: f64) -> Self {
        Self { mass }
    }
}

impl C2Fn<f64> for QuadraticPotential {
    type Output = f64;
    fn value(&self, phi: f64) -> f64 {
        0.5 * self.mass * self.mass * phi * phi
    }

    fn value_d(&self, phi: f64) -> f64 {
        self.mass * self.mass * phi
    }

    fn value_dd(&self, _: f64) -> f64 {
        self.mass * self.mass
    }
}

pub struct LinearSinePotential {
    pub coef: f64,
    pub omega: f64,
}

impl LinearSinePotential {
    pub const fn new(coef: f64, omega: f64) -> Self {
        Self { coef, omega }
    }
}

impl C2Fn<f64> for LinearSinePotential {
    type Output = f64;
    fn value(&self, phi: f64) -> f64 {
        self.coef * phi * sin(self.omega * phi)
    }

    fn value_d(&self, phi: f64) -> f64 {
        self.coef * (self.omega * phi * cos(self.omega * phi) + sin(self.omega * phi))
    }

    fn value_dd(&self, phi: f64) -> f64 {
        self.coef
            * (2.0 * self.omega * cos(self.omega * phi)
                - self.omega * self.omega * phi * sin(self.omega * phi))
    }
}

pub struct StarobinskyPotential {
    pub v0: f64,
    pub phi0: f64,
}

impl C2Fn<f64> for StarobinskyPotential {
    type Output = f64;
    fn value(&self, phi: f64) -> f64 {
        let e = 1.0 - exp(-phi * self.phi0);
        self.v0 * e * e
    }

    fn value_d(&self, phi: f64) -> f64 {
        self.v0 * 2.0 * (1.0 - exp(-phi * self.phi0)) * exp(-phi * self.phi0) * self.phi0
    }

    fn value_dd(&self, phi: f64) -> f64 {
        2.0 * self.v0
            * self.phi0
            * self.phi0
            * exp(-phi * self.phi0)
            * (2.0 * exp(-phi * self.phi0) - 1.0)
    }
}

pub struct TruncSinePotential {
    pub amp: f64,
    pub omega: f64,
    pub begin: f64,
    pub end: f64,
}

impl C2Fn<f64> for TruncSinePotential {
    type Output = f64;

    fn value(&self, phi: f64) -> Self::Output {
        if phi >= self.begin && phi <= self.end {
            sin(phi * self.omega) * self.amp
        } else {
            0.0
        }
    }

    fn value_d(&self, phi: f64) -> Self::Output {
        if phi >= self.begin && phi <= self.end {
            cos(phi * self.omega) * self.omega * self.amp
        } else {
            0.0
        }
    }

    fn value_dd(&self, phi: f64) -> Self::Output {
        if phi >= self.begin && phi <= self.end {
            -sin(phi * self.omega) * self.omega * self.omega * self.amp
        } else {
            0.0
        }
    }
}

pub struct TanhPotential {
    pub coef: f64,
    pub omega: f64,
    pub shift: f64,
}

impl C2Fn<f64> for TanhPotential {
    type Output = f64;

    fn value(&self, phi: f64) -> Self::Output {
        self.coef * tanh(self.omega * phi + self.shift)
    }

    fn value_d(&self, phi: f64) -> Self::Output {
        let s = cosh(self.omega * phi + self.shift);
        self.coef * self.omega / s / s
    }

    fn value_dd(&self, phi: f64) -> Self::Output {
        let t = tanh(self.omega * phi + self.shift);
        let s = cosh(self.omega * phi + self.shift);
        -2.0 * self.coef * self.omega * self.omega * t / s / s
    }
}
