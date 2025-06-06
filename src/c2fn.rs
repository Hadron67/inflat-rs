use std::ops::{Add, Mul};

use num_traits::FromPrimitive;

pub trait C2Fn<T> {
    fn value(&self, phi: T) -> T;
    fn value_d(&self, phi: T) -> T;
    fn value_dd(&self, phi: T) -> T;

    fn plus<F: C2Fn<T>>(self, other: F) -> Plus<Self, F>
    where
        Self: Sized,
    {
        Plus {
            f1: self,
            f2: other,
        }
    }
    fn mul<F: C2Fn<T>>(self, other: F) -> Times<Self, F>
    where
        Self: Sized,
    {
        Times {
            f1: self,
            f2: other,
        }
    }
}

pub trait C2Fn2<T1, T2> {
    type Ret;
    fn value_00(&self, phi1: T1, phi2: T2) -> Self::Ret;
    fn value_10(&self, phi1: T1, phi2: T2) -> Self::Ret;
    fn value_01(&self, phi1: T1, phi2: T2) -> Self::Ret;
    fn value_11(&self, phi1: T1, phi2: T2) -> Self::Ret;
    fn value_20(&self, phi1: T1, phi2: T2) -> Self::Ret;
    fn value_02(&self, phi1: T1, phi2: T2) -> Self::Ret;
}

pub struct Plus<F1, F2> {
    pub f1: F1,
    pub f2: F2,
}

impl<T, F1, F2> C2Fn<T> for Plus<F1, F2>
where
    F1: C2Fn<T>,
    F2: C2Fn<T>,
    T: Add<T, Output = T> + Copy,
{
    fn value(&self, phi: T) -> T {
        self.f1.value(phi) + self.f2.value(phi)
    }

    fn value_d(&self, phi: T) -> T {
        self.f1.value_d(phi) + self.f2.value_d(phi)
    }

    fn value_dd(&self, phi: T) -> T {
        self.f1.value_dd(phi) + self.f2.value_dd(phi)
    }
}

pub struct Times<F1, F2> {
    pub f1: F1,
    pub f2: F2,
}

impl<T, F1, F2> C2Fn<T> for Times<F1, F2>
where
    F1: C2Fn<T>,
    F2: C2Fn<T>,
    T: Add<T, Output = T> + Mul<T, Output = T> + FromPrimitive + Copy,
{
    fn value(&self, phi: T) -> T {
        self.f1.value(phi) * self.f2.value(phi)
    }

    fn value_d(&self, phi: T) -> T {
        self.f1.value_d(phi) * self.f2.value(phi) + self.f1.value(phi) * self.f2.value_d(phi)
    }

    fn value_dd(&self, phi: T) -> T {
        self.f1.value_dd(phi) * self.f2.value(phi)
            + self.f1.value(phi) * self.f2.value_dd(phi)
            + T::from_u8(2).unwrap() * self.f1.value_d(phi) * self.f2.value_d(phi)
    }
}
