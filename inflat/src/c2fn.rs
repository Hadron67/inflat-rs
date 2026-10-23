use std::{
    marker::PhantomData,
    ops::{Add, Mul},
};

use num_traits::{FromPrimitive, Zero};

pub trait C2Fn<T> {
    type Output;
    fn value(&self, phi: T) -> Self::Output;
    fn value_d(&self, phi: T) -> Self::Output;
    fn value_dd(&self, phi: T) -> Self::Output;

    fn plus<F>(self, other: F) -> Plus<Self, F, Self::Output>
    where
        Self: Sized,
        F: C2Fn<T, Output = Self::Output>,
    {
        Plus {
            f1: self,
            f2: other,
            _output: PhantomData,
        }
    }
    fn plus2<T2, F>(self, other: F) -> Plus2<Self, F, Self::Output>
    where
        Self: Sized,
        F: C2Fn<T2, Output = Self::Output>,
    {
        Plus2 {
            lhs: self,
            rhs: other,
            _output: PhantomData,
        }
    }
    fn mul<F>(self, other: F) -> Times<Self, F, Self::Output>
    where
        Self: Sized,
        F: C2Fn<T, Output = Self::Output>,
    {
        Times {
            f1: self,
            f2: other,
            _output: PhantomData,
        }
    }
}

impl<T, F> C2Fn<T> for &F
where
    F: C2Fn<T>,
{
    type Output = F::Output;

    fn value(&self, phi: T) -> Self::Output {
        (*self).value(phi)
    }

    fn value_d(&self, phi: T) -> Self::Output {
        (*self).value_d(phi)
    }

    fn value_dd(&self, phi: T) -> Self::Output {
        (*self).value_dd(phi)
    }
}

pub trait C2Fn2<X, Y> {
    type Ret;
    fn value_00(&self, phi1: X, phi2: Y) -> Self::Ret;
    fn value_10(&self, phi1: X, phi2: Y) -> Self::Ret;
    fn value_01(&self, phi1: X, phi2: Y) -> Self::Ret;
    fn value_11(&self, phi1: X, phi2: Y) -> Self::Ret;
    fn value_20(&self, phi1: X, phi2: Y) -> Self::Ret;
    fn value_02(&self, phi1: X, phi2: Y) -> Self::Ret;
}

// impl<X, Y, F> C2Fn2<X, Y> for &F where
//     F: C2Fn2<X, Y>,
// {
//     type Ret = F::Ret;

//     fn value_00(&self, phi1: X, phi2: Y) -> Self::Ret {
//         (*self).value_00(phi1, phi2)
//     }

//     fn value_10(&self, phi1: X, phi2: Y) -> Self::Ret {
//         (*self).value_10(phi1, phi2)
//     }

//     fn value_01(&self, phi1: X, phi2: Y) -> Self::Ret {
//         (*self).value_01(phi1, phi2)
//     }

//     fn value_11(&self, phi1: X, phi2: Y) -> Self::Ret {
//         (*self).value_11(phi1, phi2)
//     }

//     fn value_20(&self, phi1: X, phi2: Y) -> Self::Ret {
//         (*self).value_20(phi1, phi2)
//     }

//     fn value_02(&self, phi1: X, phi2: Y) -> Self::Ret {
//         (*self).value_02(phi1, phi2)
//     }
// }

pub struct Plus<F1, F2, Output> {
    pub f1: F1,
    pub f2: F2,
    _output: PhantomData<Output>,
}

impl<T, F1, F2, Output> C2Fn<T> for Plus<F1, F2, Output>
where
    F1: C2Fn<T, Output = Output>,
    F2: C2Fn<T, Output = Output>,
    T: Copy,
    Output: Add<Output, Output = Output>,
{
    type Output = Output;
    fn value(&self, phi: T) -> Self::Output {
        self.f1.value(phi) + self.f2.value(phi)
    }

    fn value_d(&self, phi: T) -> Self::Output {
        self.f1.value_d(phi) + self.f2.value_d(phi)
    }

    fn value_dd(&self, phi: T) -> Self::Output {
        self.f1.value_dd(phi) + self.f2.value_dd(phi)
    }
}

pub struct Times<F1, F2, Output> {
    pub f1: F1,
    pub f2: F2,
    _output: PhantomData<Output>,
}

impl<T, F1, F2, Output> C2Fn<T> for Times<F1, F2, Output>
where
    F1: C2Fn<T, Output = Output>,
    F2: C2Fn<T, Output = Output>,
    T: Copy,
    Output: Add<Output, Output = Output> + Mul<Output, Output = Output> + FromPrimitive,
{
    type Output = Output;

    fn value(&self, phi: T) -> Self::Output {
        self.f1.value(phi) * self.f2.value(phi)
    }

    fn value_d(&self, phi: T) -> Self::Output {
        self.f1.value_d(phi) * self.f2.value(phi) + self.f1.value(phi) * self.f2.value_d(phi)
    }

    fn value_dd(&self, phi: T) -> Self::Output {
        self.f1.value_dd(phi) * self.f2.value(phi)
            + self.f1.value(phi) * self.f2.value_dd(phi)
            + Output::from_u8(2).unwrap() * self.f1.value_d(phi) * self.f2.value_d(phi)
    }
}

pub struct Plus2<F1, F2, Output> {
    pub lhs: F1,
    pub rhs: F2,
    _output: PhantomData<Output>,
}

impl<T1, T2, F1, F2, Output> C2Fn2<T1, T2> for Plus2<F1, F2, Output>
where
    F1: C2Fn<T1, Output = Output>,
    F2: C2Fn<T2, Output = Output>,
    Output: Add<Output, Output = Output> + Zero,
{
    type Ret = Output;

    fn value_00(&self, phi1: T1, phi2: T2) -> Self::Ret {
        self.lhs.value(phi1) + self.rhs.value(phi2)
    }

    fn value_10(&self, phi1: T1, _phi2: T2) -> Self::Ret {
        self.lhs.value_d(phi1)
    }

    fn value_01(&self, _phi1: T1, phi2: T2) -> Self::Ret {
        self.rhs.value_d(phi2)
    }

    fn value_11(&self, _phi1: T1, _phi2: T2) -> Self::Ret {
        Output::zero()
    }

    fn value_20(&self, phi1: T1, _phi2: T2) -> Self::Ret {
        self.lhs.value_dd(phi1)
    }

    fn value_02(&self, _phi1: T1, phi2: T2) -> Self::Ret {
        self.rhs.value_dd(phi2)
    }
}
