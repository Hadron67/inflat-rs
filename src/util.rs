use std::{
    collections::VecDeque, error::Error, f64::consts::PI, fmt::Display, fs::File, io::{self, BufReader, BufWriter}, iter::zip, marker::PhantomData, mem::MaybeUninit, ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Range, Rem, Sub}, slice, time::{Duration, SystemTime}
};

use bincode::{
    Decode, Encode,
    config::Config,
    decode_from_std_read, encode_into_std_write,
    error::{DecodeError, EncodeError},
};
use libm::sqrt;
use ndarray::{Linspace, linspace};
use num_complex::Complex64;
use num_traits::{Float, FromPrimitive, One, Zero};
use rayon::iter::{
    IndexedParallelIterator, ParallelIterator,
    plumbing::{Consumer, Producer, ProducerCallback, UnindexedConsumer, bridge},
};

pub const ENERGY_SPECTRUM_EVAL_FACTOR: f64 = 6.8e-7;

pub fn derivative_2<T, T2>(dx1: T, dx2: T, y1: T2, y2: T2, y3: T2) -> T2
where
    T: Copy,
    T2: Copy
        + Add<T2, Output = T2>
        + Div<T2, Output = T2>
        + Mul<T2, Output = T2>
        + Sub<T2, Output = T2>
        + FromPrimitive
        + From<T>,
{
    let s: T2 = dx1.into();
    let t: T2 = dx2.into();
    let one = T2::from_usize(1).unwrap();
    (y1 / s + y3 / t - (one / s + one / t) * y2) / (s + t) * T2::from_usize(2).unwrap()
}

pub fn linear_interp<T>(a: T, b: T, i: T) -> T
where
    T: Mul<T, Output = T> + Add<T, Output = T> + Sub<T, Output = T> + Copy,
{
    a + (b - a) * i
}

pub fn log_interp<T>(a: T, b: T, i: T) -> T
where
    T: Float,
{
    a * (b / a).powf(i)
}

pub fn limit_length<T: Clone>(arr: Vec<T>, max_length: usize) -> Vec<T> {
    if arr.len() > max_length {
        let mut arr2 = vec![];
        arr2.reserve(max_length);
        for i in 0..max_length {
            arr2.push(
                arr[((i as f64) / ((max_length - 1) as f64) * ((arr.len() - 1) as f64)) as usize]
                    .clone(),
            );
        }
        arr2
    } else {
        arr
    }
}

/// Computes Gamma(n2 / 2)
pub fn half_int_gamma(mut n2: u32) -> f64 {
    let mut ret = 1.0;
    while n2 >= 2 {
        ret *= ((n2 - 2) as f64) / 2.0;
        n2 -= 2;
    }
    if n2 == 1 {
        ret *= PI.sqrt();
    }
    ret
}

#[derive(Debug)]
pub enum EncodeOrIoError {
    Encode(EncodeError),
    Decode(DecodeError),
    Io(io::Error),
}

impl From<EncodeError> for EncodeOrIoError {
    fn from(value: EncodeError) -> Self {
        Self::Encode(value)
    }
}

impl From<io::Error> for EncodeOrIoError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<DecodeError> for EncodeOrIoError {
    fn from(value: DecodeError) -> Self {
        Self::Decode(value)
    }
}

impl Display for EncodeOrIoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Encode(err) => err.fmt(f),
            Self::Decode(err) => err.fmt(f),
            Self::Io(err) => err.fmt(f),
        }
    }
}

impl Error for EncodeOrIoError {}

pub type Result<T> = std::result::Result<T, EncodeOrIoError>;

pub fn lazy_file<T, F, C>(name: &str, config: C, mut creator: F) -> Result<T>
where
    T: Encode + Decode<()>,
    F: FnMut() -> T,
    C: Config,
{
    if let Ok(file) = File::open(name) {
        Ok(decode_from_std_read(&mut BufReader::new(file), config)?)
    } else {
        let value = creator();
        let file = File::create(name).unwrap();
        encode_into_std_write(&value, &mut BufWriter::new(file), config)?;
        Ok(value)
    }
}

pub fn sigmoid<T>(x: T) -> T
where
    T: Float,
{
    T::one() / (T::one() + x.neg().exp())
}

pub trait IndexMapOp {
    type Item;

    fn map_index(&mut self, index: usize) -> Self::Item;
}

pub struct RangeBasedProducer<T, M> {
    range: Range<usize>,
    mapper: M,
    _marker: PhantomData<T>,
}

impl<T, M> ParallelIterator for RangeBasedProducer<T, M>
where
    T: Send,
    M: IndexMapOp<Item = T> + Sync + Send + Clone,
{
    type Item = T;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.range.len())
    }
}

impl<T, M> IndexedParallelIterator for RangeBasedProducer<T, M>
where
    T: Send,
    M: IndexMapOp<Item = T> + Send + Sync + Clone,
{
    fn len(&self) -> usize {
        self.range.len()
    }

    fn drive<C: Consumer<Self::Item>>(self, consumer: C) -> C::Result {
        bridge(self, consumer)
    }

    fn with_producer<CB: ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
        callback.callback(self)
    }
}

impl<T, M> Producer for RangeBasedProducer<T, M>
where
    T: Send,
    M: IndexMapOp<Item = T> + Send + Sync + Clone,
{
    type Item = T;

    type IntoIter = RangeBasedParallelIterator<T, M>;

    fn into_iter(self) -> Self::IntoIter {
        let cursor = self.range.start;
        RangeBasedParallelIterator {
            producer: self,
            cursor,
        }
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        (
            Self {
                range: self.range.start..index,
                mapper: self.mapper.clone(),
                _marker: PhantomData,
            },
            Self {
                range: index..self.range.end,
                mapper: self.mapper,
                _marker: PhantomData,
            },
        )
    }
}

pub struct RangeBasedParallelIterator<T, M> {
    producer: RangeBasedProducer<T, M>,
    cursor: usize,
}

impl<T, M> Iterator for RangeBasedParallelIterator<T, M>
where
    M: IndexMapOp<Item = T>,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor < self.producer.range.end {
            let ret = self.cursor;
            self.cursor += 1;
            Some(self.producer.mapper.map_index(ret))
        } else {
            None
        }
    }
}

impl<T, M> ExactSizeIterator for RangeBasedParallelIterator<T, M>
where
    M: IndexMapOp<Item = T>,
{
    fn len(&self) -> usize {
        self.producer.range.len()
    }
}

impl<T, M> DoubleEndedIterator for RangeBasedParallelIterator<T, M>
where
    M: IndexMapOp<Item = T>,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.cursor > self.producer.range.start {
            let ret = self.cursor;
            self.cursor += 1;
            Some(self.producer.mapper.map_index(ret))
        } else {
            None
        }
    }
}

pub struct RateLimiter {
    interval: Duration,
    last_time: Option<SystemTime>,
}

impl RateLimiter {
    pub fn new(interval: Duration) -> Self {
        Self {
            interval,
            last_time: None,
        }
    }
    pub fn run<A>(&mut self, mut action: A)
    where
        A: FnMut(),
    {
        if self
            .last_time
            .map(|last_time| {
                last_time
                    .elapsed()
                    .map(|i| i > self.interval)
                    .unwrap_or(false)
            })
            .unwrap_or(true)
        {
            self.last_time = Some(SystemTime::now());
            action();
        }
    }
}

#[derive(Clone, Copy, Encode, Decode, Debug, Hash, PartialEq, Eq)]
pub struct VecN<const N: usize, T> {
    pub value: [T; N],
}

impl<const N: usize, T> Display for VecN<N, T>
where
    T: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("({}", &self.value[0]))?;
        for i in 1..N {
            f.write_str(", ")?;
            self.value[i].fmt(f)?;
        }
        f.write_str(")")?;
        Ok(())
    }
}

impl<const N: usize, T> VecN<N, T> {
    pub fn new(value: [T; N]) -> Self {
        Self { value }
    }
    pub fn zeros() -> Self
    where
        T: Zero + Copy,
    {
        Self {
            value: [T::zero(); N],
        }
    }
    pub fn strides(&self) -> Self
    where
        T: One + Copy,
    {
        let mut ret = [T::one(); N];
        for i in 0..N - 1 {
            ret[N - 2 - i] = ret[N - 1 - i] * self[N - 1 - i];
        }
        Self::new(ret)
    }
    pub fn decode_coord(&self, mut index: T) -> Self
    where
        T: DivAssign<T> + Rem<T, Output = T> + Copy,
    {
        let mut ret = MaybeUninit::<Self>::uninit();
        let ptr = unsafe { &raw mut (*ret.as_mut_ptr()).value };
        for i in 0..N {
            let j = N - 1 - i;
            let p = unsafe { &raw mut (*ptr)[j] };
            unsafe { p.write(index % self[j]) };
            index /= self[j];
        }
        unsafe { ret.assume_init() }
    }
    pub fn encode_coord(&self, coord: &Self) -> T
    where
        T: One + Copy + AddAssign<T> + Mul<T, Output = T>,
    {
        self.strides().inner(coord)
    }
    pub fn product(&self) -> T
    where
        T: MulAssign<T> + Clone,
    {
        let mut ret = self.value[0].clone();
        for val in &self.value[1..] {
            ret *= val.clone().into();
        }
        ret
    }
    pub fn sum(&self) -> T
    where
        T: AddAssign<T> + Copy,
    {
        let mut ret = self.value[0];
        for val in &self.value[1..] {
            ret += *val;
        }
        ret
    }
    pub fn inner(&self, other: &Self) -> T
    where
        T: AddAssign<T> + Mul<T, Output = T> + Copy,
    {
        let mut ret = self[0] * other[0];
        for i in 1..N {
            ret += self[i] * other[i];
        }
        ret
    }
    pub fn map<F, T2>(&self, mut mapper: F) -> VecN<N, T2>
    where
        F: FnMut(&T) -> T2,
    {
        let mut ret = MaybeUninit::<VecN<N, T2>>::uninit();
        let ptr = unsafe { &raw mut (*ret.as_mut_ptr()).value };
        for i in 0..N {
            let p = unsafe { &raw mut (*ptr)[i] };
            unsafe { p.write(mapper(&self.value[i])) };
        }
        unsafe { ret.assume_init() }
    }
    pub fn map_at<F>(&self, index: usize, mut mapper: F) -> Self
    where
        F: FnMut(T) -> T,
        T: Clone,
    {
        let mut ret = self.clone();
        ret[index] = mapper(ret[index].clone());
        ret
    }
    pub fn flip(&self, size: &Self) -> Self
    where
        T: Zero + One + PartialOrd<T> + Sub<T, Output = T> + Copy,
    {
        let mut i = 0usize;
        self.map(|f| {
            let f = *f;
            let s = size[i];
            i += 1;
            assert!(f >= T::zero() && f < s);
            if f == T::zero() { T::zero() } else { s - f }
        })
    }
}

impl<'a, const N: usize, T: 'a> IntoIterator for &'a VecN<N, T> {
    type Item = &'a T;

    type IntoIter = slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.value.iter()
    }
}

impl<const N: usize, T> Index<usize> for VecN<N, T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.value[index]
    }
}

impl<const N: usize, T> IndexMut<usize> for VecN<N, T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.value[index]
    }
}

impl<const N: usize, T> Add<VecN<N, T>> for VecN<N, T>
where
    T: Add<T, Output = T> + Zero + Copy,
{
    type Output = Self;

    fn add(self, rhs: VecN<N, T>) -> Self::Output {
        let mut ret = VecN {
            value: [T::zero(); N],
        };
        for i in 0..N {
            ret.value[i] = self.value[i] + rhs.value[i];
        }
        ret
    }
}

impl<const N: usize, T> Mul<T> for VecN<N, T>
where
    T: Mul<T, Output = T> + Zero + Copy,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        let mut ret = VecN {
            value: [T::zero(); N],
        };
        for i in 0..N {
            ret.value[i] = self.value[i] * rhs;
        }
        ret
    }
}

#[derive(Clone, Copy)]
pub struct ParamRange<T> {
    pub start: T,
    pub end: T,
    pub count: usize,
}

impl<T> ParamRange<T> {
    pub fn new(start: T, end: T, count: usize) -> Self {
        Self { start, end, count }
    }
}

impl<T> ParamRange<T>
where
    T: Float,
{
    pub fn as_linspace(&self) -> Linspace<T> {
        linspace(self.start, self.end, self.count)
    }
    pub fn as_logspace(&self) -> LogspaceIter<T> {
        LogspaceIter {
            start: self.start,
            end: self.end,
            count: self.count,
            index: 0,
        }
    }
    pub fn linear_interp(&self, i: usize) -> T {
        assert!(i < self.count);
        linear_interp(
            self.start,
            self.end,
            T::from(i).unwrap() / T::from(self.count - 1).unwrap(),
        )
    }
    pub fn log_interp(&self, i: usize) -> T {
        assert!(i < self.count);
        log_interp(
            self.start,
            self.end,
            T::from(i).unwrap() / T::from(self.count - 1).unwrap(),
        )
    }
}

impl<T> Mul<T> for ParamRange<T>
where
    T: Mul<T, Output = T> + Copy,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        Self {
            start: self.start * rhs,
            end: self.end * rhs,
            count: self.count,
        }
    }
}

impl<T> Div<T> for ParamRange<T>
where
    T: Div<T, Output = T> + Copy,
{
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        Self {
            start: self.start / rhs,
            end: self.end / rhs,
            count: self.count,
        }
    }
}

pub struct LogspaceIter<T> {
    pub start: T,
    pub end: T,
    pub count: usize,
    index: usize,
}

impl<T> Iterator for LogspaceIter<T>
where
    T: Float,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.count {
            let p = T::from(self.index).unwrap() / T::from(self.count - 1).unwrap();
            self.index += 1;
            Some(log_interp(self.start, self.end, p))
        } else {
            None
        }
    }
}

pub struct TimeEstimator {
    pub range: Range<f64>,
    current: f64,
    last_time: SystemTime,
    speeds: VecDeque<f64>,
}

impl TimeEstimator {
    pub fn new(range: Range<f64>, speeds: usize) -> Self {
        Self {
            current: range.start,
            range,
            last_time: SystemTime::now(),
            speeds: VecDeque::with_capacity(speeds),
        }
    }
    pub fn update(&mut self, value: f64) {
        if let Ok(time) = self.last_time.elapsed() {
            self.last_time = SystemTime::now();
            let speed = (value - self.current) / time.as_secs_f64();
            while self.speeds.len() >= self.speeds.capacity() {
                self.speeds.pop_front();
            }
            self.speeds.push_back(speed);
        }
        self.current = value;
    }
    pub fn get_speed(&self) -> f64 {
        self.speeds.iter().sum::<f64>() / (self.speeds.len() as f64)
    }
    pub fn remaining_secs(&self) -> u64 {
        let secs = (self.range.end - self.current) / self.get_speed();
        if secs < 0.0 { 0 } else { secs as u64 }
    }
}

pub fn first_index_of<T, I, Idx, F>(
    arr: &I,
    search_range: Range<Idx>,
    mut predicate: F,
) -> Option<<Range<Idx> as IntoIterator>::Item>
where
    T: ?Sized,
    Range<Idx>: IntoIterator,
    <Range<Idx> as IntoIterator>::Item: Copy,
    I: ?Sized + Index<<Range<Idx> as IntoIterator>::Item, Output = T>,
    F: FnMut(&T) -> bool,
{
    for i in search_range {
        if predicate(&arr[i]) {
            return Some(i);
        }
    }
    None
}

pub struct Hms {
    pub hour: u64,
    pub min: u8,
    pub sec: u8,
}

impl Hms {
    pub fn from_secs(mut d: u64) -> Self {
        let sec = (d % 60) as u8;
        d /= 60;
        let min = (d % 60) as u8;
        d /= 60;
        Self { hour: d, min, sec }
    }
}

impl Display for Hms {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "{}:{:0>2}:{:0>2}",
            self.hour, self.min, self.sec
        ))
    }
}

pub fn cubic_discriminal(a: f64, b: f64, c: f64, d: f64) -> f64 {
    b * b * c * c
        + -4.0 * a * c * c * c
        + -4.0 * b * b * b * d
        + 18.0 * a * b * c * d
        + -27.0 * a * a * d * d
}

pub fn solve_cubic(a: Complex64, b: Complex64, c: Complex64, d: Complex64) -> [Complex64; 3] {
    let delta0 = b * b - a * c * 3.0;
    let delta1 = 2.0 * b * b * b - 9.0 * a * b * c + 27.0 * a * a * d;
    let c1 = ((delta1 + (delta1 * delta1 - delta0 * delta0 * delta0 * 4.0).sqrt()) / 2.0).cbrt();
    let c2 = Complex64::new(-0.5, sqrt(3.0) / 2.0) * c1;
    let c3 = Complex64::new(-0.5, -sqrt(3.0) / 2.0) * c1;

    let s1 = b + c1 + delta0 / c1;
    let s2 = b + c2 + delta0 / c2;
    let s3 = b + c3 + delta0 / c3;
    [s1 / (-3.0 * a), s2 / (-3.0 * a), s3 / (-3.0 * a)]
}

pub fn solve_cubic2(a: Complex64, b: Complex64, c: Complex64, d: Complex64) -> [Complex64; 3] {
    let q = (3.0 * a * c - b * b) / 9.0 / a / a;
    let r = (9.0 * a * b * c - 27.0 * a * a * d - 2.0 * b * b * b) / 54.0 / a / a / a;
    let q3 = q * q * q;
    let r2 = r * r;
    let s = (r + (q3 + r2).sqrt()).cbrt();
    let t = (r - (q3 + r2).sqrt()).cbrt();
    let omega1 = Complex64::new(-0.5, sqrt(0.75));
    [
        s + t - b / 3.0 / a,
        s * omega1 + t * omega1.conj() - b / 3.0 / a,
        s * omega1.conj() + t * omega1 - b / 3.0 / a,
    ]
}

pub fn solve_cubic_one_real(a: Complex64, b: Complex64, c: Complex64, d: Complex64) -> Option<f64> {
    let sols = solve_cubic(a, b, c, d);
    sols.iter()
        .cloned()
        .find(|a| (a.im / a.re).abs() <= 1e-20)
        .map(|f| f.re)
}

pub fn evaluate_polynomial<T>(x: T, coefs: &[T]) -> T where
    T: Clone + Mul<T, Output = T> + AddAssign<T> + MulAssign<T>,
{
    let mut ret = coefs[0].clone();
    let mut x_acc = x.clone();
    for c in &coefs[1..] {
        ret += x_acc.clone() * c.clone();
        x_acc *= x.clone();
    }
    ret
}

pub fn evaluate_polynomial_derivative<T>(x: T, coefs: &[T]) -> T where
    T: Clone + Mul<T, Output = T> + AddAssign<T> + MulAssign<T> + FromPrimitive,
{
    let mut ret = coefs[1].clone();
    let mut x_acc = x.clone();
    for (c, i) in zip(&coefs[2..], 2..) {
        ret += x_acc.clone() * c.clone() * T::from_i32(i).unwrap();
        x_acc *= x.clone();
    }
    ret
}

pub fn newton_solve_polynomial<T>(mut initial: T, coefs: &[T], tolerance: T) -> T where
    T: Clone + Mul<T, Output = T> + AddAssign<T> + MulAssign<T> + FromPrimitive + Div<T, Output = T> + Sub<T, Output = T> + Float,
{
    let mut f = evaluate_polynomial(initial.clone(), coefs);
    while f.abs() > tolerance {
        initial = initial.clone() - f / evaluate_polynomial_derivative(initial.clone(), coefs);
        f = evaluate_polynomial(initial.clone(), coefs);
    }
    initial
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use num_complex::ComplexFloat;

    use crate::util::{evaluate_polynomial, evaluate_polynomial_derivative, half_int_gamma, newton_solve_polynomial, VecN};

    #[test]
    fn coords() {
        let dim = VecN::new([10, 11, 12]);
        let index = 456;
        let coord = dim.decode_coord(index);
        assert_eq!(index, dim.encode_coord(&coord));
    }

    #[test]
    fn flip() {
        let dim = VecN::new([4; 3]);
        let coord = VecN::new([0, 1, 2]);
        let flipped = coord.flip(&dim);
        assert_eq!(flipped[0], 0);
        assert_eq!(flipped[1], 3);
        assert_eq!(flipped[2], 2);
    }

    #[test]
    fn mom_factor() {
        assert!((half_int_gamma(3) - PI.sqrt() / 2.0).abs() <= 1e-20);
    }

    #[test]
    fn cubic_eqn() {
        let coefs = [0.0000000014625333905062336, 0.0, -3.0, 0.000015611414900550057];
        let sol = newton_solve_polynomial(0.1, &coefs, 1e-20);
        println!("sol = {}", sol);
        assert!((sol / 0.0000220797 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn polynomial() {
        let coefs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let x = 3.0;
        assert_eq!(evaluate_polynomial(x, &coefs), coefs[0] + coefs[1] * x + coefs[2] * x * x + coefs[3] * x * x * x + coefs[4] * x * x * x * x);
        assert_eq!(evaluate_polynomial_derivative(x, &coefs), coefs[1] + coefs[2] * x * 2.0 + coefs[3] * x * x * 3.0 + coefs[4] * x * x * x * 4.0);
    }
}
