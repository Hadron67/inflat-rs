use std::{
    error::Error,
    fmt::Display,
    fs::File,
    io::{self, BufReader, BufWriter},
    marker::PhantomData,
    mem::MaybeUninit,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Range, Rem, Sub},
    slice,
    time::{Duration, SystemTime},
};

use bincode::{
    Decode, Encode,
    config::Config,
    decode_from_std_read, encode_into_std_write,
    error::{DecodeError, EncodeError},
};
use ndarray::{Linspace, linspace};
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
    last_time: SystemTime,
}

impl RateLimiter {
    pub fn new(interval: Duration) -> Self {
        Self {
            interval,
            last_time: SystemTime::now(),
        }
    }
    pub fn run<A>(&mut self, mut action: A)
    where
        A: FnMut(),
    {
        if self
            .last_time
            .elapsed()
            .map(|t| t > self.interval)
            .unwrap_or(false)
        {
            self.last_time = SystemTime::now();
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

pub struct TimeEstimator {
    
}

#[cfg(test)]
mod tests {
    use crate::util::VecN;

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
}
