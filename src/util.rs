use std::{
    error::Error,
    fmt::Display,
    fs::File,
    io::{self, BufReader, BufWriter},
    marker::PhantomData,
    mem::MaybeUninit,
    ops::{Add, Div, Index, IndexMut, Mul, Range, Sub},
    process::Output,
    time::{Duration, SystemTime},
};

use bincode::{
    Decode, Encode,
    config::Config,
    decode_from_std_read, encode_into_std_write,
    error::{DecodeError, EncodeError},
};
use num_traits::{Float, FromPrimitive, Zero};
use rayon::iter::{
    IndexedParallelIterator, ParallelIterator,
    plumbing::{Consumer, Producer, ProducerCallback, UnindexedConsumer, bridge},
};

use crate::c2fn::Plus;

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

pub fn power_interp<T>(a: T, b: T, i: T) -> T
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

pub struct VecN<const N: usize, T> {
    value: [T; N],
}

impl<const N: usize, T> VecN<N, T> {
    pub fn new(value: [T; N]) -> Self {
        Self { value }
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
