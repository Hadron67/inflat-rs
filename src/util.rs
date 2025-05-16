use std::{marker::PhantomData, ops::{Add, Div, Mul, Range, Sub}};

use num_traits::{Float, FromPrimitive};
use rayon::iter::{plumbing::{bridge, Consumer, Producer, ProducerCallback, UnindexedConsumer}, IndexedParallelIterator, ParallelIterator};

pub fn derivative_2<T, T2>(dx: &[T], y: &[T2], i: usize) -> T2 where
    T: Copy,
    T2: Copy + Add<T2, Output = T2> + Div<T2, Output = T2> + Mul<T2, Output = T2> + Sub<T2, Output = T2> + FromPrimitive + From<T>,
{
    let s: T2 = dx[i - 1].into();
    let t: T2 = dx[i].into();
    let one = T2::from_usize(1).unwrap();
    (y[i - 1] / s + y[i + 1] / t - (one / s + one / t) * y[i]) / (s + t) * T2::from_usize(2).unwrap()
}

pub fn linear_interp<T>(a: T, b: T, i: T) -> T where
    T: Mul<T, Output = T> + Add<T, Output = T> + Sub<T, Output = T> + Copy,
{
    a + (b - a) * i
}

pub fn power_interp<T>(a: T, b: T, i: T) -> T where
    T: Float,
{
    a * (b / a).powf(i)
}

pub fn limit_length<T: Clone>(arr: Vec<T>, max_length: usize) -> Vec<T> {
    if arr.len() > max_length {
        let mut arr2 = vec![];
        arr2.reserve(max_length);
        for i in 0..max_length {
            arr2.push(arr[((i as f64) / ((max_length - 1) as f64) * ((arr.len() - 1) as f64)) as usize].clone());
        }
        arr2
    } else {
        arr
    }
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

impl<T, M> ParallelIterator for RangeBasedProducer<T, M> where
    T: Send,
    M: IndexMapOp<Item = T> + Sync + Send + Clone,
{
    type Item = T;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result where
        C: UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.range.len())
    }
}

impl<T, M> IndexedParallelIterator for RangeBasedProducer<T, M> where
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

impl<T, M> Producer for RangeBasedProducer<T, M> where
    T: Send,
    M: IndexMapOp<Item = T> + Send + Sync + Clone,
{
    type Item = T;

    type IntoIter = RangeBasedParallelIterator<T, M>;

    fn into_iter(self) -> Self::IntoIter {
        let cursor = self.range.start;
        RangeBasedParallelIterator { producer: self, cursor }
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        (Self {
            range: self.range.start..index,
            mapper: self.mapper.clone(),
            _marker: PhantomData,
        }, Self {
            range: index..self.range.end,
            mapper: self.mapper,
            _marker: PhantomData,
        })
    }
}

pub struct RangeBasedParallelIterator<T, M> {
    producer: RangeBasedProducer<T, M>,
    cursor: usize,
}

impl<T, M> Iterator for RangeBasedParallelIterator<T, M> where
    M: IndexMapOp<Item = T>,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor < self.producer.range.end {
            let ret = self.cursor;
            self.cursor += 1;
            Some(self.producer.mapper.map_index(ret))
        } else { None }
    }
}

impl<T, M> ExactSizeIterator for RangeBasedParallelIterator<T, M> where
    M: IndexMapOp<Item = T>,
{
    fn len(&self) -> usize {
        self.producer.range.len()
    }
}

impl<T, M> DoubleEndedIterator for RangeBasedParallelIterator<T, M> where
    M: IndexMapOp<Item = T>,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.cursor > self.producer.range.start {
            let ret = self.cursor;
            self.cursor += 1;
            Some(self.producer.mapper.map_index(ret))
        } else { None }
    }
}
