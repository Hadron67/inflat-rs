use std::{marker::PhantomData, ops::Range};

use rayon::iter::{plumbing::{bridge, Consumer, Producer, ProducerCallback, UnindexedConsumer}, IndexedParallelIterator, ParallelIterator};

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
