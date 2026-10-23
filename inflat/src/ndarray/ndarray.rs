use std::{marker::PhantomData, mem::MaybeUninit, ops::{Index, IndexMut}, panic::AssertUnwindSafe};

use ndarray::{ArrayD, Dim};
use rayon::iter::{plumbing::{bridge, Consumer, Producer, ProducerCallback, UnindexedConsumer}, IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

pub trait TransposibleList<T> {
    type RemovedOneAxis: TransposibleList<T>;

    fn len(&self) -> usize;
    fn get(&self, index: usize) -> T;
    fn permute(self, perm: &[u8]) -> Self where Self: Sized;
    fn remove_one_axis(self, index: usize) -> Self::RemovedOneAxis;
}

macro_rules! impl_array_shape_for_array {
    ([$T:expr; $len:expr]) => {
        impl ArrayShapeLike for [$T; $len] {
            type RemovedOneAxis = [$T; $len - 1];

            fn len(&self) -> usize {
                <[T]>::len(self)
            }

            fn get(&self, index: usize) -> T {
                self[index]
            }

            fn permute(self, perm: &[u8]) -> Self {
                permute(self, perm);
            }

            fn remove_one_axis(self, index: usize) -> Self::RemovedOneAxis {
                let mut ret = [0 as T; $len - 1];
                let mut j = 0;
                for i in 0..<[T]>::len(self) {
                    if i != index {
                        ret[j] = self[i];
                        j += 1;
                    }
                }
                ret
            }
        }
    };
}

#[derive(Debug, Clone, Copy)]
pub struct ArrayShape<const N: usize> {
    dim: [usize; N],
    strides: [usize; N],
}

fn permute<T: Copy, const N: usize>(list: [T; N], perm: [u8; N]) -> [T; N] {
    let mut ret: [T; N] = unsafe { MaybeUninit::uninit().assume_init() };
    for i in 0..N {
        ret[perm[i] as usize] = list[i];
    }
    ret
}

impl<const N: usize> ArrayShape<N> {
    pub fn from_dims(dim: [usize; N]) -> Self {
        let mut strides = [0usize; N];
        strides[N - 1] = 1;
        for i in 0..N - 1 {
            let j = N - 1 - i;
            strides[j - 1] = dim[j] * strides[j];
        }
        Self { dim, strides }
    }
    pub fn total_size(&self) -> usize {
        let mut ret = 1;
        for index in self.dim {
            ret *= index;
        }
        ret
    }
    pub fn total_size_except_last(&self) -> usize {
        let mut ret = 1;
        for index in &self.dim[..N - 1] {
            ret *= index;
        }
        ret
    }
    pub fn transpose(&self, perm: [u8; N]) -> Self {
        Self { dim: permute(self.dim, perm), strides: permute(self.dim, perm) }
    }
    pub fn encode_coord(&self, coord: [usize; N]) -> usize {
        self.strides.iter().zip(coord.iter()).map(|(a, b)|a * b).sum::<usize>()
    }
}

pub fn decode_coord<const N: usize>(mut index: usize, dim: [usize; N]) -> [usize; N] {
    let mut ret = [0usize; N];
    for i in 0..N {
        let j = N - i - 1;
        let d = dim[j];
        ret[j] = index % d;
        index /= d;
    }
    ret
}

pub struct ArrayMutView<'data, T, const N: usize> {
    ptr: &'data mut [T],
    shape: ArrayShape<N>,
}

impl<'data, T, const N: usize> ArrayMutView<'data, T, N> {
    pub fn from_dims_ref_mut(ptr: &'data mut [T], dims: [usize; N]) -> Self {
        let shape = ArrayShape::from_dims(dims);
        assert!(shape.total_size() >= ptr.len());
        Self { ptr, shape }
    }
    pub fn transpose(self, perm: [u8; N]) -> Self {
        Self { ptr: self.ptr, shape: self.shape.transpose(perm) }
    }
    pub fn shape(&self) -> ArrayShape<N> {
        self.shape
    }
    pub fn par_iter_other_axis<'arr>(&'arr mut self, axis: u8) -> ArrayMutViewAxisParIter<'data, 'arr, T, N> {
        ArrayMutViewAxisParIter { array: self, axis }
    }
}

pub struct ArrayMutViewAxisParIter<'data, 'arr, T, const N: usize> {
    array: &'arr mut ArrayMutView<'data, T, N>,
    axis: u8,
}

impl<'data, 'arr, T, const N: usize> ParallelIterator for ArrayMutViewAxisParIter<'data, 'arr, T, N> where
    T: Sync + Send,
{
    type Item = RefSliceMut<'data, T>;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result where
        C: UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.len())
    }
}

impl<'data, 'arr, T, const N: usize> IndexedParallelIterator for ArrayMutViewAxisParIter<'data, 'arr, T, N> where
    T: Sync + Send,
{
    fn len(&self) -> usize {
        self.array.shape.total_size_except_last()
    }

    fn drive<C: Consumer<Self::Item>>(self, consumer: C) -> C::Result {
        bridge(self, consumer)
    }

    fn with_producer<CB: ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
        let mut perm = [0u8; N];
        for i in 0..N {
            perm[i] = i as u8;
        }
        perm[self.axis as usize] = (N - 1) as u8;
        perm[N - 1] = self.axis;
        callback.callback(ArrayMutViewSliceIterProducer {
            data: self.array.ptr as *mut [T] as *mut _,
            shape: self.array.shape.transpose(perm),
            start: 0,
            end: self.len(),
            _data: PhantomData,
        })
    }
}

pub struct ArrayMutViewSliceIterProducer<'data, T, const N: usize> {
    data: *mut T,
    _data: PhantomData<&'data mut [T]>,
    shape: ArrayShape<N>,
    start: usize,
    end: usize,
}

unsafe impl<'data, T, const N: usize> Send for ArrayMutViewSliceIterProducer<'data, T, N> where T: Send + Sync {}

impl<'data, T, const N: usize> ArrayMutViewSliceIterProducer<'data, T, N> {
    pub fn get_at(&mut self, index: usize) -> RefSliceMut<'data, T> {
        assert!(index >= self.start && index < self.end);
        let coord = decode_coord(index + self.start, self.shape.dim);
        let offset = coord[..N - 1]
            .iter()
            .zip(self.shape.strides[..N - 1].iter())
            .map(|(a, b)|a * b)
            .sum::<usize>();
        RefSliceMut {
            ptr: unsafe { self.data.add(offset) },
            length: self.shape.dim[N - 1],
            stride: self.shape.strides[N - 1],
            _marker: PhantomData,
        }
    }
    pub fn len(&self) -> usize { self.end - self.start }
}

pub struct ArrayMutViewSliceIter<'data, T, const N: usize> {
    producer: ArrayMutViewSliceIterProducer<'data, T, N>,
    cursor: usize,
}

impl<'data, T, const N: usize> Producer for ArrayMutViewSliceIterProducer<'data, T, N> where
    T: Send + Sync,
{
    type Item = <<Self as Producer>::IntoIter as Iterator>::Item;

    type IntoIter = ArrayMutViewSliceIter<'data, T, N>;

    fn into_iter(self) -> Self::IntoIter {
        let start = self.start;
        ArrayMutViewSliceIter {
            producer: self,
            cursor: start,
        }
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        (Self {
            data: self.data,
            shape: self.shape,
            start: self.start,
            end: index,
            _data: PhantomData,
        }, Self {
            data: self.data,
            shape: self.shape,
            start: index,
            end: self.end,
            _data: PhantomData,
        })
    }
}

impl<'data, T, const N: usize> Iterator for ArrayMutViewSliceIter<'data, T, N> {
    type Item = RefSliceMut<'data, T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor < self.producer.end {
            let ret = self.producer.get_at(self.cursor);
            self.cursor += 1;
            Some(ret)
        } else {
            None
        }
    }
}

impl<'data, T, const N: usize> ExactSizeIterator for ArrayMutViewSliceIter<'data, T, N> {
    fn len(&self) -> usize {
        self.producer.len()
    }
}

impl<'data, T, const N: usize> DoubleEndedIterator for ArrayMutViewSliceIter<'data, T, N> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.cursor > self.producer.start {
            let ret = self.producer.get_at(self.cursor);
            self.cursor -= 1;
            Some(ret)
        } else { None }
    }
}

pub struct RefSliceMut<'data, T> {
    ptr: *mut T,
    length: usize,
    stride: usize,
    _marker: PhantomData<&'data mut T>,
}

unsafe impl<'data, T> Send for RefSliceMut<'data, T> where T: Send + Sync {}

impl<'data, T> RefSliceMut<'data, T> {
    pub fn len(&self) -> usize { self.length }
    pub fn map_mut_copied_indexed<F>(&mut self, mut mapper: F) where
        F: FnMut(T, usize) -> T,
        T: Copy,
    {
        for index in 0..self.length {
            self[index] = mapper(self[index], index);
        }
    }
}

impl<'data, T> Index<usize> for RefSliceMut<'data, T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        assert!(index < self.length);
        unsafe { &*self.ptr.add(index * self.stride) }
    }
}

impl<'data, T> IndexMut<usize> for RefSliceMut<'data, T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        assert!(index < self.length);
        unsafe { &mut *self.ptr.add(index * self.stride) }
    }
}

#[cfg(test)]
mod tests {
    use rayon::iter::{IndexedParallelIterator, ParallelIterator};

    use super::ArrayMutView;

    #[test]
    fn test_iter() {
        let mut arr = unsafe { Box::new_uninit_slice(512).assume_init() };
        for i in 0..arr.len() {
            arr[i] = i as i32;
        }
        let mut view = ArrayMutView::from_dims_ref_mut(&mut arr, [8, 8, 8]);
        let shape = view.shape();
        view.par_iter_other_axis(1).zip(0..64).for_each(|(mut slice, index)|{
            slice.map_mut_copied_indexed(|_, i|i as i32);
        });
        for x in 0..8 {
            for z in 0..8 {
                for y in 0..8 {
                    let index = shape.encode_coord([x, y, z]);
                    assert_eq!(arr[index], y as i32);
                }
            }
        }
    }
}