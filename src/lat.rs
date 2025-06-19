use std::{
    cmp::min, fmt::Display, io::{self, Write}, iter::Sum, marker::PhantomData, ops::{Add, AddAssign, Div, Index, IndexMut, Mul, MulAssign, Range}, ptr::NonNull
};

use bincode::{Decode, Encode};
use num_traits::{FromPrimitive, Zero};
use rayon::iter::{
    plumbing::{bridge, Producer}, IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator
};

use crate::util::VecN;

pub struct LatticePos<const D: usize> {
    pub index: usize,
    pub coord: VecN<D, usize>,
}

pub trait Lattice<const D: usize, T> {
    fn dim(&self) -> &VecN<D, usize>;
    fn get(&self, index: usize, coord: &VecN<D, usize>) -> T;
    fn dump<R, W>(&self, tag: &R, write: &mut W) -> io::Result<()> where
        W: ?Sized + Write,
        R: ?Sized + Display,
        T: Display,
    {
        let dim = *self.dim();
        for index in 0..dim.product() {
            let coord = dim.decode_coord(index);
            writeln!(write, "{}[{}] = {}", tag, &coord, self.get(index, &coord))?;
        }
        Ok(())
    }
    fn total_size(&self) -> usize {
        self.dim().product()
    }
    fn plus<F>(self, other: F) -> LatticePlus<Self, F>
    where
        Self: Sized,
    {
        LatticePlus {
            lhs: self,
            rhs: other,
        }
    }
    fn map<M, T2>(self, mapper: M) -> LatticeMap<T, Self, M>
    where
        Self: Sized,
        M: Fn(T) -> T2,
    {
        LatticeMap {
            field: self,
            mapper,
            _src: PhantomData,
        }
    }
    fn flip(self) -> LatticeFlip<D, Self> where  Self: Sized {
        LatticeFlip { field: self }
    }
    fn mul_scalar(self, factor: T) -> LatticeScalarMul<D, T, Self> where
        Self: Sized,
    {
        LatticeScalarMul { factor, field: self }
    }
    fn sum(&self) -> T where
        T: FromPrimitive + MulAssign<T> + Send + Sum,
        Self: Sync,
    {
        let dim = self.dim();
        (0..dim.product()).into_par_iter().map(|index|self.get(index, &dim.decode_coord(index))).sum::<T>()
    }
    fn average(&self) -> T where
        T: FromPrimitive + MulAssign<T> + Div<T, Output = T> + Send + Sum,
        Self: Sync,
    {
        self.sum() / T::from_usize(self.dim().product()).unwrap()
    }
}

pub trait LatticeMut<const D: usize, T> {
    fn get_mut(&mut self, index: usize, coord: &VecN<D, usize>) -> &mut T;
    fn for_each<F>(&mut self, mut op: F) where
        F: FnMut(&mut T, usize, &VecN<D, usize>),
        Self: Lattice<D, T>,
    {
        let dim = *self.dim();
        for index in 0..dim.product() {
            let coord = dim.decode_coord(index);
            op(self.get_mut(index, &coord), index, &coord);
        }
    }
    fn par_for_each_mut<F>(&mut self, mapper: F)
    where
        F: Fn(&mut T, usize, &VecN<D, usize>) + Send + Sync,
        T: Send + Sync;
    fn par_add_assign<F>(&mut self, other: &F)
    where
        F: Lattice<D, T> + Send + Sync,
        T: AddAssign<T> + Send + Sync,
    {
        self.par_for_each_mut(move |ptr, index, coord| {
            *ptr += other.get(index, coord);
        });
    }
    fn par_assign<F>(&mut self, other: &F)
    where
        F: Lattice<D, T> + Send + Sync,
        T: AddAssign<T> + Send + Sync,
    {
        self.par_for_each_mut(move |ptr, index, coord| {
            *ptr = other.get(index, coord);
        });
    }
}

#[must_use = "Lattice operations do nothing unless used"]
pub struct LatticePlus<F1, F2> {
    lhs: F1,
    rhs: F2,
}

impl<const D: usize, T, F1, F2> Lattice<D, T> for LatticePlus<F1, F2>
where
    F1: Lattice<D, T>,
    F2: Lattice<D, T>,
    T: Add<T, Output = T>,
{
    fn dim(&self) -> &VecN<D, usize> {
        self.lhs.dim()
    }

    fn get(&self, index: usize, coord: &VecN<D, usize>) -> T {
        self.lhs.get(index, coord) + self.rhs.get(index, coord)
    }
}

#[must_use = "Lattice operations do nothing unless used"]
pub struct LatticeMap<T, F, Map> {
    field: F,
    mapper: Map,
    _src: PhantomData<T>,
}

impl<const D: usize, T, T2, F, Map> Lattice<D, T2> for LatticeMap<T, F, Map>
where
    F: Lattice<D, T>,
    Map: Fn(T) -> T2,
{
    fn dim(&self) -> &VecN<D, usize> {
        self.field.dim()
    }

    fn get(&self, index: usize, coord: &VecN<D, usize>) -> T2 {
        (self.mapper)(self.field.get(index, coord))
    }
}

pub struct LatticeSupplier<const D: usize, F> {
    dim: VecN<D, usize>,
    supplier: F,
}

impl<const D: usize, F> LatticeSupplier<D, F> {
    pub fn new<T>(dim: VecN<D, usize>, supplier: F) -> Self where F: Fn(usize, &VecN<D, usize>) -> T, {
        Self { dim, supplier }
    }
}

impl<const D: usize, T, F> Lattice<D, T> for LatticeSupplier<D, F>
where
    F: Fn(usize, &VecN<D, usize>) -> T,
{
    fn dim(&self) -> &VecN<D, usize> {
        &self.dim
    }

    fn get(&self, index: usize, coord: &VecN<D, usize>) -> T {
        (self.supplier)(index, coord)
    }
}

#[must_use = "Lattice operations do nothing unless used"]
pub struct LatticeFlip<const D: usize, F> {
    field: F,
}

impl<const D: usize, T, F> Lattice<D, T> for LatticeFlip<D, F> where
    F: Lattice<D, T>,
{
    fn dim(&self) -> &VecN<D, usize> {
        self.field.dim()
    }

    fn get(&self, _index: usize, coord: &VecN<D, usize>) -> T {
        let dim = self.dim();
        let coord2 = coord.flip(dim);
        let index2 = dim.encode_coord(&coord2);
        self.field.get(index2, &coord2)
    }
}

#[must_use = "Lattice operations do nothing unless used"]
pub struct LatticeScalarMul<const D: usize, T, F> {
    factor: T,
    field: F,
}

impl<const D: usize, T, F> Lattice<D, T> for LatticeScalarMul<D, T, F> where
    F: Lattice<D, T>,
    T: Mul<T, Output = T> + Clone,
{
    fn dim(&self) -> &VecN<D, usize> {
        self.field.dim()
    }

    fn get(&self, index: usize, coord: &VecN<D, usize>) -> T {
        self.factor.clone() * self.field.get(index, coord)
    }
}

#[derive(Encode, Decode)]
pub struct BoxLattice<const D: usize, T> {
    data: Vec<T>,
    dim: VecN<D, usize>,
}

impl<const D: usize, T> BoxLattice<D, T> {
    pub fn zeros(dim: VecN<D, usize>) -> Self where T: Zero + Copy {
        Self { data: vec![T::zero(); dim.product()], dim }
    }
    pub fn view(&self) -> LatticeView<'_, D, [T], T, DirectStride> {
        LatticeView {
            data: self.data.as_slice(),
            dim: self.dim,
            strides: DirectStride,
            _elem: PhantomData,
        }
    }
    pub fn par_axis_co_iter_mut(&mut self, axis: usize) -> CoAxisIteratorMut<'_, D, T> {
        CoAxisIteratorMut::from_slice_mut_shape(
            self.data.as_mut_slice(),
            self.dim,
            self.dim.strides(),
            axis,
        )
    }
}

impl<const D: usize, T> Lattice<D, T> for BoxLattice<D, T>
where
    T: Clone,
{
    fn dim(&self) -> &VecN<D, usize> {
        &self.dim
    }

    fn get(&self, index: usize, _coord: &VecN<D, usize>) -> T {
        self.data[index].clone()
    }
}

impl<const D: usize, T> LatticeMut<D, T> for BoxLattice<D, T> {
    fn par_for_each_mut<F>(&mut self, mapper: F)
    where
        F: Fn(&mut T, usize, &VecN<D, usize>) + Send + Sync,
        T: Send + Sync,
    {
        let len = self.data.len();
        let dim = self.dim;
        self.data
            .par_iter_mut()
            .zip(0..len)
            .for_each(move |(ptr, index)| {
                let coord = dim.decode_coord(index);
                mapper(ptr, index, &coord);
            });
    }

    fn get_mut(&mut self, index: usize, _coord: &VecN<D, usize>) -> &mut T {
        &mut self.data[index]
    }
}

impl<const D: usize, T> Index<[usize; D]> for BoxLattice<D, T> {
    type Output = T;

    fn index(&self, index: [usize; D]) -> &Self::Output {
        &self.data[self.dim.strides().inner(&VecN::new(index))]
    }
}

impl<const D: usize, T> IndexMut<[usize; D]> for BoxLattice<D, T> {
    fn index_mut(&mut self, index: [usize; D]) -> &mut Self::Output {
        &mut self.data[self.dim.strides().inner(&VecN::new(index))]
    }
}

pub trait LatticeStride<const D: usize> {
    fn map_index(&self, index: usize, coord: &VecN<D, usize>) -> usize;
}

pub struct DirectStride;
impl<const D: usize> LatticeStride<D> for DirectStride {
    fn map_index(&self, index: usize, _coord: &VecN<D, usize>) -> usize {
        index
    }
}

impl<const D: usize> LatticeStride<D> for VecN<D, usize> {
    fn map_index(&self, _index: usize, coord: &VecN<D, usize>) -> usize {
        self.inner(coord)
    }
}

pub struct LatticeView<'a, const D: usize, I: ?Sized, T, S> {
    data: &'a I,
    dim: VecN<D, usize>,
    strides: S,
    _elem: PhantomData<T>,
}

impl<'a, const D: usize, I, T, S> Lattice<D, T> for LatticeView<'a, D, I, T, S>
where
    I: ?Sized + Index<usize, Output = T>,
    T: Clone + Zero + Copy,
    S: LatticeStride<D>,
{
    fn dim(&self) -> &VecN<D, usize> {
        &self.dim
    }

    fn get(&self, index: usize, coord: &VecN<D, usize>) -> T {
        self.data[self.strides.map_index(index, coord)].clone()
    }
}

pub struct CoAxisParallelIteratorMut<'a, const D: usize, T>(CoAxisIteratorMut<'a, D, T>);

pub struct CoAxisIteratorMut<'a, const D: usize, T> {
    ptr: NonNull<T>,
    dim: VecN<D, usize>,
    strides: VecN<D, usize>,
    cursor: Range<usize>,
    result_length: usize,
    result_stride: usize,
    _marker: PhantomData<&'a mut [T]>,
}

unsafe impl<const D: usize, T: Send> Send for CoAxisIteratorMut<'_, D, T> {}
unsafe impl<const D: usize, T: Sync> Sync for CoAxisIteratorMut<'_, D, T> {}

fn move_axis_to_first<const D: usize, T: Clone>(vec: VecN<D, T>, axis: usize) -> VecN<D, T> {
    let mut ret = vec.clone();
    ret[0] = vec[axis].clone();
    let mut j = 1usize;
    for i in 0..D {
        if axis != i {
            ret[j] = vec[i].clone();
            j += 1;
        }
    }
    ret
}

impl<'a, const D: usize, T> CoAxisIteratorMut<'a, D, T> {
    pub fn from_slice_mut_shape(
        slice: &'a mut [T],
        dim: VecN<D, usize>,
        strides: VecN<D, usize>,
        axis: usize,
    ) -> Self {
        let result_length = dim[axis];
        let result_stride = strides[axis];
        let mut new_dim = move_axis_to_first(dim, axis);
        let mut new_strides = move_axis_to_first(strides, axis);
        new_dim[0] = 1;
        new_strides[0] = 1;
        Self {
            ptr: NonNull::new(slice.as_mut_ptr()).unwrap(),
            dim: new_dim,
            strides: new_strides,
            cursor: 0..new_dim.product(),
            result_length,
            result_stride,
            _marker: PhantomData,
        }
    }
    pub fn length(&self) -> usize {
        self.dim.product()
    }
    unsafe fn get_elem(&self, cursor: usize) -> (VecN<D, usize>, ArrayMut<'a, T>) {
        let coord = self.dim.decode_coord(cursor);
        let ptr = unsafe { self.ptr.add(coord.inner(&self.strides)) };
        (
            coord,
            ArrayMut {
                ptr,
                length: self.result_length,
                stride: self.result_stride,
                _marker: PhantomData,
            },
        )
    }
}

impl<'a, const D: usize, T> Iterator for CoAxisIteratorMut<'a, D, T> {
    type Item = (VecN<D, usize>, ArrayMut<'a, T>);

    fn next(&mut self) -> Option<Self::Item> {
        self.cursor
            .next()
            .map(|cursor| unsafe { self.get_elem(cursor) })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.cursor.size_hint()
    }
}

impl<'a, const D: usize, T> ExactSizeIterator for CoAxisIteratorMut<'a, D, T> {
    fn len(&self) -> usize {
        self.cursor.len()
    }
}

impl<'a, const D: usize, T> DoubleEndedIterator for CoAxisIteratorMut<'a, D, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.cursor
            .next_back()
            .map(|cursor| unsafe { self.get_elem(cursor) })
    }
}

impl<'a, const D: usize, T> Producer for CoAxisParallelIteratorMut<'a, D, T>
where
    T: Send + Sync,
{
    type Item = <Self::IntoIter as Iterator>::Item;

    type IntoIter = CoAxisIteratorMut<'a, D, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let inner = &self.0;
        (
            Self(CoAxisIteratorMut {
                ptr: inner.ptr,
                dim: inner.dim,
                strides: inner.strides,
                cursor: inner.cursor.start..index,
                result_length: inner.result_length,
                result_stride: inner.result_stride,
                _marker: PhantomData,
            }),
            Self(CoAxisIteratorMut {
                ptr: inner.ptr,
                dim: inner.dim,
                strides: inner.strides,
                cursor: index..inner.cursor.end,
                result_length: inner.result_length,
                result_stride: inner.result_stride,
                _marker: PhantomData,
            }),
        )
    }
}

impl<'a, const D: usize, T: Send + Sync> ParallelIterator for CoAxisParallelIteratorMut<'a, D, T> {
    type Item = <CoAxisIteratorMut<'a, D, T> as Iterator>::Item;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.0.cursor.len())
    }
}

impl<'a, const D: usize, T: Send + Sync> IndexedParallelIterator for CoAxisParallelIteratorMut<'a, D, T> {
    fn len(&self) -> usize {
        self.0.cursor.len()
    }

    fn drive<C: rayon::iter::plumbing::Consumer<Self::Item>>(self, consumer: C) -> C::Result {
        bridge(self, consumer)
    }

    fn with_producer<CB: rayon::iter::plumbing::ProducerCallback<Self::Item>>(
        self,
        callback: CB,
    ) -> CB::Output {
        callback.callback(self)
    }
}

pub struct ArrayMut<'a, T> {
    ptr: NonNull<T>,
    length: usize,
    stride: usize,
    _marker: PhantomData<&'a mut [T]>,
}

impl<T> ArrayMut<'_, T> {
    pub fn write(&mut self, data: &[T])
    where
        T: Clone,
    {
        for i in 0..min(self.length, data.len()) {
            self[i] = data[i].clone();
        }
    }
    pub fn read(&self, data: &mut [T])
    where
        T: Clone,
    {
        for i in 0..min(self.length, data.len()) {
            data[i] = self[i].clone();
        }
    }
    pub fn len(&self) -> usize {
        self.length
    }
}

unsafe impl<T: Send> Send for ArrayMut<'_, T> {}
unsafe impl<T: Sync> Sync for ArrayMut<'_, T> {}

impl<T> Index<usize> for ArrayMut<'_, T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        assert!(index < self.length);
        unsafe { self.ptr.add(index * self.stride).as_ref() }
    }
}

impl<T> IndexMut<usize> for ArrayMut<'_, T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        assert!(index < self.length);
        unsafe { self.ptr.add(index * self.stride).as_mut() }
    }
}

#[cfg(test)]
mod tests {
    use crate::{lat::{BoxLattice, Lattice, LatticeMut}, util::VecN};

    #[test]
    fn co_axis_iter() {
        let dim = VecN::new([4; 3]);
        let mut lat = BoxLattice::<3, i32>::zeros(dim);
        lat.par_axis_co_iter_mut(1).for_each(|(coord, mut arr)| {
            println!("coord = {}", &coord);
            arr[0] = coord[1] as i32;
            arr[1] = coord[2] as i32;
        });
        for x in 0..dim[0] {
            for z in 0..dim[2] {
                assert_eq!(lat[[x, 0, z]], x as i32);
                assert_eq!(lat[[x, 1, z]], z as i32);
            }
        }
        lat.for_each(|p, _, _|*p = 0);
        lat.par_axis_co_iter_mut(0).for_each(|(coord, mut arr)| {
            println!("coord = {}", &coord);
            arr[0] = coord[1] as i32;
            arr[1] = coord[2] as i32;
        });
        for y in 0..dim[0] {
            for z in 0..dim[2] {
                assert_eq!(lat[[0, y, z]], y as i32);
                assert_eq!(lat[[1, y, z]], z as i32);
            }
        }
    }

    #[test]
    fn flip() {
        let dim = VecN::new([4; 3]);
        let mut lat = BoxLattice::zeros(dim);
        lat.for_each(|ptr, index, _|*ptr = index);
        let flipped = lat.view().flip();
        for index in 0..dim.product() {
            let coord = dim.decode_coord(index);
            let flipped_coord = coord.flip(&dim);
            let flipped_index = dim.encode_coord(&flipped_coord);
            println!("testing coord = {}, flipped = {}", &coord, &flipped_coord);
            assert_eq!(lat.get(index, &coord), flipped.get(flipped_index, &flipped_coord));
        }
    }
}
