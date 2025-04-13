use std::{iter::Sum, ops::{Add, AddAssign, DerefMut, Div, Index, Mul, Sub}};

use num_traits::{FromPrimitive, Num, NumAssign};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

#[derive(Copy, Clone)]
pub struct Dim {
    pub x: usize,
    pub y: usize,
    pub z: usize,
}

impl Dim {
    pub fn coord_to_index(self, coord: Vec3<i32>) -> usize {
        let x: usize = coord.x.try_into().unwrap();
        let y: usize = coord.y.try_into().unwrap();
        let z: usize = coord.z.try_into().unwrap();
        x + self.x * (y + self.y * z)
    }
    pub fn index_to_coord(self, mut index: usize) -> Vec3<i32> {
        let x = (index % self.x).try_into().unwrap();
        index /= self.x;
        let y = (index % self.y).try_into().unwrap();
        index /= self.y;
        let z = index.try_into().unwrap();
        Vec3 { x, y, z }
    }
    pub fn total_size(self) -> usize {
        self.x * self.y * self.z
    }
}

#[derive(Copy, Clone)]
pub struct Vec3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T: NumAssign + PartialOrd> Vec3<T> {
    pub fn shift_x_wrap(self, by: T, size: T) -> Self {
        let mut x = self.x + by;
        if x < T::zero() {
            x += size;
        } else if x >= size {
            x -= size;
        }
        Self {
            x,
            y: self.y,
            z: self.z,
        }
    }
    pub fn shift_y_wrap(self, by: T, size: T) -> Self {
        let mut y = self.y + by;
        if y < T::zero() {
            y += size;
        } else if y >= size {
            y -= size;
        }
        Self {
            x: self.x,
            y,
            z: self.z,
        }
    }
    pub fn shift_z_wrap(self, by: T, size: T) -> Self {
        let mut z = self.z + by;
        if z < T::zero() {
            z += size;
        } else if z >= size {
            z -= size;
        }
        Self {
            x: self.x,
            y: self.y,
            z,
        }
    }
}

impl<T> Vec3<T> where
    T: Mul<T, Output = T> + Add<T, Output = T>,
{
    pub fn inner(self, other: Self) -> T {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
}

pub type Vec3i = Vec3<i32>;

pub struct Lattice<T> {
    data: Box<[T]>,
    dim: Dim,
}

impl<T> Lattice<T> {
    pub fn new(dim: Dim) -> Self {
        Self {
            data: unsafe { Box::new_uninit_slice(dim.total_size()).assume_init() },
            dim,
        }
    }
    pub fn data(&self) -> &[T] { self.data.as_ref() }
    pub fn dim(&self) -> Dim { self.dim }
}

impl<T: Copy> LatticeLike<T> for &Lattice<T> {
    fn dim(&self) -> Dim {
        self.dim
    }

    fn get_by_index(&self, index: usize) -> T {
        self.data[index]
    }
}

impl<'data, T: Copy> LatticeMutLike<'data, T> for Lattice<T> where
    [T]: IntoParallelRefMutIterator<'data>,
    <[T] as IntoParallelRefMutIterator<'data>>::Iter: IndexedParallelIterator,
    <[T] as IntoParallelRefMutIterator<'data>>::Item: DerefMut<Target = T>,
{
    fn map_mut<F>(&'data mut self, provider: F) where F: Fn(T, usize) -> T + Send + Sync {
        let len = self.data.len();
        self.data.par_iter_mut().zip(0..len).for_each(|(mut ptr, index)| {
            *ptr = provider(*ptr, index);
        });
    }
}

impl<T> Index<usize> for Lattice<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T> Index<Vec3<i32>> for Lattice<T> {
    type Output = T;

    fn index(&self, index: Vec3<i32>) -> &Self::Output {
        self.index(self.dim.coord_to_index(index))
    }
}

#[derive(Clone, Copy)]
pub struct LatticeView<'data, T> {
    ptr: &'data [T],
    dim: Dim,
}

impl<'data, T> LatticeView<'data, T> {
    pub fn new(ptr: &'data [T], dim: Dim) -> Self {
        Self { ptr, dim }
    }
    pub fn len(&self) -> usize { self.ptr.len() }
    pub fn dim(&self) -> Dim { self.dim }
}

impl<T> Index<usize> for LatticeView<'_, T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.ptr[index]
    }
}

impl<T> Index<Vec3<i32>> for LatticeView<'_, T> {
    type Output = T;

    fn index(&self, index: Vec3<i32>) -> &Self::Output {
        self.index(self.dim.coord_to_index(index))
    }
}

pub struct LatticeViewMut<'data, T> {
    ptr: &'data mut [T],
    dim: Dim,
}

impl<'data, T> LatticeViewMut<'data, T> {
    pub fn new(ptr: &'data mut [T], dim: Dim) -> Self {
        Self { ptr, dim }
    }
    pub fn len(&self) -> usize { self.ptr.len() }
}

impl<'data, T: Copy> LatticeLike<T> for LatticeViewMut<'data, T> {
    fn dim(&self) -> Dim {
        self.dim
    }

    fn get_by_index(&self, index: usize) -> T {
        self.ptr[index]
    }
}

impl<'data, T> LatticeMutLike<'data, T> for LatticeViewMut<'data, T> where
    [T]: IntoParallelRefMutIterator<'data>,
    <[T] as IntoParallelRefMutIterator<'data>>::Iter: IndexedParallelIterator,
    <[T] as IntoParallelRefMutIterator<'data>>::Item: DerefMut<Target = T>,
    T: 'data + Copy,
{
    fn map_mut<F>(&'data mut self, provider: F) where
        F: Fn(T, usize) -> T + Send + Sync,
    {
        let len = self.ptr.len();
        self.ptr.par_iter_mut().zip(0..len).for_each(|(mut ptr, index)| {
            *ptr = provider(*ptr, index);
        });
    }
}

#[must_use = "Lattice operations do nothing unless used"]
pub struct LatticeMap<F, M> {
    mapper: M,
    field: F,
}

impl<T, F, M> LatticeLike<T> for LatticeMap<F, M> where
    F: LatticeLike<T>,
    M: Fn(T) -> T,
{
    fn dim(&self) -> Dim {
        self.field.dim()
    }

    fn get_by_index(&self, index: usize) -> T {
        (self.mapper)(self.field.get_by_index(index))
    }
}

#[must_use = "Lattice operations do nothing unless used"]
pub struct LatticeDerivativeSquare<F> {
    field: F,
}

impl<T, F> LatticeLike<T> for LatticeDerivativeSquare<F> where
    F: LatticeLike<T>,
    T: Sub<T, Output = T> + Mul<T, Output = T> + Add<T, Output = T> + Div<T, Output = T> + From<i32> + Copy,
{
    fn dim(&self) -> Dim {
        self.field.dim()
    }

    fn get_by_index(&self, index: usize) -> T {
        self.get(self.dim().index_to_coord(index))
    }

    fn get(&self, coord: Vec3<i32>) -> T {
        let d = self.field.derivative_at(coord);
        d.inner(d)
    }
}

#[must_use = "Lattice operations do nothing unless used"]
pub struct LatticeLaplacian<F> {
    field: F,
}

impl<T, F> LatticeLike<T> for LatticeLaplacian<F> where
    F: LatticeLike<T>,
    T: AddAssign<T> + Mul<T, Output = T> + From<i32>,
{
    fn dim(&self) -> Dim {
        self.field.dim()
    }

    fn get_by_index(&self, index: usize) -> T {
        self.get(self.dim().index_to_coord(index))
    }

    fn get(&self, coord: Vec3<i32>) -> T {
        self.field.laplacian_at(coord)
    }
}

#[must_use = "Lattice operations do nothing unless used"]
pub struct LatticeScalarMult<T, F> {
    field: F,
    scalar: T,
}

impl<T, F> LatticeLike<T> for LatticeScalarMult<T, F> where
    F: LatticeLike<T>,
    T: Mul<T, Output = T> + Copy,
{
    fn dim(&self) -> Dim {
        self.field.dim()
    }

    fn get_by_index(&self, index: usize) -> T {
        self.field.get_by_index(index) * self.scalar
    }
}

#[must_use = "Lattice operations do nothing unless used"]
pub struct LatticeMult<F1, F2> {
    lhs: F1,
    rhs: F2,
}

impl<T, F1, F2> LatticeLike<T> for LatticeMult<F1, F2> where
    F1: LatticeLike<T>,
    F2: LatticeLike<T>,
    T: Mul<T, Output = T>,
{
    fn dim(&self) -> Dim {
        self.lhs.dim()
    }

    fn get_by_index(&self, index: usize) -> T {
        self.lhs.get_by_index(index) * self.rhs.get_by_index(index)
    }
}

#[must_use = "Lattice operations do nothing unless used"]
pub struct LatticeAdd<F1, F2> {
    lhs: F1,
    rhs: F2,
}

impl<T, F1, F2> LatticeLike<T> for LatticeAdd<F1, F2> where
    F1: LatticeLike<T>,
    F2: LatticeLike<T>,
    T: Add<T, Output = T>,
{
    fn dim(&self) -> Dim {
        self.lhs.dim()
    }

    fn get_by_index(&self, index: usize) -> T {
        self.lhs.get_by_index(index) + self.rhs.get_by_index(index)
    }
}

pub trait LatticeLike<T> {
    fn dim(&self) -> Dim;
    fn get_by_index(&self, index: usize) -> T;
    fn get(&self, coord: Vec3<i32>) -> T {
        self.get_by_index(self.dim().coord_to_index(coord))
    }
    fn derivative_at(&self, coord: Vec3<i32>) -> Vec3<T> where
        T: Sub<T, Output = T> + Div<T, Output = T> + From<i32>,
    {
        let dim = self.dim();
        let size_x = dim.x.try_into().unwrap();
        let size_y = dim.y.try_into().unwrap();
        let size_z = dim.z.try_into().unwrap();
        Vec3 {
            x: (self.get_by_index(dim.coord_to_index(coord.shift_x_wrap(1, size_x))) - self.get_by_index(dim.coord_to_index(coord.shift_x_wrap(-1, size_x)))) / 2.into(),
            y: (self.get_by_index(dim.coord_to_index(coord.shift_y_wrap(1, size_y))) - self.get_by_index(dim.coord_to_index(coord.shift_y_wrap(-1, size_y)))) / 2.into(),
            z: (self.get_by_index(dim.coord_to_index(coord.shift_z_wrap(1, size_z))) - self.get_by_index(dim.coord_to_index(coord.shift_z_wrap(-1, size_z)))) / 2.into(),
        }
    }
    fn laplacian_at(&self, coord: Vec3<i32>) -> T where
        T: AddAssign<T> + Mul<T, Output = T> + From<i32>,
    {
        let dim = self.dim();
        let size_x = dim.x.try_into().unwrap();
        let size_y = dim.y.try_into().unwrap();
        let size_z = dim.z.try_into().unwrap();
        let coords = [
            dim.coord_to_index(coord.shift_x_wrap(-1, size_x)),
            dim.coord_to_index(coord.shift_x_wrap(1, size_x)),
            dim.coord_to_index(coord.shift_y_wrap(-1, size_y)),
            dim.coord_to_index(coord.shift_y_wrap(1, size_y)),
            dim.coord_to_index(coord.shift_z_wrap(-1, size_z)),
            dim.coord_to_index(coord.shift_z_wrap(1, size_z)),
        ];
        let mut ret = self.get(coord) * (-6).into();
        for c in coords {
            ret += self.get_by_index(c);
        }
        ret
    }
    fn sum(&self) -> T where
        T: Send + Sum<T>,
        Self: Sync,
    {
        let len = self.dim().total_size();
        (0..len).into_par_iter().map(|index|self.get_by_index(index)).sum::<T>()
    }
    fn average(&self) -> T where
        T: Send + Sum<T> + Div<T, Output = T> + FromPrimitive,
        Self: Sync,
    {
        self.sum() / T::from_usize(self.dim().total_size()).unwrap()
    }
    fn map<M>(self, mapper: M) -> LatticeMap<Self, M> where
        Self: Sized,
        M: Fn(T) -> T,
    {
        LatticeMap { mapper, field: self }
    }
    fn derivative_square(self) -> LatticeDerivativeSquare<Self> where
        T: Sub<T, Output = T> + Mul<T, Output = T> + Add<T, Output = T> + Div<T, Output = T> + From<i32> + Copy,
        Self: Sized,
    {
        LatticeDerivativeSquare { field: self }
    }
    fn laplacian(self) -> LatticeLaplacian<Self> where
        Self: Sized,
    {
        LatticeLaplacian { field: self }
    }
    fn mul<F>(self, other: F) -> LatticeMult<Self, F> where
        Self: Sized,
        F: LatticeLike<T> + Sized,
    {
        LatticeMult { lhs: self, rhs: other }
    }
    fn mul_scalar(self, factor: T) -> LatticeScalarMult<T, Self> where
        Self: Sized,
    {
        LatticeScalarMult { field: self, scalar: factor }
    }
    fn add<F>(self, other: F) -> LatticeAdd<Self, F> where
        Self: Sized,
        F: LatticeLike<T>,
    {
        LatticeAdd { lhs: self, rhs: other }
    }
}

impl<T: Clone + Copy> LatticeLike<T> for LatticeView<'_, T> {
    fn dim(&self) -> Dim {
        self.dim
    }

    fn get_by_index(&self, index: usize) -> T {
        self.ptr[index]
    }
}

pub trait LatticeMutLike<'data, T> {
    fn map_mut<F>(&'data mut self, mapper: F) where F: Fn(T, usize) -> T + Send + Sync;
    fn add_assign<F>(&'data mut self, provider: F) where
        F: LatticeLike<T> + Sync + Send,
        T: Add<T, Output = T>,
    {
        // no need to move here?
        self.map_mut(move |f, index|f + provider.get_by_index(index));
    }
}

pub struct ConstantField<T> {
    dim: Dim,
    value: T,
}

impl<T: Copy> LatticeLike<T> for ConstantField<T> {
    fn dim(&self) -> Dim {
        self.dim
    }

    fn get_by_index(&self, _: usize) -> T {
        self.value
    }
}
