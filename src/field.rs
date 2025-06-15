use std::{
    fmt::Display,
    iter::Sum,
    marker::PhantomData,
    mem::MaybeUninit,
    ops::{Add, AddAssign, DerefMut, Div, Index, Mul, Sub},
};

use num_traits::{FromPrimitive, NumAssign, Zero};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};

#[derive(Copy, Clone)]
pub struct Dim {
    pub x: usize,
    pub y: usize,
    pub z: usize,
}

impl Dim {
    pub fn new_equal(size: usize) -> Self {
        Self {
            x: size,
            y: size,
            z: size,
        }
    }
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

impl<T> TryInto<Vec3<T>> for Dim
where
    T: TryFrom<usize>,
{
    type Error = <T as TryFrom<usize>>::Error;

    fn try_into(self) -> Result<Vec3<T>, Self::Error> {
        Ok(Vec3 {
            x: self.x.try_into()?,
            y: self.y.try_into()?,
            z: self.z.try_into()?,
        })
    }
}

#[derive(Copy, Clone)]
pub struct Vec3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T> Vec3<T> {
    pub fn new(x: T, y: T, z: T) -> Self {
        Self { x, y, z }
    }
    pub fn map<T2, F>(self, mut mapper: F) -> Vec3<T2>
    where
        F: FnMut(T) -> T2,
    {
        Vec3 {
            x: mapper(self.x),
            y: mapper(self.y),
            z: mapper(self.z),
        }
    }
    pub fn total(self) -> T
    where
        T: Add<T, Output = T>,
    {
        self.x + self.y + self.z
    }
}

impl<T> Vec3<T>
where
    T: Mul<T, Output = T> + Add<T, Output = T>,
{
    pub fn inner(self, other: Self) -> T {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
}

impl<T> Vec3<T>
where
    T: Sub<T, Output = T> + PartialEq<T> + Zero,
{
    pub fn flip_wrap_around(self, sizes: Vec3<T>) -> Self {
        let x = if self.x == T::zero() {
            self.x
        } else {
            sizes.x - self.x
        };
        let y = if self.y == T::zero() {
            self.y
        } else {
            sizes.y - self.y
        };
        let z = if self.z == T::zero() {
            self.z
        } else {
            sizes.z - self.z
        };
        Vec3 { x, y, z }
    }
}

impl<T> Display for Vec3<T>
where
    T: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("({}, {}, {})", self.x, self.y, self.z))
    }
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

impl<T> Mul<T> for Vec3<T>
where
    T: Mul<T, Output = T> + Copy,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl<T> Add<Vec3<T>> for Vec3<T>
where
    T: Add<T, Output = T>,
{
    type Output = Self;

    fn add(self, rhs: Vec3<T>) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl<T: Copy> Add<T> for Vec3<T>
where
    T: Add<T, Output = T>,
{
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output {
        Self {
            x: self.x + rhs,
            y: self.y + rhs,
            z: self.z + rhs,
        }
    }
}

impl<T> Sub<Vec3<T>> for Vec3<T>
where
    T: Sub<T, Output = T>,
{
    type Output = Self;

    fn sub(self, rhs: Vec3<T>) -> Self::Output {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl<T: Copy> Sub<T> for Vec3<T>
where
    T: Sub<T, Output = T>,
{
    type Output = Self;

    fn sub(self, rhs: T) -> Self::Output {
        Self {
            x: self.x - rhs,
            y: self.y - rhs,
            z: self.z - rhs,
        }
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
    pub fn data(&self) -> &[T] {
        self.data.as_ref()
    }
    pub fn dim(&self) -> Dim {
        self.dim
    }
    pub fn ref_mut(&mut self) -> &mut Self {
        self
    }
}

impl<T: Copy> LatticeLike<T> for &Lattice<T> {
    fn dim(&self) -> Dim {
        self.dim
    }

    fn get_by_index(&self, index: usize) -> T {
        self.data[index]
    }
}

impl<T: Copy> LatticeLike<T> for &mut Lattice<T> {
    fn dim(&self) -> Dim {
        self.dim
    }

    fn get_by_index(&self, index: usize) -> T {
        self.data[index]
    }
}

impl<'data, T> IntoParallelIterator for &'data mut Lattice<T>
where
    T: Send,
{
    type Iter = rayon::iter::Zip<
        <[T] as IntoParallelRefMutIterator<'data>>::Iter,
        rayon::range::Iter<usize>,
    >;

    type Item = (<[T] as IntoParallelRefMutIterator<'data>>::Item, usize);

    fn into_par_iter(self) -> Self::Iter {
        let len = self.data.len();
        self.data.par_iter_mut().zip(0..len)
    }
}

impl<T: Copy> LatticeMutLike<T> for &mut Lattice<T> {
    fn par_map_mut<'data, F>(&'data mut self, provider: F)
    where
        F: Fn(T, usize) -> T + Send + Sync,
        [T]: IntoParallelRefMutIterator<'data>,
        <[T] as IntoParallelRefMutIterator<'data>>::Iter: IndexedParallelIterator,
        <[T] as IntoParallelRefMutIterator<'data>>::Item: DerefMut<Target = T>,
    {
        let len = self.data.len();
        self.data
            .par_iter_mut()
            .zip(0..len)
            .for_each(|(mut ptr, index)| {
                *ptr = provider(*ptr, index);
            });
    }

    fn set_by_index(&mut self, index: usize, value: T) {
        self.data[index] = value;
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
    pub fn len(&self) -> usize {
        self.ptr.len()
    }
    pub fn dim(&self) -> Dim {
        self.dim
    }
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
    pub fn len(&self) -> usize {
        self.ptr.len()
    }
}

impl<T> LatticeViewMut<'_, T> {
    pub unsafe fn from_raw<'data>(ptr: *mut [T], dim: Dim) -> LatticeViewMut<'data, T> {
        LatticeViewMut {
            ptr: unsafe { &mut *ptr },
            dim,
        }
    }
}

impl<'data, T: Copy> LatticeLike<T> for LatticeViewMut<'data, T> {
    fn dim(&self) -> Dim {
        self.dim
    }

    fn get_by_index(&self, index: usize) -> T {
        self.ptr[index]
    }
}

impl<T: Copy> LatticeMutLike<T> for LatticeViewMut<'_, T> {
    fn par_map_mut<'data, F>(&'data mut self, provider: F)
    where
        F: Fn(T, usize) -> T + Send + Sync,
        [T]: IntoParallelRefMutIterator<'data>,
        <[T] as IntoParallelRefMutIterator<'data>>::Iter: IndexedParallelIterator,
        <[T] as IntoParallelRefMutIterator<'data>>::Item: DerefMut<Target = T>,
    {
        let len = self.ptr.len();
        self.ptr
            .par_iter_mut()
            .zip(0..len)
            .for_each(|(mut ptr, index)| {
                *ptr = provider(*ptr, index);
            });
    }

    fn set_by_index(&mut self, index: usize, value: T) {
        self.ptr[index] = value;
    }
}

#[must_use = "Lattice operations do nothing unless used"]
pub struct LatticeMap<T, F, M> {
    mapper: M,
    field: F,
    _marker: PhantomData<T>,
}

impl<T, T2, F, M> LatticeLike<T2> for LatticeMap<T, F, M>
where
    F: LatticeLike<T>,
    M: Fn(T) -> T2,
{
    fn dim(&self) -> Dim {
        self.field.dim()
    }

    fn get_by_index(&self, index: usize) -> T2 {
        (self.mapper)(self.field.get_by_index(index))
    }
}

#[must_use = "Lattice operations do nothing unless used"]
pub struct LatticeDerivativeSquare<F> {
    field: F,
}

impl<T, F> LatticeLike<T> for LatticeDerivativeSquare<F>
where
    F: LatticeLike<T>,
    T: Sub<T, Output = T>
        + Mul<T, Output = T>
        + Add<T, Output = T>
        + Div<T, Output = T>
        + From<i32>
        + Copy,
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

impl<T, F> LatticeLike<T> for LatticeLaplacian<F>
where
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

impl<T, F> LatticeLike<T> for LatticeScalarMult<T, F>
where
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

impl<T, F1, F2> LatticeLike<T> for LatticeMult<F1, F2>
where
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

impl<T, F1, F2> LatticeLike<T> for LatticeAdd<F1, F2>
where
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

pub struct LatticeFlip<F> {
    field: F,
}

impl<T, F> LatticeLike<T> for LatticeFlip<F>
where
    F: LatticeLike<T>,
{
    fn dim(&self) -> Dim {
        self.field.dim()
    }

    fn get_by_index(&self, index: usize) -> T {
        self.get(self.field.dim().index_to_coord(index))
    }

    fn get(&self, coord: Vec3i) -> T {
        self.field
            .get(coord.flip_wrap_around(self.field.dim().try_into().unwrap()))
    }
}

fn inverse_permute<T: Copy>(list: [T; 3], perm: [u8; 3]) -> [T; 3] {
    [
        list[perm[0] as usize],
        list[perm[1] as usize],
        list[perm[2] as usize],
    ]
}

fn permute<T: Copy>(list: [T; 3], perm: [u8; 3]) -> [T; 3] {
    let mut ret: [T; 3] = unsafe { MaybeUninit::uninit().assume_init() };
    ret[perm[0] as usize] = list[0];
    ret[perm[1] as usize] = list[1];
    ret[perm[2] as usize] = list[2];
    ret
}

fn transposed_dim(dim: Dim, perm: [u8; 3]) -> Dim {
    let d = permute([dim.x, dim.y, dim.z], perm);
    Dim {
        x: d[0],
        y: d[1],
        z: d[2],
    }
}

fn permute_vec3<T: Copy>(vec: Vec3<T>, perm: [u8; 3]) -> Vec3<T> {
    let c = permute([vec.x, vec.y, vec.z], perm);
    Vec3 {
        x: c[0],
        y: c[1],
        z: c[2],
    }
}

fn inverse_permute_vec3<T: Copy>(vec: Vec3<T>, perm: [u8; 3]) -> Vec3<T> {
    let c = inverse_permute([vec.x, vec.y, vec.z], perm);
    Vec3 {
        x: c[0],
        y: c[1],
        z: c[2],
    }
}

pub struct LatticeTranspose<T, F> {
    field: F,
    perm: [u8; 3],
    _marker: PhantomData<T>,
}

impl<T, F> LatticeLike<T> for LatticeTranspose<T, F>
where
    F: LatticeLike<T>,
{
    fn dim(&self) -> Dim {
        transposed_dim(self.field.dim(), self.perm)
    }

    fn get_by_index(&self, index: usize) -> T {
        self.get(self.dim().index_to_coord(index))
    }

    fn get(&self, coord: Vec3<i32>) -> T {
        self.field.get(inverse_permute_vec3(coord, self.perm))
    }
}

impl<T, F> LatticeMutLike<T> for LatticeTranspose<T, F>
where
    F: LatticeMutLike<T>,
{
    fn par_map_mut<'data, F2>(&'data mut self, mapper: F2)
    where
        F2: Fn(T, usize) -> T + Send + Sync,
        [T]: IntoParallelRefMutIterator<'data>,
        <[T] as IntoParallelRefMutIterator<'data>>::Iter: IndexedParallelIterator,
        <[T] as IntoParallelRefMutIterator<'data>>::Item: DerefMut<Target = T>,
    {
        let dim = self.field.dim();
        let self_dim = self.dim();
        self.field.par_map_mut(|f, index| {
            mapper(
                f,
                self_dim.coord_to_index(permute_vec3(dim.index_to_coord(index), self.perm)),
            )
        });
    }

    fn set_by_index(&mut self, index: usize, value: T) {
        let coord = self.dim().index_to_coord(index);
        self.field.set_by_index(
            self.field
                .dim()
                .coord_to_index(inverse_permute_vec3(coord, self.perm)),
            value,
        );
    }
}

pub trait LatticeLike<T> {
    fn dim(&self) -> Dim;
    fn get_by_index(&self, index: usize) -> T;
    fn get(&self, coord: Vec3<i32>) -> T {
        self.get_by_index(self.dim().coord_to_index(coord))
    }
    fn for_each<F>(&self, mut consumer: F)
    where
        F: FnMut(T, Vec3i, usize) -> (),
    {
        let dim = self.dim();
        for index in 0..dim.total_size() {
            consumer(self.get_by_index(index), dim.index_to_coord(index), index);
        }
    }
    fn derivative_at(&self, coord: Vec3<i32>) -> Vec3<T>
    where
        T: Sub<T, Output = T> + Div<T, Output = T> + From<i32>,
    {
        let dim = self.dim();
        let size_x = dim.x.try_into().unwrap();
        let size_y = dim.y.try_into().unwrap();
        let size_z = dim.z.try_into().unwrap();
        Vec3 {
            x: (self.get_by_index(dim.coord_to_index(coord.shift_x_wrap(1, size_x)))
                - self.get_by_index(dim.coord_to_index(coord.shift_x_wrap(-1, size_x))))
                / 2.into(),
            y: (self.get_by_index(dim.coord_to_index(coord.shift_y_wrap(1, size_y)))
                - self.get_by_index(dim.coord_to_index(coord.shift_y_wrap(-1, size_y))))
                / 2.into(),
            z: (self.get_by_index(dim.coord_to_index(coord.shift_z_wrap(1, size_z)))
                - self.get_by_index(dim.coord_to_index(coord.shift_z_wrap(-1, size_z))))
                / 2.into(),
        }
    }
    fn laplacian_at(&self, coord: Vec3<i32>) -> T
    where
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
    fn sum(&self) -> T
    where
        T: Send + Sum<T>,
        Self: Sync,
    {
        let len = self.dim().total_size();
        (0..len)
            .into_par_iter()
            .map(|index| self.get_by_index(index))
            .sum::<T>()
    }
    fn average(&self) -> T
    where
        T: Send + Sum<T> + Div<T, Output = T> + FromPrimitive,
        Self: Sync,
    {
        self.sum() / T::from_usize(self.dim().total_size()).unwrap()
    }
    fn as_ref<'data>(&'data self) -> RefLatticeLike<'data, Self> {
        RefLatticeLike { field: self }
    }
    fn map<T2, M>(self, mapper: M) -> LatticeMap<T, Self, M>
    where
        Self: Sized,
        M: Fn(T) -> T2,
    {
        LatticeMap {
            mapper,
            field: self,
            _marker: PhantomData,
        }
    }
    fn derivative_square(self) -> LatticeDerivativeSquare<Self>
    where
        T: Sub<T, Output = T>
            + Mul<T, Output = T>
            + Add<T, Output = T>
            + Div<T, Output = T>
            + From<i32>
            + Copy,
        Self: Sized,
    {
        LatticeDerivativeSquare { field: self }
    }
    fn laplacian(self) -> LatticeLaplacian<Self>
    where
        Self: Sized,
    {
        LatticeLaplacian { field: self }
    }
    fn mul<F>(self, other: F) -> LatticeMult<Self, F>
    where
        Self: Sized,
        F: LatticeLike<T> + Sized,
    {
        LatticeMult {
            lhs: self,
            rhs: other,
        }
    }
    fn mul_scalar(self, factor: T) -> LatticeScalarMult<T, Self>
    where
        Self: Sized,
    {
        LatticeScalarMult {
            field: self,
            scalar: factor,
        }
    }
    fn add<F>(self, other: F) -> LatticeAdd<Self, F>
    where
        Self: Sized,
        F: LatticeLike<T>,
    {
        LatticeAdd {
            lhs: self,
            rhs: other,
        }
    }
    fn transpose(self, perm: [u8; 3]) -> LatticeTranspose<T, Self>
    where
        Self: Sized,
    {
        LatticeTranspose {
            field: self,
            perm,
            _marker: PhantomData,
        }
    }
    fn flip(self) -> LatticeFlip<Self>
    where
        Self: Sized,
    {
        LatticeFlip { field: self }
    }
    fn show_as_array_rules<F>(self, filter: F) -> ShowAsArrayRules<T, Self, F>
    where
        F: Fn(T) -> bool,
        Self: Sized,
    {
        ShowAsArrayRules {
            field: self,
            filter,
            _marker: PhantomData,
        }
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

pub trait LatticeMutLike<T>: LatticeLike<T> {
    fn set_by_index(&mut self, index: usize, value: T);
    fn set(&mut self, coord: Vec3i, value: T) {
        self.set_by_index(self.dim().coord_to_index(coord), value);
    }
    fn map_mut<F>(&mut self, mut mapper: F)
    where
        F: FnMut(T, usize) -> T,
    {
        for index in 0..self.dim().total_size() {
            self.set_by_index(index, mapper(self.get_by_index(index), index));
        }
    }
    // XXX: these trait bounds are here to makes it possible to use rayon iterators, but how to put these on the impl bound?
    fn par_map_mut<'data, F>(&'data mut self, mapper: F)
    where
        [T]: IntoParallelRefMutIterator<'data>,
        <[T] as IntoParallelRefMutIterator<'data>>::Iter: IndexedParallelIterator,
        <[T] as IntoParallelRefMutIterator<'data>>::Item: DerefMut<Target = T>,
        F: Fn(T, usize) -> T + Send + Sync;
    fn as_ref_mut<'data2>(&'data2 mut self) -> RefMutLatticeLike<'data2, Self> {
        RefMutLatticeLike { field: self }
    }
    fn par_add_assign<'data, F>(&'data mut self, provider: F)
    where
        F: LatticeLike<T> + Sync + Send,
        T: Add<T, Output = T>,
        [T]: IntoParallelRefMutIterator<'data>,
        <[T] as IntoParallelRefMutIterator<'data>>::Iter: IndexedParallelIterator,
        <[T] as IntoParallelRefMutIterator<'data>>::Item: DerefMut<Target = T>,
    {
        // no need to move here?
        self.par_map_mut(move |f, index| f + provider.get_by_index(index));
    }
    fn par_add_assign2<'data, F>(&'data mut self, provider: F)
    where
        Self: IntoParallelRefMutIterator<'data, Item = (&'data mut T, usize)>,
        T: 'data,
    {
        self.par_iter_mut();
    }
    fn par_assign<'data, F>(&'data mut self, provider: F)
    where
        F: LatticeLike<T> + Sync + Send,
        T: Add<T, Output = T>,
        [T]: IntoParallelRefMutIterator<'data>,
        <[T] as IntoParallelRefMutIterator<'data>>::Iter: IndexedParallelIterator,
        <[T] as IntoParallelRefMutIterator<'data>>::Item: DerefMut<Target = T>,
    {
        self.par_map_mut(move |_, index| provider.get_by_index(index));
    }
}

pub struct RefLatticeLike<'data, F: ?Sized> {
    field: &'data F,
}

impl<T, F> LatticeLike<T> for RefLatticeLike<'_, F>
where
    F: LatticeLike<T> + ?Sized,
{
    fn dim(&self) -> Dim {
        self.field.dim()
    }

    fn get_by_index(&self, index: usize) -> T {
        self.field.get_by_index(index)
    }
}

pub struct RefMutLatticeLike<'data, F: ?Sized> {
    field: &'data mut F,
}

impl<T, F> LatticeLike<T> for RefMutLatticeLike<'_, F>
where
    F: LatticeLike<T> + ?Sized,
{
    fn dim(&self) -> Dim {
        self.field.dim()
    }

    fn get_by_index(&self, index: usize) -> T {
        self.field.get_by_index(index)
    }
}

impl<T, F> LatticeMutLike<T> for RefMutLatticeLike<'_, F>
where
    F: LatticeMutLike<T>,
{
    fn par_map_mut<'data, F2>(&'data mut self, mapper: F2)
    where
        F2: Fn(T, usize) -> T + Send + Sync,
        [T]: IntoParallelRefMutIterator<'data>,
        <[T] as IntoParallelRefMutIterator<'data>>::Iter: IndexedParallelIterator,
        <[T] as IntoParallelRefMutIterator<'data>>::Item: DerefMut<Target = T>,
    {
        self.field.par_map_mut(mapper);
    }

    fn set_by_index(&mut self, index: usize, value: T) {
        self.field.set_by_index(index, value);
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

pub struct ShowAsArrayRules<T, F, Fil> {
    field: F,
    filter: Fil,
    _marker: PhantomData<T>,
}

impl<T, F, Fil> Display for ShowAsArrayRules<T, F, Fil>
where
    F: LatticeLike<T>,
    Fil: Fn(T) -> bool,
    T: Display + Copy,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Lattice {")?;
        let mut first = true;
        let mut ret: std::fmt::Result = Ok(());
        self.field.for_each(|value, coord, _| {
            if ret.is_ok() && (self.filter)(value) {
                if first {
                    first = false;
                } else {
                    ret = f.write_str(", ");
                }
                if ret.is_err() {
                    return;
                }
                ret = f.write_fmt(format_args!("{} -> {}", coord, value));
            }
        });
        ret?;
        f.write_str("}")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{Dim, Lattice, LatticeLike, LatticeMutLike, Vec3i};

    #[test]
    fn test_transpose() {
        let dim = Dim::new_equal(16);
        let mut lat = Lattice::<i32>::new(dim);
        lat.ref_mut().par_map_mut(|_, _| 0);
        lat.ref_mut().set(Vec3i::new(1, 2, 3), -10);
        let mut transposed = lat.ref_mut().transpose([1, 2, 0]);
        assert_eq!(transposed.get(Vec3i::new(3, 1, 2)), -10);
        transposed.set(Vec3i::new(4, 5, 6), -7);
        assert_eq!(transposed.get(Vec3i::new(4, 5, 6)), -7);

        drop(lat);
        let mut lat = Lattice::<i32>::new(dim);
        lat.ref_mut().par_map_mut(|_, _| 0);
        lat.ref_mut().set(Vec3i::new(1, 2, 3), -10);
        lat.ref_mut().set(Vec3i::new(4, 5, 6), -5);
        let mut lat2 = Lattice::new(dim);
        let mut transposed = lat2.ref_mut().transpose([1, 2, 0]);
        transposed.par_assign(&lat);
        assert_eq!((&lat2).get(Vec3i::new(2, 3, 1)), -10);
        assert_eq!((&lat2).get(Vec3i::new(5, 6, 4)), -5);
    }
}
