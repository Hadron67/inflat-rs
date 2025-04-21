use std::{marker::PhantomData, mem::MaybeUninit, ops::{Index, IndexMut, Range}};

use num_traits::Zero;

use super::par_iter::ArrayViewMutParallelIter;

pub trait ListLike {
    fn zeros(len: usize) -> Self;
    fn len(&self) -> usize;
}

pub trait RemovableList {
    type Removed;

    fn remove_one_item(self, index: usize) -> Self::Removed;
}

impl<T, const N: usize> ListLike for [T; N] where
    T: Zero + Copy,
{
    fn zeros(len: usize) -> Self {
        assert!(len == N);
        [T::zero(); N]
    }

    fn len(&self) -> usize {
        <[T]>::len(self)
    }
}

fn permute<T: Copy, const N: usize>(list: [T; N], perm: &[u8]) -> [T; N] {
    let mut ret: [T; N] = unsafe { MaybeUninit::uninit().assume_init() };
    for i in 0..N {
        ret[perm[i] as usize] = list[i];
    }
    ret
}

macro_rules! impl_array_shape_for_array {
    ($len:expr) => {
        impl<T> RemovableList for [T; $len] where
            T: Copy + Zero,
        {
            type Removed = [T; $len - 1];

            fn remove_one_item(self, index: usize) -> Self::Removed {
                let mut ret = [T::zero(); $len - 1];
                let mut j = 0;
                for i in 0..$len {
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
impl_array_shape_for_array!(1);
impl_array_shape_for_array!(2);
impl_array_shape_for_array!(3);
impl_array_shape_for_array!(4);
impl_array_shape_for_array!(5);
impl_array_shape_for_array!(6);

pub struct ArrayShape<D> {
    dims: D,
    strides: D,
}

impl<D> ArrayShape<D> {
    pub fn dims(&self) -> &D {
        &self.dims
    }
    pub fn strides(&self) -> &D {
        &self.strides
    }
    pub fn remove_axis(self, axis: usize) -> ArrayShape<D::Removed> where
        D: RemovableList,
    {
        ArrayShape {
            dims: self.dims.remove_one_item(axis),
            strides: self.strides.remove_one_item(axis),
        }
    }
    pub fn size(&self) -> usize where
        D: Index<usize, Output = usize> + ListLike,
    {
        let mut ret = 1usize;
        for i in 0..self.dims.len() {
            ret *= self.dims[i];
        }
        ret
    }
    pub fn encode_coord<D2>(&self, coord: &D2) -> usize where
        D: Index<usize, Output = usize> + ListLike,
        D2: Index<usize, Output = usize>,
    {
        let mut ret = 0usize;
        for i in 0..self.strides.len() {
            let c = coord[i];
            assert!(c < self.dims[i]);
            ret += c * self.strides[i];
        }
        ret
    }
    pub fn decode_coord(&self, mut index: usize) -> D where
        D: IndexMut<usize, Output = usize> + ListLike,
    {
        let mut ret = D::zeros(self.dims.len());
        for i in 0..self.dims.len() {
            let j = self.dims.len() - 1 - i;
            let dim = self.dims[j];
            ret[j] = index % dim;
            index /= dim;
        }
        ret
    }
}

impl<D> ArrayShape<D> where
    D: IndexMut<usize, Output = usize> + ListLike,
{
    pub fn from_dims(dims: D) -> Self {
        let strides = strides_from_dim(&dims);
        Self { dims, strides }
    }
}

impl<D> Clone for ArrayShape<D> where
    D: Clone,
{
    fn clone(&self) -> Self {
        Self { dims: self.dims.clone(), strides: self.strides.clone() }
    }
}

pub fn strides_from_dim<D>(dim: &D) -> D where
    D: IndexMut<usize, Output = usize> + ListLike,
{
    let len = dim.len();
    let mut strides = D::zeros(len);
    strides[len - 1] = 1;
    for i in 0..len - 1 {
        let j = len - 1 - i;
        strides[j - 1] = strides[j] * dim[j];
    }
    strides
}

pub struct ArrayViewMut<'data, T, D> {
    ptr: *mut T,
    _ptr: PhantomData<&'data mut [T]>,
    shape: ArrayShape<D>,
}

unsafe impl<'data, T, D> Send for ArrayViewMut<'data, T, D> where T: Send + Sync {}

impl<'data, T, D> ArrayViewMut<'data, T, D> {
    pub fn shape(&self) -> &ArrayShape<D> {
        &self.shape
    }
}

impl<'data, T, D> ArrayViewMut<'data, T, D> {
    pub fn map_indexed_mut<F>(&mut self, mut mapper: F) where
        F: FnMut(T, &D) -> T,
        D: IndexMut<usize, Output = usize> + ListLike + Clone,
        T: Copy,
    {
        for i in 0..self.shape.size() {
            let coord = self.shape.decode_coord(i);
            let r = &mut self[coord.clone()];
            *r = mapper(*r, &coord);
        }
    }
}

impl<'data, T, D> ArrayViewMut<'data, T, D> where
    D: IndexMut<usize, Output = usize> + ListLike + Clone,
{
    pub fn from_dims_ref_mut(ptr: &'data mut [T], dims: D) -> Self {
        Self::from_shape_ref_mut(ptr, ArrayShape::from_dims(dims))
    }
    pub fn from_shape_ref_mut(ptr: &'data mut [T], shape: ArrayShape<D>) -> Self {
        Self {
            ptr: ptr as *mut [T] as *mut _,
            _ptr: PhantomData,
            shape,
        }
    }
    pub unsafe fn from_shape_ptr_mut(ptr: *mut T, shape: ArrayShape<D>) -> Self {
        Self {
            ptr,
            _ptr: PhantomData,
            shape,
        }
    }
    pub fn slice_axis_move(self, axis: usize, range: Range<usize>) -> ArrayViewMut<'data, T, D> where
        D: RemovableList,
    {
        let mut shape = self.shape;
        assert!(axis < shape.dims.len());
        assert!(range.start < shape.dims[axis]);
        assert!(range.end <= shape.dims[axis]);
        let offset = range.start * shape.strides[axis];
        shape.dims[axis] = range.end - range.start;
        ArrayViewMut {
            ptr: unsafe { self.ptr.add(offset) },
            _ptr: PhantomData,
            shape,
        }
    }
    pub fn split_axis_move(mut self, axis: usize, index: usize) -> (Self, Self) {
        let dim = self.shape.dims[axis];
        assert!(index < dim);
        let mut shape2 = self.shape.clone();
        self.shape.dims[axis] = index;
        shape2.dims[axis] -= index;
        (Self {
            ptr: self.ptr,
            _ptr: PhantomData,
            shape: self.shape.clone(),
        }, Self {
            ptr: unsafe { self.ptr.add(index * shape2.strides[axis]) },
            _ptr: PhantomData,
            shape: shape2,
        })
    }
    pub fn slice_iter_mut_move(self, axis: usize) -> SliceIterMut<'data, T, <D as RemovableList>::Removed> where
        D: IndexMut<usize, Output = usize> + ListLike + Clone + RemovableList,
        <D as RemovableList>::Removed: ListLike,
    {
        let len = self.shape.dims[axis];
        let stride = self.shape.strides[axis];
        SliceIterMut {
            ptr: self.ptr,
            _ptr: PhantomData,
            shape: self.shape.remove_axis(axis),
            len,
            stride,
            cursor: 0,
        }
    }
    pub fn par_iter_mut_move(self, axis: usize) -> ArrayViewMutParallelIter<'data, T, D> {
        ArrayViewMutParallelIter::new(self, axis)
    }
}

impl<'data, T, D, D2> Index<D2> for ArrayViewMut<'data, T, D> where
    D: Index<usize, Output = usize> + ListLike,
    D2: Index<usize, Output = usize>,
{
    type Output = T;

    fn index(&self, index: D2) -> &Self::Output {
        unsafe { &*self.ptr.add(self.shape.encode_coord(&index)) }
    }
}

impl<'data, T, D, D2> IndexMut<D2> for ArrayViewMut<'data, T, D> where
    D: Index<usize, Output = usize> + ListLike,
    D2: Index<usize, Output = usize>,
{
    fn index_mut(&mut self, index: D2) -> &mut Self::Output {
        unsafe { &mut *self.ptr.add(self.shape.encode_coord(&index)) }
    }
}

pub struct SliceIterMut<'a, T, D> {
    ptr: *mut T,
    _ptr: PhantomData<&'a mut T>,
    shape: ArrayShape<D>,
    stride: usize,
    len: usize,
    cursor: usize,
}

impl<'a, T, D> Iterator for SliceIterMut<'a, T, D> where
    D: IndexMut<usize, Output = usize> + Clone + ListLike,
{
    type Item = ArrayViewMut<'a, T, D>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor < self.len {
            let ret = unsafe { ArrayViewMut::from_shape_ptr_mut(self.ptr.add(self.stride * self.cursor), self.shape.clone()) };
            self.cursor += 1;
            Some(ret)
        } else {
            None
        }
    }
}

impl<'a, T, D> ExactSizeIterator for SliceIterMut<'a, T, D> where
    D: IndexMut<usize, Output = usize> + Clone + ListLike,
{
    fn len(&self) -> usize {
        self.len
    }
}

impl<'a, T, D> DoubleEndedIterator for SliceIterMut<'a, T, D> where
    D: IndexMut<usize, Output = usize> + Clone + ListLike,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.cursor > 0 {
            let ret = unsafe { ArrayViewMut::from_shape_ptr_mut(self.ptr.add(self.stride * self.cursor), self.shape.clone()) };
            self.cursor -= 1;
            Some(ret)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator};

    use super::ArrayViewMut;

    #[test]
    fn test_split_axis() {
        let mut data = unsafe { Box::<[i32]>::new_uninit_slice(512).assume_init() };
        let arr = ArrayViewMut::from_dims_ref_mut(&mut data, [8, 8, 8]);
        let (mut part1, mut part2) = arr.split_axis_move(1, 4);
        part1[[1, 0, 1]] = 12;
        part1[[1, 1, 3]] = 13;
        part2[[1, 1, 1]] = 14;
        part2[[2, 2, 1]] = 15;
        let arr =ArrayViewMut::from_dims_ref_mut(&mut data, [8, 8, 8]);
        assert_eq!(arr[[1, 0, 1]], 12);
        assert_eq!(arr[[1, 1, 3]], 13);
        assert_eq!(arr[[1, 5, 1]], 14);
        assert_eq!(arr[[2, 6, 1]], 15);
    }

    #[test]
    fn test_slice_iter_mut() {
        let mut data = unsafe { Box::<[i32]>::new_uninit_slice(512).assume_init() };
        let mut arr = ArrayViewMut::from_dims_ref_mut(&mut data, [8, 8, 8]);
        arr[[1, 0, 1]] = 12;
        arr[[1, 0, 2]] = 13;
        arr[[2, 1, 4]] = 14;
        arr[[2, 2, 1]] = 15;
        let mut iter = arr.slice_iter_mut_move(1);
        let slice = iter.next().unwrap();
        assert_eq!(slice[[1, 1]], 12);
        assert_eq!(slice[[1, 2]], 13);
        let slice = iter.next().unwrap();
        assert_eq!(slice[[2, 4]], 14);
        let slice = iter.next().unwrap();
        assert_eq!(slice[[2, 1]], 15);
    }
}
