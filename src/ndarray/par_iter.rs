use std::ops::{Index, IndexMut, Range};

use rayon::{iter::{plumbing::{bridge, Consumer, Producer, ProducerCallback}, IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator}, range};

use super::array::{ArrayViewMut, ListLike, RemovableList, SliceIterMut};

pub struct ArrayViewMutParallelIter<'data, T, D> {
    array: ArrayViewMut<'data, T, D>,
    axis: usize,
}

impl<'data, T, D> ArrayViewMutParallelIter<'data, T, D> {
    pub fn new(array: ArrayViewMut<'data, T, D>, axis: usize) -> Self {
        Self { array, axis }
    }
}

impl<'data, T, D> ParallelIterator for ArrayViewMutParallelIter<'data, T, D> where
    T: Send + Sync,
    D: IndexMut<usize, Output = usize> + RemovableList + ListLike + Clone,
    <D as RemovableList>::Removed: ListLike + IndexMut<usize, Output = usize> + Clone,
{
    type Item = ArrayViewMut<'data, T, D::Removed>;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result where
        C: rayon::iter::plumbing::UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.len())
    }
}

impl<'data, T, D> IndexedParallelIterator for ArrayViewMutParallelIter<'data, T, D> where
    T: Send + Sync,
    D: IndexMut<usize, Output = usize> + RemovableList + ListLike + Clone,
    <D as RemovableList>::Removed: ListLike + IndexMut<usize, Output = usize> + Clone,
{
    fn len(&self) -> usize {
        self.array.shape().dims()[self.axis as usize]
    }

    fn drive<C: Consumer<Self::Item>>(self, consumer: C) -> C::Result {
        bridge(self, consumer)
    }

    fn with_producer<CB: ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
        callback.callback(self)
    }
}

impl<'data, T, D> Producer for ArrayViewMutParallelIter<'data, T, D> where
    D: IndexMut<usize, Output = usize> + ListLike + RemovableList + Clone,
    <D as RemovableList>::Removed: ListLike + IndexMut<usize, Output = usize> + Clone,
    T: Send + Sync,
{
    type Item = <Self::IntoIter as Iterator>::Item;

    type IntoIter = SliceIterMut<'data, T, D::Removed>;

    fn into_iter(self) -> Self::IntoIter {
        let len = self.len();
        self.array.slice_axis_move(self.axis, 0..len).slice_iter_mut_move(self.axis)
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let (arr1, arr2) = self.array.split_axis_move(self.axis, index);
        (Self {
            array: arr1,
            axis: self.axis,
        }, Self {
            array: arr2,
            axis: self.axis,
        })
    }
}

#[cfg(test)]
mod tests {
    use rayon::iter::ParallelIterator;

    use crate::ndarray::array::{ArrayShape, ArrayViewMut};

    #[test]
    fn test_par_iter_mut() {
        let shape = ArrayShape::from_dims([8, 8, 8]);
        let mut arr_data = unsafe { Box::<[i32]>::new_uninit_slice(shape.size()).assume_init() };
        let arr = ArrayViewMut::from_shape_ref_mut(&mut arr_data, shape.clone());
        arr.par_iter_mut_move(0).for_each(|arr2|{
            arr2.par_iter_mut_move(1).for_each(|mut arr3|{
                arr3.map_indexed_mut(|val, coord|{
                    coord[0] as i32
                });
            });
        });
        let arr = ArrayViewMut::from_shape_ref_mut(&mut arr_data, shape);
        for x in 0..8 {
            for y in 0..8 {
                for z in 0..8 {
                    assert_eq!(arr[[x, y, z]], y as i32);
                }
            }
        }
    }
}