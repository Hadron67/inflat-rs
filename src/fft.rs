use std::{
    cmp::max,
    sync::{Arc, Mutex},
};

use num_traits::{Float, FloatConst};
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};
use rustfft::{Fft, FftDirection, FftNum, FftPlanner, num_complex::Complex};

use crate::{
    field::{Dim, LatticeLike, LatticeMutLike, Vec3i},
    lat::{BoxLattice, Lattice},
};

pub struct Dft3DPlan<T: FftNum> {
    plan_x: Arc<dyn Fft<T>>,
    plan_y: Arc<dyn Fft<T>>,
    plan_z: Arc<dyn Fft<T>>,
    buffer: Box<[Complex<T>]>,
}

impl<T> Dft3DPlan<T>
where
    T: FftNum + Float + FloatConst + Send + Sync,
{
    pub fn new(dim: Dim, direction: FftDirection) -> Self {
        let mut planner = FftPlanner::new();
        let plan_x = planner.plan_fft(dim.x, direction);
        let plan_y = planner.plan_fft(dim.y, direction);
        let plan_z = planner.plan_fft(dim.z, direction);
        let scratch_size = max(
            plan_x.get_inplace_scratch_len() * dim.y * dim.z,
            max(
                plan_y.get_inplace_scratch_len() * dim.x * dim.z,
                plan_z.get_inplace_scratch_len() * dim.x * dim.y,
            ),
        );
        Self {
            plan_x,
            plan_y,
            plan_z,
            buffer: unsafe { Box::new_uninit_slice(dim.total_size() + scratch_size).assume_init() },
        }
    }
    fn transform_x<F>(&mut self, mut field: F, plan: u8)
    where
        F: LatticeMutLike<Complex<T>> + Sync,
        T: From<i32>,
    {
        let dim = field.dim();
        let plan = match plan {
            0 => self.plan_x.as_ref(),
            1 => self.plan_y.as_ref(),
            2 => self.plan_z.as_ref(),
            _ => unreachable!(),
        };
        let scratch_size = plan.get_inplace_scratch_len();
        let chunk_size = scratch_size + dim.x;
        let buffer_size = chunk_size * dim.y * dim.z;
        self.buffer[0..buffer_size]
            .par_chunks_mut(chunk_size)
            .zip(0..dim.y * dim.z)
            .for_each(|(buffer, chunk_idx)| {
                let y: i32 = (chunk_idx % dim.y).try_into().unwrap();
                let z: i32 = (chunk_idx / dim.y).try_into().unwrap();
                let (data, scratch) = buffer.split_at_mut(dim.x);
                for x0 in 0..dim.x {
                    data[x0] = field.get(Vec3i::new(x0.try_into().unwrap(), y, z)).into();
                }
                plan.process_with_scratch(data, scratch);
            });
        field.par_map_mut(|_, index| {
            let coord = dim.index_to_coord(index);
            let x: usize = coord.x.try_into().unwrap();
            let y: usize = coord.y.try_into().unwrap();
            let z: usize = coord.z.try_into().unwrap();
            let chunk_idx = y + z * dim.y;
            self.buffer[chunk_idx * chunk_size + x]
        });
    }
    pub fn transform<F>(&mut self, mut field: F)
    where
        F: LatticeMutLike<Complex<T>> + Sync,
        T: From<i32>,
    {
        self.transform_x(field.as_ref_mut(), 0);
        self.transform_x(field.as_ref_mut().transpose([1, 0, 2]), 1);
        self.transform_x(field.as_ref_mut().transpose([2, 1, 0]), 2);
    }
}

pub struct DftNDPlan<const D: usize, T> {
    plans: [Arc<dyn Fft<T>>; D],
    scratch: Mutex<Vec<Vec<Complex<T>>>>,
}

impl<const D: usize, T> DftNDPlan<D, T>
where
    T: FftNum + Float + FloatConst + Send + Sync,
{
    pub fn new(dims: [usize; D], direction: FftDirection) -> Self {
        let mut planner = FftPlanner::new();
        let plans = dims.map(|dim| planner.plan_fft(dim, direction));
        Self {
            plans,
            scratch: Mutex::new(vec![]),
        }
    }
    fn alloc_scratch(&self, total_length: usize) -> Vec<Complex<T>> {
        let v = &mut *self.scratch.lock().unwrap();
        if let Some(mut ret) = v.pop() {
            ret.resize_with(total_length, || T::zero().into());
            ret
        } else {
            vec![T::zero().into(); total_length]
        }
    }
    fn retain_scratch(&self, s: Vec<Complex<T>>) {
        self.scratch.lock().unwrap().push(s);
    }
    pub fn transform_one_axis_inplace(&self, field: &mut BoxLattice<D, Complex<T>>, axis: usize) {
        let plan = self.plans[axis].as_ref();
        let dim = field.dim()[axis];
        field
            .par_axis_co_iter_mut(axis)
            .for_each(move |(_, mut arr)| {
                let mut buffer = self.alloc_scratch(dim + plan.get_inplace_scratch_len());
                let (data, scratch) = buffer.as_mut_slice().split_at_mut(dim);
                assert!(data.len() == arr.len());
                arr.read(data);
                plan.process_with_scratch(data, scratch);
                arr.write(data);
                self.retain_scratch(buffer);
            });
    }
    pub fn transform_inplace(&self, field: &mut BoxLattice<D, Complex<T>>) {
        for i in 0..D {
            self.transform_one_axis_inplace(field, i);
        }
    }
}

#[cfg(test)]
mod tests {

    use anyhow::Ok;
    use random::Source;
    use rustfft::{
        FftDirection,
        num_complex::{Complex64, ComplexFloat},
    };

    use crate::{
        fft::DftNDPlan,
        field::{Dim, Lattice, LatticeLike, LatticeMutLike},
        lat::{BoxLattice, LatticeMut},
        util::VecN,
    };

    use super::Dft3DPlan;

    #[test]
    fn fft() {
        let dim = Dim::new_equal(128);
        let mut lat = Lattice::<Complex64>::new(dim);
        let mut transformed = Lattice::<Complex64>::new(dim);
        let mut plan = Dft3DPlan::<f64>::new(dim, FftDirection::Forward);
        let mut rand = random::default(1);
        transformed.ref_mut().map_mut(|_, _| rand.read_f64().into());
        lat.ref_mut().par_assign(&mut transformed);
        plan.transform(&mut transformed);
        plan.transform(&mut transformed);
        let diff = transformed
            .mul_scalar((1.0 / (dim.total_size() as f64)).into())
            .flip()
            .add(lat.mul_scalar((-1.0).into()))
            .map(|f| f.abs() * f.abs())
            .average();
        assert!(diff <= 1e-20);
    }

    #[test]
    fn fft2() -> anyhow::Result<()> {
        use crate::lat::Lattice;
        let dim = VecN::new([8; 3]);
        let vol = dim.product() as f64;
        let mut input = BoxLattice::<3, Complex64>::zeros(dim);
        let mut rand = random::default(1);
        input.for_each(|ptr, _, _| *ptr = rand.read_f64().into());

        let mut transformed = BoxLattice::zeros(dim);
        transformed.par_assign(&input);
        let plan = DftNDPlan::new(dim.value, FftDirection::Forward);
        plan.transform_inplace(&mut transformed);
        plan.transform_inplace(&mut transformed);
        let diff = transformed
            .view()
            .mul_scalar((1.0 / vol).into())
            .flip()
            .plus(input.view().mul_scalar((-1.0).into()))
            .map(|f| f.abs() * f.abs())
            .average();
        println!("diff = {}", diff);
        assert!(diff.abs() <= 1e-20);
        Ok(())
    }
}
