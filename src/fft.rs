use std::sync::{Arc, Mutex};

use num_traits::{Float, FloatConst};
use rustfft::{Fft, FftDirection, FftNum, FftPlanner, num_complex::Complex};

use crate::lat::{BoxLattice, Lattice};

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
        lat::{BoxLattice, LatticeMut},
        util::VecN,
    };

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
