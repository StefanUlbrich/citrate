use ndarray::{prelude::*, Data};
use num_traits::Float;
use super::row_norm_l2;

pub trait PointSet<A> {
    fn get_differences<S>(&self, point: &ArrayBase<S, Ix1>) -> Array2<A>
    where
        S: Data<Elem = A>,
        A: Float;

    fn get_distances<S>(&self, point: &ArrayBase<S, Ix1>) -> Array1<A>
    where
        S: Data<Elem = A>,
        A: Float;
}

impl<A, T> PointSet<A> for ArrayBase<T, Ix2>
where
    T: Data<Elem = A>,
    A: Float,
{
    fn get_differences<S>(&self, point: &ArrayBase<S, Ix1>) -> Array2<A>
    where
        S: Data<Elem = A>,
    {
        self - &point.view().insert_axis(Axis(0))
    }

    fn get_distances<S>(&self, point: &ArrayBase<S, Ix1>) -> Array1<A>
    where
        S: Data<Elem = A>,
        A: Float,
    {
        row_norm_l2(&self.get_differences(point))
    }
}