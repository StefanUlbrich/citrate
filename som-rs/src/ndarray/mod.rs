pub mod point_set;
pub mod ndindex;


use ndarray::{prelude::*, Data};
use num_traits::Float;


pub fn argmin(a: &Array1<f64>) -> usize {
    let min = a.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    a.iter().position(|e| e.eq(&min)).unwrap()
}

pub fn row_norm_l2<A, S>(points: &ArrayBase<S, Ix2>) -> Array1<A>
where
    S: Data<Elem = A>,
    A: Float,
{
    points.mapv(|e| e.powi(2)).sum_axis(Axis(1)).mapv(A::sqrt)
}