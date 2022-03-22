//! Adds functions that extends 2D float arrays such that they behave as PointSets
use ndarray::{prelude::*, Data};
use num_traits::Float;


/// Computes the L2 norm for all rows of a `PointSet`
///
/// # Examples
///
/// ```
/// use som_rs::ndarray::point_set::row_norm_l2;
///
/// assert_eq!(row_norm_l2(points), );
/// ```
pub fn row_norm_l2<A, S>(points: &ArrayBase<S, Ix2>) -> Array1<A>
where
    S: Data<Elem = A>,
    A: Float,
{
    points.mapv(|e| e.powi(2)).sum_axis(Axis(1)).mapv(A::sqrt)
}



pub trait PointSet<A> {
    /// Computes the difference of each row to a given `point` (1D)
    ///
    /// # Examples
    ///
    /// ```
    /// // Example template not implemented for trait functions
    /// ```
    fn get_differences<S>(&self, point: &ArrayBase<S, Ix1>) -> Array2<A>
    where
        S: Data<Elem = A>,
        A: Float;

    /// Computes the Eucledean distance of each row to a given `point` (1D)
    ///
    /// # Examples
    ///
    /// ```
    /// // Example template not implemented for trait functions
    /// ```
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