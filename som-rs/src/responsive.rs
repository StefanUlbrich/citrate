use crate::Neural;

// Tunable?
/// Interface for structures encapsulating representations input patterns. See
/// [neural tuning](https://en.wikipedia.org/wiki/Neuronal_tuning)
pub trait Responsive<N>
where
    N: Neural,
{
    // Cannot be specialized in implementation
    // See https://stackoverflow.com/a/53085395/9415551
    // fn get_best_matching<N,P>(&self, neurons: &N, pattern: &P)

    // Should include also get_differences and get_distances

    // In case we don't operate in euclidian space

    fn get_best_matching(&self, neurons: &N, pattern: &ArrayView1<f64>) -> usize;
}

impl<N> Responsive<N> for Box<dyn Responsive<N>> where
    N: Neural{
    fn get_best_matching(&self, neurons: &N, pattern: &ArrayView1<f64>) -> usize {
        (**self).get_best_matching(neurons, pattern)
    }
}
pub struct CartesianResponsiveness {}

use crate::nd_tools::{
    argmin,
    point_set::{row_norm_l2, PointSet},
};
use ndarray::{prelude::*, Data};

impl<N> Responsive<N> for CartesianResponsiveness
where
    N: Neural,
{
    fn get_best_matching(&self, neurons: &N, pattern: &ArrayView1<f64>) -> usize {
        argmin(&row_norm_l2(
            &neurons.get_patterns().get_differences(&pattern),
        ))
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
