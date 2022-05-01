pub struct CartesianFeature {}

use ndarray::{prelude::*, Data};
use crate::nd_tools::{
    argmin,
    point_set::{row_norm_l2, PointSet},
};

use crate::{Neural, Competitive};

impl Competitive for CartesianFeature {
    fn get_best_matching<S, N>(&self, neurons: &N, pattern: &ArrayBase<S, Ix1>) -> usize
    where
        N: Neural,
        S: Data<Elem = f64>,
    {
        argmin(&row_norm_l2(&neurons.get_patterns().get_differences(&pattern)))
    }
}
