pub struct CartesianFeature {}

use ndarray::{prelude::*, Data};

use crate::{Neural, Tunable};

impl Tunable for CartesianFeature {
    fn get_best_matching<S, N>(&self, neurons: &N, pattern: &ArrayBase<S, Ix1>) -> usize
    where
        N: Neural,
        S: Data<Elem = f64>,
    {
        todo!()
    }
}
