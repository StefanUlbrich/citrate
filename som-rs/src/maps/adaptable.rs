pub struct ClassicAdaptivity {}

use ndarray::{prelude::*, Data};

use crate::{Adaptable, Neural, Tunable};
use crate::nd_tools::{
    argmin,
    point_set::{row_norm_l2, PointSet},
};

impl Adaptable for ClassicAdaptivity {
    fn adapt<S, N, T>(&mut self, neurons: &mut N, tuning: &mut T, pattern: &ArrayBase<S, Ix1>, influence: f64, rate: f64)
    where
        T: Tunable,
        N: Neural,
        S: Data<Elem = f64>,
    {
        // TODO!!
        // we want to reuse differences ... but best matching should be used ...  think about it!

        let differences = neurons.get_patterns().get_differences(&pattern); // in feature space

        let best_matching = argmin(&row_norm_l2(&differences)); // index
        let best_matching = neurons.get_lateral().slice(s![best_matching, ..]); // latent coordinate

        let distances = &neurons.get_lateral().get_distances(&best_matching); // in latent space

        // Gauss kernel
        let strength = distances.mapv(|e| (-1.0 * e.powi(2) / influence / 2.0).exp());

        let updated = neurons.get_patterns() - (rate * strength.insert_axis(Axis(1)) * differences); // update rule

        neurons.get_patterns_mut().assign(&updated);
    }
}

pub struct SmoothAdaptivity {}

impl Adaptable for SmoothAdaptivity {
    fn adapt<S, N, T>(&mut self, neurons: &mut N, tuning: &mut T, pattern: &ArrayBase<S, Ix1>, influence: f64, rate: f64)
    where
        T: Tunable,
        N: Neural,
        S: Data<Elem = f64>,
    {
        todo!()
    }
}
