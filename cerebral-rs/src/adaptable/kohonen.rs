#[derive(Clone)]
pub struct KohonenAdaptivity {}

use ndarray::{prelude::*, Data};

use crate::nd_tools::{
    argmin,
    point_set::{row_norm_l2, PointSet},
};
use crate::{Neural, Responsive};
use super::{BoxedAdaptable, Adaptable};

impl<N, R> Adaptable<N, R> for KohonenAdaptivity
where
    R: Responsive<N>,
    N: Neural,
{
    fn adapt(
        &mut self,
        neurons: &mut N,
        responsiveness: &mut R,
        pattern: &ArrayView1<f64>,
        influence: f64,
        rate: f64,
    ) {
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

    fn clone_dyn(&self) -> BoxedAdaptable<N,R> {
        Box::new(self.clone()) // Forward to the derive(Clone) impl    }
    }
}
// #[cfg(test)]
// mod tests {

//     #[test]
//     fn it_works() {
//         let result = 2 + 2;
//         assert_eq!(result, 4);
//     }
// }
