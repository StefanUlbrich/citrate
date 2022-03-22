//! The self-oranizing maps are implemented in this module
use crate::ndarray::{argmin, point_set::{PointSet,row_norm_l2}};
use ndarray::{prelude::*, Data};

pub mod uniform;


/// The common interface for all self-organizing map (SOM) implementations.
/// Contains the the default implementation of the learning algorithm.
/// A SOM contains a list of neurons and each points to
/// - a coordinate in the feature space (variable)
/// - a coordinate in a latent / neural space (fixed) that defines a
///   topology and thus neighborhood relations.
pub trait SelfOrganizingMap {
    //
    // Abstract methods. Must be implemented by types
    //

    fn get_feature_mut(&mut self) -> &mut Array2<f64>;
    fn get_feature(&self) -> &Array2<f64>;
    fn get_latent(&self) -> &Array2<f64>;

    //
    // Convenience methods
    //

    /// Returns the number of neurons
    fn get_number_neurons(&self) -> usize {
        self.get_feature().len_of(Axis(0))
    }

    /// Returns the dimensionality of the feature space
    fn get_feature_dim(&self) -> usize {
        self.get_feature().len_of(Axis(1))
    }

    /// Returns the dimensionality of the latent space
    fn get_latent_dim(&self) -> usize {
        self.get_latent().len_of(Axis(1))
    }

    /// Returns the index of the best matching unit given an observed `feature`
    fn get_best_matching<S>(&self, feature: &ArrayBase<S, Ix1>) -> usize
    where
        S: Data<Elem = f64>,
    {
        argmin(&row_norm_l2(&self.get_feature().get_differences(&feature)))
    }

    //
    // default implementation of training methods
    //

    /// Adapt to a single observed `feature` given a learning `rate` and an `influence`
    /// (standard variation to a Gauss kernel) in latent space.
    fn adapt<S>(&mut self, feature: &ArrayBase<S, Ix1>, influence: f64, rate: f64)
    where
        S: Data<Elem = f64>,
    {
        // To avoid double computation, the feature space differenes is computed explicitedly
        // already here and the method is not used for determination of the best matching unit
        let differences = self.get_feature().get_differences(&feature); // in feature space

        let best_matching = argmin(&row_norm_l2(&differences)); // index
        let best_matching = self.get_latent().slice(s![best_matching, ..]); // latent coordinate

        // FIXME!
        let distances = &self.get_latent().get_distances(&best_matching); // in latent space

        // Gauss kernel
        let strength = distances
            .mapv(|e| e.powi(2))
            .sum_axis(Axis(1))
            .mapv(|e| (-1.0 * e / influence / 2.0).exp());

        let updated = self.get_feature() - (rate * strength * differences); // update rule

        self.get_feature_mut().assign(&updated);
    }

    /// Batch training of a PointSet repeatedly for a number of `epochs`.
    /// Analog to the `adapt` method, `influences` and `rates` can be specified
    /// as tuples of the start and end values at the beginning and end of
    /// the learning respectively.
    fn batch<S>(
        &mut self,
        features: ArrayBase<S, Ix2>,
        influences: Option<(f64, f64)>,
        rates: Option<(f64, f64)>,
        epochs: Option<usize>,
    ) where
        S: Data<Elem = f64>,
    {
        let influences = influences.unwrap_or((3.0, 1.0));
        let rates = rates.unwrap_or((0.7, 0.1));
        let epochs = epochs.unwrap_or(1);

        let n_samples = features.len_of(Axis(0));

        for epoch in 0..epochs {
            for (i, feature) in features.outer_iter().enumerate() {
                let progress = ((epoch * n_samples + i) as f64) / ((epochs * n_samples) as f64);
                let rate = rates.0 * (rates.1 / rates.0).powf(progress);
                let influence = influences.0 * (influences.1 / influences.0).powf(progress);

                self.adapt(&feature, influence, rate);
            }
        }
    }
}
