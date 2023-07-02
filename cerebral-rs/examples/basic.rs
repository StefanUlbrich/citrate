///! The basic algorithm without any composition
use cerebral::nd_tools::argmin;
use cerebral::nd_tools::point_set::PointSet;
use itertools::Itertools;
use ndarray::prelude::*;
use ndarray::Data;

#[derive(Debug)]
pub struct SelfOrganizingMap {
    feature: Array<f64, Ix2>,
    latent: Array<f64, Ix2>,
}

impl SelfOrganizingMap {
    fn new<S, T>(feature: &ArrayBase<S, Ix2>, latent: &ArrayBase<T, Ix2>) -> SelfOrganizingMap
    where
        S: Data<Elem = f64>,
        T: Data<Elem = f64>,
    {
        // TODO: Check if #rows is equal for S, T
        let feature = feature.to_owned();
        let latent = latent.to_owned();

        SelfOrganizingMap { feature, latent }
    }

    /// Single update step
    fn adapt<S>(&mut self, feature: &ArrayBase<S, Ix1>, influence: f64, rate: f64)
    where
        S: Data<Elem = f64>,
    {
        let differences = self.feature.get_differences(&feature); // in feature space

        let best_matching = argmin(&self.feature.get_distances(&feature)); // row-index in array
                                                                           // To avoid double computation, the feature space differenes is computed explicitedly
                                                                           // already here and the method is not used for determining the best matching unit
                                                                           //let best_matching = argmin(&row_norm_l2(&differences)); // index
        let best_matching = self.latent.slice(s![best_matching, ..]); // latent coordinate

        let distances = &self.latent.get_distances(&best_matching); // in latent space

        // Compute the Gauss kernel
        let strength = distances.mapv(|e| (-1.0 * e.powi(2) / influence / 2.0).exp());

        let updated = &self.feature - (rate * strength.insert_axis(Axis(1)) * differences); // update rule

        self.feature.assign(&updated);
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

use num_traits::cast;

fn main() {
    let (n1, n2) = (10, 10);

    let mut som = SelfOrganizingMap::new(
        &Array2::<f64>::zeros((n1 * n2, 3)),
        &Array2::<f64>::zeros((n1 * n2, 2)),
    );

    // 1st variant:  "Classical" nested loops and indexing
    for x in 0..n1 {
        for y in 0..n2 {
            som.latent
                .row_mut(x * n2 + y)
                .assign(&Array::from_vec(vec![cast(x).unwrap(), cast(y).unwrap()]))
        }
    }

    // 2nd variant: More modern, functional approach with iterators
    som.latent
        .axis_iter_mut(Axis(0))
        .zip((0..n1).cartesian_product(0..n2))
        .for_each(|(mut row, (x, y))| {
            row.assign(&Array::from_vec(vec![cast(x).unwrap(), cast(y).unwrap()]))
        });

    println!("{:?}", som);
}
