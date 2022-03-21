use crate::ndarray::{argmin, row_norm_l2};
use ndarray::{prelude::*, Data};
use num_traits::Float;

#[derive(Debug, Default)]
pub struct NetworkBase {
    feature: Array<f64, Ix2>,
    latent: Array<f64, Ix2>,
}

pub trait SelfOrganizingMap {
    fn get_number_neurons(&self) -> usize;
    fn get_feature_dim(&self) -> usize;
    fn get_latent_dim(&self) -> usize;

    fn get_best_matching<S>(
        &self,
        feature: &ArrayBase<S, Ix1>,
    ) -> usize
    where
        S: Data<Elem = f64>;

    fn adapt<S>(&mut self, feature: &ArrayBase<S, Ix1>, influence: f64, rate: f64)
    where
        S: Data<Elem = f64>;

        fn batch<S>(
        &mut self,
        features: ArrayBase<S, Ix2>,
        influences: Option<(f64, f64)>,
        rates: Option<(f64, f64)>,
        epochs: Option<usize>,
    ) where
        S: Data<Elem = f64>;


}

pub fn get_differences<A, S, T>(points: &ArrayBase<S, Ix2>, point: &ArrayBase<T, Ix1>) -> Array2<A>
where
    S: Data<Elem = A>,
    T: Data<Elem = A>,
    A: Float,
{
    points - &point.view().insert_axis(Axis(0))
}



pub fn get_distances<A, S, T>(points: &ArrayBase<S, Ix2>, point: &ArrayBase<T, Ix1>) -> Array1<A>
where
    S: Data<Elem = A>,
    T: Data<Elem = A>,
    A: Float,
{
    row_norm_l2(&get_differences(points, point))
}

impl SelfOrganizingMap for NetworkBase {
    fn get_number_neurons(&self) -> usize {
        self.feature.len_of(Axis(0))
    }

    fn get_feature_dim(&self) -> usize {
        self.feature.len_of(Axis(1))
    }

    fn get_latent_dim(&self) -> usize {
        self.latent.len_of(Axis(1))
    }

    fn get_best_matching<S>(
        &self,
        feature: &ArrayBase<S, Ix1>,
    ) -> usize
    where
        S: Data<Elem = f64>
    {
        argmin(&row_norm_l2(&get_differences(&self.feature, &feature)))
    }

    fn adapt<S>(&mut self, feature: &ArrayBase<S, Ix1>, influence: f64, rate: f64)
    where
        S: Data<Elem = f64>,
    {

        let differences = get_differences(&self.feature, &feature); // in feature space

        let bmu = argmin(&row_norm_l2(&differences)); // index
        let bmu = self.latent.slice(s![bmu, ..]); // latent coordinate

        // FIXME!
        let distances = get_distances(&self.latent, &bmu); // in latent space

        // Gauss kernel
        let strength = distances
            .mapv(|e| e.powi(2))
            .sum_axis(Axis(1))
            .mapv(|e| (-1.0 * e / influence / 2.0).exp());


        let updated = &self.feature - rate * strength * differences; // update rule

        self.feature.assign(&updated);
    }

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
