pub mod categorical;

use categorical::Categorical;

use ndarray::ArrayView1;

// Maybe call feature but not component. Mixture type?
pub trait MixtureType {
    type SufficientStatistics;
    type DataIn<'a>;
    type DataOut;

    /// The E-Step. Computes the responsibility matrix and likelihood
    fn expect(&self, weights: Self::DataIn<'_>, data: &Self::DataIn<'_>) -> (Self::DataOut, f64);

    /// Computes the sufficient statistics from the responsibility matrix. Optionally, stores the
    /// sufficient statistics (for incremental learning and store.restore functionality)
    /// can be disabled for performance (defaults to `True`)
    fn compute(
        &mut self,
        responsibilities: &Self::DataIn<'_>,
        store: Option<bool>,
    ) -> Self::SufficientStatistics;

    /// Maximize the model parameters from
    fn maximize(&mut self, sufficient_statistics: &Self::SufficientStatistics);

    fn predict(&self, responsibilities: &Self::DataIn<'_>, data: &Self::DataIn<'_>) -> Self::DataOut;

    /// Update the stored sufficient statistics (for incremental learning)
    /// Weights is a tuple (a float should suffice, if summing to one)
    fn update(&mut self, sufficient_statistics: &Self::SufficientStatistics, weight: (f64, f64));

    /// merge multiple sufficient statistics into one.
    fn merge(
        sufficient_statistics: &[&Self::SufficientStatistics],
        weights: &[f64],
    ) -> Self::SufficientStatistics;

    fn initialize(&mut self, n_components: i32);

    fn store(&self) -> &Self::SufficientStatistics;
    fn restore(&mut self, sufficient_statistics: Self::SufficientStatistics);
}

pub struct MixtureModel<T>
where
    T: MixtureType,
{
    pub n_components: i32,
    pub mixture: T,
    pub weighting: Categorical,
    pub n_init: i32,
    pub n_iterations: i32,
    pub status: MixtureModelStatus,
}

pub struct MixtureModelStatus {
    pub is_fitted: bool,
    pub converged: bool,
    pub likelihood: f64
}
impl MixtureModelStatus {
    fn new() -> MixtureModelStatus {
        MixtureModelStatus { is_fitted: false, converged: false, likelihood : f64::NAN }
    }
}

impl<T> MixtureModel<T>
where
    T: MixtureType,
{
    pub fn fit(&mut self, data: T::DataIn<'_>) {
        todo!()
    }

    pub fn predict(&self, data: &T::DataIn<'_>) -> T::DataOut{
        todo!()
    }

    pub fn new(n_components: i32, mut mixture: T, prior: Option<f64>, n_init: i32, n_iterations: i32) -> MixtureModel<T> {
        let mut weighting = Categorical::new(1, prior);
        weighting.initialize(n_components);
        mixture.initialize(n_components);

        MixtureModel {
            n_components,
            mixture,
            weighting,
            n_init,
            n_iterations,
            status: MixtureModelStatus::new()
        }
    }

    pub fn store(
        &self,
    ) -> (
        &<Categorical as MixtureType>::SufficientStatistics,
        &T::SufficientStatistics,
    ) {
        (self.weighting.store(), self.mixture.store())
    }
    pub fn restore(
        &mut self,
        sufficient_statistics: (
            <Categorical as MixtureType>::SufficientStatistics,
            T::SufficientStatistics,
        ),
    ) {
        self.weighting.restore(sufficient_statistics.0);
        self.mixture.restore(sufficient_statistics.1);
    }
}



// We need a makro here
// pub struct JointDistributions<S, T> {
//     components: (S, T),
// }


/// Bayesian linear regression




///
///
///
///
///
///
///
///

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
