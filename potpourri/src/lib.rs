pub mod backend; // TODO find a better way
pub mod default;

pub use backend::ndarray::categorical::Categorical;

pub use default::{MixtureModel, MixtureModelStatus};

// Maybe call feature but not component. Mixture type?
pub trait MixtureType {
    type SufficientStatistics;
    type ModelParameters;
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

    fn predict(
        &self,
        responsibilities: &Self::DataIn<'_>,
        data: &Self::DataIn<'_>,
    ) -> Self::DataOut;

    /// Update the stored sufficient statistics (for incremental learning)
    /// Weights is a tuple (a float should suffice, if summing to one)
    fn update(&mut self, sufficient_statistics: &Self::SufficientStatistics, weight: (f64, f64));

    /// merge multiple sufficient statistics into one.
    fn merge(
        sufficient_statistics: &[&Self::SufficientStatistics],
        weights: &[f64],
    ) -> Self::SufficientStatistics;

    // Do I need this? I doubt it--initialization is done in the algorithm.
    fn initialize(&mut self, n_components: i32);

    // Do I need these? Rather try to make the models serializable
    fn store(&self) -> &Self::SufficientStatistics;
    fn restore(&mut self, sufficient_statistics: Self::SufficientStatistics);
}

/// A mixture model has a discrete and unobservable variable (i.e., latent) variable
/// associated with each data point. It can be interpreted as a pointer to the component
/// of a mixture generated the sample. This component computes weights the components
/// in the mixture, that is, the probability for each component that the next sample will
/// be drawn from it. In case of non-probabilistic models (k-mm and SOM) this is irrelevant.
pub trait LatentModel {
    type DataIn<'a>;
    type DataOut;

    type Weights<'a>;

    // look that up again
    fn maximize(&self, responsibilities: &Self::DataIn<'_>) -> &Self::Weights<'_>;
}

pub trait ExpectationMaximization<T>
where
    T: MixtureType,
{
    type SufficientStatistics;

    fn fit(&mut self, data: T::DataIn<'_>) {
        // provide a standard implementation of batch and incremental learning independent of the data and sufficient statistic types
        todo!()
    }
    fn predict(&self, data: &T::DataIn<'_>) -> T::DataOut {
        // provide a standard implementation of batch and incremental learning independent of the data and sufficient statistic types
        todo!()
    }

    // abstract methods that depend on the backend
    /// Random initialization by creating a random responsibility matrix
    fn initialize(&mut self);
    /// collects expectations from the LatentModel and Components and computes the responsibility matrix with bayes. A separate maximize is not needed
    fn expect(&self, data: &T::DataIn<'_>) -> T::DataOut;

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
