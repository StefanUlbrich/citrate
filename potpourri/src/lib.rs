/**
 * Package for models with discrete, unobservable latent variables that can be learned with the
 * Expectation Maximization algorithm.
 *
 * The package aims at highest modularity to allow for easy experimentation for research
 * such as adding parallelization on clusters and exploring new models
 *
 * Conventions:
 * * Traits: Capital letters and CamelCase, adjectives used as nouns that indicate a cabability.
 * * Structs: Capital letters and CamelCase, nouns describing things and concepts
 * * methods/functions: snake_case and imperatives or short, discriptive imperative clauses
 */

pub mod backend;

#[cfg(feature = "ndarray")]
pub use crate::backend::ndarray::MixtureModel;

#[cfg(feature = "ractor")]
pub use backend::ractor::mixture::mixture;



pub trait Mixables {
    type SufficientStatistics;
    type Likelihood;
    type DataIn<'a>;
    type DataOut;

    // weights: Self::DataIn<'_>,

    /// The E-Step. Computes the likelihood for each component in the mixture
    fn expect(&self, data: &Self::DataIn<'_>) -> (Self::Likelihood, f64);

    // Consider combining `compute` and `maximize` – no that is a bad idea
    // &mut self,
    // store: Option<bool>, // consider removing. The parent class should take care of that

    /// Computes the sufficient statistics from the responsibility matrix. The
    ///  Optionally, stores the
    /// sufficient statistics (for incremental learning and store.restore functionality)
    /// can be disabled for performance (defaults to `True`)
    fn compute(
        &self,
        responsibilities: &Self::Likelihood,
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


}

/// Probabilistic mixables should implement this trait
pub trait Probabilistic {}

// pub trait LatentModel {
//     type DataIn<'a>;
//     type DataOut;

//     type Weights<'a>;

//     // look that up again
//     fn maximize(&self, responsibilities: &Self::DataIn<'_>) -> &Self::Weights<'_>;
// }

// No ajective, something with latent? LatentStateEstimating?
/// A mixture model has a discrete and unobservable variable (i.e., latent) variable
/// associated with each data point. It can be interpreted as a pointer to the component
/// of a mixture generated the sample. This component computes weights the components
/// in the mixture, that is, the probability for each component that the next sample will
/// be drawn from it. In case of non-probabilistic models (k-mm and SOM) this is irrelevant.
pub trait ExpectationMaximizing
{
    type DataIn<'a>;
    type DataOut;

    fn fit(&mut self, data: Self::DataIn<'_>) {
        // provide a standard implementation of batch and incremental learning independent of the data and sufficient statistic types
        todo!()
    }
    fn predict(&self, data: &Self::DataIn<'_>) -> Self::DataOut {
        // provide a standard implementation of batch and incremental learning independent of the data and sufficient statistic types
        todo!()
    }

    // abstract methods that depend on the backend
    /// Random initialization by creating a random responsibility matrix
    fn initialize(&mut self);
}

/// Two options:
/// 1. implementations of ExpectationMaximization decide on whether they are probabilistic or best matching

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
