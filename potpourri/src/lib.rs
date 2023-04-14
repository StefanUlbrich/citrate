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
pub mod errors;
pub mod mixture;
pub mod probabilistic;

use std::option;

pub use mixture::MixtureModel;
pub use probabilistic::{Density, Latent, Probabilistic};

use errors::Error;

pub trait Mixables {
    type SufficientStatistics: Send + Sync;
    type LogLikelihood;
    type DataIn<'a>: Sync;
    type DataOut;

    // weights: Self::DataIn<'_>,

    /// The E-Step. Computes the likelihood for each component in the mixture
    fn expect(&self, data: &Self::DataIn<'_>) -> Result<(Self::LogLikelihood, f64), Error>;

    // Consider combining `compute` and `maximize` – no that is a bad idea
    // &mut self,
    // store: Option<bool>, // consider removing. The parent class should take care of that

    /// Computes the sufficient statistics from the responsibility matrix. The
    ///  Optionally, stores the
    /// sufficient statistics (for incremental learning and store.restore functionality)
    /// can be disabled for performance (defaults to `True`)
    fn compute(
        &self,
        data: &Self::DataIn<'_>,
        responsibilities: &Self::LogLikelihood,
    ) -> Result<Self::SufficientStatistics, Error>;

    /// Maximize the model parameters from
    fn maximize(&mut self, sufficient_statistics: &Self::SufficientStatistics)
        -> Result<(), Error>;

    fn predict(
        &self,
        // responsibilities: &Self::DataIn<'_>,
        data: &Self::DataIn<'_>,
    ) -> Result<Self::DataOut, Error>;

    /// Update the stored sufficient statistics (for incremental learning)
    /// Weights is a tuple (a float should suffice, if summing to one)
    fn update(
        &mut self,
        sufficient_statistics: &Self::SufficientStatistics,
        weight: f64,
    ) -> Result<(), Error>;

    /// merge multiple sufficient statistics into one.
    fn merge(
        sufficient_statistics: &[&Self::SufficientStatistics],
        weights: &[f64],
    ) -> Result<Self::SufficientStatistics, Error>;

    /// Generate a random expectation. Used as an initalization. It is recommended
    /// to draw the expectations from a univorm Dirichlet distribution.
    /// Note: This works better than an initialization method, because the layers
    /// such as the `Probabilistic` trait don't need to implement backend-specific
    /// random samplers.
    fn expect_rand(&self, data: &Self::DataIn<'_>, k: usize) -> Result<Self::LogLikelihood, Error> {
        todo!()
    }
}

/// Probabilistic mixables should implement this trait

/// A mixture model has a discrete and unobservable variable (i.e., latent) variable
/// associated with each data point. It can be interpreted as a pointer to the component
/// of a mixture generated the sample. This component computes weights the components
/// in the mixture, that is, the probability for each component that the next sample will
/// be drawn from it. In case of non-probabilistic models (k-mm and SOM) this is irrelevant.
pub trait ExpectationMaximizing {
    type DataIn<'a>;
    type DataOut;

    fn fit(&mut self, data: Self::DataIn<'_>) -> Result<(), Error>;
    fn predict(&self, data: &Self::DataIn<'_>) -> Result<Self::DataOut, Error>;
}
