//! <div>
//! <img src="../potpourri.svg" width="800" />
//! </div>
//!
//! Package for models with discrete, unobservable latent variables that can be learned with the
//! Expectation Maximization algorithm.
//! The package aims at highest modularity to allow for easy experimentation for research
//! such as adding parallelization on clusters and exploring new models
//!
//! Conventions:
//! * Traits: Capital letters and CamelCase, adjectives used as nouns that
//!   indicate a cabability.
//! * Structs: Capital letters and CamelCase, nouns describing things and
//!   concepts
//! * methods/functions: snake_case and imperatives or short, discriptive
//!   imperative clauses
//!
//! Concepts:
//! * Avoid making assumptions about the chosen numerical framework early. All
//!   calculations are reserved for the actual models or implementations of
//!   distributions (in case of the mixture model). This is achieved by
//!   extensively using
//!   [Generic Assotiate Types
//!  (GAT)](https://blog.rust-lang.org/2022/10/28/gats-stabilization.html)

pub mod backend;
pub mod errors;
pub mod mixture;
pub mod model;

use errors::Error;
pub use mixture::{Latent, Mixable, Mixture};
pub use model::Model;

/// Average log-likelihood used to meature convergence. This is a new type for the inbuilt
/// `f64` datatype
/// ([new type idiom](https://doc.rust-lang.org/rust-by-example/generics/new_types.html)).
#[derive(Debug, Clone)]
pub struct AvgLLH(f64);

/// The main trait all models (such as the basic Mixture Models or Hidden Markov
/// Models) need to implement. It represents only the logic and parameters of
/// a machine learning model (which it forms together with an implementation of [Learning]).
/// A parametrizable model implements the bare
/// mathematics and must be associated with a struct implements the [Learning]
/// trait and orchestrates the EM algorithm.
pub trait Parametrizable {

    /// Sufficient statistics contain all relevant information of a dataset
    /// (or a part thereof) to compute all model parameters. Models are required
    /// to be able to join a pair of sufficient statistics into a single one
    /// for distributed and incremental learning.
    type SufficientStatistics: Send + Sync;

    /// The likelihoods of the hidden states for all data points.
    type Likelihood;

    /// The type of input data. A
    /// [GAT]((https://blog.rust-lang.org/2022/10/28/gats-stabilization.html))
    /// to allow for references and array views.
    type DataIn<'a>: Sync;

    /// The data type of predictions. Typically (but not necessarily),
    /// these are unsigned integer arrays
    /// for classification tasks and floats for regressions.
    type DataOut;

    // TODO check whether this is clear later /// Note that for `Mixables`, this is the log-likelihood


    /// The Expectation or E-Step of the EM algorithm. The model computes
    /// the likelihood of the hidden states for each data point. The result
    /// is often called *Responsibility Matrix*
    fn expect(&self, data: &Self::DataIn<'_>) -> Result<(Self::Likelihood, AvgLLH), Error>;

    /// Computes the sufficient statistics from the responsibility matrix (the result of the
    /// [E-step](Parametrizable::expect)).
    fn compute(
        &self,
        data: &Self::DataIn<'_>,
        responsibilities: &Self::Likelihood,
    ) -> Result<Self::SufficientStatistics, Error>;

    /// Maximize the model parameters from a sufficient statistics (the return
    /// of the [compute](Parametrizable::compute) method.
    fn maximize(&mut self, sufficient_statistics: &Self::SufficientStatistics)
        -> Result<(), Error>;

    /// The predict method produces a response to a data set once the model
    /// has been trained. Typically, the response depends on the application
    /// as either a regression or a classification task (see [DataOut](Parametrizable::DataOut))
    fn predict(
        &self,
        // responsibilities: &Self::DataIn<'_>,
        data: &Self::DataIn<'_>,
    ) -> Result<Self::DataOut, Error>;

    // TODO: Check whether we need update at all.
    /// Update the stored sufficient statistics (for incremental learning)
    /// Weights is a tuple (a float should suffice, if summing to one)
    fn update(
        &mut self,
        sufficient_statistics: &Self::SufficientStatistics,
        weight: f64,
    ) -> Result<(), Error>;

    /// Merge multiple [sufficient statistics](Parametrizable::SufficientStatistics) into one.
    fn merge(
        sufficient_statistics: &[&Self::SufficientStatistics],
        weights: &[f64],
    ) -> Result<Self::SufficientStatistics, Error>;

    // TODO: find citation!
    /// Generate a random expectation / responsibility matrix (see [E-step](Parametrizable::expect)).
    /// Used as an initalization. It is recommended
    /// to draw the expectations from a uniform Dirichlet distribution.
    ///
    /// This kind of initialization is very effective (citation needed) despite its
    /// simplicity and easy to implement (typically just calling a Dirichlet RNG from
    /// the computation backend).
    fn expect_rand(&self, _data: &Self::DataIn<'_>, _k: usize) -> Result<Self::Likelihood, Error> {
        todo!()
    }
}

/// Simple trait for an implementation that orchestrates
/// learning a [parametrizable model](Parametrizable)
pub trait Learning {
    type DataIn<'a>;
    type DataOut;

    /// Starts a training
    fn fit(&mut self, data: &Self::DataIn<'_>) -> Result<(), Error>;
    /// Generate a response to a data set after training
    fn predict(&self, data: &Self::DataIn<'_>) -> Result<Self::DataOut, Error>;
}
