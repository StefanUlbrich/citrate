// Todo: move outside of the backend!

use crate::{Mixables, Error};

/// An additional interface for `Mixables` that can be used as latent states.
/// These can be categorical distributions, with or without finite Dirichlet
/// or infinite Dirichlet process priors. The `Mixables` are here used not
/// multiple components but only as one distribution of the latent states.



pub trait Latent<T>
where
    T: Mixables,
{
    fn join(likelihood_a: &T::LogLikelihood, likelihood_b: &T::LogLikelihood) -> Result<T::LogLikelihood, Error>;
}

pub trait Probabilistic<T>
where
    T: Mixables,
{
    fn probabilistic_predict(&self, latent_likelihood: T::LogLikelihood, data: &T::DataIn<'_>) -> Result<T::DataOut, Error>;
}

/// This trait represents the traditional mixture models with an underlying
/// probability density (as opposed to k-means or SOM). They have a soft
/// assignment, that is, for each sample and each component the likelihood
/// is computed that the sample belongs to the component. The alternative
/// is that a sample can only belong to one of the compent alone.
pub struct Density<T, L>
where
    // https://doc.rust-lang.org/nomicon/hrtb.html -- include in docs about GAT
    T: for<'a> Mixables<
        LogLikelihood = L::LogLikelihood,
        DataIn<'a> = L::DataIn<'a>,
    > + Probabilistic<T>,
    L: Mixables + Latent<T>,
{
    mixables: T,
    latent: L,
}

impl<T, L> Mixables for Density<T, L>
where
    T: for<'a> Mixables<
        LogLikelihood = L::LogLikelihood,
        DataIn<'a> = L::DataIn<'a>,
    > + Probabilistic<T>,
    L: Mixables + Latent<T>,
{
    type SufficientStatistics = (
        L::SufficientStatistics,
        T::SufficientStatistics,
    );

    type LogLikelihood = T::LogLikelihood;

    type DataIn<'a> = T::DataIn<'a>;

    type DataOut = T::DataOut;

    fn expect(&self, data: &Self::DataIn<'_>) -> Result<(Self::LogLikelihood, f64), Error >{
        // Todo compute the second parameter
        Ok((
            L::join(&self.latent.expect(data)?.0, &self.mixables.expect(data)?.0)?,
            self.mixables.expect(data)?.1,
        ))
    }

    fn compute(&self, responsibilities: &Self::LogLikelihood) -> Result<Self::SufficientStatistics, Error> {
        Ok((
            self.latent.compute(responsibilities)?,
            self.mixables.compute(responsibilities)?,
        ))
    }

    fn maximize(&mut self, sufficient_statistics: &Self::SufficientStatistics) ->  Result<(), Error> {
        self.latent.maximize(&sufficient_statistics.0);
        self.mixables.maximize(&sufficient_statistics.1);
        Ok(())
    }

    /// Prediction can be classification or regression depending on the implementation.
    fn predict(&self, data: &Self::DataIn<'_>) -> Result<Self::DataOut, Error>{

        let likelihood =self.latent.expect(data)?.0;
        self.mixables.probabilistic_predict(likelihood, data)
    }

    fn update(&mut self, sufficient_statistics: &Self::SufficientStatistics, weight: (f64, f64)) -> Result<(), Error> {
        self.latent.update(&sufficient_statistics.0, weight)?;
        self.mixables.update(&sufficient_statistics.1, weight)?;
        Ok(())
    }

    fn merge(
        sufficient_statistics: &[&Self::SufficientStatistics],
        weights: &[f64],
    ) -> Result<Self::SufficientStatistics, Error> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        // let result = add(2, 2);
        // assert_eq!(result, 4);
    }
}
