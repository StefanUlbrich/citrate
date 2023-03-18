// Todo: move outside of the backend!

use crate::Mixables;

pub trait Latent<T>
where
    T: Mixables,
{
    fn join(likelihood_a: &T::LogLikelihood, likelihood_b: &T::LogLikelihood) -> T::LogLikelihood;
}

pub struct Density<T, L>
where
    // https://doc.rust-lang.org/nomicon/hrtb.html -- include in docs about GAT
    T: for<'a> Mixables<
        LogLikelihood = L::LogLikelihood,
        DataIn<'a> = L::DataIn<'a>,
    >,
    L: Mixables + Latent<T>,
{
    mixables: T,
    categorical: L,
}

impl<T, L> Mixables for Density<T, L>
where
    T: for<'a> Mixables<
        LogLikelihood = L::LogLikelihood,
        DataIn<'a> = L::DataIn<'a>,
    >,
    L: Mixables + Latent<T>,
{
    type SufficientStatistics = (
        L::SufficientStatistics,
        T::SufficientStatistics,
    );

    type LogLikelihood = T::LogLikelihood;

    type DataIn<'a> = T::DataIn<'a>;

    type DataOut = T::DataOut;

    fn expect(&self, data: &Self::DataIn<'_>) -> (Self::LogLikelihood, f64) {
        // Todo compute the second parameter
        (
            L::join(&self.categorical.expect(data).0, &self.mixables.expect(data).0),
            self.mixables.expect(data).1,
        )
    }

    fn compute(&self, responsibilities: &Self::LogLikelihood) -> Self::SufficientStatistics {
        (
            self.categorical.compute(responsibilities),
            self.mixables.compute(responsibilities),
        )
    }

    fn maximize(&mut self, sufficient_statistics: &Self::SufficientStatistics) {
        self.categorical.maximize(&sufficient_statistics.0);
        self.mixables.maximize(&sufficient_statistics.1);
    }

    fn predict(&self, data: &Self::DataIn<'_>) -> Self::DataOut {
        todo!()
    }

    fn update(&mut self, sufficient_statistics: &Self::SufficientStatistics, weight: (f64, f64)) {
        self.categorical.update(&sufficient_statistics.0, weight);
        self.mixables.update(&sufficient_statistics.1, weight);
    }

    fn merge(
        sufficient_statistics: &[&Self::SufficientStatistics],
        weights: &[f64],
    ) -> Self::SufficientStatistics {
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
