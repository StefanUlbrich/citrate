use crate::Mixables;
use ndarray::prelude::*;

use super::categorical::{Categorical, Latent};

pub(crate) struct Density<T>
where
    // https://doc.rust-lang.org/nomicon/hrtb.html -- include in docs about GAT
    T: for<'a> Mixables<
        LogLikelihood = <Categorical as Mixables>::LogLikelihood,
        DataIn<'a> = <Categorical as Mixables>::DataIn<'a>,
    >,
{
    mixables: T,
    categorical: Categorical,
}

impl<T> Mixables for Density<T>
where
    T: for<'a> Mixables<
        LogLikelihood = <Categorical as Mixables>::LogLikelihood,
        DataIn<'a> = <Categorical as Mixables>::DataIn<'a>,
    >,
{
    type SufficientStatistics = (
        <Categorical as Mixables>::SufficientStatistics,
        T::SufficientStatistics,
    );

    type LogLikelihood = T::LogLikelihood;

    type DataIn<'a> = T::DataIn<'a>;

    type DataOut = T::DataOut;

    fn expect(&self, data: &Self::DataIn<'_>) -> (Self::LogLikelihood, f64) {

        // Todo compute the second parameter
        (
            self.categorical.expect(data).0 * self.mixables.expect(data).0,
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

    fn predict(
        &self,
        responsibilities: &Self::DataIn<'_>,
        data: &Self::DataIn<'_>,
    ) -> Self::DataOut {
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

    fn initialize(&mut self, n_components: i32) {
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
