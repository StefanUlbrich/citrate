use crate::Mixables;

pub struct Cluster<T>
where
    T: Mixables,
{
    mixables: T,
}

impl<T> Mixables for Cluster<T>
where
    T: Mixables,
{
    type SufficientStatistics = T::SufficientStatistics;

    type Likelihood = T::Likelihood;

    type DataIn<'a> = T::DataIn<'a>;

    type DataOut = T::DataOut;

    fn expect(&self, data: &Self::DataIn<'_>) -> (Self::Likelihood, f64) {
        todo!()
    }

    fn maximize(&mut self, sufficient_statistics: &Self::SufficientStatistics) {
        todo!()
    }

    fn predict(
        &self,
        responsibilities: &Self::DataIn<'_>,
        data: &Self::DataIn<'_>,
    ) -> Self::DataOut {
        todo!()
    }

    fn update(&mut self, sufficient_statistics: &Self::SufficientStatistics, weight: (f64, f64)) {
        todo!()
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

    fn compute(&self, responsibilities: &Self::Likelihood) -> Self::SufficientStatistics {
        todo!()
    }
}
