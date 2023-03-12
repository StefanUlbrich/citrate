use ndarray::prelude::*;

use super::MixtureType;


pub struct Categorical<D> {
    pub dimension: i32,
    pub prior: Option<f64>,
    pub pmf: Array2<f64>,
    sufficient_statistics: Array<f64, D>
}

impl<D> Categorical<D> {
    pub fn new(dimension: i32, prior: Option<f64>) -> Categorical {
        // let prior = prior.unwrap_or(1.0);
        Categorical { dimension, prior, pmf: Array2::<f64>::zeros((0,0)), sufficient_statistics: Array2::<f64>::zeros((0,0)) }
    }
}

impl<D> MixtureType for Categorical<D> {
    type SufficientStatistics = Array<f64, D>;

    type DataIn<'a> = ArrayView2<'a, f64>;

    type DataOut = Array2<f64>;

    fn expect(&self, weights: Self::DataIn<'_>, data: &Self::DataIn<'_>) -> (Self::DataOut, f64) {
        todo!()
    }

    fn compute(
        &mut self,
        responsibilities: &Self::DataIn<'_>,
        store: Option<bool>,
    ) -> Self::SufficientStatistics {
        todo!()
    }

    fn maximize(&mut self, sufficient_statistics:  (&<Categorical as MixtureType>::SufficientStatistics, &Self::SufficientStatistics)) {
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

    fn store(&self) -> &Self::SufficientStatistics {
        &self.sufficient_statistics
    }

    fn restore(&mut self, sufficient_statistics: Self::SufficientStatistics) {
        todo!()
    }

    fn predict(&self, responsibilities: &Self::DataIn<'_>, data: &Self::DataIn<'_>) -> Self::DataOut {
        todo!()
    }
}