use ndarray::prelude::*;

use crate::Mixables;

pub struct Gaussian {
    pub dimension: i32,
    pub means: Array2<f64>,
    pub covariances: Array4<f64>,
}

impl Gaussian {
    pub fn new(dimension: i32, prior: Option<f64>) -> Gaussian {
        todo!()
    }
}

impl Mixables for Gaussian {
    type SufficientStatistics = (Array2<f64>, Array2<f64>);

    type DataIn<'a> = ArrayView2<'a, f64>;
    type DataOut = Array2<f64>;

    fn maximize(&mut self, sufficient_statistics: &Self::SufficientStatistics) {
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

    fn predict(&self, data: &Self::DataIn<'_>) -> Self::DataOut {
        todo!()
    }

    type LogLikelihood = Array2<f64>;

    fn compute(&self, responsibilities: &Self::LogLikelihood) -> Self::SufficientStatistics {
        todo!()
    }

    fn expect(&self, data: &Self::DataIn<'_>) -> (Self::LogLikelihood, f64) {
        todo!()
    }
}
