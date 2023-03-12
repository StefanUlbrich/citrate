use ndarray::prelude::*;

use super::{MixtureType, Categorical};


type GaussianSufficientStatistics = (Array2, Array2);

pub struct Gaussian {
    pub dimension: i32,
    pub means: Array2<f64>,
    pub covariances: Array4<f64>,
    sufficient_statistics: GaussianSufficientStatistics,
}

impl Gaussian {
    pub fn new(dimension: i32, prior: Option<f64>) -> Categorical {
        Categorical { dimension, prior }
    }
}

impl MixtureType for Categorical {
    type SufficientStatistics =GaussianSufficientStatistics;

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

    fn maximize(&mut self, sufficient_statistics: (&<Categorical as MixtureType>::SufficientStatistics, &Self::SufficientStatistics)) {
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

    fn store(&self) -> Self::SufficientStatistics {
        todo!()
    }

    fn restore(&mut self, sufficient_statistics: Self::SufficientStatistics) {
        todo!()
    }

    fn predict(&self, responsibilities: &Self::DataIn<'_>, data: &Self::DataIn<'_>) -> Self::DataOut {
        todo!()
    }
}