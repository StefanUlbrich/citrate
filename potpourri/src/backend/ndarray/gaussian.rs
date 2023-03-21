use ndarray::prelude::*;

use crate::{Error, Mixables, Probabilistic};

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

    type LogLikelihood = Array2<f64>;
    type DataIn<'a> = ArrayView2<'a, f64>;

    type DataOut = Array2<f64>;

    fn expect(&self, data: &Self::DataIn<'_>) -> Result<(Self::LogLikelihood, f64), Error> {
        todo!()
    }

    fn compute(
        &self,
        responsibilities: &Self::LogLikelihood,
    ) -> Result<Self::SufficientStatistics, Error> {
        todo!()
    }

    fn maximize(
        &mut self,
        sufficient_statistics: &Self::SufficientStatistics,
    ) -> Result<(), Error> {
        todo!()
    }

    fn predict(&self, data: &Self::DataIn<'_>) -> Result<Self::DataOut, Error> {
        panic!("Not implemented");
    }

    fn update(
        &mut self,
        sufficient_statistics: &Self::SufficientStatistics,
        weight: (f64, f64),
    ) -> Result<(), Error> {
        todo!()
    }

    fn merge(
        sufficient_statistics: &[&Self::SufficientStatistics],
        weights: &[f64],
    ) -> Result<Self::SufficientStatistics, Error> {
        todo!()
    }
}

impl Probabilistic<Gaussian> for Gaussian {
    fn probabilistic_predict(
        &self,
        latent_likelihood: <Gaussian as Mixables>::LogLikelihood,
        data: &<Gaussian as Mixables>::DataIn<'_>,
    ) -> Result<<Gaussian as Mixables>::DataOut, Error> {
        todo!()
    }
}
