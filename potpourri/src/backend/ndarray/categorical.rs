


use ndarray::prelude::*;

use crate::{Mixables, Latent, Error};


pub struct Categorical {
    pub dimension: i32,
    pub prior: Option<f64>,
    pub pmf: Array1<f64>,
    sufficient_statistics: Array2<f64>
}

impl Categorical {
    pub fn new(dimension: i32, prior: Option<f64>) -> Categorical {
        // let prior = prior.unwrap_or(1.0);
        Categorical { dimension, prior, pmf: Array1::<f64>::zeros(0), sufficient_statistics: Array2::<f64>::zeros((0,0)) }
    }
}

impl Mixables for Categorical {
    type SufficientStatistics = Array1<f64>;

    type LogLikelihood = Array2<f64>;

    type DataIn<'a> = ArrayView2<'a, f64>;

    type DataOut = Array2<f64>;

    fn expect(&self, data: &Self::DataIn<'_>) -> Result<(Self::LogLikelihood, f64), Error> {
        Ok((self.pmf.clone().into_shape((1,2)).expect("Could not reshape"), f64::NAN))
    }

    // fn compute(
    //     &self,
    //     responsibilities: &Self::DataIn<'_>,
    // ) -> Self::SufficientStatistics {
    // }

    fn maximize(&mut self, sufficient_statistics: &Self::SufficientStatistics) -> Result<(), Error> {


        if let Some(x) = self.prior {

        } else {

        }

        Ok(())
    }

    fn predict(
        &self,
        data: &Self::DataIn<'_>,
    ) -> Result<Self::DataOut, Error> {
        todo!()
    }

    fn update(&mut self, sufficient_statistics: &Self::SufficientStatistics, weight: (f64, f64)) -> Result<(), Error> {
        todo!()
    }

    fn merge(
        sufficient_statistics: &[&Self::SufficientStatistics],
        weights: &[f64],
    ) -> Result<Self::SufficientStatistics, Error> {
        todo!()
    }


    fn compute(
        &self,
        responsibilities: &Self::LogLikelihood,
    ) -> Result<Self::SufficientStatistics, Error> {
        Ok(responsibilities.sum_axis(Axis(0)))

    }

}

impl Latent<Categorical> for Categorical {
    fn join(likelihood_a: &<Categorical as Mixables>::LogLikelihood, likelihood_b: &<Categorical as Mixables>::LogLikelihood) -> Result< <Categorical as Mixables>::LogLikelihood, Error> {
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