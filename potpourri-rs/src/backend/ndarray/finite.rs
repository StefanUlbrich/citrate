use ndarray::prelude::*;

use crate::{Error, Latent, Parametrizable};

use super::utils::generate_random_expections;

#[derive(Clone, Debug)]

pub struct Finite {
    // pub dimension: i32,
    pub prior: Option<f64>,
    pub pmf: Array1<f64>,
    sufficient_statistics: Array2<f64>,
}

impl Finite {
    pub fn new(prior: Option<f64>) -> Finite {
        // let prior = prior.unwrap_or(1.0);
        Finite {
            // dimension,
            prior,
            pmf: Array1::<f64>::zeros(0),
            sufficient_statistics: Array2::<f64>::zeros((0, 0)),
        }
    }
}

impl Parametrizable for Finite {
    type SufficientStatistics = Array1<f64>;

    type Likelihood = Array2<f64>;

    type DataIn<'a> = ArrayView2<'a, f64>;

    type DataOut = Array2<f64>;

    fn expect(&self, _data: &Self::DataIn<'_>) -> Result<(Self::Likelihood, f64), Error> {
        Ok((
            self.pmf.slice(s![NewAxis, ..]).mapv(|x| x.ln()), //.to_owned()
            f64::NAN,
        ))
    }

    fn compute(
        &self,
        _data: &Self::DataIn<'_>,
        responsibilities: &Self::Likelihood,
    ) -> Result<Self::SufficientStatistics, Error> {
        let sum_responsibilities = responsibilities.sum_axis(Axis(0)); // (K)

        Ok(sum_responsibilities)
    }

    fn maximize(
        &mut self,
        sufficient_statistics: &Self::SufficientStatistics,
    ) -> Result<(), Error> {
        if let Some(_x) = self.prior {
        } else {
            self.pmf = sufficient_statistics / sufficient_statistics.sum();
        }

        Ok(())
    }

    fn predict(&self, _data: &Self::DataIn<'_>) -> Result<Self::DataOut, Error> {
        Err(Error::ForbiddenCode)
    }

    fn update(
        &mut self,
        sufficient_statistics: &Self::SufficientStatistics,
        weight: f64,
    ) -> Result<(), Error> {
        self.sufficient_statistics =
            &self.sufficient_statistics * (1.0 - weight) + sufficient_statistics * weight;
        Ok(())
    }

    fn merge(
        sufficient_statistics: &[&Self::SufficientStatistics],
        weights: &[f64],
    ) -> Result<Self::SufficientStatistics, Error> {
        Ok(sufficient_statistics
            .iter()
            .zip(weights.iter())
            .map(|(s, w)| *s * *w)
            .reduce(|s1, s2| &s1 + &s2)
            .unwrap())
    }

    fn expect_rand(&self, data: &Self::DataIn<'_>, k: usize) -> Result<Self::Likelihood, Error> {
        generate_random_expections(data, k)
    }
}
use tracing::info;

/// FIXME compute the likelihood.. is it just the sum of all?
impl Latent<Finite> for Finite {
    // fn join(
    //     likelihood_a: &<Finite as Parametrizable>::Likelihood,
    //     likelihood_b: &<Finite as Parametrizable>::Likelihood,
    // ) -> Result<(<Finite as Parametrizable>::Likelihood, f64), Error> {
    // }

    fn expect(
        &self,
        data: &<Finite as Parametrizable>::DataIn<'_>,
        likelihood_b: &<Finite as Parametrizable>::Likelihood,
    ) -> Result<(<Finite as Parametrizable>::Likelihood, f64), Error> {
        let likelihood_a = Parametrizable::expect(self, data)?.0;
        info!(%likelihood_a);
        let log_weighted = likelihood_a + likelihood_b; // n x k
        info!(%likelihood_b);
        let weighted = log_weighted.mapv(|x| x.exp()); // sum?
        let s = weighted.shape();
        info!("{}x{}", s[0], s[1]);
        let log_weighted_norm = weighted.sum_axis(Axis(1)).mapv(|x| x.ln());
        let s = log_weighted_norm.shape();
        info!("{}", s[0]);

        let log_responsibilities = log_weighted - &log_weighted_norm.slice(s![.., NewAxis]);

        Ok((
            log_responsibilities.mapv(|x| x.exp()),
            log_weighted_norm.mean().unwrap(),
        ))
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn it_works() {
//         // let result = add(2, 2);
//         // assert_eq!(result, 4);
//     }
// }
