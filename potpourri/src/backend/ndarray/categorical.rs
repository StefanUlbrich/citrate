use ndarray::prelude::*;

use crate::{Error, Latent, Mixables};

use super::utils::generate_random_expections;

#[derive(Clone, Debug)]

pub struct Categorical {
    pub dimension: i32,
    pub prior: Option<f64>,
    pub pmf: Array1<f64>,
    sufficient_statistics: Array2<f64>,
}

impl Categorical {
    pub fn new(dimension: i32, prior: Option<f64>) -> Categorical {
        // let prior = prior.unwrap_or(1.0);
        Categorical {
            dimension,
            prior,
            pmf: Array1::<f64>::zeros(0),
            sufficient_statistics: Array2::<f64>::zeros((0, 0)),
        }
    }
}

impl Mixables for Categorical {
    type SufficientStatistics = Array1<f64>;

    type LogLikelihood = Array2<f64>;

    type DataIn<'a> = ArrayView2<'a, f64>;

    type DataOut = Array2<f64>;

    fn expect(&self, _data: &Self::DataIn<'_>) -> Result<(Self::LogLikelihood, f64), Error> {
        Ok((self.pmf.slice(s![NewAxis, ..]).to_owned(), f64::NAN))
    }

    fn compute(
        &self,
        _data: &Self::DataIn<'_>,
        responsibilities: &Self::LogLikelihood,
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

    fn expect_rand(&self, data: &Self::DataIn<'_>, k: usize) -> Result<Self::LogLikelihood, Error> {
        generate_random_expections(data, k)
    }
}

/// FIXME compute the likelihood.. is it just the sum of all?
impl Latent<Categorical> for Categorical {
    fn join(
        likelihood_a: &<Categorical as Mixables>::LogLikelihood,
        likelihood_b: &<Categorical as Mixables>::LogLikelihood,
    ) -> Result<(<Categorical as Mixables>::LogLikelihood, f64), Error> {
        let log_weighted = likelihood_a + likelihood_b;
        let weighted = log_weighted.mapv(|x| x.exp());
        let responsibilities = (&log_weighted - &weighted).mapv(|x| x.ln());
        weighted.sum_axis(Axis(1)).mapv(|x| x.ln()).mean();

        Ok((
            likelihood_a + likelihood_b,
            weighted.sum_axis(Axis(1)).mapv(|x| x.ln()).mean().unwrap(),
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
