use std::f64::consts::PI;

use crate::{Error, Mixables, Probabilistic};
use itertools::izip;
use ndarray::parallel::prelude::*;
use ndarray::prelude::*;

use super::utils::{
    get_det_spd, get_shape2, get_shape3, get_weighted_means, get_weighted_sum, invert_spd,
};

#[derive(Default, Debug, Clone)]
pub struct Gaussian {
    pub means: Array2<f64>,
    pub covariances: Array3<f64>,
    pub precisions: Array3<f64>,
    pub summands: Array1<f64>,
    sufficient_statistics: <Gaussian as Mixables>::SufficientStatistics,
}

impl Gaussian {
    pub fn new() -> Gaussian {
        Gaussian {
            ..Default::default()
        }
    }
}

impl Mixables for Gaussian {
    /// The sufficient statistics that can be extracted from
    /// the data to maximize the parameters. It is a triple of
    /// arrays (names as in Kimura et al.):
    ///
    ///  $$ e=m\cdot c^2 $$
    ///
    /// $$ \begin{aligned}
    /// a_j = \sum_i^n r_{ij}&, \qquad (k) \\\\
    /// b_j = \sum_i^n r_{ij} \cdot x_i &, \qquad (k \times d) \\\\
    /// c_j = \sum_i^n r_{ij} \cdot x_i^T\cdot x_i&, \qquad (k \times d \times d) \\\\
    /// \end{aligned} $$
    type SufficientStatistics = (Array1<f64>, Array2<f64>, Array3<f64>);

    type LogLikelihood = Array2<f64>;
    type DataIn<'a> = ArrayView2<'a, f64>;

    type DataOut = Array2<f64>;

    fn expect(&self, data: &Self::DataIn<'_>) -> Result<(Self::LogLikelihood, f64), Error> {
        let [k, _d, _] = get_shape3(&self.covariances)?;

        // n x k x d
        let adjusted = &data.slice(s![.., NewAxis, ..]) - &self.means.slice(s![NewAxis, .., ..]);

        let [n, _d] = get_shape2(data)?;

        let mut responsibilities = Array2::<f64>::default((n, k));

        // TODO analyse whether it's faster to parallelize in the outer loop
        // Then we would have to work with enumerates
        izip!(
            adjusted.axis_iter(Axis(0)),
            responsibilities.axis_iter_mut(Axis(0)),
        ) // iterate n
        .for_each(|(sample, mut responsibility)| {
            (
                sample.axis_iter(Axis(0)),
                responsibility.axis_iter_mut(Axis(0)),
                self.precisions.axis_iter(Axis(0)),
                self.summands.axis_iter(Axis(0)),
            ) //iterate k
                .into_par_iter()
                .for_each(|(x, mut r, precision, summand)| {
                    let x = x.slice(s![.., NewAxis]);
                    let x = &x.t().dot(&precision).dot(&x).into_shape(()).unwrap();
                    let x = &summand - x;
                    r.assign(&x);
                })
        });

        Ok((responsibilities, f64::NAN))
    }

    fn compute(
        &self,
        data: &Self::DataIn<'_>,
        responsibilities: &Self::LogLikelihood,
    ) -> Result<Self::SufficientStatistics, Error> {
        let sum_responsibilities = responsibilities.sum_axis(Axis(0)); // (K)

        let weighted_sum = get_weighted_sum(&data, &responsibilities); // (K x d)

        let [k, d] = get_shape2(&weighted_sum.view())?;

        let mut covs = Array3::<f64>::zeros((k, d, d)); // (K x d x d)

        // Einstein sum: NxD, NxK -> KxDxD ??
        (
            covs.axis_iter_mut(Axis(0)),
            responsibilities.axis_iter(Axis(1)),
        ) // iterate k (can be parallelized)
            .into_par_iter()
            .for_each(|(mut cov, resp)| {
                data.axis_iter(Axis(0)) // iterate n (cannot be parallelized)
                    .zip(resp.axis_iter(Axis(0)))
                    .for_each(|(x, r)| {
                        let x = x.slice(s![NewAxis, ..]);
                        let x = &r.slice(s![NewAxis, NewAxis]) * &x.t().dot(&x);
                        cov += &x;
                    });
            });

        Ok((sum_responsibilities, weighted_sum, covs))
    }

    fn maximize(
        &mut self,
        sufficient_statistics: &Self::SufficientStatistics,
    ) -> Result<(), Error> {
        self.means = get_weighted_means(&sufficient_statistics.1, &sufficient_statistics.0); // (K x D)

        let [k, d, _d] = get_shape3(&sufficient_statistics.2)?;
        let mut product = Array3::<f64>::zeros((k, d, d)); // means.T * means

        (
            product.axis_iter_mut(Axis(0)), // einsum('kd,ke->kde', mean, mean)
            self.means.axis_iter(Axis(0)),
        )
            .into_par_iter()
            .for_each(|(mut prod, mean)| {
                prod.assign(
                    &mean
                        .slice(s![.., NewAxis])
                        .dot(&mean.slice(s![NewAxis, ..])),
                )
            });

        self.covariances = &sufficient_statistics.2
            / &sufficient_statistics.0.slice(s![.., NewAxis, NewAxis])
            - &product;

        // Compute variables needed in the expectation step
        self.precisions = Array3::<f64>::zeros((k, d, d));
        self.summands = Array1::<f64>::zeros(k);

        (
            self.covariances.axis_iter(Axis(0)),
            self.precisions.axis_iter_mut(Axis(0)),
            self.summands.axis_iter_mut(Axis(0)),
        )
            .into_par_iter()
            .for_each(|(cov, mut prec, mut summand)| {
                prec.assign(&invert_spd(&cov).unwrap());
                summand.assign(&arr0(
                    -(k as f64) / 2.0 * (2.0 * PI).ln() - get_det_spd(&cov).unwrap().ln(),
                ))
            });

        Ok(())
    }

    fn predict(&self, _data: &Self::DataIn<'_>) -> Result<Self::DataOut, Error> {
        panic!("Not implemented");
    }

    fn update(
        &mut self,
        sufficient_statistics: &Self::SufficientStatistics,
        weight: f64,
    ) -> Result<(), Error> {
        // check values of weight
        self.sufficient_statistics.0 =
            &self.sufficient_statistics.0 * (1.0 - weight) + &sufficient_statistics.0 * weight;
        self.sufficient_statistics.1 =
            &self.sufficient_statistics.1 * (1.0 - weight) + &sufficient_statistics.1 * weight;
        self.sufficient_statistics.2 =
            &self.sufficient_statistics.2 * (1.0 - weight) + &sufficient_statistics.2 * weight;
        Ok(())
    }

    fn merge(
        sufficient_statistics: &[&Self::SufficientStatistics],
        weights: &[f64],
    ) -> Result<Self::SufficientStatistics, Error> {
        // if time matters fill arrays initialized outside of the closure
        // Reduct is slower.
        Ok(sufficient_statistics
            .iter()
            .zip(weights.iter())
            .map(|(s, w)| (&s.0 * *w, &s.1 * *w, &s.2 * *w))
            .reduce(|s1, s2| (&s1.0 + &s2.0, &s1.1 + &s2.1, &s1.2 + &s2.2))
            .unwrap())
    }
}

impl Probabilistic<Gaussian> for Gaussian {
    fn probabilistic_predict(
        &self,
        _latent_likelihood: <Gaussian as Mixables>::LogLikelihood,
        _data: &<Gaussian as Mixables>::DataIn<'_>,
    ) -> Result<<Gaussian as Mixables>::DataOut, Error> {
        Err(Error::NotImplemented)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::ndarray::utils::{filter_data, generate_samples};
    use tracing::info;
    use tracing_test::traced_test;

    #[traced_test]
    #[test]
    fn check_maximization() {
        let (data, responsibilities, covariances) = generate_samples(300000, 3, 2);

        let mut gaussian = Gaussian::new();

        let sufficient_statistics = gaussian.compute(&data.view(), &responsibilities).unwrap();

        gaussian.maximize(&sufficient_statistics).unwrap();

        assert!(covariances.abs_diff_eq(&gaussian.covariances, 1e-1))
    }

    #[traced_test]
    #[test]
    fn check_expectation() {
        // Simulate an expectation maximization step:
        // 1. generate data with responsibility, 2. maximize, 3. expect, 3. maximize again, 4. compare to ground trouth
        let (data, responsibilities, covariances) = generate_samples(60000, 3, 2);

        let mut gaussian = Gaussian::new();

        let sufficient_statistics = gaussian.compute(&data.view(), &responsibilities).unwrap();
        gaussian.maximize(&sufficient_statistics).unwrap();
        // assert!(covariances.abs_diff_eq(&gaussian.covariances, 1e-1));
        let (responsibilities, _) = gaussian.expect(&data.view()).unwrap();
        let responsibilities = responsibilities.map(|x| x.exp());
        let responsibilities =
            &responsibilities / &responsibilities.sum_axis(Axis(1)).slice(s![.., NewAxis]);

        let sufficient_statistics = gaussian.compute(&data.view(), &responsibilities).unwrap();
        gaussian.maximize(&sufficient_statistics).unwrap();

        info!(%covariances);
        info!(%gaussian.covariances);
        assert!(covariances.abs_diff_eq(&gaussian.covariances, 2e-1));
    }

    // #[traced_test]
    #[test]
    fn check_merge() {
        let (data, responsibilities, covariances) = generate_samples(30000, 3, 2);
        let (data_1, responsibilities_1) =
            filter_data(&data.view(), &responsibilities.view(), |x, _y| x[1] > 0.5).unwrap();
        let (data_2, responsibilities_2) =
            filter_data(&data.view(), &responsibilities.view(), |x, _y| x[1] <= 0.5).unwrap();

        let mut gaussian = Gaussian::new();

        let sufficient_statistics_1 = gaussian
            .compute(&data_1.view(), &responsibilities_1)
            .unwrap();
        let sufficient_statistics_2 = gaussian
            .compute(&data_2.view(), &responsibilities_2)
            .unwrap();

        gaussian.maximize(&sufficient_statistics_1).unwrap();

        // This should fail--we ignored much of the data
        assert!(!covariances.abs_diff_eq(&gaussian.covariances, 1e-1));

        gaussian.maximize(&sufficient_statistics_2).unwrap();

        // This should fail--we ignored much of the data
        assert!(!covariances.abs_diff_eq(&gaussian.covariances, 1e-1));

        let sufficient_statistics = Gaussian::merge(
            &[&sufficient_statistics_1, &sufficient_statistics_2],
            &[0.5, 0.5],
        )
        .unwrap();
        gaussian.maximize(&sufficient_statistics).unwrap();

        // This should fail--we ignored much of the data
        assert!(covariances.abs_diff_eq(&gaussian.covariances, 1e-1));
    }

    // #[traced_test]
    // #[test]
    // fn how_to_deal_with_zero_dim() {
    //     let mut x = arr0(0.0);
    //     x.assign(&arr0(1.2));
    //     let y = x.get(()).unwrap();
    //     x[()] = 4.0;
    // }
}
