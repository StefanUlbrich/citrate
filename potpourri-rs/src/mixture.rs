// Todo: move outside of the backend!

use crate::{Error, Parametrizable};

/// An additional interface for `Mixables` that can be used as latent states.
/// These can be categorical distributions, with or without finite Dirichlet
/// or infinite Dirichlet process priors. The `Mixables` are here used not
/// multiple components but only as one distribution of the latent states.

pub trait Latent<T>
where
    T: Parametrizable,
{

    fn expect(
        &self,
        data: &T::DataIn<'_>,
        likelihood: &T::Likelihood,
    ) -> Result<(T::Likelihood, f64), Error>;
}

pub trait Mixable<T>
where
    T: Parametrizable,
{
    fn predict(
        &self,
        latent_likelihood: T::Likelihood,
        data: &T::DataIn<'_>,
    ) -> Result<T::DataOut, Error>;
}

/// This trait represents the traditional mixture models with an underlying
/// probability density (as opposed to k-means or SOM). They have a soft
/// assignment, that is, for each sample and each component the likelihood
/// is computed that the sample belongs to the component. The alternative
/// is that a sample can only belong to one of the compent alone.
///
/// Warning: we don't enforce trait bounds here due to a possible
/// [compiler bug](https://github.com/rust-lang/rust/issues/110136)
///
#[derive(Clone, Debug)]
pub struct Mixture<T, L>
where
    // https://doc.rust-lang.org/nomicon/hrtb.html -- include in docs about GAT
    // T: for<'a, 'b> Mixables<Likelihood = L::Likelihood, DataIn<'a> = L::DataIn<'a>>
    //     + Probabilistic<T>,
    T: Parametrizable<Likelihood = L::Likelihood>,
    // for<'a> <T as Mixables>::DataIn<'a>: Into<L::DataIn<'a>>,
    L: Parametrizable + Latent<L>,
{
    pub mixables: T,
    pub latent: L,
}

impl<T, L> Mixture<T, L>
where
    // https://doc.rust-lang.org/nomicon/hrtb.html -- include in docs about GAT
    // T: for<'a> Mixables<Likelihood = L::Likelihood, DataIn<'a> = L::DataIn<'a>>
    //     + Probabilistic<T>,
    T: Parametrizable<Likelihood = L::Likelihood>,
    // for<'a> <T as Mixables>::DataIn<'a>: Into<L::DataIn<'a>>,
    L: Parametrizable + Latent<L>,
{
    pub fn new(mixables: T, latent: L) -> Self {
        Mixture {
            latent: latent,
            mixables: mixables,
        }
    }
}

impl<T, L> Parametrizable for Mixture<T, L>
where
    T: for<'a> Parametrizable<Likelihood = L::Likelihood, DataIn<'a> = L::DataIn<'a>>
        + Mixable<T>,
    L: Parametrizable + Latent<L>,
{
    type SufficientStatistics = (L::SufficientStatistics, T::SufficientStatistics);

    type Likelihood = T::Likelihood;

    type DataIn<'a> = T::DataIn<'a>;

    type DataOut = T::DataOut;

    fn expect(&self, data: &Self::DataIn<'_>) -> Result<(Self::Likelihood, f64), Error> {
        // Todo compute the second parameter
        Latent::expect(&self.latent, data, &self.mixables.expect(data)?.0)

        // Ok(L::join(
        //     &self.latent.expect(data.into())?.0,
        //     ,
        // )?)
    }

    fn compute(
        &self,
        data: &Self::DataIn<'_>,
        responsibilities: &Self::Likelihood,
    ) -> Result<Self::SufficientStatistics, Error> {
        Ok((
            self.latent.compute(&data, responsibilities)?,
            self.mixables.compute(&data, responsibilities)?,
        ))
    }

    fn maximize(
        &mut self,
        sufficient_statistics: &Self::SufficientStatistics,
    ) -> Result<(), Error> {
        self.latent.maximize(&sufficient_statistics.0)?;
        self.mixables.maximize(&sufficient_statistics.1)?;
        Ok(())
    }

    /// Prediction can be classification or regression depending on the implementation.
    fn predict(&self, data: &Self::DataIn<'_>) -> Result<Self::DataOut, Error> {
        // TODO not tested
        let likelihood = Parametrizable::expect(&self.latent, data)?.0;
        Mixable::predict(&self.mixables, likelihood, data)
    }

    fn update(
        &mut self,
        sufficient_statistics: &Self::SufficientStatistics,
        weight: f64,
    ) -> Result<(), Error> {
        self.latent.update(&sufficient_statistics.0, weight)?;
        self.mixables.update(&sufficient_statistics.1, weight)?;
        Ok(())
    }

    fn merge(
        sufficient_statistics: &[&Self::SufficientStatistics],
        weights: &[f64],
    ) -> Result<Self::SufficientStatistics, Error> {
        // boah.
        let a: Vec<_> = sufficient_statistics.iter().map(|x| &x.0).collect();
        let b: Vec<_> = sufficient_statistics.iter().map(|x| &x.1).collect();

        Ok((L::merge(&a[..], weights)?, T::merge(&b[..], weights)?))
    }

    fn expect_rand(&self, data: &Self::DataIn<'_>, k: usize) -> Result<Self::Likelihood, Error> {
        self.latent.expect_rand(data, k)
    }
}

#[cfg(all(test, feature = "ndarray"))]
mod tests {
    use super::*;
    use crate::backend::ndarray::{
        finite::Finite,
        gaussian::Gaussian,
        utils::{generate_random_expections, generate_samples},
    };
    use tracing::info;
    use tracing_test::traced_test;

    #[traced_test]
    #[test]
    fn em_step() {
        let k = 3;
        // let (data, responsibilities, _means, _covariancess) = generate_samples(30, k, 2);
        // info!(%data);
        // info!(%responsibilities);
        let (data, _, means, _covariances) = generate_samples(30000, k, 2);

        info!(%means);

        let gaussian = Gaussian::new();
        let categorial = Finite::new(None);
        let mut mixture = Mixture {
            mixables: gaussian,
            latent: categorial,
        };

        let mut likelihood: f64;
        let mut responsibilities = generate_random_expections(&data.view(), k).unwrap();
        for _ in 1..20 {
            let stat = mixture.compute(&data.view(), &responsibilities).unwrap();
            mixture.maximize(&stat).unwrap();
            // info!("maximized");
            info!(%mixture.mixables.means);
            info!(%mixture.latent.pmf);

            (responsibilities, likelihood) = mixture.expect(&data.view()).unwrap();
            info!(%likelihood);
        }

        info!(%means);

        // println!("{:?}", result)
    }
}
