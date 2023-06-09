use crate::{Error, Learning, Parametrizable};
use rayon::prelude::*;

use tracing::{debug, info};

/// The basis struct to use for models
pub struct Model<T>
where
    T: Parametrizable,
{
    pub mixable: T,
    pub n_components: usize,
    pub max_iterations: usize,
    pub n_init: usize,
    pub incremental: bool,
    pub incremental_weight: f64,
    pub tol: f64,
    last_sufficient_statistics: Option<T::SufficientStatistics>,
    // pub initialization: Option<T::LogLikelihood>,
    pub info: ModelInfo,
}

pub struct ModelInfo {
    pub fitted: bool,
    pub converged: bool,
    pub n_iterations: usize,
    pub likelihood: f64,
    // pub initialized: bool,
}

impl<T> Model<T>
where
    T: Parametrizable + Sync,
{
    pub fn new(
        mixable: T,
        n_components: usize,
        max_iterations: usize,
        n_init: usize,
        incremental: bool,
    ) -> Model<T> {
        Model {
            mixable,
            n_components,
            max_iterations,
            n_init,
            incremental,
            incremental_weight: 0.8,
            tol: 1e-6,
            last_sufficient_statistics: None,
            // initialization: None,
            info: ModelInfo {
                fitted: false,
                converged: false,
                n_iterations: 0,
                likelihood: f64::NAN,
                // initialized: false,
            },
        }
    }
}

/// Intermediate result from a single EM training (better than just using tuples)
#[derive(Debug)]
struct Intermediate<T: Parametrizable> {
    sufficient_statistics: T::SufficientStatistics,
    converged: bool,
    n_iterations: usize,
    likelihood: f64,
}

impl<T: Parametrizable> Intermediate<T> {
    fn new(
        sufficient_statistics: T::SufficientStatistics,
        converged: bool,
        n_iterations: usize,
        likelihood: f64,
    ) -> Self {
        Intermediate {
            sufficient_statistics,
            converged,
            n_iterations,
            likelihood,
        }
    }
}

impl<T> Model<T>
where
    T: Parametrizable + Sync,
{
    /// Single EM iteration.
    fn single_fit(&self, mut mixable: T, data: &T::DataIn<'_>) -> Result<Intermediate<T>, Error> {
        // If the model has not been fitted yet, do a random initialization

        // use random sufficient statistics for variable initialization
        let mut sufficient_statistics = self
            .mixable
            .compute(&data, &mixable.expect_rand(&data, self.n_components)?)?;

        // .. and optional model initialization
        if !self.info.fitted {
            mixable.maximize(&sufficient_statistics)?;
        }

        let mut n_iterations = 0;
        let mut converged = false;
        let mut last_likelihood = f64::NEG_INFINITY;

        for i in 0..self.max_iterations {
            info!(%i);
            let (responsibilities, likelihood) = mixable.expect(&data)?;
            sufficient_statistics = mixable.compute(&data, &responsibilities)?;
            mixable.maximize(&sufficient_statistics)?;
            if f64::abs(likelihood - last_likelihood) > self.tol {
                converged = true;
                n_iterations = i;
                break;
            }
            last_likelihood = likelihood;
        }

        Ok(Intermediate::new(
            sufficient_statistics,
            converged,
            n_iterations,
            f64::NEG_INFINITY,
        ))
    }
}

impl<T> Learning for Model<T>
where
    T: Parametrizable + Sync + Clone + Send,
{
    type DataIn<'a> = T::DataIn<'a>;
    type DataOut = T::DataOut;

    fn fit(&mut self, data: Self::DataIn<'_>) -> Result<(), Error> {
        if !self.incremental {
            if self.n_init > 0 && self.info.fitted {
                return Err(Error::ParameterError {
                    n_init: self.n_init,
                    fitted: self.info.fitted,
                });
            }

            // https://stackoverflow.com/a/36371890
            let results: Result<Vec<_>, _> = (0..self.n_init)
                .into_par_iter()
                .map(|_| self.mixable.clone())
                .map(|x| self.single_fit(x, &data))
                .collect();
            // results variable required to determine the lifetime of references returned by `max_by`
            let results = results?;

            let best = results
                .iter()
                .max_by(|a, b| a.likelihood.total_cmp(&b.likelihood))
                .unwrap();

            // restore winning model from the sufficient statistics
            // Additional step,  maybe a bit more expensive but elegant
            self.mixable.maximize(&best.sufficient_statistics)?;
            self.info.converged = best.converged;
            self.info.n_iterations = best.n_iterations;
            self.info.likelihood = best.likelihood;
        } else {
            // incremental learning

            if !self.info.fitted {
                let sufficient_statistics = self
                    .mixable
                    .compute(&data, &self.mixable.expect_rand(&data, self.n_components)?)?;
                self.mixable.maximize(&sufficient_statistics)?;
                self.info.fitted = true
            }

            panic!();
            // Guess I will have to read the paper
            let (responsibilities, _) = self.mixable.expect(&data)?;
            let mut sufficient_statistics = self.mixable.compute(&data, &responsibilities)?;
            sufficient_statistics = T::merge(
                &[
                    &self
                        .last_sufficient_statistics
                        .as_ref()
                        .expect("Model has not been trained before"),
                    &sufficient_statistics,
                ],
                &[1.0 - self.incremental_weight, self.incremental_weight],
            )?;
            // todo!
            // self.batch()
        }
        Ok(())
    }

    fn predict(&self, data: &Self::DataIn<'_>) -> Result<Self::DataOut, Error> {
        let (responsibilities, likelihood) = self.mixable.expect(&data)?;
        // self.mixable.predict(&responsibilities, data)
        todo!()
    }
}

// TODO: Move to integration tests
#[cfg(all(test, feature = "ndarray"))]
mod tests {
    use super::*;
    use crate::backend::ndarray::utils::generate_samples;
    use crate::backend::ndarray::{categorical::Finite, gaussian::Gaussian};
    use crate::mixture::Mixture;
    use crate::model::Model;
    use tracing_test::traced_test;

    #[test]
    #[traced_test]
    fn single_gmm_em() {
        let (data, _, _, covariances) = generate_samples(30000, 3, 2);

        let gaussian = Gaussian::new();
        let categorial = Finite::new(None);
        let mixture = Mixture {
            mixables: gaussian,
            latent: categorial,
        };
        // let density = Density::new(gaussian, categorial);

        let gaussian = Gaussian::new();
        let categorial = Finite::new(None);

        let density = Mixture::new(gaussian, categorial);

        let gmm = Model::new(density, 3, 200, 1, false);

        let result = gmm.single_fit(gmm.mixable.clone(), &data.view()).unwrap();

        println!("{:?}", result)
    }
}
