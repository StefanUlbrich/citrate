use crate::{AvgLLH, Error, Learning, Parametrizable};
use rayon::prelude::*;

use tracing::info;

/// The basis struct to use for models
#[derive(Debug)]
pub struct Model<T>
where
    T: Parametrizable,
{
    pub parametrizable: T,
    pub n_components: usize,
    pub max_iterations: usize,
    pub n_init: usize,
    pub incremental: bool,
    pub incremental_weight: f64,
    pub tol: f64,
    // last_sufficient_statistics: Option<T::SufficientStatistics>,
    // pub initialization: Option<T::Likelihood>,
    pub info: ModelInfo,
}

#[derive(Debug)]
pub struct ModelInfo {
    pub fitted: bool,
    pub converged: bool,
    pub n_iterations: usize,
    pub likelihood: AvgLLH,
    // pub initialized: bool,
}

impl<T> Model<T>
where
    T: Parametrizable + Sync,
{
    pub fn new(
        parametrizable: T,
        n_components: usize,
        max_iterations: usize,
        n_init: usize,
        incremental: bool,
    ) -> Model<T> {
        Model {
            parametrizable,
            n_components,
            max_iterations,
            n_init,
            incremental,
            incremental_weight: 0.8,
            tol: 1e-6,
            // last_sufficient_statistics: None,
            // initialization: None,
            info: ModelInfo {
                fitted: false,
                converged: false,
                n_iterations: 0,
                likelihood: AvgLLH(f64::NAN),
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
    likelihood: AvgLLH,
}

impl<T: Parametrizable> Intermediate<T> {
    fn new(
        sufficient_statistics: T::SufficientStatistics,
        converged: bool,
        n_iterations: usize,
        likelihood: AvgLLH,
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
    /// Single EM iteration. Consumes a copy of a parametrizable
    fn single_fit(
        &self,
        mut parametrizable: T,
        data: &T::DataIn<'_>,
    ) -> Result<Intermediate<T>, Error> {
        // If the model has not been fitted yet, do a random initialization

        // use random sufficient statistics for variable initialization
        let mut sufficient_statistics = self.parametrizable.compute(
            &data,
            &parametrizable.expect_rand(&data, self.n_components)?,
        )?;

        // .. and optional model initialization
        if !self.info.fitted {
            parametrizable.maximize(&sufficient_statistics)?;
        }

        let mut n_iterations = 0;
        let mut converged = false;
        let mut last_likelihood: AvgLLH = AvgLLH(f64::NEG_INFINITY);

        for i in 0..self.max_iterations {
            let (responsibilities, likelihood) = parametrizable.expect(&data)?;
            sufficient_statistics = parametrizable.compute(&data, &responsibilities)?;
            parametrizable.maximize(&sufficient_statistics)?;

            let diff = f64::abs(likelihood.0 - last_likelihood.0);
            last_likelihood = likelihood;

            info!("{}: {}", i, diff);
            if diff < self.tol {
                converged = true;
                n_iterations = i;
                break;
            }
        }

        Ok(Intermediate::new(
            sufficient_statistics,
            converged,
            n_iterations,
            last_likelihood,
        ))
    }
}

impl<T> Learning for Model<T>
where
    T: Parametrizable + Sync + Clone + Send,
{
    type DataIn<'a> = T::DataIn<'a>;
    type DataOut = T::DataOut;

    fn fit(&mut self, data: &Self::DataIn<'_>) -> Result<(), Error> {
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
                .map(|_| self.parametrizable.clone())
                .map(|x| self.single_fit(x, &data))
                .collect();
            // results variable required to determine the lifetime of references returned by `max_by`
            let results = results?;

            info!(
                "Likelihood of all runs {:?}",
                results.iter().map(|i| i.likelihood.0).collect::<Vec<_>>()
            );

            let best = results
                .iter()
                .max_by(|a, b| a.likelihood.0.total_cmp(&b.likelihood.0))
                .unwrap();

            info!("Best {:?}", best.likelihood.0);
            // restore winning model from the sufficient statistics
            // Additional step,  maybe a bit more expensive but elegant
            self.parametrizable.maximize(&best.sufficient_statistics)?;
            self.info.converged = best.converged;
            self.info.n_iterations = best.n_iterations;
            self.info.likelihood = best.likelihood.clone();
        } else {
            // incremental learning

            return Err(Error::NotImplemented);

            // Todo: only a draft yet
            // if !self.info.fitted {
            //     let sufficient_statistics = self
            //         .mixable
            //         .compute(&data, &self.mixable.expect_rand(&data, self.n_components)?)?;
            //     self.mixable.maximize(&sufficient_statistics)?;
            //     self.info.fitted = true
            // }

            // panic!();
            // // Guess I will have to read the paper
            // let (responsibilities, _) = self.mixable.expect(&data)?;
            // let mut sufficient_statistics = self.mixable.compute(&data, &responsibilities)?;
            // sufficient_statistics = T::merge(
            //     &[
            //         &self
            //             .last_sufficient_statistics
            //             .as_ref()
            //             .expect("Model has not been trained before"),
            //         &sufficient_statistics,
            //     ],
            //     &[1.0 - self.incremental_weight, self.incremental_weight],
            // )?;
            // // self.batch()
        }
        Ok(())
    }

    fn predict(&self, _data: &Self::DataIn<'_>) -> Result<Self::DataOut, Error> {
        // let (responsibilities, likelihood) = self.mixable.expect(&data)?;
        // self.mixable.predict(&responsibilities, data)
        todo!()
    }
}

// TODO: Move to integration tests
#[cfg(all(test, feature = "ndarray"))]
mod tests {
    use crate::backend::ndarray::gaussian::sort_parameters;
    use crate::backend::ndarray::utils::generate_samples;
    use crate::backend::ndarray::{finite::Finite, gaussian::Gaussian};
    use crate::mixture::Mixture;
    use crate::model::Model;
    use crate::Learning;
    use tracing::info;
    use tracing_test::traced_test;

    #[test]
    #[traced_test]
    fn single_gmm_em() {
        let (data, _, _, _covariances) = generate_samples(&[15000, 10000, 50000], 2);

        let gaussian = Gaussian::new();
        let categorial = Finite::new(None);
        let _mixture = Mixture {
            mixables: gaussian,
            latent: categorial,
        };
        // let density = Density::new(gaussian, categorial);

        let gaussian = Gaussian::new();
        let categorial = Finite::new(None);

        let density = Mixture::new(gaussian, categorial);

        let gmm = Model::new(density, 3, 200, 1, false);

        let result = gmm
            .single_fit(gmm.parametrizable.clone(), &data.view())
            .unwrap();

        info!(?result);
        assert!(result.n_iterations < 35);
        assert!(result.converged == true);
    }

    #[test]
    #[traced_test]
    fn test_multi_pass() {
        // Samples must be sorted in decreasing order
        let (data, _, means, _) = generate_samples(&[5000, 10000, 15000], 2);

        let mut model = Model::new(
            Mixture::new(Gaussian::new(), Finite::new(None)),
            3,
            50,
            4,
            false,
        );

        model.fit(&data.view()).unwrap();
        info!(?model.info);

        let (means_sorted, _) = sort_parameters(
            &model.parametrizable.mixables,
            &model.parametrizable.latent.pmf.view(),
        );

        println!("{}", model.parametrizable.mixables.means);
        println!("{}", model.parametrizable.latent.pmf);

        info!(%means);
        // info!(?model.parametrizable.mixables.means),

        info!(%means_sorted);
        info!("{}", &means_sorted - &means);

        assert!(means.abs_diff_eq(&means_sorted, 1e-2));

        // time cargo test -F ndarray -p potpourri test_multi_pass --release
        // 6,17s user 0,60s system 703% cpu 0,962 total

        // cargo test -F ndarray -p potpourri test_multi_pass
        // 270,15s user 0,57s system 804% cpu 33,635 total
    }
}
