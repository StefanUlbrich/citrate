use crate::{errors, ExpectationMaximizing, Mixables, Initializable};


/// The basis struct to use for models
pub struct MixtureModel<T>
where
    T: Mixables,
{
    pub mixable: T,
    pub n_components: usize,
    pub max_iterations: usize,
    pub n_init: usize,
    pub incremental: bool,
    pub incremental_weight: f64,
    pub tol: f64,
    last_sufficient_statistics: Option<T::SufficientStatistics>,
    pub initialization: Option<T::LogLikelihood>,
    pub info: MixtureInfo,
}

pub struct MixtureInfo {
    pub fitted: bool,
    pub converged: bool,
    pub n_iterations: usize,
    pub likelihood: f64,
    pub initialized: bool,
}

impl<T> MixtureModel<T>
where
    T: Mixables,
{
    fn new(
        mixable: T,
        n_components: usize,
        max_iterations: usize,
        n_init: usize,
        incremental: bool,
    ) -> MixtureModel<T> {
        MixtureModel {
            mixable,
            n_components,
            max_iterations,
            n_init,
            incremental,
            incremental_weight: 0.8,
            tol: 1e-6,
            last_sufficient_statistics: None,
            initialization: None,
            info: MixtureInfo {
                fitted: false,
                converged: false,
                n_iterations: 0,
                likelihood: f64::NAN,
                initialized: false,
            },
        }
    }

    fn initialize_manually(&mut self, responsibilities: T::LogLikelihood) {
        self.info.initialized = true;
        self.initialization = Some(responsibilities);
    }
}

impl<T> ExpectationMaximizing for MixtureModel<T>
where
    T: Mixables,
{
    type DataIn<'a> = T::DataIn<'a>;
    type DataOut = T::DataOut;

    fn fit(&mut self, data: Self::DataIn<'_>) -> Result<(), errors::Error> {
        if !self.info.fitted || !self.incremental {
            if (self.n_init > 0 || self.info.fitted) && self.info.initialized {
                return Err(errors::Error::InvalidTrainingConfiguration {
                    n_init: self.n_init,
                    incremental: self.incremental,
                    initialized: self.incremental,
                });
            }

            // multiple initializations can be parallelized when using iterator funtions
            let best = (0..self.n_init)
                .map(|_| {
                    if !self.info.initialized && !self.info.fitted {
                        self.initialize(None);
                    }

                    let mut converged = true;
                    let mut n_iterations = 0;

                    let mut last_likelihood = f64::NEG_INFINITY;

                    let mut sufficient_statistics =
                        self.mixable.compute(self.initialization.as_ref().unwrap());
                    self.mixable.maximize(&sufficient_statistics);

                    // the inner loop cannot be parallelized
                    for i in 0..self.max_iterations {
                        let (responsibilities, likelihood) = self.mixable.expect(&data);
                        sufficient_statistics = self.mixable.compute(&responsibilities);
                        self.mixable.maximize(&sufficient_statistics);
                        if f64::abs(likelihood - last_likelihood) > self.tol {
                            converged = true;
                            n_iterations = i;
                            break;
                        }
                        last_likelihood = likelihood;
                    }

                    if converged {
                        (sufficient_statistics, true, n_iterations, last_likelihood)
                    } else {
                        (
                            sufficient_statistics,
                            false,
                            n_iterations,
                            f64::NEG_INFINITY,
                        )
                    }
                })
                .max_by(|a, b| a.3.total_cmp(&b.3))
                .unwrap();

            // from the sufficient statistics we can restore the winning model by simply maximizing
            self.mixable.maximize(&best.0);
            self.info.converged = best.1;
            self.info.n_iterations = best.2;
            self.info.likelihood = best.3;
        } else {
            // incremental learning

            // Guess I will have to read the paper
            let (responsibilities, _) = self.mixable.expect(&data);
            let mut sufficient_statistics = self.mixable.compute(&responsibilities);
            sufficient_statistics = T::merge(
                &[
                    &self
                        .last_sufficient_statistics
                        .as_ref()
                        .expect("Model has not been trained before"),
                    &sufficient_statistics,
                ],
                &[1.0 - self.incremental_weight, self.incremental_weight],
            );
            // todo!
            // self.batch()
        }
        Ok(())
    }

    fn predict(&self, data: &Self::DataIn<'_>) -> T::DataOut {
        let (responsibilities, likelihood) = self.mixable.expect(&data);
        // self.mixable.predict(&responsibilities, data)
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
