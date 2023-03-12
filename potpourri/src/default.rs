use ndarray::prelude::*;
use ndarray_rand::{rand_distr::Dirichlet, rand, rand::prelude::*};

use super::{MixtureType, ExpectationMaximization};

pub struct MixtureModel<T>
where
    T: MixtureType,
{
    pub n_components: i32,
    pub mixture: T,
    pub n_init: i32,
    pub n_iterations: i32,
    pub weights: Array1<f64>,
    pub status: MixtureModelStatus,
}

pub struct MixtureModelStatus {
    pub is_fitted: bool,
    pub converged: bool,
    pub likelihood: f64
}
impl MixtureModelStatus {
    fn new() -> MixtureModelStatus {
        MixtureModelStatus { is_fitted: false, converged: false, likelihood : f64::NAN }
    }
}

impl<T> MixtureModel<T>
where
    T: MixtureType,
{

    pub fn new(n_components: i32, mut mixture: T, prior: Option<f64>, n_init: i32, n_iterations: i32) -> MixtureModel<T> {
        // let mut weighting = Categorical::new(1, prior);
        // weighting.initialize(n_components);
        mixture.initialize(n_components);
        let weights = Array1::<f64>::zeros(0);

        MixtureModel {
            n_components,
            mixture,
            n_init,
            n_iterations,
            weights,
            status: MixtureModelStatus::new()
        }
    }

    pub fn store(
        &self,
    ) -> (
        &<Categorical as MixtureType>::SufficientStatistics,
        &T::SufficientStatistics,
    ) {
        (self.weighting.store(), self.mixture.store())
    }
    pub fn restore(
        &mut self,
        sufficient_statistics: (
            <Categorical as MixtureType>::SufficientStatistics,
            T::SufficientStatistics,
        ),
    ) {
        self.weighting.restore(sufficient_statistics.0);
        self.mixture.restore(sufficient_statistics.1);
    }

    fn expect(){}

    fn maximize(&mut self, responsibilities: &T::DataIn<'_>){
        responsibilities; // Damn this needs to be an ndarray already! Unless we move this into a `latent` trait
    }
}

impl<T> ExpectationMaximization<T> for MixtureModel<T> where
T: MixtureType {

    type SufficientStatistics = Array1<f64>;

    fn fit(&mut self, data: T::DataIn<'_>) {
        todo!()
    }

    fn predict(&self, data: &T::DataIn<'_>) -> T::DataOut{
        todo!()
    }

    fn initialize(&mut self) {

        let dirichlet = Dirichlet::new(&vec![1.0; self.n_components] ).unwrap();

        let responsibilities = dirichlet.sample(&mut rand::thread_rng());
        // Standard.sample_iter(&mut rng).take(16).collect();
    }


}