pub mod categorical;
pub mod gaussian;
pub mod kmeans;
pub mod linear;
pub mod som;

use crate::{Initializable, Mixables, MixtureModel};

use ndarray_rand::{rand, rand::prelude::*, rand_distr::Dirichlet};

// Todo: move outside of the backend! .. that is put the standard implementation of fit to the EM trait
// Problem is the initialized, this is a matrix.
// We can make an fn initialize_responsibility<T>

impl<T> Initializable<T::LogLikelihood> for MixtureModel<T>
where
    T: Mixables,
{
    fn initialize(&mut self, responsibilities: Option<T::LogLikelihood>) {
        if let Some(r) = responsibilities {
            self.initialization = Some(r);
        } else {
            let dirichlet = Dirichlet::new(&vec![1.0; self.n_components]).unwrap();

            let responsibilities = dirichlet.sample(&mut rand::thread_rng());
            // Standard.sample_iter(&mut rng).take(16).collect();
        }
    }
}
