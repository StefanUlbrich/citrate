//! <div>
//! <img src="../cerebral.svg" width="800" />
//! </div>
//!
//!
//! Naming convenctions
//! * traits: Start with capital letter and are adjectives
//! * structs: Start with capital letter and are substantives
//!
//! ```rust
//! use som_rs::default::*;
//! use som_rs::{NeuralLayer, Neurons, SelfOrganizing};
//!
//! fn main() {
//!     println!("Hello, SOM!");
//!
//!     let seed = 42;
//!
//!     let mut rng = Isaac64Rng::seed_from_u64(seed);
//!
//!     let mut som = NeuralLayer {
//!         neurons: Neurons {
//!             // lateral: Array2::<f64>::zeros((0,0)),
//!             patterns: Array::random_using((100, 3), Uniform::new(0., 10.), &mut rng),
//!             ..Default::default()
//!         },
//!         adaptivity: KohonenAdaptivity {},
//!         topology: CartesianTopology::new((10, 10)),
//!         responsiveness: CartesianResponsiveness {},
//!         training: BatchTraining {
//!             radii: (2.0, 0.2),
//!             rates: (0.7, 0.1),
//!             epochs: 1,
//!         },
//!     };
//!
//!     println!("{}", som.neurons.lateral);
//!
//!     som.init_lateral();
//!     let training = Array::random_using((5000, 2), Uniform::new(0., 9.), &mut rng);
//!     som.train(&training);
//!     som.adapt(&training.row(0), 0.7, 0.7);
//! }
//! ```

pub mod neural;
pub mod selforganizing;

pub mod adaptable;
pub mod responsive;
pub mod topological;
pub mod trainable;

pub use neural::{Neural, NeuralLayer};
pub use selforganizing::{BoxedSelforganizing, Selforganizing, SelforganizingNetwork};

pub use adaptable::{Adaptable, BoxedAdaptable};
pub use responsive::{BoxedResponsive, Responsive};
pub use topological::{BoxedTopological, Topological};
pub use trainable::{BoxedTrainable, Trainable};

pub mod default {
    pub use crate::adaptable::KohonenAdaptivity;
    pub use crate::responsive::CartesianResponsiveness;
    pub use crate::topological::CartesianTopology;
    pub use crate::trainable::IncrementalLearning;
}

// #[cfg(feature = "ndarray")]
pub mod nd_tools;

#[cfg(test)]
mod tests {

    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
