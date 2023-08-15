//! <div>
//! <img src="../cerebral.svg" width="800" />
//! </div>
//!
//! ## About
//!
//! This crate provides a library for creating highly customizable
//! self-organizing networks.
//!
//!  ## Example
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
//!
//! ## (Naming) convenctions
//!
//! * traits: Start with capital letter and are adjectives
//! * structs: Start with capital letter and are substantives
//!
//! * All aspects of a model (update rules, training, etc.) are
//!   represented by a struct that holds it parameters and provides
//!   an implementation for the respective trait
//!
//!
//! ## Glossary / Synonyms / Definitions
//!
//!
//! * *Stimulus* (pattern, data point, feature), see [neural::Neural], [neural::NeuralLayer]
//!
//!     Stimuli are data points that trigger a response from the neurons
//!     of the self organizing network. In silico, this is simply a data
//!     point from a data set.
//!
//! * *codebook* (neural weights / weight vector, feature, tuning), see [neural::Neural], [neural::NeuralLayer]
//!
//!     Each neuron has a stimulus that it reacts stronger to than all other neurons. This
//!     pattern is sometimes called weight (vector).
//!     One can say that a neuron is tuned to its weight vector
//!     All weights together form a matrix
//!     which is called codebook.
//!
//! * *lateral space* (neural coordinate/space/lattice, latent space, hidden space),
//!     see [topological::Topological], [topological::CartesianTopology],  [neural::Neural], [neural::NeuralLayer]
//!
//!     In a simplified reality, each neuron has a 2D coordinate marking its physcal
//!     position on the cortex (the neural space). In a self-organizing network, the
//!     topology and dimension of this space plays an important role
//!
//! * *Response*, see [responsive::Responsive], [responsive::CartesianResponsiveness]
//!
//!     Presented a stimuli, each neuron of a self-organizing network
//!     can produce a response. Within the network, exactly one neuron
//!     can be triggered from an individual stimulus. In vivo, this neuron
//!     suppresses possible responses from other neurons.
//!
//!
//! * *Best-matching unit* (BMU, winning neuron, competive learning), see [responsive::Responsive], [responsive::CartesianResponsiveness]
//!
//!     The neuron the creates the stongest (and in vivo, fastest) response to
//!     a stimulus. Being the fastest and strongest, it is often referred to as the
//!     winning neuron and it suppresses the response of all other neurons. In silico
//!     this is the neuron which neural weights resembles most the input given a
//!     metric.
//!
//!
//! * *Adaptation* (update, update rule, tuning), see [adaptable::Adaptable], [adaptable::KohonenAdaptivity]
//!
//!     The ability of a network to respond and tune to a stimuli. The BMU tunes itself further
//!     to the stimulus but so do its adjacent neighbors but with less intensity diminishing with
//!     the distance in the neural space
//!
//! * *Training* (learning), see [trainable::Trainable], [trainable::IncrementalLearning]
//!
//!     In this context, learning is a process of adapting to multiple stimuli
//!     and over an extended period of time possibly with multiple repetitions (i.e., epochs)
//!
//! * Self-organization, see [selforganizing::Selforganizing], [selforganizing::SelforganizingNetwork]
//!
//!     A trait of a neural network that emerges when competive learning is implemented.
//!     The network starts to map similar input stimuli to adjacent regions in the network
//!     thus mapping the topology of input space onto its own neural lattice.

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

// FIXME remove
// pub mod default {
//     pub use crate::adaptable::KohonenAdaptivity as Adaptivity;
//     pub use crate::responsive::CartesianResponsiveness as Responsiveness;
//     pub use crate::topological::CartesianTopology;
//     pub use crate::trainable::IncrementalLearning;
// }

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
