use ndarray::{prelude::*, Data};
///! Naming convenctions
/// * traits: Start with capital letter and are adjectives
/// * structs: Start with capital letter and are substantives
use std::fmt::Debug;

pub mod cartesian_map;
pub mod maps;
pub mod neural_layer;

// #[cfg(feature = "ndarray")]
pub mod nd_tools;

pub use crate::{
    cartesian_map::topological::CartesianTopology,
    maps::{adaptable::ClassicAdaptivity, trainable::BatchTraining, tunable::CartesianFeature},
};

/// Provides access to the neurons of a neural network.
/// The data is separated in
/// * lateral connections: Depending on the network type,
///   can be absolute coordinates (SOM) or connection
///   distances (Growing neural gases) for instance.
/// * tuning patterns: List of patterns of the feature space
///   each individual neural is tuned to
/// Provides read-only, modifying and consuming access.
pub trait Neural {
    fn get_lateral(&self) -> &Array2<f64>;
    fn get_lateral_mut(&mut self) -> &mut Array2<f64>;
    fn set_lateral(&mut self, lateral: Array2<f64>);
    fn get_patterns(&self) -> &Array2<f64>;
    fn get_patterns_mut(&mut self) -> &mut Array2<f64>;
    fn set_patterns(&mut self, patterns: Array2<f64>);
}

/// Interface to all self-organizing neural layers in this crate
pub trait SelfOrganizing {
    // Associated to topology

    /// Init the lateral connections according to network type
    fn init_lateral(&mut self);// -> Self;

    /// Get the distance/connection between a selected neuron
    /// and the rest of the layer
    fn get_lateral_distance(&mut self, index: usize) -> Array2<f64>;

    // Associated to the feature space

    /// Get the best matching neuron given a pattern
    fn get_best_matching<S>(&mut self, pattern: &ArrayBase<S, Ix1>) -> usize
    where
        S: Data<Elem = f64>;

    // Associated to adaptivity (Single Datapoints)
    // Ownership has to be transferred.
    // The object needs to be partially deconstructed to
    // grant write access to data and the adaptivity data structure
    // returns ownership.
    // The doubtfully best alternative is making adaptivity and training
    // Copyable

    /// Adapt the layer to an input pattern. Note this consumes
    /// the current later and returns a new created (zero-copy)
    fn adapt<S>(&mut self, pattern: &ArrayBase<S, Ix1>, influence: f64, rate: f64)
    //-> Self
    where
        S: Data<Elem = f64>;

    // Train a layer given a training set
    fn train<S>(&mut self, patterns: &ArrayBase<S, Ix2>)
    //-> Self
    where
        S: Data<Elem = f64>;
}

/// Interface for structures encapsulating algorithms for self-organization
pub trait Adaptable {
    fn adapt<S, N, T>(
        &mut self,
        neurons: &mut N,
        tuning: &mut T,
        pattern: &ArrayBase<S, Ix1>,
        influence: f64,
        rate: f64,
    )
    //&Self::ArgType)
    where
        T: Competitive,
        N: Neural,
        S: Data<Elem = f64>;
}

/// Interface for structures encapsulating algorithms for training from data sets
pub trait Trainable {
    fn train<S, N, A, F>(
        &mut self,
        neurons: &mut N,
        adaptation: &mut A,
        feature: &mut F,
        patterns: &ArrayBase<S, Ix2>,
    ) where
        N: Neural,
        F: Competitive,
        A: Adaptable,
        S: Data<Elem = f64>;
}

/// Interface for structures encapsulating representations of network layer topologies.
pub trait Topological {
    fn get_lateral_connections<N>(&mut self, data: &N) -> f64
    where
        N: Neural;

    fn init_lateral<N>(&self, neurons: &mut N)
    where
        N: Neural;
}

// Tunable?
/// Interface for structures encapsulating representations input patterns. See
/// [neural tuning](https://en.wikipedia.org/wiki/Neuronal_tuning)
pub trait Competitive {
    // Cannot be specialized in implementation
    // See https://stackoverflow.com/a/53085395/9415551
    // fn get_best_matching<N,P>(&self, neurons: &N, pattern: &P)

    fn get_best_matching<S, N>(&self, neurons: &N, pattern: &ArrayBase<S, Ix1>) -> usize
    where
        N: Neural,
        S: Data<Elem = f64>;
}

/// Data for the neurons of a layer
pub struct Neurons {
    /// Lateral layer. Can be coordinates or connections (depending on method)
    pub lateral: Array2<f64>,
    /// Tuned Patterns the neurons
    pub patterns: Array2<f64>,
}

impl Neurons {
    pub fn new() -> Neurons{
        Neurons{
            lateral: Array2::<f64>::zeros((0,0)),
            patterns: Array2::<f64>::zeros((0,0)),
        }
    }
}

/// Composable representation of a neural layer
/// capable of self-organization
pub struct NeuralLayer<A, T, C, L>
where
    A: Adaptable,
    T: Topological,
    C: Competitive,
    L: Trainable,
    // B: Trainable<D1,D2> + Copy,
{
    /// needs to be nested to share it with the algorithms
    pub neurons: Neurons,
    /// Algorithm for adaptivity
    pub adaptivity: A,
    /// Algorithm related to topology
    pub topology: T,
    /// Algorithm to feature pattern matching
    pub competition: C,
    /// Algorithm related to batch processing
    pub training: L // Box<B>,
}

// struct ToroidTopology {}
// impl<V> Topological<V> for ToroidTopology
// where
//     V: Debug,
// {
//     fn init_lateral<T>(&self, nn: &T)
//     where
//         T: SelfOrganizing<V>+ Neural<V>,
//     {
//         println!("ToroidTopology {:?}", nn.get_lateral());
//     }

//     fn get_lateral_connections<T>(&mut self, nn: &T) -> f64
//     where
//         T: SelfOrganizing<V>,
//     {
//         42.0
//     }
// }

#[test]
fn main() {
    // use maps::{adaptable::ClassicAdaptivity, trainable::BatchTraining};
    // use cartesian_map::{topological::CartesianTopology, tunable::CartesianFeature};

    // let mut nn = NeuralLayer {
    //     neurons: Neurons{
    //         lateral: 3.0,
    //         patterns: 4.0,
    //     },
    //     adaptivity: ClassicAdaptivity {},
    //     topology: CartesianTopology { shape: (10,10)},
    //     tuning: CartesianFeature {},
    //     training: BatchTraining {
    //         start_rate: 0.9,
    //         end_rate: 0.1,
    //     },
    // };

    // nn = nn.adapt(&2.0);
    // println!("{}", nn.neurons.lateral);
}

// #[cfg(test)]
// mod tests {

//     #[test]
//     fn it_works() {
//         let result = 2 + 2;
//         assert_eq!(result, 4);
//     }

//     // fn test_uniform() {
//     //     let a = uniform((2, 3));
//     //     let b = array![
//     //         [0.0, 0.0],
//     //         [0.0, 1.0],
//     //         [1.0, 0.0],
//     //         [1.0, 1.0],
//     //         [2.0, 0.0],
//     //         [2.0, 1.0]
//     //     ];

//     //     assert_eq!(a, b);
//     // }
// }
