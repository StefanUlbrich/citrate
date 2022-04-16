///! Naming convenctions
/// * traits: Start with capital letter and are adjectives
/// * structs: Start with capital letter and are substantives



use std::fmt::Debug;

pub mod neural_layer;
pub mod cartesian_map;
pub mod maps;

#[cfg(feature="ndarray")]
pub mod nd_tools;

/// Provides access to the neurons of a neural network.
/// The data is separated in
/// * lateral connections: Depending on the network type,
///   can be absolute coordinates (SOM) or connection
///   distances (Growing neural gases) for instance.
/// * tuning patterns: List of patterns of the feature space
///   each individual neural is tuned to
/// Provides read-only, modifying and consuming access.
trait Neural<D1: Debug, D2: Debug> {


    fn get_lateral(&self) -> &D1;
    fn get_lateral_mut(&mut self) -> &mut D1;
    fn set_lateral(&mut self, lateral: D1);
    fn get_patterns(&self) -> &D2;
    fn get_patterns_mut(&mut self) -> &mut D2;
    fn set_patterns(&mut self, patterns: D2);

}

/// Interface to all self-organizing neural layers in this crate
trait SelfOrganizing<D1: Debug, D2: Debug> {

    type ArgTypeA;
    type ArgTypeF;
    type ArgTypeB;


    // Associated to topology

    /// Init the lateral connections according to network type
    fn init_lateral(self) -> Self;

    /// Get the distance/connection between a selected neuron
    /// and the rest of the layer
    fn get_lateral_distance(&mut self, index: usize) -> D1;

    // Associated to the feature space

    /// Get the best matching neuron given a pattern
    fn get_best_matching(&self, pattern: &Self::ArgTypeF) -> usize;

    // Associated to adaptivity (Single Datapoints)
    // Ownership has to be transferred.
    // The object needs to be partially deconstructed to
    // grant write access to data and the adaptivity data structure
    // returns ownership.
    // The doubtfully best alternative is making adaptivity and training
    // Copyable

    /// Adapt the layer to an input pattern. Note this consumes
    /// the current later and returns a new created (zero-copy)
    fn adapt<T>(self, pattern: T) -> Self;

    // Train a layer given a training set
    fn train(self, patterns: &Self::ArgTypeB) -> Self;
}


/// Interface for structures encapsulating algorithms for self-organization
trait Adaptable<D1: Debug,D2: Debug> {
    type ArgType;

    fn adapt<N,F,T>(&mut self, neurons: &mut N, feature: &mut F, pattern: T) //&Self::ArgType)
    where
        F: Tunable<D1,D2>,
        N: Neural<D1,D2>;

}

/// Interface for structures encapsulating algorithms for training from data sets
trait Trainable<D1: Debug,D2: Debug> {
    type ArgType;

    fn train<N, A, F>(&mut self, neurons: &mut N, adaptation: &mut A, feature: &mut F, data: &Self::ArgType)
        where
            N: Neural<D1,D2>,
            F: Tunable<D1,D2>,
            A: Adaptable<D1,D2>;
}

/// Interface for structures encapsulating representations of network layer topologies.
trait Topological<D1: Debug,D2: Debug> {
    fn get_lateral_connections<D>(&mut self, data: &D) -> f64
    where
        D: Neural<D1,D2>;

    fn init_lateral<N>(&self, neurons: &mut N)
    where
        N: Neural<D1,D2>;
}

// Tunable?
/// Interface for structures encapsulating representations input patterns. See
/// [neural tuning](https://en.wikipedia.org/wiki/Neuronal_tuning)
trait Tunable<D1: Debug,D2: Debug> {
    type ArgType;

    // Cannot be specialized in implementation
    // See https://stackoverflow.com/a/53085395/9415551
    // fn get_best_matching<N,P>(&self, neurons: &N, pattern: &P)

    fn get_best_matching<N>(&self, neurons: &N, pattern: &Self::ArgType) -> usize
    where
        N: Neural<D1,D2>;
}


/// Data for the neurons of a layer
struct Neurons<D1: Debug,D2: Debug> {
    /// Lateral layer. Can be coordinates or connections (depending on method)
    lateral: D1,
    /// Tuned Patterns the neurons
    patterns: D2,
}

/// Composable representation of a neural layer
/// capable of self-organization
struct NeuralLayer<D1, D2, A, T, F, B>
where
    D1: Debug,
    D2: Debug,
    A: Adaptable<D1,D2>,
    T: Topological<D1,D2>,
    F: Tunable<D1,D2>,
    B: Trainable<D1,D2>,
    // B: Trainable<D1,D2> + Copy,
{
    /// needs to be nested to share it with the algorithms
    neurons: Neurons<D1,D2>,
    /// Algorithm for adaptivity
    adaptivity: A,
    /// Algorithm related to topology
    topology: T,
    /// Algorithm to feature pattern matching
    tuning: F,
    /// Algorithm related to batch processing
    training: B,
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
