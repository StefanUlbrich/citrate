///! Naming convenctions
/// * traits: Start with capital letter and are adjectives
/// * structs: Start with capital letter and are substantives



use std::fmt::Debug;

pub mod neural_layer;
pub mod cartesian_map;
pub mod classic_adaptivity;
pub mod tuning;

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
trait Neural<V: Debug> {
    fn get_lateral(&self) -> &V;
    fn get_lateral_mut(&mut self) -> &mut V;
    fn set_lateral(&mut self, lateral: V);
    fn get_patterns(&self) -> &V;
    fn get_patterns_mut(&mut self) -> &mut V;
    fn set_patterns(&mut self, patterns: V);

}

/// Interface to all self-organizing neural layers in this crate
trait SelfOrganizing<V: Debug> {
    // Associated to topology

    /// Init the lateral connections according to network type
    fn init_lateral(&self);

    /// Get the distance/connection between a selected neuron
    /// and the rest of the layer
    fn get_lateral_distance(&mut self, index: usize) -> V;

    // Associated to the feature space

    /// Get the best matching neuron given a pattern
    fn get_best_matching(&self) -> usize;

    // Associated to adaptivity (Single Datapoints)
    // Ownership has to be transferred.
    // The object needs to be partially deconstructed to
    // grant write access to data and the adaptivity data structure
    // returns ownership.
    // The doubtfully best alternative is making adaptivity and training
    // Copyable

    /// Adapt the layer to an input pattern. Note this consumes
    /// the current later and returns a new created (zero-copy)
    fn adapt(self, data: &V) -> Self;

    // Train a layer given a training set
    fn train(self, data: &V) -> Self;
}


/// Interface for structures encapsulating algorithms for self-organization
trait Adaptable<V: Debug> {
    fn adapt<D,F>(&mut self, data: &mut D, feature: &mut F)
    where
        F: Tunable<V>,
        D: Neural<V>;

}

/// Interface for structures encapsulating algorithms for training from data sets
trait Trainable<V: Debug> {
    fn train<D, A, F>(&mut self, data: &mut D, adaptation: &mut A, feature: &mut F)
        where
            D: Neural<V>,
            F: Tunable<V>,
            A: Adaptable<V>;
}

/// Interface for structures encapsulating representations of network layer topologies.
trait Topological<V: Debug> {
    fn get_lateral_connections<T>(&mut self, nn: &T) -> f64
    where
        T: SelfOrganizing<V>;
    fn init_lateral<T>(&self, nn: &T)
    where
        T: SelfOrganizing<V> + Neural<V>;
}

// Tunable?
/// Interface for structures encapsulating representations input patterns. See
/// [neural tuning](https://en.wikipedia.org/wiki/Neuronal_tuning)
trait Tunable<V: Debug> {
    fn get_best_matching<T>(&self, nn: &T) -> usize
    where
        T: SelfOrganizing<V> + Neural<V>;
}


/// Data for the neurons of a layer
struct Neurons<V: Debug> {
    /// Lateral layer. Can be coordinates or connections (depending on method)
    lateral: V,
    /// Tuned Patterns the neurons
    patterns: V,
}

/// Composable representation of a neural layer
/// capable of self-organization
struct NeuralLayer<V, A, T, F, B>
where
    V: Debug,
    A: Adaptable<V>,
    T: Topological<V>,
    F: Tunable<V>,
    B: Trainable<V>,
    // B: Trainable<V> + Copy,
{
    /// needs to be nested to share it with the algorithms
    neurons: Neurons<V>,
    /// Algorithm for adaptivity
    adaptivity: A,
    /// Algorithm related to topology
    topology: T,
    /// Algorithm to feature pattern matching
    tuning: F,
    /// Algorithm related to batch processing
    training: B,
}






struct ToroidTopology {}
impl<V> Topological<V> for ToroidTopology
where
    V: Debug,
{
    fn init_lateral<T>(&self, nn: &T)
    where
        T: SelfOrganizing<V>+ Neural<V>,
    {
        println!("ToroidTopology {:?}", nn.get_lateral());
    }

    fn get_lateral_connections<T>(&mut self, nn: &T) -> f64
    where
        T: SelfOrganizing<V>,
    {
        42.0
    }
}







// #[derive(Copy,Clone)]
struct BatchTraining {
    start_rate: f64,
    end_rate: f64,
}
impl Trainable<f64> for BatchTraining {
    fn train<D, A, F>(&mut self, data: &mut D, adaptation: &mut A, feature: &mut F)
        where
            D: Neural<f64>,
            F: Tunable<f64>,
            A: Adaptable<f64>
    {
        println!("Batch");
        // (nn.get_lateral(), nn.get_patterns())
    }
}


#[test]
fn main() {
    use classic_adaptivity::SmoothAdaptivity;
    use cartesian_map::simple::CartesianTopology;
    use tuning::CartesianFeature;

    let mut nn = NeuralLayer {
        neurons: Neurons{
            lateral: 3.0,
            patterns: 4.0,
        },
        adaptivity: SmoothAdaptivity {},
        topology: CartesianTopology {},
        tuning: CartesianFeature {},
        training: BatchTraining {
            start_rate: 0.9,
            end_rate: 0.1,
        },
    };

    nn = nn.adapt(&2.0);
    println!("{}", nn.neurons.lateral);
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
