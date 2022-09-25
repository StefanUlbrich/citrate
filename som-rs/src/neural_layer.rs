use ndarray::{prelude::*, Data};

use crate::{Adaptable, Neural, Neurons, Responsive, Topological, Trainable};

/// Public trait that defines the concept of self organization
pub trait SelfOrganizing {
    // Associated to topology

    /// Init the lateral connections according to network type
    fn init_lateral(&mut self); // -> Self;

    /// Get the distance/connection between a selected neuron
    /// and the rest of the layer
    fn get_lateral_distance(&mut self, index: usize) -> Array2<f64>;

    // Associated to the feature space

    /// Get the best matching neuron given a pattern
    fn get_best_matching(&mut self, pattern: &ArrayView1<f64>) -> usize;

    // Associated to adaptivity (Single Datapoints)
    // Ownership has to be transferred.
    // The object needs to be partially deconstructed to
    // grant write access to data and the adaptivity data structure
    // returns ownership.
    // The doubtfully best alternative is making adaptivity and training
    // Copyable

    // fn get_best_matching<S>(&mut self, pattern: &ArrayBase<S, Ix1>) -> usize
    // where
    //     S: Data<Elem = f64>;
    // fn adapt<S>(&mut self, pattern: &ArrayBase<S, Ix1>, influence: f64, rate: f64)
    // where
    //     S: Data<Elem = f64>;
    // fn train<S>(&mut self, patterns: &ArrayBase<S, Ix2>)
    // where
    //     S: Data<Elem = f64>;

    /// Adapt the layer to an input pattern. Note this consumes
    /// the current later and returns a new created (zero-copy)
    fn adapt(&mut self, pattern: &ArrayView1<f64>, influence: f64, rate: f64);
    //-> Self

    // Train a layer given a training set
    fn train(&mut self, patterns: &ArrayView2<f64>);
    //-> Self
}

/// Struct that implements structural composition
pub struct NeuralLayer<A, T, R, L>
where
    A: Adaptable<Neurons, R>,
    T: Topological<Neurons>,
    R: Responsive<Neurons>,
    L: Trainable<Neurons, A, R>,
    // B: Trainable<D1,D2> + Copy,
{
    /// needs to be nested to share it with the algorithms
    pub neurons: Neurons,
    /// Algorithm for adaptivity
    pub adaptivity: A,
    /// Algorithm related to topology
    pub topology: T,
    /// Algorithm to feature pattern matching and lateral inhibition
    pub responsiveness: R,
    /// Algorithm related to batch processing
    pub training: L, // Box<B>,
}

impl<A, T, R, B> SelfOrganizing for NeuralLayer<A, T, R, B>
where
    A: Adaptable<Neurons, R>,
    T: Topological<Neurons>,
    R: Responsive<Neurons>,
    B: Trainable<Neurons, A, R>,
{
    fn init_lateral(&mut self) //-> Self
    {
        self.topology.init_lateral(&mut self.neurons);
        // self
    }

    fn get_lateral_distance(&mut self, index: usize) -> Array2<f64> {
        todo!()
    }

    fn get_best_matching(&mut self, pattern: &ArrayView1<f64>) -> usize {
        self.responsiveness.get_best_matching(&self.neurons, pattern)
    }

    fn adapt(&mut self, pattern: &ArrayView1<f64>, influence: f64, rate: f64) {
        self.adaptivity.adapt(
            &mut self.neurons,
            &mut self.responsiveness,
            pattern,
            influence,
            rate,
        );
        //self
    }

    fn train(&mut self, patterns: &ArrayView2<f64>) {
        self.training.train(
            &mut self.neurons,
            &mut self.adaptivity,
            &mut self.responsiveness,
            patterns,
        );
        // self
    }
}

// #[cfg(feature = "ndarray")]

impl<A, T, R, B> Neural for NeuralLayer<A, T, R, B>
where
    A: Adaptable<Neurons, R>,
    T: Topological<Neurons>,
    R: Responsive<Neurons>,
    B: Trainable<Neurons, A, R>,
{
    fn get_lateral(&self) -> &Array2<f64> {
        &self.neurons.lateral
    }

    fn get_lateral_mut(&mut self) -> &mut Array2<f64> {
        todo!()
    }
    fn set_lateral(&mut self, lateral: Array2<f64>) {
        todo!()
    }

    fn get_patterns(&self) -> &Array2<f64> {
        &self.neurons.patterns
    }

    fn get_patterns_mut(&mut self) -> &mut Array2<f64> {
        todo!()
    }

    fn set_patterns(&mut self, patterns: Array2<f64>) {
        todo!()
    }
}

pub trait SelforganizingNeural: SelfOrganizing + Neural {}
impl<A, T, R, B> SelforganizingNeural for NeuralLayer<A, T, R, B>
where
    A: Adaptable<Neurons, R>,
    T: Topological<Neurons>,
    R: Responsive<Neurons>,
    B: Trainable<Neurons, A, R>,
{
}

// #[cfg(not(feature = "ndarray"))]

// #[cfg(test)]
// mod tests {

//     #[test]
//     fn it_works() {
//         let result = 2 + 2;
//         assert_eq!(result, 4);
//     }
// }
