//! The model for a self-organizing neural network

use ndarray::{prelude::*, Data};

use crate::{Adaptable, Neural, NeuralLayer, Responsive, Topological, Trainable};

// TODO why a trait and not on the struct itself?

/// Public trait for a model of self-organization.
/// It combines the methods of [Neural] (which it extends, i.e.,
/// its implementations need to implement it as well),
/// [Adaptable], [Responsive], [Topological] and [Trainable]
/// but with different parameters. Implementations
/// are supposed to delegate calls to instances of said traits.
pub trait Selforganizing: Neural {
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

/// Default struct for self-organization
pub struct SelforganizingNetwork<A, T, R, L>
where
    A: Adaptable<NeuralLayer, R>,
    T: Topological<NeuralLayer>,
    R: Responsive<NeuralLayer>,
    L: Trainable<NeuralLayer, A, R>,
    // B: Trainable<D1,D2> + Copy,
{
    /// needs to be nested to share it with the algorithms
    pub neurons: NeuralLayer,
    /// Algorithm for adaptivity
    pub adaptivity: A,
    /// Algorithm related to topology
    pub topology: T,
    /// Algorithm to feature pattern matching and lateral inhibition
    pub responsiveness: R,
    /// Algorithm related to batch processing
    pub training: L, // Box<B>,
}

// Implementation of neural

impl<A, T, R, B> Selforganizing for SelforganizingNetwork<A, T, R, B>
where
    A: Adaptable<NeuralLayer, R>,
    T: Topological<NeuralLayer>,
    R: Responsive<NeuralLayer>,
    B: Trainable<NeuralLayer, A, R>,
{
    fn init_lateral(&mut self) {
        self.topology.init_lateral(&mut self.neurons);
    }

    // TODO think about removing if unused
    fn get_lateral_distance(&mut self, index: usize) -> Array2<f64> {
        todo!()
    }

    fn get_best_matching(&mut self, pattern: &ArrayView1<f64>) -> usize {
        self.responsiveness
            .get_best_matching(&self.neurons, pattern)
    }

    fn adapt(&mut self, pattern: &ArrayView1<f64>, influence: f64, rate: f64) {
        self.adaptivity.adapt(
            &mut self.neurons,
            &mut self.responsiveness,
            pattern,
            influence,
            rate,
        );
    }

    fn train(&mut self, patterns: &ArrayView2<f64>) {
        self.training.train(
            &mut self.neurons,
            &mut self.adaptivity,
            &mut self.responsiveness,
            patterns,
        );
    }
}

// Big TODO: why fullfilling [Neural]? Returning a reference to self.neural would do
// we would need a get_neural_mut and get_neural .. probably doesn't make much of a difference

impl<A, T, R, B> Neural for SelforganizingNetwork<A, T, R, B>
where
    A: Adaptable<NeuralLayer, R>,
    T: Topological<NeuralLayer>,
    R: Responsive<NeuralLayer>,
    B: Trainable<NeuralLayer, A, R>,
{
    // TODO fill these
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

// TODO do I need to implement the traits for this type?

pub type BoxedSelforganizing = Box<dyn Selforganizing + Send>;

// TODO what about this comment block
// pub trait SelforganizingNeural: SelfOrganizing + Neural {}
// impl<A, T, R, B> SelforganizingNeural for NeuralLayer<A, T, R, B>
// where
//     A: Adaptable<Neurons, R>,
//     T: Topological<Neurons>,
//     R: Responsive<Neurons>,
//     B: Trainable<Neurons, A, R>,
// {
// }
// pub type BoxedSelforganizingNeural = Box<dyn SelforganizingNeural + Send>;
