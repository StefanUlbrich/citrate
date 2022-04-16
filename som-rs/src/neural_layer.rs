use ndarray::{prelude::*, Data};
use std::fmt::Debug;

use super::{
    Adaptable, Neural, NeuralLayer, Neurons, SelfOrganizing, Topological, Trainable, Tunable,
};

impl<A, T, F, B> SelfOrganizing for NeuralLayer<A, T, F, B>
where
    A: Adaptable,
    T: Topological,
    F: Tunable,
    B: Trainable,
{
    fn init_lateral(mut self) -> Self {
        self.topology.init_lateral(&mut self.neurons);
        self
    }

    fn get_lateral_distance(&mut self, index: usize) -> Array2<f64> {
        todo!()
    }

    fn get_best_matching<S>(&self, pattern: &ArrayBase<S, Ix1>) -> usize
    where
        S: Data<Elem = f64>,
    {
        self.tuning.get_best_matching(self, pattern)
    }

    fn adapt<S>(mut self, pattern: &ArrayBase<S, Ix1>, influence: f64, rate: f64) -> Self
    where
        S: Data<Elem = f64>,
    {
        self.adaptivity
            .adapt(&mut self.neurons, &mut self.tuning, pattern, influence, rate);
        self
    }

    fn train<S>(mut self, patterns: &ArrayBase<S, Ix2>) -> Self
    where
        S: Data<Elem = f64>,
    {
        self.training.train(
            &mut self.neurons,
            &mut self.adaptivity,
            &mut self.tuning,
            patterns,
        );
        self
    }
}

// #[cfg(feature = "ndarray")]
impl Neural for Neurons {
    fn get_lateral(&self) -> &Array2<f64> {
        &self.lateral
    }

    fn get_lateral_mut(&mut self) -> &mut Array2<f64> {
        &mut self.lateral
    }
    fn set_lateral(&mut self, lateral: Array2<f64>) {
        todo!()
    }

    fn get_patterns(&self) -> &Array2<f64> {
        &self.patterns
    }

    fn get_patterns_mut(&mut self) -> &mut Array2<f64> {
        &mut self.patterns
    }

    fn set_patterns(&mut self, patterns: Array2<f64>) {
        todo!()
    }
}

impl<A, T, F, B> Neural for NeuralLayer<A, T, F, B>
where
    A: Adaptable,
    T: Topological,
    F: Tunable,
    B: Trainable,
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

// #[cfg(not(feature = "ndarray"))]
