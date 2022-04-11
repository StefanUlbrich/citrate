use std::fmt::Debug;

use super::{
    Adaptable, Neural, NeuralLayer, Neurons, SelfOrganizing, Topological, Trainable, Tunable,
};

impl<V, A, T, F, B> SelfOrganizing<V> for NeuralLayer<V, A, T, F, B>
where
    V: Debug,
    A: Adaptable<V>,
    T: Topological<V>,
    F: Tunable<V>,
    B: Trainable<V>,
{
    fn adapt(mut self, data: &V) -> Self {
        self.adaptivity.adapt(&mut self.neurons, &mut self.tuning);
        self
    }

    fn init_lateral(&self) {
        self.topology.init_lateral(self);
    }

    fn get_lateral_distance(&mut self, index: usize) -> V {
        todo!()
    }

    fn get_best_matching(&self) -> usize {
        self.tuning.get_best_matching(self)
    }

    fn train(mut self, data: &V) -> Self {
        self.training
            .train(&mut self.neurons, &mut self.adaptivity, &mut self.tuning);
        self
    }
}

#[cfg(feature = "simple")]
impl<V: Debug> Neural<V> for Neurons<V> {
    fn get_lateral(&self) -> &V {
        &self.lateral
    }

    fn get_lateral_mut(&mut self) -> &mut V {
        &mut self.lateral
    }
    fn set_lateral(&mut self, lateral: V) {
        todo!()
    }

    fn get_patterns(&self) -> &V {
        &self.patterns
    }

    fn get_patterns_mut(&mut self) -> &mut V {
        &mut self.patterns
    }

    fn set_patterns(&mut self, patterns: V) {
        todo!()
    }
}

#[cfg(feature = "simple")]
impl<V, A, T, F, B> Neural<V> for NeuralLayer<V, A, T, F, B>
where
    V: Debug,
    A: Adaptable<V>,
    T: Topological<V>,
    F: Tunable<V>,
    B: Trainable<V>,
{
    fn get_lateral(&self) -> &V {
        &self.neurons.lateral
    }

    fn get_lateral_mut(&mut self) -> &mut V {
        todo!()
    }
    fn set_lateral(&mut self, lateral: V) {
        todo!()
    }

    fn get_patterns(&self) -> &V {
        &self.neurons.patterns
    }

    fn get_patterns_mut(&mut self) -> &mut V {
        todo!()
    }

    fn set_patterns(&mut self, patterns: V) {
        todo!()
    }
}

#[cfg(feature = "ndarray")]
impl<V: Debug> Neural<V> for Neurons<V> {
    fn get_lateral(&self) -> &V {
        &self.lateral
    }

    fn get_lateral_mut(&mut self) -> &mut V {
        &mut self.lateral
    }
    fn set_lateral(&mut self, lateral: V) {
        todo!()
    }

    fn get_patterns(&self) -> &V {
        &self.patterns
    }

    fn get_patterns_mut(&mut self) -> &mut V {
        &mut self.patterns
    }

    fn set_patterns(&mut self, patterns: V) {
        todo!()
    }
}
#[cfg(feature = "ndarray")]
impl<V, A, T, F, B> Neural<V> for NeuralLayer<V, A, T, F, B>
where
    V: Debug,
    A: Adaptable<V>,
    T: Topological<V>,
    F: Tunable<V>,
    B: Trainable<V>,
{
    fn get_lateral(&self) -> &V {
        &self.neurons.lateral
    }

    fn get_lateral_mut(&mut self) -> &mut V {
        todo!()
    }
    fn set_lateral(&mut self, lateral: V) {
        todo!()
    }

    fn get_patterns(&self) -> &V {
        &self.neurons.patterns
    }

    fn get_patterns_mut(&mut self) -> &mut V {
        todo!()
    }

    fn set_patterns(&mut self, patterns: V) {
        todo!()
    }
}
