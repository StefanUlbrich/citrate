use std::fmt::Debug;

use super::{
    Adaptable, Neural, NeuralLayer, Neurons, SelfOrganizing, Topological, Trainable, Tunable,
};

impl<D1, D2, A, T, F, B> SelfOrganizing<D1, D2> for NeuralLayer<D1, D2, A, T, F, B>
where
    D1: Debug,
    D2: Debug,
    A: Adaptable<D1, D2>,
    T: Topological<D1, D2>,
    F: Tunable<D1, D2>,
    B: Trainable<D1, D2>,
{
    type ArgTypeA = A::ArgType;
    type ArgTypeF = F::ArgType;
    type ArgTypeB = B::ArgType;

    fn init_lateral(mut self) -> Self {
        self.topology.init_lateral(&mut self.neurons);
        self
    }

    fn get_lateral_distance(&mut self, index: usize) -> D1 {
        todo!()
    }

    fn get_best_matching(&self, pattern: &Self::ArgTypeF) -> usize {
        self.tuning.get_best_matching(self, &pattern)
    }

    fn adapt<P>(mut self, pattern: P //&Self::ArgTypeA
    ) -> Self {
        self.adaptivity.adapt(&mut self.neurons, &mut self.tuning, pattern);
        self
    }

    fn train(mut self, patterns: &Self::ArgTypeB) -> Self {
        self.training.train(
            &mut self.neurons,
            &mut self.adaptivity,
            &mut self.tuning,
            patterns,
        );
        self
    }
}

#[cfg(feature = "ndarray")]
impl<D1: Debug, D2: Debug> Neural<D1, D2> for Neurons<D1, D2> {
    fn get_lateral(&self) -> &D1 {
        &self.lateral
    }

    fn get_lateral_mut(&mut self) -> &mut D1 {
        &mut self.lateral
    }
    fn set_lateral(&mut self, lateral: D1) {
        todo!()
    }

    fn get_patterns(&self) -> &D2 {
        &self.patterns
    }

    fn get_patterns_mut(&mut self) -> &mut D2 {
        &mut self.patterns
    }

    fn set_patterns(&mut self, patterns: D2) {
        todo!()
    }
}
#[cfg(feature = "ndarray")]
impl<D1, D2, A, T, F, B> Neural<D1, D2> for NeuralLayer<D1, D2, A, T, F, B>
where
    D1: Debug,
    D2: Debug,
    A: Adaptable<D1, D2>,
    T: Topological<D1, D2>,
    F: Tunable<D1, D2>,
    B: Trainable<D1, D2>,
{
    fn get_lateral(&self) -> &D1 {
        &self.neurons.lateral
    }

    fn get_lateral_mut(&mut self) -> &mut D1 {
        todo!()
    }
    fn set_lateral(&mut self, lateral: D1) {
        todo!()
    }

    fn get_patterns(&self) -> &D2 {
        &self.neurons.patterns
    }

    fn get_patterns_mut(&mut self) -> &mut D2 {
        todo!()
    }

    fn set_patterns(&mut self, patterns: D2) {
        todo!()
    }
}

#[cfg(not(feature = "ndarray"))]
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

#[cfg(not(feature = "ndarray"))]
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
