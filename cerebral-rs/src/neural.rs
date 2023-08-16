//! Properties of lateral and pattern spaces of a network

use ndarray::prelude::*;

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

#[derive(Default)]
pub struct NeuralLayer {
    /// Lateral layer that defines the topology. Can be coordinates or connections (depending on method). Row matrix.
    pub lateral: Array2<f64>,
    /// Tuning Patterns the neurons. This is the codebook. Row matrix.
    pub patterns: Array2<f64>,
}

impl NeuralLayer {
    pub fn new() -> NeuralLayer {
        NeuralLayer {
            lateral: Array2::<f64>::zeros((0, 0)),
            patterns: Array2::<f64>::zeros((0, 0)),
        }
    }
}

impl Neural for NeuralLayer {
    fn get_lateral(&self) -> &Array2<f64> {
        &self.lateral
    }

    fn get_lateral_mut(&mut self) -> &mut Array2<f64> {
        &mut self.lateral
    }
    fn set_lateral(&mut self, lateral: Array2<f64>) {
        self.lateral = lateral;
    }

    fn get_patterns(&self) -> &Array2<f64> {
        &self.patterns
    }

    fn get_patterns_mut(&mut self) -> &mut Array2<f64> {
        &mut self.patterns
    }

    fn set_patterns(&mut self, patterns: Array2<f64>) {
        self.patterns = patterns;
    }
}
