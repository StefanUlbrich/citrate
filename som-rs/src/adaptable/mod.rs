pub mod kohonen;

pub use kohonen::KohonenAdaptivity;
use ndarray::prelude::*;

use crate::{Neural, Responsive};

/// Interface for structures encapsulating algorithms for self-organization
pub trait Adaptable<N, R>
where
    R: Responsive<N>,
    N: Neural,
{
    fn adapt(
        &mut self,
        neurons: &mut N,
        responsiveness: &mut R,
        pattern: &ArrayView1<f64>,
        influence: f64,
        rate: f64,
    );
    //&Self::ArgType)
}

impl<N, R> Adaptable<N, R> for Box<dyn Adaptable<N, R>>
where
    N: Neural,
    R: Responsive<N>,
{
    fn adapt(
        &mut self,
        neurons: &mut N,
        responsiveness: &mut R,
        pattern: &ArrayView1<f64>,
        influence: f64,
        rate: f64,
    ) {
        (**self).adapt(neurons, responsiveness, pattern, influence, rate)
    }
}
