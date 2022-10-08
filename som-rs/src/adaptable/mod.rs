pub mod kohonen;

pub use kohonen::KohonenAdaptivity;
use ndarray::prelude::*;

use crate::{Neural, Responsive};

pub type BoxedAdaptable<N, R> = Box<dyn Adaptable<N, R> + Send>;

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
    fn clone_dyn(&self) -> BoxedAdaptable<N, R>;
}

impl<N, R> Adaptable<N, R> for BoxedAdaptable<N, R>
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

    fn clone_dyn(&self) -> BoxedAdaptable<N, R> {
        panic!()
    }
}

impl<N, R> Clone for BoxedAdaptable<N, R>
where
    N: Neural,
    R: Responsive<N>,
{
    fn clone(&self) -> Self {
        (**self).clone_dyn()
    }
}
