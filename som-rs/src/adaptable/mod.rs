pub mod kohonen;

pub use kohonen::KohonenAdaptivity;
use ndarray::{prelude::*, Data};

use crate::{Neural, Responsive};

/// Interface for structures encapsulating algorithms for self-organization
pub trait Adaptable {
    fn adapt<S, N, T>(
        &mut self,
        neurons: &mut N,
        tuning: &mut T,
        pattern: &ArrayBase<S, Ix1>,
        influence: f64,
        rate: f64,
    )
    //&Self::ArgType)
    where
        T: Responsive,
        N: Neural,
        S: Data<Elem = f64>;
}
