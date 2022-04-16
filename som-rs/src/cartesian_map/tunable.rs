pub struct CartesianFeature {}

#[cfg(feature = "ndarray")]
mod nd {
    use ndarray::prelude::*;

    use super::CartesianFeature;
    use crate::{Tunable, Neural};


    impl Tunable<Array2<f64>, Array2<f64>> for CartesianFeature
    {
        fn  get_best_matching<N>(&self, neurons: &N, pattern: &Self::ArgType) -> usize
        where
            N: Neural<Array2<f64>, Array2<f64>>,
        {
            todo!()
        }

        type ArgType=Array2<f64>;
    }
}

#[cfg(not(feature = "ndarray"))]
mod pure {
    use super::CartesianFeature;
    use crate::{Neural, Tunable};
    use std::fmt::Debug;

    impl Tunable<D1, D2> for CartesianFeature
    where
        D1: Debug,
        D2: Debug,
    {
        fn get_best_matching<D>(&self, data: &D, pattern: &D2) -> usize
        where
            D: crate::Neural<D1, D2>,
        {
            todo!()
        }
    }
}

#[cfg(feature = "ndarray")]
pub use nd::*;
#[cfg(not(feature = "ndarray"))]
pub use pure::*;
