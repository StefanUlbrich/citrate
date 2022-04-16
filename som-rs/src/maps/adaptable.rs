pub struct ClassicAdaptivity {}
pub struct SmoothAdaptivity {}

#[cfg(feature = "ndarray")]
mod nd {
    use ndarray::{prelude::*, Data};

    use super::{ClassicAdaptivity, SmoothAdaptivity};
    use crate::{Adaptable, Neural, Tunable};

    impl Adaptable<Array2<f64>, Array2<f64>> for ClassicAdaptivity
    {
        type ArgType = ArrayView2<f64>; //

        fn adapt<D, F>(&mut self, data: &mut D, feature: &mut F, p: &Self::ArgType)
        where
            F: Tunable<Array2<f64>, Array2<f64>>,
            D: Neural<Array2<f64>, Array2<f64>>,

        {
            todo!()
        }
    }

    impl Adaptable<Array2<f64>, Array2<f64>> for SmoothAdaptivity {
        type ArgType = Array2<f64>;

        fn adapt<D, F>(&mut self, data: &mut D, feature: &mut F, pattern: &Self::ArgType)
        where
            F: Tunable<Array2<f64>, Array2<f64>>,
            D: Neural<Array2<f64>, Array2<f64>>,
        {
            todo!()
        }
    }
}

#[cfg(not(feature = "ndarray"))]
mod pure {

    use super::{ClassicAdaptivity, SmoothAdaptivity};
    use crate::{Adaptable, Neural, Tunable};

    use std::fmt::Debug;

    impl<D1, D2> Adaptable<D1, D2> for ClassicAdaptivity
    where
        D1: Debug,
        D2: Debug,
    {
        fn adapt<D, F>(&mut self, data: &mut D, feature: &mut F, pattern: &Self::ArgType)
        where
            F: Tunable<D1, D2>,
            D: Neural<D1, D2>,
        {
            todo!()
        }
        type ArgType;
    }

    impl<D1, D2> Adaptable<D1, D2> for SmoothAdaptivity
    where
        D1: Debug,
        D2: Debug,
    {
        fn adapt<D, F>(&mut self, data: &mut D, feature: &mut F, pattern: &Self::ArgType)
        where
            F: Tunable<D1, D2>,
            D: Neural<D1, D2>,
        {
            todo!()
        }
        type ArgType;
    }
}

#[cfg(feature = "ndarray")]
pub use nd::*;
#[cfg(not(feature = "ndarray"))]
pub use pure::*;
