pub struct ClassicAdaptivity {}

use ndarray::{prelude::*, Data};

use crate::{Adaptable, Neural, Tunable};

impl Adaptable for ClassicAdaptivity {
    fn adapt<S, N, T>(&mut self, neurons: &mut N, tuning: &mut T, pattern: &ArrayBase<S, Ix1>)
    where
        T: Tunable,
        N: Neural,
        S: Data<Elem = f64>,
    {
        todo!()
    }
}

pub struct SmoothAdaptivity {}

impl Adaptable for SmoothAdaptivity {
    fn adapt<S, N, T>(&mut self, neurons: &mut N, tuning: &mut T, pattern: &ArrayBase<S, Ix1>)
    where
        T: Tunable,
        N: Neural,
        S: Data<Elem = f64>,
    {
        todo!()
    }
}
