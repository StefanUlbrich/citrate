// #[derive(Copy,Clone)]
pub struct BatchTraining {
    pub radii: (f64, f64),
    pub rates: (f64, f64),
    pub epochs: usize,
}


#[cfg(feature = "ndarray")]
mod nd {
    use ndarray::prelude::*;

    use super::BatchTraining;
    use crate::{Adaptable, Neural, Tunable, Trainable};

    impl Trainable<Array2<f64>, Array2<f64>> for BatchTraining {

        type ArgType = Array2<f64>;
        // Expect troubles with the data argument .. that needs to be changed to allow for views
        fn train<N, A, F>(&mut self, neurons: &mut N, adaptation: &mut A, feature: &mut F, data: &Array2<f64>)
        where
            N: Neural<Array2<f64>, Array2<f64>>,
            F: Tunable<Array2<f64>, Array2<f64>>,
            A: Adaptable<Array2<f64>, Array2<f64>>,
        {
            for epoch in 0..self.epochs {
                println!("{}", epoch);
                // for (i, feature) in features.outer_iter().enumerate() {
                //     let progress = ((epoch * n_samples + i) as f64) / ((epochs * n_samples) as f64); // check the use of epochs
                //     let rate = rates.0 * (rates.1 / rates.0).powf(progress);
                //     let influence = influences.0 * (influences.1 / influences.0).powf(progress);

                //     self.adapt(&feature, influence, rate);
                //     // println!("{}:{}: {},{}", epoch, i,rate, influence);
                // }
            }
        }
    }
}

#[cfg(not(feature = "ndarray"))]
mod pure {
    use std::fmt::Debug;

    use super::BatchTraining;
    use crate::{Adaptable, Neural, Tunable, Trainable};

    impl<D1: Debug, D2: Debug> Trainable<D1, D2> for BatchTraining {
        fn train<D, A, F>(&mut self, data: &mut D, adaptation: &mut A, feature: &mut F)
        where
            D: Neural<D1, D2>,
            F: Tunable<D1, D2>,
            A: Adaptable<D1, D2>,
        {
            todo!()
        }
    }
}

#[cfg(feature = "ndarray")]
pub use nd::*;
#[cfg(not(feature = "ndarray"))]
pub use pure::*;
