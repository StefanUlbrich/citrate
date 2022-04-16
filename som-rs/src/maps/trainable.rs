// #[derive(Copy,Clone)]
pub struct BatchTraining {
    pub radii: (f64, f64),
    pub rates: (f64, f64),
    pub epochs: usize,
}

use ndarray::{prelude::*, Data};

use crate::{Adaptable, Neural, Trainable, Tunable};

impl Trainable for BatchTraining {
    fn train<S, N, A, T>(
        &mut self,
        neurons: &mut N,
        adaptation: &mut A,
        tuning: &mut T,
        patterns: &ArrayBase<S, Ix2>,
    ) where
        N: Neural,
        T: Tunable,
        A: Adaptable,
        S: Data<Elem = f64>,
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
