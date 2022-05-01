// #[derive(Copy,Clone)]
pub struct BatchTraining {
    pub radii: (f64, f64),
    pub rates: (f64, f64),
    pub epochs: usize,
}

use ndarray::{prelude::*, Data};

use crate::{Adaptable, Neural, Trainable, Competitive};

impl Trainable for BatchTraining {
    fn train<S, N, A, T>(
        &mut self,
        neurons: &mut N,
        adaptation: &mut A,
        tuning: &mut T,
        patterns: &ArrayBase<S, Ix2>,
    ) where
        N: Neural,
        T: Competitive,
        A: Adaptable,
        S: Data<Elem = f64>,
    {
        let n_samples = patterns.len_of(Axis(0));

        for epoch in 0..self.epochs {
            println!("{}", epoch);
            for (i, pattern) in patterns.outer_iter().enumerate() {
                let progress = ((epoch * n_samples + i) as f64) / ((self.epochs * n_samples) as f64);
                let rate = self.rates.0 * (self.rates.1 / self.rates.0).powf(progress);
                let influence = self.radii.0 * (self.radii.1 / self.radii.0).powf(progress);

                adaptation.adapt(neurons, tuning, &pattern, influence, rate);
                // println!("{}:{}: {},{}", epoch, i,rate, influence);
            }
        }
    }
}
