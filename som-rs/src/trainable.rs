use crate::{Adaptable, Neural, Responsive};
/// Interface for structures encapsulating algorithms for training from data sets
pub trait Trainable {
    fn train<S, N, A, F>(
        &mut self,
        neurons: &mut N,
        adaptation: &mut A,
        feature: &mut F,
        patterns: &ArrayBase<S, Ix2>,
    ) where
        N: Neural,
        F: Responsive,
        A: Adaptable,
        S: Data<Elem = f64>;
}

pub struct BatchTraining {
    pub radii: (f64, f64),
    pub rates: (f64, f64),
    pub epochs: usize,
}

use ndarray::{prelude::*, Data};

impl Trainable for BatchTraining {
    fn train<S, N, A, T>(
        &mut self,
        neurons: &mut N,
        adaptation: &mut A,
        tuning: &mut T,
        patterns: &ArrayBase<S, Ix2>,
    ) where
        N: Neural,
        T: Responsive,
        A: Adaptable,
        S: Data<Elem = f64>,
    {
        let n_samples = patterns.len_of(Axis(0));

        for epoch in 0..self.epochs {
            println!("{}", epoch);
            for (i, pattern) in patterns.outer_iter().enumerate() {
                let progress =
                    ((epoch * n_samples + i) as f64) / ((self.epochs * n_samples) as f64);
                let rate = self.rates.0 * (self.rates.1 / self.rates.0).powf(progress);
                let influence = self.radii.0 * (self.radii.1 / self.radii.0).powf(progress);

                adaptation.adapt(neurons, tuning, &pattern, influence, rate);
                // println!("{}:{}: {},{}", epoch, i,rate, influence);
            }
        }
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}