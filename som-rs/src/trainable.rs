use core::panic;

use crate::{Adaptable, Neural, Responsive};

pub type BoxedTrainable<N, A, R> = Box<dyn Trainable<N, A, R> + Send>;
/// Interface for structures encapsulating algorithms for training from data sets
pub trait Trainable<N, A, R>
where
    N: Neural,
    A: Adaptable<N, R>,
    R: Responsive<N>,
{
    fn train(
        &mut self,
        neurons: &mut N,
        adaptation: &mut A,
        feature: &mut R,
        patterns: &ArrayView2<f64>,
    );
    fn clone_dyn(&self) -> Box<dyn Trainable<N, A, R> + Send>;
}

impl<N, A, R> Trainable<N, A, R> for Box<dyn Trainable<N, A, R> + Send>
where
    N: Neural,
    A: Adaptable<N, R>,
    R: Responsive<N>,
{
    fn train(
        &mut self,
        neurons: &mut N,
        adaptation: &mut A,
        feature: &mut R,
        patterns: &ArrayView2<f64>,
    ) {
        (**self).train(neurons, adaptation, feature, patterns)
    }

    fn clone_dyn(&self) -> Box<dyn Trainable<N, A, R> + Send> {
        panic!()
    }
}

impl<N, A, R> Clone for Box<dyn Trainable<N, A, R> + Send>
where
    N: Neural,
    A: Adaptable<N, R>,
    R: Responsive<N>,
{
    fn clone(&self) -> Self {
        (**self).clone_dyn()
    }
}

#[derive(Clone)]
pub struct BatchTraining {
    pub radii: (f64, f64),
    pub rates: (f64, f64),
    pub epochs: usize,
}

use ndarray::{prelude::*, Data};

impl<N, A, R> Trainable<N, A, R> for BatchTraining
where
    N: Neural,
    A: Adaptable<N, R>,
    R: Responsive<N>,
{
    fn train(
        &mut self,
        neurons: &mut N,
        adaptation: &mut A,
        tuning: &mut R,
        patterns: &ArrayView2<f64>,
    ) {
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

    fn clone_dyn(&self) -> Box<dyn Trainable<N, A, R> + Send> {
        Box::new(self.clone()) // Forward to the derive(Clone) impl    }
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
