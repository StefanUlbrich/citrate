//! Extensions to the ndarray crate. General functions are defined in the top-level

pub mod ndindex;
pub mod point_set;

use ndarray::prelude::*;

/// Returns the index of the smallest element of a vector. Panics if vector is of size 0.
pub fn argmin(a: &Array1<f64>) -> usize {
    let min = a.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    a.iter().position(|e| e.eq(&min)).unwrap()
}
