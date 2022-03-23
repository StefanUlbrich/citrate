//! This module defines a self organizing maps with neurons
//! distributed in a regular grid in the latent space

use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use super::SelfOrganizingMap;

use crate::ndarray::{ndindex::get_ndindex_array};


#[derive(Debug, Default)]
pub struct CartesianGrid {
    feature: Array<f64, Ix2>,
    latent: Array<f64, Ix2>,
}

impl CartesianGrid {

    /// Creates a self organizing map with neurons
    /// arranged in a regular grid according to the `shape` parameter
    /// # Examples
    ///
    /// ```
    /// use som_rs::som::uniform::Uniform;
    ///
    /// assert_eq!(Uniform::new(shape), );
    /// ```
    pub fn new<Sh>(shape: Sh, output_dim: usize, low: Option<f64>, high: Option<f64>) -> CartesianGrid
    where
        Sh: ShapeBuilder
    {
        let latent =  get_ndindex_array(shape);

        let low = low.unwrap_or(0.0);
        let high = high.unwrap_or(1.0);

        CartesianGrid {
            feature: Array::<f64, Ix2>::random((latent.shape()[0], output_dim), Uniform::new(low,high)),
            latent: latent
        }
    }
}

impl SelfOrganizingMap for CartesianGrid {
    fn get_latent(&self) -> &Array2<f64> {
        &self.latent
    }

    fn get_feature_mut(&mut self) -> &mut Array2<f64> {
        &mut self.feature
    }

    fn get_feature(&self) -> &Array2<f64> {
        &self.feature
    }
}
