//! This module defines a self organizing maps with neurons
//! distributed in a regular grid in the latent space

use ndarray::prelude::*;
use super::SelfOrganizingMap;
use crate::ndarray::{ndindex::get_ndindex_array};

#[derive(Debug, Default)]
pub struct Uniform {
    feature: Array<f64, Ix2>,
    latent: Array<f64, Ix2>,
}

impl Uniform {

    /// Creates a self organizing map with neurons
    /// arranged in a regular grid according to the `shape` parameter
    /// # Examples
    ///
    /// ```
    /// use som_rs::som::uniform::Uniform;
    ///
    /// assert_eq!(Uniform::new(shape), );
    /// ```
    pub fn new<Sh>(shape: Sh) -> Uniform
    where
        Sh: ShapeBuilder
    {
        Uniform {
            feature: Array::<f64, Ix2>::zeros((3, 2)),
            latent: get_ndindex_array(shape),
        }
    }
}

impl SelfOrganizingMap for Uniform {
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
