//! This module defines a self organizing maps with neurons
//! distributed in a regular grid in the latent space

use super::SelfOrganizingMap;
use ndarray::prelude::*;
use ndarray_rand::rand::{distributions::Distribution, Rng};
use ndarray_rand::RandomExt;

use crate::ndarray::ndindex::get_ndindex_array;

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
    pub fn new<Sh, IdS, R>(
        shape: Sh,
        output_dim: usize,
        distribution: IdS,
        rng: &mut R,
    ) -> CartesianGrid
    where
        Sh: ShapeBuilder,
        IdS: Distribution<f64>,
        R: Rng + ?Sized,
    {
        let latent = get_ndindex_array(shape);

        CartesianGrid {
            feature: Array::<f64, Ix2>::random_using(
                (latent.shape()[0], output_dim),
                distribution,
                rng,
            ),
            latent: latent,
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
