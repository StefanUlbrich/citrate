pub mod finite;
pub mod gaussian;
pub mod kmeans;
pub mod linear;
pub mod som;
pub mod utils;

// Add errors specific to ndarray

use crate::errors::Error;
use ndarray::ShapeError;
use ndarray_linalg::error::LinalgError;

impl std::convert::From<LinalgError> for Error {
    fn from(_: LinalgError) -> Self {
        Error::LinalgError
    }
}
impl std::convert::From<ShapeError> for Error {
    fn from(_: ShapeError) -> Self {
        Error::ShapeError
    }
}
