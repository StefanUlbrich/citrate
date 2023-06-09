pub mod categorical;
pub mod gaussian;
pub mod kmeans;
pub mod linear;
pub mod som;
pub mod utils;

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use crate::backend::ndarray::utils::{filter_data, generate_samples};
//     use ndarray::prelude::*;

//     #[test]
//     fn em_step() {
//     }
// }

use ndarray::ShapeError;
use ndarray_linalg::error::LinalgError;

use crate::errors::Error;

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
