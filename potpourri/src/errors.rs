// #[cfg(feature = "ndarray")]
use ndarray::ShapeError;

use ndarray_linalg::error::LinalgError;

#[derive(thiserror::Error, Debug, Clone)]
pub enum Error {
    #[error("Multiple iterations overwrite a fitted model: {n_init:?}, {fitted:?}")]
    ParameterError { n_init: usize, fitted: bool },
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),
    #[error("Medthod not implemented")]
    NotImplemented,
    #[error("Dimension mismatch")] // TODO make parameterizable
    DimensionMismatch,
    #[error("Error in ndarray_linalg")]
    LinalgError,
    #[error("Wrong shapes")]
    ShapeError,
    #[error("Should never be executed")]
    ForbiddenCode,
}

impl std::convert::From<std::num::ParseIntError> for Error {
    fn from(err: std::num::ParseIntError) -> Self {
        Error::InvalidArgument(err.to_string())
    }
}

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
