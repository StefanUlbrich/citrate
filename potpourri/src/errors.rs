#[cfg(feature = "ndarray")]
use ndarray::ErrorKind;

#[derive(thiserror::Error, Debug, Clone)]
pub enum Error {
    #[error("The combination of training parameters is invalid: {0}")]
    InvalidTrainingConfiguration {
        n_init: usize,
        incremental: bool,
        initialized: bool
    },
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

}

impl std::convert::From<std::num::ParseIntError> for Error {
    fn from(err: std::num::ParseIntError) -> Self {
        Error::InvalidArgument(err.to_string())
    }
}