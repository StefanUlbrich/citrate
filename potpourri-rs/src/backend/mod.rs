//! Computation backends such as [ndrray
//! ](https://docs.rs/ndarray/latest/ndarray/) or [ractor
//! ](https://github.com/slawlor/ractor) (more to follow)

#[cfg(feature = "ndarray")]
pub mod ndarray;
#[cfg(feature = "ractor")]
pub mod ractor;
