use super::{Tunable, SelfOrganizing};
use std::fmt::Debug;

#[cfg(feature = "simple")]
pub struct CartesianFeature {}
#[cfg(feature = "simple")]
impl<V> Tunable<V> for CartesianFeature
where
    V: Debug,
{
    fn get_best_matching<T>(&self, nn: &T) -> usize
    where
        T: SelfOrganizing<V>,
    {
        todo!()
    }
}


#[cfg(feature = "ndarray")]
pub struct CartesianFeature {}

#[cfg(feature = "ndarray")]
impl<V> Tunable<V> for CartesianFeature
where
    V: Debug,
{
    fn get_best_matching<T>(&self, nn: &T) -> usize
    where
        T: SelfOrganizing<V>,
    {
        todo!()
    }
}