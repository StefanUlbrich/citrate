use super::super::{Neural, SelfOrganizing, Topological};
use std::fmt::Debug;

#[cfg(feature = "simple")]
pub struct CartesianTopology {}
#[cfg(feature = "simple")]
impl<V> Topological<V> for CartesianTopology
where
    V: Debug,
{
    fn init_lateral<T>(&self, nn: &T)
    where
        T: SelfOrganizing<V> + Neural<V>,
    {
        println!("Cartesian {:?}", nn.get_lateral());
    }

    fn get_lateral_connections<T>(&mut self, nn: &T) -> f64
    where
        T: SelfOrganizing<V>,
    {
        42.0
    }
}

#[cfg(feature = "ndarray")]
pub struct CartesianTopology {}
#[cfg(feature = "ndarray")]
impl<V> Topological<V> for CartesianTopology
where
    V: Debug,
{
    fn init_lateral<T>(&self, nn: &T)
    where
        T: SelfOrganizing<V> + Neural<V>,
    {
        println!("Cartesian {:?}", nn.get_lateral());
    }

    fn get_lateral_connections<T>(&mut self, nn: &T) -> f64
    where
        T: SelfOrganizing<V>,
    {
        42.0
    }
}