use super::{Adaptable, Tunable, Neural};
use std::fmt::Debug;

#[cfg(feature = "simple")]
struct ClassicAdaptivity {}

#[cfg(feature = "simple")]
impl<V> Adaptable<V> for ClassicAdaptivity
where
    V: Debug,
{
    fn adapt<D,F>(&mut self, data: &mut D, feature: &mut F)
    where
        F: Tunable<V>,
        D: Neural<V>
    {
        println!("Classic {:?}", data.get_patterns());
    }
}

#[cfg(feature = "ndarray")]
struct ClassicAdaptivity {}

#[cfg(feature = "ndarray")]
impl<V> Adaptable<V> for ClassicAdaptivity
where
    V: Debug,
{
    fn adapt<D,F>(&mut self, data: &mut D, feature: &mut F)
    where
        F: Tunable<V>,
        D: Neural<V>
    {
        println!("Classic {:?}", data.get_patterns());
    }
}

pub struct SmoothAdaptivity {}
impl<V> Adaptable<V> for SmoothAdaptivity
where
    V: Debug,
{
    fn adapt<D,F>(&mut self, data: &mut D, feature: &mut F)
    where
        F: Tunable<V>,
        D: Neural<V>
    {
        println!("Smooth {:?}", data.get_patterns());
    }
}