use ndarray::{prelude::*, Shape};
use crate::nd_tools::ndindex::get_ndindex_array;
use crate::{Neural, Topological};

pub struct CartesianTopology<D> where D: Dimension {
    shape: Shape<D>,
}

impl<D> CartesianTopology<D>
where D: Dimension
{
    pub fn new<Sh>(shape: Sh) -> Self
    where Sh: ShapeBuilder<Dim = D> {
        CartesianTopology { shape: shape.into_shape() }
    }
}



impl<D> Topological for CartesianTopology<D>
where D: Dimension
{
    fn get_lateral_connections<N>(&mut self, neurons: &N) -> f64
    where
        N: Neural,
    {
        todo!()
    }

    fn init_lateral<N>(&self, neurons: &mut N)
    where
        N: Neural,
    {
        neurons
            .get_lateral_mut()
            .assign(&get_ndindex_array(&self.shape));
    }
}
