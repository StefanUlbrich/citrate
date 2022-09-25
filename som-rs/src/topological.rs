use crate::nd_tools::ndindex::get_ndindex_array;
use crate::Neural;
use ndarray::{prelude::*, Shape};

/// Interface for structures encapsulating representations of network layer topologies.
pub trait Topological<N>
where
    N: Neural,
{
    // fn get_lateral_connections<N>(&mut self, data: &N) -> f64;

    fn init_lateral(&self, neurons: &mut N);
}

impl<N> Topological<N> for Box<dyn Topological<N>>
where
    N: Neural,
{
    fn init_lateral(&self, neurons: &mut N) {
        (**self).init_lateral(neurons)
    }
}

pub struct CartesianTopology<D>
where
    D: Dimension,
{
    shape: Shape<D>,
}

impl<D> CartesianTopology<D>
where
    D: Dimension,
{
    pub fn new<Sh>(shape: Sh) -> Self
    where
        Sh: ShapeBuilder<Dim = D>,
    {
        CartesianTopology {
            shape: shape.into_shape(),
        }
    }
}

impl<D, N> Topological<N> for CartesianTopology<D>
where
    N: Neural,
    D: Dimension,
{
    // fn get_lateral_connections<N>(&mut self, neurons: &N) -> f64
    // where
    //     N: Neural,
    // {
    //     todo!()
    // }

    fn init_lateral(&self, neurons: &mut N) {
        neurons
            .get_lateral_mut()
            .assign(&get_ndindex_array(&self.shape));
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
