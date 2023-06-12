use crate::nd_tools::ndindex::get_ndindex_array;
use crate::Neural;
use ndarray::{prelude::*, Shape};

pub type BoxedTopological<N> = Box<dyn Topological<N> + Send>;
/// Interface for structures encapsulating representations of network layer topologies.
pub trait Topological<N>
where
    N: Neural,
{
    fn init_lateral(&self, neurons: &mut N);

    // Todo can we make this non public?
    fn clone_dyn(&self) -> BoxedTopological<N>;
}

impl<N> Topological<N> for BoxedTopological<N>
where
    N: Neural,
{
    fn init_lateral(&self, neurons: &mut N) {
        (**self).init_lateral(neurons)
    }

    fn clone_dyn(&self) -> BoxedTopological<N> {
        panic!()
    }
}

impl<N> Clone for BoxedTopological<N>
where
    N: Neural,
{
    fn clone(&self) -> Self {
        (**self).clone_dyn()
    }
}

#[derive(Clone)]
pub struct CartesianTopology<D>
where
    D: Dimension,
{
    pub shape: Shape<D>,
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
    fn init_lateral(&self, neurons: &mut N) {
        neurons
            .get_lateral_mut()
            .assign(&get_ndindex_array(&self.shape));
    }

    fn clone_dyn(&self) -> BoxedTopological<N> {
        // Cloning leads to "parameter may not live long enough"
        // Box::new(self.clone())
        // Convert into dyn object first. Unclear whether this is the "right" thing to do
        let dim = self.shape.raw_dim().clone().into_dyn();
        return Box::new(CartesianTopology {
            shape: Shape::from(dim),
        });
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
