pub struct CartesianTopology<Sh> {
    shape: Sh,
}

// https://stackoverflow.com/a/43070938/9415551 (reduce number of cfg)
#[cfg(feature = "ndarray")]
mod ndarray_based {
    use ndarray::prelude::*;
    // use ndarray::Data;
    use super::CartesianTopology;
    use crate::nd_tools::ndindex::get_ndindex_array;
    use crate::{Neural, Topological};
    // use std::fmt::Debug;
    // use num_traits::Float;

    impl<Sh> Topological<Array2<f64>, Array2<f64>> for CartesianTopology<Sh>
    where
        // F: Float + Debug,
        // D1: Debug,
        // D2: Debug,
        // S: Data<Elem = T>,
        Sh: ShapeBuilder,
    {
        fn get_lateral_connections<N>(&mut self, neurons: &N) -> f64
        where
            N: Neural<Array2<f64>, Array2<f64>>,
        {
            todo!()
        }

        fn init_lateral<N>(&self, neurons: &mut N)
        where
            N: Neural<Array2<f64>, Array2<f64>>,
        {
            neurons
                .get_lateral_mut()
                .assign(&get_ndindex_array(self.shape));
        }
    }
}

#[cfg(not(feature = "ndarray"))]
mod standard_based {
    use super::CartesianTopology;
    use crate::{Neural, Topological};

    impl<V> Topological<V> for CartesianTopology
    where
        V: Debug,
    {
        fn get_lateral_connections<D>(&mut self, data: &D) -> f64
        where
            D: Neural<V>,
        {
            todo!()
        }

        fn init_lateral<D>(&self, data: &D)
        where
            D: Neural<V>,
        {
            todo!()
        }
    }
}

#[cfg(feature = "ndarray")]
pub use ndarray_based::*;
#[cfg(not(feature = "ndarray"))]
pub use standard_based::*;
