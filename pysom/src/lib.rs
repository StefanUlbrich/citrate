use pyo3::prelude::*;
use pyo3::Python;

use ndarray::Shape;
use ndarray::{prelude::*, Ix};
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::{rand_distr::Uniform, RandomExt};
use pyo3::types::PySequence;
use rand_isaac::isaac64::Isaac64Rng;

// use som_rs::som::cartesian::CartesianGrid;
// use som_rs::som::SelfOrganizingMap;

use som_rs::{default::*, BoxedSelforganizingNeural};
use som_rs::{BoxedAdaptable, BoxedResponsive, BoxedTopological, BoxedTrainable};

// use som_rs::neurons;
use som_rs::{NeuralLayer, Neurons};

#[pyclass]
#[derive(Clone)]
struct PyAdaptivity {
    // Why send? https://stackoverflow.com/a/60109068
    __component: BoxedAdaptable<Neurons, BoxedResponsive<Neurons>>,
}
#[pymethods]
impl PyAdaptivity {
    #[new]
    fn new() -> Self {
        PyAdaptivity {
            __component: Box::new(KohonenAdaptivity {}),
        }
    }
}

#[pyclass]
#[derive(Clone)]
struct PyResponsiveness {
    // __component: Arc<dyn Responsive<Neurons> >,
    __component: BoxedResponsive<Neurons>,
}
#[pymethods]
impl PyResponsiveness {
    #[new]
    fn new() -> Self {
        PyResponsiveness {
            __component: Box::new(CartesianResponsiveness {}),
        }
    }
}

#[pyclass]
#[derive(Clone)]
struct PyTopology {
    __component: BoxedTopological<Neurons>,
}
#[pymethods]
impl PyTopology {
    #[new]
    fn new(shape: &PySequence) -> Result<Self, PyErr> {
        let shape: Vec<Ix> = shape.extract()?;
        let shape = Dim(shape);
        let shape = Shape::from(shape);
        Ok(PyTopology {
            __component: Box::new(CartesianTopology { shape: shape }),
        })
    }
}

#[pyclass]
#[derive(Clone)]
struct PyTraining {
    __component: BoxedTrainable<
        Neurons,
        BoxedAdaptable<Neurons, BoxedResponsive<Neurons>>,
        BoxedResponsive<Neurons>,
    >,
}
#[pymethods]
impl PyTraining {
    /// .
    #[new]
    fn new(radii: (f64, f64), rates: (f64, f64), epochs: usize) -> Self {
        PyTraining {
            __component: Box::new(BatchTraining {
                radii,
                rates,
                epochs,
            }),
        }
    }
}

#[pyclass]
struct PyNeuralLayer {
    __som: BoxedSelforganizingNeural,
}

#[pymethods]
impl PyNeuralLayer {
    #[new]
    fn new(
        adaptivity: PyAdaptivity,
        topology: PyTopology,
        responsiveness: PyResponsiveness,
        training: PyTraining,
    ) -> Self {
        let seed = 42;
        let mut rng = Isaac64Rng::seed_from_u64(seed);
        let neurons = Neurons {
            lateral: Array::random_using((1, 1), Uniform::new(0., 10.), &mut rng),
            patterns: Array::random_using((1, 1), Uniform::new(0., 10.), &mut rng),
            ..Default::default()
        };
        return PyNeuralLayer {
            __som: Box::new(NeuralLayer {
                neurons: neurons,
                adaptivity: adaptivity.__component,
                topology: topology.__component,
                responsiveness: responsiveness.__component,
                training: training.__component,
            }),
        };
    }
}

// #[pymethods]
// impl PyNeuralLayer {

//     #[new]
//     fn new(shape: (usize, usize), output_dim: usize /*, string->parameters */) -> Self {
//         let seed = 42;
//         let mut rng = Isaac64Rng::seed_from_u64(seed);
//         let mut som = NeuralLayer {
//             neurons: Neurons {
//                 lateral: Array::random_using(shape, Uniform::new(0., 10.), &mut rng),
//                 patterns: Array::random_using(
//                     (shape.0 * shape.1, output_dim),
//                     Uniform::new(0., 10.),
//                     &mut rng,
//                 ),
//                 ..Default::default()
//             },
//             adaptivity: KohonenAdaptivity {},
//             topology: CartesianTopology::new((10, 10)),
//             responsiveness: CartesianResponsiveness {},
//             training: BatchTraining {
//                 radii: (2.0, 0.2),
//                 rates: (0.7, 0.1),
//                 epochs: 1,
//             },
//         };

//         // println!("{}", som.neurons.lateral);

//         som.init_lateral();
//         PyNeuralLayer {
//             __som: Box::new(som),
//         }
//     }

//     #[getter]
//     fn get_feature<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
//         self.__som.get_patterns().to_pyarray(py)
//     }

//     fn get_best_matching(&mut self, feature: PyReadonlyArray1<f64>) -> usize {
//         self.__som.get_best_matching(&feature.as_array())
//     }

//     fn adapt(&mut self, feature: PyReadonlyArray1<f64>, influence: f64, rate: f64) {
//         self.__som.adapt(&feature.as_array(), influence, rate)
//     }
//     fn batch(
//         &mut self,
//         features: PyReadonlyArray2<f64>,
//         radii: Option<(f64, f64)>,
//         rates: Option<(f64, f64)>,
//         epochs: Option<usize>,
//     ) {
//         // if let Some(r) = rates {
//         //     self.__som.training.rates = r;
//         // }
//         // if let Some(e) = epochs {
//         //     self.__som.training.epochs = e;
//         // }
//         // if let Some(r) = radii {
//         //     self.__som.training.radii = r;
//         // }
//         self.__som.train(&features.as_array())
//     }
// }

#[pymodule]
fn pysom(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_class::<PyAdaptivity>()?;
    module.add_class::<PyResponsiveness>()?;
    module.add_class::<PyTopology>()?;
    module.add_class::<PyTraining>()?;
    module.add_class::<PyNeuralLayer>()?;

    Ok(())
}

// /// Formats the sum of two numbers as string.
// #[pyfunction]
// fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
//     Ok((a + b).to_string())
// }

// #[pyfunction]
// fn again(a: usize, b: usize) -> PyResult<String> {
//     Ok((a + b).to_string())
// }
// /// A Python module implemented in Rust.
// #[pymodule]
// fn pysom(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
//     m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
//     m.add_function(wrap_pyfunction!(again, m)?)?;

//     fn mult(a: f64, mut x: ArrayViewMutD<'_, f64>) {
//         x *= a;
//     }

//     #[pyfn(m)]
//     #[pyo3(name = "mult")]
//     fn mult_py(_py: Python<'_>, a: f64, x: &PyArrayDyn<f64>) {
//         let x = unsafe { x.as_array_mut() };
//         mult(a, x);
//     }

//     fn axpy(a: f64, x: ArrayViewD<'_, f64>, y: ArrayViewD<'_, f64>) -> ArrayD<f64> {
//         a * &x + &y
//     }

//     // wrapper of `axpy`
//     #[pyfn(m)]
//     #[pyo3(name = "axpy")]
//     fn axpy_py<'py>(
//         py: Python<'py>,
//         a: f64,
//         x: PyReadonlyArrayDyn<f64>,
//         y: PyReadonlyArrayDyn<f64>,
//     ) -> &'py PyArrayDyn<f64> {
//         let x = x.as_array();
//         let y = y.as_array();
//         let z = axpy(a, x, y);
//         z.into_pyarray(py)
//     }

//     #[pyfn(m)]
//     #[pyo3(name = "demo")]
//     fn demo_som_py<'py>(py: Python<'py>, samples: usize, epochs: usize) -> &'py PyArray3<f64> {
//         let seed = 42;
//         let mut rng = Isaac64Rng::seed_from_u64(seed);

//         let mut som = CartesianGrid::new((10, 10), 2, Uniform::new(0., 9.), &mut rng);
//         // println!("{:?}", som);

//         let training = Array::random_using((samples, 2), Uniform::new(0., 9.), &mut rng);
//         // println!("{:?}", training);

//         som.batch(&training, None, None, Some(epochs));

//         let training = som.get_feature().view().into_shape((10, 10, 2)).unwrap();

//         training.to_pyarray(py)

//         // .clone().to_pyarray(py)
//     }

//         m.add_class::<PyCartesianGrid>()?;
//     Ok(())
// }
