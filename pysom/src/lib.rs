use numpy::ndarray::ArrayViewMutD;
use numpy::ndarray::{ArrayD, ArrayViewD};
use numpy::{IntoPyArray, PyReadonlyArrayDyn};
use numpy::{PyArray2, PyArray3, PyArrayDyn, ToPyArray};
use pyo3::prelude::*;
use pyo3::{pymodule, types::PyModule, PyResult, Python};

use ndarray::prelude::*;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rand_isaac::isaac64::Isaac64Rng;
use som_rs::som::regular::CartesianGrid;
use som_rs::som::SelfOrganizingMap;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn pysom(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;

    fn mult(a: f64, mut x: ArrayViewMutD<'_, f64>) {
        x *= a;
    }

    #[pyfn(m)]
    #[pyo3(name = "mult")]
    fn mult_py(_py: Python<'_>, a: f64, x: &PyArrayDyn<f64>) {
        let x = unsafe { x.as_array_mut() };
        mult(a, x);
    }

    fn axpy(a: f64, x: ArrayViewD<'_, f64>, y: ArrayViewD<'_, f64>) -> ArrayD<f64> {
        a * &x + &y
    }

    // wrapper of `axpy`
    #[pyfn(m)]
    #[pyo3(name = "axpy")]
    fn axpy_py<'py>(
        py: Python<'py>,
        a: f64,
        x: PyReadonlyArrayDyn<f64>,
        y: PyReadonlyArrayDyn<f64>,
    ) -> &'py PyArrayDyn<f64> {
        let x = x.as_array();
        let y = y.as_array();
        let z = axpy(a, x, y);
        z.into_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "demo")]
    fn demo_som_py<'py>(py: Python<'py>, samples: usize, epochs: usize) -> &'py PyArray3<f64> {
        let seed = 42;
        let mut rng = Isaac64Rng::seed_from_u64(seed);

        let mut som = CartesianGrid::new((10, 10), 2, Uniform::new(0., 9.), &mut rng);
        // println!("{:?}", som);

        let training = Array::random_using((samples, 2), Uniform::new(0., 9.), &mut rng);
        // println!("{:?}", training);

        som.batch(training, None, None, Some(epochs));

        let training = som.get_feature().view().into_shape((10, 10, 2)).unwrap();

        training.to_pyarray(py)

        // .clone().to_pyarray(py)
    }
    Ok(())
}
