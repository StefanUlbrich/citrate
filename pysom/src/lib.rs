
use pyo3::prelude::*;
use numpy::ndarray::{ArrayViewMutD};
use numpy::{PyArrayDyn};

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

    Ok(())
}