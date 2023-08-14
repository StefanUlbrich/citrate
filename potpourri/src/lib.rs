use numpy::{IntoPyArray, PyReadonlyArray2, PyArray2, PyArray3, PyArray1};
use pyo3::{pymodule, types::PyModule, PyResult, Python};

use ndarray::parallel::prelude::*;
use ndarray::prelude::*;


#[pymodule]
fn potpourri(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // example using immutable borrows producing a new array
    // fn axpy(a: f64, x: ArrayViewD<'_, f64>, y: ArrayViewD<'_, f64>) -> ArrayD<f64> {
    //     a * &x + &y
    // }

    // wrapper of `axpy`
    #[pyfn(m)]
    #[pyo3(name = "maximize")]
    fn maximize_py<'py>(
        py: Python<'py>,
        data: PyReadonlyArray2<f64>,
        responsibilities: PyReadonlyArray2<f64>,
    ) { // -> (&'py PyArray2<f64>, &'py PyArray3<f64>, &'py PyArray1<f64> ){

    }

    Ok(())
}
