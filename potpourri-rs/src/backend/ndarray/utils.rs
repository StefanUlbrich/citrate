//! Additional support functions

use ndarray::parallel::prelude::*;
use ndarray::prelude::*;

use crate::Error;
// use ndarray_rand::rand::prelude::*;
use ndarray_rand::rand_distr::Standard;
use ndarray_rand::RandomExt;
use ndarray_rand::{rand, rand_distr::Dirichlet};
use rand_distr::Distribution;

use statrs::distribution::MultivariateNormal;

use ndarray_linalg::cholesky::*;

/* ****************************************************************************
 *                               Convenience
 * ****************************************************************************/

/// Gets the shape of an Array3 object or raise an error if not possible
#[inline(always)]
pub fn get_shape3(array: &Array3<f64>) -> Result<[usize; 3], Error> {
    Ok(if let [n, m, k] = array.shape() {
        [*n, *m, *k]
    } else {
        return Err(Error::DimensionMismatch);
    })
}

// FIXME make generic over the DataOwnership
/// Gets the shape of an Array2 object or raise an error if not possible
#[inline(always)]
pub fn get_shape2(array: &ArrayView2<f64>) -> Result<[usize; 2], Error> {
    Ok(if let [n, m] = array.shape() {
        [*n, *m]
    } else {
        return Err(Error::DimensionMismatch);
    })
}

// TODO make this conditional depending on the lib (nd-linalg/nalgbra)
#[inline(always)]
pub fn invert_spd(matrix: &ArrayView2<f64>) -> Result<Array2<f64>, Error> {
    Ok(matrix.invc()?)
}

#[inline(always)]
pub fn get_det_spd(matrix: &ArrayView2<f64>) -> Result<f64, Error> {
    Ok(matrix.detc()?)
}

/* ****************************************************************************
 *                               Data Generation
 * ****************************************************************************/

/// Create data generated with a Gaussian mixture model.
/// Returns $n_1+\ldots +n_k$ samples from a Gaussian mixture with $k$ components
/// in a $d$-dimensional feature space. It also returns the $(n_1+\ldots +n_k) \times k$
/// "true" responisiblity matrix (i.e., only ones and zeros in its elements).
/// For testing, it returns also the generated covariances
/// Returns: (samples, responsibilities, means, covariances)
pub fn generate_samples(
    nk: &[usize],
    d: usize,
) -> (Array2<f64>, Array2<f64>, Array2<f64>, Array3<f64>) {
    // TODO simplify!

    let n_total = nk.into_iter().sum();
    let k = nk.len();

    let mut covariances = Array3::<f64>::default((k, d, d));
    covariances.axis_iter_mut(Axis(0)).for_each(|mut x| {
        let y = Array2::<f64>::random((d, d), Standard) / 5.0;
        x.assign(&y.t().dot(&y));
    });

    let mut means = Array2::<f64>::default((k, d));
    means.axis_iter_mut(Axis(0)).for_each(|mut x| {
        x.assign(&Array1::<f64>::random(d, Standard));
    });

    let mvn: Vec<_> = (means.axis_iter(Axis(0)), covariances.axis_iter(Axis(0)))
        .into_par_iter()
        .map(|(m, x)| {
            MultivariateNormal::new(m.into_owned().into_raw_vec(), x.into_owned().into_raw_vec())
                .unwrap()
        })
        .collect();

    let mut samples = Array2::<f64>::default((n_total, d));
    let mut responsibilities = Array2::<f64>::default((n_total, k));

    (
        samples.axis_iter_mut(Axis(0)),
        responsibilities.axis_iter_mut(Axis(0)),
    )
        .into_par_iter()
        .enumerate()
        .for_each(|(i, (mut s_row, mut r_row))| {
            let mut component: usize = 0;
            component = loop {
                if i < nk[0..component + 1].into_iter().sum() {
                    break component;
                }
                component += 1;
            };
            let sample = mvn[component].sample(&mut rand::thread_rng());
            r_row[component] = 1.0;

            s_row.assign(&Array::from_shape_vec((2,), sample.data.into()).unwrap());
        });

    (samples, responsibilities, means, covariances)
}

// todo: remove when above works
// pub fn generate_samples_old(
//     n: usize,
//     k: usize,
//     d: usize,
// ) -> (Array2<f64>, Array2<f64>, Array2<f64>, Array3<f64>) {
//     // TODO simplify!

//     let mut covariances = Array3::<f64>::default((k, d, d));
//     covariances.axis_iter_mut(Axis(0)).for_each(|mut x| {
//         let y = Array2::<f64>::random((d, d), Standard) / 5.0;
//         x.assign(&y.t().dot(&y));
//     });

//     let mut means = Array2::<f64>::default((k, d));
//     means.axis_iter_mut(Axis(0)).for_each(|mut x| {
//         x.assign(&Array1::<f64>::random(d, Standard));
//     });

//     let mvn: Vec<_> = (means.axis_iter(Axis(0)), covariances.axis_iter(Axis(0)))
//         .into_par_iter()
//         .map(|(m, x)| {
//             MultivariateNormal::new(m.into_owned().into_raw_vec(), x.into_owned().into_raw_vec())
//                 .unwrap()
//         })
//         .collect();

//     let mut samples = Array2::<f64>::default((n, d));
//     let mut responsibilities = Array2::<f64>::default((n, k));

//     (
//         samples.axis_iter_mut(Axis(0)),
//         responsibilities.axis_iter_mut(Axis(0)),
//     )
//         .into_par_iter()
//         .enumerate()
//         .for_each(|(i, (mut s_row, mut r_row))| {
//             let component = i / (n / k);
//             let sample = mvn[component].sample(&mut rand::thread_rng());
//             r_row[component] = 1.0;

//             s_row.assign(&Array::from_shape_vec((2,), sample.data.into()).unwrap());
//         });

//     (samples, responsibilities, means, covariances)
// }

/// Splits a dataset consiting of two arrays according to a row-wise criteria
pub fn filter_data<F>(
    data_a: &ArrayView2<f64>,
    data_b: &ArrayView2<f64>,
    predicate: F,
) -> Result<(Array2<f64>, Array2<f64>), Error>
where
    F: Fn(&ArrayView1<f64>, &ArrayView1<f64>) -> bool + Sync,
{
    let [_, a] = get_shape2(&data_a)?;
    let [_, b] = get_shape2(&data_b)?;

    let selected_a: Vec<_> = (data_a.axis_iter(Axis(0)), data_b.axis_iter(Axis(0)))
        .into_par_iter()
        .filter(|(x, y)| predicate(&x, &y))
        .map(|(x, _y)| x.into_owned().into_raw_vec())
        .collect();

    let n = selected_a.len();

    let selected_a: Vec<f64> = selected_a.iter().flatten().cloned().collect();

    let selected_b: Vec<_> = (data_a.axis_iter(Axis(0)), data_b.axis_iter(Axis(0)))
        .into_par_iter()
        .filter(|(x, y)| predicate(&x, &y))
        .map(|(_x, y)| y.into_owned().into_raw_vec())
        .collect();

    let selected_b: Vec<_> = selected_b.iter().flatten().cloned().collect();

    Ok((
        Array::from_shape_vec((n, a), selected_a)?,
        Array::from_shape_vec((n, b), selected_b)?,
    ))
}

/* ****************************************************************************
 *                      Shared functionality for models
 * ****************************************************************************/

/// Generate random initializations from a dirichlet distribution.
#[inline(always)]
pub fn generate_random_expections(data: &ArrayView2<f64>, k: usize) -> Result<Array2<f64>, Error> {
    let [n, _d] = get_shape2(data)?;
    let alpha: Vec<_> = (0..k).map(|_| 1.0f64).collect();
    let dirichlet = Dirichlet::new(&alpha).unwrap();

    let mut x = Array2::<f64>::zeros((n, k));
    x.axis_iter_mut(Axis(0)).for_each(|mut x| {
        x.assign(&Array::from_shape_vec(k, dirichlet.sample(&mut rand::thread_rng())).unwrap())
    });
    Ok(x)
}

/// Get the  means ($k\times d$) of the $n$ weighted samples
/// for $k$ components using the weighted sum of the samples ($n \times k \times d$)
/// and the sum of the responsibilities ($k$).
/// Useful for multiple distributions (e.g., Gaussian, Poisson).
#[inline(always)]
pub(crate) fn get_weighted_means(
    weighted_sum: &Array2<f64>,
    sum_responsibilities: &Array1<f64>,
) -> Array2<f64> {
    weighted_sum / &sum_responsibilities.slice(s![.., NewAxis])
}

/// Get the sum ($n \times k \times d$) of the samples ($n \times d$) weighted by the
/// responsibilities ($k \times d$)
/// Useful for multiple distributions (e.g., Gaussian, Poisson).
#[inline(always)]
pub(crate) fn get_weighted_sum(
    samples: &ArrayView2<f64>,
    responsibilities: &Array2<f64>,
) -> Array2<f64> {
    (&responsibilities.slice(s![.., .., NewAxis]) * &samples.slice(s![.., NewAxis, ..]))
        .sum_axis(Axis(0))
}

/// Mean adjust samples ($n \times d$) set given a responsibility matrix
/// ($n \times k$). Useful for multiple distributions. Returns a $n \times k \times d$
/// array.
/// Obsolete with the use of sufficient statistics
// pub(crate) fn adjust_weighted_means(
//     samples: &ArrayView2<f64>,
//     responsibilities: &Array2<f64>,
//     means: &Array2<f64>,
// ) -> Array3<f64> {
//     let adjusted = (&samples.slice(s![.., NewAxis, ..]) - &means.slice(s![NewAxis, .., ..]))
//         * responsibilities.slice(s![.., .., NewAxis]);

//     adjusted
// }

#[cfg(test)]
mod tests {
    use ndarray::Array2;
    use ndarray_rand::RandomExt;
    use rand_distr::Standard;

    use crate::backend::ndarray::utils::invert_spd;

    // use tracing::debug;
    // use tracing_test::traced_test;

    // #[traced_test]
    #[test]
    fn check_invc_1() {
        let n = 5;
        let mut matrix = Array2::<f64>::random((n, n), Standard);
        matrix = matrix.t().dot(&matrix);
        let inverted = invert_spd(&matrix.view()).unwrap();
        // debug!(%matrix);
        // debug!(%inverted);
        let eye = &matrix.dot(&inverted);
        assert!(eye.abs_diff_eq(&Array2::<f64>::eye(n), 1e-5));
        let eye = &inverted.dot(&matrix);
        assert!(eye.abs_diff_eq(&Array2::<f64>::eye(n), 1e-5));
    }
}
