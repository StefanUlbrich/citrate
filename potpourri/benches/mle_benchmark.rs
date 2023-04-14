use criterion::{black_box, criterion_group, criterion_main, Criterion};
use itertools::izip;
use ndarray::parallel::prelude::*;
use ndarray::prelude::*;
use ndarray_rand::rand_distr::Standard;
use ndarray_rand::RandomExt;
use potpourri::backend::ndarray::utils::get_shape2;
use rand_distr::num_traits::zero;
use rand_distr::Distribution;
use statrs::distribution::MultivariateNormal;

// extern crate blas_src;

fn loops(adjusted: &Array3<f64>) {
    let mut covs = Array3::<f64>::zeros((3, 2, 2));

    for (row, mut cov) in adjusted.axis_iter(Axis(1)).zip(covs.axis_iter_mut(Axis(0))) {
        // iterate over n
        for x in row.axis_iter(Axis(0)) {
            let x = x.slice(s![NewAxis, ..]);
            cov += &x.t().dot(&x);
            // cov.assign(&x.t().dot(&x));
        }
    }
}

fn iterators(adjusted: &Array3<f64>) {
    let mut covs = Array3::<f64>::zeros((3, 2, 2));

    adjusted
        .axis_iter(Axis(1))
        .zip(covs.axis_iter_mut(Axis(0)))
        .for_each(|(row, mut cov)| {
            cov +=
                // iterate over n
                &row.axis_iter(Axis(0))
                    .map(|x| {
                        let x = x.slice(s![NewAxis, ..]);
                        x.t().dot(&x)
                    })
                    .reduce(|x, y| &x + &y)
                    // .last()
                    .unwrap();
        });
}

fn iterators_assign(adjusted: &Array3<f64>) {
    let mut covs = Array3::<f64>::zeros((3, 2, 2));

    adjusted
        .axis_iter(Axis(1))
        .zip(covs.axis_iter_mut(Axis(0)))
        .for_each(|(row, mut cov)| {
            cov.assign(
                // iterate over n
                &row.axis_iter(Axis(0))
                    .map(|x| {
                        let x = x.slice(s![NewAxis, ..]);
                        x.t().dot(&x)
                    })
                    .reduce(|x, y| &x + &y)
                    .unwrap(),
            );
        });
}

fn izip(adjusted: &Array3<f64>) {
    let mut covs = Array3::<f64>::zeros((3, 2, 2));

    izip!(adjusted.axis_iter(Axis(1)), covs.axis_iter_mut(Axis(0))).for_each(|(row, mut cov)| {
        cov.assign(
            // iterate over n
            &row.axis_iter(Axis(0))
                .map(|x| {
                    let x = x.slice(s![NewAxis, ..]);
                    x.t().dot(&x)
                })
                .reduce(|x, y| &x + &y)
                .unwrap(),
        );
    });
}

fn parallel(adjusted: &Array3<f64>) {
    let mut covs = Array3::<f64>::zeros((3, 2, 2));

    (adjusted.axis_iter(Axis(1)), covs.axis_iter_mut(Axis(0)))
        .into_par_iter()
        .for_each(|(row, mut cov)| {
            cov.assign(
                // iterate over n
                &row.axis_iter(Axis(0))
                    .map(|x| {
                        let x = x.slice(s![NewAxis, ..]);
                        x.t().dot(&x)
                    })
                    .reduce(|x, y| &x + &y)
                    .unwrap(),
            );
        });
}

fn iterators_no_reduce(adjusted: &Array3<f64>) {
    let mut covs = Array3::<f64>::zeros((3, 2, 2));

    adjusted
        .axis_iter(Axis(1))
        .zip(covs.axis_iter_mut(Axis(0)))
        .for_each(|(row, mut cov)| {
            // iterate over n
            row.axis_iter(Axis(0)).for_each(|x| {
                let x = x.slice(s![NewAxis, ..]);
                cov += &x.t().dot(&x);
            })
        });
}

fn parallel_no_reduce(adjusted: &Array3<f64>) {
    let mut covs = Array3::<f64>::zeros((3, 2, 2));

    (adjusted.axis_iter(Axis(1)), covs.axis_iter_mut(Axis(0)))
        .into_par_iter()
        .for_each(|(row, mut cov)| {
            row.axis_iter(Axis(0))
                // .into_par_iter() // cannot borrow cov any more
                .for_each(|x| {
                    let x = x.slice(s![NewAxis, ..]);
                    cov += &x.t().dot(&x);
                });
        });
}

fn sufficient_statistics(data: &Array2<f64>, responsibilities: &Array2<f64>) {
    let [_n, d] = get_shape2(&data.view()).unwrap();
    let [_n, k] = get_shape2(&responsibilities.view()).unwrap();

    let mut covs = Array3::<f64>::zeros((k, d, d));

    // Einstein sum: NxD, NxK -> KxDxD ??
    (
        covs.axis_iter_mut(Axis(0)),
        responsibilities.axis_iter(Axis(1)),
    ) // iterate k (can be parallelized)
        .into_par_iter()
        .for_each(|(mut cov, resp)| {
            data.axis_iter(Axis(0)) // iterate n (cannot be parallelized)
                .zip(resp.axis_iter(Axis(0)))
                .for_each(|(x, r)| {
                    let x = x.slice(s![NewAxis, ..]);
                    let x = &r.slice(s![NewAxis, NewAxis]) * &x.t().dot(&x);
                    cov += &x;
                });
        });
}

fn data(n: usize) -> (Array2<f64>, Array2<f64>, Array3<f64>) {
    // TODO simplify!
    let mut rng = rand::thread_rng();

    let covariance = (
        Array2::<f64>::random((2, 2), Standard),
        Array2::<f64>::random((2, 2), Standard),
        Array2::<f64>::random((2, 2), Standard),
    );
    let covariance = (
        covariance.0.t().dot(&covariance.0),
        covariance.1.t().dot(&covariance.1),
        covariance.2.t().dot(&covariance.2),
    );
    let mvn = (
        MultivariateNormal::new(vec![1.0, 2.0], covariance.0.into_raw_vec()).unwrap(),
        MultivariateNormal::new(vec![2.0, 1.0], covariance.1.into_raw_vec()).unwrap(),
        MultivariateNormal::new(vec![3.0, 3.0], covariance.2.into_raw_vec()).unwrap(),
    );
    let mut samples = Array2::<f64>::default((n, 2));
    let mut responsibilities = Array2::<f64>::default((n, 3));
    // we could use izip too: https://stackoverflow.com/a/29669741
    for (i, (mut s_row, mut r_row)) in samples
        .axis_iter_mut(Axis(0))
        .zip(responsibilities.axis_iter_mut(Axis(0)))
        .enumerate()
    {
        let (s, r) = if i > (2.0 / 3.0 * n as f64) as usize {
            (mvn.0.sample(&mut rng), array![1.0, 0.0, 0.0])
        } else if i > (1.0 / 3.0 * n as f64) as usize {
            (mvn.1.sample(&mut rng), array![0.0, 1.0, 0.0])
        } else {
            (mvn.2.sample(&mut rng), array![0.0, 0.0, 1.0])
        };

        // Convert nalgebra matrix to ndarray!
        // from ndarray to nalgebra: https://github.com/rust-ndarray/ndarray-linalg/issues/121#issuecomment-441818907

        // This should be zero copy (data does not have copy trait)
        let ss = Array::from_shape_vec((2,), s.data.into()).unwrap();
        s_row.assign(&ss);
    }

    let sum_responsibilities = responsibilities.sum_axis(Axis(0));

    // todo: check whether div/mul take ownership of arguments (desired)

    let means = (&responsibilities.slice(s![.., .., NewAxis])
        * &samples.slice(s![.., NewAxis, ..]))
        .sum_axis(Axis(0))
        / sum_responsibilities.slice(s![.., NewAxis]);

    let adjusted = (&samples.slice(s![.., NewAxis, ..]) - &means.slice(s![NewAxis, .., ..]))
        * responsibilities.slice(s![.., .., NewAxis]);

    (samples, responsibilities, adjusted)
}

fn loops_benchmark(c: &mut Criterion) {
    let (samples, responsibilities, adjusted) = data(30000);
    c.bench_function("loops", |b| b.iter(|| loops(&adjusted)));
}
fn iterators_benchmark(c: &mut Criterion) {
    let (samples, responsibilities, adjusted) = data(30000);
    c.bench_function("iterators", |b| b.iter(|| iterators(&adjusted)));
}
fn iterators_benchmark2(c: &mut Criterion) {
    let (samples, responsibilities, adjusted) = data(30000);
    c.bench_function("iterators assign", |b| {
        b.iter(|| iterators_assign(&adjusted))
    });
}

fn izip_benchmark(c: &mut Criterion) {
    let (samples, responsibilities, adjusted) = data(30000);
    c.bench_function("izip", |b| b.iter(|| izip(&adjusted)));
}
fn parallel_benchmark(c: &mut Criterion) {
    let (samples, responsibilities, adjusted) = data(30000);
    c.bench_function("parallel", |b| b.iter(|| parallel(&adjusted)));
}

fn iterators_benchmark_no_reduce(c: &mut Criterion) {
    let (samples, responsibilities, adjusted) = data(30000);
    c.bench_function("iterators_no_reduce", |b| {
        b.iter(|| iterators_no_reduce(&adjusted))
    });
}
fn parallel_benchmark_no_reduce(c: &mut Criterion) {
    let (samples, responsibilities, adjusted) = data(30000);
    c.bench_function("parallel_no_reduce", |b| {
        b.iter(|| parallel_no_reduce(&adjusted))
    });
}
fn sufficient_statistics_benchmark(c: &mut Criterion) {
    let (samples, responsibilities, adjusted) = data(30000);
    c.bench_function("sufficient_statistics", |b| {
        b.iter(|| sufficient_statistics(&samples, &responsibilities))
    });
}

criterion_group!(
    benches,
    loops_benchmark,
    iterators_benchmark,
    iterators_benchmark2,
    izip_benchmark,
    parallel_benchmark,
    iterators_benchmark_no_reduce,
    parallel_benchmark_no_reduce,
    sufficient_statistics_benchmark
);
criterion_main!(benches);
