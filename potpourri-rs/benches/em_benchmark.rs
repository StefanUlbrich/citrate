use criterion::{black_box, criterion_group, criterion_main, Criterion};
use itertools::izip;
use ndarray::parallel::prelude::*;
use ndarray::prelude::*;
use ndarray_rand::rand_distr::Standard;
use ndarray_rand::RandomExt;
use potpourri::backend::ndarray::gaussian::Gaussian;
use potpourri::backend::ndarray::utils::{generate_samples, get_shape2};
use potpourri::Parametrizable;
use rand_distr::num_traits::zero;
use rand_distr::Distribution;
use statrs::distribution::MultivariateNormal;

// extern crate blas_src;

fn expect_bench(gaussian: &Gaussian, data: &Array2<f64>) {
    let (responsibilities, likelihood) = gaussian.expect(&data.view()).unwrap();
}

fn data(n: usize) -> (Array2<f64>, Array2<f64>, Array2<f64>, Array3<f64>) {
    generate_samples(&[10000, 10000, 10000], 2)
}

fn expect_benchmark(c: &mut Criterion) {
    let (samples, _, _, _) = data(30000);
    let gaussian = Gaussian::new();
    c.bench_function("expect_bench", |b| {
        b.iter(|| expect_bench(&gaussian, &samples))
    });
}

criterion_group!(benches, expect_benchmark);
criterion_main!(benches);
