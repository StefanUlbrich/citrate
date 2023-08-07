use ndarray::prelude::*;
use ndarray_npy::NpzWriter;
use potpourri::backend::ndarray::utils::generate_samples;
use std::fs::File;
use tracing::info;

/// Generates data for testing.
/// The user has to check manually whether
/// the clusters can be distinguised by sorting along the first axis.
/// Sorting is not trivial unfortunately.
fn main() {
    let (data, responsibilities, means, covariances) = generate_samples(&[1000, 2000, 3000], 2);

    let mut npz = NpzWriter::new(File::create("potpourri-rs/data/test.npz").unwrap());
    npz.add_array("data", &data).unwrap();
    npz.add_array("responsibilities", &responsibilities)
        .unwrap();
    npz.add_array("means", &means).unwrap();
    npz.add_array("covariances", &covariances).unwrap();
    npz.finish().unwrap();
    println!("{:?}", means);
    info!(%means);
}
