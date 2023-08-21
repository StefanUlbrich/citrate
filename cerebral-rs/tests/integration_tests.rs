use ndarray::prelude::*;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rand_isaac::isaac64::Isaac64Rng;

use cerebral::{default::*, Adaptable, BoxedAdaptable, BoxedSelforganizing, BoxedTrainable};
use cerebral::{NeuralLayer, Selforganizing, SelforganizingNetwork};

use tracing_test::traced_test;

use itertools::Itertools;

/// Checks whether convergence works in a simple case.
///
/// It trains on uniformely distributed points in
/// the 2D plane and assumes that the learned net is
/// a Voronoi decomposition of the 2D plane. Each cell
/// would then hold approximately the same amount of samples.
/// We then check if the average amount per cell deviates
/// less than 20% of the expected average. Note that
/// there will be outlier so checking for the max deviation
/// does not make sense.
#[test]
#[traced_test]
fn test_kohonen_conversion() {
    let seed = 31;

    let mut rng = Isaac64Rng::seed_from_u64(seed);

    let (nx, ny, n): (usize, usize, usize) = (10, 10, 5000);

    let mut som = SelforganizingNetwork {
        neurons: NeuralLayer {
            // lateral: Array2::<f64>::zeros((0,0)),
            patterns: Array::random_using((nx * ny, 2), Uniform::new(0., (nx as f64)), &mut rng),
            ..Default::default()
        },
        adaptivity: KohonenAdaptivity {},
        topology: CartesianTopology::new((nx, ny)),
        responsiveness: CartesianResponsiveness {},
        training: IncrementalLearning {
            radii: (2.5, 0.15),
            rates: (0.9, 0.01),
            epochs: 5,
        },
    };

    // println!("{}", som.neurons.lateral);

    som.init_lateral();
    let training = Array::random_using((n, 2), Uniform::new(0., 9.), &mut rng);
    som.adapt(&training.row(0), 0.7, 0.7);
    som.train(&training.view());

    // println!("{:?}", som.get_patterns());

    let bmu: Vec<_> = training
        .axis_iter(Axis(0))
        .map(|x| (som.get_best_matching(&x)))
        .sorted()
        .group_by(|x| x.clone())
        // Warning group by only groups consecutive equal keys. You need to sort
        // to achieve the "regular" group by.
        .into_iter()
        .map(|(g, group)| (g, group.count()))
        .collect();

    println!("{:?}", bmu.iter().count());

    // abs(i - n / nx / ny) n * nx * ny

    for i in bmu.iter() {
        println!("{:?}", i);
    }

    // Average expected samples per neuron (equally distributed)
    let avg = (n as f64) / (nx as f64) / (nx as f64);

    // relative error
    let rel_errors: Vec<_> = bmu
        .iter()
        .map(|x| ((x.1 as f64) - avg).abs() / avg)
        .collect();

    for i in rel_errors.iter() {
        println!("{:?}", i);
    }

    let avg_rel_error = rel_errors.iter().sum::<f64>() / (nx as f64) / (nx as f64);

    // https://stackoverflow.com/a/28446718
    let max_rel_error = rel_errors.iter().max_by(|a, b| a.partial_cmp(b).unwrap());

    assert!(avg_rel_error < 0.2);

    println!(
        "avg: {:?}, max: {:?}",
        avg_rel_error, //(|a, b| a.partial_cmp(b).unwrap())
        max_rel_error
    );
}

/// Checks whether components of the model can be
/// created dynamically at runtime.
/// Does not check the validity of the solution (i.e., conversion)
/// as in [test_kohonen_conversion]; only whether it compiles
/// and runs.
#[test]
fn test_boxed_components() {
    let seed = 42;
    let mut rng = Isaac64Rng::seed_from_u64(seed);

    // does not work (see next unit test)
    // let adaptivity = Box::<dyn Adaptable>::new(KohonenAdaptivity {}) ;
    let adaptivity: BoxedAdaptable<NeuralLayer, CartesianResponsiveness> =
        Box::new(KohonenAdaptivity {});

    fn create() -> BoxedTrainable<
        NeuralLayer,
        BoxedAdaptable<NeuralLayer, CartesianResponsiveness>,
        CartesianResponsiveness,
    > {
        Box::new(IncrementalLearning {
            radii: (2.0, 0.2),
            rates: (0.7, 0.1),
            epochs: 1,
        })
    }
    let training = create();

    // println!("{}", som.neurons.lateral);
    let mut som = SelforganizingNetwork {
        neurons: NeuralLayer {
            // lateral: Array2::<f64>::zeros((0,0)),
            patterns: Array::random_using((100, 2), Uniform::new(0., 10.), &mut rng),
            ..Default::default()
        },
        adaptivity: adaptivity,
        topology: CartesianTopology::new((10, 10)),
        responsiveness: CartesianResponsiveness {},
        training: training,
    };
    som.init_lateral();
    let training = Array::random_using((5000, 2), Uniform::new(0., 9.), &mut rng);
    som.train(&training.view());
    som.adapt(&training.row(0), 0.7, 0.7);
}

/// checks various methods of whether the model itself can be boxed
#[test]
fn test_boxed_model() {
    let seed = 42;
    let mut rng = Isaac64Rng::seed_from_u64(seed);

    fn create() -> BoxedTrainable<NeuralLayer, KohonenAdaptivity, CartesianResponsiveness> {
        Box::new(IncrementalLearning {
            radii: (2.0, 0.2),
            rates: (0.7, 0.1),
            epochs: 1,
        })
    }
    let training = create();
    // let adaptivity = Box::<dyn Adaptable>::new(KohonenAdaptivity {});

    fn create_som(
        training: BoxedTrainable<NeuralLayer, KohonenAdaptivity, CartesianResponsiveness>,
    ) -> BoxedSelforganizing {
        let seed = 42;

        let mut rng = Isaac64Rng::seed_from_u64(seed);
        Box::new(SelforganizingNetwork {
            neurons: NeuralLayer {
                // lateral: Array2::<f64>::zeros((0,0)),
                patterns: Array::random_using((100, 3), Uniform::new(0., 10.), &mut rng),
                ..Default::default()
            },
            adaptivity: KohonenAdaptivity {},
            topology: CartesianTopology::new((10, 10)),
            responsiveness: CartesianResponsiveness {},
            training: training,
        })
    }

    // does not workworks not
    // let mut som = Box::<dyn Selforganizing>::new(..);

    // That way, it works
    let mut som: Box<dyn Selforganizing> = create_som(training);

    let training = create();

    // This works too
    let mut som: Box<dyn Selforganizing> = Box::new(SelforganizingNetwork {
        neurons: NeuralLayer {
            // lateral: Array2::<f64>::zeros((0,0)),
            patterns: Array::random_using((100, 3), Uniform::new(0., 10.), &mut rng),
            ..Default::default()
        },
        // adaptivity: adaptivity,
        adaptivity: KohonenAdaptivity {},
        topology: CartesianTopology::new((10, 10)),
        responsiveness: CartesianResponsiveness {},
        training: training,
        // training: BatchTraining {
        //     radii: (2.0, 0.2),
        //     rates: (0.7, 0.1),
        //     epochs: 1,
        // },
    });
    som.init_lateral();
    let training = Array::random_using((5000, 3), Uniform::new(0., 9.), &mut rng);
    som.train(&training.view());
    som.adapt(&training.row(0), 0.7, 0.7);
}
