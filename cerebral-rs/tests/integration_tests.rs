use ndarray::prelude::*;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rand_isaac::isaac64::Isaac64Rng;

use cerebral::{default::*, BoxedSelforganizing, BoxedTrainable, Neural};
use cerebral::{NeuralLayer, Selforganizing, SelforganizingNetwork};

use tracing_test::traced_test;

use itertools::Itertools;

#[test]
#[traced_test]
fn test_kohonen() {
    let seed = 31;

    let mut rng = Isaac64Rng::seed_from_u64(seed);

    let (nx, ny, n): (usize, usize, usize) = (10, 10, 5000);

    let mut som = SelforganizingNetwork {
        neurons: NeuralLayer {
            // lateral: Array2::<f64>::zeros((0,0)),
            patterns: Array::random_using((nx * ny, 2), Uniform::new(0., 10.), &mut rng),
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

    // Warning group by only groups consecutive equal keys. You need to sort
    // to achieve the "regular" group by.
    let bmu: Vec<_> = training
        .axis_iter(Axis(0))
        .map(|x| (som.get_best_matching(&x)))
        .sorted()
        .group_by(|x| x.clone())
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
    let max_rel_error = rel_errors.iter().max_by(|a, b| a.partial_cmp(b).unwrap());

    assert!(avg_rel_error < 0.2);

    println!(
        "avg: {:?}, max: {:?}",
        avg_rel_error, //(|a, b| a.partial_cmp(b).unwrap())
        max_rel_error
    );
}

#[test]

fn test_boxed() {
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
    // does not work (see next unit test)
    // let adaptivity = Box::<dyn Adaptable>::new(KohonenAdaptivity {}) ;

    // println!("{}", som.neurons.lateral);
    let mut som = SelforganizingNetwork {
        neurons: NeuralLayer {
            // lateral: Array2::<f64>::zeros((0,0)),
            patterns: Array::random_using((100, 2), Uniform::new(0., 10.), &mut rng),
            ..Default::default()
        },
        adaptivity: KohonenAdaptivity {},
        topology: CartesianTopology::new((10, 10)),
        responsiveness: CartesianResponsiveness {},
        training: training,
    };
    som.init_lateral();
    let training = Array::random_using((5000, 2), Uniform::new(0., 9.), &mut rng);
    som.train(&training.view());
    som.adapt(&training.row(0), 0.7, 0.7);
}

#[test]
fn test_boxed_2() {
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
    // let adaptivity = Box::<dyn Adaptable>::new(KohonenAdaptivity {}) ;

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
        })
    }

    // does not workworks not
    // let mut som = Box::<dyn Selforganizing>::new(..);

    // That way, it works
    // let mut som: Box<dyn Selforganizing> = create_som(training);

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
