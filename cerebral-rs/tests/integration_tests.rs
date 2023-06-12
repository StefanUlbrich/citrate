use ndarray::prelude::*;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rand_isaac::isaac64::Isaac64Rng;

use cerebral::{default::*, BoxedSelforganizing, BoxedTrainable};
use cerebral::{NeuralLayer, Selforganizing, SelforganizingNetwork};

#[test]
fn test_kohonen() {
    let seed = 42;

    let mut rng = Isaac64Rng::seed_from_u64(seed);

    let mut som = SelforganizingNetwork {
        neurons: NeuralLayer {
            // lateral: Array2::<f64>::zeros((0,0)),
            patterns: Array::random_using((100, 3), Uniform::new(0., 10.), &mut rng),
            ..Default::default()
        },
        adaptivity: KohonenAdaptivity {},
        topology: CartesianTopology::new((10, 10)),
        responsiveness: CartesianResponsiveness {},
        training: IncrementalLearning {
            radii: (2.0, 0.2),
            rates: (0.7, 0.1),
            epochs: 1,
        },
    };

    // println!("{}", som.neurons.lateral);

    som.init_lateral();
    let training = Array::random_using((5000, 2), Uniform::new(0., 9.), &mut rng);
    som.train(&training.view());
    som.adapt(&training.row(0), 0.7, 0.7);
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
            patterns: Array::random_using((100, 3), Uniform::new(0., 10.), &mut rng),
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
    let training = Array::random_using((5000, 2), Uniform::new(0., 9.), &mut rng);
    som.train(&training.view());
    som.adapt(&training.row(0), 0.7, 0.7);
}
