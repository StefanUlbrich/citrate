use ndarray::prelude::*;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rand_isaac::isaac64::Isaac64Rng;
use som_rs::som::cartesian::CartesianGrid;
use som_rs::som::SelfOrganizingMap;

fn main() {
    let seed = 42;
    let mut rng = Isaac64Rng::seed_from_u64(seed);

    let mut som = CartesianGrid::new((10, 10), 2, Uniform::new(0., 9.), &mut rng);
    // println!("{:?}", som);

    let training = Array::random_using((5000, 2), Uniform::new(0., 9.), &mut rng);
    println!("{:?}", training);

    som.batch(&training, None, None, None);

    let trained = som.get_feature().view().into_shape((10, 10, 2));

    println!("{:?}", trained.unwrap().sum_axis(Axis(1)));

    // Use this to test!
    // x = a[1:,:,:] - a[:-1,:,:]
    // y = a[:, 1:,:] - a[:, :-1,:]
    // x.mean(axis=0).mean(axis=0), x.std(axis=0).std(axis=0), y.mean(axis=0).mean(axis=0), y.std(axis=0).std(axis=0)
    // (array([-0.03560061, -0.90005994]),
    //  array([0.05568317, 0.02590234]),
    //  array([ 0.90100552, -0.01948181]),
    //  array([0.01654651, 0.04664077]))
}
