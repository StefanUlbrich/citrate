///! Naming convenctions
/// * traits: Start with capital letter and are adjectives
/// * structs: Start with capital letter and are substantives
pub mod neural_layer;
pub mod neurons;

pub mod adaptable;
pub mod responsive;
pub mod topological;
pub mod trainable;

pub use neural_layer::{NeuralLayer, SelfOrganizing};
pub use neurons::{Neural, Neurons};

pub use adaptable::Adaptable;
pub use responsive::Responsive;
pub use topological::Topological;
pub use trainable::Trainable;

pub mod default {
    pub use crate::adaptable::KohonenAdaptivity;
    pub use crate::responsive::CartesianResponsiveness;
    pub use crate::topological::CartesianTopology;
    pub use crate::trainable::BatchTraining;
}

// #[cfg(feature = "ndarray")]
pub mod nd_tools;

#[test]
fn main() {
    // use maps::{adaptable::ClassicAdaptivity, trainable::BatchTraining};
    // use cartesian_map::{topological::CartesianTopology, tunable::CartesianFeature};

    // let mut nn = NeuralLayer {
    //     neurons: Neurons{
    //         lateral: 3.0,
    //         patterns: 4.0,
    //     },
    //     adaptivity: ClassicAdaptivity {},
    //     topology: CartesianTopology { shape: (10,10)},
    //     tuning: CartesianFeature {},
    //     training: BatchTraining {
    //         start_rate: 0.9,
    //         end_rate: 0.1,
    //     },
    // };

    // nn = nn.adapt(&2.0);
    // println!("{}", nn.neurons.lateral);
}

// #[cfg(test)]
// mod tests {

//     #[test]
//     fn it_works() {
//         let result = 2 + 2;
//         assert_eq!(result, 4);
//     }

//     // fn test_uniform() {
//     //     let a = uniform((2, 3));
//     //     let b = array![
//     //         [0.0, 0.0],
//     //         [0.0, 1.0],
//     //         [1.0, 0.0],
//     //         [1.0, 1.0],
//     //         [2.0, 0.0],
//     //         [2.0, 1.0]
//     //     ];

//     //     assert_eq!(a, b);
//     // }
// }
