var searchIndex = JSON.parse('{\
"cerebral":{"doc":"Naming convenctions","t":"CCCCCCCCCCCCCAAAAAAAAIGCKLLKLADLLLLLLLLLLLLLLLLCCCCFAADLLLLLLFLLLFLLLLIKKFIDLLLLLLLKLKLKLKLLLMLMKLKLLLLGDILLLLKLLLLLLLKLLLLLLLLGIDKLMLLLLLLKLLKLLLLLKLLMMLLMKLMLLLGDILLLLKLLLLLLLLKLLLLMLLLLGDILLLKLLLLLMLLLMMLKLLLL","n":["Adaptable","BoxedAdaptable","BoxedResponsive","BoxedSelforganizing","BoxedTopological","BoxedTrainable","Neural","NeuralLayer","Responsive","Selforganizing","SelforganizingNetwork","Topological","Trainable","adaptable","default","nd_tools","neural","responsive","selforganizing","topological","trainable","Adaptable","BoxedAdaptable","KohonenAdaptivity","adapt","adapt","clone","clone_dyn","clone_dyn","kohonen","KohonenAdaptivity","adapt","borrow","borrow_mut","clone","clone_dyn","clone_into","deref","deref_mut","drop","from","init","into","to_owned","try_from","try_into","type_id","CartesianResponsiveness","CartesianTopology","IncrementalLearning","KohonenAdaptivity","argmin","ndindex","point_set","NdIndexIterator","borrow","borrow_mut","deref","deref_mut","drop","from","get_ndindex_array","init","into","into_iter","ndindex","next","try_from","try_into","type_id","PointSet","get_differences","get_distances","row_norm_l2","Neural","NeuralLayer","borrow","borrow_mut","default","deref","deref_mut","drop","from","get_lateral","get_lateral","get_lateral_mut","get_lateral_mut","get_patterns","get_patterns","get_patterns_mut","get_patterns_mut","init","into","lateral","new","patterns","set_lateral","set_lateral","set_patterns","set_patterns","try_from","try_into","type_id","BoxedResponsive","CartesianResponsiveness","Responsive","borrow","borrow_mut","clone","clone","clone_dyn","clone_dyn","clone_dyn","clone_into","deref","deref_mut","drop","from","get_best_matching","get_best_matching","get_best_matching","init","into","to_owned","try_from","try_into","type_id","BoxedSelforganizing","Selforganizing","SelforganizingNetwork","adapt","adapt","adaptivity","borrow","borrow_mut","deref","deref_mut","drop","from","get_best_matching","get_best_matching","get_lateral","get_lateral_distance","get_lateral_distance","get_lateral_mut","get_patterns","get_patterns_mut","init","init_lateral","init_lateral","into","neurons","responsiveness","set_lateral","set_patterns","topology","train","train","training","try_from","try_into","type_id","BoxedTopological","CartesianTopology","Topological","borrow","borrow_mut","clone","clone","clone_dyn","clone_dyn","clone_dyn","clone_into","deref","deref_mut","drop","from","init","init_lateral","init_lateral","init_lateral","into","new","shape","to_owned","try_from","try_into","type_id","BoxedTrainable","IncrementalLearning","Trainable","borrow","borrow_mut","clone","clone_dyn","clone_dyn","clone_into","deref","deref_mut","drop","epochs","from","init","into","radii","rates","to_owned","train","train","try_from","try_into","type_id"],"q":["cerebral","","","","","","","","","","","","","","","","","","","","","cerebral::adaptable","","","","","","","","","cerebral::adaptable::kohonen","","","","","","","","","","","","","","","","","cerebral::default","","","","cerebral::nd_tools","","","cerebral::nd_tools::ndindex","","","","","","","","","","","","","","","","cerebral::nd_tools::point_set","","","","cerebral::neural","","","","","","","","","","","","","","","","","","","","","","","","","","","","","cerebral::responsive","","","","","","","","","","","","","","","","","","","","","","","","cerebral::selforganizing","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","cerebral::topological","","","","","","","","","","","","","","","","","","","","","","","","","","cerebral::trainable","","","","","","","","","","","","","","","","","","","","","","",""],"d":["","","","","","","","","","","","","","","","This module defines extensions to the ndarray crate. …","","","","","","Interface for structures encapsulating algorithms for …","","","","","","","","","","","","","","","","","","","Returns the argument unchanged.","","Calls <code>U::from(self)</code>.","","","","","","","","","Returns the index of the smallest element of a vector. …","","Adds functions that extends 2D float arrays such that they …","","","","","","","Returns the argument unchanged.","Creates an array with rows that hold the indices generated …","","Calls <code>U::from(self)</code>.","","Creates an iterator that generates indices for an array of …","","","","","","Computes the difference of each row to a given <code>point</code> (1D)","Computes the Eucledean distance of each row to a given …","Computes the L2 norm for all rows of a <code>PointSet</code>","Provides access to the neurons of a neural network. The …","","","","","","","","Returns the argument unchanged.","","","","","","","","","","Calls <code>U::from(self)</code>.","Lateral layer that defines the topology. Can be …","","Tuning Patterns the neurons. This is the codebook. Row …","","","","","","","","","","Interface for structures encapsulating representations …","","","","","","","","","","","","Returns the argument unchanged.","","","","","Calls <code>U::from(self)</code>.","","","","","","Public trait that defines the concept of self organization","Struct that implements structural composition","Adapt the layer to an input pattern. Note this consumes …","","Algorithm for adaptivity","","","","","","Returns the argument unchanged.","Get the best matching neuron given a pattern","","","Get the distance/connection between a selected neuron and …","","","","","","Init the lateral connections according to network type","","Calls <code>U::from(self)</code>.","needs to be nested to share it with the algorithms","Algorithm to feature pattern matching and lateral …","","","Algorithm related to topology","","","Algorithm related to batch processing","","","","","","Interface for structures encapsulating representations of …","","","","","","","","","","","","Returns the argument unchanged.","","","","","Calls <code>U::from(self)</code>.","","","","","","","","","Interface for structures encapsulating algorithms for …","","","","","","","","","","","Returns the argument unchanged.","","Calls <code>U::from(self)</code>.","","","","","","","",""],"i":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,24,3,3,24,3,0,0,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,0,0,0,0,0,0,0,0,11,11,11,11,11,11,0,11,11,11,0,11,11,11,11,0,25,25,0,0,0,14,14,14,14,14,14,14,26,14,26,14,26,14,26,14,14,14,14,14,14,26,14,26,14,14,14,14,0,0,0,16,16,15,16,27,15,16,16,16,16,16,16,27,15,16,16,16,16,16,16,16,0,0,0,28,17,17,17,17,17,17,17,17,28,17,17,28,17,17,17,17,17,28,17,17,17,17,17,17,17,28,17,17,17,17,17,0,0,0,20,20,19,20,29,19,20,20,20,20,20,20,20,29,19,20,20,20,20,20,20,20,20,0,0,0,21,21,21,22,21,21,21,21,21,21,21,21,21,21,21,21,22,21,21,21,21],"f":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,[[1,2,2]],[[3,1,2,2]],[3,3],[[],3],[3,3],0,0,[[4,1,2,2]],[[]],[[]],[4,4],[4,3],[[]],[5],[5],[5],[[]],[[],5],[[]],[[]],[[],6],[[],6],[[],7],0,0,0,0,[8,5],0,0,0,[[]],[[]],[5],[5],[5],[[]],[9,[[10,[2]]]],[[],5],[[]],[[]],[[],11],[11,12],[[],6],[[],6],[[],7],0,[13,10],[13,8],[13,8],0,0,[[]],[[]],[[],14],[5],[5],[5],[[]],[[],10],[14,10],[[],10],[14,10],[[],10],[14,10],[[],10],[14,10],[[],5],[[]],0,[[],14],0,[[[10,[2]]]],[[14,[10,[2]]]],[[[10,[2]]]],[[14,[10,[2]]]],[[],6],[[],6],[[],7],0,0,0,[[]],[[]],[15,15],[16,16],[[],15],[15,15],[16,15],[[]],[5],[5],[5],[[]],[1,5],[[15,1],5],[[16,1],5],[[],5],[[]],[[]],[[],6],[[],6],[[],7],0,0,0,[[1,2,2]],[[17,1,2,2]],0,[[]],[[]],[5],[5],[5],[[]],[1,5],[[17,1],5],[17,10],[5,[[10,[2]]]],[[17,5],[[10,[2]]]],[17,10],[17,10],[17,10],[[],5],[[]],[17],[[]],0,0,[[17,[10,[2]]]],[[17,[10,[2]]]],0,[18],[[17,18]],0,[[],6],[[],6],[[],7],0,0,0,[[]],[[]],[19,19],[20,20],[[],19],[19,19],[20,19],[[]],[5],[5],[5],[[]],[[],5],[[]],[19],[20],[[]],[[],20],0,[[]],[[],6],[[],6],[[],7],0,0,0,[[]],[[]],[21,21],[[],[[23,[22]]]],[21,[[23,[22]]]],[[]],[5],[5],[5],0,[[]],[[],5],[[]],0,0,[[]],[18],[[21,18]],[[],6],[[],6],[[],7]],"p":[[6,"ArrayView1"],[15,"f64"],[6,"BoxedAdaptable"],[3,"KohonenAdaptivity"],[15,"usize"],[4,"Result"],[3,"TypeId"],[6,"Array1"],[3,"Shape"],[6,"Array2"],[3,"NdIndexIterator"],[4,"Option"],[3,"ArrayBase"],[3,"NeuralLayer"],[6,"BoxedResponsive"],[3,"CartesianResponsiveness"],[3,"SelforganizingNetwork"],[6,"ArrayView2"],[6,"BoxedTopological"],[3,"CartesianTopology"],[3,"IncrementalLearning"],[8,"Trainable"],[3,"Box"],[8,"Adaptable"],[8,"PointSet"],[8,"Neural"],[8,"Responsive"],[8,"Selforganizing"],[8,"Topological"]]},\
"potpourri":{"doc":"Package for models with discrete, unobservable latent …","t":"DQQQQCIQCCCIQALLKLLLAKLKLLLLLKKAAKKLLLLLKLAAAAAAADLLLLLLLLLLLLLLLLLLLLMLMLLLLLLLLDLLLLLMLLLLLLLLLLLLMLLMLLLLLLLLLLDLLLLLLLLLLLLLLLLDLLLLLLLLLLLLLLLLDLLLLLLLLLLLLLLLLFFFFFFFNENNNNNNLLLLLLLLLLLLLLLLLLLLLLLLLLMMIIDLLLLLLLLKLLLLLLLLMLLMLKLLLLLLLLLDDLLLLMLLLLLLLMLLLLMMMLLLLLLMMMMMMLLLLLLMLLLLLLLL","n":["AvgLLH","DataIn","DataIn","DataOut","DataOut","Latent","Learning","Likelihood","Mixable","Mixture","Model","Parametrizable","SufficientStatistics","backend","borrow","borrow_mut","compute","deref","deref_mut","drop","errors","expect","expect_rand","fit","from","from_subset","init","into","is_in_subset","maximize","merge","mixture","model","predict","predict","to_subset","to_subset_unchecked","try_from","try_into","type_id","update","vzip","ndarray","finite","gaussian","kmeans","linear","som","utils","Finite","borrow","borrow_mut","clone","clone_into","compute","deref","deref_mut","drop","expect","expect","expect_rand","fmt","from","from_subset","init","into","is_in_subset","maximize","merge","new","pmf","predict","prior","to_owned","to_subset","to_subset_unchecked","try_from","try_into","type_id","update","vzip","Gaussian","borrow","borrow_mut","clone","clone_into","compute","covariances","default","deref","deref_mut","drop","expect","fmt","from","from_subset","init","into","is_in_subset","maximize","means","merge","new","precisions","predict","predict","to_owned","to_subset","to_subset_unchecked","try_from","try_into","type_id","update","vzip","KMeans","borrow","borrow_mut","deref","deref_mut","drop","from","from_subset","init","into","is_in_subset","to_subset","to_subset_unchecked","try_from","try_into","type_id","vzip","Linear","borrow","borrow_mut","deref","deref_mut","drop","from","from_subset","init","into","is_in_subset","to_subset","to_subset_unchecked","try_from","try_into","type_id","vzip","SOM","borrow","borrow_mut","deref","deref_mut","drop","from","from_subset","init","into","is_in_subset","to_subset","to_subset_unchecked","try_from","try_into","type_id","vzip","filter_data","generate_random_expections","generate_samples","get_det_spd","get_shape2","get_shape3","invert_spd","DimensionMismatch","Error","ForbiddenCode","InvalidArgument","LinalgError","NotImplemented","ParameterError","ShapeError","borrow","borrow_mut","clone","clone_into","deref","deref_mut","drop","fmt","fmt","from","from","from","from","from_subset","init","into","is_in_subset","provide","to_owned","to_string","to_subset","to_subset_unchecked","try_from","try_into","type_id","vzip","fitted","n_init","Latent","Mixable","Mixture","borrow","borrow_mut","clone","clone_into","compute","deref","deref_mut","drop","expect","expect","expect_rand","fmt","from","from_subset","init","into","is_in_subset","latent","maximize","merge","mixables","new","predict","predict","to_owned","to_subset","to_subset_unchecked","try_from","try_into","type_id","update","vzip","Model","ModelInfo","borrow","borrow","borrow_mut","borrow_mut","converged","deref","deref","deref_mut","deref_mut","drop","drop","fit","fitted","from","from","from_subset","from_subset","incremental","incremental_weight","info","init","init","into","into","is_in_subset","is_in_subset","likelihood","max_iterations","mixable","n_components","n_init","n_iterations","new","predict","to_subset","to_subset","to_subset_unchecked","to_subset_unchecked","tol","try_from","try_from","try_into","try_into","type_id","type_id","vzip","vzip"],"q":["potpourri","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","potpourri::backend","potpourri::backend::ndarray","","","","","","potpourri::backend::ndarray::finite","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","potpourri::backend::ndarray::gaussian","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","potpourri::backend::ndarray::kmeans","","","","","","","","","","","","","","","","","potpourri::backend::ndarray::linear","","","","","","","","","","","","","","","","","potpourri::backend::ndarray::som","","","","","","","","","","","","","","","","","potpourri::backend::ndarray::utils","","","","","","","potpourri::errors","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","potpourri::errors::Error","","potpourri::mixture","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","potpourri::model","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","",""],"d":["Average log-likelihood. Used to meature convergence","","","","","","Probabilistic mixables should implement this trait A …","","","","","","","","","","Computes the sufficient statistics from the responsibility …","","","","","The E-Step. Computes the likelihood for each component in …","Generate a random expectation. Used as an initalization. …","","Returns the argument unchanged.","","","Calls <code>U::from(self)</code>.","","Maximize the model parameters from","merge multiple sufficient statistics into one.","","","","","","","","","","Update the stored sufficient statistics (for incremental …","","","","","","","","Additional support functions","","","","","","","","","","","","","","Returns the argument unchanged.","","","Calls <code>U::from(self)</code>.","","","","","","","","","","","","","","","","Represents a gaussian mixture model. The number of …","","","","","","The covariance matrices), $(k\\\\times d\\\\times d)$","","","","","","","Returns the argument unchanged.","","","Calls <code>U::from(self)</code>.","","","The mean values, $ k\\\\times d $","","","The precision matrices (inverted coariances), $(k\\\\times …","","","","","","","","","","","","","","","","","Returns the argument unchanged.","","","Calls <code>U::from(self)</code>.","","","","","","","","","","","","","","Returns the argument unchanged.","","","Calls <code>U::from(self)</code>.","","","","","","","","","","","","","","Returns the argument unchanged.","","","Calls <code>U::from(self)</code>.","","","","","","","","Splits a dataset consiting of two arrays according to a …","Generate random initializations from a dirichlet …","Create data generated with a Gaussian mixture model. …","","Gets the shape of an Array2 object or raise an error if …","Gets the shape of an Array3 object or raise an error if …","","","","","","","","","","","","","","","","","","","","","","Returns the argument unchanged.","","","Calls <code>U::from(self)</code>.","","","","","","","","","","","","","An additional interface for <code>Mixables</code> that can be used as …","","This trait represents the traditional mixture models with …","","","","","","","","","","","","","Returns the argument unchanged.","","","Calls <code>U::from(self)</code>.","","","","","","","","Prediction can be classification or regression depending …","","","","","","","","","The basis struct to use for models","","","","","","","","","","","","","","","Returns the argument unchanged.","Returns the argument unchanged.","","","","","","","","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","","","","","","","","","","","","","","","","","","","","","","",""],"i":[0,22,23,22,23,0,0,22,0,0,0,0,22,0,24,24,22,24,24,24,0,22,22,23,24,24,24,24,24,22,22,0,0,22,23,24,24,24,24,24,22,24,0,0,0,0,0,0,0,0,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,0,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,0,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,0,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,0,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,0,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,28,28,0,0,0,20,20,20,20,20,20,20,20,29,20,20,20,20,20,20,20,20,20,20,20,20,20,30,20,20,20,20,20,20,20,20,20,0,0,21,31,21,31,31,21,31,21,31,21,31,21,31,21,31,21,31,21,21,21,21,31,21,31,21,31,31,21,21,21,21,31,21,21,21,31,21,31,21,21,31,21,31,21,31,21,31],"f":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,[[]],[[]],[[],[[2,[1]]]],[3],[3],[3],0,[[],[[2,[1]]]],[3,[[2,[1]]]],[[],[[2,[1]]]],[[]],[[]],[[],3],[[]],[[],4],[[],[[2,[1]]]],[[],[[2,[1]]]],0,0,[[],[[2,[1]]]],[[],[[2,[1]]]],[[],5],[[]],[[],2],[[],2],[[],6],[7,[[2,[1]]]],[[]],0,0,0,0,0,0,0,0,[[]],[[]],[8,8],[[]],[8,[[2,[1]]]],[3],[3],[3],[8,[[2,[1]]]],[8,[[2,[1]]]],[[8,3],[[2,[1]]]],[[8,9],10],[[]],[[]],[[],3],[[]],[[],4],[8,[[2,[1]]]],[[],[[2,[1]]]],[[[5,[7]]],8],0,[8,[[2,[1]]]],0,[[]],[[],5],[[]],[[],2],[[],2],[[],6],[[8,7],[[2,[1]]]],[[]],0,[[]],[[]],[11,11],[[]],[11,[[2,[1]]]],0,[[],11],[3],[3],[3],[11,[[2,[1]]]],[[11,9],10],[[]],[[]],[[],3],[[]],[[],4],[11,[[2,[1]]]],0,[[],[[2,[1]]]],[[],11],0,[11,[[2,[1]]]],[11,[[2,[1]]]],[[]],[[],5],[[]],[[],2],[[],2],[[],6],[[11,7],[[2,[1]]]],[[]],0,[[]],[[]],[3],[3],[3],[[]],[[]],[[],3],[[]],[[],4],[[],5],[[]],[[],2],[[],2],[[],6],[[]],0,[[]],[[]],[3],[3],[3],[[]],[[]],[[],3],[[]],[[],4],[[],5],[[]],[[],2],[[],2],[[],6],[[]],0,[[]],[[]],[3],[3],[3],[[]],[[]],[[],3],[[]],[[],4],[[],5],[[]],[[],2],[[],2],[[],6],[[]],[[12,12],[[2,[1]]]],[[12,3],[[2,[[13,[7]],1]]]],[[3,3,3]],[12,[[2,[7,1]]]],[12,[[2,[1]]]],[14,[[2,[1]]]],[12,[[2,[[13,[7]],1]]]],0,0,0,0,0,0,0,0,[[]],[[]],[1,1],[[]],[3],[3],[3],[[1,9],10],[[1,9],10],[15,1],[16,1],[17,1],[[]],[[]],[[],3],[[]],[[],4],[18],[[]],[[],19],[[],5],[[]],[[],2],[[],2],[[],6],[[]],0,0,0,0,0,[[]],[[]],[20,20],[[]],[20,[[2,[1]]]],[3],[3],[3],[[],[[2,[1]]]],[20,[[2,[1]]]],[[20,3],[[2,[1]]]],[[20,9],10],[[]],[[]],[[],3],[[]],[[],4],0,[20,[[2,[1]]]],[[],[[2,[1]]]],0,[[],20],[[],[[2,[1]]]],[20,[[2,[1]]]],[[]],[[],5],[[]],[[],2],[[],2],[[],6],[[20,7],[[2,[1]]]],[[]],0,0,[[]],[[]],[[]],[[]],0,[3],[3],[3],[3],[3],[3],[21,[[2,[1]]]],0,[[]],[[]],[[]],[[]],0,0,0,[[],3],[[],3],[[]],[[]],[[],4],[[],4],0,0,0,0,0,0,[[3,3,3,4],21],[21,[[2,[1]]]],[[],5],[[],5],[[]],[[]],0,[[],2],[[],2],[[],2],[[],2],[[],6],[[],6],[[]],[[]]],"p":[[4,"Error"],[4,"Result"],[15,"usize"],[15,"bool"],[4,"Option"],[3,"TypeId"],[15,"f64"],[3,"Finite"],[3,"Formatter"],[6,"Result"],[3,"Gaussian"],[6,"ArrayView2"],[6,"Array2"],[6,"Array3"],[3,"ShapeError"],[3,"ParseIntError"],[4,"LinalgError"],[3,"Demand"],[3,"String"],[3,"Mixture"],[3,"Model"],[8,"Parametrizable"],[8,"Learning"],[3,"AvgLLH"],[3,"KMeans"],[3,"Linear"],[3,"SOM"],[13,"ParameterError"],[8,"Latent"],[8,"Mixable"],[3,"ModelInfo"]]}\
}');
if (typeof window !== 'undefined' && window.initSearch) {window.initSearch(searchIndex)};
if (typeof exports !== 'undefined') {exports.searchIndex = searchIndex};