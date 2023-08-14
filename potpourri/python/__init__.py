'''A very basic, yet agruably elegant implementation of Gaussian mixture models'''

import timeit
from dataclasses import dataclass
from typing import NewType

import numpy as np
# from numpy.typing import NDArray
from sklearn.datasets import make_blobs
# from sklearn.mixture import GaussianMixture
# from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt

# import potpourri extension

def make_data():
    '''Generate data for the rust implementation'''
    data, _ = make_blobs(n_samples=10000, centers=10, n_features=2, random_state=7)  # pylint: disable=W0632
    responsibilities = np.random.dirichlet(10 * [1.0], data.shape[0])

    # Create Gaussian Mixture and run single maximization step
    # with precalculated responsibilities to save results

    # Pseudo-code
    # gmm = GaussianMixture(...)
    # gmm.maximize(gmm, responsibilities.T, data)

    # np.save("data.npy", data)
    # np.save("responsibilities.npy", responsibilities.T)
    # np.save("means.npy", gmm._means)
    # np.save("weights.npy", gmm._weights)
    # np.save("covs.npy", gmm._covs)
