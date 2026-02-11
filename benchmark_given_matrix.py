import ..markovchain
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from benchmark_source import benchmark_dynamic
from benchmark_source import benchmark_source


def benchmark_given_matrix(np_true_matrix, state_space, initial_state, n_samples,
                           size_smallest_subsample, size_largest_subsample, step_of_subsampling,
                           adress_results,
                           LIST_METHODS):
    """

    :param np_true_matrix: numpy array. Transition matrix of the Markov chain
    :param state_space: State space of the Markov chain
    :param initial_state: Initial state taken by the Markov chain

    All other parameters are defined in benchmark_source
    """
    MC = markovchain.MarkovChain()

    true_matrix = pd.DataFrame(np_true_matrix, index=state_space, columns=state_space, dtype='float64')

    MC.sample_according_to_matrix(initial_state=initial_state, matrix=true_matrix, state_space=state_space,
                                  n_samples=n_samples)

    benchmark_dynamic(MC=MC,state_space=state_space,
                     size_smallest_subsample=size_smallest_subsample,
                     size_largest_subsample=size_largest_subsample,
                     step_of_subsampling=step_of_subsampling,
                     adress_results=adress_results,
                     LIST_METHODS=LIST_METHODS)