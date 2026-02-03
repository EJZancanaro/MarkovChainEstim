import markovchain
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from benchmark_source import benchmark_source

#This benchmarking could have been made way faster by implementing a dynamic programming
#approach to the computing of the confidence intervals, given that we are progressively revealing
#a larger chunk of the Markov Chain. This is however not done.

def benchmark_from_AFICS(rmsd_file,
                         size_smallest_subsample, size_largest_subsample, step_of_subsampling,
                         adress_results,
                         LIST_METHODS):
    """Given the RMSD file output by the AFICS application, study the convergences of the confidence intervals of the Markov chains describing the transitions between geometries.
    :string:rmsd_file : adress of the RMSD file output by the AFICS application.
    All other parameters are defined in benchmark_source
    """

    MC = markovchain.MarkovChain()
    MC.load_rmsd_geometries(rmsd_file)
    state_space = np.array(list(MC.state_space))

    if len(MC.states) < size_largest_subsample:
        print(f"Warning: The value of largest_subsample the user wishes to study is {size_largest_subsample}. This is larger than the size of the simulation {len(MC.states)}")
        size_largest_subsample = len(MC.states)
        print(f"The value of size_largest_subsample has been reduced to {size_largest_subsample} " )
    MC = markovchain.MarkovChain()
    MC.load_rmsd_geometries(rmsd_file)
    state_space = np.array(list(MC.state_space))

    print("Benchmarking from an AFICS geometry trajectory started")
    print("methods to be used: ", LIST_METHODS)
    print(f"State space of the Markov chain: {MC.state_space}")
    print(f"length of the trajectory: {len(MC.states)}")

    benchmark_source(MC=MC, state_space=state_space,
                     size_smallest_subsample=size_smallest_subsample,
                     size_largest_subsample=size_largest_subsample,
                     step_of_subsampling=step_of_subsampling,
                     adress_results=adress_results,
                     LIST_METHODS=LIST_METHODS)


