from pandas.core.interchange.dataframe_protocol import DataFrame

import markovchain
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':

    LIST_METHODS = ['Gaussian', 'GaussianSlutsky', 'BasicChi2', 'BasicSlutskyChi2', 'FreerChi2', 'FreerSlutskyChi2']

    list_lower_matrices = pd.DataFrame(index=LIST_METHODS)
    list_upper_matrices = pd.DataFrame(index=LIST_METHODS)

    difference_matrices = pd.DataFrame(index=LIST_METHODS)

    MC = markovchain.MarkovChain()

    state_space = ["A", "B", "C"]

    np_true_matrix = np.array([[1 / 3, 1 / 2, 1 / 6],
                               [1 / 2, 0, 1 / 2],
                               [0, 1, 0]])

    true_matrix = pd.DataFrame(np_true_matrix, index=state_space, columns=state_space, dtype='float64')

    n_samples = 1000
    MC.sample_according_to_matrix(initial_state="A", matrix=true_matrix, state_space=state_space,
                                  n_samples=n_samples)

    plt.figure(0)
    plt.title("Convergence of largest confidence interval length per method")
    for method in LIST_METHODS:
        print("Current method", method)
        # new stuff
        ##############
        list_norms = []

        N_range = np.arange(int(n_samples/10),n_samples,int(n_samples/10))

        for n in N_range:

            smaller_mc = markovchain.MarkovChain()
            state_space = ["A", "B", "C"]
            smaller_mc.states = MC.states[:n]
            smaller_mc.state_space = state_space

            upper_matrix = pd.DataFrame(index=state_space, columns=state_space, dtype='float64')
            lower_matrix = pd.DataFrame(index=state_space, columns=state_space, dtype='float64')

            for state_i in smaller_mc.state_space:
                for state_j in smaller_mc.state_space:
                    lower_matrix.loc[state_i, state_j], upper_matrix.loc[state_i, state_j] = smaller_mc.confidence_intervals(
                        state_i=state_i, state_j=state_j, method=method)

            list_lower_matrices.loc[method] = lower_matrix
            list_upper_matrices.loc[method] = upper_matrix

            list_norms.append(np.linalg.norm(upper_matrix-lower_matrix, np.inf))

        plt.figure(0)
        plt.loglog(N_range, list_norms,label=method)

    plt.figure(0)

    plt.xlabel("Size of the markov chain sample")
    plt.ylabel("Largest confidence interval length among all pairs of states")
    plt.legend()
    plt.grid(True)
    plt.show()

