from pandas.core.interchange.dataframe_protocol import DataFrame

import markovchain
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#This benchmarking could have been made way faster by implementing a dynamic programming
#approach to the computing of the confidence intervals, given that we are progressively revealing
#a larger chunk of the Markov Chain. This is however not done.

if __name__ == '__main__':

    LIST_METHODS = ['Gaussian', 'GaussianSlutsky', 'BasicChi2', 'BasicSlutskyChi2', 'FreerChi2', 'FreerSlutskyChi2']

    MC = markovchain.MarkovChain()

    state_space = ["A", "B", "C"]



    #P_1 matrix
    #np_true_matrix = np.array([[1 / 3, 1 / 2, 1 / 6],
    #                           [1 / 2, 0, 1 / 2],
    #                           [0, 1, 0]])
    #P_2 matrix
    #np_true_matrix = np.array([[1 / 3, 1 / 2, 1 / 6],
    #                           [1 / 3, 1 / 3, 1 / 3],
    #                           [1 / 2,  1/4 , 1 / 4 ]])

    #P3 matrix
    np_true_matrix = np.array([
        [1 / 3, 1 / 3, 1 / 3],
        [1 / 100, 99 / 100, 0],
        [1 / 100, 0, 99 / 100]
    ])

    #P4 matrix
    #np_true_matrix = np.array([
    #    [1 / 1000, 999 / 1000, 0],
    #    [0, 1 / 1000, 999 / 1000],
    #    [999 / 1000, 0, 1 / 1000]
    #])

    true_matrix = pd.DataFrame(np_true_matrix, index=state_space, columns=state_space, dtype='float64')

    n_samples = 1000
    MC.sample_according_to_matrix(initial_state="A", matrix=true_matrix, state_space=state_space,
                                  n_samples=n_samples)

    #Figure for the largest confidence interval plots
    plt.figure(0)
    plt.title("Convergence of largest confidence interval length per method")

    #FIgure for the smallest confidence interval plots
    plt.figure(1)
    plt.title("Convergence of smallest confidence interval length per method")

    #Will serve to save results of the run for inspection
    # Per-method, per-n storage
    history_lower = {m: [] for m in LIST_METHODS}
    history_upper = {m: [] for m in LIST_METHODS}
    history_diff = {m: [] for m in LIST_METHODS}
    history_max = {m: [] for m in LIST_METHODS}
    history_min = {m: [] for m in LIST_METHODS}

    for method in LIST_METHODS:
        print("Current method", method)

        #For plotting largest confidence interval length
        list_largest = []
        #For plotting smallest confidence interval length
        list_smallest = []

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

            #Plotting the largest confidence intervals lengths
            list_largest.append(np.max(upper_matrix-lower_matrix))
            #Plotting the smallest confidence intervals lengths
            list_smallest.append(np.min(upper_matrix-lower_matrix))
            #############################################
            ##Section dedicated to saving the results in a file:
            # Add subsample size as a column
            lower_matrix_copy = lower_matrix.copy()
            lower_matrix_copy["n"] = n
            upper_matrix_copy = upper_matrix.copy()
            upper_matrix_copy["n"] = n
            # Correct diff: subtract the matrices BEFORE adding 'n' column
            diff_matrix_copy = upper_matrix - lower_matrix
            diff_matrix_copy["n"] = n
            # Append to per-method storage lists
            history_lower[method].append(lower_matrix_copy)
            history_upper[method].append(upper_matrix_copy)
            history_diff[method].append(diff_matrix_copy)
            # Save max and min CI lengths for this subsample
            history_max[method].append(np.max(diff_matrix_copy.drop(columns="n").to_numpy()))
            history_min[method].append(np.min(diff_matrix_copy.drop(columns="n").to_numpy()))
        #Plotting this method:
        plt.figure(0)
        plt.loglog(N_range, list_largest,label=method)
        plt.figure(1)
        plt.loglog(N_range, list_smallest,label=method)

        ###########################################
        #Saving in a file the results of this method
        # Concatenate all subsample matrices into one
        all_lower = pd.concat(history_lower[method])
        all_upper = pd.concat(history_upper[method])
        all_diff = pd.concat(history_diff[method])
        # Save to one CSV per method
        all_lower.to_csv(f"./benchmark_results/lower_{method}.csv", index=True)
        all_upper.to_csv(f"./benchmark_results/upper_{method}.csv", index=True)
        all_diff.to_csv(f"./benchmark_results/diff_{method}.csv", index=True)
        summary = pd.DataFrame({
            "n": N_range,
            "max_length": history_max[method],
            "min_length": history_min[method]
        })
        summary.to_csv(f"./benchmark_results/summary_{method}.csv", index=False)

    plt.figure(0)
    plt.xlabel("Size of the markov chain sample")
    plt.ylabel("Largest confidence interval length among all pairs of states")
    plt.legend()
    plt.grid(True)

    plt.figure(1)
    plt.xlabel("Size of the markov chain sample")
    plt.ylabel("Smallest confidence interval length among all pairs of states")
    plt.legend()
    plt.grid(True)
    plt.show()



