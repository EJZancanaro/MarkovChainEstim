from pandas.core.interchange.dataframe_protocol import DataFrame

import markovchain
import numpy as np
import pandas as pd

if __name__=='__main__' :

    LIST_METHODS = ['Gaussian', 'GaussianSlutsky' ,'BasicChi2', 'BasicSlutskyChi2', 'FreerChi2', 'FreerSlutskyChi2']

    list_lower_matrices = pd.DataFrame(index=LIST_METHODS)
    list_upper_matrices = pd.DataFrame(index=LIST_METHODS)
    for method in LIST_METHODS :
        MC = markovchain.MarkovChain()

        state_space = ["A", "B", "C"]

        np_true_matrix = np.array([[1/3, 1/2, 1/6],
                                [1/2, 0, 1/2  ],
                                [0  , 1, 0]   ] )

        true_matrix = pd.DataFrame(np_true_matrix,index = state_space, columns = state_space, dtype='float64')

        MC.sample_according_to_matrix(initial_state="A", matrix=true_matrix, state_space = state_space, n_samples=1000)

        upper_matrix = pd.DataFrame(index = state_space, columns = state_space, dtype='float64')
        lower_matrix = pd.DataFrame(index = state_space, columns = state_space, dtype='float64')

        for state_i in MC.state_space:
            for state_j in MC.state_space:
                lower_matrix.loc[state_i,state_j], upper_matrix.loc[state_i,state_j] = MC.confidence_intervals(state_i=state_i, state_j=state_j, method=method)

        print(f"current method {method}")
        print(f"Matrix of lower bounds \n{lower_matrix}")
        list_lower_matrices.loc[method] = lower_matrix
        print(f"Matrix of upper bounds \n{upper_matrix}")
        list_upper_matrices.loc[method] = upper_matrix


