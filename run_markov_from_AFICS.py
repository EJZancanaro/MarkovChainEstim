
import markovchain
import pandas as pd
import numpy as np
import sys



rmsd_df = pd.read_csv("./AFICS/results/Cr-results-RMSDs.csv")

rmsd_df['Best geometry'] = rmsd_df.idxmin(axis=1)

trajectory = rmsd_df['Best geometry'].tolist()

MC = markovchain.MarkovChain()

for element in trajectory:
    MC.next_state(element)

print(trajectory)

state_space = list(MC.state_space)

print(state_space)


#######
# #####Copier coller à peu de choses près de benchmark, peu être effacé
#######

LIST_METHODS = ['Gaussian', 'GaussianSlutsky' ,'BasicChi2', 'BasicSlutskyChi2', 'FreerChi2', 'FreerSlutskyChi2']

list_lower_matrices = pd.DataFrame(index=LIST_METHODS)
list_upper_matrices = pd.DataFrame(index=LIST_METHODS)

difference_matrices = pd.DataFrame(index=LIST_METHODS)
for method in LIST_METHODS :
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

for method in LIST_METHODS :
    difference_matrices.loc[method] = list_upper_matrices.loc[method] - list_lower_matrices.loc[method]
best_method = np.argmin(np.linalg.norm(difference_matrices.loc[method], np.inf) for method in LIST_METHODS)
print(LIST_METHODS[best_method])
