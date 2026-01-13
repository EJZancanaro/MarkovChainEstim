

import numpy as np
import pandas as pd
import scipy.stats

class MarkovChain():
    def __init__(self):
        self.states = []
        self.state_space = set([])

    def sample_according_to_matrix(self, state_space, initial_state ,matrix, n_samples):
        self.state_space = set(state_space)

        assert matrix.shape[0] == len(state_space)
        assert matrix.shape[1] == len(state_space)
        assert initial_state in state_space
        assert np.max(matrix)<=1 and np.min(matrix)>=0

        current_state = initial_state
        for time in range(n_samples):
            probabilities_next_states = [probs for probs in matrix.loc[current_state, :]]

            self.next_state(np.random.choice(a=state_space, p=probabilities_next_states))

            current_state = self.states[-1]

    def next_state(self, element):
        """
        Add a new sample to the Markov Chain
        :param element:
        :return:
        """
        self.states.append(element)

        if element not in self.state_space:
            self.state_space.add(element)

    def __print__(self):
        print(f"State space: {self.state_space} \n Unbroken chain:{self.states}")

    def MLE_stationary(self, fail_at_missing_transitions=False):
        """Maximum likelihood estimator of p_{i,j} for all i and j in the state space

        fail_at_missing_transitions: If set to False, the estimator of the probability of an event that never happened is set to 0. Otherwise, the program crashes.
        """
        assert self.states is not []

        # Initialize the matrix ; and the array storing the n_i^* = \sum_t\sum_j n_{i,j}(t)
        transition_matrix_estimate = pd.DataFrame(
            0,
            index=list(self.state_space),
            columns=list(self.state_space),
            dtype='float64'
        )

        counts_of_starting_state = pd.Series(0, index=list(self.state_space), dtype='int')

        #Compute number of transitions
        for time in range(len(self.states) - 1):

            starting_state = self.states[time]
            following_state = self.states[time + 1]

            transition_matrix_estimate.loc[starting_state, following_state] += 1

            counts_of_starting_state[starting_state] += 1

        #normalise them by the n_i^* as to get probabilities
        for state_1 in self.state_space:
            for state_2 in self.state_space:
                if counts_of_starting_state[state_1]!=0:
                    transition_matrix_estimate.loc[state_1, state_2] /= counts_of_starting_state[state_1]
                else:
                    if not fail_at_missing_transitions:
                        transition_matrix_estimate.loc[state_1, state_2]=0
                    else:
                        raise AssertionError("Tried to estimate the probability of an event that was never witnessed, while having set the fail_at_missing_transitions flag to True")
        return transition_matrix_estimate

    def confidence_intervals(self, state_i, state_j, alpha=0.05, method='BasicChi2', avoid_trivial='True'):
        """Gives a confidence intervals for p_{i,j}

        :param state_i:

        :param avoid_trivial:whether lower bounds smaller than 0 should be put to 0, upper bounds larger than 1 should be put to 1, and whether CI for probabilities of non-measured transitions should be defined as [0,1]
        """
        assert self.states is not []

        assert method in ['Gaussian', 'GaussianSlutsky' ,'BasicChi2', 'BasicSlutskyChi2', 'FreerChi2', 'FreerSlutskyChi2']

        MLE_matrix = self.MLE_stationary()

        dim = len(self.state_space)

        counts_of_starting_state = pd.Series(0, index=list(self.state_space), dtype='int')

        for current_state in self.states[:-1]:
            counts_of_starting_state[current_state]+=1

        if counts_of_starting_state[state_i] == 0:
            if avoid_trivial:
                return (0,1)
            else:
                return (np.nan, np.nan) #No method can estimate confidence intervals without any measures

        if method in ['Gaussian', 'GaussianSlutsky']:

            #computing gamma
            quantile = scipy.stats.norm.ppf(1-alpha/2)

            A = MLE_matrix.to_numpy()
            matrix = sum([np.linalg.matrix_power(A,t-1) for t in range(1,len(self.states))]) # TODO can be optimised with dynamic programming
            matrix = pd.DataFrame(matrix, index=MLE_matrix.index, columns=MLE_matrix.columns)
            phi_i = matrix.loc[self.states[0], state_i]

            if phi_i==0 :
                if avoid_trivial:
                    return (0,1)
                else:
                    return (np.nan, np.nan)

            gamma = quantile**2/(counts_of_starting_state[state_i]*phi_i)

            if method=='Gaussian' :
                lower_bound = MLE_matrix.loc[state_i, state_j]*(1+gamma) + gamma*(1+gamma)/2 - (1+gamma) * np.sqrt(gamma*MLE_matrix.loc[state_i, state_j]*(1-MLE_matrix.loc[state_i, state_j])+gamma**2/4)
                upper_bound =MLE_matrix.loc[state_i, state_j]*(1+gamma) + gamma*(1+gamma)/2 + (1+gamma) * np.sqrt(gamma*MLE_matrix.loc[state_i, state_j]*(1-MLE_matrix.loc[state_i, state_j])+gamma**2/4)

            elif method=='GaussianSlutsky':
                lower_bound = MLE_matrix.loc[state_i, state_j] - np.sqrt(MLE_matrix.loc[state_i,state_j]*(1-MLE_matrix.loc[state_i, state_j])*gamma)
                upper_bound = MLE_matrix.loc[state_i, state_j] + np.sqrt(MLE_matrix.loc[state_i, state_j] * (1 - MLE_matrix.loc[state_i, state_j]) * gamma)

        if method in ['BasicChi2', 'BasicSlutskyChi2']:
            quantile = scipy.stats.chi2.ppf(q=1 - alpha, df=dim - 1)
        elif method in ['FreerChi2', 'FreerSlutskyChi2' ] : #still in development
            quantile = scipy.stats.chi2.ppf(q=1 - alpha, df=dim*(dim - 1))

        if method in ['BasicChi2', 'FreerChi2'] :
            lower_bound = (
                    2*MLE_matrix.loc[state_i, state_j]+quantile/(counts_of_starting_state[state_i])
                    - np.sqrt( (4*MLE_matrix.loc[state_i, state_j] + quantile/(counts_of_starting_state[state_i]) )* quantile/(counts_of_starting_state[state_i]))
            )/2

            upper_bound = (
                    2*MLE_matrix.loc[state_i, state_j]+quantile/(counts_of_starting_state[state_i])
                    + np.sqrt( (4*MLE_matrix.loc[state_i, state_j] + quantile/(counts_of_starting_state[state_i]) )* quantile/(counts_of_starting_state[state_i]))
            )/2
        elif method in ['BasicSlutskyChi2', 'FreerSlutskyChi2']:
            lower_bound = MLE_matrix.loc[state_i, state_j] - np.sqrt(
                MLE_matrix.loc[state_i, state_j] * quantile / counts_of_starting_state[state_i])

            upper_bound = MLE_matrix.loc[state_i, state_j] + np.sqrt(
                MLE_matrix.loc[state_i, state_j] * quantile / counts_of_starting_state[state_i])

        if avoid_trivial :
            if lower_bound<0:
                lower_bound = 0

            if lower_bound>1 : #avoids the Gaussian method's problem of having lower bounds that are above 1 for extreme values of gamma
                lower_bound=0

            if upper_bound>1:
                upper_bound = 1
        return (lower_bound, upper_bound)
