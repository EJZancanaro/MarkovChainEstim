import numpy as np
import pandas as pd

class MarkovChain():
    def __init__(self):
        self.states = []
        self.state_space = set([])
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

    def MLE_stationary(self):
        """Maximum likelihood estimator of p_{i,j} for all i and j in the state space"""

        length_of_chain = len(self.states)
        state_dimension = len(self.state_space)

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
                transition_matrix_estimate.loc[state_1, state_2] /= counts_of_starting_state[state_1]

        return transition_matrix_estimate
