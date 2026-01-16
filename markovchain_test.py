import pandas as pd
import numpy as np

import markovchain

def test_creation():
    """Tests whether the class can be instantiated."""
    MChain = markovchain.MarkovChain()
def test_sequential():
    """Tests whether the sequential increments of the class behave properly"""
    MChain = markovchain.MarkovChain()

    MChain.next_state("State A")
    MChain.next_state("State B")
    MChain.next_state("State A")
    MChain.next_state("State C")
    MChain.next_state("State B")

    assert MChain.state_space == {"State A", "State B", "State C"}
    assert MChain.states == ["State A", "State B","State A","State C", "State B"]

    MChain_numeric = markovchain.MarkovChain()
    MChain_numeric.next_state(0)
    MChain_numeric.next_state(1)
    MChain_numeric.next_state(0)
    MChain_numeric.next_state(2)
    MChain_numeric.next_state(1)

    assert MChain_numeric.state_space == {0,1,2}
    assert MChain_numeric.states == [0,1,0,2,1]

def test_sample_according_to_matrix():

    MC = markovchain.MarkovChain()
    state_space = ["A", "B", "C"]
    #Trivial Markov chain that always stays where it started
    matrix = np.array([[1,0,0],[0,1,0],[0,0,1]])
    matrix = pd.DataFrame(matrix, index=state_space, columns=state_space)

    #TEST The initial state is taken into account:
    MC.sample_according_to_matrix(state_space=state_space , initial_state="A", matrix=matrix, n_samples=1)
    assert MC.states == ["A"]
    #TEST It works as supposed to for static cases
    ##Starting at A
    MC = markovchain.MarkovChain()
    MC.sample_according_to_matrix(state_space=state_space , initial_state="A", matrix=matrix, n_samples=100)
    assert np.all(np.array(MC.states)==np.repeat("A", 100) )
    ##Starting at B
    MC = markovchain.MarkovChain()
    MC.sample_according_to_matrix(state_space=state_space, initial_state="B", matrix=matrix, n_samples=100)
    assert np.all(np.array(MC.states) == np.repeat("B", 100))
    ##Starting at C
    MC = markovchain.MarkovChain()
    MC.sample_according_to_matrix(state_space=state_space, initial_state="C", matrix=matrix, n_samples=100)
    assert np.all(np.array(MC.states) == np.repeat("C", 100))

def test_sample_according_to_matrix_asymptotic():
    MC = markovchain.MarkovChain()
    state_space = ["A", "B", "C"]

    matrix = np.array([[1/3, 1/3, 1/3], [1/3, 1/3, 1/3], [1/3, 1/3, 1/3]])
    matrix = pd.DataFrame(matrix, index=state_space, columns=state_space)

    MC.sample_according_to_matrix(state_space=state_space, initial_state="A", matrix=matrix, n_samples=100000)
    matrix_estimate = MC.MLE_stationary()
    print("Estimated matrix: \n", matrix_estimate)
    print("True matrix: \n", matrix)
    assert np.allclose(matrix_estimate, matrix, atol=1e-8, rtol= 0.05)

def test_MLE_binary_state_space_simple():
    """Tests whether the transition probability matrix is estimated properly"""

    MChain = markovchain.MarkovChain()

    MChain.next_state(1)
    MChain.next_state(0)
    MChain.next_state(1)
    MChain.next_state(0)
    MChain.next_state(1)
    MChain.next_state(0)
    MChain.next_state(1)
    MChain.next_state(0)

    transition_matrix_estimate = MChain.MLE_stationary()

    dim = len(MChain.state_space)
    assert transition_matrix_estimate.shape == (dim, dim)
    assert min(transition_matrix_estimate) >= 0
    assert max(transition_matrix_estimate) <= 1

    assert transition_matrix_estimate[0][0] == 0
    assert transition_matrix_estimate[0][1] == 1
    assert transition_matrix_estimate[1][0] == 1
    assert transition_matrix_estimate[1][1] == 0


def test_MLE_binary_state_space():
    """Tests whether the transition probability matrix is estimated properly"""

    MChain = markovchain.MarkovChain()

    MChain.next_state(1)
    MChain.next_state(0)
    MChain.next_state(1)
    MChain.next_state(0)
    MChain.next_state(1)
    MChain.next_state(0)
    MChain.next_state(1)
    MChain.next_state(0)

    transition_matrix_estimate = MChain.MLE_stationary()

    dim = len(MChain.state_space)
    assert transition_matrix_estimate.shape == (dim, dim)
    assert min(transition_matrix_estimate) >= 0
    assert max(transition_matrix_estimate) <= 1

    assert transition_matrix_estimate[0][0] == 0
    assert transition_matrix_estimate[0][1] == 1
    assert transition_matrix_estimate[1][0] == 1
    assert transition_matrix_estimate[1][1] == 0

    print("Test failed because the sum of estimated probabilities isn't close enough to one")
    print("MLE matrix: ", np.array(transition_matrix_estimate))
    print("Vector of sums over rows: ", np.sum(np.array(transition_matrix_estimate), axis=1))
    assert np.allclose(np.sum(np.array(transition_matrix_estimate), axis=1),
                       np.ones(len(MChain.state_space)),
                       atol=1e-8, rtol=0.05)

def test_MLE_complicated():
    """
    Tests the asymptotic convergence of the MLE estimator for a more complicated MC
    """
    state_space = [0,1,2]
    true_matrix = pd.DataFrame(index = state_space, columns = state_space, dtype='float64')

    true_matrix.loc[0,0] = 2/3
    true_matrix.loc[0, 1] = 1/6
    true_matrix.loc[0,2] = 1/6

    true_matrix.loc[1,0] = 1/4
    true_matrix.loc[1,1] = 1/2
    true_matrix.loc[1,2] = 1/4

    true_matrix.loc[2,0] = 1/2
    true_matrix.loc[2,1] = 1/2
    true_matrix.loc[2,2] = 0

    MChain = markovchain.MarkovChain()

    current_state = 0
    for time in range(100000):

        probabilities_next_states = [probs for probs in true_matrix.loc[current_state,:]]

        MChain.next_state(np.random.choice(a=[0,1,2], p=probabilities_next_states) )

        current_state = MChain.states[-1]

    MLE_matrix = np.array(MChain.MLE_stationary())

    print( np.max( np.array(MLE_matrix) - np.array(true_matrix) ) )
    print(np.array(MLE_matrix))
    print(np.array(true_matrix))

    assert np.allclose(np.array(MLE_matrix), np.array(true_matrix), atol=1e-8, rtol= 0.05)

    print("Test failed because the sum of estimated probabilities isn't close enough to one")
    print("MLE matrix: ",np.array(MLE_matrix))
    print("Vector of sums over rows: ", np.sum(np.array(MLE_matrix), axis=1))
    assert np.allclose( np.sum(np.array(MLE_matrix), axis=1) ,
                        np.ones(len(MChain.state_space)),
                        atol=1e-8, rtol= 0.05 )

def test_confidence_intervals() :
    MChain = markovchain.MarkovChain()

    state_space = ["A", "B", "C"]

    p_matrix = pd.DataFrame(
        index = state_space,
        columns = state_space,
        dtype= "float64"
    )
    p_matrix.loc["A","A"] = 1/3
    p_matrix.loc["A","B"] = 1/3
    p_matrix.loc["A","C"] = 1/3

    p_matrix.loc["B","A"] = 1/2
    p_matrix.loc["B","B"] = 1/2
    p_matrix.loc["B","C"] = 0

    p_matrix.loc["C","A"] = 0
    p_matrix.loc["C","B"] = 1
    p_matrix.loc["C","C"] = 0


    MChain.sample_according_to_matrix(state_space=state_space, initial_state="A", matrix=p_matrix, n_samples=1000)

    for method in ['BasicChi2', 'BasicSlutskyChi2', 'FreerChi2', 'FreerSlutskyChi2', 'Gaussian', 'GaussianSlutsky'] :

        lower, upper = MChain.confidence_intervals(state_i="A", state_j="A", alpha=0.05, method=method)

        estimate = MChain.MLE_stationary().loc["A","A"]
        print(f'METHOD : {method}')
        print(f'Upper bound { upper }, estimate = { estimate } , lower bound { lower }')

        assert (lower < estimate) and (estimate < upper)