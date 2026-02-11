"""
Example usage of dynamic Markov chain benchmarking.

This shows how to use the DynamicMarkovBenchmark class to efficiently
benchmark confidence interval methods on a Markov chain.
"""

import numpy as np
import pandas as pd

from dynamic_benchmark import DynamicMarkovBenchmark


def test_dynamic_vs_original():
    """
    Compare dynamic benchmarking with the original method.
    This helps verify correctness and measure speedup.
    """
    try:
        from MarkovChain.markovchain import MarkovChain
        from MarkovChain.benchmark_source import benchmark_source
    except ImportError:
        print("Could not import MarkovChain modules. Skipping comparison.")
        return

    import time

    # parameters: must be identical to both methods

    size_smallest_subsample = 2
    size_largest_subsample = 100
    step_of_subsampling = 5
    LIST_METHODS = ['Gaussian', 'GaussianSlutsky', 'BasicChi2', 'BasicSlutskyChi2', 'FreerChi2', 'FreerSlutskyChi2']

    # Create a test chain

    state_space = ['A', 'B', 'C']
    transition_matrix = pd.DataFrame(
        [[0.7, 0.2, 0.1],
         [0.3, 0.4, 0.3],
         [0.2, 0.3, 0.5]],
        index=state_space,
        columns=state_space
    )

    MC = MarkovChain()
    MC.sample_according_to_matrix(
        state_space=state_space,
        initial_state='A',
        matrix=transition_matrix,
        n_samples=size_largest_subsample
    )

    # Time the dynamic version
    print("Testing DYNAMIC benchmarking...")
    benchmark = DynamicMarkovBenchmark(MC)

    start = time.time()
    benchmark.benchmark_dynamic(
        size_smallest_subsample=size_smallest_subsample,
        size_largest_subsample=size_largest_subsample,
        step_of_subsampling=step_of_subsampling,
        adress_results='./benchmark_results/comparison_dynamic_vs_original/results_dynamic',
        LIST_METHODS=LIST_METHODS
    )
    dynamic_time = time.time() - start

    print(f"\nDynamic benchmarking took: {dynamic_time:.2f} seconds")

    # Time the original version
    print("\nTesting ORIGINAL benchmarking...")
    start = time.time()
    benchmark_source(
        MC, state_space,
        size_smallest_subsample=size_smallest_subsample,
        size_largest_subsample=size_largest_subsample,
        step_of_subsampling=step_of_subsampling,
        adress_results='./benchmark_results/comparison_dynamic_vs_original/results_original',
        LIST_METHODS=LIST_METHODS
    )
    original_time = time.time() - start

    print(f"Original benchmarking took: {original_time:.2f} seconds")
    print(f"Speedup: {original_time / dynamic_time:.2f}x")

    # Verify results match
    print("\nVerifying results match...")
    dynamic_summary = pd.read_csv(
        'benchmark_results/comparison_dynamic_vs_original/results_dynamic/summary_BasicChi2.csv')
    original_summary = pd.read_csv(
        'benchmark_results/comparison_dynamic_vs_original/results_original/summary_BasicChi2.csv')

    max_diff = np.nanmax(np.abs(dynamic_summary['max_length'].values - original_summary['max_length'].values))
    print(f"Maximum difference in CI lengths: {max_diff:.2e}")

    if max_diff < 1e-10:
        print("✓ Results match perfectly!")
    else:
        print("⚠ Small numerical differences detected")
        assert False



if __name__ == "__main__":
    # Compare performance with original
    test_dynamic_vs_original()