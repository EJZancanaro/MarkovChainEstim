import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy.stats
from markovchain import compute_phi_from_MLE

class DynamicMarkovBenchmark:
    """
    Dynamically benchmark confidence intervals by incrementally updating statistics
    as the Markov chain grows, using lazy evaluation for MLE computation.

    This avoids numerical errors from incremental scaling while still being efficient.
    Works with MarkovChain objects that have .states (list) and .state_space (set).
    """

    def __init__(self, MC):
        """
        Initialize the dynamic benchmark.

        :param MC: MarkovChain object with .states (list of states) and .state_space (set of states)
        """
        self.MC = MC
        self.state_space = sorted(list(MC.state_space))  # Convert set to sorted list for consistent ordering
        self.n_states = len(self.state_space)

        # Initialize cumulative statistics
        self.transition_counts = pd.DataFrame(
            0, index=self.state_space, columns=self.state_space, dtype='int'
        )
        self.state_counts = pd.Series(0, index=self.state_space, dtype='int')

        # For Gaussian methods, we need to track phi_i dynamically
        self.A_powers_sum = None  # Will store sum of A^t for t=0..n-1
        self.current_A_power = None  # Will store A^n
        self.compensation = None  # Kahan summation compensation
        self.last_phi_update_n = 0  # Track when we last updated phi

        # Cached MLE matrix with lazy evaluation
        self.MLE_matrix_cache = None
        self.MLE_cache_dirty = True

        self.current_n = 0
        self.initial_state = None

    def add_transition(self, from_state, to_state):
        """
        Add one transition to the running statistics.

        :param from_state: starting state of the transition
        :param to_state: ending state of the transition
        """
        self.transition_counts.loc[from_state, to_state] += 1
        self.state_counts[from_state] += 1
        self.current_n += 1
        self.MLE_cache_dirty = True  # Mark MLE as needing recomputation

    def get_MLE_matrix(self):
        """
        Get the MLE transition matrix with lazy evaluation.
        Only recomputes when the cache is dirty (after new transitions added).
        """
        if self.MLE_cache_dirty:
            # Recompute MLE from counts - numerically stable
            self.MLE_matrix_cache = pd.DataFrame(
                0.0, index=self.state_space, columns=self.state_space, dtype='float64'
            )

            for i in self.state_space:
                if self.state_counts[i] > 0:
                    for j in self.state_space:
                        self.MLE_matrix_cache.loc[i, j] = (
                                self.transition_counts.loc[i, j] / self.state_counts[i]
                        )

            self.MLE_cache_dirty = False

        return self.MLE_matrix_cache

    def update_phi_matrices_incremental(self, new_transitions_count):
        """
        Incrementally update the sum of powers of A using Kahan summation.

        Adds terms from A^(last_n) to A^(current_n-1) to the running sum.
        This is called only at evaluation points.

        :param new_transitions_count: how many new transitions since last phi update
        """
        A = self.get_MLE_matrix().to_numpy()

        if self.A_powers_sum is None:
            # First time: initialize with A^0 = I
            self.A_powers_sum = np.eye(self.n_states, dtype=np.float64)
            self.current_A_power = np.eye(self.n_states, dtype=np.float64)
            self.compensation = np.zeros((self.n_states, self.n_states), dtype=np.float64)
            new_transitions_count -= 1  # We already included A^0

        # Add the next new_transitions_count powers of A
        for _ in range(new_transitions_count):
            # Compute next power: A^(n+1) = A^n @ A
            self.current_A_power = self.current_A_power @ A

            # Add A^n to the sum with Kahan compensation for numerical stability
            y = self.current_A_power - self.compensation
            t = self.A_powers_sum + y
            self.compensation = (t - self.A_powers_sum) - y
            self.A_powers_sum = t

    def compute_confidence_interval(self, state_i, state_j, method, alpha=0.05, avoid_trivial=True):
        """
        Compute confidence interval for p_{i,j} using current statistics.

        :param state_i: starting state
        :param state_j: ending state
        :param method: one of ['Gaussian', 'GaussianSlutsky', 'BasicChi2', 'BasicSlutskyChi2', 'FreerChi2', 'FreerSlutskyChi2']
        :param alpha: significance level
        :param avoid_trivial: whether to clip bounds to [0,1]
        :return: (lower_bound, upper_bound)
        """

        if self.state_counts[state_i] == 0:
            return (0, 1) if avoid_trivial else (np.nan, np.nan)

        MLE_matrix = self.get_MLE_matrix()
        p_hat = MLE_matrix.loc[state_i, state_j]
        n_i = self.state_counts[state_i]

        # Gaussian methods
        if method in ['Gaussian', 'GaussianSlutsky']:
            quantile = scipy.stats.norm.ppf(1 - alpha / 2)

            # Get phi_i from the sum of powers matrix
            phi_matrix = pd.DataFrame(
                self.A_powers_sum,
                index=self.state_space,
                columns=self.state_space
            )
            phi_i = phi_matrix.loc[self.initial_state, state_i]

            if phi_i == 0:
                return (0, 1) if avoid_trivial else (np.nan, np.nan)

            gamma = quantile ** 2 / (n_i * phi_i)

            if method == 'Gaussian':
                sqrt_term = np.sqrt(gamma * p_hat * (1 - p_hat) + gamma ** 2 / 4)
                lower_bound = p_hat * (1 + gamma) + gamma * (1 + gamma) / 2 - (1 + gamma) * sqrt_term
                upper_bound = p_hat * (1 + gamma) + gamma * (1 + gamma) / 2 + (1 + gamma) * sqrt_term
            else:  # GaussianSlutsky
                sqrt_term = np.sqrt(p_hat * (1 - p_hat) * gamma)
                lower_bound = p_hat - sqrt_term
                upper_bound = p_hat + sqrt_term

        # Chi-squared methods
        elif method in ['BasicChi2', 'BasicSlutskyChi2']:
            quantile = scipy.stats.chi2.ppf(q=1 - alpha, df=self.n_states - 1)

            if method == 'BasicChi2':
                q_over_n = quantile / n_i
                sqrt_term = np.sqrt((4 * p_hat + q_over_n) * q_over_n)
                lower_bound = (2 * p_hat + q_over_n - sqrt_term) / 2
                upper_bound = (2 * p_hat + q_over_n + sqrt_term) / 2
            else:  # BasicSlutskyChi2
                sqrt_term = np.sqrt(p_hat * quantile / n_i)
                lower_bound = p_hat - sqrt_term
                upper_bound = p_hat + sqrt_term

        # Freer Chi-squared methods
        elif method in ['FreerChi2', 'FreerSlutskyChi2']:
            quantile = scipy.stats.chi2.ppf(q=1 - alpha, df=self.n_states * (self.n_states - 1))

            if method == 'FreerChi2':
                q_over_n = quantile / n_i
                sqrt_term = np.sqrt((4 * p_hat + q_over_n) * q_over_n)
                lower_bound = (2 * p_hat + q_over_n - sqrt_term) / 2
                upper_bound = (2 * p_hat + q_over_n + sqrt_term) / 2
            else:  # FreerSlutskyChi2
                sqrt_term = np.sqrt(p_hat * quantile / n_i)
                lower_bound = p_hat - sqrt_term
                upper_bound = p_hat + sqrt_term
        else:
            raise ValueError(f"Unknown method: {method}")

        # Apply trivial bounds if requested
        if avoid_trivial:
            lower_bound = max(0, min(lower_bound, 1))
            upper_bound = max(0, min(upper_bound, 1))

            if lower_bound > 1:
                lower_bound = 0

        return (lower_bound, upper_bound)

    def benchmark_dynamic(self,
                          size_smallest_subsample,
                          size_largest_subsample,
                          step_of_subsampling,
                          adress_results,
                          LIST_METHODS,
                          alpha=0.05):
        """
        Dynamically benchmark methods by incrementally processing the chain.

        :param size_smallest_subsample: starting sample size
        :param size_largest_subsample: ending sample size
        :param step_of_subsampling: increment between samples
        :param adress_results: directory to save results
        :param LIST_METHODS: list of methods to benchmark
        :param alpha: significance level
        """

        # Create output directory if needed
        os.makedirs(adress_results, exist_ok=True)

        # Initialize storage
        history_max = {m: [] for m in LIST_METHODS}
        history_min = {m: [] for m in LIST_METHODS}
        history_lower = {m: [] for m in LIST_METHODS}
        history_upper = {m: [] for m in LIST_METHODS}
        history_diff = {m: [] for m in LIST_METHODS}

        N_range = list(range(size_smallest_subsample, size_largest_subsample, step_of_subsampling))

        # Process the chain incrementally
        self.initial_state = self.MC.states[0]

        # Track which evaluation points we need to hit
        eval_points = set(N_range)

        # Check if we need Gaussian methods
        needs_gaussian = any(m in LIST_METHODS for m in ['Gaussian', 'GaussianSlutsky'])

        print(f"Processing Markov chain of length {len(self.MC.states)}")
        print(f"Evaluation points: {len(N_range)}")

        for idx in range(len(self.MC.states) - 1):
            from_state = self.MC.states[idx]
            to_state = self.MC.states[idx + 1]

            # Add this transition (O(1) operation)
            self.add_transition(from_state, to_state)

            current_chain_length = idx + 2  # We've processed idx+1 transitions, starting from state 0

            # Check if we should evaluate at this point
            if current_chain_length in eval_points:
                n = current_chain_length
                print(f"Evaluating at n={n} (transition {idx + 1}/{len(self.MC.states) - 1})")

                # Update phi matrices for Gaussian methods if needed
                if needs_gaussian:
                    # Recompute phi from scratch using current MLE (matches original behavior)
                    # This avoids mixing powers of different MLE matrices
                    A = self.get_MLE_matrix().to_numpy()

                    # Import the function from markovchain module
                    from markovchain import compute_phi_from_MLE
                    self.A_powers_sum = compute_phi_from_MLE(A, size_chain=n)

                # Compute confidence intervals for all methods
                for method in LIST_METHODS:
                    lower_matrix = pd.DataFrame(
                        index=self.state_space, columns=self.state_space, dtype='float64'
                    )
                    upper_matrix = pd.DataFrame(
                        index=self.state_space, columns=self.state_space, dtype='float64'
                    )

                    # Compute CIs for ALL state pairs in the full state space
                    # States with no visits will return (0,1) due to avoid_trivial=True
                    for state_i in self.state_space:
                        for state_j in self.state_space:
                            lb, ub = self.compute_confidence_interval(
                                state_i, state_j, method, alpha=alpha
                            )
                            lower_matrix.loc[state_i, state_j] = lb
                            upper_matrix.loc[state_i, state_j] = ub

                    diff_matrix = upper_matrix - lower_matrix

                    # Store results
                    lower_copy = lower_matrix.copy()
                    lower_copy["n"] = n
                    upper_copy = upper_matrix.copy()
                    upper_copy["n"] = n
                    diff_copy = diff_matrix.copy()
                    diff_copy["n"] = n

                    history_lower[method].append(lower_copy)
                    history_upper[method].append(upper_copy)
                    history_diff[method].append(diff_copy)

                    # Save max and min CI lengths - match original exactly
                    # Original uses: np.max(diff_matrix_copy.drop(columns="n").to_numpy())
                    diff_for_stats = diff_matrix.to_numpy()
                    history_max[method].append(np.max(diff_for_stats))
                    history_min[method].append(np.min(diff_for_stats))

        # Generate plots and save results
        self._generate_results(N_range, history_max, history_min,
                               history_lower, history_upper, history_diff,
                               LIST_METHODS, adress_results)

    def _generate_results(self, N_range, history_max, history_min,
                          history_lower, history_upper, history_diff,
                          LIST_METHODS, adress_results):
        """Generate plots and save CSV results - matches original benchmark_source output exactly."""

        # Create plots
        plt.figure(0, figsize=(10, 6))
        plt.clf()

        plt.figure(1, figsize=(10, 6))
        plt.clf()

        for method in LIST_METHODS:
            # Save CSVs
            all_lower = pd.concat(history_lower[method])
            all_upper = pd.concat(history_upper[method])
            all_diff = pd.concat(history_diff[method])

            all_lower.to_csv(os.path.join(adress_results, f"lower_{method}.csv"), index=True)
            all_upper.to_csv(os.path.join(adress_results, f"upper_{method}.csv"), index=True)
            all_diff.to_csv(os.path.join(adress_results, f"diff_{method}.csv"), index=True)

            length_history = min(len(history_max[method]),len(history_min[method]))
            if len(N_range)> length_history:
                N_range=N_range[:length_history] #cut the subsample sizes that were not visited because the step was not an exact divisor of the length of the space to be explored

            # Summary CSV - EXACT same format as original benchmark_source
            summary = pd.DataFrame({
                "n": N_range,
                "max_length": history_max[method],
                "min_length": history_min[method]
            })
            summary.to_csv(os.path.join(adress_results, f"summary_{method}.csv"), index=False)

            # Plot
            plt.figure(0)
            plt.loglog(N_range, history_max[method], label=method)

            plt.figure(1)
            plt.loglog(N_range, history_min[method], label=method)

        # Finalize plots
        plt.figure(0)
        plt.xlabel("Size of the markov chain sample")
        plt.ylabel("Largest confidence interval length among all pairs of states")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(adress_results, "largest.png"))

        plt.figure(1)
        plt.xlabel("Size of the markov chain sample")
        plt.ylabel("Smallest confidence interval length among all pairs of states")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(adress_results, "smallest.png"))

        plt.show()

        print(f"\nResults saved to {adress_results}")