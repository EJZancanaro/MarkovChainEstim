import argparse
import numpy as np

from benchmark_from_AFICS import benchmark_from_AFICS
from benchmark_given_matrix import benchmark_given_matrix


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark confidence interval methods for Markov chains: for given subsamples, see how the lengths of the confidence intervals converge."
    )

    # General parameters
    parser.add_argument("--size_smallest_subsample", type=int, default=1, help="Size of smallest subsample used to obtain the confidence interval" )
    parser.add_argument("--step_of_subsampling", type=int, default=1,help="Step of size increase of subsample for evaluation of the CIs")
    parser.add_argument("--adress_results", required=True ,type=str, help="Where to store the results of the benchmarking, in case one whishes to analyze them.")

    parser.add_argument(
        "--methods",
        nargs="+",
        default=['Gaussian', 'GaussianSlutsky', 'BasicChi2',
                 'BasicSlutskyChi2', 'FreerChi2', 'FreerSlutskyChi2'],
        help="List of CI estimation methods to be used"
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["matrix", "afics"],
        required=True,
        help="Choose benchmark mode: from an RMSD file or from a known theoretical Markov Chain matrix?"
    )

    # ===== Matrix mode arguments =====
    parser.add_argument("--matrix_id", type=int,
                        help="Choose example matrix (1-4) among those presented in the article")
    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--initial_state", type=str, default="A")
    parser.add_argument("--state_space", type=str, default= ["A","B", "C"])
    # ===== AFICS mode arguments =====
    parser.add_argument("--rmsd_file", type=str)

    return parser.parse_args()


def load_matrix(matrix_id):
    if matrix_id == 1:
        return np.array([
            [1/3, 1/2, 1/6],
            [1/3, 1/3, 1/3],
            [1/2, 1/4, 1/4]
        ])
    elif matrix_id == 2:
        return np.array([
            [1/3, 1/2, 1/6],
            [1/2, 0, 1/2],
            [0, 1, 0]
        ])
    elif matrix_id == 3:
        return np.array([
            [1/3, 1/3, 1/3],
            [1/100, 99/100, 0],
            [1/100, 0, 99/100]
        ])
    elif matrix_id == 4:
        return np.array([
            [1/1000, 999/1000, 0],
            [0, 1/1000, 999/1000],
            [999/1000, 0, 1/1000]
        ])
    else:
        raise ValueError("matrix_id must be between 1 and 4")


def main():
    args = parse_args()

    if args.mode == "matrix":
        np_true_matrix = load_matrix(args.matrix_id)

        benchmark_given_matrix(
            np_true_matrix=np_true_matrix,
            state_space=args.state_space,
            initial_state=args.initial_state,
            adress_results=args.adress_results,
            size_smallest_subsample=args.size_smallest_subsample,
            size_largest_subsample=args.n_samples,
            step_of_subsampling=args.step_of_subsampling,
            n_samples=args.n_samples,
            LIST_METHODS=args.methods
        )

    elif args.mode == "afics":

        benchmark_from_AFICS(
            rmsd_file=args.rmsd_file,
            size_smallest_subsample=args.size_smallest_subsample,
            step_of_subsampling=args.step_of_subsampling,
            size_largest_subsample=np.inf,
            adress_results=args.adress_results,
            LIST_METHODS=args.methods
        )


if __name__ == "__main__":
    main()
