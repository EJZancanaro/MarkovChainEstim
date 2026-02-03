import pandas as pd
import matplotlib.pyplot as plt
import os


if __name__=='__main__':
    adress_results = "./benchmark_results/Results_Simulation_PIR/2PVB-RMSDs" # same folder
    LIST_METHODS = ['Gaussian', 'GaussianSlutsky', 'BasicChi2', 'BasicSlutskyChi2', 'FreerChi2','FreerSlutskyChi2']

    # Largest CI plot
    plt.figure()
    plt.title("Largest confidence interval length among all pairs of states")
    for method in LIST_METHODS:
        summary = pd.read_csv(os.path.join(adress_results, f"summary_{method}.csv"))
        plt.loglog(summary["n"], summary["max_length"], label=method)

    plt.xlabel("Size of the Markov chain sample")
    plt.ylabel("Largest confidence interval length")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(adress_results, f"largest_clean.png"))

    # Smallest CI plot
    plt.figure()
    plt.title("Smallest confidence interval length among all pairs of states")
    for method in LIST_METHODS:
        summary = pd.read_csv(os.path.join(adress_results, f"summary_{method}.csv"))
        plt.loglog(summary["n"], summary["min_length"], label=method)

    plt.xlabel("Size of the Markov chain sample")
    plt.ylabel("Smallest confidence interval length")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(adress_results, "smallest_clean.png"))

    plt.show()
