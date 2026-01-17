
import os
import numpy as np
from benchmark_from_AFICS import benchmark_from_AFICS


if __name__ == '__main__':
    size_smallest_subsample = 1
    step_of_subsampling = 1
    size_largest_subsample = np.inf
    adress_results = "./benchmark_results/"
    LIST_METHODS = ['Gaussian', 'GaussianSlutsky', 'BasicChi2', 'BasicSlutskyChi2', 'FreerChi2', 'FreerSlutskyChi2']



    list_files= ["./AFICS/results/Cr/Cr-water-RMSDs.csv",
                          "./AFICS/results/Eu-EDTA/Eu-EDTA-RMSDs.csv",
                          "./AFICS/results/Ca_water/CA_water-RMSDs.csv",
                          "./AFICS/results/2pvb/2PVB-RMSDs.csv"
                ]

    # Ensure the parent results folder exists
    os.makedirs(adress_results, exist_ok=True)

    for rmsd_file in list_files:
        file_name_no_ext = os.path.splitext(os.path.basename(rmsd_file))[0]
        folder_for_file = os.path.join(adress_results, file_name_no_ext)
        os.makedirs(folder_for_file, exist_ok=True)
        benchmark_from_AFICS(rmsd_file=rmsd_file,
                             size_smallest_subsample=size_smallest_subsample,
                             step_of_subsampling=step_of_subsampling,
                             size_largest_subsample=size_largest_subsample,
                             adress_results=folder_for_file,
                             LIST_METHODS=LIST_METHODS)