# MarkovChain
Tool for statistical estimation of parameters of sequences of values, supposed to be samplings of first order homogenous Markov Chains.

A class called MarkovChain is the main object at play, and the methods are the novelties of the project. In particular the objective is to implement trustworthy confidence-interval estimation, with theoretical garanties.


## How to use MarkovChainEstim

After cloning this repository and installing an environment with all the libraries specified in requirements.txt, MarkovChainEstim is ready to be used.

### From MD simulation data to the (modified) AFICS

First, insert MD simulation data in the "AFICS/data/" folder. An example of a Cromium trajectory, and of a smaller Cromium trajectory are already present there.

You can then modify the parameters of run_AFICS_multiple_CN.py so that they fit your MD data:
- results_folder: folder in which to save the outputs AFICS, i.e the fitting ideal geometries taken, the RDF plot, the ADF plot, and the diffrent RMSE values of the system compared to all taken geometries frame by frame.
- prefix_output_files: prefix to add to the files output by AFICS.
- data_address: where the .xyz file to be studied is
- IonID: Name of the ion of interest in the xyz file. Beware, it is case sensitive.
- elements: List elements to be taken into account as coordination elements.
- framesforRMSD: number of frames a choosen ideal geometry is considered to remain true before rechecking if the geometry hasn't changed. The larger this number is, the faster the computation but the less reliable the obtained geometries are.
- binSize: width of the histogram to compute the RDF.
- startFrame: first frame of the .xyz file to be taken into account.
- endFrame: last frame of the .xyz file to be taken into account. Should equal the number of frames in the .xyz file if we wish to take into account the entire MD simulation.

Executing the run_AFICS_multiple_CN.py file after choosing all these parameters outputs the AFICS realization into the folder written in the "results_folder" variable.

The exact modifications applied to the original AFICS for our purposes are described at AFICS/modifications_in_AFICS.md
## From the (modified) AFICS to MarkovChains

The RMSD.csv file output during previous step can then be fed into markov chains by executing
```python
import pandas as pd
import markovchain
rmsd_df = pd.read_csv(rmsd_file) #where rmsd_file is the previously output file
rmsd_df['Best geometry'] = rmsd_df.idxmin(axis=1)
trajectory = rmsd_df['Best geometry'].tolist()
MC = markovchain.MarkovChain()
for element in trajectory:
    MC.next_state(element)
```

This instantiates a MarkovChain object with the ideal geometries of the molecular dynamics system.

Examples of usage, specifically for benchmarking, are in the "run_markov_from_AFICS.py" file.

Further help on the implementation of the MarcovChain class can be found in the markovchain.py file.

## Benchmarking the confidence interval methods

Depending on whether the user wishes to see how the length of the confidence intervals evolves for either a theoretical Markov chain or for one obtained from the ideal geometries of AFICS, the user can respectively either call benchmark_given_matrix or benchmark_from_AFICS.

The benchmarking procedure consists in subsampling the entire trajectory and see how the confidence intervals estimated by each method behave with these successively larger subsamples the sample realization.

run_markov_from_AFICS.py shows examples on how to use both benchmarking functions.

## Convergence for multiple RMSD data

run_all_AFICS_data.py is specifically a file for launching this bencharking on multiple AFICS RMSDs results.