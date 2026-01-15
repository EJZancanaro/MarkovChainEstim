# MarkovChain
Tool for statistical estimation of parameters of sequences of values, supposed to be samplings of first order Markov Chains.

A class called MarkovChain is the main object at play, and the methods are the novelties of the project. In particular the objective is to implement trustworthy confidence-interval estimation, with theoretical garanties.


## From AFICS to MarkovChains

Here we do not use the original AFICS, but a modified one that allows for taking into account ion-solvent systems with time-dependent coordination number.

After generating a (modified) AFICS analysis of data thanks to "AFICS/run_AFICS_multiple_CN.py", data will be stored in the "AFICS/data/" folder (or any folder specified by the user in said file).

Then, this data can be accessed by the "run_markov_from_AFICS.py" file. In that same file, an example of simple usage is implemented.

## Benchmarking the confidence interval methods
The benchmark_convergence.py file generates a Markov chain according to one of four matrices (the user can choose which one by commenting and uncommenting the sections of code defining the matrices P1, P2, P3, or P4), of specified length that user can also natively modify. Then, it subsamples the first $n$ samples, where $n$ goes from a given number to the nb_samples by a step the user can define. For every method, each one of these subsamples gives us a matrix of confidence intervals, of which we plot the largest and the smallest length.
The matrices and other information are saved in files that will be saved in a benchmark results folder.
