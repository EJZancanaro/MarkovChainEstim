# MarkovChain
Tool for statistical estimation of parameters of sequences of values, supposed to be samplings of first order Markov Chains.

A class called MarkovChain is the main object at play, and the methods are the novelties of the project. In particular the objective is to implement trustworthy confidence-interval estimation, with theoretical garanties.

After generating an AFICS analysis of data thanks to "AFICS/run_AFICS.py", data will be stored in the "AFICS/data/" folder.
Then, this data can be accessed by the "run_markov_from_AFICS.py" file, in order