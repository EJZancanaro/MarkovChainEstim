
import markovchain
import pandas as pd


rmsd_df = pd.read_csv("./AFICS/results/Cr-results-RMSDs.csv")

rmsd_df['Best geometry'] = rmsd_df.idxmin(axis=1)

trajectory = rmsd_df['Best geometry'].tolist()

mchain = markovchain.MarkovChain()

for element in trajectory:
    mchain.next_state(element)

