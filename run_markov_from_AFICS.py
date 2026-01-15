
import markovchain
import pandas as pd
import numpy as np
import sys



#rmsd_df = pd.read_csv("AFICS/results/short-Cr/Cr-results-RMSDs.csv")

rmsd_df = pd.read_csv("AFICS/results/Ca_water/Ca_water-RMSDs.csv")
rmsd_df['Best geometry'] = rmsd_df.idxmin(axis=1)
trajectory = rmsd_df['Best geometry'].tolist()

MC = markovchain.MarkovChain()

for element in trajectory:
    MC.next_state(element)

print(trajectory)

state_space = list(MC.state_space)

print(state_space)
