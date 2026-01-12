from MarkovChain.AFICS import rmsd


results_folder = "./results/"

outputFile = results_folder + 'Cr-results'
traj = rmsd.Trajectory(ionID='Cr', elements=['O'], boxSize=12.42, framesForRMSD=100, binSize=0.02, startFrame=1, endFrame=10000) 
traj.getAtoms('./data/Cr-aqua-veryshort.xyz')

traj.getIonNum()
if (traj.ionNum > 1): 
    traj.getWhichIon()
traj.getRDF()
traj.getDist()
traj.getMaxR()
traj.printRDF(outputFile)
traj.checkWithUser() 
traj.getThresholdAtoms()
traj.getADF()
traj.printADF(outputFile)
traj.getIdealGeos()
traj.getRMSDs()
traj.printRMSDs(outputFile)
traj.outputIdealGeometries(results_folder)
