import rmsd_multiple_CN

results_folder = "./results/"

outputFile = results_folder + '2pvb_results'
traj = rmsd_multiple_CN.Trajectory(ionID='CA', elements=['O'], boxSize=12.42, framesForRMSD=100, binSize=0.02, startFrame=1, endFrame=101)
traj.getAtoms('./data/2pvb_final_aqv_lbda60_soltip3_md10ns.xyz')


traj.getIonNum()
if (traj.ionNum > 1): 
    traj.getWhichIon() 
traj.getRDF() 
traj.getDist()
traj.getMaxR()
traj.printRDF(outputFile)
#traj.checkWithUser() #This may be commented out if user does not wish to be prompted to confirm/reject predicted RDF maximum, first shell threshold, and coordination number
traj.getThresholdAtoms()
traj.getADF()
traj.printADF(outputFile)
traj.getIdealGeos()
traj.getRMSDs()
traj.printRMSDs(outputFile)
traj.outputIdealGeometries(results_folder)
traj.printNbCN(outputFile) #Location name may be entered in string if user desires geometries to be written to subfolder instead of working directory 
traj.printPlace(outputFile)