import rmsd_multiple_CN

#in what folder to store the results. Make sure the folder already exists.
results_folder = "./results/Cr/"

#Prefix_of_AFICS_output_files (not including geometry fitting)
prefix_output_files="Cr-water"

#MD simulation data to analyse
data_address='./data/Cr-aqua.xyz'
outputFile = results_folder + prefix_output_files

#DO NOT FORGET TO UPDATE ionID, elemtents, and endFrame in the following function
traj = rmsd_multiple_CN.Trajectory(ionID='Cr', elements=['O'], boxSize=12.42, framesForRMSD=100, binSize=0.02, startFrame=1, endFrame=10000)

#Nothing to change below this line
traj.getAtoms(data_address)
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
