import rmsd_multiple_CN

#in what folder to store the results. Make sure the folder already exists.
results_folder = "./results/Ca_water/"

#Prefix_of_AFICS_output_files (not including geometry fitting)
prefix_output_files="CA_water"

#MD simulation data to analyse
data_address='./data/CA_center_lbda60_soltip3_md10ns.xyz'
outputFile = results_folder + prefix_output_files

#DO NOT FORGET TO UPDATE ionID, elemtents, and endFrame in the following function
traj = rmsd_multiple_CN.Trajectory(ionID='CA', elements=['O'], boxSize=12.42, framesForRMSD=100, binSize=0.02, startFrame=1, endFrame=101)

#Nothing to change below
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

print(traj.RMSDs)