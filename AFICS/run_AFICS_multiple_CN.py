import rmsd_multiple_CN

#in what folder to store the results. Make sure the folder already exists.
results_folder = "./results/Ca_c27/"

#Prefix_of_AFICS_output_files (not including geometry fitting)
prefix_output_files="CA-water-c27"

#MD simulation data to analyse
data_address='./data/calcium_c27_soltip3_md10ns_1000pts_centerCAL.xyz'
outputFile = results_folder + prefix_output_files

#Trajectory parameters

ionID='Ca' #ion of interest's name in the .xyz file
elements=['O'] # list of elements to be taken into account in the computations of RDF, ADF and CN.
framesForRMSD=100 #number of frames a given ideal geometry is used for before reconsidering what the ideal geometry is
binSize=0.02 #width of the histograms for approximating the RDF.
startFrame=1 #In the .xyz file, what frame is the first to be taken into account
endFrame=1001# In the .xys file, what frame is the last to be taken into account,
                # this should be exactly equal to the number of frames in the simulation if one whishes to analyse with AFICS the entire simulation.


#DO NOT FORGET TO UPDATE ionID, elemtents, and endFrame in the following function
traj = rmsd_multiple_CN.Trajectory(ionID=ionID, elements=elements, boxSize=12.42, framesForRMSD=framesForRMSD, binSize=binSize, startFrame=startFrame, endFrame=endFrame)

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
