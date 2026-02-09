import numpy as np
import csv
import geometries
import itertools
from scipy.interpolate import CubicSpline
import os

class Trajectory ():
    def __init__(
                    self,
                    atoms=[],
                    atomNum=0,
                    frameCount=0,
                    ionID='',
                    ionNum=0,
                    whichIon=1,
                    elements=[],
                    boxSize=10.0,
                    rdf=[],
                    rdfIntegral=[],
                    adf=[],
                    binSize=0.05,
                    dist=0.0,
                    coorNum=0,
                    maxR=0.0,
                    missingFrames=[],
                    RMSDs=[],
                    framesForRMSD=100,
                    idealGeos=[],
                    startFrame=None,
                    endFrame=100000000,
                    
                    nbCN=[],
                    idealGeos2=[],
                    autres_frames=[],
                    place_atome=[]
                    ):
        self.atoms = atoms              # array of atoms and their respective coordinates
        self.atomNum = atomNum          # number of atoms per frame
        self.frameCount = frameCount    # number of frames in the trajectory
        self.ionID = ionID              # string of the name of the central ion for rdf, ie. 'Lu' (needs to changed to accomadate more than one of the same ion)
        self.ionNum = ionNum            # number of ions with the ion name, ionID
        self.whichIon = whichIon        # if there are multiple of the same ion, this specifies which one to center the rdf around (not indexed at 0), by default gets 1st one
        self.elements = elements        # elements to include in the rdf/adf ie. 'O', 'N'
        self.boxSize = boxSize          # size of the cell
        self.rdf = rdf                  # array for rdf bins used in histogram
        self.rdfIntegral = rdfIntegral  # array for integral of rdf used in histogram
        self.adf = adf
        self.binSize = binSize          # size of bin in histogram
        self.dist = dist                # average distance in first sphere, ie max of rdf
        self.coorNum = coorNum          # coordination number
        self.maxR = maxR                # maximum distance of atoms within the first coordination sphere
        self.thresholdAtoms = []        # atoms within the first coordination sphere
        self.missingFrames = missingFrames  # index of frames that were not able to be collected
        self.RMSDs = RMSDs              # array holding RMSD values
        self.framesForRMSD = framesForRMSD  # number of frames each optimal permutation is used for, higher numbers make the program faster
        self.idealGeos = idealGeos      # array containing the ideal geometries
        self.startFrame = startFrame
        self.endFrame = endFrame
        
        
        self.nbCN=nbCN
        self.idealGeos2 = idealGeos2
        self.autres_frames=autres_frames
        self.place_atome=place_atome

    def getRMSDs(self):
        # [ ['geometry name', [[x,y,z],[x,y,z],...], ... ]
        print ('Calculating RMSDs... (this may take a while)')
        geoNames = []
        geo_temp=[]
        temp2=[]
        
        for ideal in self.idealGeos:
            
            for i in range(len(ideal[1])):
                    
                geo_temp.append(ideal[1][i][0])
            
            temp2.append(ideal[0])
            temp2.append(geo_temp.copy())
            geoNames.append(temp2.copy())
            temp2.clear()
            geo_temp.clear()

        nbcoor=0

        for i, frame in enumerate(self.thresholdAtoms):
            if (i % self.framesForRMSD == 0):
                for k,geopos in enumerate(self.idealGeos):
                    
                    if frame[0] == geopos[0]:
                        self.idealGeos[k][1] = reorderIdealGeometries(frame[1], geopos[1])
            tempRMSDs = []
            temp_frames=[]
            for ideal in self.idealGeos:
                if frame[0] == ideal[0]:
                    for i in ideal[1]:
                        tempRMSDs.append(str(kabschRMSD(frame[1], i[1])))
                        nbcoor=ideal[0]
            for names in geoNames:
                if nbcoor == names[0]:
                    temp_frames.append(names[1])
                    
            temp_frames.append(tempRMSDs.copy())
            self.RMSDs.append(temp_frames.copy())
            self.nbCN.append(frame[0])
        print ('RMSDs done')

    def getAtoms(self, filename):
        '''
        Retrieves atoms from an .xyz file and saves all the elements and their respective
        coordinates per frame as an array in the form:
        [ [ ['element', [x, y, z]], ['element', [x, y, z]], ... ]... ] 
        '''
        with open(filename, 'r') as f:
            frame = []
            if self.startFrame is not None and self.endFrame: # if startFrame and endFrame are numbers
                if self.startFrame < 1:
                    print ('Starting frame must be greater than 0')
                    return None
                if self.startFrame < self.endFrame:
                    if self.endFrame != 100000000:
                        print ('Collecting atoms from frame '+str(self.startFrame)+' to frame '+str(self.endFrame)+'...')
                    else:
                        print ('Collecting atoms starting from frame'+str(self.startFrame)+'...')
                    framesCounter = 0
                    for i, row in enumerate(f):
                        row = row.split()
                        if i == 0:
                                self.atomNum = int(row[0])
                        if (i % (self.atomNum + 2) == 0):
                            if row: # ensures its not an empty line at the end of the file
                                framesCounter += 1
                        if (framesCounter >= self.startFrame and framesCounter <= (self.endFrame + 1)):
                            if (i % (self.atomNum + 2) != 0) and (i % (self.atomNum + 2) != 1): # + 2 for the atom count and comment line of .xyz
                                frame.append([row[0], [float(row[1]), float(row[2]), float(row[3])]])
                            if (i % (self.atomNum + 2) == 0) and (self.frameCount != 0):
                                self.atoms.append(frame.copy())
                                frame.clear()
                            if (i % (self.atomNum + 2) == 0):
                                if row:
                                    self.frameCount += 1
                else:
                    print ('End frame must be a larger number than startframe')
            else:
                print ('Collecting all atoms...')
                for i, row in enumerate(f):
                    row = row.split()
                    if i == 0:
                        self.atomNum = int(row[0])
                    if (i % (self.atomNum + 2) != 0) and (i % (self.atomNum + 2) != 1): # + 2 for the atom count and comment line of .xyz
                        frame.append([row[0], [float(row[1]), float(row[2]), float(row[3])]])
                    if (i % (self.atomNum + 2) == 0) and (i != 0):
                        self.atoms.append(frame.copy())
                        frame.clear()
                    if (i % (self.atomNum + 2) == 0):
                        if row:
                            self.frameCount += 1

        print ('Atoms collected')
    
    def getThresholdAtoms(self):
        '''
        Retrieves atoms in the first coordinate sphere of the ion
        Format:
        [ [ [x, y, z], [x, y, z], ... ], ... ]
        '''
        frames_cn=[]
        frames_cn2=[]
        
        place=[]
        for i, frame in enumerate(self.atoms):
            frameAtoms = []
            ionCoords = []
            ionCount = 0

            #Find the coordinates of the ion of interest among the class of ions we wish to study
            for atom in frame:
                if atom[0] == self.ionID:
                    ionCount += 1
                    if self.whichIon == ionCount:
                        ionCoords = atom[1]
                        break
                    
            # on parcourt tous les atomes de chaque frame, puis on garde dans frameAtoms
            # ceux qui sont dans les éléments choisis et qui sont dans la 1ere sphère coordination

            for compt,atom in enumerate(frame):
                
                if (atom[0] in self.elements) and (np.linalg.norm(np.array(atom[1]) - np.array(ionCoords),2) <= self.maxR):
                    frameAtoms.append(atom[1])
                    place.append(atom[0])
                    place.append(compt)
                    place.append(i)
                    self.place_atome.append(place.copy())
                    place.clear()
                    
                
                    
            # on stocke dans thresholdAtoms les frames pour lesquelles on a le bon CN
            # ie le bon nb d'atomes autour de l'ion 
            #si c'est pas le cas on clear la liste on stocke dans missingFrames le numéro de la frame où y'a pas le bon CN
            # on affihe le numero de ces frames
            
            frames_cn.append(len(frameAtoms))
            frames_cn.append(np.array(frameAtoms.copy()))

            # every thresholdAtoms[i] is list, whose 0th element contains the number of atoms of interest (i.e the coordination number for this frame),
            # the next elements are the coordinates of those atoms of interest
            self.thresholdAtoms.append(frames_cn.copy())

            
            frames_cn.clear()
        print ('First sphere atoms complete')
        
        
        
    def printPlace(self,filename):
        # Nom du fichier CSV de sortie
        
        
        # Écriture dans le fichier CSV
        with open(filename+'-emplacement_atomes.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            
            # Écriture de l'en-tête
            writer.writerow(['Atome', 'Numéro de l\'atome', 'Frame'])
            
            # Écriture des données
            writer.writerows(self.place_atome)

    def getDist(self):
        '''
        Calculates the maximum of the RDF to be used as the average distance
        in generating ideal structures
        '''
        
        # on parcourt les valeurs de la rdf, on récupère la valeur de la rdf tant que la rdf
        # est croissante et on récupère l'index a chaque fois
        # quand la rdf commence a decroitre on incrémente breakcount jusqu'a 5

        bin = self.binSize # bin size is not adjustable by user and is set at 0.05 A
        r_max = 6.0 # the max the rdf will go to is 6 A
        data_num = int(r_max / bin)
        
        x= np.arange(data_num)
        x=x/(data_num/r_max)
        spline = CubicSpline(x, self.rdf)
        derivative = spline.derivative()
        roots = derivative.roots()
        
        Max=0
        
        for i in roots:
            if spline(i) > Max:
                Max = spline(i)
                max_ind=i
        
        
        self.dist = max_ind

    def getMaxR(self):
        '''
        Calculates the maximum radius for determining atoms within the first coordination sphere and the coordination number
        '''
        breakCount = 0
        # on parcourt les valeurs de l'intégrale de la rdf
        for i, value in enumerate(self.rdfIntegral):
            
            # on regarde uniquement les bins supérieurs au max de rdf (en abscisse)
            # on commence sur la moitié "montée" de l'intégrale, dans cette phase on ne rentre dans aucune boucle if
            # break count est alors egale à 0, une fois que la courbe se stabilise sur un bin break count prend la valeur 1
            # puis si on a encore une légère variation <0.002 c'est stable, on est sur notre
            # sur la fin de notre pic
            # donc on récupère la valeur de l'intégrale de RDF qui est le CN et 
            # maxR = maximum distance of atoms within the first coordination sphere
            if (i > (self.dist/self.binSize)):
                if (value - self.rdfIntegral[i - 1] < 0.002) and (breakCount == 0):
                    breakCount +=1
                elif (value - self.rdfIntegral[i - 1] > 0.002) and (breakCount == 1):
                    breakCount -=1
                elif (value - self.rdfIntegral[i - 1] < 0.002) and (breakCount == 1):
                    self.coorNum = round(value)
                    self.maxR = self.binSize * i
                    break

    def printRDF(self, filename):
        '''
        Outputs RDF and RDF integral as .csv files
        '''
        RDFHeader = ['Distance (A)', 'Count']
        RDFIntHeader = ['Distance (A)', 'Integral']
        RDFout = []
        RDFIntout = []
        for i, value in enumerate(self.rdf):
            RDFout.append([(i * self.binSize), value])
        for i, value in enumerate(self.rdfIntegral):
            RDFIntout.append([(i * self.binSize), value])
        with open(filename+'-rdf.csv', 'w') as rdffile:
            rdfwriter = csv.writer(rdffile)
            rdfwriter.writerow(RDFHeader)
            rdfwriter.writerows(RDFout)
        with open(filename+'-rdf-integral.csv', 'w') as rdffile:
            rdfwriter = csv.writer(rdffile)
            rdfwriter.writerow(RDFIntHeader)
            rdfwriter.writerows(RDFIntout)

    def printADF(self, filename):
        '''
        Outputs RDF as .csv files
        '''
        #print("adf",self.adf)
        ADFdict = {}
        for i in range(181):
            ADFdict[i] = 0 # initializes 180 entries for adf histogram of bin size 1 degree
        for value in self.adf:
            ADFdict[int(np.floor(value))] += 1
        with open(filename+'-adf.csv', 'w') as adffile:
            adfwriter = csv.writer(adffile)
            adfwriter.writerow(['Angle', 'Count'])
            for key in ADFdict:
                adfwriter.writerow([key, ADFdict[key]])
                
    
    def printNbCN(self, filename):
        with open(filename + '-Nbcoordination.csv', 'w', newline='') as csvfile:
            nbCN_writer = csv.writer(csvfile)
            nbCN_writer.writerow(['Coordination'])  # Header for the column
            nbCN_writer.writerows(map(lambda x: [x], self.nbCN))

    def printRMSDs(self, filename):
        '''
        Génère un fichier .csv contenant les RMSDs
        '''
        # Récupère toutes les géométries idéales uniques
        all_geometries = set()
        for frame_data in self.RMSDs:
            all_geometries.update(set(frame_data[0]))
    
        # Trie la liste des géométries idéales
        sorted_geometries = sorted(list(all_geometries))
    
        # Ouvre le fichier CSV en mode écriture
        with open(filename + '-RMSDs.csv', 'w') as rmsdfile:
            # Crée un objet writer pour écrire dans le fichier CSV
            rmsdfwriter = csv.writer(rmsdfile)
    
            # Écrit la première ligne du fichier CSV avec les noms de géométries idéales
            rmsdfwriter.writerow([''] + sorted_geometries)
    
            # Parcourt les données pour chaque frame et écrit les lignes correspondantes dans le fichier CSV
            for frame_data in self.RMSDs:
                frame_row = ['']  # La première colonne de chaque ligne est laissée vide
                rmsd_dict = dict(zip(frame_data[0], frame_data[1]))
    
                # Remplit les colonnes avec les valeurs RMSDs correspondantes ou 100 si la géométrie n'est pas présente
                for geometry in sorted_geometries:
                    #frame_row.append(rmsd_dict.get(geometry, 100))
                    frame_row.append(rmsd_dict.get(geometry))
    
                # Écrit la ligne dans le fichier CSV
                rmsdfwriter.writerow(frame_row)

    def outputIdealGeometries(self, subfolder=''):
        import os
        if len(subfolder) and not os.path.exists(subfolder):
            os.makedirs(subfolder)

        # Iterate through the Union we built in getIdealGeos
        for cn_entry in self.idealGeos:
            current_cn = cn_entry[0]
            geos_for_this_cn = cn_entry[1]

            for ideal in geos_for_this_cn:
                name = ideal[0]
                coords = ideal[1]
                # Header must match the actual number of atoms in THIS geometry
                header = [
                    [str(len(coords) + 1)],
                    [f"CN={current_cn}: {name}"]
                ]
                path = os.path.join(subfolder, f"CN{current_cn}_{name}.xyz")
                with open(path, 'w') as f:
                    csvwriter = csv.writer(f)
                    csvwriter.writerows(header)
                    csvwriter.writerow(['O 0.0 0.0 0.0'])  # Ion
                    for row in coords:
                        csvwriter.writerow(['H ' + str(row[0]) + ' ' + str(row[1]) + ' ' + str(row[2])])
    def getIonNum(self):
        '''
        Finds the total number of ions specified by the ionID per frame
        '''
        for frame in self.atoms:
            for atom in frame:
                if atom[0] == self.ionID:
                    self.ionNum += 1
            break

    def getWhichIon(self):
        ionsCoords = []
        for frame in self.atoms:
            for atom in frame:
                if atom[0] == self.ionID:
                    ionsCoords.append(atom[1])
        print ('There are '+str(self.ionNum)+' '+str(self.ionID)+' ions.')
        print ('Enter which one you would like to use:')
        for i in range(self.ionNum):
            print (str(i + 1)+' - '+str(self.ionID)+' at ('+str(ionsCoords[i][0])+', '+str(ionsCoords[i][1])+', '+str(ionsCoords[i][2])+')')
        while (True):
            usr_input = input ('Enter which ion to use: ')
            try:
                if (int(usr_input) > 0 and int(usr_input) <= self.ionNum):
                    self.whichIon = int(usr_input)
                    break
                else:
                    print ('Enter an integer between 1 and '+str(self.ionNum))
            except:
                print ('Enter an integer')


    def getRDF(self):
        """
        Obtains the RDF and integral of the RDF using bins of 0.05 A width
        """
        print ('Calculating RDF... (this may take a while)')
        bin = self.binSize # bin size is not adjustable by user and is set at 0.05 A
        r_max = 6.0 # the max the rdf will go to is 6 A
        data_num = int(r_max / bin)

        vol = 4.0/3.0 * np.pi * np.power(r_max, 3)
        density_A = 1.0 / vol

        distances = []

        for frame in self.atoms:
            frame_dist = []
            ion_coords = []
            # get ion coordinates
            ionCount = 0
            for atom in frame:
                if atom[0] == self.ionID:
                    ionCount += 1
                    if self.whichIon == ionCount:
                        ion_coords = atom[1]
                        break
            for atom in frame:
                if atom[0] in self.elements:
                    frame_dist.append(np.linalg.norm(np.array(ion_coords) - np.array(atom[1]),2))
            distances.append(frame_dist.copy())
            frame_dist.clear()
            
            
        atom_B_tol_num = []

        rdf_values = []
        rdfint_values = []
        for i, frame in enumerate(distances):
            
            
            tmp_dist_count = 0
            for distance in frame:
                if distance <= r_max:
                    tmp_dist_count += 1
            atom_B_tol_num.append(tmp_dist_count)
            
            
            density_B = atom_B_tol_num[i] / vol
            tmp_rdfint_values = []

            tmp_rdf_values = []
            for binNum in range(data_num):
                num_B_r = 0.0
                for distance in frame:
                    
                    if (distance >= binNum*bin) and (distance < (binNum + 1)*bin):
                        num_B_r += 1
                tmp_rdf_values.append(1.0 / (density_A*atom_B_tol_num[i])*num_B_r / (4.0*np.pi*(binNum + 1)**2 * bin**3))
                
                sum_int = 0.0
                for k in range(binNum):
                    
                    sum_int += 4*np.pi*density_B*((k + 1)*bin)**2 * tmp_rdf_values[k]*bin
                tmp_rdfint_values.append(sum_int)
            rdfint_values.append(tmp_rdfint_values.copy())
            tmp_rdfint_values.clear()
            
            rdf_values.append(tmp_rdf_values.copy())
            tmp_rdf_values.clear()


     
       
        for i in range(data_num):
            sum1 = 0.0
            sum2 = 0.0
            for j in range(len(rdfint_values)):
                sum1 += rdfint_values[j][i]
            self.rdfIntegral.append(sum1 / self.frameCount)
            for j in range(len(rdf_values)):
                sum2 += rdf_values[j][i]
            self.rdf.append(sum2 / self.frameCount)
        print ('RDF complete')
        
        
        
        

    def getADF(self):
        '''
        Calculates the ADF for the treshold atoms in each frame
        '''
        
        for frame in self.thresholdAtoms:
            
            # center each frame using centroid
            frame[1] -= centroid(frame[1])
            for i in range(len(frame[1]) - 1):
                for j in range(i + 1, len(frame[1])):
                    self.adf.append(angle_eq(frame[1][i], frame[1][j]))
                    
            
        print ('ADF complete')
        

    def getIdealGeos(self):
        if len(self.idealGeos)!=0:
            exit("Fatal error: getIdealGeos was called multiple times.")

        cn_geo=[]
        
        list_cn=[]

        #For every frame
        for i,fr in enumerate(self.thresholdAtoms):
            #every thresholdAtoms[i] is list, whose 0th element contains the number of atoms of interest (i.e the coordination number for this frame),
            # the next elements are the coordinates of those atoms of interest

            if self.thresholdAtoms[i][0] not in list_cn:
                list_cn.append(self.thresholdAtoms[i][0])
        #val_cn now contains all values the coordination number takes (with unique elements)
        for i,val_cn in enumerate(list_cn):
            # every element of idealGeos will have val_cn and a list of the possible ideal geometries for that value of coordination number
            # [CN, [list_of_geometries]]
            self.idealGeos.append([val_cn, geometries.idealGeometries(val_cn, self.dist)])

    def checkWithUser(self):
        print ('Do the following look OK?')
        while (True):
            print ('\tMax of RDF: '+str('{:.2f}'.format(self.dist)))
            YN_inp = input ('Y/N?: ')
            try:
                if (YN_inp.upper() == 'Y' or YN_inp.upper() == 'YES'):
                    break
                elif (YN_inp.upper() == 'N' or YN_inp.upper() == 'NO'):
                    r_inp = input ('Enter value for the max of the RDF: ')
                    try:
                        self.dist = float(r_inp)
                        break
                    except:
                        print ('Enter a floating point number')
                else:
                    print ('Enter Y or N')
            except:
                print ('Enter Y or N')
        while (True):
            print ('\tThreshold of 1st coordination sphere: '+str('{:.2f}'.format(self.maxR)))
            YN_inp = input ('Y/N?: ')
            try:
                if (YN_inp.upper() == 'Y' or YN_inp.upper() == 'YES'):
                    break
                elif (YN_inp.upper() == 'N' or YN_inp.upper() == 'NO'):
                    r_inp = input ('Enter value for the threshold of the coordination sphere: ')
                    try:
                        self.maxR = float(r_inp)
                        break
                    except:
                        print ('Enter a floating point number')
                else:
                    print ('Enter Y or N')
            except:
                print ('Enter Y or N')
        while (True):
            print ('\tCoordination number: '+str(int(self.coorNum)))
            YN_inp = input ('Y/N?: ')
            try:
                if (YN_inp.upper() == 'Y' or YN_inp.upper() == 'YES'):
                    break
                elif (YN_inp.upper() == 'N' or YN_inp.upper() == 'NO'):
                    r_inp = input ('Enter value for the coordination number: ')
                    try:
                        self.coorNum = int(r_inp)
                        break
                    except:
                        print ('Enter an integer number')
                else:
                    print ('Enter Y or N')
            except:
                print ('Enter Y or N')


def dist_eq(c1, c2):
    return np.sqrt( (c1[0] - c2[0])**2 + (c1[1] - c2[1])**2 + (c1[2] - c2[2])**2 )

def angle_eq(c1, c2):
    # make sure to center coordinates around origin first
    a = dist_eq(c1, [0., 0., 0.])
    b = dist_eq(c2, [0., 0., 0.])
    c = dist_eq(c1, c2)
    return np.arccos( (a**2 + b**2 - c**2) / (2*a*b) ) * (180. / np.pi)

def centroid(arr):
    # a - 3D array of coordinates
    return np.mean(arr, axis=0)

def kabschRMSD(A, B):
    # kabsch algorithm for rotating/translating set of coordinates A onto B
    A = A - centroid(A)
    B = B - centroid(B)
    A = kabschRotate(A, B)
    return rmsd(A, B)

def kabschRotate(A, B):
    U = kabsch(A, B)
    A = np.dot(A, U)
    return A

def kabsch(A, B):
    C = np.dot(np.transpose(A), B)
    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]
    U = np.dot(V, W)
    return U

def rmsd(A, B):
    diff = np.array(A) - np.array(B)
    N = len(A)
    return np.sqrt((diff * diff).sum() / N)

def reorderIdealGeometries(real, ideals):
    minIdeals = ideals
    for i, ideal in enumerate(ideals):
        min = 999.
        for permutation in itertools.permutations(ideal[1]):
            checkPerm = kabschRMSD(real, permutation)
            if (checkPerm < min):
                min = checkPerm
                minIdeals[i][1] = np.array(permutation)
    return minIdeals

