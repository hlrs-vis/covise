try:
    import cPickle as pickle
except:
    import pickle

import sys, os, shutil
from coPyModules import RWCovise, GetSubset
from coviseCase import (
    CoviseCaseFileItem,
    CoviseCaseFile,
    GEOMETRY_2D,
    GEOMETRY_3D,
    SCALARVARIABLE,
    VECTOR3DVARIABLE)

# params
theCoCaseFileName = "/data/Kunden/Daimler/W221_Familientag/Temperature/CoviseDaten/fahrzeug.cocase"
theOutputSubDir = "reduced"
#theFilterSelection = "1-40"
#theFilterSelection = "1 5 9 13 17 21 25 29 33 37 41 45 49 53 57 61 65 69 73 77 81 85 89 93 97 101 105 109 113 117 121 125 129 133 137 141 145 149 153 157"
#theFilterSelection = "1-10"
#theFilterSelection = "0 40 80 120"
theFilterSelection = "1 5 9 13 17 21 25 29 33 37"

theNet = net()
theRCovise = RWCovise()
theWCovise = RWCovise()
theGetSubset = GetSubset()
theNet.add(theRCovise)
theNet.add(theWCovise)
theNet.add(theGetSubset)
theGetSubset.set_selection(theFilterSelection)
theNet.connect(theRCovise, 'mesh', theGetSubset, 'DataIn0')
theNet.connect(theGetSubset, 'DataOut0', theWCovise, 'mesh_in')

cocase = CoviseCaseFile()
inputFile = open(theCoCaseFileName, 'rb')
cocase = pickle.load(inputFile)
inputFile.close()

try:
    os.mkdir(os.path.dirname(theCoCaseFileName) + '/' + theOutputSubDir)
except:
    pass
shutil.copy(theCoCaseFileName, os.path.dirname(theCoCaseFileName) + '/' + theOutputSubDir + '/.')

# get all used .covise filenames from the cocase
coviseFileNames = []
for item in cocase.items_:
    coviseFileNames.append(item.geometryFileName_)
    for var in item.variables_:
        coviseFileNames.append(os.path.basename(var[1]))


# do the reducing
i = 1
for coviseFileName in coviseFileNames:
    theRCovise.set_grid_path(os.path.dirname(theCoCaseFileName) + '/' + coviseFileName)
    theWCovise.set_grid_path(os.path.dirname(theCoCaseFileName) + '/' + theOutputSubDir + '/' + coviseFileName)
    
    print(i, "/", len(coviseFileNames), ": ", coviseFileName)
    
    theRCovise.execute()
    theNet.finishedBarrier()
    
    i += 1


sys.exit()
