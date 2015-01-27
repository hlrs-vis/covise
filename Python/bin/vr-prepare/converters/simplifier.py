# NOTE: Only 2D parts will be processed. 3D parts will be ignored and must be copied by hand if needed.
# TODO: I'm not sure if this works with multiple variables. Depends on wether the SimplifySurface produces exactly the same result on each run.
try:
    import cPickle as pickle
except:
    import pickle

import sys, os, shutil
from coPyModules import RWCovise, SimplifySurface
from coviseCase import (
    CoviseCaseFileItem,
    CoviseCaseFile,
    GEOMETRY_2D,
    GEOMETRY_3D,
    SCALARVARIABLE,
    VECTOR3DVARIABLE)

# params
theCoCaseFileName = "/data/Kunden/Daimler/W221_Familientag/Temperature/CoviseDaten/reduced/fahrzeug.cocase"
theOutputSubDir = "simplified"
thePercent = 5.0 # lower means smaller
theMaxNormalDeviation = 5.0

theNet = net()
theGridRCovise = RWCovise()
theDataRCovise = RWCovise()
theGridWCovise = RWCovise()
theDataWCovise = RWCovise()
theSimplify = SimplifySurface()
theNet.add(theGridRCovise)
theNet.add(theDataRCovise)
theNet.add(theGridWCovise)
theNet.add(theDataWCovise)
theNet.add(theSimplify)
theSimplify.set_percent(thePercent)
theSimplify.set_max_normaldeviation(theMaxNormalDeviation)

cocase = CoviseCaseFile()
inputFile = open(theCoCaseFileName, 'rb')
cocase = pickle.load(inputFile)
inputFile.close()

try:
    os.mkdir(os.path.dirname(theCoCaseFileName) + '/' + theOutputSubDir)
except:
    pass
shutil.copy(theCoCaseFileName, os.path.dirname(theCoCaseFileName) + '/' + theOutputSubDir + '/.')

i = 1
for item in cocase.items_:
    print(i, "/", len(cocase.items_), ":", item.geometryFileName_)
    if (item.dimensionality_ == GEOMETRY_2D):
        print("CONVERTING")
        theGridRCovise.set_grid_path(os.path.dirname(theCoCaseFileName) + '/' + item.geometryFileName_)
        theGridWCovise.set_grid_path(os.path.dirname(theCoCaseFileName) + '/' + theOutputSubDir + '/' + item.geometryFileName_)
        theNet.connect(theGridRCovise, 'mesh', theSimplify, 'meshIn')
        theNet.connect(theSimplify, 'meshOut', theGridWCovise, 'mesh_in')
        if (len(item.variables_) == 0):
            theGridRCovise.execute()
            theNet.finishedBarrier()
        else:
            theNet.connect(theDataRCovise, 'mesh', theSimplify, 'dataIn_0')
            theNet.connect(theSimplify, 'dataOut_0', theDataWCovise, 'mesh_in')
            firstRun = True
            for var in item.variables_:
                theDataRCovise.set_grid_path(os.path.dirname(theCoCaseFileName) + '/' + os.path.basename(var[1]))
                theDataWCovise.set_grid_path(os.path.dirname(theCoCaseFileName) + '/' + theOutputSubDir + '/' + os.path.basename(var[1]))
                if firstRun:
                    theGridRCovise.execute()
                theDataRCovise.execute()
                theNet.finishedBarrier()
                if firstRun:
                    theNet.disconnect(theSimplify, 'meshOut', theGridWCovise, 'mesh_in')
                    firstRun = False
        theNet.disconnectAllFromModule(theSimplify)
    else:
        print("IGNORING")
    i += 1

sys.exit()
