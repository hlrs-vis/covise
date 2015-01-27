# Delete covise-files smaller than this value.
MIN_FILESIZE = 80


import sys
import os

from time import sleep

from CoviseMsgLoop import CoviseMsgLoop, CoviseMsgLoopAction
from paramAction import NotifyAction
from coviseCase import (
    CoviseCaseFileItem,
    CoviseCaseFile,
    GEOMETRY_2D,
    GEOMETRY_3D,
    SCALARVARIABLE,
    VECTOR3DVARIABLE)
from ErrorLogAction import ErrorLogAction




# /dir/filename.ext -> filename
def getName(inFile):
    return os.path.splitext(os.path.basename(inFile))[0]

# /dir/filename.ext, -grid-2D -> ../CoviseDaten/filename-grid-2D.covise
def makeCoviseFilename(inFile, suffix):
    return coviseDatenDir + '/' + getName(inFile) + suffix + '.covise'



# prepare variables
# note: all files except the cocase name come with full path
if (scaleFactor == 1.0):
    scaleFactor = None
if (bofname == ""):
    bofname = None
if (cocasename == ""):
    if bofname:
        cocasename = getName(bofname)
    else:
        cocasename = "BifBofConversion"
if bofname and (variable == ""):
    variable = getName(bofname)



# LOG

logFileName = coviseDatenDir + '/' + cocasename + ".log"
logFile = open(logFileName, 'w')

logFile.write("\nOptions:")
logFile.write("\nCovise Data Directory = %s"%(coviseDatenDir,))
logFile.write("\nBof File Name = %s"%(bofname,))
logFile.write("\nScale Factor = %s"%(scaleFactor,))
logFile.write("\nCovise Case Name = %s"%(cocasename,))
logFile.write("\nVariable Name = %s"%(variable,))
logFile.write("\nBif File Names = %s"%(str(bifFiles),))
logFile.write("\n")
logFile.flush()
print("Options:")
print("Covise Data Directory = ", coviseDatenDir)
print("Bof File Name = ", bofname)
print("Scale Factor = ", scaleFactor)
print("Covise Case Name = ", cocasename)
print("Variable Name = ", variable)
print("Bif File Names = ", str(bifFiles))
print(" ")



# PREPARE

aErrorLogAction = ErrorLogAction()
CoviseMsgLoop().register(aErrorLogAction)

cocase = CoviseCaseFile()



# PREPARE MODULES

theNet = net()

ReadBIFBOF_1 = ReadBIFBOF()
theNet.add( ReadBIFBOF_1 )

SplitUsg_1 = SplitUsg()
theNet.add( SplitUsg_1 )

if scaleFactor:
    Transform_1 = Transform()
    theNet.add( Transform_1 )
    Transform_1.set_Transform( 5 )
    Transform_1.set_scaling_factor( scaleFactor )
    Transform_1.set_createSet( "FALSE" )
    theNet.connect( ReadBIFBOF_1, "GridOut", Transform_1, "geo_in" )
    theNet.connect( Transform_1, "geo_out", SplitUsg_1, "Grid" )
else:
    theNet.connect( ReadBIFBOF_1, "GridOut", SplitUsg_1, "Grid" )

theNet.connect( ReadBIFBOF_1, "skalarData", SplitUsg_1, "S_Data" ) # no need to scale scalar data

RWCovise3DGrid = RWCovise()
theNet.add( RWCovise3DGrid )
RWCovise3DGrid.set_stepNo( 0 )
RWCovise3DGrid.set_rotate_output( "FALSE" )
theNet.connect( SplitUsg_1, "Grid3D", RWCovise3DGrid, "mesh_in" )

RWCovise2DGrid = RWCovise()
theNet.add( RWCovise2DGrid )
RWCovise2DGrid.set_stepNo( 0 )
RWCovise2DGrid.set_rotate_output( "FALSE" )
theNet.connect( SplitUsg_1, "Grid2D", RWCovise2DGrid, "mesh_in" )

if bofname:

    RWCovise3DData = RWCovise()
    theNet.add( RWCovise3DData )
    RWCovise3DData.set_stepNo( 0 )
    RWCovise3DData.set_rotate_output( "FALSE" )
    theNet.connect( SplitUsg_1, "S_Grid3D_Data", RWCovise3DData, "mesh_in" )

    RWCovise2DData = RWCovise()
    theNet.add( RWCovise2DData )
    RWCovise2DData.set_stepNo( 0 )
    RWCovise2DData.set_rotate_output( "FALSE" )
    theNet.connect( SplitUsg_1, "S_Grid2D_Data", RWCovise2DData, "mesh_in" )




# CONVERT FILES

for bifname in bifFiles:

    logFile.write("\nCONVERTING %s" % (bifname,))
    logFile.flush()
    print("CONVERTING", bifname)

    ReadBIFBOF_1.set_BIFFile(bifname)
    if bofname:
        ReadBIFBOF_1.set_BOFFile(bofname)

    covFile3Dgrid = makeCoviseFilename(bifname, "-grid-3D")
    covFile3Ddata = makeCoviseFilename(bifname, "-data-3D")
    covFile2Dgrid = makeCoviseFilename(bifname, "-grid-2D")
    covFile2Ddata = makeCoviseFilename(bifname, "-data-2D")
    name = getName(bifname)
    RWCovise3DGrid.set_grid_path(covFile3Dgrid)
    RWCovise2DGrid.set_grid_path(covFile2Dgrid)
    if bofname:
        RWCovise3DData.set_grid_path(covFile3Ddata)
        RWCovise2DData.set_grid_path(covFile2Ddata)

    runMap()
    theNet.finishedBarrier()

    # 3D
    item3D = None
    if os.access(covFile3Dgrid, os.F_OK):
        if (os.path.getsize(covFile3Dgrid) >= MIN_FILESIZE):
            # valid file
            item3D = CoviseCaseFileItem(name, GEOMETRY_3D, covFile3Dgrid)
            cocase.add(item3D)
        else:
            os.remove(covFile3Dgrid)
    if os.access(covFile3Ddata, os.F_OK):
        if bofname and (os.path.getsize(covFile3Ddata) >= MIN_FILESIZE):
            # valid file
            if item3D:
                item3D.addVariableAndFilename(variable, covFile3Ddata, SCALARVARIABLE)
        else:
            os.remove(covFile3Ddata)

    # 2D
    item2D = None
    if os.access(covFile2Dgrid, os.F_OK):
        if (os.path.getsize(covFile2Dgrid) >= MIN_FILESIZE):
            # valid file
            item2D = CoviseCaseFileItem(name, GEOMETRY_2D, covFile2Dgrid)
            cocase.add(item2D)
        else:
            os.remove(covFile2Dgrid)
    if os.access(covFile2Ddata, os.F_OK):
        if bofname and (os.path.getsize(covFile2Ddata) >= MIN_FILESIZE):
            # valid file
            if item2D:
                item2D.addVariableAndFilename(variable, covFile2Ddata, SCALARVARIABLE)
        else:
            os.remove(covFile2Ddata)

    logFile.write("\nDONE\n")
    logFile.flush()
    print("DONE\n")




# FINISH

try:
    import cPickle as pickle
except:
    import pickle

pickleFile = coviseDatenDir + '/' + cocasename + '.cocase'
output = open(pickleFile, 'wb')
pickle.dump(cocase,output)
output.close()

logFile.write("\ncocasefile = %s\n"%(pickleFile,))
logFile.flush()
print("cocase file = %s\n"%(pickleFile,))

CoviseMsgLoop().unregister(aErrorLogAction)

logFile.write("\nConversion finished\n")
logFile.flush()
logFile.close()
print("Conversion finished see log file %s\n"%(logFileName,))

sys.exit()
