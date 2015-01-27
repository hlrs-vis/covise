import re
import sys
from time import sleep
from printing import InfoPrintCapable
from coviseCase import (
    CoviseCaseFileItem,
    CoviseCaseFile,
    GEOMETRY_2D,
    GEOMETRY_3D,
    SCALARVARIABLE,
    VECTOR3DVARIABLE)
from ErrorLogAction import ErrorLogAction
from CoviseMsgLoop import CoviseMsgLoop, CoviseMsgLoopAction

infoer = InfoPrintCapable()
infoer.doPrint = False # True  Toggle talkativeness of this script.

cocase = CoviseCaseFile()

logFileName = cocasename + ".log"
logFile = open(logFileName, 'w')
logFile.write("Options:\n")
logFile.write("Covise Case Name = %s\n"%(cocasename,))
logFile.write("Covise Data Directory = %s\n"%(coviseDatenDir,))
logFile.write("scaleZ = %s\n"%(scaleZ,))
logFile.write("waterSurfaceOffset = %s\n"%(waterSurfaceOffset,))
logFile.write("\n")
logFile.flush()
print("Options:")
print("Covise Case Name = ", cocasename)
print("Covise Data Directory = ", coviseDatenDir)
print("format = ", format)
print("scaleZ = ", scaleZ)
print("waterSurfaceOffset = ", waterSurfaceOffset)
print(" ")


aErrorLogAction = ErrorLogAction()
CoviseMsgLoop().register(aErrorLogAction)

#
# formats=
#

#
# create global net
#
theNet = net()

