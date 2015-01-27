import re
import sys
import os
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

try:
    import cPickle as pickle
except:
    import pickle


infoer = InfoPrintCapable()
infoer.doPrint = False # True  Toggle talkativeness of this script.

cocase = CoviseCaseFile()
if coCaseFile != "None":
    inputFile = open(coCaseFile, 'rb')
    cocase = pickle.load(inputFile)
    inputFile.close()


logFileName = cocasename + ".log"
logFile = open(logFileName, 'w')
logFile.write("Options:\n")
logFile.write("Covise Case Name = %s\n"%(cocasename,))
logFile.write("Covise Data Directory = %s\n"%(coviseDatenDir,))
logFile.write("scale = %s\n"%(scale,))
logFile.write("\n")
logFile.flush()
print("Options:")
print("Covise Case Name = ", cocasename)
print("Covise Data Directory = ", coviseDatenDir)
print("scale = ", scale)
print("noGrid = ", noGrid)
print("Composed grid = ", composedGrid)
print("Calculate PDYN = ", calculatePDYN)
print("coCaseFile = ", coCaseFile)
print("readBoundaries = ", readBoundaries)
print("transient = ", readTransient)
print(" ")


aErrorLogAction = ErrorLogAction()
CoviseMsgLoop().register(aErrorLogAction)

#
# create global net
#
theNet = net()
