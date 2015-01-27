

import re
import sys

from time import sleep

from CoviseMsgLoop import CoviseMsgLoop, CoviseMsgLoopAction
from paramAction import NotifyAction
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


class PartDescriptionAnalyzer(object):

    def __init__(self, candidateString):
        # Regexp fitting for a 2d-part-description that comes with a covise-message.
        self.regexp2dPartDescription = r'(.*)\|(.*)\|(.*)\|(.*)\|.*(2D|d).*'
        self.regexp3dPartDescription = r'(.*)\|(.*)\|(.*)\|(.*)\|.*(3D|d).*'
        self.regexp2dOr3dPartDescription = r'(.*)\|(.*)\|(.*)\|(.*)\|.*(2|3D|d).*'
        self.candidateString = candidateString

    def is2dPartDescription(self):

        """Return if self.candidateString is a part-2d-description.

        For to know what a part-description is use the
        source here and in ReadEnsight until there is a
        better documentation.

        """

        return bool(
            re.match(
            self.regexp2dPartDescription, self.candidateString))

    def is3dPartDescription(self):

        """Return if self.candidateString is a part-3d-description.

        For to know what a part-description is use the
        source here and in ReadEnsight until there is a
        better documentation.

        """

        return bool(
            re.match(
            self.regexp3dPartDescription, self.candidateString))

    def pullOutPartIdAndName(self):

        """Return part-id and name from self.candidateString."""

        assert self.is2dPartDescription() or self.is3dPartDescription()
        matchie = re.match(
            self.regexp2dOr3dPartDescription, self.candidateString)
        coviseId = int(matchie.group(1).strip())
        ensightId = int(matchie.group(2).strip())
        name = matchie.group(3).strip()
        return coviseId, ensightId, name


class PartsCollectorAction(CoviseMsgLoopAction):

    """Action to collect part information when using
    covise with an ReadEnsight.

    Gets the information from covise-info-messages.

    """

    def __init__ (self):
        CoviseMsgLoopAction.__init__(
            self,
            self.__class__.__module__ + '.' + self.__class__.__name__,
            56, # Magic number 56 is covise-msg type INFO
            'Collect parts-names and numbers from covise-info-messages.')

        self.__partsinfoFinished = False

        self.__refAndNameDict2d = {}
        self.__refAndNameDict3d = {}


    def run(self, param):
        assert 4 == len(param)
        # assert param[0] is a modulename
        # assert param[1] is a number
        # assert param[2] is an ip
        # assert param[3] is a string

#       print(str(self.run))

        msgText = param[3]

        #print(str(msgText))
        analyzer = PartDescriptionAnalyzer(msgText)
        if analyzer.is2dPartDescription():
            coviseId, ensightId, name = analyzer.pullOutPartIdAndName()
            self.__refAndNameDict2d[ensightId] = name
            #logFile.write("Part: id = %d name = %s\n"%(partid, name))
            #logFile.flush()
            #print("CoviseId = %d\tEnsightId = %d\tName = %s"%(coviseId, ensightId, name))

        if analyzer.is3dPartDescription():
            coviseId, ensightId, name = analyzer.pullOutPartIdAndName()
            self.__refAndNameDict3d[ensightId] = name
            #logFile.write("Part: id = %d name = %s\n"%(coviseId, name))
            #logFile.flush()
            #print("CoviseId = %d\tEnsightId = %d\tName = %s"%(coviseId, ensightId, name))
            
        if msgText == "...Finished: List of Ensight Parts":
            #logFile.write("\n")
            #logFile.flush()
            self.__partsinfoFinished = True
#        else:
#            infoer.function = str(self.run)
#            infoer.write(
#                'Covise-message "%s" doesn\'t look like a parts-description.'
#                % str(msgText))

    def getRefNameDict2dParts(self):
        return self.__refAndNameDict2d

    def getRefNameDict3dParts(self):
        return self.__refAndNameDict3d

    def waitForPartsinfoFinished(self):
        #print("Ensight2CoviseBegin.py PartsCollectorAction.waitForPartsinfoFinished")
        while not self.__partsinfoFinished: 
            pass



######## START #########

cocase = CoviseCaseFile()

logFileName = cocasename + ".log"
logFile = open(logFileName, 'w')
logFile.write("Options:\n")
logFile.write("Covise Case Name = %s\n"%(cocasename,))
logFile.write("Covise Data Directory = %s\n"%(coviseDatenDir,))
logFile.write("scale = %s\n"%(scale,))
logFile.write("byteswap = %s\n"%(byteswap,))
logFile.write("startId = %s\n"%(startId,))
logFile.write("\n")
logFile.flush()
print("Options:")
print("Covise Case Name = ", cocasename)
print("Covise Data Directory = ", coviseDatenDir)
print("scale = ", scale)
print("byteswap = ", byteswap)
print("startId = ", startId)
print(" ")


aErrorLogAction = ErrorLogAction()
CoviseMsgLoop().register(aErrorLogAction)

#
# create global net
#
theNet = net()
