
# Part of the vr-prepare 

# Copyright (c) 2008 Visenso GmbH


""" +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ """
"""                                                                                             """
""" reduces the num of used set elements of the ImportModule with ReduceSet                     """
"""                                                                                             """
""" +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ """


from ImportGroupSimpleFilter import ImportGroupSimpleFilter
from ImportGroupFilter import ImportGroupFilter
from VRPCoviseNetAccess import connect, disconnect, ReduceSetModule, theNet

from ImportManager import ImportModule

import traceback


################################################################################
# ImportReduceSetModule                                                        #
################################################################################

ParentClass = ImportModule

class ImportReduceSetModule(ParentClass):
    def __init__(self, dimension, partcase):
        ParentClass.__init__(self, dimension, partcase)

        self.__needExecute = False
        self.__dataConnectionPoints = {}

        self.__reduceSetModules = []  # the ReduceSet modules

        self.__reductionFactor = 1


    def geoConnectionPoint(self):
        if self.__reduceSetModules != []:
            return self.__reduceSetModules[0].geoOutConnectionPoint()
        else:
            return ParentClass.geoConnectionPoint(self)

    def dataConnectionPoint(self, varName):
        if varName in self.__dataConnectionPoints:
            return self.__dataConnectionPoints[varName]
        else:
            return ParentClass.dataConnectionPoint(self, varName)

    def executeGeo(self):
        # reconnect geometry
        if self.__reduceSetModules != []:
            theNet().disconnectAllFromModulePort(self.__reduceSetModules[0].geoInConnectionPoint().module, self.__reduceSetModules[0].geoInConnectionPoint().port)
            connect( ParentClass.geoConnectionPoint(self), self.__reduceSetModules[0].geoInConnectionPoint() )

        # wenn voriges Modul tatsaechlich executed wurde, dann nicht mehr selbst executen
        if ParentClass.executeGeo(self) == True:
            return True

        if not self.__needExecute:
            return False
        self.__needExecute = False

        if self.__reduceSetModules != []:
            self.__reduceSetModules[0].execute()
        return True

    def executeData(self, varName):
        # reconnect data
        if varName in self.__dataConnectionPoints.keys():
            theNet().disconnectAllFromModulePort(self.__dataConnectionPoints[varName].module, self.__dataConnectionPoints[varName].port.replace("out", "in"))
            theNet().connect(ParentClass.dataConnectionPoint(self, varName).module, ParentClass.dataConnectionPoint(self, varName).port,
                             self.__dataConnectionPoints[varName].module, self.__dataConnectionPoints[varName].port.replace("out", "in"))

        # wenn voriges Modul tatsaechlich executed wurde, dann nicht mehr selbst executen
        if ParentClass.executeData(self, varName) == True:
            return True

        if not self.__needExecute:
            return False
        self.__needExecute = False

        for rS in self.__reduceSetModules:
            rS.execute()
        return True


    #############################
    # filter specific functions #
    #############################

    def setReductionFactor(self, rf, execute=True):
        self.__reductionFactor = rf

        if rf == 1 and self.__reduceSetModules == []:
            return
            
        if rf < 0:
            timesteps = self.getNumTimeSteps()
            self.__reductionFactor = timesteps / 2

        self.__needExecute = True

        # create new modules if necessary
        if self.__reduceSetModules == []:
            numNeededReduceSetModules = 1
            numConnectedVariables = 1   # 1, so data variable connecting will start at port 'input_1' ('input_0' is geometry)

            # create initial module
            self.__reduceSetModules.append(ReduceSetModule())

            # connect geometry
            connect( ParentClass.geoConnectionPoint(self), self.__reduceSetModules[0].geoInConnectionPoint() )

            # connect data
            for name in self._dataFileNames:

                # TODO: should be made unnecessary
                self.executeData(name)
                
                portNum = numConnectedVariables % 8
                if len(self.__reduceSetModules) < numNeededReduceSetModules:
                    self.__reduceSetModules.append(ReduceSetModule())

#                theNet().disconnectAllFromModulePort(ParentClass.dataConnectionPoint(self, name).module, ParentClass.dataConnectionPoint(self, name).port)
                if portNum == 0:
                    self.__dataConnectionPoints[name] = self.__reduceSetModules[numNeededReduceSetModules-1].data0OutConnectionPoint()
                    connect(ParentClass.dataConnectionPoint(self, name), self.__reduceSetModules[numNeededReduceSetModules-1].data0InConnectionPoint())
                elif portNum == 1:
                    self.__dataConnectionPoints[name] = self.__reduceSetModules[numNeededReduceSetModules-1].data1OutConnectionPoint()
                    connect(ParentClass.dataConnectionPoint(self, name), self.__reduceSetModules[numNeededReduceSetModules-1].data1InConnectionPoint())
                elif portNum == 2:
                    self.__dataConnectionPoints[name] = self.__reduceSetModules[numNeededReduceSetModules-1].data2OutConnectionPoint()
                    connect(ParentClass.dataConnectionPoint(self, name), self.__reduceSetModules[numNeededReduceSetModules-1].data2InConnectionPoint())
                elif portNum == 3:
                    self.__dataConnectionPoints[name] = self.__reduceSetModules[numNeededReduceSetModules-1].data3OutConnectionPoint()
                    connect(ParentClass.dataConnectionPoint(self, name), self.__reduceSetModules[numNeededReduceSetModules-1].data3InConnectionPoint())
                elif portNum == 4:
                    self.__dataConnectionPoints[name] = self.__reduceSetModules[numNeededReduceSetModules-1].data4OutConnectionPoint()
                    connect(ParentClass.dataConnectionPoint(self, name), self.__reduceSetModules[numNeededReduceSetModules-1].data4InConnectionPoint())
                elif portNum == 5:
                    self.__dataConnectionPoints[name] = self.__reduceSetModules[numNeededReduceSetModules-1].data5OutConnectionPoint()
                    connect(ParentClass.dataConnectionPoint(self, name), self.__reduceSetModules[numNeededReduceSetModules-1].data5InConnectionPoint())
                elif portNum == 6:
                    self.__dataConnectionPoints[name] = self.__reduceSetModules[numNeededReduceSetModules-1].data6OutConnectionPoint()
                    connect(ParentClass.dataConnectionPoint(self, name), self.__reduceSetModules[numNeededReduceSetModules-1].data6InConnectionPoint())
                elif portNum == 7:
                    self.__dataConnectionPoints[name] = self.__reduceSetModules[numNeededReduceSetModules-1].data7OutConnectionPoint()
                    connect(ParentClass.dataConnectionPoint(self, name), self.__reduceSetModules[numNeededReduceSetModules-1].data7InConnectionPoint())

                numConnectedVariables = numConnectedVariables + 1
                if numConnectedVariables % 8 == 0:
                    numNeededReduceSetModules = numNeededReduceSetModules + 1
            
        for rS in self.__reduceSetModules:
            rS.setReductionFactor(self.__reductionFactor)

        if execute:
            for rS in self.__reduceSetModules:
                rS.execute()
            self.__needExecute = False
            return True
        return False
        
    def getNumReductionModules(self):
        return len(self.__reduceSetModules)
        
    def getReductionFactor(self):
        return self.__reductionFactor


################################################################################
# ImportReduceSet3DModule                                                      #
################################################################################

class ImportReduceSet3DModule( ImportReduceSetModule ):

    def __init__(self, partcase ):
        ImportReduceSetModule.__init__(self, 3, partcase )

################################################################################
# ImportReduceSet2DModule                                                      #
################################################################################

class ImportReduceSet2DModule( ImportReduceSetModule ):

    def __init__(self, partcase ):
        ImportReduceSetModule.__init__(self, 2, partcase )
