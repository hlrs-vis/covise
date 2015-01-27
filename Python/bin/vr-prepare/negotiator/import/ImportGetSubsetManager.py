
# Part of the vr-prepare 

# Copyright (c) 2008 Visenso GmbH


""" +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ """
"""                                                                                             """
""" reduces the num of used set elements of the ImportModule with ReduceSet                     """
"""                                                                                             """
""" +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ """


from ImportGroupSimpleFilter import ImportGroupSimpleFilter
from ImportGroupFilter import ImportGroupFilter
from VRPCoviseNetAccess import connect, disconnect, saveExecute, GetSubsetModule, theNet

from ImportManager import ImportModule

import traceback


################################################################################
# ImportGetSubsetModule                                                        #
################################################################################

ParentClass = ImportModule

class ImportGetSubsetModule(ParentClass):
    def __init__(self, dimension, partcase):
        ParentClass.__init__(self, dimension, partcase)

        self.__needExecuteGeo = False
        self.__needExecuteData = {}
        self.__geoConnectionPoint = None
        self.__geoInConnectionPoint = None
        self.__dataConnectionPoints = {}
        self.__dataInConnectionPoints = {}

        self.__getSubsetModuleData = {}  # the GetSubset modules for variables
        self.__getSubsetModuleGeo = None

        self.__selectionString = None   # should take a correct default?
        self.__reductionFactor = 1      # for compatibility to former ImportReduceSetManager


    def geoConnectionPoint(self):
        self.__initGeo()

        if self.__geoConnectionPoint != None:
            return self.__geoConnectionPoint
        else:
            return ParentClass.geoConnectionPoint(self)

    def dataConnectionPoint(self, varName):
        self.__initData(varName)

        if varName in self.__dataConnectionPoints:
            return self.__dataConnectionPoints[varName]
        else:
            return ParentClass.dataConnectionPoint(self, varName)

    def executeGeo(self):
        self.__initGeo()

        # reconnect geometry
        if self.__geoInConnectionPoint != None:
            theNet().disconnectAllFromModulePort(self.__geoInConnectionPoint.module, self.__geoInConnectionPoint.port)
            connect( ParentClass.geoConnectionPoint(self), self.__geoInConnectionPoint )

        # wenn voriges Modul tatsaechlich executed wurde, dann nicht mehr selbst executen
        if ParentClass.executeGeo(self) == True:
            self.__needExecuteGeo = False
            return True

        if not self.__needExecuteGeo:
            return False
        self.__needExecuteGeo = False

        if self.__geoInConnectionPoint != None:
            saveExecute(self.__geoInConnectionPoint.module)
        return True

    def executeData(self, varName):
        self.__initData(varName)

        # reconnect data
        if varName in self.__dataConnectionPoints.keys():
            theNet().disconnectAllFromModulePort(self.__dataInConnectionPoints[varName].module, self.__dataInConnectionPoints[varName].port)
            theNet().connect(ParentClass.dataConnectionPoint(self, varName).module, ParentClass.dataConnectionPoint(self, varName).port,
                             self.__dataInConnectionPoints[varName].module, self.__dataInConnectionPoints[varName].port)

        # wenn voriges Modul tatsaechlich executed wurde, dann nicht mehr selbst executen
        if ParentClass.executeData(self, varName) == True:
            self.__needExecuteData[varName] = False
            return True

        if not self.__needExecuteData[varName]:
            return False
        self.__needExecuteData[varName] = False

        if varName in self.__dataInConnectionPoints.keys():
            saveExecute(self.__dataInConnectionPoints[varName].module)
        return True

    def __initGeo(self):
        if (self.__selectionString == None or self.__selectionString == "") and (self.__getSubsetModuleGeo == None):
            return

        if self.__geoInConnectionPoint == None:
            if self.__getSubsetModuleGeo == None:
                self.__getSubsetModuleGeo = GetSubsetModule()

            self.__getSubsetModuleGeo.setSelectionString(self.__selectionString)

            self.__geoConnectionPoint = self.__getSubsetModuleGeo.geoOutConnectionPoint()
            self.__geoInConnectionPoint = self.__getSubsetModuleGeo.geoInConnectionPoint()

            self.__needExecuteGeo = True

    def __initData(self, varName):
        if not varName in self.__needExecuteData.keys():
            self.__needExecuteData[varName] = False

        if (self.__selectionString == None or self.__selectionString == "") and (varName not in self.__getSubsetModuleData.keys()):
            return

        if not varName in self.__getSubsetModuleData.keys():

            self.__getSubsetModuleData[varName] = GetSubsetModule()
            self.__getSubsetModuleData[varName].setSelectionString(self.__selectionString)

            self.__dataConnectionPoints[varName] = self.__getSubsetModuleData[varName].geoOutConnectionPoint()
            self.__dataInConnectionPoints[varName] = self.__getSubsetModuleData[varName].geoInConnectionPoint()

            self.__needExecuteData[varName] = True

    def delete(self):
        if hasattr(self, "_ImportGetSubsetModule__getSubsetModuleGeo") and self.__getSubsetModuleGeo != None:
            self.__getSubsetModuleGeo.remove()
        if hasattr(self, "_ImportGetSubsetModule__getSubsetModuleData"):
            for module in self.__getSubsetModuleData.values(): module.remove()
        ParentClass.delete(self)

    #############################
    # filter specific functions #
    #############################

    # for compatibility to former ImportReduceSetManager

    def setReductionFactor(self, rf, execute=False):
        if rf == 1 and self.__getSubsetModuleGeo == None:
            return

        if rf < 0:
            self.__reductionFactor = (self.getNumTimeSteps() + 1) / 2
        else:
            self.__reductionFactor = rf

        selectionString = ""
        for i in range(self.getNumTimeSteps()):
            if i % self.__reductionFactor == 0:
                selectionString = selectionString + str(i) + " "
        return self.setSelectionString(selectionString, execute)

    def getNumReductionModules(self):
        return len(self.__getSubsetModuleData)

    def getReductionFactor(self):
        return self.__reductionFactor


    def setSelectionString(self, selection, execute=False):
        if self.__selectionString != selection:
            self.__selectionString = selection
            self.__needExecuteGeo = True
            for key in self.__needExecuteData.keys():
                self.__needExecuteData[key] = True

        if self.__getSubsetModuleGeo != None:
            self.__getSubsetModuleGeo.setSelectionString(self.__selectionString)
        for gS in self.__getSubsetModuleData.values():
            gS.setSelectionString(self.__selectionString)

        if execute:
            self.__getSubsetModuleGeo.execute()
            for gS in self.__getSubsetModuleData.values():
                gS.execute()
            self.__needExecuteGeo = False
            for key in self.__needExecuteData.keys():
                self.__needExecuteData[key] = False
            return True
        return False

    def getNumGetSubsetModules(self):
        return len(self.__getSubsetModules)

    def getSelectionString(self):
        return self.__selectionString


################################################################################
# ImportGetSubset3DModule                                                      #
################################################################################

class ImportGetSubset3DModule( ImportGetSubsetModule ):

    def __init__(self, partcase ):
        ImportGetSubsetModule.__init__(self, 3, partcase )

################################################################################
# ImportReduceSet2DModule                                                      #
################################################################################

class ImportGetSubset2DModule( ImportGetSubsetModule ):

    def __init__(self, partcase ):
        ImportGetSubsetModule.__init__(self, 2, partcase )
