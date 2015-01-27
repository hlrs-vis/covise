
# Part of the vr-prepare 

# Copyright (c) 2008 Visenso GmbH


""" +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ """
"""                                                                                             """
""" crops the import with CropUsg                                                               """
"""                                                                                             """
""" +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ """


from VRPCoviseNetAccess import connect, disconnect, CropUsgModule, theNet, saveExecute

from ImportGetSubsetManager import ImportGetSubsetModule
from vrpconstants import SCALARVARIABLE, VECTOR3DVARIABLE

import traceback


################################################################################
# ImportCropUsgModule                                                        #
################################################################################

ParentClass = ImportGetSubsetModule

class ImportCropUsgModule(ParentClass):
    def __init__(self, dimension, partcase):
        ParentClass.__init__(self, dimension, partcase)

        self.__needExecuteGeo = False
        self.__needExecuteData = {}
        self.__geoConnectionPoint = None
        self.__geoInConnectionPoint = None
        self.__dataConnectionPoints = {}
        self.__dataInConnectionPoints = {}

        self.__cropUsgModules = []  # the CropUsg modules

        self.__cropMin = [0, 0, 0]
        self.__cropMax = [0, 0, 0]
        self.__cropInvert = False


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

            # reconnect geometry port
            theNet().disconnectAllFromModulePort(self.__dataInConnectionPoints[varName].module, 'GridIn0')
            theNet().connect(ParentClass.geoConnectionPoint(self).module, ParentClass.geoConnectionPoint(self).port,
                             self.__dataInConnectionPoints[varName].module, 'GridIn0')


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
        if self.__cropMin == [0, 0, 0] and self.__cropMax == [0, 0, 0] and self.__cropUsgModules == []:
            return

        if self.__geoInConnectionPoint == None:
            newModule = CropUsgModule(3) # 3: would handle vector data, but data port is never connected
            newModule.setCropMin(self.__cropMin[0], self.__cropMin[1], self.__cropMin[2])
            newModule.setCropMax(self.__cropMax[0], self.__cropMax[1], self.__cropMax[2])
            self.__geoConnectionPoint = newModule.geoOutConnectionPoint()
            self.__geoInConnectionPoint = newModule.geoInConnectionPoint()
            self.__cropUsgModules.append(newModule)

            self.__needExecuteGeo = True

    def __initData(self, varName):
        if not varName in self.__needExecuteData.keys():
            self.__needExecuteData[varName] = False

        if self.__cropMin == [0, 0, 0] and self.__cropMax == [0, 0, 0] and self.__cropUsgModules == []:
            return

        if not varName in self.__dataInConnectionPoints.keys():
            # create module for cropped data (distinguish between scalar/vector data)
            dim = self._dataVariableType[varName]
            newModule = None
            if dim == SCALARVARIABLE:
                newModule = CropUsgModule(1)
            elif dim == VECTOR3DVARIABLE:
                newModule = CropUsgModule(3)
            else:
                assert False, "unknown dimensionality: self._dataVariableType[varName = %s ] = %s" % (varName, dim)
            self.__cropUsgModules.append(newModule)
            newModule.setCropMin(self.__cropMin[0], self.__cropMin[1], self.__cropMin[2])
            newModule.setCropMax(self.__cropMax[0], self.__cropMax[1], self.__cropMax[2])
            self.__dataConnectionPoints[varName] = newModule.dataOutConnectionPoint()
            self.__dataInConnectionPoints[varName] = newModule.dataInConnectionPoint()

            self.__needExecuteData[varName] = True

    def delete(self):
        if hasattr(self, "_ImportCropUsgModule__cropUsgModules"):
            for module in self.__cropUsgModules: module.remove()
        ParentClass.delete(self)

    #############################
    # filter specific functions #
    #############################

    def setCropMin(self, x, y, z):
        if self.__cropMin != [x, y, z]:
            self.__cropMin = [x, y, z]
            self.__needExecuteGeo = True
            for key in self.__needExecuteData.keys():
                self.__needExecuteData[key] = True

        for cU in self.__cropUsgModules:
            cU.setCropMin(self.__cropMin[0], self.__cropMin[1], self.__cropMin[2])

    def setCropMax(self, x, y, z):
        if self.__cropMax != [x, y, z]:
            self.__cropMax = [x, y, z]
            self.__needExecuteGeo = True
            for key in self.__needExecuteData.keys():
                self.__needExecuteData[key] = True

        for cU in self.__cropUsgModules:
            cU.setCropMax(self.__cropMax[0], self.__cropMax[1], self.__cropMax[2])


################################################################################
# ImportReduceSet3DModule                                                      #
################################################################################

class ImportCropUsg3DModule( ImportCropUsgModule ):

    def __init__(self, partcase ):
        ImportCropUsgModule.__init__(self, 3, partcase )

################################################################################
# ImportReduceSet2DModule                                                      #
################################################################################

class ImportCropUsg2DModule( ImportCropUsgModule ):

    def __init__(self, partcase ):
        ImportCropUsgModule.__init__(self, 2, partcase )
