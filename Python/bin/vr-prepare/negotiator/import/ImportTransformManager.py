
# Part of the vr-prepare 

# Copyright (c) 2008 Visenso GmbH


""" +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ """
"""                                                                                             """
"""      transform the content of the ImportGroup with Transform                                """
"""                                                                                             """
""" +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ """


from ImportGroupSimpleFilter import ImportGroupSimpleFilter
from ImportGroupFilter import ImportGroupFilter
from VRPCoviseNetAccess import connect, disconnect, TransformModule, theNet, saveExecute
from coviseCase import SCALARVARIABLE

from ImportCropUsgManager import ImportCropUsgModule

import traceback


################################################################################
# ImportTransformModule                                                        #
################################################################################

ParentClass = ImportCropUsgModule

class ImportTransformModule(ParentClass):
    def __init__(self, dimension, partcase):
        ParentClass.__init__(self, dimension, partcase)

        self.__needExecuteGeo = False
        self.__needExecuteData = {}
        self.__dataConnectionPoints = {}
        self.__dataInConnectionPoints = {}
        self.__geoConnectionPoint = None
        self.__geoInConnectionPoint = None

        self.__numNeededRotationModules = 1
        self.__numConnectedVectorVariables = 0
        self.__rotations = []
        self.__translation = None

        self.__rotAngle = 0
        self.__rotX = 1
        self.__rotY = 1
        self.__rotZ = 1

        self.__transX = 0
        self.__transY = 0
        self.__transZ = 0


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
            connect(ParentClass.geoConnectionPoint(self), self.__geoInConnectionPoint)

        # wenn voriges Modul tatsaechlich executed wurde, dann nicht mehr selbst executen
        if ParentClass.executeGeo(self) == True:
            self.__needExecuteGeo = False
            return True

        if not self.__needExecuteGeo:
            return False
        self.__needExecuteGeo = False

        if self.__geoInConnectionPoint != None:
            saveExecute(self.__geoInConnectionPoint.module)
        return False

    def executeData(self, varName):
        self.__initData(varName)

        # reconnect data
        if varName in self.__dataInConnectionPoints.keys():
            theNet().disconnectAllFromModulePort(self.__dataInConnectionPoints[varName].module, self.__dataInConnectionPoints[varName].port)
            connect(ParentClass.dataConnectionPoint(self, varName), self.__dataInConnectionPoints[varName])

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

        return False

    def __initGeo(self):
        # init rotation
        if self.__rotAngle == 0 and self.__rotations == []:
            pass
        else:
            if self.__rotAngle != 0 and self.__rotations == []:
                self.__rotations.append(TransformModule())
                self.__rotations[0].setRotation(self.__rotAngle, self.__rotX, self.__rotY, self.__rotZ)
                self.__needExecuteGeo = True

                self.__geoInConnectionPoint = self.__rotations[0].geoInConnectionPoint()
                if self.__translation == None:
                    self.__geoConnectionPoint = self.__rotations[0].geoOutConnectionPoint()
                else:
                    theNet().disconnectAllFromModulePort(self.__translation.geoInConnectionPoint().module, self.__translation.geoInConnectionPoint().port)
                    connect(self.__rotations[0].geoOutConnectionPoint(), self.__translation.geoInConnectionPoint())

        # init translation
        if (self.__transX, self.__transY, self.__transZ) == (0,0,0) and self.__translation == None:
            pass
        else:
            if (self.__transX, self.__transY, self.__transZ) != (0,0,0) and self.__translation == None:
                self.__translation = TransformModule()
                self.__translation.setTranslation(self.__transX, self.__transY, self.__transZ)
                self.__needExecuteGeo = True

                if self.__rotations == []:
                    self.__geoInConnectionPoint = self.__translation.geoInConnectionPoint()
                else:
                    theNet().disconnectAllFromModulePort(self.__rotations[0].geoOutConnectionPoint().module, self.__rotations[0].geoOutConnectionPoint().port)
                    connect(self.__rotations[0].geoOutConnectionPoint(), self.__translation.geoInConnectionPoint())
                self.__geoConnectionPoint = self.__translation.geoOutConnectionPoint()

    def __initData(self, varName):
        if not varName in self.__needExecuteData.keys():
            self.__needExecuteData[varName] = False

        if self.__rotAngle == 0 and self.__rotations == []:
            pass
        else:
            if (self._dataVariableType[varName] != SCALARVARIABLE) and (not varName in self.__dataInConnectionPoints.keys()) and (self.__rotAngle != 0):

                portNum = self.__numConnectedVectorVariables % 4
                if len(self.__rotations) < self.__numNeededRotationModules:
                    self.__rotations.append(TransformModule())
                    self.__rotations[self.__numNeededRotationModules-1].setRotation(self.__rotAngle, self.__rotX, self.__rotY, self.__rotZ)
                    self.__needExecuteData[varName] = True
                    # verbinde notwendige geometrie
                    connect(ParentClass.geoConnectionPoint(self), self.__rotations[self.__numNeededRotationModules-1].geoInConnectionPoint())

                if portNum == 0:
                    self.__dataConnectionPoints[varName] = self.__rotations[self.__numNeededRotationModules-1].data0OutConnectionPoint()
                    self.__dataInConnectionPoints[varName] = self.__rotations[self.__numNeededRotationModules-1].data0InConnectionPoint()
                elif portNum == 1:
                    self.__dataConnectionPoints[varName] = self.__rotations[self.__numNeededRotationModules-1].data1OutConnectionPoint()
                    self.__dataInConnectionPoints[varName] = self.__rotations[self.__numNeededRotationModules-1].data1InConnectionPoint()
                elif portNum == 2:
                    self.__dataConnectionPoints[varName] = self.__rotations[self.__numNeededRotationModules-1].data2OutConnectionPoint()
                    self.__dataInConnectionPoints[varName] = self.__rotations[self.__numNeededRotationModules-1].data2InConnectionPoint()
                elif portNum == 3:
                    self.__dataConnectionPoints[varName] = self.__rotations[self.__numNeededRotationModules-1].data3OutConnectionPoint()
                    self.__dataInConnectionPoints[varName] = self.__rotations[self.__numNeededRotationModules-1].data3InConnectionPoint()

                self.__numConnectedVectorVariables = self.__numConnectedVectorVariables + 1
                if self.__numConnectedVectorVariables % 4 == 0:
                    self.__numNeededRotationModules = self.__numNeededRotationModules + 1


                if self.__translation == None:
                    pass
                else:
                    theNet().disconnectAllFromModulePort(self.__translation.geoInConnectionPoint().module, self.__translation.geoInConnectionPoint().port)
                    connect(self.__rotations[0].geoOutConnectionPoint(), self.__translation.geoInConnectionPoint())

    def delete(self):
        if hasattr(self, "_ImportTransformModule__rotations"):
            for module in self.__rotations: module.remove()
        if hasattr(self, "_ImportTransformModule__translation") and self.__translation: self.__translation.remove()
        ParentClass.delete(self)

    #############################
    # filter specific functions #
    #############################

    def setRotation(self, angle, x,y,z):
        if (angle, x, y, z) != (self.__rotAngle, self.__rotX, self.__rotY, self.__rotZ):
            self.__rotX = x
            self.__rotY = y
            self.__rotZ = z
            self.__rotAngle = angle

            self.__needExecuteGeo = True
            for key in self.__needExecuteData.keys():
                self.__needExecuteData[key] = True

        for r in self.__rotations:
            r.setRotation(self.__rotAngle, self.__rotX, self.__rotY, self.__rotZ)

    def setTranslation(self, x,y,z):
        if (self.__transX, self.__transY, self.__transZ) != (x,y,z):
            self.__transX = x
            self.__transY = y
            self.__transZ = z

            self.__needExecuteGeo = True

        if self.__translation != None:
            self.__translation.setTranslation(self.__transX, self.__transY, self.__transZ)

    def hasTransformationModules(self):
        return (self.__translation != None) or (self.__rotations != [])

    def hasTranslationModule(self):
        return self.__translation != None

    def hasRotationModules(self):
        return self.__rotations != []


################################################################################
# ImportTransform3DModule                                                      #
################################################################################

class ImportTransform3DModule( ImportTransformModule ):

    def __init__(self, partcase ):
        ImportTransformModule.__init__(self, 3, partcase )

################################################################################
# ImportTransform2DModule                                                      #
################################################################################

class ImportTransform2DModule( ImportTransformModule ):

    def __init__(self, partcase ):
        ImportTransformModule.__init__(self, 2, partcase )
