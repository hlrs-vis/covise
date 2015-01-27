
# Part of the vr-prepare 

# Copyright (c) 2008 Visenso GmbH


""" +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ """
"""                                                                                             """
"""      transform the content of the ImportGroup with Transform                                """
"""                                                                                             """
""" +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ """


from ImportGroupSimpleFilter import ImportGroupSimpleFilter
from VRPCoviseNetAccess import connect, disconnect, TransformModule


class ImportGroupTransform(ImportGroupSimpleFilter):
    def __init__(self, iGroupModule):
        ImportGroupSimpleFilter.__init__(self, iGroupModule, TransformModule)

        self.__rotation = None
        self.__translation = None

        self.__rotAngle = 0.0
        self.__rotX = 1.0
        self.__rotY = 1.0
        self.__rotZ = 1.0

        self.__transX = 0.0
        self.__transY = 0.0
        self.__transZ = 0.0

    def execute(self):
        #self._initTrans()
        if self.__rotation != None:
            self.__rotation.execute()
        if self.__translation != None:
            self.__translation.execute()


    def setRotation(self, angle, x,y,z , execute=True):
        self.__rotX = x
        self.__rotY = y
        self.__rotZ = z
        self.__rotAngle = angle

        # create new module if necessary
        if self.__rotation == None:
            self.__rotation = TransformModule()
            connect( self.iGroupModule.geoConnectionPoint(), self.__rotation.geoInConnectionPoint() )
            if self.__translation == None:
                pass
            else:
                disconnect(self.iGroupModule.geoConnectionPoint(), self.__translation.geoInConnectionPoint())
                connect(self.__rotation.geoOutConnectionPoint(), self.__translation.geoInConnectionPoint())
                connect(self.__rotation.data0OutConnectionPoint(), self.__translation.data0InConnectionPoint())
                connect(self.__rotation.data1OutConnectionPoint(), self.__translation.data1InConnectionPoint())
                connect(self.__rotation.data2OutConnectionPoint(), self.__translation.data2InConnectionPoint())
                connect(self.__rotation.data3OutConnectionPoint(), self.__translation.data3InConnectionPoint())
        
        self.__rotation.setRotation(self.__rotAngle, self.__rotX, self.__rotY, self.__rotZ)
        if execute:
            self.__rotation.execute()
            return True
        return False

    def setTranslation(self, x,y,z, execute=True):
        self.__transX = x
        self.__transY = y
        self.__transZ = z

        # create new module if necessary
        if self.__translation == None:
            self.__translation = TransformModule()
            if self.__rotation == None:
                connect(self.iGroupModule.geoConnectionPoint(), self.__translation.geoInConnectionPoint())
            else:
                connect(self.__rotation.geoOutConnectionPoint(), self.__translation.geoInConnectionPoint())
                connect(self.__rotation.data0OutConnectionPoint(), self.__translation.data0InConnectionPoint())
                connect(self.__rotation.data1OutConnectionPoint(), self.__translation.data1InConnectionPoint())
                connect(self.__rotation.data2OutConnectionPoint(), self.__translation.data2InConnectionPoint())
                connect(self.__rotation.data3OutConnectionPoint(), self.__translation.data3InConnectionPoint())

        self.__translation.setTranslation(self.__transX, self.__transY, self.__transZ)
        if execute:
            r.execute()
            return True
        return False

