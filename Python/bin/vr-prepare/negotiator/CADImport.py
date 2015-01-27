
# Part of the vr-prepare program for dc

# Copyright (c) 2007 Visenso GmbH

from VRPCoviseNetAccess import saveExecute, connect, disconnect, theNet, ConnectionPoint
from coPyModules import ReadCAD, Collect

""" ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ """
"""                                                                """
"""    import a cad part                                           """
"""                                                                """
""" ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ """

class ImportCADModule(object):

    def __init__(self, filename, index=0 ):
        self.__filename = filename
        self.__index = index
        self.__readCad = None
        self.__collect = None

    def __init(self):
        """ start COVISE ReadCAD Module """
        if self.__readCad==None :
            self.__readCad = ReadCAD()
            theNet().add(self.__readCad)
            self.__readCadPolyOut = ConnectionPoint( self.__readCad, 'mesh' )
            self.__readCadNormalsOut = ConnectionPoint( self.__readCad, 'Normals' )
            self.__readCadColorsOut = ConnectionPoint( self.__readCad, 'colorOut' )
            self.__readCad.set_catia_server('obie')
            self.__readCad.set_catia_server_port('7000')
            self.__readCad.set_partIndex(self.__index)
            self.__readCad.set_file_path( self.__filename )

            self.__collect = Collect()
            theNet().add(self.__collect)
            connect( self.__readCadPolyOut, ConnectionPoint(self.__collect, 'GridIn0') )
            connect( self.__readCadNormalsOut, ConnectionPoint(self.__collect, 'DataIn1' ) )
            connect( self.__readCadColorsOut, ConnectionPoint(self.__collect, 'DataIn0' ) )
            self.__connectGeoOut =  ConnectionPoint(self.__collect, 'GeometryOut0' )

    def __update(self, featureAngle, max_Dev_mm, max_Size_mm ):
        self.__init()
        self.__readCad.set_FeatureAngle( featureAngle-1, featureAngle-1, featureAngle )
        self.__readCad.set_Max_Dev_mm( max_Dev_mm-1, max_Dev_mm+1, max_Dev_mm )
        self.__readCad.set_Max_Size_mm( max_Size_mm-1, max_Size_mm+1,max_Size_mm )

    def setTesselationParams( self, featureAngle, max_Dev_mm, max_Size_mm ):
        self.__update( featureAngle, max_Dev_mm, max_Size_mm )

    """ ------------------------ """
    """ connection points        """
    """ ------------------------ """

    def geoConnectionPoint(self):
        self.__init()
        return self.__connectGeoOut

    def normalsConnectionPoint(self):
        self.__init()
        return self.__readCadNormalsOut

    def colorsConnectionPoint(self):
        self.__init()
        return self.__readCadColorsOut

    def execute(self):
        self.__init()
        saveExecute(self.__readCad)

    def executeGeo(self):
        self.execute()

    def getCoObjName(self):
        if not self.__readCad==None:
            return self.__readCad.getCoObjName('mesh')
