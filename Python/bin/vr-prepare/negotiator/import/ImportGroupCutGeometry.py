
# Part of the vr-prepare 

# Copyright (c) 2008 Visenso GmbH


""" +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ """
"""                                                                                             """
"""      simplify the content of the ImportGroup with AssembleUsg                               """
"""                                                                                             """
""" +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ """


from ImportGroupSimpleFilter import ImportGroupSimpleFilter
from VRPCoviseNetAccess import CutGeometryModule


class ImportGroupCutGeometry(ImportGroupSimpleFilter):
    def __init__(self, iGroupModule):
        ImportGroupSimpleFilter.__init__(self, iGroupModule, CutGeometryModule)
        self.__normal = (1.,0.,0.)
        self.__distance = 0. 
        
    def setNormal( self, nx, ny, nz):
        self.__normal = ( nx, ny, nz)

        self.setNeedExecute(True)

    def setDistance( self, d ):
        self.__distance = d

        self.setNeedExecute(True)
        
    def _update(self, varname=None):
        if varname==None:
            self._filterGeo.setPlane(self.__normal[0], self.__normal[1], self.__normal[2], self.__distance)
        else:
            self._filterVar[varname].setPlane(self.__normal[0], self.__normal[1], self.__normal[2], self.__distance)

    def getBox(self, execute = False):
        """ custom getBox() needed to have a AABB greater than what is visible """
        return self._importGroupModule.getBoxFromGeoRWCovise()

    def getBoxFromGeoRWCovise(self):
        return self._importGroupModule.getBoxFromGeoRWCovise()
