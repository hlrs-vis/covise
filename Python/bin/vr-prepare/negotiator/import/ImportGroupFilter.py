
# Part of the vr-prepare 

# Copyright (c) 2008 Visenso GmbH

# definition of class ImportGroupFilter

""" +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ """
"""                                                                                             """
"""      base class for all filter classes that have an importGroupModule as input and output   """
"""                                                                                             """
""" +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ """


# This class only calls the functions of the class ImportGroupModule.
# Interface has to be adapted to every change in class ImportGroupModule


class ImportGroupFilter(object):
    def __init__(self, iGroupModule):
        self._importGroupModule = iGroupModule

    """ ------------------------ """
    """ connection points        """
    """ ------------------------ """

    def geoConnectionPoint(self):
        return self._importGroupModule.geoConnectionPoint()

    def dataConnectionPoint( self, vname ):
        return self._importGroupModule.dataConnectionPoint(vname)

    def octTreeConnectionPoint( self ):        
        return self._importGroupModule.octTreeConnectionPoint()

    def geoSampleConnectionPoint(self, varname, outside=2):
        return self._importGroupModule.geoSampleConnectionPoint(varname, outside)
        
    def dataSampleConnectionPoint(self, varname, outside=2):
        return self._importGroupModule.dataSampleConnectionPoint(varname, outside)

    """ ------------------------ """
    """ execution methods        """
    """ ------------------------ """

    def executeGeo(self):
        return  self._importGroupModule.executeGeo()

    def executeData(self, vname):
        return  self._importGroupModule.executeData(vname)

    def executeSampleData(self, vname, bbox=None, outside=2 ):
        return  self._importGroupModule.executeSampleData(vname, bbox, outside )

    def executeOct(self):
        return  self._importGroupModule.executeOct()

    def reloadGeo(self):
        return  self._importGroupModule.reloadGeo()

    """ ------------------------ """
    """ read private variables   """
    """ ------------------------ """

    def getDimension(self):
        return  self._importGroupModule.getDimension()

    def getParts(self):
        return  self._importGroupModule.getParts()

    def getIsTransient(self):
        return self._importGroupModule.getIsTransient()

    def getCoObjName(self):
        return  self._importGroupModule.getCoObjName()

    def getBox(self, forceExecute=False):
        return self._importGroupModule.getBox(forceExecute)

    def getDataMinMax( self, variable):
        return  self._importGroupModule.getDataMinMax(variable)



    def setRotation(self, angle, x,y,z , execute=True):
        return

    def setTranslation(self, x,y,z, execute=True):
        return

#