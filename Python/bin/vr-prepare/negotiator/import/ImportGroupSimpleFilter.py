
# Part of the vr-prepare 

# Copyright (c) 2008 Visenso GmbH

# definition of class ImportGroupSimpleFilter

""" ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ """
"""                                                                              """
"""      general simple filter class for ImportGroupModules                      """
"""                                                                              """
""" ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ """


# This class only calls the functions of the class ImportGroupModule.
# Interface has to be adapted to every change in class ImportGroupModule

from ImportGroupFilter import ImportGroupFilter
from VRPCoviseNetAccess import theNet,saveExecute, connect, disconnect, ConnectionPoint
from BoundingBox import BoundingBoxParser, Box
import coviseStartup

from coPyModules import BoundingBox, Colors

# this class depends on a module class which supports the functions geoInConnectionPoint, geoOutConnectionPoint,
# dataInConnectionPoint, dataOutConnectionPoint and execute

class ImportGroupSimpleFilter(ImportGroupFilter):
    def __init__(self, iGroupModule, covModuleName):
        ImportGroupFilter.__init__(self,iGroupModule)
        self.__filterName = covModuleName       
        # one filter per variable
        # series of filter modules. Referenced by variable name
        self._filterVar = {}
        self._filterGeo = None

        self.__needExecuteData = {}
        self.__needExecuteGeo = False

        self._bb = None
        self._minMax = None

    def __initFilterGeo(self):
        if self._filterGeo==None:
            self._filterGeo = self.__filterName()
            self.__needExecuteGeo = True

    def __initFilterVar(self, varname):
        if not varname in self._filterVar:
            self._filterVar[varname] = self.__filterName()
            self.__needExecuteData[varname] = True

    def __initFilter( self, varname=None):
        if varname==None:
            return self.__initFilterGeo()
        else:
            return self.__initFilterVar(varname)

    def _update(self, varname=None):
        # update parameter of filter. To be overwritten by child class
        return

    def delete(self):
        if hasattr(self, "_filterGeo") and self._filterGeo: self._filterGeo.remove()
        if hasattr(self, "_filterVar"):
            for module in self._filterVar.values(): module.remove()
        if hasattr(self, "_bb") and self._bb: theNet().remove(self._bb)
        if hasattr(self, "_minMax") and self._minMax: theNet().remove(self._minMax)

    def geoConnectionPoint(self):
        if self._filterGeo==None:
            self.__initFilter()
            connect( self._importGroupModule.geoConnectionPoint(), self._filterGeo.geoInConnectionPoint() )
        return  self._filterGeo.geoOutConnectionPoint()

    def dataConnectionPoint(self, varname):
        if not varname in self._filterVar:
            self.__initFilter(varname)
            connect( self._importGroupModule.geoConnectionPoint(), self._filterVar[varname].geoInConnectionPoint() )
            connect( self._importGroupModule.dataConnectionPoint(varname), self._filterVar[varname].dataInConnectionPoint() )
        return  self._filterVar[varname].dataOutConnectionPoint()

    def executeGeo(self):
        self.geoConnectionPoint()
        self._update()
        # TODO make clean

        if ImportGroupFilter.executeGeo(self):
            return True

        if self.__needExecuteGeo:
            self._filterGeo.execute()
            self.__needExecuteGeo = False
            return True
        return False
            

    def executeData(self, varname):
        self.dataConnectionPoint(varname)
        self._update(varname)
        # TODO make clean
        #ImportGroupFilter.executeGeo(self)

        if ImportGroupFilter.executeData(self, varname):
            return True

        if self.__needExecuteData[varname]:
            self._filterVar[varname].execute()
            self.__needExecuteData[varname] = False
            return True
        return False

    def getBox(self, forceExecute = False):
        """ return the bounding box """
        if self._bb==None:
            self._bb = BoundingBox()
            theNet().add( self._bb)
            connect( self.geoConnectionPoint(), ConnectionPoint( self._bb, 'GridIn0' ) )
            # Clear info queue so we dont read a previous BB output.
            # (If something goes wrong with the queue, this could be the reason.)
            coviseStartup.globalReceiverThread.infoQueue.clear()
            if not self.executeGeo():
                saveExecute(self._bb)
            boxParser = BoundingBoxParser()
            boxParser.parseQueue(coviseStartup.globalReceiverThread.infoQueue)
            self._boundingBox = boxParser.getBox()
        elif forceExecute:
            # Clear info queue so we dont read a previous BB output.
            # (If something goes wrong with the queue, this could be the reason.)
            coviseStartup.globalReceiverThread.infoQueue.clear()
            if not self.executeGeo():
                saveExecute(self._bb)
            boxParser = BoundingBoxParser()
            boxParser.parseQueue(coviseStartup.globalReceiverThread.infoQueue)
            try:
                oldbb = self._boundingBox
                self._boundingBox = boxParser.getBox()
            except (ValueError):
                self._boundingBox = oldbb
        return self._boundingBox

    def getDataMinMax(self, variable):
        """ return min and max value of variable """
        if variable==None:
            return

        if self._minMax==None:
            self._minMax = Colors()
            theNet().add(self._minMax)
        theNet().disconnectAllFromModulePort( self._minMax, 'DataIn0' )
        connect(self.dataConnectionPoint(variable), ConnectionPoint(self._minMax, 'DataIn0'))

        self.executeData( variable )
        saveExecute(self._minMax)

        return ( float(self._minMax.getParamValue('MinMax')[0]),\
                 float(self._minMax.getParamValue('MinMax')[1]) )

    def setNeedExecute(self, b = True):
        self.__needExecuteGeo = b
        for key in self.__needExecuteData.keys():
            self.__needExecuteData[key] = b
