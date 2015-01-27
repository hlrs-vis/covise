# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH

import copy
from VRPCoviseNetAccess import ConnectionPoint, theNet

from VisItem import VisItem, VisItemParams
from PartModuleVis import PartModuleVis, PartModuleVisParams

from coPyModules import DomainSurface, Collect

from Utils import ParamsDiff, mergeGivenParams
from KeydObject import globalKeyHandler, RUN_ALL, VIS_DOMAINLINES


from printing import InfoPrintCapable
_infoer = InfoPrintCapable()
_infoer.doPrint = False # True


class PartDomainLinesVis(PartModuleVis):
    """ VisItem to compute iso cropped polygons from polygons """

    def __init__(self):
        PartModuleVis.__init__(self, DomainSurface, VIS_DOMAINLINES, self.__class__.__name__,['GridIn0'],[],[],[],[],[],[],None,False,False)
        
        self.params = PartDomainLinesVisParams()
        self.__initBase()

    def __initBase(self):
        """
            + __initBase is called from the constructor and after the class was unpickled
            + add privately created modules here
        """
        self.__firstTime = True

        # create custom modules here
        self.__myCollect = Collect()
        theNet().add(self.__myCollect)

    def __init(self, negMsgHandler):
        """ __init is called from __update
            + start _module module if it does not already exist and connect output to COVER
            + set default parameters of module
            + set default boundingBox which is also used in the GUI to set borders of the float sliders
            + set color inport and corresponding module
            + send params to the gui
        """
        if self.__firstTime == True:
            self.__firstTime = False
            PartModuleVis._init(self, negMsgHandler)
            
            # make my custom connections between further self created modules here
            theNet().connect(self._module, 'GridOut1', self.__myCollect, 'GridIn0')

            # send params to gui
            self.sendParams()

    def __getstate__(self):
        mycontent = PartModuleVis.__getstate__(self)
        del mycontent['_PartDomainLinesVis__myCollect']
        return mycontent

    def recreate(self, negMsgHandler, parentKey, offset):
        """ recreate is called after all classes of the session have been unpickled """
        PartDomainLinesVisParams.mergeDefaultParams(self.params) # explicitly call mergeDefaultParams of this class
        self.__initBase()
        PartModuleVis.recreate(self, negMsgHandler, parentKey, offset, DomainSurface, ['GridIn0'],[],[],[],[],[],[],None,False,False )

    def delete(self, isInitialized, negMsgHandler=None):
        if isInitialized:
            if hasattr(self, '_PartDomainLinesVis__myCollect') and self.__myCollect: theNet().remove(self.__myCollect)
        PartModuleVis.delete(self, isInitialized, negMsgHandler)

    def __update(self, negMsgHandler):
        """ __update is called from the run method to update the module parameter before execution
            + do init of the IsoCutter module if necessary
            + update module parameters """
        self.__init(negMsgHandler)
        PartModuleVis._update(self, negMsgHandler)

    def run(self, runmode, negMsgHandler):
        """ create a new visulisation
            + register for events from Covise if not done yet
            + runmode RUN_GEO and RUN_OCT are ignored
            + update module parameter
            + exec the module
        """
        assert negMsgHandler
        if runmode==RUN_ALL:
            _infoer.function = str(self.run)
            _infoer.write("go")

            if not hasattr(self, 'importModule'):
                return
            
            self.__update(negMsgHandler)
            PartModuleVis.run(self, runmode, negMsgHandler)
    
    def setParams( self, params, negMsgHandler=None, sendToCover=True):
        """ set parameters from outside
            + init module if necessary
            + mainly receive parameter changes from Gui
            + send status messages to COVER if state has changed
        """
        realChange = ParamsDiff( self.params, params )  

        PartModuleVis.setParams(self, params, negMsgHandler, sendToCover, realChange)
        
    def connectionPoint(self):
        """ return the connection (module/outport) to be connected to COVER
            called by the class VisItem """
        return ConnectionPoint(self.__myCollect, 'GeometryOut0')
        #return PartModuleVis.connectionPoint(self, 'linesOut')

    def getCoObjName(self):
        """ return the generated covise object name
            called by the class VisItem to check if an object name registered by the COVER was created by the iso cutter
        """
        # the returned string is independent of the given portname, but portname must match the name of an out port
        return self.__myCollect.getCoObjName('GeometryOut0')
        #return PartModuleVis.getCoObjName(self, 'linesOut')


class PartDomainLinesVisParams(PartModuleVisParams):
    def __init__(self):
        PartModuleVisParams.__init__(self)
        self.isVisible = True
        self.name = 'DomainLines'
        PartDomainLinesVisParams.mergeDefaultParams(self) # explicitly call mergeDefaultParams of this class

    def mergeDefaultParams(self):
        defaultParams = {
        }
        mergeGivenParams(self, defaultParams)
