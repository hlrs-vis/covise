


# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH


import VRPCoviseNetAccess

from VisItem import VisItem, VisItemParams
from PartModuleVis import PartModuleVis, PartModuleVisParams    
from PartColoredVis import PartColoredVis, PartColoredVisParams

from BoundingBox import Box
from coPyModules import IsoSurfaceComp

from Utils import ParamsDiff, CopyParams
from KeydObject import globalKeyHandler, RUN_ALL, VIS_ISOPLANE
from coColorTable import coColorTableParams


from printing import InfoPrintCapable
import os
_infoer = InfoPrintCapable()
_infoer.doPrint = False # True

class PartIsoSurfaceVis(PartModuleVis, PartColoredVis):
    """ VisItem to compute iso surfaces for 2d or 3d objects """
    def __init__(self):
        if os.getenv('VR_PREPARE_DEBUG_RUN'):
            print("PartIsoSurfaceVis.__init__")
        PartModuleVis.__init__(self,IsoSurfaceComp, VIS_ISOPLANE, self.__class__.__name__,['GridIn0'],[],[],[],['DataIn0'],['DataIn1'])
        PartColoredVis.__init__(self)
        self.params = PartIsoSurfaceVisParams()
        self.__initBase()
        
    def __initBase(self):
        """ __initBase is called from the constructor and after the class was unpickled """
        self.__firstTime = True

    def __init(self, negMsgHandler):
        if os.getenv('VR_PREPARE_DEBUG_RUN'):
            print("PartIsoSurfaceVis.__init")
        """ __init is called from __update

            + start _module module if it does not already exist and connect output to COVER
            + set default parameters of Tracer
            + set default boundingBox which is also used in the GUI to set borders of the float sliders
            + send params to the gui
        """
        if self.__firstTime == True:
            self.__firstTime = False
            PartModuleVis._init(self, negMsgHandler)
            PartColoredVis._init(self, negMsgHandler, self._module) 
            self.__register(negMsgHandler)

            #init params, the both are static at the moment
            self._module.set_Interactor(self.params.Interactor) 
            self._module.set_vector(self.params.vector)

            # send params to gui
            self.sendParams()
    
    def recreate(self, negMsgHandler, parentKey, offset):
        """ recreate is called after all classes of the session have been unpickled """
        self.__initBase()
        PartModuleVis.recreate(self, negMsgHandler, parentKey, offset, IsoSurfaceComp, ['GridIn0'],[],[],[],['DataIn0'],['DataIn1'] )
        PartColoredVis.recreate(self, negMsgHandler, parentKey, offset)
        PartColoredVis.setColorMap(self,True)        
        
    def __update(self, negMsgHandler):
        """ __update is called from the run method to update the module parameter before execution
            + do init if the tracer module if necessary
            + update module parameters """
        
        self.__init(negMsgHandler)
        PartModuleVis._update(self, negMsgHandler)
        PartColoredVis._update(self, negMsgHandler)
        
        #colorTable = globalKeyHandler().getObject(self.params.colorTableKey[self.params.variable])
        #if colorTable.params.mode==coColorTableParams.LOCAL :
        #    self._module.set_autominmax('TRUE')
        #else :
        #    self._module.set_autominmax('FALSE')

        # other parameters
        self._module.set_isovalue(self.params.isomin,self.params.isomax,self.params.isovalue)
        self._module.setTitle( self.params.name )


    def run(self, runmode, negMsgHandler):
        if os.getenv('VR_PREPARE_DEBUG_RUN'):
            print("PartIsoSurfaceVis.run")
        """ create a new visulisation
            + register for events from Covise if not done yet
            + runmode RUN_GEO and RUN_OCT are ignored
            + update module parameter
            + exec the tracer module
        """
        
        assert negMsgHandler

        if runmode==RUN_ALL:
            _infoer.function = str(self.run)
            _infoer.write("go")

            if not hasattr(self, 'importModule'):
                return

            self.__update(negMsgHandler)

            PartColoredVis.run(self, runmode, negMsgHandler, self._module, self.fromRecreation )
            PartModuleVis.run(self, runmode, negMsgHandler)


    def __register(self, negMsgHandler):
        """ register to receive events from covise """
        PartModuleVis.register( self, negMsgHandler, [ 'isopoint', 'isovalue', 'Min/Max' ] )   
    
    def setParamsByModule( self, mparam, mvalue):
        """ receives parameter changes from the iso surface module
            return a list of objKey and their parameters to be set by the Neg2Gui class
        """

        _infoer.function = str(self.setParamsByModule)
        _infoer.write("param: %s, value: %s" % (mparam, str(mvalue)))
        pChangeList = []
        newparams = CopyParams(self.params)
        if mparam == 'vector':
            newparams.vector = int(mvalue[0])
        if mparam == 'isovalue':
            newparams.isomin = float(mvalue[0])
            newparams.isomax = float(mvalue[1])
            newparams.isovalue = float(mvalue[2])

        pChangeList.append( (self.key, newparams) )

        if mparam == 'isovalue':
            cTableObject = globalKeyHandler().getObject(self.params.colorTableKey[self.currentVariable()])
            if cTableObject.params.baseMin != float(mvalue[0]) or cTableObject.params.baseMax != float(mvalue[1]):
                cTableObject.params.baseObjectName = self.params.name
                cTableObject.params.baseMin = float(mvalue[0])
                cTableObject.params.baseMax = float(mvalue[1])
                if self.key not in cTableObject.params.dependantKeys:
                    cTableObject.params.dependantKeys.append(self.key)
                pChangeList.append( (cTableObject.key, cTableObject.params) )
            
        return pChangeList

    def setParams( self, params, negMsgHandler=None, sendToCover=True):
        """ set parameters from outside
            + init tracer module if necessary
            + mainly receive parameter changes from Gui
            + send status messages to COVER if state has changed
        """
        realChange = ParamsDiff( self.params, params )  

        PartModuleVis.setParams(self, params, negMsgHandler, sendToCover, realChange)
        PartColoredVis.setParams(self, params, negMsgHandler, sendToCover, realChange)


class PartIsoSurfaceVisParams(PartModuleVisParams, PartColoredVisParams):
    def __init__(self):
        PartModuleVisParams.__init__(self)
        PartColoredVisParams.__init__(self)
        self.isVisible = True
        self.Interactor = 2
        self.vector = 2
        self.name = 'IsoSurface'
        self.len = 1.
        self.isovalue = 0.5
        self.isomin = 0.0
        self.isomax = 1.0
        
