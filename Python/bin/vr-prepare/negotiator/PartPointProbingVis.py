


# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH


from VisItem import VisItem, VisItemParams
from PartModuleVis import PartModuleVis, PartModuleVisParams
from PartColoredVis import PartColoredVis, PartColoredVisParams
from PartInteractorVis import PartInteractorVis, PartInteractorVisParams

from coPyModules import Probe3D 

from Utils import ParamsDiff, CopyParams
from KeydObject import RUN_ALL, VIS_POINTPROBING

from printing import InfoPrintCapable
_infoer = InfoPrintCapable()
_infoer.doPrint = False # True

class PartPointProbingVis(PartModuleVis, PartColoredVis, PartInteractorVis):
    """ VisItem to compute probing point for 3d objects """
    def __init__(self):
        PartModuleVis.__init__(self,Probe3D, VIS_POINTPROBING, self.__class__.__name__,['meshIn'],['gOcttreesIn'],[],[],['gdataIn'],[],[], 1, True, True)
        PartColoredVis.__init__(self)
        PartInteractorVis.__init__(self)
        self.params = PartPointProbingVisParams()
        self.__initBase()

    def __initBase(self):
        """ __initBase is called from the constructor and after the class was unpickled """
        self.__startpoint1 = None
        self.__firstTime = True

    def __init(self, negMsgHandler):
        """ __init is called from __update

            + start probe3D module if it does not already exist and connect output to COVER
            + set default parameters of Tracer
            + set default boundingBox which is also used in the GUI to set borders of the float sliders
            + send params to the gui
        """
        if self.__firstTime==True:
            PartModuleVis._init(self, negMsgHandler)
            PartColoredVis._init(self, negMsgHandler, self._module, 'colorMapIn')
            PartInteractorVis._init(self, negMsgHandler)      
            self.__register(negMsgHandler)

            #init params
            self._module.set_dimensionality(1) 
            self._module.set_probe_type(self.params.type)
            self._module.set_startpoint1(self.params.startpoint[0],self.params.startpoint[1],self.params.startpoint[2])
            self._module.set_startpoint2(self.params.startpoint[0],self.params.startpoint[1],self.params.startpoint[2])

            # send params to gui
            self.sendParams()

    def __update(self, negMsgHandler):
        """ __update is called from the run method to update the module parameter before execution
            + do init if the tracer module if necessary
            + update module parameters """
        self.__init(negMsgHandler)

        PartModuleVis._update(self, negMsgHandler)
        PartColoredVis._update(self, negMsgHandler)
        PartInteractorVis._update(self, negMsgHandler)           
        
        if not hasattr(self, 'importModule'):
             return

        self._module.set_startpoint1(self.params.startpoint[0],self.params.startpoint[1],self.params.startpoint[2])
        self._module.set_startpoint2(self.params.startpoint[0],self.params.startpoint[1],self.params.startpoint[2])

        self._module.setTitle( self.params.name )

    def connectionPoint(self):
        """ return the object to be displayed
            called by the class VisItem """
        return PartModuleVis.connectionPoint(self, 'ggeometry')

    def getCoObjName(self):
        """ return the generated covise object name
            called by the class VisItem to check if an object name registered by the COVER was created by the tracer
        """
        return PartModuleVis.getCoObjName(self, 'ggeometry')

    def run(self, runmode, negMsgHandler):

        return # TODO: probe is not working at the moment: stuck in saveExecute (data or grid missing)

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
            PartInteractorVis.run(self, runmode, negMsgHandler)
            PartModuleVis.run(self, runmode, negMsgHandler)

    def __register(self, negMsgHandler):
        """ register to receive events from covise """
        PartModuleVis.register( self, negMsgHandler, ['startpoint1', 'startpoint2'] )        

    def registerCOVISEkey( self, covise_key):
        """ called during registration if key received from COVER
            + update states in COVER
        """
        (registered, firstTime) = VisItem.registerCOVISEkey( self, covise_key)
        if registered:
            self.sendInteractorStatus()
            self.sendSmokeStatus()
            return (True, firstTime)
        return (False, False)

    def recreate(self, negMsgHandler, parentKey, offset):

        return # TODO: probe is not working at the moment: stuck in saveExecute (data or grid missing)

        """ recreate is called after all classes of the session have been unpickled """
        self.__initBase()
        PartModuleVis.recreate(self, negMsgHandler, parentKey, offset, Probe3D, ['meshIn'],['gOcttreesIn'],[],[],['gdataIn'],['gdataIn'])
        PartColoredVis.recreate(self, negMsgHandler, parentKey, offset)
        
    def setParamsByModule( self, mparam, mvalue):
        """ receives parameter changes from the tracer module
            return a list of objKey and their parameters to be set by the Neg2Gui class
        """
        _infoer.function = str(self.setParamsByModule)
        _infoer.write(" ")
        pChangeList = []
        newparams = CopyParams(self.params)
        if mparam == 'startpoint1':
            self.__startpoint1 = float(mvalue[0]),float(mvalue[1]),float(mvalue[2])
            return []
        if mparam == 'startpoint2':
            newparams.startpoint = (self.__startpoint1[0]+float(mvalue[0]))/2.0, (self.__startpoint1[1]+float(mvalue[1]))/2.0, (self.__startpoint1[2]+float(mvalue[2]))/2.0

        pChangeList.append( (self.key, newparams) )

        return pChangeList

    def setParams( self, params, negMsgHandler=None, sendToCover=True):

        return # TODO: probe is not working at the moment: stuck in saveExecute (data or grid missing)

        """ set parameters from outside
            + init tracer module if necessary
            + mainly receive parameter changes from Gui
            + send status messages to COVER if state has changed
        """
        _infoer.function = str(self.setParams)
        _infoer.write("setParams")

        realChange = ParamsDiff( self.params, params )
        
        PartModuleVis.setParams(self, params, negMsgHandler, sendToCover, realChange)
        PartColoredVis.setParams(self, params, negMsgHandler, sendToCover, realChange)
        PartInteractorVis.setParams(self, params, negMsgHandler, sendToCover, realChange)        

class PartPointProbingVisParams(PartModuleVisParams, PartColoredVisParams, PartInteractorVisParams):
    def __init__(self):
        PartModuleVisParams.__init__(self)
        PartColoredVisParams.__init__(self)
        PartInteractorVisParams.__init__(self)
        self.isVisible = True
        self.Interactor = 2
        self.name = 'ProbingPoint'
        self.startpoint = 0, 0, 0 
        self.type = 1
        



