


# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH

import copy
from VRPCoviseNetAccess import ConnectionPoint, RWCoviseModule, connect, theNet

from VisItem import VisItem, VisItemParams
from PartModuleVis import PartModuleVis, PartModuleVisParams
from PartColoredVis import PartColoredVis, PartColoredVisParams

from coPyModules import IsoCutter, Collect, Colors

from Utils import ParamsDiff, CopyParams, mergeGivenParams
from KeydObject import globalKeyHandler, RUN_ALL, VIS_ISOCUTTER
from coColorTable import coColorTableParams


from printing import InfoPrintCapable
_infoer = InfoPrintCapable()
_infoer.doPrint = False # True


class PartIsoCutterVis(PartModuleVis, PartColoredVis):
    """ VisItem to compute iso cropped polygons from polygons """

    def __init__(self):
        PartModuleVis.__init__(self,IsoCutter, VIS_ISOCUTTER, self.__class__.__name__,['inPolygons'],[],[],[],['inData'])
        PartColoredVis.__init__(self)

        self.params = PartIsoCutterVisParams()
        self.__initBase()

    def __initBase(self):
        """
            + __initBase is called from the constructor and after the class was unpickled
            + add privately created modules here
        """
        self.__firstTime = True

        # create custom modules
        self.__myCollect = Collect()
        self.__myColors  = Colors()
        theNet().add(self.__myCollect)
        theNet().add(self.__myColors)


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
            PartColoredVis._init(self, negMsgHandler, self.__myColors, 'ColormapIn0')
            self.__register(negMsgHandler)

            # make my custom connections between further self created modules
            theNet().connect(self._module, 'outPolygons', self.__myCollect, 'GridIn0')
            theNet().connect(self.__myColors, 'DataOut0', self.__myCollect, 'DataIn0')
            theNet().connect(self._module, 'outData', self.__myColors, 'DataIn0')

            # send params to gui
            self.sendParams()

    def __getstate__(self):
        mycontent = PartModuleVis.__getstate__(self)
        del mycontent['_PartIsoCutterVis__myCollect']
        del mycontent['_PartIsoCutterVis__myColors']
        return mycontent
    
    def recreate(self, negMsgHandler, parentKey, offset):
        """ recreate is called after all classes of the session have been unpickled """
        PartIsoCutterVisParams.mergeDefaultParams(self.params) # explicitly call mergeDefaultParams of this class
        self.__initBase()
        PartModuleVis.recreate(self, negMsgHandler, parentKey,offset,IsoCutter, ['inPolygons'],[],[],[],['inData'] )
        PartColoredVis.recreate(self, negMsgHandler, parentKey, offset)
        PartColoredVis.setColorMap(self,True)

    def delete(self, isInitialized, negMsgHandler=None):
        if isInitialized:
            if hasattr(self, '_PartIsoCutterVis__myCollect') and self.__myCollect: theNet().remove(self.__myCollect)
            if hasattr(self, '_PartIsoCutterVis__myColors') and self.__myColors: theNet().remove(self.__myColors)
        PartModuleVis.delete(self, isInitialized, negMsgHandler)

    def __update(self, negMsgHandler):
        """ __update is called from the run method to update the module parameter before execution
            + do init of the IsoCutter module if necessary
            + update module parameters """
        self.__init(negMsgHandler)
        PartModuleVis._update(self, negMsgHandler)
        PartColoredVis._update(self, negMsgHandler)

        # other parameters
        self._module.set_iso_value(self.params.isomin, self.params.isomax, self.params.isovalue)
        self._module.set_cutoff_side(str(self.params.cutoff_side))   # must be done so for bool
        self._module.setTitle( self.params.name )


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

            PartColoredVis.run(self, runmode, negMsgHandler, self._module, self.fromRecreation )   # self._module is not used!
            PartModuleVis.run(self, runmode, negMsgHandler)

    def __register(self, negMsgHandler):
        """ register to receive events from covise """
        PartModuleVis.register( self, negMsgHandler, ['iso_value'] )

    def setParamsByModule( self, mparam, mvalue):
        """ receives parameter changes from the iso cutter module
            return a list of objKey and their parameters to be set by the Neg2Gui class
        """
        _infoer.function = str(self.setParamsByModule)
        _infoer.write("param: %s, value: %s" % (mparam, str(mvalue)))
        pChangeList = []
        newparams = CopyParams(self.params)
        if mparam == 'iso_value':
            newparams.isomin = float(mvalue[0])
            newparams.isomax = float(mvalue[1])
            newparams.isovalue = float(mvalue[2])
            #print " mvalue[0] = ", float(mvalue[0]), " mvalue[1] = ", float(mvalue[1]), " mvalue[2] = ", float(mvalue[2])

        pChangeList.append( (self.key, newparams) )

        return pChangeList

    def setParams( self, params, negMsgHandler=None, sendToCover=True):
        """ set parameters from outside
            + init module if necessary
            + mainly receive parameter changes from Gui
            + send status messages to COVER if state has changed
        """
        realChange = ParamsDiff( self.params, params )  

        PartModuleVis.setParams(self, params, negMsgHandler, sendToCover, realChange)
        PartColoredVis.setParams(self, params, negMsgHandler, sendToCover, realChange)

    def connectionPoint(self):
        """ return the connection (module/outport) to be connected to COVER
            called by the class VisItem """
        return ConnectionPoint(self.__myCollect, 'GeometryOut0')

    def getCoObjName(self):
        """ return the generated covise object name
            called by the class VisItem to check if an object name registered by the COVER was created by the iso cutter
        """
        # the returned string is independent of the given portname, but portname must match the name of an out port
        #return PartModuleVis.getCoObjName(self, 'outPolygons')
        return self.__myCollect.getCoObjName('GeometryOut0')


class PartIsoCutterVisParams(PartModuleVisParams, PartColoredVisParams):
    def __init__(self):
        PartModuleVisParams.__init__(self)
        PartColoredVisParams.__init__(self)
        self.isVisible = True
        self.name = 'Iso Cropped Surface'
        PartIsoCutterVisParams.mergeDefaultParams(self) # explicitly call mergeDefaultParams of this class

    def mergeDefaultParams(self):
        defaultParams = {
            'isovalue' : 0.5,
            'cutoff_side' : True,
            'isomin' : 0.0,
            'isomax' : 1.0
        }
        mergeGivenParams(self, defaultParams)

