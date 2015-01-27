# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH

import copy
from VRPCoviseNetAccess import ConnectionPoint, RWCoviseModule, connect, theNet

from VisItem import VisItem, VisItemParams
from PartModuleVis import PartModuleVis, PartModuleVisParams
from PartColoredVis import PartColoredVis, PartColoredVisParams

from coPyModules import MagmaTrace, Collect, Colors

from Utils import ParamsDiff, CopyParams, mergeGivenParams
from KeydObject import globalKeyHandler, RUN_ALL, VIS_MAGMATRACE
from coColorTable import coColorTableParams


from printing import InfoPrintCapable
_infoer = InfoPrintCapable()
_infoer.doPrint = False # True


class PartMagmaTraceVis(PartModuleVis, PartColoredVis):
    """ VisItem to compute iso cropped polygons from polygons """

    def __init__(self):
        PartModuleVis.__init__(self, MagmaTrace, VIS_MAGMATRACE, self.__class__.__name__,['geo_in'],[],[],[],['data_in'])
        PartColoredVis.__init__(self)

        self.params = PartMagmaTraceVisParams()
        self.__initBase()

    def __initBase(self):
        """
            + __initBase is called from the constructor and after the class was unpickled
            + add privately created modules here
        """
        self.__firstTime = True

        # create custom modules here
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
            PartColoredVis._init(self, negMsgHandler, self.__myColors, 'cmapIn')
            self.__register(negMsgHandler)

            # make my custom connections between further self created modules here
            theNet().connect(self._module, 'geo_out', self.__myCollect, 'grid')
            theNet().connect(self.__myColors, 'colors', self.__myCollect, 'colors')
            theNet().connect(self._module, 'data_out', self.__myColors, 'Data')

            # send params to gui
            self.sendParams()

    def __getstate__(self):
        mycontent = PartModuleVis.__getstate__(self)
        del mycontent['_PartMagmaTraceVis__myCollect']
        del mycontent['_PartMagmaTraceVis__myColors']
        return mycontent

    def recreate(self, negMsgHandler, parentKey, offset):
        """ recreate is called after all classes of the session have been unpickled """
        PartMagmaTraceVisParams.mergeDefaultParams(self.params) # explicitly call mergeDefaultParams of this class
        self.__initBase()
        PartModuleVis.recreate(self, negMsgHandler, parentKey, offset, MagmaTrace, ['geo_in'],[],[],[],['data_in'] )
        PartColoredVis.recreate(self, negMsgHandler, parentKey, offset)
        PartColoredVis.setColorMap(self,True)

    def delete(self, isInitialized, negMsgHandler=None):
        if isInitialized:
            if hasattr(self, '_PartMagmaTraceVis__myCollect') and self.__myCollect: theNet().remove(self.__myCollect)
            if hasattr(self, '_PartMagmaTraceVis__myColors') and self.__myColors: theNet().remove(self.__myColors)
        PartModuleVis.delete(self, isInitialized, negMsgHandler)

    def __update(self, negMsgHandler):
        """ __update is called from the run method to update the module parameter before execution
            + do init of the IsoCutter module if necessary
            + update module parameters """
        self.__init(negMsgHandler)
        PartModuleVis._update(self, negMsgHandler)
        PartColoredVis._update(self, negMsgHandler)

        # other parameters
        self._module.set_len(self.params.length)
        self._module.set_skip(self.params.skip)
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

            PartModuleVis.run(self, runmode, negMsgHandler)
            PartColoredVis.run(self, runmode, negMsgHandler, self._module, self.fromRecreation )   # self._module is not used!

    def __register(self, negMsgHandler):
        """ register to receive events from covise """
        PartModuleVis.register( self, negMsgHandler, ['len', 'skip'] )

    def setParamsByModule( self, mparam, mvalue):
        """ receives parameter changes from the iso cutter module
            return a list of objKey and their parameters to be set by the Neg2Gui class
        """
        _infoer.function = str(self.setParamsByModule)
        _infoer.write("param: %s, value: %s" % (mparam, str(mvalue)))
        pChangeList = []
        newparams = CopyParams(self.params)
        if mparam == 'len':
            try:
                newparams.length= int(mvalue[0])
            except ValueError:
                newparams.length= int(float(mvalue[0]))

        if mparam == 'skip':
            try:
                newparams.skip= int(mvalue[0])
            except ValueError:
                newparams.skip= int(float(mvalue[0]))

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
        return ConnectionPoint(self.__myCollect, 'geometry')

    def getCoObjName(self):
        """ return the generated covise object name
            called by the class VisItem to check if an object name registered by the COVER was created by the iso cutter
        """
        # the returned string is independent of the given portname, but portname must match the name of an out port
        return self.__myCollect.getCoObjName('geometry')


class PartMagmaTraceVisParams(PartModuleVisParams, PartColoredVisParams):
    def __init__(self):
        PartModuleVisParams.__init__(self)
        PartColoredVisParams.__init__(self)
        self.isVisible = True
        self.name = 'MagmaTrace'
        PartMagmaTraceVisParams.mergeDefaultParams(self) # explicitly call mergeDefaultParams of this class

    def mergeDefaultParams(self):
        defaultParams = {
            'length' : 10,
            'skip' : 20
        }
        mergeGivenParams(self, defaultParams)

