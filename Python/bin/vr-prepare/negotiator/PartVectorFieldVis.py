# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH

import copy
from VRPCoviseNetAccess import ConnectionPoint, RWCoviseModule, connect, theNet

from VisItem import VisItem, VisItemParams
from PartModuleVis import PartModuleVis, PartModuleVisParams
from PartColoredVis import PartColoredVis, PartColoredVisParams
from PartTransform import PartTransform, PartTransformParams
from coGRMsg import coGRMsg, coGRObjColorObjMsg
import covise

from coPyModules import VectorField, Collect, Colors

from Utils import ParamsDiff, CopyParams, mergeGivenParams
from KeydObject import globalKeyHandler, RUN_ALL, VIS_VECTORFIELD
from coColorTable import coColorTableParams


from printing import InfoPrintCapable
_infoer = InfoPrintCapable()
_infoer.doPrint = False # True

# used to distinguish coloring options from the panel
RGB_COLOR = 0
COLOR_MAP = 1


class PartVectorFieldVis(PartModuleVis, PartColoredVis, PartTransform):
    """ VisItem to compute iso cropped polygons from polygons """

    def __init__(self):
        PartModuleVis.__init__(self, VectorField, VIS_VECTORFIELD, self.__class__.__name__,['meshIn'],[],[],[],['vdataIn'])
        PartColoredVis.__init__(self)
        PartTransform.__init__(self, True)

        self.params = PartVectorFieldVisParams()
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
            PartColoredVis._init(self, negMsgHandler, self.__myColors, 'ColormapIn0')
            self.__register(negMsgHandler)

            # make my custom connections between further self created modules here
            theNet().connect(self._module, 'linesOut', self.__myCollect, 'GridIn0')
            theNet().connect(self.__myColors, 'DataOut0', self.__myCollect, 'DataIn0')
            theNet().connect(self._module, 'dataOut', self.__myColors, 'DataIn0')

            # adjust slider range from bounding box
            # done only once, range wont change anymore
#            maxSideLength = self.params.boundingBox.getMaxEdgeLength()
#            self.params.minScalingValue = -maxSideLength
#            self.params.maxScalingValue =  maxSideLength

            # append variable name to real name if not fromRecreation
            if not self.fromRecreation:
                self.params.name = self.params.name + " (" + self.params.variable + ")"

            # send params to gui
            self.sendParams()

    def __getstate__(self):
        mycontent = PartModuleVis.__getstate__(self)
        del mycontent['_PartVectorFieldVis__myCollect']
        del mycontent['_PartVectorFieldVis__myColors']
        return mycontent

    def recreate(self, negMsgHandler, parentKey, offset):
        """ recreate is called after all classes of the session have been unpickled """
        PartVectorFieldVisParams.mergeDefaultParams(self.params) # explicitly call mergeDefaultParams of this class
        self.__initBase()
        PartModuleVis.recreate(self, negMsgHandler, parentKey, offset, VectorField, ['meshIn'],[],[],[],['vdataIn'] )
        PartColoredVis.recreate(self, negMsgHandler, parentKey, offset)
        PartColoredVis.setColorMap(self,True)
        PartTransform.recreate(self, negMsgHandler, parentKey, offset)  

    def delete(self, isInitialized, negMsgHandler=None):
        if isInitialized:
            if hasattr(self, '_PartVectorFieldVis__myCollect') and self.__myCollect: theNet().remove(self.__myCollect)
            if hasattr(self, '_PartVectorFieldVis__myColors') and self.__myColors: theNet().remove(self.__myColors)
        PartModuleVis.delete(self, isInitialized, negMsgHandler)

    def __update(self, negMsgHandler):
        """ __update is called from the run method to update the module parameter before execution
            + do init of the IsoCutter module if necessary
            + update module parameters """
        self.__init(negMsgHandler)
        PartModuleVis._update(self, negMsgHandler)
        PartColoredVis._update(self, negMsgHandler)

#        self._module.set_scale(self.params.minScalingValue, self.params.maxScalingValue, self.params.scalingValue)
        self._module.set_scale(self.params.scalingValue-1.0, self.params.scalingValue+1.0, self.params.scalingValue)
        self._module.set_length(self.params.scalingType + 1)    # +1 because covise choices start from 1
        self._module.set_arrow_head_factor(self.params.arrowHeadFactor)
        self._module.set_num_sectors(3)
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
            #self._sendMatrix()

            if self.params.coloringOption == RGB_COLOR:
                self.__sendColor()

    def __register(self, negMsgHandler):
        """ register to receive events from covise """
        PartModuleVis.register( self, negMsgHandler, [] )

    def setParamsByModule( self, mparam, mvalue):
        """ receives parameter changes from the iso cutter module
            return a list of objKey and their parameters to be set by the Neg2Gui class
        """
        _infoer.function = str(self.setParamsByModule)
        _infoer.write("param: %s, value: %s" % (mparam, str(mvalue)))
        pChangeList = []
        newparams = CopyParams(self.params)

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
        PartTransform.setParams(self, params, negMsgHandler, sendToCover, realChange)

        if self.params.coloringOption == RGB_COLOR:
            self.__sendColor()

    def connectionPoint(self):
        """ return the connection (module/outport) to be connected to COVER
            called by the class VisItem """
        #return PartModuleVis.connectionPoint(self, 'linesOut')
        return ConnectionPoint(self.__myCollect, 'GeometryOut0')

    def getCoObjName(self):
        """ return the generated covise object name
            called by the class VisItem to check if an object name registered by the COVER was created by the iso cutter
        """
        # the returned string is independent of the given portname, but portname must match the name of an out port
        #return PartModuleVis.getCoObjName(self, 'linesOut')
        return self.__myCollect.getCoObjName('GeometryOut0')

    def __sendColor(self):
        """ send interactor geometry to cover """
        if self.keyRegistered():
            _infoer.function = str(self.__sendColor)
            _infoer.write("send")
            msg = coGRObjColorObjMsg( coGRMsg.COLOR_OBJECT, self.covise_key, self.params.r, self.params.g, self.params.b)
            covise.sendRendMsg(msg.c_str())



class PartVectorFieldVisParams(PartModuleVisParams, PartColoredVisParams, PartTransformParams):
    def __init__(self):
        PartModuleVisParams.__init__(self)
        PartColoredVisParams.__init__(self)
        PartTransformParams.__init__(self)
        self.isVisible = True
        self.name = 'VectorField'
        PartVectorFieldVisParams.mergeDefaultParams(self) # explicitly call mergeDefaultParams of this class

    def mergeDefaultParams(self):
        defaultParams = {
            'scalingValue' : 1.0,
            'scalingType' : 0,
            #'minScalingValue' : -1.0,
            #'maxScalingValue' :  1.0,
            'arrowHeadFactor' : 0.2,
            'r' : 255,
            'g' : 255,
            'b' : 255,
            'coloringOption' : COLOR_MAP
        }
        mergeGivenParams(self, defaultParams)