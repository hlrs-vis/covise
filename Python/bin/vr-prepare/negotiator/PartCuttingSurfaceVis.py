
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH


import VRPCoviseNetAccess
import copy
from VRPCoviseNetAccess import (
    connect,
    disconnect,
    ConnectionPoint,
    globalRenderer,
    theNet)

from VisItem import VisItem, VisItemParams
from PartModuleVis import PartModuleVis, PartModuleVisParams
from PartColoredVis import PartColoredVis, PartColoredVisParams
from PartInteractorVis import PartInteractorVis, PartInteractorVisParams, CUT
from ImportSampleManager import USER_DEFINED, MAX_FLT
from coPyModules import CuttingSurfaceComp
import covise
from coGRMsg import coGRMsg, coGRObjAttachedClipPlaneMsg

from Utils import  ParamsDiff, CopyParams, convertAlignedRectangleToCutRectangle, AxisAlignedRectangleIn3d, RectangleIn3d1Mid1Norm, convertCutRectangleToAlignedRectangle

import KeydObject
from KeydObject import globalKeyHandler, RUN_ALL, VIS_PLANE
from coColorTable import coColorTableParams
from printing import InfoPrintCapable
import os

_infoer = InfoPrintCapable()
_infoer.doPrint = False # True

class PartCuttingSurfaceVis(PartModuleVis, PartColoredVis, PartInteractorVis):
    """ VisItem to compute cutting surfaces for 2d or 3d objects """
    def __init__(self, geoInput=[], geoSampleInput=[], dataSampleInput=[], dataInInput=[], sampleType=MAX_FLT):
        if os.getenv('VR_PREPARE_DEBUG_RUN'):
            print("PartCuttingSurfaceVis.__init__")
        PartModuleVis.__init__(self,CuttingSurfaceComp, VIS_PLANE, self.__class__.__name__, geoInput, [], geoSampleInput, dataSampleInput, dataInInput, [], [], sampleType)
        PartColoredVis.__init__(self)
        PartInteractorVis.__init__(self, 1) # mode 1 means cuttingsurface interactor
        self.params = PartPlaneVisParams()
        self.__initBase()

    def __initBase(self):
        """ __initBase is called from the constructor and after the class was unpickled """
        #save last middlepoint from COVER
        self.__point = None
        self.__firstTime = True

    def __init(self, negMsgHandler):
        if os.getenv('VR_PREPARE_DEBUG_RUN'):
            print("PartCuttingSurfaceVis.__init")
        """ __init is called from __update
            + start CuttingSurfaceComp module if it does not already exist and connect output to COVER
            + set default parameters of cuttingsurface
            + set default boundingBox which is also used in the GUI to set borders of the float sliders
            + send params to the gui
        """
        if self.__firstTime==True:
            self.__firstTime = False

            # disable sampling if import is transient
            if self.importModule.getIsTransient():
                self._geoSampleNames  = []
                self._dataSampleNames = []

                # special case for PartVectorVis:
                # transient data cannot have sample modules attached to, so do not try to connect CuttingSurface module to sample modules
                if self.params.vector == 3:
                    PartModuleVis._initBase(self, CuttingSurfaceComp, ['GridIn0'], [], [], [], ['DataIn0'], [], [], USER_DEFINED)

            PartModuleVis._init(self, negMsgHandler)
            PartColoredVis._init(self, negMsgHandler, self._module)  
            PartInteractorVis._init(self, negMsgHandler)
            self.__register(negMsgHandler)  

            #init params
            self._module.set_option(1) # plane
            self._module.set_vector(self.params.vector)

            if hasattr(self.params.boundingBox, 'getXMin' ) and hasattr(self.params, 'scale'):
                if self.params.scale < 0:
                    self.params.scale = self.params.boundingBox.getMaxEdgeLength() * 0.1

            # send params to gui
            self.sendParams()

    def __update(self, negMsgHandler):
        """ __update is called from the run method to update the module parameter before execution
            + do init if the cutting surface module if necessary
            + update module parameters """
        self.__init(negMsgHandler)
        PartModuleVis._update(self, negMsgHandler)
        PartColoredVis._update(self, negMsgHandler)
        PartInteractorVis._update(self, negMsgHandler)
        
        # other parameters
        aRectangleIn3d1Mid1Norm = convertAlignedRectangleToCutRectangle( self.params.alignedRectangle)
        self._module.set_point(*aRectangleIn3d1Mid1Norm.point)
        self._module.set_vertex(*aRectangleIn3d1Mid1Norm.normal)
        self._module.set_option(1)
        self._module.set_vector(self.params.vector)
        if globalKeyHandler().getObject(self.params.colorTableKey[self.currentVariable()]).params.mode==coColorTableParams.LOCAL :
            self._module.set_autoScales('TRUE')
        else :
            self._module.set_autoScales('FALSE')
        self._module.setTitle( self.params.name )

        # init params in case of arrows
        if hasattr(self.params, 'length'): self._module.set_length(self.params.length)
        if hasattr(self.params, 'scale'):
            self._module.set_scale(0.0, 1.0, self.params.scale)
            self._module.set_num_sectors(3)
        if hasattr(self.params, 'arrow_head_factor'): self._module.set_arrow_head_factor(self.params.arrow_head_factor)
        if hasattr(self.params, 'project_arrows'): self._module.set_project_lines(str(self.params.project_arrows))

    def run(self, runmode, negMsgHandler):
        if os.getenv('VR_PREPARE_DEBUG_RUN'):
            print("PartCuttingSurfaceVis.run")
        """ create a new visulisation
            + register for events from Covise if not done yet
            + runmode RUN_GEO and RUN_OCT are ignored
            + update module parameter
            + exec the cutting surface module
        """
        #assert negMsgHandler

        if runmode==RUN_ALL:
            _infoer.function = str(self.run)
            _infoer.write("go")

            if not hasattr(self, 'importModule'):
                return

            self.__update(negMsgHandler)

            PartColoredVis.run(self, runmode, negMsgHandler, self._module)
            PartModuleVis.run(self, runmode, negMsgHandler)
            PartInteractorVis.run(self, runmode, negMsgHandler)


    def __register(self, negMsgHandler):
        """ register to receive events from covise """
        PartModuleVis.register( self, negMsgHandler, [ 'vertex', 'point', 'Min/Max'] )

    def registerCOVISEkey( self, covise_key):
        """ called during registration if key received from COVER
            + update states in COVER
        """
        (registered, firstTime) = VisItem.registerCOVISEkey( self, covise_key)
        if registered:
            self.sendInteractorStatus()
            self.sendSmokeStatus()
            self.sendInteractorAxis()
            self.sendClipPlane()
            return (True, firstTime)
        return (False, False)

    def recreate(self, negMsgHandler, parentKey, geoInput=[], geoSampleInput=[], dataSampleInput=[], dataInInput=[], sampleType=MAX_FLT, offset=0):
        """ recreate is called after all classes of the session have been unpickled """
        self.__initBase()
        self.params.setMoreParams()
        PartInteractorVis.setFormat(self, 1)
        PartModuleVis.recreate(self, negMsgHandler, parentKey, offset, CuttingSurfaceComp, geoInput, [], geoSampleInput, dataSampleInput, dataInInput, [], [], sampleType)
        PartColoredVis.recreate(self,negMsgHandler, parentKey, offset )
        PartColoredVis.setColorMap(self,True)
        
    def setParamsByModule( self, mparam, mvalue):
        """ receives parameter changes from the cutting surface module
            return a list of objKey and their parameters to be set by the Neg2Gui class
        """
        _infoer.function = str(self.setParamsByModule)
        _infoer.write(" ")
        pChangeList = []
        newparams = CopyParams(self.params)
        if mparam == 'point':
            self.__point = mvalue
            return pChangeList
        if mparam == 'vertex':
            aaa = RectangleIn3d1Mid1Norm()#convertAlignedRectangleToGeneral(self.params.alignedRectangle)
            if self.__point:
                aaa.point=(float(self.__point[0]), float(self.__point[1]), float(self.__point[2]) )
            aaa.normal=(float(mvalue[0]), float(mvalue[1]), float(mvalue[2]) )
            xxx=convertCutRectangleToAlignedRectangle(aaa, self.params.alignedRectangle.orthogonalAxis)
            newparams.alignedRectangle=xxx
            self.__point=None
        if mparam == 'option':
            newparams.option= int(mvalue[0])
        if mparam == 'scale':
            newparams.scale= float(mvalue[2])
        if mparam == 'length':
            newparams.length= int(mvalue[0])

        pChangeList.append( (self.key, newparams) )

        if mparam == 'Min/Max':
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
            + init cutting surface module if necessary
            + mainly receive parameter changes from Gui
            + send status messages to COVER if state has changed
        """
        _infoer.function = str(self.setParams)
        _infoer.write("setParams")

        realChange = ParamsDiff( self.params, params )

        PartModuleVis.setParams(self, params, negMsgHandler, sendToCover, realChange)
        PartColoredVis.setParams(self, params, negMsgHandler, sendToCover, realChange)
        PartInteractorVis.setParams(self, params, negMsgHandler, sendToCover, realChange)
        
        if 'isVisible' in realChange:
            if self.params.isVisible:
                self.sendClipPlane()
            else:
                self.sendClipPlaneOFF()

        if (   ('attachedClipPlane_index' in realChange)
            or ('attachedClipPlane_offset' in realChange)
            or ('attachedClipPlane_flip' in realChange)):
            self.sendClipPlane()

    def sendClipPlaneOFF(self):
        msg = coGRObjAttachedClipPlaneMsg( coGRMsg.ATTACHED_CLIPPLANE, self.covise_key, -1, self.params.attachedClipPlane_offset, self.params.attachedClipPlane_flip )
        covise.sendRendMsg(msg.c_str())

    def sendClipPlane(self):
        msg = coGRObjAttachedClipPlaneMsg( coGRMsg.ATTACHED_CLIPPLANE, self.covise_key, self.params.attachedClipPlane_index, self.params.attachedClipPlane_offset, self.params.attachedClipPlane_flip )
        covise.sendRendMsg(msg.c_str())


class PartCuttingSurfaceVisParams(PartModuleVisParams, PartColoredVisParams, PartInteractorVisParams):
    def __init__(self):
        PartModuleVisParams.__init__(self)
        PartColoredVisParams.__init__(self)
        PartInteractorVisParams.__init__(self)
        self.isVisible = True
        self.option = 1
        self.vector = 1
        self.name = 'PartCuttingSurfaceVisParams'
        self.setMoreParams()

    def setMoreParams(self):
        if not hasattr(self, 'attachedClipPlane_index'): self.attachedClipPlane_index = -1
        if not hasattr(self, 'attachedClipPlane_offset'): self.attachedClipPlane_offset = 0.0005
        if not hasattr(self, 'attachedClipPlane_flip'): self.attachedClipPlane_flip = False


class PartPlaneVis(PartCuttingSurfaceVis):
    """ VisItem to compute a plane for 2d or 3d objects """
    def __init__(self):
        PartCuttingSurfaceVis.__init__(self,['GridIn0'],['GridIn1'],['DataIn4'],['DataIn0'], MAX_FLT)
        self.params = PartPlaneVisParams()
        self.params.name = 'Plane'
        NameColoredCuttingSurface = covise.getCoConfigEntry("vr-prepare.NameColoredCuttingSurface")
        if NameColoredCuttingSurface:
            self.params.name = NameColoredCuttingSurface
        self.params.option = 1
        self.params.vector = 2

    def recreate(self, negMsgHandler, parentKey, offset):
        PartCuttingSurfaceVis.recreate(self, negMsgHandler, parentKey, ['GridIn0'],['GridIn1'],['DataIn4'],['DataIn0'], MAX_FLT, offset)

class PartPlaneVisParams(PartCuttingSurfaceVisParams):
    def __init__(self):
        PartCuttingSurfaceVisParams.__init__(self)

class PartVectorVis(PartCuttingSurfaceVis):
    """ VisItem to compute vectors for 2d or 3d objects """
    def __init__(self):
        # start cutting suface not on sampled grid
        if not covise.coConfigIsOn("vr-prepare.ArrowsOnSampledGrid", True) or not covise.coConfigIsOn("vr-prepare.UseSamplingModules", False):
            PartCuttingSurfaceVis.__init__(self,  ['GridIn0'], [], [], ['DataIn0'], USER_DEFINED)
        # start cutting surface on sampled grid
        else:
            PartCuttingSurfaceVis.__init__(self, [],  ['GridIn0'], ['DataIn0'], [], USER_DEFINED)
        self.params = PartVectorVisParams()
        self.params.name = 'Arrows'
        NameArrowsCuttingSurface = covise.getCoConfigEntry("vr-prepare.NameArrowsCuttingSurface")
        if NameArrowsCuttingSurface:
            self.params.name = NameArrowsCuttingSurface
        self.params.option = 1
        self.params.vector = 3

    def recreate(self, negMsgHandler, parentKey, offset):
        # start cutting suface not on sampled grid
        if not covise.coConfigIsOn("vr-prepare.ArrowsOnSampledGrid", True) or not covise.coConfigIsOn("vr-prepare.UseSamplingModules", False):
            PartCuttingSurfaceVis.recreate(self, negMsgHandler, parentKey, ['GridIn0'], [], [], ['DataIn0'], USER_DEFINED, offset)
        # start cutting surface on sampled grid
        else:
            PartCuttingSurfaceVis.recreate(self, negMsgHandler, parentKey, [], ['GridIn0'], ['DataIn0'], [], USER_DEFINED, offset)
        #self.params.arrow_head_factor = 0.2
        self.params.setMoreParams()
    def setParams(self, params, negMsgHandler=None, sendToCover=True):
        # set new param if old param class was stored by the presentation manager
        params.setMoreParams()
        PartCuttingSurfaceVis.setParams(self, params, negMsgHandler, sendToCover)
        if self._module:
            self._module.set_length(self.params.length)
            self._module.set_scale(0.0, 1.0, self.params.scale)
            self._module.set_arrow_head_factor(self.params.arrow_head_factor)
            self._module.set_project_lines(str(self.params.project_arrows))


    def __update(self):
        if self._module:
            self._module.set_length(self.params.length)
            self._module.set_scale(0.0, 1.0, self.params.scale)
            self._module.set_arrow_head_factor(self.params.arrow_head_factor)
            self._module.set_project_lines(str(self.params.project_arrows))

    def run(self, runmode, negMsgHandler=None):
        self.__update()
        PartCuttingSurfaceVis.run(self, runmode, negMsgHandler)


class PartVectorVisParams(PartCuttingSurfaceVisParams):
    def __init__(self):
        PartCuttingSurfaceVisParams.__init__(self)
        self.length = 2
        self.scale = -1.0
        self.setMoreParams()

    def setMoreParams(self):
        PartCuttingSurfaceVisParams.setMoreParams(self)
        if not hasattr(self, 'arrow_head_factor'): self.arrow_head_factor = 0.2
        if not hasattr(self, 'project_arrows'): self.project_arrows = False
