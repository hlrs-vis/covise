
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH

import math
import numpy

import VRPCoviseNetAccess
import covise

from VRPCoviseNetAccess import (
    connect,
    disconnect,
    ConnectionPoint,
    globalRenderer,
    theNet)

from VisItem import VisItem, VisItemParams
from PartModuleVis import PartModuleVis, PartModuleVisParams
from PartTransform import PartTransform, PartTransformParams
from coPyModules import BoundingBox
from KeydObject import globalKeyHandler, RUN_ALL, VIS_3D_BOUNDING_BOX, TYPE_3D_PART
from VRPCoviseNetAccess import saveExecute
from Utils import ParamsDiff, multMatVec
import Neg2Gui

from VRPCoviseNetAccess import theNet

import traceback

from printing import InfoPrintCapable
_infoer = InfoPrintCapable()
_infoer.doPrint = False # True

class Part3DBoundingBoxVis(PartModuleVis, PartTransform):
    """ VisItem to color an object with an rgba color """
    def __init__(self):
        PartModuleVis.__init__(self, BoundingBox, VIS_3D_BOUNDING_BOX, self.__class__.__name__,['GridIn0'],[],[],[],[],[],[])
        PartTransform.__init__(self)
        self.params = Part3DBoundingBoxVisParams()
        self.__initBase()

    def __initBase(self):
        self.__firstTime = True
        self.__hasTranslatedChildren = False
        self.__hasRotatedChildren = False

    def __init(self, negMsgHandler):
        """ start COVISE Module and connect output to COVER """
        if self.__firstTime == True:
            self.__firstTime = False
            PartModuleVis._init(self, negMsgHandler)
            #PartModuleVis.register( self, negMsgHandler, [] )

    def connectionPoint(self):
        #return self.__boundingBoxOut
        return PartModuleVis.connectionPoint(self, 'GridOut0')

    def recreate(self, negMsgHandler, parentKey, offset):
        self.__initBase()
        PartTransform.recreate(self, negMsgHandler, parentKey, offset)
        PartModuleVis.recreate(self, negMsgHandler, parentKey, offset, BoundingBox, ['GridIn0'],[],[],[],[],[],[])
        if (self.params.name == "Part3DBoundingBoxVisParams") or (self.params.name == "BoundingBox"):
            # The name of Part3DBoundingBoxVis was never changed in previous versions. Copy it from parent if it's the default name.
            self.params.name = globalKeyHandler().getObject(parentKey).params.name


    def registerCOVISEkey( self, covise_key):
        """ called during registration if key received from COVER
            + update states in COVER
        """
        (registered, firstTime) = VisItem.registerCOVISEkey( self, covise_key)
        if registered:
            # set parameters of transformation modules, but dont execute
            self._setTransform()
            # run visualizer so it gets correctly connected to previous modules of the import managers
            PartModuleVis.run(self, RUN_ALL, None)
            return (True, firstTime)
        return (False, False)

    def __update(self, negMsgHandler):
        """ __update is called from the run method to update the module parameter before execution
            + do init 
            + update module parameters """
        self.__init(negMsgHandler)
        PartModuleVis._update(self, negMsgHandler)

        self._setTransform()

    def run(self, runmode, negMsgHandler=None):
        if runmode==RUN_ALL:
            _infoer.function = str(self.run)
            _infoer.write("go")

            self.__update(negMsgHandler)
            PartModuleVis.run(self, runmode, negMsgHandler)

            self.__runChildren(runmode, negMsgHandler)

    def __runChildren(self, runmode, negMsgHandler=None):
        # after an initial translation all children objects must reconnect to the 3D part and execute
        # exception: when in recreation
        if self.importModule.hasTranslationModule() and not self.__hasTranslatedChildren and (not negMsgHandler or not negMsgHandler.getInRecreation()):
            self.__hasTranslatedChildren = True

            # do this, because a rotation after a translation needs no __runChildren()
            self.__hasRotatedChildren = True

            for obj in globalKeyHandler().getObject(self.parentKey).objects:
                # dont execute me or duplicated parts
                if obj.key != self.key and not obj.typeNr in [self.typeNr, TYPE_3D_PART]:
                    obj.run(RUN_ALL, negMsgHandler)

        # after an initial rotation all children objects must reconnect to the 3D part and execute
        # exception: when in recreation
        # exception: after an initial translation
        if self.importModule.hasRotationModules() and not self.__hasRotatedChildren and (not negMsgHandler or not negMsgHandler.getInRecreation()):
            self.__hasRotatedChildren = True
            for obj in globalKeyHandler().getObject(self.parentKey).objects:
                # dont execute me or duplicated parts
                if obj.key != self.key and not obj.typeNr in [self.typeNr, TYPE_3D_PART]:
                    obj.run(RUN_ALL, negMsgHandler)


    def setParams( self, params, negMsgHandler=None, sendToCover=True):
        """ set parameters from outside
            + init tracer module if necessary
            + mainly receive parameter changes from Gui
            + send status messages to COVER if state has changed
        """
        _infoer.function = str(self.setParams)
        _infoer.write(" ")

        realChange = ParamsDiff( self.params, params )
        PartModuleVis.setParams(self, params, negMsgHandler, sendToCover, realChange)
        PartTransform.setParams(self, params, negMsgHandler, sendToCover, realChange)

    def getCoObjName(self):
        return PartModuleVis.getCoObjName(self, 'GridOut0')
    

class Part3DBoundingBoxVisParams(PartModuleVisParams, PartTransformParams):
    def __init__(self):
        PartModuleVisParams.__init__(self)
        PartTransformParams.__init__(self)
        self.name = 'BoundingBox'
