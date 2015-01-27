
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH
import math
import numpy

import VRPCoviseNetAccess

from VRPCoviseNetAccess import (
    connect,
    disconnect,
    ConnectionPoint,
    globalRenderer,
    theNet)

from VisItem import VisItem, VisItemParams
import KeydObject
from KeydObject import globalKeyHandler, RUN_ALL, VIS_2D_RAW, TYPE_2D_PART, TYPE_COLOR_TABLE, TYPE_COLOR_CREATOR
from Utils import ParamsDiff, mergeGivenParams
from coGRMsg import coGRMsg, coGRObjColorObjMsg, coGRObjSetTransparencyMsg, coGRObjMaterialObjMsg, coGRObjShaderObjMsg
import covise
from coColorCreator import coColorCreatorParams
from coColorTable import coColorTableParams
from coPyModules import Collect

from PartModuleVis import PartModuleVis, PartModuleVisParams
from PartColoredVis import PartColoredVis, PartColoredVisParams
from PartTransform import PartTransform, PartTransformParams

from ImportManager import Import2DModule
from printing import InfoPrintCapable
import os

_infoer = InfoPrintCapable()
_infoer.doPrint = False # True

NO_COLOR = 0
RGB_COLOR = 1
MATERIAL = 2
VARIABLE = 3

class Part2DRawVis(PartModuleVis, PartColoredVis, PartTransform):
    """ VisItem to show an 2D object with colors """
    def __init__(self):
        PartModuleVis.__init__(self,Collect, VIS_2D_RAW, self.__class__.__name__,['GridIn0'],[],[],[],[],[],[],1,False, False)
        PartColoredVis.__init__(self, False)
        PartTransform.__init__(self)
        self.params = Part2DRawVisParams()
        self.params.isVisible = True
        self.__initBase()

    def __initBase(self):
        """ __initBase is called from the constructor and after the class was unpickled """
        self.__lastColorConnection=None   #color connection into collect module
        self.__firstTime = True
        self.__hasTranslatedChildren = False
        self.__hasRotatedChildren = False

    def __init(self, negMsgHandler):
        """ __init is called from __update"""
        if self.__firstTime == True:
            self.__firstTime = False

            PartModuleVis._init(self, negMsgHandler)
            PartColoredVis._initColor(self, negMsgHandler)
            #PartModuleVis.register(self, negMsgHandler, [])

    def connectionPoint(self):
        """ connection point for COVER """
        return PartModuleVis.connectionPoint(self, 'GeometryOut0')

    def getCoObjName(self):
        """ return the generated covise object name
            called by the class VisItem to check if an object name registered by the COVER was created by the tracer
        """
        return PartModuleVis.getCoObjName(self, 'GeometryOut0')

    def registerCOVISEkey( self, covise_key):
        """ check if object name was created by this visItem
            and if yes store it """
        (registered, firstTime) = VisItem.registerCOVISEkey( self, covise_key )
        if registered:
            self._setTransform()
            if self.params.color==MATERIAL:
                self.__sendMaterial()
            else:
                if self.params.color==RGB_COLOR:
                    self.__sendColor()
            self.__sendTransparency()
            if (self.params.shaderFilename != ""):
                self.__sendShader()
            return (True, firstTime)
        return (False, False)

    def __getstate__(self):
        """ __getstate__ returns a cleaned dictionary
            only called while class is pickled
        """
        mycontent = PartModuleVis.__getstate__(self)#copy.copy(self.__dict__)
        del mycontent['_Part2DRawVis__lastColorConnection']
        _infoer.function = str(self.__getstate__)
        _infoer.write("storing dict %s " % str(self.params.__dict__) )
        return mycontent

    def recreate(self, negMsgHandler, parentKey, offset):
        self.__initBase()
        self.params.mergeDefaultParams()
        PartColoredVis.recreate(self, negMsgHandler, parentKey, offset)
        PartColoredVis.setColorMap(self,False)
        PartTransform.recreate(self, negMsgHandler, parentKey, offset)
        PartModuleVis.recreate(self, negMsgHandler, parentKey, offset, Collect, ['GridIn0'],[],[],[],[],[],[],1,False, False )
        if (self.params.name == "Part2DRawVisParams") or (self.params.name == "Appearance"):
            # The name of Part2DRawVis was never changed in previous versions. Copy it from parent if it's the default name.
            self.params.name = globalKeyHandler().getObject(parentKey).params.name

    def __update(self, negMsgHandler):
        """ __update is called from the run method to update the module parameter before execution
            + do init
            + update module parameters """
        self.__init(negMsgHandler)

        PartModuleVis._update(self, negMsgHandler)
        if self.params.color==VARIABLE:
            PartColoredVis._update(self, negMsgHandler)
        self._setTransform()

        if not hasattr(self, 'importModule'):
            return

        #set color for variable
        if not self.__lastColorConnection==None:
            disconnect(self.__lastColorConnection, ConnectionPoint(self._module, 'DataIn0'))
            self.__lastColorConnection=None
        if self.__lastColorConnection==None and self.params.variable!=None \
            and self.params.variable!='Select a variable' and self.params.color == VARIABLE and len(self.objects)>0:
            _infoer.function = str(self.__update)
            _infoer.write("connection color for variable %s " % self.params.variable)
            self.__lastColorConnection =  (self.objects[0]).colorContConnectionPoint()
            connect(self.__lastColorConnection, ConnectionPoint(self._module, 'DataIn0'))

    def run(self, runmode, negMsgHandler=None):
        if os.getenv('VR_PREPARE_DEBUG_RUN'):
            print("coColorCreator.run")
        if runmode==RUN_ALL:
            _infoer.function = str(self.run)
            _infoer.write("go Part2DRawVis")


            self.__update(negMsgHandler)


            #if a variable is chosen, run PartColoredVis
            if self.params.variable!=None and self.params.variable!= 'Select a variable' and self.params.color == VARIABLE:
                # At the moment the collect might be executed twice.
                # The "pre-run"-disconnect of PartColoredVis does not work here,
                # because our collect is connected to a different port of the colors module (compared to all the other visualizers).
                colorExecuted = PartColoredVis.run(self, runmode, negMsgHandler, self._module)

            PartModuleVis.run(self, runmode, negMsgHandler)

            if self.params.color==MATERIAL:
                self.__sendMaterial()
            elif self.params.color==RGB_COLOR:
                    self.__sendColor()
            self.__sendTransparency()
            if (self.params.shaderFilename != ""):
                self.__sendShader()

            self.__runChildren(runmode, negMsgHandler)

    def __runChildren(self, runmode, negMsgHandler=None):
        # after an initial translation all children objects must reconnect to the 2D part and execute
        # exception: when in recreation
        if hasattr(self.importModule, 'hasTranslationModule') and self.importModule.hasTranslationModule() and not self.__hasTranslatedChildren and (not negMsgHandler or not negMsgHandler.getInRecreation()):
            self.__hasTranslatedChildren = True

            # do this, because a rotation after a translation needs no __runChildren()
            self.__hasRotatedChildren = True

            for obj in globalKeyHandler().getObject(self.parentKey).objects:
                # dont execute me or duplicated parts
                if obj.key != self.key and not obj.typeNr in [self.typeNr, TYPE_2D_PART]:
                    obj.run(RUN_ALL, negMsgHandler)

        # after an initial rotation all children objects must reconnect to the 2D part and execute
        # exception: when in recreation
        # exception: after an initial translation
        if hasattr(self.importModule, 'hasRotationModules') and self.importModule.hasRotationModules() and not self.__hasRotatedChildren and (not negMsgHandler or not negMsgHandler.getInRecreation()):
            self.__hasRotatedChildren = True
            for obj in globalKeyHandler().getObject(self.parentKey).objects:
                # dont execute me or duplicated parts
                if obj.key != self.key and not obj.typeNr in [self.typeNr, TYPE_2D_PART]:
                    obj.run(RUN_ALL, negMsgHandler)



    def setParams( self, params, negMsgHandler=None, sendToCover=True):
        """ set parameters from outside
            + init tracer module if necessary
            + mainly receive parameter changes from Gui
            + send status messages to COVER if state has changed
        """
        _infoer.function = str(self.setParams)
        _infoer.write(" ")
        # save the old variable to delete from colorTable
        oldVariable = self.currentVariable()
        oldTable = None
        if oldVariable!=None and oldVariable!= 'Select a variable':
            if hasattr(self.params.colorTableKey, oldVariable):
                oldTable = globalKeyHandler().getObject(self.params.colorTableKey[oldVariable])

        realChange = ParamsDiff( self.params, params )

        PartModuleVis.setParams(self, params, negMsgHandler, sendToCover, realChange)
        if self.params.color == VARIABLE:
            PartColoredVis.setParams(self, params, negMsgHandler, sendToCover, realChange)
        PartTransform.setParams(self, params, negMsgHandler, sendToCover, realChange)

        needsTransparency = False
        needsShader = False

        if hasattr (params, 'color') and (params.color == NO_COLOR) \
                and ('color' in realChange):
            # disconnect colors module and run the collect module
            # (we dont want to call run since we dont need the children to be executed)
            if not self.__lastColorConnection==None:
                disconnect(self.__lastColorConnection, ConnectionPoint(self._module, 'DataIn0'))
                self.__lastColorConnection=None
            PartModuleVis.run(self, RUN_ALL, negMsgHandler)
            needsTransparency = True
        elif hasattr (params, 'color') and (params.color==RGB_COLOR) \
                and (('color' in realChange) or ('r' in realChange) or ('g' in realChange) or ('b' in realChange)):
            self.__sendColor()
            needsTransparency = True
        elif hasattr (params, 'color') and (params.color==MATERIAL) \
                and (('color' in realChange) or ('r' in realChange) or ('g' in realChange) or ('b' in realChange) \
                  or ('ambient' in realChange) or ('specular' in realChange) or ('shininess' in realChange)):
            self.__sendMaterial()
            needsTransparency = True
        elif hasattr(params, 'transparency') and ('transparency' in realChange):
            needsTransparency = True

        if 'variable' in realChange:
            # if variable changed append key to colorTable dependant keys
            # make sure this part is updated if colormap changes
            if params.variable!=None and params.variable!= 'Select a variable' and params.color == VARIABLE:
                # delete key from old colorTable
                if not  oldTable == None and self.key in oldTable.params.dependantKeys:
                    oldTable.params.dependantKeys.remove(self.key)
                # add key to new colorTable
                cTableObject = globalKeyHandler().getObject(params.colorTableKey[params.variable])
                params.baseObjectName = params.name
                if self.key not in cTableObject.params.dependantKeys:
                    cTableObject.params.dependantKeys.append(self.key)
                    if negMsgHandler:
                        negMsgHandler.internalRecvParams( cTableObject.key, cTableObject.params  )
                        negMsgHandler.sendParams( cTableObject.key, cTableObject.params )

        if ('shaderFilename' in realChange):
            needsTransparency = True
            needsShader = True

        # always send transparency before shader:
        # sendTransparency will ignore any shader transparency but sendShader respects the regular transparency if possible
        if needsTransparency and (params.shaderFilename != ""):
            needsShader = True
        if needsTransparency:
            self.__sendTransparency()
        if needsShader:
            self.__sendShader()
            
    def __sendColor(self):
        """ send interactor geometry to cover """
        if self.keyRegistered():
            _infoer.function = str(self.__sendColor)
            _infoer.write("send")
            msg = coGRObjColorObjMsg( coGRMsg.COLOR_OBJECT, self.covise_key, self.params.r, self.params.g, self.params.b)
            covise.sendRendMsg(msg.c_str())

    def __sendTransparency(self):
        if self.keyRegistered():
            if self.params.transparencyOn:
                transparency = self.params.transparency
            else:
                transparency = 1.0
            msg = coGRObjSetTransparencyMsg(coGRMsg.SET_TRANSPARENCY, self.covise_key, transparency)
            covise.sendRendMsg(msg.c_str())

    def __sendShader(self):
        if (len(self.objects) > 0):
            return # we only send shader messages to leafs (geodes or EOT-nodes)
        msg = coGRObjShaderObjMsg(coGRMsg.SHADER_OBJECT, self.covise_key, self.params.shaderFilename, "", "", "", "", "", "", "", "", "")
        covise.sendRendMsg(msg.c_str())

    def __sendMaterial(self):
        if self.keyRegistered():
            _infoer.function = str(self.__sendMaterial)
            _infoer.write("send")
            if not self.params.transparencyOn:
                transparency = 1.0
            else:
                transparency = self.params.transparency

            msg = coGRObjMaterialObjMsg( coGRMsg.MATERIAL_OBJECT, self.covise_key, \
                  self.params.ambient[0],self.params.ambient[1],self.params.ambient[2], \
                  self.params.r, self.params.g, self.params.b, \
                  self.params.specular[0],self.params.specular[1],self.params.specular[2], self.params.shininess, transparency)
            covise.sendRendMsg(msg.c_str())



class Part2DRawVisParams(PartModuleVisParams, PartColoredVisParams, PartTransformParams):
    def __init__(self):
        PartModuleVisParams.__init__(self)
        PartColoredVisParams.__init__(self)
        PartTransformParams.__init__(self)
        #coloring and shader
        self.name = 'Appearance'
        self.r=200
        self.g=200
        self.b=200
        self.transparency=1.0
        self.transparencyOn=True
        self.color = 0
        self.ambient = 180, 180, 180
        self.specular = 255, 255, 130
        self.shininess = 16.0
        #visualization
        self.variable = 'Select a variable'
        self.allVariables = []
        self.mergeDefaultParams()

    def mergeDefaultParams(self):
        defaultParams = {
                'shaderFilename' : "",
        }
        mergeGivenParams(self, defaultParams)
