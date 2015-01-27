
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH


import VRPCoviseNetAccess

from VRPCoviseNetAccess import (
    connect,
    disconnect,
    ConnectionPoint,
    globalRenderer,
    theNet,
    saveExecute)

from VisItem import VisItem, VisItemParams
try:
    from coPyModules import PerformerScene, AddAttribute
except:
    from coPyModules import PerformerScene
    class AddAttribute(object):
        pass # TODO (FIX AddAttribute)
from KeydObject import RUN_ALL, VIS_VRML, globalKeyHandler
from Utils import ParamsDiff, mergeGivenParams
from coGRMsg import coGRMsg, coGRObjSensorEventMsg
import covise
from Utils import getExistingFilename

import Neg2Gui

from ErrorManager import CoviseFileNotFoundError


from printing import InfoPrintCapable
_infoer = InfoPrintCapable()
_infoer.doPrint = False # True

class VrmlVis(VisItem):
    """ VisItem to show an vrml object """
    def __init__(self):
        VisItem.__init__(self, VIS_VRML, self.__class__.__name__)
        self.params = VrmlVisParams()
        self.params.isVisible = True

        sgm = globalKeyHandler().getObject(globalKeyHandler().globalSceneGraphMgrKey)
        # immediately increase maximumIndex by at least 1, so if for some reason we don't get any SceneGraphItems, the next VrmlVis still uses another startIndex
        sgm.sceneGraphItems_maximumIndex = sgm.sceneGraphItems_maximumIndex + 1
        self.params.sceneGraphItems_startIndex = sgm.sceneGraphItems_maximumIndex

        self.__initBase()

    def __initBase(self):
        self.__connected = False
        self.performerScene = None
        self.addAttribute = None
        self.__loaded = False

    def __update(self):
        """ __update is called from the run method to update the module parameter before execution
            + update module parameters """
        if self.performerScene==None:
            self.performerScene = PerformerScene()
            theNet().add(self.performerScene)
            self.addAttribute = AddAttribute()
            theNet().add(self.addAttribute)
            theNet().connect(self.performerScene, 'model', self.addAttribute, 'inObject')
            # we dont get a register message for some filetypes so dont expect one!
            if self.params.filename.split(".")[-1].lower() in ["via", "vim", "vis",  # Molecules Plugin
                                                               "dyn", "geoall", "str", "sensor" # VRAnim Plugin
                                                               ]:
                Neg2Gui.theGrMsgHandler().decreaseNumVisItemsToBeRegistered()
        # update params
        self.performerScene.set_modelPath( self.params.filename )
        self.performerScene.set_scale(self.params.scale)
        if (self.params.backface == True):
            self.performerScene.set_backface('TRUE')
        else:
            self.performerScene.set_backface('FALSE')
        if (self.params.orientation_iv == True):
            self.performerScene.set_orientation_iv('TRUE')
        else:
            self.performerScene.set_orientation_iv('FALSE')
        if (self.params.convert_xforms_iv == True):
            self.performerScene.set_convert_xforms_iv('TRUE')
        else:
            self.performerScene.set_convert_xforms_iv('FALSE')
        # add attribute
        self.addAttribute.set_attrName('SCENEGRAPHITEMS_STARTINDEX')
        self.addAttribute.set_attrVal(str(self.params.sceneGraphItems_startIndex))

    def createdKey(self, key):
        """ called during registration if key received from COVER """
        _infoer.function = str(self.createdKey)
        _infoer.write(" ")
        importKey = self.addAttribute.getCoObjName('outObject')
        posCover  = key.find("OUT")
        posImport = importKey.find("OUT")
        return ( importKey[0:posImport]==key[0:posCover] )

    def connectionPoint(self):
        """ return the object to be displayed
            called by the class VisItem """
        if self.performerScene:
            return ConnectionPoint(self.addAttribute, 'outObject')

    def recreate(self, negMsgHandler, parentKey, offset):
        VrmlVisParams.mergeDefaultParams(self.params) # explicitly call mergeDefaultParams of this class
        self.__initBase()
        VisItem.recreate(self, negMsgHandler, parentKey, offset)
        if getExistingFilename(self.params.filename) == None:
            raise CoviseFileNotFoundError(self.params.filename)


    def run(self, runmode, negMsgHandler=None):
        if runmode==RUN_ALL:
            _infoer.function = str(self.run)
            _infoer.write("go")
            self.__update()
            if not self.__connected and self.params.isVisible:
                VisItem.connectToCover( self, self )
                self.__connected=True
            if not self.__loaded:
                saveExecute(self.performerScene)
                self.__loaded=True

        # dont auto-activate sensors in recreation
        if negMsgHandler and negMsgHandler.getInRecreation() == False:
            # bei run() die sensoren starten
            # wird (ausserhalb von recreate) nur vom presentation manager aufgerufen
            for sensorID in self.params.autoActiveSensorIDs:
                msg = coGRObjSensorEventMsg(coGRMsg.SENSOR_EVENT, self.params.filename, sensorID, True, True)
                covise.sendRendMsg(msg.c_str())
                msg = coGRObjSensorEventMsg(coGRMsg.SENSOR_EVENT, self.params.filename, sensorID, True, False)
                covise.sendRendMsg(msg.c_str())

    def delete(self, isInitialized, negMsgHandler=None):
        ''' delete this CoviseVis: remove the module '''
        _infoer.function = str(self.delete)
        _infoer.write(" ")
        if isInitialized:
            theNet().remove(self.performerScene)
            theNet().remove(self.addAttribute)
        VisItem.delete(self, isInitialized, negMsgHandler)


    def setParams( self, params, negMsgHandler=None, sendToCover=True):
        """ set parameters from outside """
        _infoer.function = str(self.setParams)
        _infoer.write("setParams")

        if not params.isVisible==self.params.isVisible:
            if params.isVisible and not self.__connected:
                connect(self.connectionPoint()  ,globalRenderer().connectionPoint() )
                self.__connected = True
            elif not params.isVisible and self.__connected:
                disconnect(self.connectionPoint()  ,globalRenderer().connectionPoint() )
                self.__connected = False

        realChange = ParamsDiff(self.params, params)
        VisItem.setParams(self, params, negMsgHandler, sendToCover)

        if params.clickedSensorID != None:
            msg = coGRObjSensorEventMsg(coGRMsg.SENSOR_EVENT, params.filename, params.clickedSensorID, True, True)
            covise.sendRendMsg(msg.c_str())
            msg = coGRObjSensorEventMsg(coGRMsg.SENSOR_EVENT, params.filename, params.clickedSensorID, True, False)
            covise.sendRendMsg(msg.c_str())

            # clickedSensorButtonEvent was handled and can be removed
            self.params.clickedSensorID = None
            self.sendParams()


class VrmlVisParams(VisItemParams):
    def __init__(self):
        VisItemParams.__init__(self)
        self.name = 'VrmlVisParams'
        VrmlVisParams.mergeDefaultParams(self) # explicitly call mergeDefaultParams of this class

    def mergeDefaultParams(self):
        defaultParams = {
            'filename' : '',
            'scale' : -1,
            'backface' : False,
            'orientation_iv' : False,
            'convert_xforms_iv' : False,
            'sensorIDs' : [],
            'autoActiveSensorIDs' : [],
            'clickedSensorID' : None,
            'sceneGraphItems_startIndex' : 0
        }
        mergeGivenParams(self, defaultParams)

    def isStaticParam(self, paramname):
        if VisItemParams.isStaticParam(self, paramname):
            return True
        return paramname in ["filename", "sensorIDs", "sceneGraphItems_startIndex"]
