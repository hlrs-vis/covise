# Part of the vr-prepare program for dc

# Copyright (c) 2006-2007 Visenso GmbH

import time

from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import QObject, pyqtSignal

import numpy
import os
import os.path
from stat import *
import socket

from Neg2GuiMessages import ( initMsg, deleteMsg, paramMsg, bboxMsg, keyWordMsg, finishLoadingMsg, isTransientMsg,
     setParamsMsg, requestParamsMsg, requestObjMsg, loadObjectMsg,
     negOKMsg, negErrorMsg, negChangePathMsg, SESSION_KEY, NO_REQ_NR, runObjMsg , negAskTimestepReductionMsg, REQUEST_OBJ_SIGNAL, LOAD_OBJECT_SIGNAL, requestDelObjMsg, setPresentationPointMsg,
     moveObjMsg, varNotFoundMsg, selectMsg, deleteFromCOVERMsg)
from coGRMsg import (coGRMsg, coGRObjRegisterMsg, coGRObjVisMsg, coGRCreateViewpointMsg, coGRChangeViewpointIdMsg, coGRViewpointChangedMsg,
     coGRObjSetTransparencyMsg, coGRKeyWordMsg, coGRShowViewpointMsg, coGRCreateDefaultViewpointMsg, coGRToggleFlymodeMsg, coGRToggleVPClipPlaneModeMsg, coGRObjRestrictAxisMsg,
     coGRGraphicRessourceMsg, coGRShowPresentationpointMsg, coGRObjSensorMsg, coGRSendDocNumbersMsg, coGRSetAnimationSpeedMsg, coGRSetTimestepMsg, coGRAnimationOnMsg,
     coGRObjAttachedClipPlaneMsg, coGRActivatedViewpointMsg, coGRSendCurrentDocMsg, coGRSetDocPageSizeMsg, coGRObjMoveObjMsg, coGRSnapshotMsg, coGRGenericParamRegisterMsg,
     coGRGenericParamChangedMsg, coGRObjMovedMsg, coGRObjSetConnectionMsg, coGRObjTransformMsg, coGRObjSelectMsg, coGRObjDelMsg, coGRObjGeometryMsg, coGRObjAddChildMsg, coGRObjKinematicsStateMsg)


from ErrorManager import CoviseFileNotFoundError, TimestepFoundError

import coprjVersion
import KeydObject
from KeydObject import ( globalKeyHandler, globalProjectKey, globalPresentationMgrKey, nameOfCOType, RUN_ALL,
     TYPE_PROJECT, TYPE_CASE, TYPE_2D_GROUP, TYPE_3D_GROUP, TYPE_2D_PART, TYPE_3D_PART, TYPE_3D_COMPOSED_PART,
     TYPE_COLOR_CREATOR, TYPE_COLOR_TABLE, TYPE_COLOR_MGR, TYPE_PRESENTATION_STEP, TYPE_PRESENTATION,
     TYPE_JOURNAL_STEP, TYPE_JOURNAL,
     TYPE_VIEWPOINT_MGR, TYPE_VIEWPOINT, TYPE_SCENEGRAPH_MGR, TYPE_SCENEGRAPH_ITEM, TYPE_DNA_MGR, TYPE_DNA_ITEM, TYPE_GENERIC_OBJECT_MGR, TYPE_GENERIC_OBJECT, TYPE_2D_CUTGEOMETRY_PART,
     VIS_2D_STATIC_COLOR, VIS_2D_SCALAR_COLOR, VIS_2D_RAW, VIS_3D_BOUNDING_BOX,
     VIS_STREAMLINE, VIS_STREAMLINE_2D, VIS_MOVING_POINTS, VIS_PATHLINES, VIS_VRML, VIS_COVISE, VIS_PLANE, VIS_VECTOR,
     VIS_ISOPLANE, VIS_ISOCUTTER, VIS_CLIPINTERVAL, VIS_VECTORFIELD, VIS_DOCUMENT, VIS_DOMAINLINES, VIS_DOMAINSURFACE, TYPE_2D_COMPOSED_PART,
     VIS_MAGMATRACE, TYPE_TRACKING_MGR, VIS_SCENE_OBJECT)#VIS_POINTPROBING,



from coProjectMgr import coProjectMgr
from coCaseMgr import coCaseMgr
from coGroupMgr import co2DGroupMgr, co3DGroupMgr
from co3DPartMgr import co3DPartMgr
from co3DComposedPartMgr import co3DComposedPartMgr
from co2DPartMgr import co2DPartMgr
from co2DComposedPartMgr import co2DComposedPartMgr
from coSessionMgr import coSessionMgr
from coColorCreator import coColorCreator, coColorTable
from coColorMgr import coColorMgr
from coPresentationMgr import coPresentationMgr, coPresentationStep, coPresentationStepParams, coPresentationMgrParams
from coJournalMgr import coJournalMgr, coJournalStep, coJournalStepParams, STEP_PARAM
from coViewpointMgr import coViewpointMgr, coViewpoint, VIEWPOINT_ID_STRING
from coTrackingMgr import coTrackingMgr
from coDocumentMgr import coDocumentMgr, DOCUMENT_ID_STRING
from co2DCutGeometryPartMgr import co2DCutGeometryPartMgr
from coSceneGraphMgr import coSceneGraphMgr, coSceneGraphItem, SCENEGRAPH_PARAMS_STRING
from coGenericObjectMgr import coGenericObjectMgr, coGenericObject
from coDNAMgr import coDNAMgr, coDNAItem
from Part2DStaticColorVis import Part2DStaticColorVis
from Part2DScalarColorVis import Part2DScalarColorVis
from Part2DRawVis import Part2DRawVis, VARIABLE
from Part3DBoundingBoxVis import Part3DBoundingBoxVis
from PartTracerVis import PartStreamlineVis, PartStreamline2DVis,PartMovingPointsVis, PartPathlinesVis
from VrmlVis import VrmlVis
from SceneObjectVis import SceneObjectVis
from CoviseVis import CoviseVis
from PartCuttingSurfaceVis import PartPlaneVis, PartVectorVis
from PartIsoSurfaceVis import PartIsoSurfaceVis
from PartIsoCutterVis import PartIsoCutterVis
from PartClipIntervalVis import PartClipIntervalVis
from PartVectorFieldVis import PartVectorFieldVis
from Part2DRawVis import Part2DRawVis
#from PartPointProbingVis import PartPointProbingVis
from PartDomainLinesVis import PartDomainLinesVis
from PartDomainSurfaceVis import PartDomainSurfaceVis
#from PartMagmaTraceVis import PartMagmaTraceVis

from VRPCoviseNetAccess import globalRenderer

from UI_Actions import COPY_EVENT, DELETE_EVENT, RENDERER_CRASH_EVENT, NOTIFIER_EVENT, GRMSG_EVENT, EXEC_EVENT, VRC_EVENT, theVRCAction, theModuleMsgHandler, theGrMsgAction, theModuleNotifier, theStartMsgHandler
from Utils import CopyParams, ParamsDiff
import Utils

import covise

from printing import InfoPrintCapable
_infoer = InfoPrintCapable()
_infoer.doPrint =  False #True #

import auxils
transactionKeyHandler = auxils.KeyHandler()

import ccNotifier

class _neg2Gui(QtCore.QObject):
    """ send and receive messages from negotiator to gui (negotiator side)"""
    sigInit = pyqtSignal(int, initMsg)
    sigBbox = pyqtSignal(int, bboxMsg)
    sigKeyWord = pyqtSignal(int, keyWordMsg)
    sigSetPresentationPoint = pyqtSignal(int, setPresentationPointMsg)
    sigFinishLoading = pyqtSignal(int, finishLoadingMsg)
    sigSelectObj = pyqtSignal(int, selectMsg)
    sigDelFromCoverObj = pyqtSignal(int, deleteFromCOVERMsg)
    sigIsTransient = pyqtSignal(int, isTransientMsg)
    sigDelObj = pyqtSignal(int, deleteMsg)
    sigParam = pyqtSignal(int, paramMsg)
    sigAskTimestepReduction = pyqtSignal(int, negAskTimestepReductionMsg)
    sigChangePath = pyqtSignal(int, negChangePathMsg)
    sigVarNotFound = pyqtSignal(int, varNotFoundMsg)
    sigNegOk = pyqtSignal(int, negOKMsg)
    sigNegError = pyqtSignal(int, negErrorMsg)
    
    def __init__(self, coverWidgetId=None):
        QtCore.QObject.__init__(self)
        self.__coverWidgetId=coverWidgetId
        self.__inRecreation = False
        self.__inLoadingCoProject = False
        self.__keyHasTempParam = {}
        self._replaceInPathList = []
        self.reductionFactor = False
        if covise.coConfigIsOn("vr-prepare.ShowReductionDialog", False):
            self.setReductionFactor = False
        else:
            self.setReductionFactor = True
        self.__initDataRequest = None
        self.__objKey = None
        self.session = None

        theGrMsgAction().register(self)
        theVRCAction().register(self)
        
    # NOTE: request is not the same request as below (here we just return the key, below we create an object)
    #       bad naming
    def internalRequestObjectCoviseKey(self, key):
        obj = globalKeyHandler().getObject(key)
        if hasattr( obj, 'covise_key'):
            return obj.covise_key
        return 'No key'

    def internalRequestObject( self, typeNr, parentKey=SESSION_KEY, initData=None):
        """ create object and initialize it with initData """
        _infoer.write("_neg2Gui.internalRequestObject, typeNr: %s, parentKey: %s" % (typeNr, parentKey))
        return self.requestObject( -5, requestObjMsg( -1, typeNr, parentKey), initData )

    def internalRequestObjectDuringRecreate( self, typeNr, parentKey=SESSION_KEY, initData=None):
        """ create object and initialize it with initData """
        return self.requestObject( -11, requestObjMsg( -1, typeNr, parentKey), initData )

    def internalDeleteObject( self, key ):
        """ delete object """
        self.requestDelObject( -5, requestDelObjMsg( -1, key ) )

    def internalRecvParams( self, key, params):
        """ set params from internal """
        self.recvParams( -5, setParamsMsg( -1, key, params ) )

    def internalSyncParams( self, key, params):
        """ set params from internal """
        self.recvParams( -6, setParamsMsg( -1, key, params ) )

    def presentationRecvParams( self, key, params, force=False):
        """ set params from internal """
        if not force:
            self.recvParams( -10, setParamsMsg( -1, key, params ) )
        else:
            #TODO
            # ueberpruefen, ob das hier noch benoetigt wird, wenn GUI komplett portiert ist.
            # eigentlich muessten sich dann alle Fenster veraendert haben und neusetzen der paramter sollte korrekte executes ausloesen
            # dann nur recvParams aufrufen
            obj = globalKeyHandler().getObject(key)

            changedParams = ParamsDiff( obj.params, params )

            # synchronization
            for p in changedParams:
                synclist = globalKeyHandler().getObject(KeydObject.globalProjectKey).getSync( obj.key,p )
                if synclist:
                    for pair in synclist:
                        newparam = CopyParams( globalKeyHandler().getObject(pair[0]).params)
                        newparam.__dict__[pair[1]] = msg.params.__dict__[p]
                        self.internalSyncParams( pair[0], newparam )
                        #globalKeyHandler().getObject(pair[0]).setParams(newparam)
                        self.sendParams( pair[0], newparam )
            # journal
            #store old params if necessary

            if not ( obj.typeNr==TYPE_JOURNAL or obj.typeNr==TYPE_JOURNAL_STEP or obj.typeNr==TYPE_CASE):
                glJ = globalKeyHandler().getObject(globalKeyHandler().globalJournalMgrKey)
                if hasattr( glJ, 'hasStatusForKey' ):
                    self.storeParams(obj)

            obj.setParams(params, self )
            self.__keyHasTempParam[obj.key]=False

            #store params in history
            if not ( obj.typeNr==TYPE_JOURNAL or obj.typeNr==TYPE_JOURNAL_STEP or obj.typeNr==TYPE_CASE):
                if hasattr( glJ, 'hasStatusForKey' ):
                    self.storeParams(obj)

            # if request from presentationMgr call run
            vis = hasattr(obj.params, 'isVisible') and obj.params.isVisible
            if not obj.typeNr == VIS_3D_BOUNDING_BOX and vis or obj.typeNr==TYPE_VIEWPOINT_MGR:
                obj.run(RUN_ALL, self)


    def requestObject(self, requestNr, msg, initData=None):
        if (requestNr != -11) and self.__inRecreation: # -11 allows new objects during recreate (in order to request a missing TrackingMgr)
            self.answerOK(requestNr)
            return

        renderer = globalRenderer(self, self.__coverWidgetId)
        if renderer: theModuleMsgHandler().register( renderer.getCovModule(), 0, self)

        typeNr = msg.typeNr
        parentKey = msg.getParentKey()
        _infoer.function = str(self.requestObject)
        _infoer.write("request object of type %s of parent %s" % (
            nameOfCOType[typeNr], parentKey))

        if typeNr == TYPE_PROJECT: #new project
            obj = coProjectMgr()

        elif typeNr == TYPE_CASE: #new coCase
            obj = coCaseMgr()

        elif typeNr==TYPE_2D_GROUP: # new import 2d group
            obj = co2DGroupMgr()

        elif typeNr==TYPE_3D_GROUP: # new import 3d group
            obj = co3DGroupMgr()

        elif typeNr==TYPE_2D_PART: # new 2dpart
            obj = co2DPartMgr()

        elif typeNr==TYPE_2D_COMPOSED_PART: # new 2dpart
            obj = co2DComposedPartMgr()

        elif typeNr==TYPE_2D_CUTGEOMETRY_PART: # new 2dpart
            obj = co2DCutGeometryPartMgr()
            # copy partcase + name
            obj.params.name = globalKeyHandler().getObject(parentKey).params.name + " (cut)"
            if hasattr(globalKeyHandler().getObject(parentKey).params, "partcase"):
                obj.params.partcase = globalKeyHandler().getObject(parentKey).params.partcase

        elif typeNr==TYPE_3D_PART: # new 3dpart
            obj = co3DPartMgr()

        elif typeNr==TYPE_3D_COMPOSED_PART: # new 3dpart
            obj = co3DComposedPartMgr()

        elif typeNr==TYPE_COLOR_TABLE: # new colorTable
            obj = coColorTable()
            parentKey = KeydObject.globalColorMgrKey

        elif typeNr==TYPE_COLOR_CREATOR: # new colorTable
            obj = coColorCreator()

        elif typeNr==TYPE_COLOR_MGR: # new colorMgr
            obj = coColorMgr()

        elif typeNr==TYPE_PRESENTATION:
            obj = coPresentationMgr()

        elif typeNr==TYPE_PRESENTATION_STEP:
            obj = coPresentationStep()

        elif typeNr==TYPE_JOURNAL:
            obj = coJournalMgr()

        elif typeNr==TYPE_JOURNAL_STEP:
            obj = coJournalStep()

        elif typeNr==TYPE_VIEWPOINT_MGR:
            obj = coViewpointMgr()

        elif typeNr==TYPE_VIEWPOINT:
            obj = coViewpoint()

        elif typeNr==TYPE_SCENEGRAPH_MGR:
            obj = coSceneGraphMgr()

        elif typeNr==TYPE_SCENEGRAPH_ITEM:
            obj = coSceneGraphItem()

        elif typeNr==TYPE_GENERIC_OBJECT_MGR:
            obj = coGenericObjectMgr()

        elif typeNr==TYPE_GENERIC_OBJECT:
            obj = coGenericObject()

        elif typeNr==TYPE_DNA_MGR:
            obj = coDNAMgr()

        elif typeNr==TYPE_DNA_ITEM:
            obj = coDNAItem()

        elif typeNr==VIS_2D_RAW: # new 2d visualization with raw 2d data
            obj = Part2DRawVis()
            #for copy 2d and 3d parts
            if not msg.params == None:
                obj.setParams(msg.params)

        elif typeNr==VIS_2D_STATIC_COLOR: # new 2d visualization with static rgb color
            obj = Part2DStaticColorVis()

        elif typeNr==VIS_2D_SCALAR_COLOR: # new 2d visualization with colormapn over value
            obj = Part2DScalarColorVis()

        elif typeNr==VIS_3D_BOUNDING_BOX: # new 3d bounding box visualization
            obj = Part3DBoundingBoxVis()
            #for copy 2d and 3d parts
            if not msg.params == None:
                obj.setParams(msg.params)

        elif typeNr==VIS_STREAMLINE: # new streamline visualization
            obj = PartStreamlineVis()
            #for copy of visualizer
            if not msg.params == None:
                obj.setParams(msg.params)

        elif typeNr==VIS_STREAMLINE_2D:
            obj = PartStreamline2DVis()

        elif typeNr==VIS_MOVING_POINTS:
            obj = PartMovingPointsVis()
            #for copy of visualizer
            if not msg.params == None:
                obj.setParams(msg.params)

        elif typeNr==VIS_PATHLINES:
            obj = PartPathlinesVis()
            #for copy of visualizer
            if not msg.params == None:
                obj.setParams(msg.params)

        elif typeNr == VIS_VRML:
            obj = VrmlVis()

        elif typeNr == VIS_SCENE_OBJECT:
            obj = SceneObjectVis()

        elif typeNr == VIS_COVISE:
            obj = CoviseVis()

        elif typeNr == VIS_PLANE:
            obj = PartPlaneVis()
            #for copy of visualizer
            if not msg.params == None:
                obj.setParams(msg.params)

        elif typeNr == VIS_VECTOR:
            obj = PartVectorVis()
            #for copy of visualizer
            if not msg.params == None:
                obj.setParams(msg.params)

        elif typeNr == VIS_ISOPLANE:
            obj = PartIsoSurfaceVis()
            #for copy of visualizer
            if not msg.params == None:
                obj.setParams(msg.params)

        elif typeNr == VIS_ISOCUTTER:
            obj = PartIsoCutterVis()

        elif typeNr == VIS_CLIPINTERVAL:
            obj = PartClipIntervalVis()

        elif typeNr == VIS_VECTORFIELD:
            obj = PartVectorFieldVis()
            #for copy of visualizer
            if not msg.params == None:
                obj.setParams(msg.params)

        #elif typeNr == VIS_POINTPROBING:
        #    obj = PartPointProbingVis()

        elif typeNr == VIS_DOCUMENT:
            obj = coDocumentMgr()

        elif typeNr == VIS_DOMAINLINES:
            obj = PartDomainLinesVis()

        elif typeNr == VIS_DOMAINSURFACE:
            obj = PartDomainSurfaceVis()

        #elif typeNr == VIS_MAGMATRACE:
        #    obj = PartMagmaTraceVis()

        elif typeNr == TYPE_TRACKING_MGR:
            obj = coTrackingMgr()

        #else:
        #    assert False

        if hasattr(initData,'filename'):
            if self.setReductionFactor:
                obj.init( initData, self.reductionFactor )
            else:
                if not obj.init( initData ):
                    self.__initDataRequest = initData
                    self.__objKey = obj.key
                    self.askTimestepReduction(requestNr, msg)

        # for duplicate 2d and 3d parts
        elif hasattr(initData, 'partcase'):
            if self.setReductionFactor:
                if not obj.init( initData.partcase, self.reductionFactor ) :
                    self.__initDataRequest = initData
                    self.__objKey = obj.key
                    self.askTimestepReduction(requestNr, msg)
            else:
                if not obj.init( initData.partcase ):
                    self.__initDataRequest = initData
                    self.__objKey = obj.key
                    self.askTimestepReduction(requestNr, msg)

        if not parentKey == SESSION_KEY:
            parent = globalKeyHandler().getObject(parentKey)
            parent.addObject(obj)
        self.initObj(obj.key, typeNr, requestNr, parentKey, msg.params)

        if typeNr == TYPE_PROJECT: #new project
            self.internalRequestObject( TYPE_COLOR_MGR, obj.key )
            self.internalRequestObject( TYPE_PRESENTATION, obj.key )
            self.internalRequestObject( TYPE_VIEWPOINT_MGR, obj.key )
            self.internalRequestObject( TYPE_JOURNAL, obj.key )
            self.internalRequestObject( TYPE_SCENEGRAPH_MGR, obj.key )
            self.internalRequestObject( TYPE_TRACKING_MGR, obj.key )
            self.internalRequestObject( TYPE_DNA_MGR, obj.key )
            self.internalRequestObject( TYPE_GENERIC_OBJECT_MGR, obj.key )
        elif typeNr==TYPE_2D_CUTGEOMETRY_PART: # additional VIS_2D_RAW
            rawVis = self.internalRequestObject(VIS_2D_RAW, obj.key, msg.params)
            rawVis.run(RUN_ALL)

        #send params if not duplicated
        if not requestNr==-6:
            self.sendParams(obj.key, obj.getParams(), requestNr)

        if typeNr==VIS_VRML:
            obj.run(RUN_ALL, self)

        return obj

    def requestDelObject(self, requestNr, msg):
        if not globalKeyHandler().hasKey(msg.getKey()):
            return
        if self.__inRecreation:
            _infoer.write("Ignore delete event in loading")
            self.answerOK(requestNr)
            return
        obj = globalKeyHandler().getObject(msg.getKey())
        #if obj.typeNr in [TYPE_PROJECT, TYPE_CASE, TYPE_2D_GROUP, TYPE_3D_GROUP]:
            #return
        # delete
        _infoer.function = str(self.requestDelObject)
        _infoer.write("object %s with key %s " % (obj.name, obj.key))
        obj.delete(True, self)

    def requestDuplicateObject(self, requestNr, msg):
        if self.__inRecreation:
            _infoer.write("IGNORE duplicate in loading")
            self.answerOK(requestNr)
            return
        oldObj = globalKeyHandler().getObject(msg.getKey())
        newObj = self.requestObject(requestNr, requestObjMsg(-6, msg.typeNr, msg.getParentKey(), msg.params), oldObj.params)
        params = newObj.getParams()
        params.name = msg.newName
        newObj.setParams(params)
        self.sendParams(newObj.key, params, requestNr)
        newObj.run(RUN_ALL, self)

        return newObj

    # receives the reduction factor form the import manager
    # and sets the reduction factor and number of timesteps in the coProjectMgr and sends them to the gui
    def internalRecvReductionFactor(self, reductionFactor, numTimeSteps):
        obj = globalKeyHandler().getObject(0)
        params = CopyParams(obj.params)
        if reductionFactor != None:
            params.reductionFactor = reductionFactor
        params.numTimeSteps = numTimeSteps
        obj.setParams(params, self, False)
        self.sendParams(0, params)

    def recvParams(self, requestNr, msg):
        if self.__inRecreation or msg.getKey()==-1:
            _infoer.write("Ignore params settings in loading")
            self.answerOK(requestNr)
            return

        _infoer.function = str(self.recvParams)
        obj = globalKeyHandler().getObject(msg.getKey())

        #if KeyHandler is deleted there is no obj for the viewpoints
        if obj == None :
            return

        changedParams = ParamsDiff( obj.params, msg.params )
        if requestNr==-10 and obj.typeNr == TYPE_COLOR_TABLE and ('colorMapIdx' in changedParams or 'colorMapList' in changedParams):
            oldColorMap = obj.params.colorMapList[obj.params.colorMapIdx]
            newColorMap = msg.params.colorMapList[msg.params.colorMapIdx]
        

        # do nothing if params have not changed and setParams is called from presentationMgr
        sendSensors = False
        if (requestNr==-10):
            if (hasattr(msg.params,"autoActiveSensorIDs") and (len(msg.params.autoActiveSensorIDs) > 0)):
                sendSensors = True
            if not sendSensors and len(changedParams)==0:
                return

        _infoer.write(
            "recv param for object %s with key %s " % (obj.name, obj.key))

        # synchronization
        if requestNr!=-6:
            for p in changedParams:
                synclist = globalKeyHandler().getObject(KeydObject.globalProjectKey).getSync( obj.key,p )
                if synclist:
                    for pair in synclist:
                        newparam = CopyParams( globalKeyHandler().getObject(pair[0]).params)
                        newparam.__dict__[pair[1]] = msg.params.__dict__[p]
                        self.internalSyncParams( pair[0], newparam )
                        #globalKeyHandler().getObject(pair[0]).setParams(newparam)
                        self.sendParams( pair[0], newparam )
        # journal
        #store old params if necessary

        if not ( obj.typeNr==TYPE_JOURNAL or obj.typeNr==TYPE_JOURNAL_STEP or obj.typeNr==TYPE_CASE):
            glJ = globalKeyHandler().getObject(globalKeyHandler().globalJournalMgrKey)
            if hasattr( glJ, 'hasStatusForKey' ):
                self.storeParams(obj)

        if requestNr==-5:
            obj.setParams(msg.params, self, False)
        else:
            obj.setParams(msg.params, self )
        self.__keyHasTempParam[obj.key]=False

        #store params in history
        if not ( obj.typeNr==TYPE_JOURNAL or obj.typeNr==TYPE_JOURNAL_STEP or obj.typeNr==TYPE_CASE):
            if hasattr( glJ, 'hasStatusForKey' ):
                self.storeParams(obj)

        # if request from presentationMgr call run
        if requestNr==-10:
            if not obj.typeNr == VIS_3D_BOUNDING_BOX and self.__needExecute(changedParams):
                obj.run(RUN_ALL, self)
            elif sendSensors:
                obj.run(RUN_ALL, self)
            elif not obj.typeNr == VIS_3D_BOUNDING_BOX and 'color' in changedParams:
                if msg.params.color==VARIABLE:
                    obj.run(RUN_ALL, self)
            elif obj.typeNr == TYPE_COLOR_TABLE and ('colorMapIdx' in changedParams or 'colorMapList' in changedParams):
                if oldColorMap != newColorMap:
                    obj.run(RUN_ALL, self)
                
        self.answerOK( requestNr )

    def __needExecute(self, changedParams):
        '''check if module needs to be executed '''
        # these params need no execute
        noExec = ['isVisible', 'name',
                'r', 'g', 'b',
                'transparency', 'transparencyOn',
                'shaderFilename', 'shaderName', 'shaderParamsFloat', 'shaderParamsVec2', 'shaderParamsVec3', 'shaderParamsVec4', 'shaderParamsInt', 'shaderParamsBool', 'shaderParamsMat2', 'shaderParamsMat3', 'shaderParamsMat4', 'shader',
                'ambient', 'specular', 'shininess',
                'rotX', 'rotY', 'rotZ', 'rotAngle', 'transX', 'transY', 'transZ',
                'showInteractor',
                'dependantKeys', 'baseMin', 'baseMax', 'baseObjectName', 'colorMapList',
                'attachedClipPlane_index', 'attachedClipPlane_offset', 'attachedClipPlane_flip', 'isomin', 'isomax', 'color', 'colorMapIdx', 'colorMapList']
        for para in changedParams:
            if not para in noExec:
                # execute if we have a changed param not in noExec
                return True
        return False

    def recvTempParams(self, requestNr, msg):
        if self.__inRecreation:
            _infoer.write("Ignore params settings in loading")
            self.answerOK(requestNr)
            return

        _infoer.function = str(self.recvTempParams)
        obj = globalKeyHandler().getObject(msg.getKey())
        changedParams = ParamsDiff( obj.params, msg.params )
        if len( changedParams )==0:
            _infoer.write(
            "recv equal temp param for object %s with key %s " % (obj.name, obj.params.__dict__))
            self.answerOK(requestNr)
            return

        _infoer.write(
            "recv temp param for object %s with key %s " % (obj.name, obj.key))
        if requestNr==-5:
            obj.setParams(msg.params, self, False)
        else:
            obj.setParams(msg.params, self )
        self.__keyHasTempParam[obj.key]=True
        self.answerOK( requestNr )

    def storeParams( self, obj ):
        """ store parameter in history """

        # not used in version 1
        return


        new_step_obj = self.internalRequestObject( TYPE_JOURNAL_STEP, globalKeyHandler().globalJournalMgrKey )
        new_step_param = coJournalStepParams()
        new_step_param.name = 'Param change of ' + obj.params.name
        new_step_param.action  = STEP_PARAM
        new_step_param.key     = obj.key
        new_step_param.param   = CopyParams(obj.params)
        new_step_obj.params = new_step_param
        # send info to gui
        pMgr = globalKeyHandler().getObject( globalKeyHandler().globalJournalMgrKey )
        self.sendParams( pMgr.key, pMgr.params )
        self.sendParams( new_step_obj.key, new_step_obj.params )

    def requestParams(self, requestNr, msg):
        obj = globalKeyHandler().getObject(msg.getKey())
        _infoer.function = str(self.requestParams)
        _infoer.write(
            "request param for object %s with key %s " % (obj.name, obj.key))
        self.sendParams(obj.key, obj.getParams(), requestNr)

    def run(self, objKey):
        self.runObject(-1, runObjMsg( -1, objKey))

    def runObject( self, requestNr, msg ):
        if self.__inRecreation:
            _infoer.write("Ignore run event in loading")
            self.answerOK(requestNr)
            return
        obj = globalKeyHandler().getObject(msg.getKey())
        _infoer.function = str(self.runObject)
        _infoer.write(
            "run object for object %s with key %s " % (obj.name, obj.key))
        if not self.__inRecreation:
            obj.run(msg.runmode, self)

            # synchronize
            #if requestNr != -6:
            #    synclist = globalKeyHandler().getObject(KeydObject.globalProjectKey).getSyncKeys( obj.key )
            #    for key in synclist:
            #        self.runObject(-6, runObjMsg( -6, key))

        # print("Run ", msg.getKey(), globalKeyHandler().getObject(msg.getKey()).params.__dict__)
        self.answerOK( requestNr )

    def setInRecreation( self, inrecreation ):
        self.__inRecreation = inrecreation

    def getInRecreation(self):
        return self.__inRecreation

    def setInLoadingCoProject(self, inLoading):
        self.__inLoadingCoProject = inLoading

    def getInLoadingCoProject(self):
        return self.__inLoadingCoProject

    def loadObject( self, requestNr, msg ):
        if self.__inRecreation:
            _infoer.write("Ignore load event in loading")
            self.answerOK(requestNr)
            return
        _infoer.function = str(self.loadObject)
        _infoer.write("load Object %s" % (msg.filename))
        for path in msg.replaceInPathList:
            self._replaceInPathList.append(path)
        if not self.session:
            self.session = coSessionMgr()
            self.setInLoadingCoProject(True)  # turned off after all items got registered
        if not msg.addToCurrent:
            globalRenderer(self,self.__coverWidgetId)
            # erst nach recreate kann ein fehlendes .covise erkannt werden -> woher die replaceInPathList nehmen?
            try:
                self.session.recreate(self, msg.filename, 0, self._replaceInPathList, None, False)
                # normally there are always items still waiting to be registered, except for empty projects
                if theGrMsgHandler().getNumVisItemsToBeRegistered() == 0:
                    self.sendFinishLoading()
                    globalKeyHandler().getObject(KeydObject.globalProjectKey).params.coprjVersion = coprjVersion.version
                    self.setInLoadingCoProject(False)
            except CoviseFileNotFoundError as wrongFileName:
                theGrMsgHandler().resetNumRegisteredVisItems()
                self.answerChangePath(requestNr, msg, str(wrongFileName))
                return
            except TimestepFoundError:
                theGrMsgHandler().resetNumRegisteredVisItems()
                self.askTimestepReduction(requestNr, msg)
                return
        else :
            try:
                self.session.recreate(self, msg.filename, globalKeyHandler().getOffset(), self._replaceInPathList, None, msg.autoSync)
            except CoviseFileNotFoundError as wrongFileName:
                theGrMsgHandler().resetNumRegisteredVisItems()
                self.answerChangePath(requestNr, msg, str(wrongFileName))
                return
            except TimestepFoundError:
                theGrMsgHandler().resetNumRegisteredVisItems()
                self.askTimestepReduction(requestNr, msg)
                return
        self.answerOK( requestNr )

    def saveObject( self, requestNr, msg ):
        if self.__inRecreation:
            _infoer.write("Ignore save event in loading")
            self.answerOK(requestNr)
            return
        _infoer.function = str(self.saveObject)
        _infoer.write(
            "save Object %s" % (msg.filename))
        self.answerOK( requestNr )
        session = coSessionMgr()
        session.setProject( globalKeyHandler().getObject(msg.getKey()) )
        session.save(msg.filename)


    def recvReductionFactor(self, requestNr, msg):
        """ set the reduction factor (ReduceSet) for the project """
        globalKeyHandler().getObject(KeydObject.globalProjectKey).setReductionFactor(msg.reductionFactor, self)
        
        # execute the 2D/3D parts so everyone knows they are reduced/subsetted
        for key in globalKeyHandler().getAllElements():
            if globalKeyHandler().getObject(key) and globalKeyHandler().getObject(key).typeNr in [TYPE_3D_PART, TYPE_2D_PART]:
                globalKeyHandler().getObject(key).run(RUN_ALL, self)

        self.answerOK(requestNr)

    def recvSelectionString(self, requestNr, msg):
        """ set the selection string (GetSubset) for the project """
        globalKeyHandler().getObject(KeydObject.globalProjectKey).setSelectionString(msg.selectionString)

        # execute the 2D/3D parts so everyone knows they are reduced/subsetted
        for key in globalKeyHandler().getAllElements():
            if globalKeyHandler().getObject(key) and globalKeyHandler().getObject(key).typeNr in [TYPE_3D_PART, TYPE_2D_PART]:
                globalKeyHandler().getObject(key).run(RUN_ALL, self)

        self.answerOK(requestNr)

    def recvCropMinMax(self, requestNr, msg):
        """ set the cropping min/max (CropUsg) for the project """
        globalKeyHandler().getObject(KeydObject.globalProjectKey).setCropMin(*msg.cropMin)
        globalKeyHandler().getObject(KeydObject.globalProjectKey).setCropMax(*msg.cropMax)

        # execute the 2D/3D parts so everyone knows they are cropped
        for key in globalKeyHandler().getAllElements():
            if globalKeyHandler().getObject(key) and globalKeyHandler().getObject(key).typeNr in [TYPE_3D_PART, TYPE_2D_PART]:
                globalKeyHandler().getObject(key).run(RUN_ALL, self)

        self.answerOK(requestNr)

    def recvChangedPath(self, requestNr, msg):
        replaceInPathList = [(os.path.dirname(msg.wrongFileName), msg.correctedPath)]
        msgWithReplacings = loadObjectMsg( msg.filename, msg.addToCurrent, msg.autoSync, msg.getRequestNr(), replaceInPathList )
        self.loadObject(requestNr, msgWithReplacings)

    def recvAskedReductionFactor(self, requestNr, msg):
        self.reductionFactor = msg.reduction
        self.setReductionFactor = True
        if msg.oldMsg.getSignal()==QtCore.SIGNAL(REQUEST_OBJ_SIGNAL):
            if self.reductionFactor:
                globalKeyHandler().getObject(KeydObject.globalProjectKey).setReductionFactor(self.reductionFactor, self)
            obj = globalKeyHandler().getObject(self.__objKey)
            obj.setReductionFactor(self.reductionFactor)
            self.__initDataRequest = None
            self.__objKey = None
        elif msg.oldMsg.getSignal()==QtCore.SIGNAL(LOAD_OBJECT_SIGNAL):
            msgWithReduction = loadObjectMsg( msg.oldMsg.filename, msg.oldMsg.addToCurrent, msg.oldMsg.autoSync, msg.getRequestNr(), [])
            self.loadObject(requestNr, msgWithReduction)

    def recvOk(self, requestNr):
        _infoer.function = str(self.recvOk)
        _infoer.write("neg2Gui::recvOk, RequestNr: %s" % requestNr)

    def recvError(self, requestNr):
        _infoer.function = str(self.recvError)
        _infoer.startString = '(error)'
        _infoer.write("neg2Gui::recvError, RequestNr: %s" % requestNr)
        _infoer.reset()

    def recvExit(self, requestNr):
        """ Clean Covise and delete the temprary files """
        _infoer.function = str(self.recvExit)
        _infoer.startString = '(exit)'
        _infoer.write("neg2Gui::recvExit, RequestNr: %s" % requestNr)
        _infoer.reset()
        # quit covise
        covise.clean()
        covise.quit()

    def initObj(self, obj_key, typeNr, requestNr, parentKey, params=None ):
        msg = initMsg(obj_key, typeNr, requestNr, parentKey, params )
        
        self.sigInit.emit( requestNr, msg)
        # msg.send can not be used due to qt problems
        # msg.send()

    # Called only from KeydObject when an object is deleted.
    # NEVER call this if you want to delete an object, just delete the object!
    def deletingObject(self, obj_key, typeNr, requestNr, parentKey ):
        # remove synchronization entries
        sync = globalKeyHandler().getObject(KeydObject.globalProjectKey).params.sync
        newSync = {}
        for key, synclist in iter(sync.items()):
            if key[0] != obj_key:
                newSyncList = filter(lambda x: x[0] != obj_key, synclist)
                if len(newSyncList) > 0:
                    newSync[key] = newSyncList
        globalKeyHandler().getObject(KeydObject.globalProjectKey).params.sync = newSync
        # remove from presentation steps
        presentationMgr = globalKeyHandler().getObject(globalPresentationMgrKey)
        if (presentationMgr):
            presentationMgr.removeKey(obj_key)
        # remove from theGrMsgHandler (includes the covise key mapping)
        theGrMsgHandler().remove(globalKeyHandler().getObject(obj_key))
        # send message to gui
        msg = deleteMsg(obj_key, typeNr, requestNr, parentKey)
        self.sigDelObj.emit(requestNr, msg)
        
    def sendParams(self, obj_key, obj_params, requestNr=NO_REQ_NR):
        msg = paramMsg(obj_key, CopyParams(obj_params), requestNr) # CopyParams removes the huge status dictionary which we don't need in the GUI part
        self.sigParam.emit(requestNr, msg)

    def askTimestepReduction(self, requestNr, loadObjectMessage):
        msg = negAskTimestepReductionMsg(loadObjectMessage, requestNr)
        self.sigAskTimestepReduction.emit(requestNr, msg)

    def answerChangePath(self, requestNr, loadObjectMessage, wrongFileName):
        msg = negChangePathMsg(loadObjectMessage.filename, loadObjectMessage.addToCurrent, loadObjectMessage.autoSync, requestNr, loadObjectMessage.replaceInPathList, wrongFileName)
        self.sigChangePath.emit(requestNr, msg)

    def raiseVariableNotFound(self, wrongVarName):
        msg = varNotFoundMsg(wrongVarName, NO_REQ_NR)
        self.sigVarNotFound.emit(NO_REQ_NR, msg)

    def waitForAnswer(self, requestNr):
        while transactionKeyHandler.hasKey(requestNr):
            time.sleep(1.)

    def answerOK(self, requestNr):
        msg = negOKMsg(requestNr)
        self.sigNegOk.emit(requestNr, msg)

    def answerError(self, requestNr):
        msg = negErrorMsg(requestNr)
        self.sigNegError.emit(requestNr, msg)

    def registerParamsNotfier( self, covModule, objKey, params ):
        theModuleNotifier(objKey).register(self)
        for param in params:
            covModule.addNotifier(param,theModuleNotifier(objKey))

    def registerStartNotifier( self, covModule, obj ):
        theStartMsgHandler().register(covModule, obj, self)

    def registerCopyModules( self, covModules, obj ):
        for module in covModules:
            theModuleMsgHandler().register( module, obj, self)

    def registerVisItem( self, visItem ):
        theGrMsgHandler().register(visItem)

    def sendBBox(self, key, bbox, requestNr=NO_REQ_NR):
        msg = bboxMsg(key, CopyParams(bbox), requestNr)
        self.sigBbox.emit(requestNr, msg)

    def sendKeyWord(self, keyWord, requestNr=NO_REQ_NR):
        msg = keyWordMsg(keyWord, requestNr)
        self.sigKeyWord.emit(requestNr, msg)

    def sendPresentationPointID(self, pid, requestNr=NO_REQ_NR):
        msg = setPresentationPointMsg(pid, requestNr)
        self.sigSetPresentationPoint.emit(requestNr, msg)

    def sendFinishLoading(self):
        msg = finishLoadingMsg(NO_REQ_NR)
        self.sigFinishLoading.emit(NO_REQ_NR, msg)
        
    def sendSelect(self, params):
        msg = selectMsg(params[0], params[1], NO_REQ_NR)
        self.sigSelectObj.emit(NO_REQ_NR, msg)

    def sendDeleteFromCOVER(self, key):
        msg = deleteFromCOVERMsg(key, NO_REQ_NR)
        self.sigDelFromCoverObj.emit(NO_REQ_NR, msg)
        
    def sendIsTransient(self, isTransient = True):
        """ tell the gui that we have transient data, so it can show appropriate windows """
        msg = isTransientMsg(NO_REQ_NR, isTransient)
        self.sigIsTransient.emit(NO_REQ_NR, msg)
    def restartRenderer(self):
        globalRenderer(self.__coverWidgetId).restart()
        globalKeyHandler().getObject(globalProjectKey).reconnect()
        globalKeyHandler().getObject(globalProjectKey).sendMessages()
        globalRenderer(self.__coverWidgetId).setAlive(True)


    def customEvent(self,e):
        if self.__inRecreation:
            return

        if e.type() == COPY_EVENT :
            obj = globalKeyHandler().getObject(e.key)
            newObj = self.internalRequestObject( obj.typeNr, obj.parentKey)
            #newObj.params = obj.params
            newObj.params.variable = obj.params.variable
            newObj.params.name = 'Copy of ' + obj.params.name
            newObj.setParams( newObj.params, self )
            newObj.run(RUN_ALL, self )
            self.sendParams( newObj.key, newObj.params )
        elif e.type() == DELETE_EVENT :
            self.internalDeleteObject( e.key )
        elif e.type() == EXEC_EVENT :
            # ignore incoming exec events to prevent ping-pong behavior
            theStartMsgHandler().setEnabled(False)

            obj = globalKeyHandler().getObject(e.key)

            # synchronization
            synclist = globalKeyHandler().getObject(KeydObject.globalProjectKey).getSyncKeys( obj.key )
            for key in synclist:
                self.runObject(-6, runObjMsg( -6, key))

            theStartMsgHandler().setEnabled(True)
        elif e.type() == RENDERER_CRASH_EVENT :
            RendererRestartMode = covise.getCoConfigEntry("vr-prepare.RendererRestartMode")
            if RendererRestartMode and RendererRestartMode == "RENDERER_RESTART_AUTOMATIC":
                self.restartRenderer()
            print("Too much")
            globalRenderer().died()
            self.sendFinishLoading()

        elif e.type() == NOTIFIER_EVENT :
            obj = globalKeyHandler().getObject(e.key)
            for changes in obj.setParamsByModule( e.param, e.value ):
                key = changes[0]
                newparams = changes[1]
                self.internalRecvParams( key, newparams )
                self.sendParams( key, newparams )
        elif e.type() == GRMSG_EVENT :
            (key, params) = theGrMsgHandler().run(e.msgstring)
            
            if type(key) == str:
                if key == "keyWord":
                    self.sendKeyWord(params)
                    return
                if key == "finishLoading":
                    self.sendFinishLoading()

                    # if we are loading a coprj, then we just registered all items and are done
                    if self.getInLoadingCoProject() == True:
                        globalKeyHandler().getObject(KeydObject.globalProjectKey).params.coprjVersion = coprjVersion.version
                        self.setInLoadingCoProject(False)

                    # automatic testing
                    self.test_finished_loading = True
                    self.test_continue()
                if key == "select":
                    self.sendSelect(params)
                if key == "delete":
                    self.sendDeleteFromCOVER(params)
            elif type(key) == int:
                if key == globalPresentationMgrKey:
                    self.sendPresentationPointID(params)
                    
                elif key>-1 and params:
                    self.internalRecvParams( key, params )
                    self.sendParams( key, params )
            else:
                print ("Unknown type of key: ", key, ", type = ", type(key))
                
        elif e.type() == VRC_EVENT :
            param = e.param
            if len(param[3]) == 1:
                x, y, z =  self.parseVRCPos( param[4], param[5], param[6] )
            else:
                x, y, z =  self.parseVRCPos( param[3], param[4], param[5] )
            button = int(param[2])
            self.handleVRCButton(button)
            trackingMgr = globalKeyHandler().getObject(globalKeyHandler().globalTrackingMgrKey)
            if trackingMgr != None:
                trackingMgr.setVRC(x,y,z, int(button), self)

    __handleVRCButton_oldButton = 0
    __handleVRCButton_timeoutTimer = QtCore.QTimer()
    __handleVRCButton_timeoutTimer.setSingleShot(True)
    def handleVRCButton(self, button):
        """
        if button from previous event is still pressed than
        don't react again, but wait till button is released.
        only react, when button is pressed first time.
        """
        if not (self.__handleVRCButton_oldButton == 0 and button != 0):
            self.__handleVRCButton_oldButton = button
            return

        if self.__handleVRCButton_timeoutTimer.isActive():
            return

        #buttons for presentation conrtrol
        ResetPresentation = covise.getCoConfigEntry("vr-prepare.ResetPresentation")
        if ResetPresentation:
            resetPresentation = int(ResetPresentation)
        else:
            resetPresentation = -1
        ForwardPresentation = covise.getCoConfigEntry("vr-prepare.ForwardPresentation")
        if ForwardPresentation:
            forwardPresentation = int(ForwardPresentation)
        else:
            forwardPresentation = -1
        BackwardPresentation = covise.getCoConfigEntry("vr-prepare.BackwardPresentation")
        if BackwardPresentation:
            backwardPresentation = int(BackwardPresentation)
        else:
            backwardPresentation = -1
        if button==resetPresentation:
            self.sendKeyWord("PRESENTATION_GO_TO_START")
            self.__handleVRCButton_timeoutTimer.start(1000)
        elif button == forwardPresentation:
            self.sendKeyWord("PRESENTATION_FORWARD")
            self.__handleVRCButton_timeoutTimer.start(1000)
        elif button == backwardPresentation:
            self.sendKeyWord("PRESENTATION_BACKWARD")
            self.__handleVRCButton_timeoutTimer.start(1000)
        self.__handleVRCButton_oldButton = button

    def parseVRCPos( self, xpos, ypos, zpos ):
        if xpos[0]=='[':
            xstring = xpos[1:len(xpos)]
        else:
            xstring = xpos
        x = float(xstring)

        y = float(ypos)

        if zpos[len(zpos)-1]==']':
            z = float(zpos[:len(zpos)-1])
        else:
            z = float(zpos[:len(zpos)])
        return x, y, z

        if self.choiceOrigin!=None:
            self.dChoiceX = x - self.choiceOrigin[0]
            self.dChoiceY = y - self.choiceOrigin[1]
            self.dChoiceZ = z - self.choiceOrigin[2]

    def keyWord(self, requestNr, msg):
        keyWord = msg.keyWord
        if keyWord == "changePresentationStep":
            presentationMgr = globalKeyHandler().getObject(globalPresentationMgrKey)
            if presentationMgr != None:
                presentationMgr.change()
        else:
            msg = coGRKeyWordMsg(keyWord, True)
            covise.sendRendMsg(msg.c_str())
        

    def moveObj(self, requestNr, msg):
        move = msg.move
        x = msg.x
        y = msg.y
        z = msg.z
        msg = coGRObjMoveObjMsg("", move, x, y, z)
        covise.sendRendMsg(msg.c_str())
            
    def get_current_viewpoint_key_index(self):
        from KeydObject import globalViewpointMgrKey
        gVMK = globalKeyHandler().getObject(globalViewpointMgrKey)
        return gVMK.params.currentKey
        
    # automatic testing
    def test_continue(self):
        # NOTE: We dont want to get here too soon, so FlightTime in the config.xml must not be 0.0.
        #       Otherwise some params are not yet set correctly when finishLoading (or maybe ACTIVATED_VIEWPOINT) is received.
        #       A sleep here would not be enough since vr-prepare does not do anything during this wait,
        if not hasattr(self, "test_finished_loading") or ( not (self.get_current_viewpoint_key_index() < 0) and not hasattr(self, "test_viewpoint_activated")):
            return # we need both messages
        ccNotifier.sendLoadingFinishedUdpMessage()
        # take snapshot
        if os.getenv('VR_PREPARE_DEBUG_SNAPSHOTS_DIR'):
            # first set timestep
            msg = coGRAnimationOnMsg(False)
            covise.sendRendMsg(msg.c_str())
            timesteps = globalKeyHandler().getObject(0).params.numTimeSteps
            msg = coGRSetTimestepMsg(int(timesteps/2), int(timesteps))
            covise.sendRendMsg(msg.c_str())
            time.sleep(2)
            # now take the snapshot
            filepath = os.getenv('VR_PREPARE_DEBUG_SNAPSHOTS_DIR')
            msg = coGRSnapshotMsg( filepath, "snapOnce" )
            covise.sendRendMsg(msg.c_str())
            time.sleep(4) # give the renderer time to make the snapshot
        # select next step
        if os.getenv('VR_PREPARE_DEBUG_PRESENTATION_STEPS'):
            mgr = globalKeyHandler().getObject(globalPresentationMgrKey)
            numSteps = mgr.numSteps()
            if (numSteps > 1):
                if not hasattr(self, "test_current_step"):
                    self.test_current_step = 1 # skip the first step, the first step should already be active
                else:
                    self.test_current_step = self.test_current_step + 1
                if (self.test_current_step < numSteps):
                    point = mgr.objects[self.test_current_step]
                    # Do not change or remove the following print! (It only appears in the tests and is expected in this form.)
                    print("+++++ TEST_VISITEMS +++++ Select presentation step (key=%d, index=%d, name=%s)" % (point.key, point.params.index, point.params.name))
                    #newParams = coPresentationMgrParams()
                    #newParams.currentKey = point.key
                    #mgr.setParams(newParams, theNegMsgHandler())
                    self.sendPresentationPointID(point.params.index)
                    return # don't quit just yet
        # quit
        if os.getenv('VR_PREPARE_DEBUG_QUIT'):
            time.sleep(8) # wait, something might still not be completed
            covise.clean()
            covise.quit()

_negMsgHandler = None
def theNegMsgHandler(coverWidgetId=None):

    """Assert instance and access to the gui-message-handler."""

    global _negMsgHandler
    if None == _negMsgHandler: _negMsgHandler = _neg2Gui(coverWidgetId)
    return _negMsgHandler


class _GuiRenderMessageHandler(object):

    def __init__(self):
        self.visItems = []
        self.coverKey2visItem = {}
        self.__numRegisterVistem = 0

    def resetNumRegisteredVisItems(self):
        self.__numRegisterVistem = 0

    def decreaseNumVisItemsToBeRegistered(self):
        self.__numRegisterVistem = self.__numRegisterVistem-1
        if (self.__numRegisterVistem < 0):
            print("ERROR: more objects registered than expected")

    def getNumVisItemsToBeRegistered(self):
        return self.__numRegisterVistem

    def register( self, visItem ):

        vIToRegister = [ VIS_2D_RAW, VIS_3D_BOUNDING_BOX, VIS_STREAMLINE, VIS_MOVING_POINTS, VIS_PATHLINES, VIS_VRML, \
                         VIS_COVISE, VIS_PLANE, VIS_VECTOR, VIS_ISOPLANE, VIS_ISOCUTTER, VIS_CLIPINTERVAL, VIS_VECTORFIELD, \
                         VIS_STREAMLINE_2D, VIS_VECTORFIELD, VIS_DOMAINLINES, VIS_DOMAINSURFACE, VIS_DOCUMENT, TYPE_DNA_ITEM, VIS_SCENE_OBJECT] # VIS_POINTPROBING, VIS_MAGMATRACE,
        if visItem.typeNr in  vIToRegister:
            self.__numRegisterVistem = self.__numRegisterVistem+1
        self.visItems.append(visItem)

    def remove(self, visItem):
        if hasattr(visItem, "covise_key"):
            keydName = self.__keydName(visItem.covise_key)
            if keydName in self.coverKey2visItem:
                del self.coverKey2visItem[keydName]
        if (visItem in self.visItems):
            self.visItems.remove(visItem)

    def __keydName( self, objname ):
        end = objname.find("(")
        if end > 0:
            return objname[0:objname.find("(")]
        return objname

    def __getVisItem( self, objmsg ):
        keydName = self.__keydName(objmsg.getObjName())
        if keydName in self.coverKey2visItem:
            return self.coverKey2visItem[keydName]
        else:
            return None

    def run(self, msgString):
        _infoer.function = str(self.run)
        _infoer.write("%s" % msgString)
        renderMsg = coGRMsg(msgString)
        if renderMsg.isValid():
            if renderMsg.getType()==coGRMsg.REGISTER:
                msg = coGRObjRegisterMsg(msgString)
                objname = msg.getObjName()
                unregister = msg.getUnregister()
                parentObjName = msg.getParentObjName()
                if unregister==0:
                    for item in self.visItems:
                        if item.params.name and item.params.name.find('SceneGraphMgrParams')>-1:
                            (registered, firstTime) = item.registerCOVISEkey(objname)###, parentObjName)
                            if registered:
                                if firstTime:
                                    self.decreaseNumVisItemsToBeRegistered()
                                    if self.__numRegisterVistem ==0 :
                                        return ( "finishLoading", None)
                                break
                        elif item.params.name and item.params.name.find('DNAMgrParams')>-1:
                            (registered, firstTime) = item.registerCOVISEkey(objname)###, parentObjName)
                            item = item.getItem( objname )
                        else:
                            (registered, firstTime) = item.registerCOVISEkey(objname)
                        if registered:
                            self.coverKey2visItem[ self.__keydName(objname) ] = item
                            if firstTime:
                                self.decreaseNumVisItemsToBeRegistered()
                                if self.__numRegisterVistem ==0 :
                                    return ( "finishLoading", None)
                                break
                #This is not nescessary any more (since the SceneGraphItems are now children of the VRML_VIS)
                #else:
                    #return (None, None)
                    #scgManager = globalKeyHandler().getObject(globalKeyHandler().globalSceneGraphMgrKey)
                    #keys = scgManager.KeysToDelete(objname)
                    ## Actually it should be sufficient to delete just the root since all children will be deleted as well.
                    ## Since we have all children, it doesn't hurt to delete them anyway (if still present).
                    #for key in keys:
                        #if globalKeyHandler().hasKey(key):
                            #theNegMsgHandler().internalDeleteObject(key)

            elif renderMsg.getType()==coGRMsg.GEO_VISIBLE:
                visMsg = coGRObjVisMsg(msgString)
                visItem = self.__getVisItem( visMsg )
                if visItem:
                    param = visItem.params
                    if visMsg.isVisible():
                        param.isVisible = True
                    else :
                        param.isVisible = False
                    return ( visItem.key, param )

            elif renderMsg.getType()==coGRMsg.INTERACTOR_VISIBLE:
                visMsg = coGRObjVisMsg(msgString)
                visItem = self.__getVisItem( visMsg )
                if visItem:
                    param = visItem.params
                    if visMsg.isVisible():
                        param.showInteractor = True
                    else:
                        param.showInteractor = False
                    return ( visItem.key, param )

            elif renderMsg.getType()==coGRMsg.SMOKE_VISIBLE:
                visMsg = coGRObjVisMsg(msgString)
                visItem = self.__getVisItem( visMsg )
                if visItem:
                    param = visItem.params
                    if visMsg.isVisible():
                        param.showSmoke = True
                    else:
                        param.showSmoke = False
                    return ( visItem.key, param )

            elif renderMsg.getType()==coGRMsg.CREATE_VIEWPOINT:
                vMsg = coGRCreateViewpointMsg(msgString)
                if not VIEWPOINT_ID_STRING in self.coverKey2visItem:
                    for item in self.visItems:
                        if item.registerCOVISEkey(VIEWPOINT_ID_STRING)[0]:
                            self.coverKey2visItem[VIEWPOINT_ID_STRING] = item
                if VIEWPOINT_ID_STRING in self.coverKey2visItem:
                    visItem = self.coverKey2visItem[VIEWPOINT_ID_STRING]
                    param = CopyParams(visItem.params)
                    param.newViewpoint = ( vMsg.getViewpointId(), vMsg.getName(), vMsg.getView(), vMsg.getClipplane() )
                    return ( visItem.key, param )

            elif renderMsg.getType() == coGRMsg.CREATE_DEFAULT_VIEWPOINT:
                visMsg = coGRCreateDefaultViewpointMsg(msgString)
                if not VIEWPOINT_ID_STRING in self.coverKey2visItem:
                    for item in self.visItems:
                        if item.registerCOVISEkey(VIEWPOINT_ID_STRING)[0]:
                            self.coverKey2visItem[VIEWPOINT_ID_STRING] = item
                if VIEWPOINT_ID_STRING in self.coverKey2visItem:
                    visItem = self.coverKey2visItem[VIEWPOINT_ID_STRING]
                    param = CopyParams(visItem.params)
                    param.newDefaultViewpoint = ( visMsg.getViewpointId(), visMsg.getName() )
                    return ( visItem.key, param )

            elif renderMsg.getType()==coGRMsg.VIEWPOINT_CHANGED:
                vMsg = coGRViewpointChangedMsg(msgString)
                if not VIEWPOINT_ID_STRING in self.coverKey2visItem:
                    for item in self.visItems:
                        if item.registerCOVISEkey(VIEWPOINT_ID_STRING)[0]:
                            self.coverKey2visItem[VIEWPOINT_ID_STRING] = item
                if VIEWPOINT_ID_STRING in self.coverKey2visItem:
                    visItem = self.coverKey2visItem[VIEWPOINT_ID_STRING]
                    param = CopyParams(visItem.params)
                    param.changedViewpoint = ( vMsg.getViewpointId(), vMsg.getName(), vMsg.getView() ) #, vMsg.getClipplane()
                    return ( visItem.key, param )

            elif renderMsg.getType() == coGRMsg.SENSOR:
                # TODO: Don't use the name to identify the VRML since there might be several with the same name. (Maybe we can use the SceneGraphItemStartId)
                sensorMsg = coGRObjSensorMsg(msgString)
                vrmlFileName = os.path.basename(sensorMsg.getObjName())
                visItem = None
                for item in self.visItems:
                    if (item.typeNr == VIS_VRML) and (os.path.basename(item.params.filename) == vrmlFileName):
                        visItem = item
                        break
                if visItem:
                    params = CopyParams(visItem.params)
                    if not sensorMsg.getSensorId() in params.sensorIDs:
                        params.sensorIDs.append( sensorMsg.getSensorId() )
                    return ( visItem.key, params )

            elif renderMsg.getType() == coGRMsg.SEND_DOCUMENT_NUMBERS:
                sendDocNumMsg = coGRSendDocNumbersMsg(msgString)
                documentIdentifier = self.__keydName(sendDocNumMsg.getObjName())
                visItem = None
                if documentIdentifier in self.coverKey2visItem:
                    visItem = self.coverKey2visItem[documentIdentifier]
                if visItem!= None :
                    params = CopyParams(visItem.params)
                    minPage = sendDocNumMsg.getMinPage()
                    maxPage = sendDocNumMsg.getMaxPage()
                    maxValue = maxPage-minPage+1
                    params.minPage = 1
                    params.maxPage = maxValue
                    return ( visItem.key, params )

            elif renderMsg.getType() == coGRMsg.SEND_CURRENT_DOCUMENT:
                currentDocMsg = coGRSendCurrentDocMsg(msgString)
                currentDoc = currentDocMsg.getCurrentDocument()
                documentIdentifier = self.__keydName(currentDocMsg.getObjName())
                visItem = None
                if documentIdentifier in self.coverKey2visItem:
                    visItem = self.coverKey2visItem[documentIdentifier]
                if visItem!= None :
                    params = CopyParams(visItem.params)
                    params.currentImage = currentDoc
                    return ( visItem.key, params )

            elif renderMsg.getType() == coGRMsg.SET_DOCUMENT_PAGESIZE:
                pageSizeMsg = coGRSetDocPageSizeMsg(msgString)
                documentIdentifier = self.__keydName(pageSizeMsg.getObjName())
                visItem = None
                if documentIdentifier in self.coverKey2visItem:
                    visItem = self.coverKey2visItem[documentIdentifier]
                # do not overwrite horizontal and vertical size of recreated project with COVER values
                if (visItem!= None) and (visItem.params.size == (-1, -1)):
                    params = CopyParams(visItem.params)
                    hs = pageSizeMsg.getHSize()
                    vs = pageSizeMsg.getVSize()
                    params.size = (hs, vs)
                    return ( visItem.key, params )

            elif renderMsg.getType() == coGRMsg.SHOW_VIEWPOINT:
                visMsg = coGRShowViewpointMsg(msgString)
                if VIEWPOINT_ID_STRING in self.coverKey2visItem:
                    visItem = self.coverKey2visItem[VIEWPOINT_ID_STRING]
                    param = CopyParams(visItem.params)
                    param.selectedKey = visMsg.getViewpointId()
                    return ( visItem.key, param )

            elif renderMsg.getType() == coGRMsg.FLYMODE_TOGGLE:
                visMsg = coGRToggleFlymodeMsg(msgString)
                if VIEWPOINT_ID_STRING in self.coverKey2visItem:
                    visItem = self.coverKey2visItem[VIEWPOINT_ID_STRING]
                    param = CopyParams(visItem.params)
                    param.flyingMode = visMsg.getMode()
                    return ( visItem.key, param )

            elif renderMsg.getType() == coGRMsg.VPCLIPPLANEMODE_TOGGLE:
                visMsg = coGRToggleVPClipPlaneModeMsg(msgString)
                if VIEWPOINT_ID_STRING in self.coverKey2visItem:
                    visItem = self.coverKey2visItem[VIEWPOINT_ID_STRING]
                    param = CopyParams(visItem.params)
                    param.clipplaneMode = visMsg.getMode()
                    return ( visItem.key, param )

            elif renderMsg.getType()==coGRMsg.SET_TRANSPARENCY:
                visMsg = coGRObjSetTransparencyMsg(msgString)
                visItem = self.__getVisItem( visMsg )
                if visItem:
                    param = visItem.params
                    if visMsg.getTransparency() < 1.0:
                        param.transparencyOn = True
                    param.transparency = visMsg.getTransparency()
                    return (visItem.key, param)

            elif renderMsg.getType()==coGRMsg.KEYWORD:
                visMsg = coGRKeyWordMsg(msgString)
                msg = visMsg.getKeyWord()
                if(msg.find(SCENEGRAPH_PARAMS_STRING) != -1):
                    scenegraphMgr = globalKeyHandler().getObject(globalKeyHandler().globalSceneGraphMgrKey)
                    scenegraphMgr.setCOVERParams(msg)
                else:
                    return ( "keyWord", visMsg.getKeyWord())

            elif renderMsg.getType()==coGRMsg.RESTRICT_AXIS:
                visMsg = coGRObjRestrictAxisMsg(msgString)
                visItem = self.__getVisItem( visMsg )
                if visItem:
                    param = visItem.params
                    if visMsg.getAxisName() == 'xAxis':
                        param.alignedRectangle.orthogonalAxis = 'x'
                    elif visMsg.getAxisName() == 'yAxis':
                        param.alignedRectangle.orthogonalAxis = 'y'
                    elif visMsg.getAxisName() == 'zAxis':
                        param.alignedRectangle.orthogonalAxis = 'z'
                    return ( visItem.key, param )

            elif renderMsg.getType()==coGRMsg.ATTACHED_CLIPPLANE:
                visMsg = coGRObjAttachedClipPlaneMsg(msgString)
                visItem = self.__getVisItem( visMsg )
                if visItem:
                    param = visItem.params
                    param.attachedClipPlane_index = visMsg.getClipPlaneIndex()
                    param.attachedClipPlane_offset = visMsg.getOffset()
                    param.attachedClipPlane_flip = visMsg.isFlipped()
                    return ( visItem.key, param )

            elif renderMsg.getType()== coGRMsg.CHANGE_VIEWPOINT_ID:
                visMsg = coGRChangeViewpointIdMsg(msgString)
                if not VIEWPOINT_ID_STRING in self.coverKey2visItem:
                    for item in self.visItems:
                        if item.registerCOVISEkey(VIEWPOINT_ID_STRING)[0]:
                            self.coverKey2visItem[VIEWPOINT_ID_STRING] = item
                if VIEWPOINT_ID_STRING in self.coverKey2visItem:
                    visItem = self.coverKey2visItem[VIEWPOINT_ID_STRING]
                    param = CopyParams(visItem.params)
                    param.changeID = True
                    param.oldID = visMsg.getOldId()
                    param.newID = visMsg.getNewId()
                    return ( visItem.key, param )

            elif renderMsg.getType() == coGRMsg.SHOW_PRESENTATIONPOINT:
                visMsg = coGRShowPresentationpointMsg(msgString)
                presentationMgr = globalKeyHandler().getObject(globalPresentationMgrKey)
                return (presentationMgr.key, visMsg.getPresentationpointId())

            elif renderMsg.getType() == coGRMsg.ANIMATION_ON:
                visMsg = coGRAnimationOnMsg(msgString)
                projectMgr = globalKeyHandler().getObject(globalProjectKey)
                if projectMgr == None:
                    return (None, None)
                param = projectMgr.params
                param.animateOn = visMsg.getMode()
                return (globalProjectKey, param )

            elif renderMsg.getType() == coGRMsg.ANIMATION_SPEED:
                visMsg = coGRSetAnimationSpeedMsg(msgString)
                projectMgr = globalKeyHandler().getObject(globalProjectKey)
                if projectMgr == None:
                    return (None, None)
                param = projectMgr.params
                #print("set Animation Speed ",  visMsg.getAnimationSpeedMin(), visMsg.getAnimationSpeed(), visMsg.getAnimationSpeedMax())
                param.animationSpeed = visMsg.getAnimationSpeed()
                param.animationSpeedMin = visMsg.getAnimationSpeedMin()
                param.animationSpeedMax = visMsg.getAnimationSpeedMax()
                return (globalProjectKey, param )

            elif renderMsg.getType() == coGRMsg.ANIMATION_TIMESTEP:
                visMsg = coGRSetTimestepMsg(msgString)
                projectMgr = globalKeyHandler().getObject(globalProjectKey)
                if projectMgr == None:
                    return (None, None)
                param = projectMgr.params
                param.actTimeStep = visMsg.getActualTimeStep()
                param.numTimeSteps = visMsg.getNumTimeSteps()
                #print("ANimation Timesteps: act", param.actTimeStep, " num:", param.numTimeSteps)
                return (globalProjectKey, param )

            elif renderMsg.getType() == coGRMsg.ACTIVATED_VIEWPOINT:
                visMsg = coGRActivatedViewpointMsg(msgString)
                if VIEWPOINT_ID_STRING in self.coverKey2visItem:
                    visItem = self.coverKey2visItem[VIEWPOINT_ID_STRING]
                    param = CopyParams(visItem.params)
                    param.selectedKey = visMsg.getViewpointId()
                    # automatic testing
                    theNegMsgHandler().test_viewpoint_activated = True
                    theNegMsgHandler().test_continue()
                    return ( visItem.key, param )
            elif renderMsg.getType() == coGRMsg.GENERIC_PARAM_REGISTER:
                regMsg = coGRGenericParamRegisterMsg(msgString)
                globalKeyHandler().getObject(globalKeyHandler().globalGenericObjectMgrKey).addGenericParamFromRenderer(regMsg.getObjectName(), regMsg.getParamName(), regMsg.getParamType(), regMsg.getDefaultValue())
            elif renderMsg.getType() == coGRMsg.GENERIC_PARAM_CHANGED:
                changeMsg = coGRGenericParamChangedMsg(msgString)
                globalKeyHandler().getObject(globalKeyHandler().globalGenericObjectMgrKey).changeGenericParamFromRenderer(changeMsg.getObjectName(), changeMsg.getParamName(), changeMsg.getValue())
            elif renderMsg.getType()==coGRMsg.OBJECT_TRANSFORMED:
                visMsg = coGRObjMovedMsg(msgString)
                visItem = self.__getVisItem( visMsg )
                if visItem:
                    param = visItem.params
                    param.rotX = visMsg.getRotX()
                    param.rotY = visMsg.getRotY()
                    param.rotZ = visMsg.getRotZ()
                    param.rotAngle = visMsg.getRotAngle()
                    param.transX = visMsg.getTransX()
                    param.transY = visMsg.getTransY()
                    param.transZ = visMsg.getTransZ()
                    return ( visItem.key, param )
            elif renderMsg.getType()==coGRMsg.GEOMETRY_OBJECT:
                visMsg = coGRObjGeometryMsg(msgString)
                visItem = self.__getVisItem( visMsg )
                if visItem:
                    param = visItem.params
                    param.width = visMsg.getWidth()
                    param.height = visMsg.getHeight()
                    param.length = visMsg.getLength()
                    return ( visItem.key, param )                    
            elif renderMsg.getType()==coGRMsg.TRANSFORM_OBJECT:
                visMsg = coGRObjTransformMsg(msgString)
                visItem = self.__getVisItem( visMsg )
                if visItem:
                    param = visItem.params
                    param.matrix = [visMsg.getMatrix(0,0), visMsg.getMatrix(0,1), visMsg.getMatrix(0,2), visMsg.getMatrix(0,3),visMsg.getMatrix(1,0), visMsg.getMatrix(1,1), visMsg.getMatrix(1,2), visMsg.getMatrix(1,3),visMsg.getMatrix(2,0), visMsg.getMatrix(2,1), visMsg.getMatrix(2,2), visMsg.getMatrix(2,3),visMsg.getMatrix(3,0), visMsg.getMatrix(3,1), visMsg.getMatrix(3,2), visMsg.getMatrix(3,3)]
                    return ( visItem.key, param )
            elif renderMsg.getType()==coGRMsg.SET_CONNECTIONPOINT:
                visMsg = coGRObjSetConnectionMsg(msgString)
                visItem = self.__getVisItem( visMsg )
                if visItem:
                    param = visItem.params
                    name = visMsg.getConnPoint1() + " - " + visMsg.getConnPoint2()
                    if visMsg.isConnected() >=0 and visMsg.isEnabled() >= 0:
                        param.connectionPoints[str(name)] = visMsg.isConnected()
                        param.connectionPointsDisable[str(name)] = visMsg.isEnabled()
                        param.connectedUnit[str(name)] = visMsg.getSecondObjName()
                    elif str(name) not in param.connectionPoints:
                        param.connectionPoints[str(name)] = 0
                        param.connectionPointsDisable[str(name)] = 1
                        param.connectedUnit[str(name)] = visMsg.getSecondObjName()
                    return ( visItem.key, param )
            elif renderMsg.getType()==coGRMsg.SELECT_OBJECT:
                visMsg = coGRObjSelectMsg(msgString)
                visItem = self.__getVisItem( visMsg )
                if visItem:
                    return ("select", [visItem.key, visMsg.isSelected()])
            elif renderMsg.getType()==coGRMsg.DELETE_OBJECT:
                visMsg = coGRObjDelMsg(msgString)
                visItem = self.__getVisItem( visMsg )
                if visItem:
                    return ("delete", visItem.key)
            elif renderMsg.getType()==coGRMsg.ADD_CHILD_OBJECT:
                visMsg = coGRObjAddChildMsg(msgString)
                visItem = self.__getVisItem( visMsg )
                if visItem:
                    params = visItem.params
                    childVisItem = None
                    keydName = self.__keydName(visMsg.getChildObjName())
                    if keydName in self.coverKey2visItem:
                        childVisItem = self.coverKey2visItem[keydName]
                    if not childVisItem:
                        return (None, None)
                    childKey = childVisItem.key
                    if (visMsg.getRemove() == 0) and (not childKey in params.children):
                        params.children.append(childKey)   
                    elif (visMsg.getRemove() == 1) and (childKey in params.children):
                        while params.children.count(childKey) > 0:
                            params.children.remove(childKey)
                    return (visItem.key, params)
            elif renderMsg.getType()==coGRMsg.KINEMATICS_STATE:
                visMsg = coGRObjKinematicsStateMsg(msgString)
                visItem = self.__getVisItem( visMsg )
                if visItem:
                    param = visItem.params
                    param.kinematics_state = visMsg.getState()
                    return ( visItem.key, param )                    
            else:
                print("WARNING: wrong message received")

        return (None, None)

_grMsgHandler = None
def theGrMsgHandler():

    """Assert instance and access to the gui-render-message-handler."""

    global _grMsgHandler
    if None == _grMsgHandler:
        _grMsgHandler = _GuiRenderMessageHandler()
    return _grMsgHandler
# eof
