
from PyQt5 import QtCore
from PyQt5.QtCore import QObject, pyqtSignal

import os

import Application

from KeydObject import (
    VIS_3D_BOUNDING_BOX,
    VIS_STREAMLINE,
    VIS_STREAMLINE_2D,
    VIS_MOVING_POINTS,
    VIS_PATHLINES,
    VIS_2D_RAW,
    VIS_COVISE,
    VIS_VRML,
    VIS_PLANE,
    VIS_ISOPLANE,
    VIS_ISOCUTTER,
    VIS_CLIPINTERVAL,
    VIS_VECTORFIELD,
    VIS_VECTOR,
    VIS_DOCUMENT,
    VIS_SCENE_OBJECT,
    #VIS_POINTPROBING,
    VIS_DOMAINLINES,
    VIS_DOMAINSURFACE,
    VIS_MAGMATRACE,
    TYPE_PROJECT,
    TYPE_CASE,
    TYPE_2D_GROUP,
    TYPE_3D_GROUP,
    TYPE_2D_PART,
    TYPE_2D_COMPOSED_PART,
    TYPE_2D_CUTGEOMETRY_PART,
    TYPE_3D_PART,
    TYPE_3D_COMPOSED_PART,
    TYPE_COLOR_CREATOR,
    TYPE_COLOR_TABLE,
    TYPE_COLOR_MGR,
    TYPE_PRESENTATION,
    TYPE_PRESENTATION_STEP,
    TYPE_JOURNAL,
    TYPE_JOURNAL_STEP,
    TYPE_VIEWPOINT,
    TYPE_VIEWPOINT_MGR,
    TYPE_TRACKING_MGR,
    TYPE_CAD_PRODUCT,
    TYPE_CAD_PART,
    TYPE_SCENEGRAPH_MGR,
    TYPE_SCENEGRAPH_ITEM,
    TYPE_DNA_MGR,
    TYPE_DNA_ITEM,
    TYPE_GENERIC_OBJECT_MGR,
    TYPE_GENERIC_OBJECT,
    RUN_GEO,
    RUN_OCT,
    RUN_ALL,
    nameOfCOType,
    globalKeyHandler)
from auxils import NamedCheckable
from Utils import CopyParams, getExistingFilename
from Neg2GuiMessages import SESSION_KEY
from coCaseMgr import coCaseMgrParams
from negGuiHandlers import theGuiMsgHandler
from ErrorManager import ( CoviseFileNotFoundError, TIMESTEP_ERROR)
import coviseCase
import covise

from printing import InfoPrintCapable
_infoer = InfoPrintCapable()
_infoer.doPrint = False # True # 

# emit signal for param change
GUI_PARAM_CHANGED_SIGNAL = 'sigGuiParamChanged'
GUI_OBJECT_ADDED_SIGNAL = 'sigGuiObjectAdded'
GUI_OBJECT_DELETED_SIGNAL = 'sigGuiObjectDeleted'

class _ObjMgr(QtCore.QObject):
    """Stroes information about all objects."""
    sigGuiParamChanged = pyqtSignal(int,name=GUI_PARAM_CHANGED_SIGNAL)
    igGuiObjectAdded = pyqtSignal(int, int, int, name=GUI_OBJECT_ADDED_SIGNAL)
    sigGuiObjectDeleted = pyqtSignal(int, int, int, name=GUI_OBJECT_DELETED_SIGNAL)
    def __init__(self):
        QtCore.QObject.__init__(self)
        self.__objects = {}
        # whether init of an object was already done
        self.__initHistory = {}
        self.__projectKey = None
        self.__lastCaseKey = None
        self.__presentationKey = None
        self.__viewpointMgrKey = None
        self.__trackingMgrKey = None
        

        # which variable to set if a new visItem was requested from gui
        self.__initVariable = None

        # list of names used for copy
        self.__copyNames = []

        # last request number for object creation
        self.__newObjReqNr = -1
        self.__newObjKey = -1

        #todo remove
        self.newCoviseFileName=None

        theGuiMsgHandler().registerAddCallback(SESSION_KEY, self.addObjCB)
        theGuiMsgHandler().registerFinishLoadingCallback(self.finishLoading)
        theGuiMsgHandler().registerReduceTimestepCallback(self.reduceTimestep)
        theGuiMsgHandler().registerIsTransientCallback(self.isTransientCB)
        theGuiMsgHandler().registerKeyWordCallback('DELETE_OBJECT', self.deleteObject)        

    def reduceTimestep(self, pMsg):
        return Application.vrpApp.mw.errorCallback(pMsg.getRequestNr(), TIMESTEP_ERROR, pMsg)

    def requestObjForVariable( self, typeNr, parentKey, variable):
        # request for the visualization objects panel
        self.__initVariable = variable
        requestNr = theGuiMsgHandler().requestObject( typeNr, None, parentKey )
        _infoer.write('requestObjForVariable (requestNr %s) ' %  str(requestNr))
        # ATTENTION: the call requestObject returns AFTER new AddObj and Param signals are received. 
        # => self.__newObjNr is set in addObjCB
        # HINT:  This timing can be different on different computers
        if self.__newObjReqNr==requestNr:
            Application.vrpApp.mw.raisePanelForKey(self.__newObjKey)
            self.__newObjReqNr=-1

    # duplicates a 2d- or 3d-part
    def duplicatePart(self, objKey, rotX, rotY, rotZ, rotAngle, transX, transY, transZ):
        # get the typ of the object
        typeNr = self.__objects[objKey].type
        # copy only 2d and 3d parts
        if not typeNr in [TYPE_2D_PART, TYPE_3D_PART]:
            return
        # get the parent of the object
        # parent means 2d_group odr 3d_group
        parent = self.getParentOfObject(objKey)
        while not (self.__objects[parent].type == TYPE_2D_GROUP or self.__objects[parent].type == TYPE_3D_GROUP):
            parent = self.getParentOfObject(parent)
        # get the name of the object
        name = self.getParamsOfObject(objKey).name
        # copy the params 
        newParams = CopyParams(self.getParamsOfObject(objKey))
        # create new name with copy at the end
        newName = name + '(Copy)'
        i = 2
        while newName in self.__copyNames:
            newName = name+'(Copy '+str(i)+')'
            i = i+1
        # set the new params
        newParams.name = newName
        newParams.rotX = rotX
        newParams.rotY = rotY
        newParams.rotZ = rotZ
        newParams.rotAngle = rotAngle
        if hasattr(newParams, 'transX'):
            newParams.transX = newParams.transX + transX
            newParams.transY = newParams.transY + transY
            newParams.transZ = newParams.transZ + transZ
        else:
            newParams.transX = transX
            newParams.transY = transY
            newParams.transZ = transZ
        # send duplicate message
        request = theGuiMsgHandler().requestDuplicateObject( objKey, newName, typeNr, None, parent, newParams)
        theGuiMsgHandler().waitforAnswer(request)

    # duplicates a visualizer
    def duplicateVisualizer(self, objKey):
        # get the typ of the object
        typeNr = self.__objects[objKey].type
        # copy only 2d and 3d parts
        if not typeNr in [VIS_STREAMLINE, VIS_MOVING_POINTS, VIS_PATHLINES, VIS_ISOPLANE, VIS_VECTORFIELD, VIS_PLANE, VIS_VECTOR]:
            return
        # get the parent of the object
        parent = self.getParentOfObject(objKey)
        # get the name of the object
        name = self.getParamsOfObject(objKey).name
        # copy the params 
        newParams = CopyParams(self.getParamsOfObject(objKey))
        # create new name with copy at the end
        newName = name + '(Copy)'
        i = 2
        while newName in self.__copyNames:
            newName = name+'(Copy '+str(i)+')'
            i = i+1
        # set the new params
        newParams.name = newName
        # send duplicate message
        request = theGuiMsgHandler().requestDuplicateObject( objKey, newName, typeNr, None, parent, newParams)
        theGuiMsgHandler().waitforAnswer(request)

    def finishLoading(self):
        Application.vrpApp.mw.unSpawnPatienceDialog()

    def deleteProject(self):
        if (self.__projectKey != None):
            self.deleteObject(self.__projectKey)
        self.__projectKey = None
        # globalKeyHandler will be deleted upon creating/loading a project (init/recreate)

    def initProject(self):
        # delete old project
        self.deleteProject()
        # create new project
        print('new project')
        reqId = theGuiMsgHandler().requestObject(typeNr = TYPE_PROJECT, callback = None, parentKey = SESSION_KEY)
        
        print('wait')
        theGuiMsgHandler().waitforAnswer(reqId)
        print('waitDone')
        # reset Navigation Mode
        navMode = covise.getCoConfigEntry("COVER.NavigationMode")
        if (navMode == "NavNone"):
            Application.vrpApp.mw.navigationModeNone()
        if (navMode == "Transform"):
            Application.vrpApp.mw.navigationModeTransform()
        if (navMode == "Measure"):
            Application.vrpApp.mw.navigationModeMeasure()

        # add Coxml Intitial File
        # NOTE: might be better in negotiator but the current file importing process does not easily allow that
        if (covise.getCoConfigEntry("vr-prepare.Coxml.InitialFile", "") != ""):
            resourceDir = covise.getCoConfigEntry("vr-prepare.Coxml.ResourceDirectory")
            initFile = covise.getCoConfigEntry("vr-prepare.Coxml.InitialFile")
            if (resourceDir != None) and (initFile != None):
                if os.path.exists(resourceDir + "/coxml/" + initFile):
                    self.importFile(initFile)
                else:
                    print("Error: Initial coxml file does not exist (%s)" % initFile)

    def isTransientCB(self, isTransient):
        if covise.coConfigIsOn("vr-prepare.Panels.AnimationManager", "visible", True):
            Application.vrpApp.mw.showAnimationWindow(isTransient)
            Application.vrpApp.mw.windowAnimation_ManagerAction.setEnabled(isTransient)

    def getNewCoviseFileName(self):
        return self.newCoviseFileName

    def deleteObjCB (self, requestNr, typeNr, key, parentKey):
        self.sigGuiObjectDeleted.emit( key, parentKey, typeNr )

        del self.__objects[key]
        del Application.vrpApp.key2params[key]
        del Application.vrpApp.key2type[key]
        if (key in Application.vrpApp.visuKey2GuiKey):
            del Application.vrpApp.visuKey2GuiKey[key]
        if (parentKey in Application.vrpApp.guiKey2visuKey):
            del Application.vrpApp.guiKey2visuKey[parentKey]
        
        # type depending actions
        if TYPE_PRESENTATION_STEP == typeNr:
            Application.vrpApp.mw.presenterManager.objDeleted( key, parentKey )
        elif TYPE_VIEWPOINT == typeNr:
            Application.vrpApp.mw.viewpointManager.objDeleted( key, parentKey )


    def addObjCB (self, requestNr, typeNr, key, parentKey, params):
        """Set up right initializations when object key comes to existance.

        """
        _infoer.write('addObjCB(requestNr %s, typeNr %s, key %s, parentKey %s)  '
                      'Element has just been created.' % (
            str(requestNr), str(typeNr), str(key), str(parentKey)))
        self.__objects[key] = GuiObj( key, parentKey, typeNr )
        if requestNr>0:
            self.__newObjKey = key
            self.__newObjReqNr = requestNr

        # for compat
        Application.vrpApp.key2type[key] = typeNr

        theGuiMsgHandler().registerAddCallback(key, self.addObjCB)
        theGuiMsgHandler().registerDelCallback(key, self.deleteObjCB)
        theGuiMsgHandler().registerParamCallback(key, self.setParamsCB)
        theGuiMsgHandler().registerBBoxForChildrenCallback(key, self.setBBoxForChildrenCB)

        self.sigGuiObjectAdded.emit(key, parentKey, typeNr )

        # type depending actions
        if TYPE_PROJECT == typeNr:
            self.__projectKey = key
        elif TYPE_CASE == typeNr:
            self.__lastCaseKey = key
        elif TYPE_2D_PART == typeNr or TYPE_CAD_PART == typeNr:
            #move to negotiator    
            reqId = theGuiMsgHandler().requestObject(VIS_2D_RAW, parentKey=key , params=params)
            theGuiMsgHandler().waitforAnswer(reqId)
        elif TYPE_3D_PART == typeNr:
            #move to negotiator
            reqId = theGuiMsgHandler().requestObject(VIS_3D_BOUNDING_BOX, None, key, params=params)
            theGuiMsgHandler().waitforAnswer(reqId)
        elif TYPE_PRESENTATION == typeNr:
            self.__presentationKey = key
            Application.vrpApp.mw.presenterManager.setPresentationMgrKey(key)
        elif TYPE_JOURNAL == typeNr:
            Application.vrpApp.globalJournalMgrKey = key
        elif TYPE_PRESENTATION_STEP == typeNr:
            Application.vrpApp.mw.presenterManager.addStep(key)
        elif VIS_2D_RAW == typeNr:
            Application.vrpApp.visuKey2GuiKey[key] = parentKey
            Application.vrpApp.guiKey2visuKey[parentKey] = key
        elif VIS_3D_BOUNDING_BOX == typeNr:
            Application.vrpApp.visuKey2GuiKey[key] = parentKey
            Application.vrpApp.guiKey2visuKey[parentKey] = key
        elif TYPE_COLOR_TABLE==typeNr:
            Application.vrpApp.mw.globalColorManager().setNewKey(key)
            #VRPMainWindow.globalColorManager.setNewKey(key)  
        elif TYPE_VIEWPOINT_MGR == typeNr:
            self.__viepointMgrKey = key
            Application.vrpApp.mw.viewpointManager.setViewpointMgrKey(key)
        elif TYPE_TRACKING_MGR == typeNr:
            self.__trackingMgrKey = key
            Application.vrpApp.mw.trackingManager.setTrackingMgrKey(key)
        elif VIS_DOMAINLINES==typeNr: 
            theGuiMsgHandler().answerOk(requestNr)
            theGuiMsgHandler().runObject(key)
        theGuiMsgHandler().answerOk(requestNr)


    def getProjectKey(self):
        _infoer.write('getProjectKey : %s ' % str(self.__projectKey) )
        if not self.__projectKey==None: 
            return self.__projectKey

    def getPresentationKey(self):
        _infoer.write('getPresentationKey : %s ' % str(self.__presentationKey) )
        if not self.__presentationKey==None: 
            return self.__presentationKey

    def getViewpointMgrKey(self):
        _infoer.write('getViewpointMgrKey : %s ' % str(self.__viewpointMgrKey) )
        if not self.__viewpointMgrKey==None: 
            return self.__viewpointMgrKey

    def getTrackingMgrKey(self):
        _infoer.write('getTrackingMgrKey : %s ' % str(self.__trackingMgrKey) )
        if not self.__trackingMgrKey==None: 
            return self.__trackingMgrKey

    def getLastCaseKey(self):
        if not self.__lastCaseKey==None:
            return self.__lastCaseKey

    # call from GUI part when you want an object deleted
    def deleteObject(self, key):
        if key in Application.vrpApp.visuKey2GuiKey:
            key = Application.vrpApp.visuKey2GuiKey[key]
        if key in self.__objects:
            parentKey = self.getParentOfObject(key)
            theGuiMsgHandler().requestDelObject(key)
            if (parentKey != -1):
                Application.vrpApp.mw.raisePanelForKey(parentKey)

    def setParams( self, key, params ):
        """ params changed in gui """
        if not key in self.__objects:
            return
        self.__objects[key].params = params
        self.sigGuiParamChanged.emit( key ) # we need this, for example to update the tree checkboxes in case of contextMenu/hideParts
        Application.vrpApp.mw.updatePanel(key)
        theGuiMsgHandler().setParams(key, params)

    def setParamsCB(self, requestNr, key, params):
        """ received params from negotiator """
        _infoer.write('setParamsCB(requestNr %s, key %s, paramName= %s)' % ( str(requestNr), str(key), str(params.name)))

        if not key in self.__objects:
            return

        self.__objects[key].params = params

        self.sigGuiParamChanged.emit(key )
        #for compatibility
        Application.vrpApp.key2params[key] = params
        typeNr = self.__objects[key].type

        if (typeNr in [TYPE_SCENEGRAPH_ITEM, TYPE_COLOR_CREATOR, TYPE_COLOR_MGR, VIS_DOMAINLINES, TYPE_SCENEGRAPH_MGR, TYPE_DNA_MGR, TYPE_DNA_ITEM, TYPE_GENERIC_OBJECT_MGR, TYPE_GENERIC_OBJECT, TYPE_JOURNAL_STEP, TYPE_PROJECT, TYPE_CASE, TYPE_2D_GROUP, TYPE_3D_GROUP, TYPE_2D_CUTGEOMETRY_PART]):
            pass
        elif typeNr in [VIS_STREAMLINE, VIS_MOVING_POINTS, VIS_PATHLINES, VIS_STREAMLINE_2D, VIS_PLANE, VIS_VECTOR, VIS_ISOPLANE, VIS_ISOCUTTER, VIS_CLIPINTERVAL, VIS_VECTORFIELD, VIS_DOMAINSURFACE]: #, VIS_POINTPROBING, VIS_MAGMATRACE
            #Application.vrpApp.mw.raisePanelForKey(key)
            theGuiMsgHandler().answerOk(requestNr)
            self.checkForUpdate( key, params)
        elif TYPE_CAD_PART == typeNr:#TYPE_2D_PART == typeNr or 
            """ CAD Change"
            if hasattr(params, 'featureAngleDefault'):
                VRPMainWindow.tesselationPanel.setParams(key)
                VRPMainWindow.tesselationPanel.setParams(params)
            """
            # vis = VRPMainWindow.globalAccessToTreeView.getItemData(key).isChecked
            vis = False
            paramsForTree = NamedCheckable(params.name, vis)
            Application.vrpApp.key2params[key] = params
            #Application.vrpApp.mw.raisePanelForKey(key)
        elif TYPE_2D_PART == typeNr:
            self.__copyNames.append(params.name)
        elif TYPE_2D_COMPOSED_PART == typeNr:
            if self.__objects[key].params.name == 'Composed.2DPart':
                parentPanelParams = Application.vrpApp.mw.getPanelForKey(self.getParentOfObject(key)).getParams()
                self.__objects[key].params.name = parentPanelParams.name
                self.__objects[key].params.subKeys = parentPanelParams.subKeys
                self.__objects[key].params.definitions = parentPanelParams.definitions
                #VRPMainWindow.globalAccessToTreeView.setItemData(key, parentPanelParams.name)
                Application.vrpApp.mw.globalAccessToTreeView().setItemData(key, parentPanelParams.name)
                theGuiMsgHandler().setParams( key, params )
            #Application.vrpApp.mw.raisePanelForKey(key)
            theGuiMsgHandler().answerOk(requestNr)
            theGuiMsgHandler().runObject(key)
        elif TYPE_3D_PART == typeNr:
            self.__copyNames.append(params.name)
        elif TYPE_3D_COMPOSED_PART == typeNr:
            if self.__objects[key].params.name == 'Composed.3DPart':
                parentPanelParams = Application.vrpApp.mw.getPanelForKey(self.getParentOfObject(key)).getParams()
                self.__objects[key].params.name = parentPanelParams.name
                self.__objects[key].params.subKeys = parentPanelParams.subKeys
                self.__objects[key].params.velDefinitions = parentPanelParams.velDefinitions
                #VRPMainWindow.globalAccessToTreeView.setItemData(key, parentPanelParams.name)
                Application.vrpApp.mw.globalAccessToTreeView().setItemData(key, parentPanelParams.name)
                theGuiMsgHandler().setParams( key, params )
            #Application.vrpApp.mw.raisePanelForKey(key)
            theGuiMsgHandler().answerOk(requestNr)
            theGuiMsgHandler().runObject(key)
        elif TYPE_PRESENTATION == typeNr:
            Application.vrpApp.mw.presenterManager.updateForObject(key)
            #Application.vrpApp.mw.raisePanelForKey(key)
        elif TYPE_JOURNAL == typeNr:
            Application.vrpApp.globalJournalMgrParams = params
            """
            if params.currentIdx==params.maxIdx:
                Application.vrpApp.mw.editRedoAction.setEnabled(False)
                Application.vrpApp.mw.editUndoAction.setEnabled(True)
            else:
                Application.vrpApp.mw.editUndoAction.setEnabled(True)
                Application.vrpApp.mw.editRedoAction.setEnabled(True)
            if params.currentIdx<=0:
                Application.vrpApp.mw.editUndoAction.setEnabled(False)
            """
        elif TYPE_PRESENTATION_STEP == typeNr:
            Application.vrpApp.mw.presenterManager.setParams( key, params )
        elif TYPE_VIEWPOINT == typeNr:
            # do not add the default viewpoints to listview
            if not params.view == None and not params.view == 'default':
                Application.vrpApp.mw.viewpointManager.addViewpoint(key)
            Application.vrpApp.mw.viewpointManager.setParams( key, params )
            #VRPMainWindow.globalAccessToTreeView.setItemSelected(key, True)
        elif TYPE_VIEWPOINT_MGR == typeNr: 
            Application.vrpApp.mw.viewpointManager.updateForObject(key)
        elif TYPE_TRACKING_MGR == typeNr: 
            Application.vrpApp.mw.trackingManager.updateForObject(key)
        elif VIS_2D_RAW == typeNr:
            """ CAD Change
            paramsForTreeView = NamedCheckable(params.name, params.isVisible)
            VRPMainWindow.globalAccessToTreeView.setItemData(key, paramsForTreeView)
            """
            self.refactoringThing02(requestNr, key, params)
            self.checkForUpdate( key, params)
            #need to do this for copieing 2d parts
            self.setParams(key, params)
        elif VIS_3D_BOUNDING_BOX == typeNr:
            self.refactoringThing02(requestNr, key, params)
        elif VIS_COVISE==typeNr or VIS_DOCUMENT==typeNr or TYPE_CAD_PRODUCT==typeNr:
            if not self.newCoviseFileName==None:
                if hasattr( params, 'documentName' ):
                    # check if document exists
                    filename = self.newCoviseFileName
                    if getExistingFilename(filename) == None:
                        raise CoviseFileNotFoundError(filename)

                    params.documentName = os.path.basename(filename)#(self.newCoviseFileName)
                    params.documentName = params.documentName[:len(params.documentName)-4]
                    params.imageName = filename#self.newCoviseFileName
                    params.name = params.documentName
                else :
                    params.filename = self.newCoviseFileName
                    params.name = os.path.basename(self.newCoviseFileName)
                self.newCoviseFileName=None
            if hasattr( params, 'documentName' ):
                #Application.vrpApp.mw.raisePanelForKey(key)
                theGuiMsgHandler().answerOk(requestNr)
                # check if document exists
                filename = params.imageName
                if getExistingFilename(filename) == None:
                    raise CoviseFileNotFoundError(filename)
                params.imageName = filename
            if hasattr( params, 'isVisible'):
                p = NamedCheckable(params.name, params.isVisible)
            else:
                p = params.name
            Application.vrpApp.mw.globalAccessToTreeView().setItemData(key, p)
            theGuiMsgHandler().setParams( key, params )
            theGuiMsgHandler().runObject(key)
        elif VIS_VRML==typeNr:
            if not self.newCoviseFileName==None:
                params.filename = self.newCoviseFileName
                params.name = os.path.basename(self.newCoviseFileName)
                self.newCoviseFileName=None
                theGuiMsgHandler().setParams( key, params )
            p = params.name
            Application.vrpApp.mw.globalAccessToTreeView().setItemData(key, p)
        elif VIS_SCENE_OBJECT==typeNr:
            if not self.newCoviseFileName==None:
                params.filename = self.newCoviseFileName
                params.name = os.path.basename(self.newCoviseFileName)
                self.newCoviseFileName=None
                theGuiMsgHandler().setParams( key, params )
                theGuiMsgHandler().runObject(key)
            p = params.name
            Application.vrpApp.mw.globalAccessToTreeView().setItemData(key, p)
        elif TYPE_COLOR_TABLE==typeNr:
            Application.vrpApp.mw.globalColorManager().setParams( key, params)
        else:
            assert False, 'unknown type'

        if key in Application.vrpApp.guiKey2visuKey:
            visKey = Application.vrpApp.guiKey2visuKey[key]
            self.__objects[visKey].params.name = params.name
            theGuiMsgHandler().setParams( visKey, self.__objects[visKey].params )

        theGuiMsgHandler().answerOk(requestNr)


    def setBBoxForChildrenCB(self, requestNr, key, bbox):
        key = Application.vrpApp.visuKey2GuiKey[key]

        # set BB in OBJMgr
        params = self.__objects[key].params
        params.boundingBox = bbox

        # HACK: doesn't work correctly on trunk

        """
        typeNr = self.__objects[key].type
        if  not (VIS_3D_BOUNDING_BOX == typeNr or VIS_2D_RAW==typeNr):
            return
        self.__objects[key].params.boundingBox = bbox

        for child in self.getChildrenOfObject(key):
            params = self.__objects[child].params
            params.boundingBox = bbox
            if hasattr(params, 'boundingBox'):
                if hasattr(params, 'alignedRectangle'):
                    middle = []
                    middle.append(params.alignedRectangle.middle[0])
                    middle.append(params.alignedRectangle.middle[1])
                    middle.append(params.alignedRectangle.middle[2])
                    if params.alignedRectangle.middle[0] < bbox.getXMin():
                        middle[0] = bbox.getXMin()
                    elif params.alignedRectangle.middle[0] > bbox.getXMax():
                        middle[0] = bbox.getXMax()
                    if params.alignedRectangle.middle[1] < bbox.getYMin():
                        middle[1] = bbox.getYMin()
                    elif params.alignedRectangle.middle[1] > bbox.getYMax():
                        middle[1] = bbox.getYMax()
                    if params.alignedRectangle.middle[2] < bbox.getZMin():
                       middle [2] = bbox.getZMin()
                    elif params.alignedRectangle.middle[2] > bbox.getZMax():
                        middle[2] = bbox.getZMax()
                    params.alignedRectangle.middle = middle
                self.__objects[child].params = params

                self.sigGuiParamChanged.emit( child )
                #for compatibility
                Application.vrpApp.key2params[child].boundingBox = bbox   
                theGuiMsgHandler().setParams( child, params )
                theGuiMsgHandler().answerOk(requestNr)    
                #theGuiMsgHandler().runObject(child)
        """


    def refactoringThing02(self, requestNr, key, params):
        # propagate visibility-state to gui
        class ParamDeliverer(object):
            pass
        paramsForTreeView = ParamDeliverer()
        paramsForTreeView.isChecked = params.isVisible
        Application.vrpApp.mw.globalAccessToTreeView().setItemData(
            Application.vrpApp.visuKey2GuiKey[key], paramsForTreeView)

        theGuiMsgHandler().answerOk(requestNr)
        # Do only once
#        if not key in self.__initHistory:
#            reqid = theGuiMsgHandler().runObject(key)
#            self.__initHistory[key]=reqid

    def checkForUpdate( self, key, params ):
        if hasattr(params, 'variable') and params.variable!='unset' :
            Application.vrpApp.key2params[key] = params
        elif self.__initVariable!=None:
            params.variable = self.__initVariable
            # remark:  init params must have been deposited streamlineInitParams
            theGuiMsgHandler().setParams( key, params )
            theGuiMsgHandler().runObject(key)
            self.__initVariable = None


    def importFile(self, filename ):
        self.newCoviseFileName = filename
        if filename.lower().endswith('.covise'):
            theGuiMsgHandler().requestObject( VIS_COVISE, callback = None, parentKey = self.__projectKey)
        elif filename.lower().endswith('.tif') or filename.lower().endswith('.tiff') or filename.lower().endswith('.png'):
            theGuiMsgHandler().requestObject( VIS_DOCUMENT, callback = None, parentKey = self.__projectKey)
        elif filename.lower().endswith('.coxml') :
            theGuiMsgHandler().requestObject( VIS_SCENE_OBJECT, callback = None, parentKey = self.__projectKey)
        else:
            theGuiMsgHandler().requestObject( VIS_VRML, callback = None, parentKey = self.__projectKey)

    def importCases(self, dscsFineRaw):
        for dsc, rawDsc in zip(*dscsFineRaw):
            reqId = theGuiMsgHandler().requestObject(typeNr = TYPE_CASE, callback = None, parentKey = self.__projectKey)
            theGuiMsgHandler().waitforAnswer(reqId)
            case_key = self.getLastCaseKey() 
            caseP = coCaseMgrParams()
            caseP.filteredDsc = dsc
            caseP.origDsc = rawDsc
            theGuiMsgHandler().setParams(case_key, caseP)
            # start geo reading and octtree building
            reqId = theGuiMsgHandler().runObject(case_key, RUN_OCT)
            theGuiMsgHandler().waitforAnswer(reqId)
            # start all vis items
            reqId = theGuiMsgHandler().runObject(case_key, RUN_ALL)

    def getAllCaseKeys(self):
        caseKeys = []
        for key in self.__objects:
            if self.__objects[key].type==TYPE_CASE:
                caseKeys.append( key )
        return caseKeys

    def getCaseKeyOfObj( self, obj):
        parentKey = obj.parentKey
        while parentKey>0:
            if self.__objects[parentKey].type==TYPE_CASE:
                return parentKey
            else :
                parentKey = self.__objects[parentKey].parentKey
        return None

    def getAll2DElements( self ):
        all2DKeys = []
        for key in self.__objects:
            if self.__objects[key].type==TYPE_2D_PART:
                caseKey = self.getCaseKeyOfObj(self.__objects[key])
                if not caseKey==None:
                    all2DKeys.append( (caseKey, key) )
        return all2DKeys

    def getAllElementsOfType( self , t):
        elems = []
        for key in self.__objects:
            if self.__objects[key].type==t:
                elems.append( self.__objects[key] )
        return elems

    def getAll3DElements( self ):
        all3DKeys = []
        for key in self.__objects:
            if self.__objects[key].type==TYPE_3D_PART:
                caseKey = self.getCaseKeyOfObj(self.__objects[key])
                if not caseKey==None:
                    all3DKeys.append( (caseKey, key) )
        return all3DKeys

    def getList2dPartsForVisItem( self, key):
        partList = []
        caseKeys = self.getAllCaseKeys()
        if len(caseKeys)>1:
            appendCaseName = True
        else :
            appendCaseName = False

        for element in self.getAll2DElements():
            caseKey = element[0]
            objKey = element[1]
            if appendCaseName :
                keydPartName = ( objKey, self.__objects[caseKey].params.name+" - "+self.__objects[objKey].params.name )
            else :
                keydPartName = ( objKey, self.__objects[objKey].params.name )
            partList.append( keydPartName )
        return partList

    def getTypeOfObject( self, key):
        return self.__objects[key].type

    def hasKey(self, key):
        return key in self.__objects

    def getParentOfObject( self, key):
        return self.__objects[key].parentKey

    def getParamsOfObject( self, key):
        if key in self.__objects and self.getTypeOfObject(key) in [TYPE_2D_PART, TYPE_3D_PART, TYPE_2D_CUTGEOMETRY_PART] and key in Application.vrpApp.guiKey2visuKey:
            return self.__objects[Application.vrpApp.guiKey2visuKey[key]].params
        if key in self.__objects:
            return self.__objects[key].params
        else:
            return None

    def getRealParamsOfObject( self, key):
        if key in self.__objects:
            return self.__objects[key].params
        else:
            return None

    def getChildrenOfObject( self, key):
        return Application.vrpApp.mw.globalAccessToTreeView().childKeys(key)

    def __getPartWithVars( self, key ):
        parentKey = self.getParentOfObject(key)
        params = self.getParamsOfObject(parentKey)
        if params == None or hasattr(params, 'subKeys') or not hasattr(params, 'partcase'):
            return None
        return params.partcase

    def getPossibleScalarVariablesForVisItem( self, key ):
        parentKey = self.getParentOfObject(key)
        return self.getPossibleScalarVariablesForType(parentKey)
        #partWithVars = self.__getPartWithVars(key)
        #if type(partWithVars) == type(None):
        #    return []
        #return coviseCase.getScalarVariableNames(partWithVars)

    def getPossibleVectorVariablesForVisItem( self, key ):
        parentKey = self.getParentOfObject(key)
        return self.getPossibleVectorVariablesForType(parentKey)
        #partWithVars = self.__getPartWithVars(key)
        #if type(partWithVars) == type(None):
        #    return []
        #return coviseCase.getVectorVariableNames(partWithVars)

    def getPossibleVariablesForVisItem( self, key ):
        varList = getPossibleScalarVariablesForVisItem(key)
        for item in getPossibleVectorVariablesForVisItem(key):
            varList.append(item)
        return varList

    def getPossibleScalarVariablesForType( self, key ):
        params = self.getRealParamsOfObject(key)
        if hasattr(params, 'partcase') and (params.partcase != None):
            partWithVars = params.partcase
        else :
            return self.getScalarVariablesIntersectionForType(key)
            #return []
        if type(partWithVars) == type(None):
            return []
        return coviseCase.getScalarVariableNames(partWithVars)

    def getPossibleVectorVariablesForType( self, key ):
        params = self.getRealParamsOfObject(key)
        if hasattr(params, 'partcase') and (params.partcase != None):
            partWithVars = params.partcase
        else :
            return self.getVectorVariablesIntersectionForType(key)
            #return []
        if type(partWithVars) == type(None):
            return []
        return coviseCase.getVectorVariableNames(partWithVars)

    def getScalarVariablesIntersectionForType(self, key):
        myObject = globalKeyHandler().getObject(key)
        while (myObject.typeNr == TYPE_2D_CUTGEOMETRY_PART):
            myObject = globalKeyHandler().getObject(myObject.parentKey)
        intersection = []
        if myObject and hasattr(myObject, 'importModule') and hasattr(myObject.importModule, 'getParts'):
            for curImportModule in myObject.importModule.getParts():
                if hasattr(curImportModule, 'getPartCase'):
                    partWithVars = curImportModule.getPartCase()
                    if intersection == []:
                        intersection = set( coviseCase.getScalarVariableNames(partWithVars) )
                    else:
                        intersection = intersection & set( coviseCase.getScalarVariableNames(partWithVars) )
        return list(intersection)

    def getVectorVariablesIntersectionForType(self, key):
        myObject = globalKeyHandler().getObject(key)
        while (myObject.typeNr == TYPE_2D_CUTGEOMETRY_PART):
            myObject = globalKeyHandler().getObject(myObject.parentKey)
        intersection = []
        if myObject and hasattr(myObject, 'importModule') and hasattr(myObject.importModule, 'getParts'):
            for curImportModule in myObject.importModule.getParts():
                if hasattr(curImportModule, 'getPartCase'):
                    partWithVars = curImportModule.getPartCase()
                    if intersection == []:
                        intersection = set( coviseCase.getVectorVariableNames(partWithVars) )
                    else:
                        intersection = intersection & set( coviseCase.getVectorVariableNames(partWithVars) )
        return list(intersection)

    def getNameOfType(self, key):
        return self.__objects[key].params.name


    def getAllChildrenGridsOfObject( self, key):
        children = self.getChildrenOfObject( key)
        KeyGridVectorVarsList = []
        for child in children:
            gridName = self.getNameOfType(child)
            if child in self.__objects:
                vectorVars = self.getPossibleVectorVariablesForType(child)
                if not len(vectorVars) == 0:
                    KeyGridVectorVarsList.append([child, gridName, vectorVars]) 
                if self.getTypeOfObject(child) == TYPE_3D_PART:
                    KeyGridVectorVarsList = KeyGridVectorVarsList + self.getAllChildrenGridsOfObject(child)
        return KeyGridVectorVarsList

    def getAllChildrenPartsOfObject( self, key):
        children = self.getChildrenOfObject( key)
        KeyGridScalarVarsList = []
        for child in children:
            typeName = self.getNameOfType(child)
            if child in self.__objects:
                scalarVars = self.getPossibleScalarVariablesForType(child)
                if not len(scalarVars) == 0:
                    KeyGridScalarVarsList.append([child, typeName, scalarVars]) 
                if self.getTypeOfObject(child) == TYPE_2D_PART:
                    KeyGridScalarVarsList = KeyGridScalarVarsList + self.getAllChildrenPartsOfObject(child)
        return KeyGridScalarVarsList


_theObjMgr = None
def ObjectMgr():
    """Access the singleton."""

    global _theObjMgr
    if None == _theObjMgr: _theObjMgr = _ObjMgr()
    return _theObjMgr

class GuiObj(object):
    """ Store information about KeydObjects of the negotiator """
    def __init__(self, objKey, parentKey, objType):
        self.key = objKey
        self.parentKey = parentKey
        self.type = objType
        self.params = None

# eof
