
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH

from KeydObject import coKeydObject, RUN_ALL, globalPresentationMgrKey, globalKeyHandler, globalProjectKey, TYPE_PRESENTATION, TYPE_PRESENTATION_STEP
from Utils import CopyParams, ParamsDiff, mergeGivenParams
from VisItem import VisItem
import PartCuttingSurfaceVis
import coCaseMgr, coColorTable, coProjectMgr, coTrackingMgr, coGenericObjectMgr, coDNAMgr
import os
import copy
import covise
from KeydObject import VIS_DOCUMENT, TYPE_TRACKING_MGR
from coGRMsg import coGRKeyWordMsg, coGRChangeViewpointMsg

# needed to correct wrong freeStartPoints
import PartTracerVis

STEP_PARAM = 0
STEP_ADD   = 1
STEP_DEL   = 2

from printing import InfoPrintCapable
_infoer = InfoPrintCapable()
_infoer.doPrint = False #True #

"""
   Presentation Mgr cares for the presentation steps
   the list 'objects' saves the single Presentationsteps
"""

def objIsRelevant(obj):
    # "isVisible" covers all objects inheriting from VisItem (including ViewPointMgr and ViewPoint)
    return hasattr( obj.params, 'isVisible' ) \
           or isinstance( obj, coCaseMgr.coCaseMgr ) \
           or isinstance( obj, coColorTable.coColorTable ) \
           or isinstance( obj, coProjectMgr.coProjectMgr ) \
           or isinstance( obj, coTrackingMgr.coTrackingMgr ) \
           or isinstance( obj, coGenericObjectMgr.coGenericObjectMgr ) \
           or isinstance( obj, coGenericObjectMgr.coGenericObject )


class coPresentationMgr(coKeydObject):
    """ class to handle project files """
    def __init__(self):
        coKeydObject.__init__(self, TYPE_PRESENTATION, 'Presentation')
        self.params = coPresentationMgrParams()
        self.name = self.params.name
        # children build history of params: coPresentationStep

    def addObject( self, obj ):
        _infoer.function = str(self.addObject)
        _infoer.write("name: %s" % obj.params.name)
        coKeydObject.addObject(self,obj)

    def numSteps(self):
        return len(self.objects)

    def recreate(self, negMsgHandler, parentKey, offset):
        coPresentationMgrParams.mergeDefaultParams(self.params) # explicitly call mergeDefaultParams of this class
        coKeydObject.recreate(self, negMsgHandler, parentKey, offset)
        if offset>0 :
            globalKeyHandler().getObject(globalPresentationMgrKey).merge(self)

    def setParams(self, params, negMsgHandler=None, sendToCover=False):
        _infoer.function = str(self.setParams)
        _infoer.write(" ")

        # change of maxIdx means shorten the history
        oldParams = self.params
        changedParams = ParamsDiff( self.params, params)

        if hasattr(params, 'currentStep'):
            diffStep = params.currentStep - oldParams.currentStep
        else:
            diffStep = params.currentKey - oldParams.currentKey

        coKeydObject.setParams( self, params )
        if 'currentKey' in changedParams or 'reloadStep' in changedParams:
            if os.getenv('VR_PREPARE_DEBUG_VISITEMS_DIR'):
                for key in globalKeyHandler().getAllElements():
                    if globalKeyHandler().getObject(key) and isinstance(globalKeyHandler().getObject(key), VisItem):
                        globalKeyHandler().getObject(key).updateDebugFilename(self.params.currentKey)

            # send message presentationstep changed
            if diffStep == 1:
                msg = coGRKeyWordMsg("presForward", True)
                covise.sendRendMsg(msg.c_str())
            elif diffStep ==  -1:
                msg = coGRKeyWordMsg("presBackward", True)
                covise.sendRendMsg(msg.c_str())
            elif 'reloadStep' in changedParams:
                msg = coGRKeyWordMsg("presReload", True)
                covise.sendRendMsg(msg.c_str())
            else:
                msg_str = "goToStep "+str(params.currentStep)
                msg = coGRKeyWordMsg(msg_str, True)
                covise.sendRendMsg(msg.c_str())

            if negMsgHandler:
                #if hasattr(globalKeyHandler().getObject(self.params.currentKey).params, 'status'):
                key2stateParam  = globalKeyHandler().getObject(self.params.currentKey).params.status
                project = globalKeyHandler().getObject(0)
                keysInProject = []
                keysInProject.append(globalProjectKey)
                self.__addToList(project, keysInProject)
                orderedKeysInProject1 = []
                orderedKeysInProject2 = []
                for key in keysInProject: # put some objects at the beginning of the list
                    obj = globalKeyHandler().getObject(key)
                    #check if visItem is readyToChange otherwise send message
                    if diffStep > 0 and hasattr(obj.params, 'nextPresStep') and not obj.params.nextPresStep and covise.coConfigIsOn("vr-prepare.SolvePresentationStep"):
                        #roll back changes
                        negMsgHandler.sendParams(globalPresentationMgrKey, oldParams)
                        msg = coGRKeyWordMsg("showNotReady", True)
                        covise.sendRendMsg(msg.c_str())
                        return
                    if (obj.typeNr in [VIS_DOCUMENT, TYPE_TRACKING_MGR]):
                        orderedKeysInProject1.append(key)
                    else:
                        orderedKeysInProject2.append(key)
                orderedKeysInProject1.extend(orderedKeysInProject2)
                for key in orderedKeysInProject1:
                    if key in key2stateParam:
                        params = key2stateParam[key]
                        if not hasattr(params, 'flyingMode'): # do not save settings of viewpointMgr
                            obj = globalKeyHandler().getObject(key)
                            newparams = CopyParams(obj.getParams()) # take the objects params as base (so we have all the merged defaultParams)
                            paramChanged = False
                            for pkey in params.__dict__:
                                if pkey in newparams.__dict__:
                                    if covise.coConfigIsOn("vr-prepare.DoNotUpdateCuttingSurfaces", False) \
                                       and isinstance(obj, PartCuttingSurfaceVis. PartCuttingSurfaceVis) and pkey=='isVisible':
                                        pass
                                    elif hasattr(newparams, "isStaticParam") and newparams.isStaticParam(pkey):
                                        # skip static params
                                        pass
                                    elif pkey=='actTimeStep' and (not hasattr(params, 'version') or params.__dict__['version'] < 7):
                                        # change actual timestep for old 6.0 projects
                                        newparams.__dict__[pkey] = params.__dict__[pkey] -1
                                        if newparams.__dict__[pkey] < 0:
                                            newparams.__dict__[pkey] = newparams.__dict__['numTimeSteps'] - 1
                                        paramChanged = True
                                    elif (pkey=='autoActiveSensorIDs') and (len(params.autoActiveSensorIDs) > 0):
                                        # always set params if the new step has autoActiveSensorIDs
                                        newparams.__dict__[pkey] = copy.deepcopy(params.__dict__[pkey])
                                        paramChanged = True
                                    else:
                                        if (newparams.__dict__[pkey] != params.__dict__[pkey]):
                                            newparams.__dict__[pkey] = copy.deepcopy(params.__dict__[pkey]) # need a deepcopy in case we have a list/dict
                                            paramChanged = True
                            if (paramChanged):
                                if key != globalProjectKey and 'currentKey' in changedParams:
                                    negMsgHandler.presentationRecvParams( key, newparams )
                                    negMsgHandler.sendParams(key, newparams)
                                elif key != globalProjectKey:
                                    negMsgHandler.presentationRecvParams( key, newparams, True) #TODO, ueberpruefen, ob das nach kompletter portierung noetig ist
                                    negMsgHandler.sendParams(key, newparams)
                                else :
                                    negMsgHandler.sendParams(key, newparams)
                                    project.setParams(newparams)
                                    project.sendMessages()

                    # hide all visItem which are not in list
                    else:
                        params =  globalKeyHandler().getObject(key).params
                        if params:
                            #do not save settings of viewpointMgr
                            if not hasattr(params, 'flyingMode'):
                                if hasattr(params, 'isVisible') and params.isVisible:
                                    cparams = CopyParams(params)
                                    cparams.isVisible = False
                                    negMsgHandler.presentationRecvParams( key, cparams)
                                    negMsgHandler.sendParams(key, cparams)


    def __addToList(self, obj, listOfKeys):
        for o in obj.objects:
            if objIsRelevant(o):
                listOfKeys.append(o.key)
            if o.objects:
                self.__addToList(o, listOfKeys)

    #TODO: see coViewpointParams.isStaticParam
    def changeViewPointID(self, oldID, newID, negMsgHandler=None):
        for obj in self.objects:
            obj.changePresentationStepID(oldID, newID,negMsgHandler )

    # searches unconfirmed viewpoints, changes the id and makes them confirmed
    #TODO: see coViewpointParams.isStaticParam
    def changeUnconfirmedViewpointID(self, oldID, newID, negMsgHandler):
        for obj in self.objects:
            obj.changeUnconfirmedViewpointID(oldID, newID, negMsgHandler)

    #TODO: see coViewpointParams.isStaticParam
    def setViewpointConfirmed(self, viewpointID, negMsgHandler):
        for obj in self.objects:
            obj.setViewpointConfirmed(viewpointID, negMsgHandler)

    #TODO: see coViewpointParams.isStaticParam
    def viewpointChanged(self, vpID, view, negMsgHandler=None):
        for obj in self.objects:
            obj.viewpointChanged(vpID, view, negMsgHandler)

    """ change current presenttionstep from Gui
    """
    def change(self):
        obj = self.objects[self.params.currentStep]
        obj.change()

    # changes the documents values in presentation steps
    def documentChanged(self, documentName, pos, isVisible, scaling, size, negMsgHandler):
        #change document values for all steps
        for obj in self.objects:
            obj.documentChanged(documentName, pos, isVisible, scaling, size, negMsgHandler)

    def removeKey(self, key):
        for obj in self.objects:
            obj.removeKey(key)

class coPresentationMgrParams(object):
    def __init__(self):
        self.name       = 'Presentation'
        self.currentKey = 0
        coPresentationMgrParams.mergeDefaultParams(self) # explicitly call mergeDefaultParams of this class

    def mergeDefaultParams(self):
        defaultParams = {
                'reloadStep' : False,
                'currentStep' : 0
            }
        mergeGivenParams(self, defaultParams)

"""
   coPresentationStep keeps the parameters for a step
   within the dictionary status the parameters for the step are saved. {key: MgrParams, key: MgrParams,...}
"""
class coPresentationStep(coKeydObject):
    """ class to handle session history
        TODO handle add/del of objects """
    def __init__(self):
        coKeydObject.__init__(self, TYPE_PRESENTATION_STEP, 'Presentation')
        self.params = coPresentationStepParams()
        # walk throgh all keys (assuming own key is last in list)
        for key in range(self.key):
            obj = globalKeyHandler().getObject(key)
            if obj:
                if objIsRelevant(obj):
                    self.params.status[key] = CopyParams(obj.params)

    """
       change this presentation step
           walk trough all parameters (even the ones which where not in this step before)
           set them to the values they have right now
           changing the viewpoint for all steps
    """
    def change(self):

        # for change viewpoint send msg to COVER with ViewpointID
        # get COVER id of viewpoint for message
        id = -1
        for key, param in self.params.status.items():
            # viewpoints have parameter 'view', the viewpoint connected to this step is visible
            if hasattr(param, 'view') and hasattr(param, 'isVisible') and param.isVisible == True:
                id = param.id
        # send ChangeViewpointMsg
        if id != -1:
            msg = coGRChangeViewpointMsg( id )
            covise.sendRendMsg(msg.c_str())

        # find the number of all keys
        numkeys = len(globalKeyHandler().getAllElements())
        if numkeys <1: # if that goes wrong, use only the keys in this step
            numkeys = self.key

        # walk through all keys, even if they are not allready in the step
        for key in range(numkeys):
            obj = globalKeyHandler().getObject(key)
            if obj:
                if not hasattr(obj.params, 'view') and objIsRelevant(obj):
                    self.params.status[key] = CopyParams(obj.params)

    def recreate(self, negMsgHandler, parentKey, offset):
        coPresentationStepParams.mergeDefaultParams(self.params) # explicitly call mergeDefaultParams of this class
        self.__correctParams()
        coKeydObject.recreate(self, negMsgHandler, parentKey, offset)
        if offset>0:
            # change keys in params.status according to offset (i.e. add the offset)
            self.params.status = dict([(key+offset, value) for (key, value) in self.params.status.items()])

        # recreated presentation steps have unconfirmed viewpoints
        for k in self.params.status:
            params = self.params.status[k]
            if hasattr(params, 'view'):
                params.confirmed = False
        if self.params.timeout == 0:
            self.params.timeout = 10

        #self.params.changed = False

    def setParams(self, params, negMsgHandler=None, sendToCover=False):
        _infoer.function = str(self.setParams)
        _infoer.write(" ")
        if hasattr(params, "status") and (params.status == None): # if we get params back from the GUI part, status is always None for performance reasons
            params.status = self.params.status
        coKeydObject.setParams( self, params )

    # corrects wrong parameters in a presentation
    # must only be called in recreate after all parameters are unpickled and merged
    # currently only turns freeStartPoints that are tuples into strings
    def __correctParams(self):
        for (key, value) in iter(self.params.status.items()):
            if isinstance(value, PartTracerVis.PartTracerVisParams) and hasattr(value, "freeStartPoints") and type(value.freeStartPoints) == tuple:
                self.params.status[key].freeStartPoints = '[0.01, 0.01, 0.01]'

    def changePresentationStepID(self, oldID, newID, negMsgHandler):
        for k in self.params.status:
            params = self.params.status[k]
            if hasattr(params, 'view'):
                if params.id == oldID:
                    params.id = newID
                    if negMsgHandler:
                        negMsgHandler.internalRecvParams(self.key, self.params)
                        negMsgHandler.sendParams(self.key, self.params)

    # searches unconfirmed viewpoints, changes the id and makes them confirmed
    def changeUnconfirmedViewpointID(self, oldID, newID, negMsgHandler):
        for k in self.params.status:
            params = self.params.status[k]
            if hasattr(params, 'view'):
                if params.id == oldID and params.confirmed == False:
                    params.id = newID
                    params.confirmed = True
                    if negMsgHandler:
                        negMsgHandler.internalRecvParams(self.key, self.params)
                        negMsgHandler.sendParams(self.key, self.params)

    def setViewpointConfirmed(self, viewpointID, negMsgHandler):
        for k in self.params.status:
            params = self.params.status[k]
            if hasattr(params, 'view'):
                if params.id == viewpointID:
                    params.confirmed = True
                    if negMsgHandler:
                        negMsgHandler.internalRecvParams(self.key, self.params)
                        negMsgHandler.sendParams(self.key, self.params)

    # changes the viewpoint view of all viewpoints with vpID in list for all presentation steps
    def viewpointChanged(self, vpID, view, negMsgHandler):
        for k in self.params.status:
            params = self.params.status[k]
            if hasattr(params, 'view'):
                if params.id == vpID:
                    params.view = view
                    if negMsgHandler:
                        negMsgHandler.internalRecvParams(self.key, self.params)
                        negMsgHandler.sendParams(self.key, self.params)

    # changes the documents values of all documents with documentname in list (not only in the current step) for all presentation steps
    def documentChanged(self, documentName, pos, isVisible, scaling, size, negMsgHandler):
        for k in self.params.status:
            params = self.params.status[k]
            if hasattr(params, 'documentName'):
                if params.documentName == documentName:
                    params.pos = pos
                    params.isVisible = isVisible
                    params.scaling = scaling
                    params.size = size
        ###print "[documentChanged]", self.params.status[3].__dict__

    def removeKey(self, key):
        if key in self.params.status:
            del self.params.status[key]

class coPresentationStepParams(object):
    def __init__(self):
        self.name    = 'PresentationStep'
        self.status  = { }
        self.timeout = 10
        #self.nameChanged = False
        self.index = -1
        #self.changed = False
        coPresentationStepParams.mergeDefaultParams(self) # explicitly call mergeDefaultParams of this class

    def mergeDefaultParams(self):
        defaultParams = {
            'nameChanged' : False
        }
        mergeGivenParams(self, defaultParams)
