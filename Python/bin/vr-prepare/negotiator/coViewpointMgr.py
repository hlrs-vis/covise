# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH


from KeydObject import globalKeyHandler, globalViewpointMgrKey, globalPresentationMgrKey, TYPE_VIEWPOINT, TYPE_VIEWPOINT_MGR, RUN_ALL
from Utils import CopyParams, ParamsDiff, mergeGivenParams
from VisItem import VisItem, VisItemParams
from KeydObject import NO_PARENT, coKeydObject
from coGRMsg import coGRShowViewpointMsg, coGRCreateViewpointMsg, coGRToggleFlymodeMsg, coGRDeleteViewpointMsg, coGRKeyWordMsg, coGRToggleVPClipPlaneModeMsg, coGRChangeViewpointNameMsg
import covise

from printing import InfoPrintCapable
_infoer = InfoPrintCapable()
_infoer.doPrint =  False #True #

VIEWPOINT_ID_STRING = 'VIEWPOINT'

class coViewpointMgr(VisItem):
    """ class to handle project files """
    def __init__(self):
        VisItem.__init__(self, TYPE_VIEWPOINT_MGR, 'ViewpointMgr')
        self.params = coViewpointMgrParams()
        self.name = self.params.name
        
        # tell cover to send the default viewpoints
        msg = coGRKeyWordMsg( "sendDefaultViewPoint" , True)
        covise.sendRendMsg(msg.c_str())


    def registerCOVISEkey( self, covise_key):
        """ called during registration if key received from COVER """
        _infoer.function = str(self.registerCOVISEkey)
        _infoer.write("%s" % covise_key)
        return (covise_key==VIEWPOINT_ID_STRING, False)

    def run(self, runmode, negMsgHandler=None):
        if runmode==RUN_ALL:
            _infoer.function = str(self.run) 
            _infoer.write("go coViewpointMgr")

            VisItem.run(self, runmode, negMsgHandler)
            self.sendFlyingMode()
            self.sendClipplaneMode()

    def setParams( self, params, negMsgHandler=None, sendToCover=True):
        _infoer.function = str(self.setParams)
        realChange = ParamsDiff( self.params, params )

        if 'newViewpoint' in realChange :
            _infoer.write("new viewpoint")
            newVp = negMsgHandler.internalRequestObject(TYPE_VIEWPOINT, self.key)
            newVp.params = coViewpointParams()
            newVp.params.id   = params.newViewpoint[0]
            newVp.params.name = params.newViewpoint[1]
            newVp.params.view = params.newViewpoint[2]
            newVp.params.clipplane = params.newViewpoint[3]
            newVp.params.confirmed = True
            newVp.params.isVisible = True
            self.makeOthersInvisible(newVp, negMsgHandler)
            negMsgHandler.sendParams( newVp.key, newVp.params )

        elif 'newDefaultViewpoint' in realChange or (not hasattr(self.params, 'newDefaultViewpoint') and (hasattr(params, 'newDefaultViewpoint') and not params.newDefaultViewpoint == None)):
            _infoer.write("new default viewpoint")
            newVp = negMsgHandler.internalRequestObject(TYPE_VIEWPOINT, self.key)
            newVp.params = coViewpointParams()
            newVp.params.id   = params.newDefaultViewpoint[0]
            newVp.params.name = params.newDefaultViewpoint[1]
            newVp.params.view = "default"
            newVp.params.clipplane = "0"
            newVp.params.confirmed = True
            negMsgHandler.sendParams( newVp.key, newVp.params )

        elif 'changedViewpoint' in realChange :
            _infoer.write("change viewpoint")
            vpId   = params.changedViewpoint[0]
            vpName = params.changedViewpoint[1]
            for obj in self.objects:
                if obj.params.id == vpId and obj.params.name == vpName:
                    p = CopyParams(obj.params)
                    p.view = params.changedViewpoint[2]
                    p.changed = True
                    if negMsgHandler:
                        negMsgHandler.internalRecvParams( obj.key, p )
                        negMsgHandler.sendParams( obj.key, p )
                    break

        elif ('changeID' in realChange) or (hasattr(params, 'changeID') and params.changeID):
            _infoer.write("change id")
            for obj in self.objects:
                if obj.params.id == params.oldID and obj.params.confirmed==False:
                    p = CopyParams(obj.params)
                    p.id = params.newID
                    p.confirmed = True
                    if negMsgHandler:
                        negMsgHandler.internalRecvParams( obj.key, p )
                        negMsgHandler.sendParams( obj.key, p )
            self.params.changedID = False

        elif 'flyingMode' in realChange:
            _infoer.write("change flying Mode")
            self.params.flyingMode = params.flyingMode
            self.sendFlyingMode()

        #elif 'clipplaneMode' in realChange:
        #    _infoer.write("change clipplane Mode")
        #    self.params.clipplaneMode = params.clipplaneMode
        #    self.sendClipplaneMode()
        self.params.clipplaneMode = True
        self.sendClipplaneMode()

        if 'selectedKey' in realChange:
            _infoer.write("selected new viewpoint")
            for obj in self.objects:
                if obj.params.id != params.selectedKey:
                    p = CopyParams(obj.params)
                    p.isVisible=False
                    if negMsgHandler:
                        negMsgHandler.internalRecvParams( obj.key, p )
                        negMsgHandler.sendParams( obj.key, p )
                else:
                    p = CopyParams(obj.params)
                    p.isVisible=True
                    self.params.currentKey = params.selectedKey
                    if negMsgHandler:
                        negMsgHandler.internalRecvParams( obj.key, p )
                        negMsgHandler.sendParams( obj.key, p )


    def recreate(self, negMsgHandler, parentKey, offset):
        """ recreate is called after all classes of the session have been unpickled """
        coViewpointMgrParams.mergeDefaultParams(self.params) # explicitly call mergeDefaultParams of this class
        if offset>0 :
            for obj in self.objects:
                obj.params.name = str(self.key) + "_" + obj.params.name

        # tell cover to send the default viewpoints again
        msg = coGRKeyWordMsg( "sendDefaultViewPoint" , True)
        covise.sendRendMsg(msg.c_str())

        VisItem.recreate(self, negMsgHandler, parentKey, offset)
        self.params.selectedKey  = None

        if offset>0 :
            globalKeyHandler().getObject(globalViewpointMgrKey).merge(self)
            #return

        for obj in self.objects:
            if obj.params.isVisible:
                obj.show()
                self.params.currentKey = obj.key
                return

    def reconnect(self):
        """ recreate is called after all classes of the session have been unpickled """
        self.sendFlyingMode()
        self.sendClipplaneMode()


    def makeOthersInvisible( self, visobj, negMsgHandler=None):
        for obj in self.objects:
            if obj!=visobj:
                p = CopyParams(obj.params)
                p.isVisible=False
                if negMsgHandler:
                    negMsgHandler.internalRecvParams( obj.key, p )
                    negMsgHandler.sendParams( obj.key, p )
            else:
                p = CopyParams(obj.params)
                p.isVisible=True
                if negMsgHandler:
                    negMsgHandler.internalRecvParams( obj.key, p )
                    negMsgHandler.sendParams( obj.key, p )

    def sendFlyingMode(self ):
        if hasattr(self.params, 'flyingMode'):
            msg = coGRToggleFlymodeMsg( self.params.flyingMode )
            covise.sendRendMsg(msg.c_str())

    def sendClipplaneMode(self ):
        if hasattr(self.params, 'clipplaneMode'):
            msg = coGRToggleVPClipPlaneModeMsg( True )
            covise.sendRendMsg(msg.c_str())

class coViewpointMgrParams(VisItemParams):
    def __init__(self):
        self.name         = 'ViewpointMgrParams'
        self.flyingMode      = False
        self.newViewpoint = None
        self.newDefaultViewpoint = None
        self.changedViewpoint = None
        self.currentKey   = -1
        self.selectedKey  = None
        self.flyingMode   = True
        self.changeID = False
        self.oldID = None
        self.newID = None
        coViewpointMgrParams.mergeDefaultParams(self) # explicitly call mergeDefaultParams of this class

    def mergeDefaultParams(self):
        defaultParams = {
            'clipplaneMode' : True,
            'changedViewpoint' : None
        }
        mergeGivenParams(self, defaultParams)

class coViewpoint(VisItem):
    """ class to handle viewpoints """
    def __init__(self):
        VisItem.__init__(self, TYPE_VIEWPOINT, 'Viewpoint')
        self.params = coViewpointParams()

    def setParams( self, params, negMsgHandler=None, sendToCover=True):
        _infoer.function = str(self.setParams)
        _infoer.write("%s" % params.isVisible)
        if params.id == None:
            return
        realChange = ParamsDiff( self.params, params )
        oldID = self.params.id
        if hasattr(self.params, 'confirmed'):
            oldConfirmed = self.params.confirmed
        else:
            oldConfirmed = True
        if hasattr(self.params, 'changed'):
            oldChanged = self.params.changed
        else:
            oldChanged = False
        VisItem.setParams(self, params, negMsgHandler, sendToCover)
        self.params.confirmed = oldConfirmed
        self.params.changed = oldChanged

        # confirmed and id in realChange happens when loading a project and having different default viewpoints
        if 'confirmed' in realChange and 'id' in realChange:
            self.params.confirmed = True
            if self.params.isVisible:
                self.show()
            # tell the coPresentationMgr that viewpoint is now confirmed and the id has changed
            globalKeyHandler().getObject(globalPresentationMgrKey).changeUnconfirmedViewpointID(oldID, params.id, negMsgHandler)
        else:
            if 'confirmed' in realChange:
                self.params.confirmed = True
                if self.params.isVisible:
                    self.show()
                # tell the coPresentationMgr that viewpoint is confirmed
                globalKeyHandler().getObject(globalPresentationMgrKey).setViewpointConfirmed(params.id, negMsgHandler)
            if 'id' in realChange:
                # tell the presenterManager that my id has changed
                globalKeyHandler().getObject(globalPresentationMgrKey).changeViewPointID(oldID, params.id, negMsgHandler)
        if hasattr(self.params, 'delete'):
            self.delete(False, negMsgHandler)
            return
        if 'isVisible' in realChange and sendToCover and params.isVisible and hasattr(self.params, 'confirmed') and self.params.confirmed:
            _infoer.function = str(self.setParams)
            _infoer.write("send viewpoint")
            self.show()
            globalKeyHandler().getObject(self.parentKey).makeOthersInvisible(self, negMsgHandler)
        if 'name' in realChange and sendToCover:
            #send new name to cover
            msg = coGRChangeViewpointNameMsg( self.params.id, params.name )
            covise.sendRendMsg(msg.c_str())

        # changed in realChange happens when viewpoint is changed in gui
        if 'changed' in realChange:
            # tell the coPresentationMgr that obj has changed
            globalKeyHandler().getObject(globalPresentationMgrKey).viewpointChanged(self.params.id, self.params.view, negMsgHandler)

    def show( self ):
        if hasattr(self.params, 'confirmed') and self.params.confirmed:
            print("send show vp", self.params.id)
            msg = coGRShowViewpointMsg( self.params.id )
            covise.sendRendMsg(msg.c_str())

    def delete(self, isInitialized, negMsgHandler=None):
        if hasattr(self.params, 'confirmed') and self.params.confirmed == False:
            self.params.delete = True
            return
        if isInitialized:
            if self.params.view != 'default':
                msg = coGRDeleteViewpointMsg( self.params.id )
                covise.sendRendMsg(msg.c_str())

#        VisItem.delete(self, isInitialized, negMsgHandler)
        return coKeydObject.delete(self, isInitialized, negMsgHandler)


    def recreate(self, negMsgHandler, parentKey, offset):
        """ recreate is called after all classes of the session have been unpickled """
        coViewpointParams.mergeDefaultParams(self.params) # explicitly call mergeDefaultParams of this class
        #do not recreate default viewpoints
        if self.params.view == 'default' or self.params.id == None:
            self.delete(False, negMsgHandler)
            return
        VisItem.recreate(self, negMsgHandler, parentKey, offset)
        self.params.confirmed = False
        self.params.changed = False
        # the id is not used by cover
        # could be out of sync
        msg = coGRCreateViewpointMsg( self.params.name, self.params.id, self.params.view, self.params.clipplane )
        covise.sendRendMsg(msg.c_str())

    def reconnect(self):
        """ recreate is called after all classes of the session have been unpickled """
        #do not recreate default viewpoints
        if self.params.view == 'default' or self.params.id == None:
            return
        self.params.confirmed = False
        self.params.changed = False
        # the id is not used by cover
        # could be out of sync
        msg = coGRCreateViewpointMsg( self.params.name, self.params.id, self.params.view, self.params.clipplane  )
        covise.sendRendMsg(msg.c_str())


class coViewpointParams(VisItemParams):
    def __init__(self):
        VisItemParams.__init__(self)
        self.name    = 'Viewpoint'
        self.view    = None
        self.id      = None
        self.confirmed = True # viewpoint id need to be confirmed from cover
        self.changed = False
        coViewpointParams.mergeDefaultParams(self) # explicitly call mergeDefaultParams of this class

    def mergeDefaultParams(self):
        defaultParams = {
                'clipplane' : "0 ",
        }
        mergeGivenParams(self, defaultParams)

    # TODO: Most if not all of the params should be static.
    #       That way, changeUnconfirmedViewpointID, setViewpointConfirmed and viewpointChanged in coPresentationMgr might become obsolete.
    #       Life would be easier.
    #       We could also exclude ViewPoints from the PresentationManager entirely.
    #def isStaticParam(self, paramname):
    #    return paramname in ["name", "view", "id", "confirmed", "changed", "clipplane"]

