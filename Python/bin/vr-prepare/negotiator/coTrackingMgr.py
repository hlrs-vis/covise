
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH


from KeydObject import coKeydObject, RUN_ALL, globalKeyHandler, TYPE_TRACKING_MGR, globalKeyHandler, globalPresentationMgrKey
from Utils import ParamsDiff, mergeGivenParams
from coGRMsg import coGRSetTrackingParamsMsg, coGRObjMoveObjMsg, coGRMsg, coGRObjSensorEventMsg
import covise
#import copy

# warning: not all Qt-Objects can be pickled and must excluded beforehand

from PyQt5 import QtCore, QtGui

NO_NAV = 0
SHOW_NAME = 1
HAS_SOUND = False

import sys
if (sys.platform == "win32"):
    HAS_SOUND = True
    import winsound

class VRCInput(object):
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0
        self.button = 0

class coTrackingMgr(coKeydObject):
    """ class to handle project files """
    def __init__(self):
        coKeydObject.__init__(self, TYPE_TRACKING_MGR, 'Tracking Manager')
        globalKeyHandler().globalTrackingMgrKey = self.key
        self.params = coTrackingMgrParams()
        self.name = self.params.name

        self.__readButtonConfig()
        self.__setupTimeout()
        self.sendParamsToCover()

    """
    remove Qt objects etc. before pickling
    """
    def __getstate__(self):
        #myContent = copy.copy(self.__dict__)
        myContent = coKeydObject.__getstate__(self)
        del myContent['_coTrackingMgr__1sTimer']
        del myContent['_coTrackingMgr__negMsgHandler']
        del myContent['_coTrackingMgr__oldVRCInput']
        return myContent

    """
    create timer to check every second for an idle VRC input
    """
    def __setupTimeout(self):
        self.__1sTimer = QtCore.QTimer()
        self.__idleTimeElapsed = 0        # elapsed idle time in seconds
        self.__oldVRCInput = VRCInput()
        self.__negMsgHandler = None        # store a reference to the negotiator to send keywords

        self.__1sTimer.timeout.connect(self.__1sTimerTimeout)

        VRCTimeout = covise.getCoConfigEntry("vr-prepare.TrackingManager.VRCTimeout")
        if VRCTimeout and int(VRCTimeout) != 0:
            self.__VRCTimeout = int(VRCTimeout)
            self.__1sTimer.start(1000)

    """
    increase idle time steadily. gets reset by setVRC(self)
    """
    def __1sTimerTimeout(self):
        self.__idleTimeElapsed = self.__idleTimeElapsed + 1

        if self.__idleTimeElapsed == self.__VRCTimeout:
            print("Idle VRC input for " + str(self.__VRCTimeout) + " seconds. Resetting Presentation...")
            # send KeyWordMsg to cover
            if self.__negMsgHandler != None:
                self.__negMsgHandler.sendKeyWord("PRESENTATION_GO_TO_START")

    """ read button config from config.vr-prepare.xml, must be done for init AND unpickle """    
    def __readButtonConfig(self):
        # read buttons from config
        buttonList = covise.getCoConfigSubEntries("vr-prepare.TrackingManager.ButtonMap")
        completeButtonList = ['TRANS_LEFT', 'TRANS_RIGHT', 'TRANS_UP', 'TRANS_DOWN', 'TRANS_FRONT', 'TRANS_BACK', \
                           'ROT_X_MINUS', 'ROT_X_PLUS', 'ROT_Y_MINUS', 'ROT_Y_PLUS', 'ROT_Z_MINUS', 'ROT_Z_PLUS', \
                           'SCALE_PLUS', 'SCALE_MINUS']

        for b in buttonList:
            if b in completeButtonList:
                self.__dict__[b] = int(covise.getCoConfigEntry("vr-prepare.TrackingManager.ButtonMap." + b, "button"))
        
        # if not all buttons are described in config.vr-prepare.xml
        for b in completeButtonList:
            if not b in self.__dict__.keys():
                self.__dict__[b] = -1

        TranslateBarrier = covise.getCoConfigEntry("vr-prepare.TrackingManager.TranslateBarrier")
        if TranslateBarrier:
            self.TranslateBarrier = float(TranslateBarrier)
        else:
            self.TranslateBarrier = 1.0

        self.VRCJoystick = covise.coConfigIsOn("vr-prepare.TrackingManager.VRCJoystick", False)

        # buttons for vrml sensors
        ButtonSensor1 = covise.getCoConfigEntry("vr-prepare.TrackingManager.ButtonSensor1")
        ButtonSensor2 = covise.getCoConfigEntry("vr-prepare.TrackingManager.ButtonSensor2")
        if ButtonSensor1:
            self.BUTTON_SENSOR_1 = int(ButtonSensor1)
        else:
            self.BUTTON_SENSOR_1 = -1
        if ButtonSensor2:
            self.BUTTON_SENSOR_2 = int(ButtonSensor2)
        else:
            self.BUTTON_SENSOR_2 = -1

        WrlName = covise.getCoConfigEntry("vr-prepare.TrackingManager.WrlName")
        if WrlName:
            self.wrl_name = WrlName
        else:
            self.wrl_name = ""

    def recreate(self, negMsgHandler, parentKey, offset):
        coTrackingMgrParams.mergeDefaultParams(self.params) # explicitly call mergeDefaultParams of this class
        coKeydObject.recreate(self, negMsgHandler, parentKey, offset)
        globalKeyHandler().globalTrackingMgrKey = self.key
        self.__readButtonConfig()
        self.__setupTimeout()
        self.sendParamsToCover()

    def setParams( self, params, negMsgHandler=None, sendToCover=True):
        coKeydObject.setParams(self, params, negMsgHandler, sendToCover)
        if sendToCover:
            self.sendParamsToCover()

    # SPECIAL CASE: routes the tracking input through vr-prepare
    def setVRC(self, x, y, z, button, negMsgHandler = None):
        # recognize idle VRC inputs
        if ((self.__oldVRCInput.x == x) and (self.__oldVRCInput.y == y) and (self.__oldVRCInput.z == z) and (self.__oldVRCInput.button == button)):
            pass
        else:
            self.__idleTimeElapsed = 0
        self.__oldVRCInput.x = x
        self.__oldVRCInput.y = y
        self.__oldVRCInput.z = z
        self.__oldVRCInput.button = button
        self.__negMsgHandler = negMsgHandler

        if not self.params.trackingGUIOn:
            return

        if self.VRCJoystick:
            temp = x
            x = -y
            y = -temp
            z = -z



        if self.params.oldX < x and x-self.params.oldX > self.TranslateBarrier:
            msg = coGRObjMoveObjMsg("", "translate", 0, -1, 0)
            self.params.oldX = x
            covise.sendRendMsg(msg.c_str())
        elif self.params.oldX > x and self.params.oldX-x > self.TranslateBarrier:
            msg = coGRObjMoveObjMsg("", "translate", 0, 1, 0)
            self.params.oldX = x
            covise.sendRendMsg(msg.c_str())
        if self.params.oldY < y and y-self.params.oldY > self.TranslateBarrier:
            msg = coGRObjMoveObjMsg("", "translate", -1, 0, 0)
            self.params.oldY = y
            covise.sendRendMsg(msg.c_str())
        elif self.params.oldY > y and self.params.oldY-y > self.TranslateBarrier:
            msg = coGRObjMoveObjMsg("", "translate", 1, 0, 0)
            self.params.oldY = y
            covise.sendRendMsg(msg.c_str())
        if self.params.oldZ < z and z-self.params.oldZ > self.TranslateBarrier:
            msg = coGRObjMoveObjMsg("", "translate", 0, 0, -1)
            self.params.oldZ = z
            covise.sendRendMsg(msg.c_str())
        elif self.params.oldZ > z and self.params.oldZ-z > self.TranslateBarrier:
            msg = coGRObjMoveObjMsg("", "translate", 0, 0, 1)
            self.params.oldZ = z 
            covise.sendRendMsg(msg.c_str())
        if button == self.TRANS_LEFT:
            msg = coGRObjMoveObjMsg("", "translate", -1, 0, 0)
            covise.sendRendMsg(msg.c_str())
        elif button == self.TRANS_RIGHT:
            msg = coGRObjMoveObjMsg("", "translate", 1, 0, 0)
            covise.sendRendMsg(msg.c_str())
        elif button == self.TRANS_UP:
            msg = coGRObjMoveObjMsg("", "translate", 0, 0, 1)
            covise.sendRendMsg(msg.c_str())
        elif button == self.TRANS_DOWN:
            msg = coGRObjMoveObjMsg("", "translate", 0, 0, -1)
            covise.sendRendMsg(msg.c_str())
        elif button == self.TRANS_FRONT:
            msg = coGRObjMoveObjMsg("", "translate", 0, -1, 0)
            covise.sendRendMsg(msg.c_str())
        elif button == self.TRANS_BACK:
            msg = coGRObjMoveObjMsg("", "translate", 0, 1, 0)
            covise.sendRendMsg(msg.c_str())

        # rotation AND zoom must work simultaneously
        if button & self.ROT_X_PLUS == self.ROT_X_PLUS:
            msg = coGRObjMoveObjMsg("", "rotate", 1, 0, 0)
            covise.sendRendMsg(msg.c_str())
        elif button & self.ROT_X_MINUS == self.ROT_X_MINUS:
            msg = coGRObjMoveObjMsg("", "rotate", -1, 0, 0)
            covise.sendRendMsg(msg.c_str())
        elif button & self.ROT_Y_PLUS == self.ROT_Y_PLUS:
            msg = coGRObjMoveObjMsg("", "rotate", 0, 1, 0)
            covise.sendRendMsg(msg.c_str())
        elif button & self.ROT_Y_MINUS == self.ROT_Y_MINUS:
            msg = coGRObjMoveObjMsg("", "rotate", 0, -1, 0)
            covise.sendRendMsg(msg.c_str())
        elif button & self.ROT_Z_PLUS == self.ROT_Z_PLUS:
            msg = coGRObjMoveObjMsg("", "rotate", 0, 0, -1)
            covise.sendRendMsg(msg.c_str())
        elif button & self.ROT_Z_MINUS == self.ROT_Z_MINUS:
            msg = coGRObjMoveObjMsg("", "rotate", 0, 0, 1)
            covise.sendRendMsg(msg.c_str())
        if button & self.SCALE_PLUS == self.SCALE_PLUS:
            msg = coGRObjMoveObjMsg("", "scale", 1, 0, 0)
            covise.sendRendMsg(msg.c_str())
        elif button & self.SCALE_MINUS == self.SCALE_MINUS:
            msg = coGRObjMoveObjMsg("", "scale", -1, 0, 0)
            covise.sendRendMsg(msg.c_str())

        if button == self.BUTTON_SENSOR_1:
            msg = coGRObjSensorEventMsg(coGRMsg.SENSOR_EVENT, self.wrl_name, 0, True, True)
            covise.sendRendMsg(msg.c_str())
            msg = coGRObjSensorEventMsg(coGRMsg.SENSOR_EVENT, self.wrl_name, 0, True, False)
            covise.sendRendMsg(msg.c_str())
            if HAS_SOUND:
                winsound.PlaySound(None, winsound.SND_ASYNC)
                winsound.PlaySound(covise.getCoConfigEntry("vr-prepare.TrackingManager.ButtonSensor1Sound"), winsound.SND_ASYNC)
        elif button == self.BUTTON_SENSOR_2:
            msg = coGRObjSensorEventMsg(coGRMsg.SENSOR_EVENT, self.wrl_name, 1, True, True)
            covise.sendRendMsg(msg.c_str())
            msg = coGRObjSensorEventMsg(coGRMsg.SENSOR_EVENT, self.wrl_name, 1, True, False)
            covise.sendRendMsg(msg.c_str())
            if HAS_SOUND:
                winsound.PlaySound(None, winsound.SND_ASYNC)
                winsound.PlaySound(covise.getCoConfigEntry("vr-prepare.TrackingManager.ButtonSensor2Sound"), winsound.SND_ASYNC)

    def sendParamsToCover(self):
       msg = coGRSetTrackingParamsMsg( \
                self.params.rotateMode == 'Point', \
                self.params.showRotatePoint, \
                self.params.rotationPointSize, \
                self.params.rotatePointX, \
                self.params.rotatePointY, \
                self.params.rotatePointZ, \
                self.params.rotateMode == 'Axis', \
                self.params.rotateAxisX, \
                self.params.rotateAxisY, \
                self.params.rotateAxisZ, \
                self.params.translateRestrict, \
                self.params.translateMinX, \
                self.params.translateMaxX, \
                self.params.translateMinY, \
                self.params.translateMaxY, \
                self.params.translateMinZ, \
                self.params.translateMaxZ, \
                self.params.translateFactor, \
                self.params.scaleRestrict, \
                self.params.scaleMin, \
                self.params.scaleMax, \
                self.params.scaleFactor, \
                self.params.trackingOn, \
                self.params.navigationMode)
       covise.sendRendMsg(msg.c_str())


class coTrackingMgrParams(object):
    def __init__(self):
        self.name       = 'Tracking Manager'
        coTrackingMgrParams.mergeDefaultParams(self) # explicitly call mergeDefaultParams of this class

    def mergeDefaultParams(self):
        defaultParams = {
            'rotateMode' : 'Free',
            'rotationPointSize': 0.5,
            'rotatePointX' : 0.0,
            'rotatePointY' : 0.0,
            'rotatePointZ' : 0.0,
            'rotateAxisX' : 1.0,
            'rotateAxisY' : 0.0,
            'rotateAxisZ' : 0.0,
            'translateRestrict' : False,
            'translateMinX' : 0.0,
            'translateMaxX' : 0.0,
            'translateMinY' : 0.0,
            'translateMaxY' : 0.0,
            'translateMinZ' : 0.0,
            'translateMaxZ' : 0.0,
            'translateFactor' : 1.0,
            'scaleRestrict' : False,
            'scaleMin' : 0.0,
            'scaleMax' : 99999.0,
            'scaleFactor' : 0.1,
            'trackingOn' : True,
            'trackingGUIOn' : False,
            'navigationMode' : '',
            # Values for navigation mode are the same as in COVER:
            # XForm,Scale,Fly,Walk,Drive,ShowName,XFormTranslate,XFormRotate,ShowName,Menu,Measure
            # (plus 'NavNone' for NavNone and '' for Undefined (i.e. do not change))
            'oldX' : 0.0,
            'oldY' : 0.0,
            'oldZ' : 0.0,
            'showRotatePoint' : False
        }
        mergeGivenParams(self, defaultParams)
