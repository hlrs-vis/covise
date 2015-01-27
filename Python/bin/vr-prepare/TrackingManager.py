
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH

import time

from PyQt5 import QtCore, QtGui, QtWidgets

from printing import InfoPrintCapable
import MainWindow
import Application
import covise

from coTrackingMgr import *

from Gui2Neg import theGuiMsgHandler
from TrackingManagerBase import Ui_TrackingManagerBase
from Utils import CopyParams, getDoubleInLineEdit
from ObjectMgr import ObjectMgr, GUI_PARAM_CHANGED_SIGNAL

from vtrans import coTranslate

class TrackingManagerBase(QtWidgets.QWidget, Ui_TrackingManagerBase):
    def __init__(self, parent=None, f=QtCore.Qt.WindowFlags()):
        QtWidgets.QWidget.__init__(self, parent, f)
        self.setupUi(self)
        
class TrackingManager(QtWidgets.QDockWidget):

    """ Handling of animation steps """
    def __init__(self, parent):
        QtWidgets.QDockWidget.__init__(self, self.__tr("Tracking Manager"), parent)
        self._key = -1
        self.setWidget( TrackingManagerBase(self) )

        ObjectMgr().sigGuiParamChanged.connect(self.paramChanged)
        # connection of the DockWidget visibilityChanged
        self.visibilityChanged.connect(self.visibilityChangedS)
        # connection of the elements
        self.widget().RotateFreeRadio.clicked.connect(self.paramsChanged)
        self.widget().RotatePointRadio.clicked.connect(self.paramsChanged)
        self.widget().RotationPointSize.returnPressed.connect(self.paramsChanged)
        self.widget().RotatePointX.returnPressed.connect(self.paramsChanged)
        self.widget().RotatePointY.returnPressed.connect(self.paramsChanged)
        self.widget().RotatePointZ.returnPressed.connect(self.paramsChanged)
        self.widget().RotateAxisRadio.clicked.connect(self.paramsChanged)
        self.widget().RotateAxisX.returnPressed.connect(self.paramsChanged)
        self.widget().RotateAxisY.returnPressed.connect(self.paramsChanged)
        self.widget().RotateAxisZ.returnPressed.connect(self.paramsChanged)
        self.widget().TranslateRestrictCB.clicked.connect(self.paramsChanged)
        self.widget().TranslateMinX.returnPressed.connect(self.paramsChanged)
        self.widget().TranslateMinY.returnPressed.connect(self.paramsChanged)
        self.widget().TranslateMinZ.returnPressed.connect(self.paramsChanged)
        self.widget().TranslateMaxX.returnPressed.connect(self.paramsChanged)
        self.widget().TranslateMaxY.returnPressed.connect(self.paramsChanged)
        self.widget().TranslateMaxZ.returnPressed.connect(self.paramsChanged)
        self.widget().TranslateFactor.returnPressed.connect(self.paramsChanged)
        self.widget().ScaleRestrictCB.clicked.connect(self.paramsChanged)
        self.widget().ScaleMin.returnPressed.connect(self.paramsChanged)
        self.widget().ScaleMax.returnPressed.connect(self.paramsChanged)
        self.widget().ScaleFactor.returnPressed.connect(self.paramsChanged)
        self.widget().checkBoxTracking.clicked.connect(self.paramsChanged)
        self.widget().checkBoxTrackingGUI.clicked.connect(self.paramsChanged)
        self.widget().NavigationModeCombo.activated.connect(self.paramsChanged)
        self.widget().ShowRotPoint.clicked.connect(self.paramsChanged)
        # connections of the buttons
        
        self.widget().RotateLeftButton.clicked.connect(self.rotateLeft)
        self.widget().RotateRightButton.clicked.connect(self.rotateRight)
        self.widget().RotateFrontButton.clicked.connect(self.rotateFront)
        self.widget().RotateBackButton.clicked.connect(self.rotateBack)
        self.widget().RotateUpButton.clicked.connect(self.rotateUp)
        self.widget().RotateDownButton.clicked.connect(self.rotateDown)
        self.widget().TranslateLeftButton.clicked.connect(self.translateLeft)
        self.widget().TranslateRightButton.clicked.connect(self.translateRight)
        self.widget().TranslateFrontButton.clicked.connect(self.translateFront)
        self.widget().TranslateBackButton.clicked.connect(self.translateBack)
        self.widget().TranslateUpButton.clicked.connect(self.translateUp)
        self.widget().TranslateDownButton.clicked.connect(self.translateDown)
        self.widget().ScalePlusButton.clicked.connect(self.scalePlus)
        self.widget().ScaleMinusButton.clicked.connect(self.scaleMinus)

        # set validators, allow only double values for changeIndicatedLEs
        doubleValidator = QtGui.QDoubleValidator(self)
        posDoubleValidator = QtGui.QDoubleValidator(self)
        posDoubleValidator.setBottom(0)
        # rotate validators
        self.widget().RotationPointSize.setValidator(doubleValidator)
        self.widget().RotatePointX.setValidator(doubleValidator)
        self.widget().RotatePointY.setValidator(doubleValidator)
        self.widget().RotatePointZ.setValidator(doubleValidator)
        self.widget().RotateAxisX.setValidator(doubleValidator)
        self.widget().RotateAxisY.setValidator(doubleValidator)
        self.widget().RotateAxisZ.setValidator(doubleValidator)
        # translate
        self.widget().TranslateMinX.setValidator(doubleValidator)
        self.widget().TranslateMaxX.setValidator(doubleValidator)
        self.widget().TranslateMinY.setValidator(doubleValidator)
        self.widget().TranslateMaxY.setValidator(doubleValidator)
        self.widget().TranslateMinZ.setValidator(doubleValidator)
        self.widget().TranslateMaxZ.setValidator(doubleValidator)
        self.widget().TranslateFactor.setValidator(posDoubleValidator)
        # scale
        self.widget().ScaleMin.setValidator(doubleValidator)
        self.widget().ScaleMax.setValidator(doubleValidator)
        self.widget().ScaleFactor.setValidator(posDoubleValidator)

    def visibilityChangedS(self, visibility):
        if Application.vrpApp.mw:
            Application.vrpApp.mw.windowTracking_ManagerAction.setChecked(self.isVisible()) # don't use visibility !! (see below)
        # If the DockWidget is displayed tabbed with other DockWidgets and the tab becomes inactive, visiblityChanged(false) is called.
        # Using visibility instead of self.isVisible() this would uncheck the menuentry and hide the DockWidget (including the tab).

    def show(self):
        QtWidgets.QWidget.show(self)

    def hide(self):
        QtWidgets.QWidget.hide(self)

    def paramChanged(self, key):
        """ params of object key changed"""
        if key==0:
            pass

    def updateForObject( self, key ):
        """ called from MainWindow to update the content to the choosen object key """
        self._key = key
        params = ObjectMgr().getParamsOfObject(key)
        self._setParams( params )

    def _setParams(self, params):
        # TODO: shouldn't there be a BlockSignals() before setting the values???

        # rotate
        self.widget().RotateFreeRadio.setChecked(params.rotateMode == 'Free')
        self.widget().RotatePointRadio.setChecked(params.rotateMode == 'Point')
        self.widget().RotateAxisRadio.setChecked(params.rotateMode == 'Axis')
        self.widget().ShowRotPoint.setChecked(params.showRotatePoint)
        self.widget().RotationPointSize.setText(str(params.rotationPointSize))
        self.widget().RotatePointX.setText(str(params.rotatePointX))
        self.widget().RotatePointY.setText(str(params.rotatePointY))
        self.widget().RotatePointZ.setText(str(params.rotatePointZ))
        self.widget().RotateAxisX.setText(str(params.rotateAxisX))
        self.widget().RotateAxisY.setText(str(params.rotateAxisY))
        self.widget().RotateAxisZ.setText(str(params.rotateAxisZ))
        # translate
        self.widget().TranslateRestrictCB.setChecked(params.translateRestrict)
        self.widget().TranslateMinX.setText(str(params.translateMinX))
        self.widget().TranslateMaxX.setText(str(params.translateMaxX))
        self.widget().TranslateMinY.setText(str(params.translateMinY))
        self.widget().TranslateMaxY.setText(str(params.translateMaxY))
        self.widget().TranslateMinZ.setText(str(params.translateMinZ))
        self.widget().TranslateMaxZ.setText(str(params.translateMaxZ))
        self.widget().TranslateFactor.setText(str(params.translateFactor))
        # scale
        self.widget().ScaleRestrictCB.setChecked(params.scaleRestrict)
        self.widget().ScaleMin.setText(str(params.scaleMin))
        self.widget().ScaleMax.setText(str(params.scaleMax))
        self.widget().ScaleFactor.setText(str(params.scaleFactor))
        #navigation
        self.widget().checkBoxTracking.setChecked(params.trackingOn)
        self.widget().checkBoxTrackingGUI.setChecked(params.trackingGUIOn)
        index = self.widget().NavigationModeCombo.findText(str(params.navigationMode))
        if (index < 0):
            index = 0
        self.widget().NavigationModeCombo.setCurrentIndex(index)
        # update
        self.updateGui()

    def _getParams(self):
        params = coTrackingMgrParams()
        # rotate
        if self.widget().RotatePointRadio.isChecked():
            params.rotateMode = 'Point'
        elif self.widget().RotateAxisRadio.isChecked():
            params.rotateMode = 'Axis'
        else:
            params.rotateMode = 'Free'
        params.showRotatePoint = self.widget().ShowRotPoint.isChecked()
        params.rotationPointSize = getDoubleInLineEdit(self.widget().RotationPointSize)
        params.rotatePointX = getDoubleInLineEdit(self.widget().RotatePointX)
        params.rotatePointY = getDoubleInLineEdit(self.widget().RotatePointY)
        params.rotatePointZ = getDoubleInLineEdit(self.widget().RotatePointZ)
        params.rotateAxisX = getDoubleInLineEdit(self.widget().RotateAxisX)
        params.rotateAxisY = getDoubleInLineEdit(self.widget().RotateAxisY)
        params.rotateAxisZ = getDoubleInLineEdit(self.widget().RotateAxisZ)
        # translate
        params.translateRestrict = self.widget().TranslateRestrictCB.isChecked()
        params.translateMinX = getDoubleInLineEdit(self.widget().TranslateMinX)
        params.translateMaxX = getDoubleInLineEdit(self.widget().TranslateMaxX)
        params.translateMinY = getDoubleInLineEdit(self.widget().TranslateMinY)
        params.translateMaxY = getDoubleInLineEdit(self.widget().TranslateMaxY)
        params.translateMinZ = getDoubleInLineEdit(self.widget().TranslateMinZ)
        params.translateMaxZ = getDoubleInLineEdit(self.widget().TranslateMaxZ)
        params.translateFactor = getDoubleInLineEdit(self.widget().TranslateFactor)
        # scale
        params.scaleRestrict = self.widget().ScaleRestrictCB.isChecked()
        params.scaleMin = getDoubleInLineEdit(self.widget().ScaleMin)
        params.scaleMax = getDoubleInLineEdit(self.widget().ScaleMax)
        params.scaleFactor = getDoubleInLineEdit(self.widget().ScaleFactor)
        #navigation
        params.trackingOn = self.widget().checkBoxTracking.isChecked()
        params.trackingGUIOn = self.widget().checkBoxTrackingGUI.isChecked()
        params.navigationMode = str(self.widget().NavigationModeCombo.currentText())
        return params
    
    def setNavigationMode(self, mode):
        index = self.widget().NavigationModeCombo.findText(str(mode))
        if (index >= 0):
            self.widget().NavigationModeCombo.setCurrentIndex(index)
            self.paramsChanged();

    def paramsChanged(self, dummy=None):
        self.updateGui()
        params = self._getParams()
        Application.vrpApp.key2params[self._key] = params
        ObjectMgr().setParams( self._key, params )

    def setTrackingMgrKey( self, key):
        self._key = key

    def __tr(self,s,c = None):
        return coTranslate(s)

    def updateGui(self):
        if self.widget().RotateAxisRadio.isChecked():
            self.widget().RotateLeftButton.setText("+")
            self.widget().RotateRightButton.setText("-")
        else:
            self.widget().RotateLeftButton.setText("X +")
            self.widget().RotateRightButton.setText("X -")

        self.widget().RotateFrontButton.setVisible(not self.widget().RotateAxisRadio.isChecked())
        self.widget().RotateBackButton.setVisible(not self.widget().RotateAxisRadio.isChecked())
        self.widget().RotateUpButton.setVisible(not self.widget().RotateAxisRadio.isChecked())
        self.widget().RotateDownButton.setVisible(not self.widget().RotateAxisRadio.isChecked())

    def __str__(self):
        ret = "Tracking Manager\n"
        return ret

    # functions for buttons
    # send message to opencover
    def rotateLeft(self):
        theGuiMsgHandler().sendMoveObj("rotate", 1, 0 ,0)
    def rotateRight(self):
        theGuiMsgHandler().sendMoveObj("rotate", -1, 0 ,0)
    def rotateFront(self):
        theGuiMsgHandler().sendMoveObj("rotate", 0, 1 ,0)
    def rotateBack(self):
        theGuiMsgHandler().sendMoveObj("rotate", 0, -1 ,0)
    def rotateUp(self):
        theGuiMsgHandler().sendMoveObj("rotate", 0, 0 ,1)
    def rotateDown(self):
        theGuiMsgHandler().sendMoveObj("rotate", 0, 0 ,-1)

    def translateLeft(self):
        theGuiMsgHandler().sendMoveObj("translate", -1, 0 ,0)
    def translateRight(self):
        theGuiMsgHandler().sendMoveObj("translate", 1, 0 ,0)
    def translateFront(self):
        theGuiMsgHandler().sendMoveObj("translate", 0, -1 ,0)
    def translateBack(self):
        theGuiMsgHandler().sendMoveObj("translate", 0, 1 ,0)
    def translateUp(self):
        theGuiMsgHandler().sendMoveObj("translate", 0, 0 ,1)
    def translateDown(self):
        theGuiMsgHandler().sendMoveObj("translate", 0, 0 ,-1)

    def scaleMinus(self):
        theGuiMsgHandler().sendMoveObj("scale", -1, 0, 0)
    def scalePlus(self):
        theGuiMsgHandler().sendMoveObj("scale", 1, 0, 0)
# eof
