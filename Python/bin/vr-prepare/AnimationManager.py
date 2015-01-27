
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH

import time

from PyQt5 import QtCore, QtGui, QtWidgets

from printing import InfoPrintCapable
import MainWindow
import Application
import covise

from Gui2Neg import theGuiMsgHandler
from AnimationManagerBase import Ui_AnimationManagerBase
from Utils import CopyParams, getIntInLineEdit
from ObjectMgr import ObjectMgr, GUI_PARAM_CHANGED_SIGNAL
from TimestepSelectorDialog import TimestepSelectorDialog
from coGRMsg import coGRMsg, coGRAnimationOnMsg, coGRSetTimestepMsg, coGRSetAnimationSpeedMsg

from vrpconstants import REDUCTION_FACTOR, SELECTION_STRING

from vtrans import coTranslate 

class AnimationManagerBase(QtWidgets.QWidget, Ui_AnimationManagerBase):
    def __init__(self, parent=None, f=QtCore.Qt.WindowFlags()):
        QtWidgets.QWidget.__init__(self, parent, f)
        self.setupUi(self)


class AnimationManager(QtWidgets.QDockWidget):

    """ Handling of animation steps """
    def __init__(self, parent):
        QtWidgets.QDockWidget.__init__(self, self.__tr("Animation Manager"), parent)
        # default is prenseter manager
        self._key = -1

        #self.setSizePolicy(QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum,QtWidgets.QSizePolicy.Minimum))
        #properties of QDockWindow
        #REM self.setCloseMode(QtWidgets.QDockWindow.Always) 
        #REM self.setResizeEnabled(True)
        #REM self.setMovingEnabled(True)        #allow to be outside the window
        #REM self.setHorizontallyStretchable(True)
        #REM self.setVerticallyStretchable(True)
        #REM self.setOrientation(Qt.Vertical)
        self.setWidget( AnimationManagerBase(self) )

        ObjectMgr().sigGuiParamChanged.connect(self.paramChanged)
        #connection of the DockWidget visibilityChanged
        self.visibilityChanged.connect(self.visibilityChangedS)
        #connections of the buttons
        self.widget().ReductionLE.returnPressed.connect(self.changedReductionParams)
        self.widget().pushButtonEditSelectionString.clicked.connect(self.openTimestepSelectorDialog)
        self.widget().pushButtonApply.clicked.connect(self.applyTimestepSelection)
        self.widget().radioButton15.clicked.connect(self.emitChangedRadioGroup)
        self.widget().radioButtonSelectionString.clicked.connect(self.emitChangedRadioGroup)
        self.widget().ToStartButton.clicked.connect(self.emitFirstAnimationStep)
        self.widget().BackButton.clicked.connect(self.emitAnimationStepBack)
        self.widget().StopButton.clicked.connect(self.emitAnimationModeOff)
        self.widget().PlayButton.clicked.connect(self.emitAnimationModeOn)
        self.widget().ForwardButton.clicked.connect(self.emitAnimationStepForward)
        self.widget().ToEndButton.clicked.connect(self.emitLastAnimationStep)
        self.widget().timestepSlider.sigSliderReleased.connect(self.emitTimeStep)
        self.widget().sliderSpeed.sigSliderReleased.connect(self.emitAnimationSpeed)
        #validators
        # allow only int values for changeIndicatedLEs
        intValidator = QtGui.QIntValidator(self)
        intValidator.setBottom(0)
        self.widget().ReductionLE.setValidator(intValidator)
        self.widget().timestepSlider.lineEdit.setValidator(intValidator)
        self.widget().timestepSlider.slider.setSingleStep(1)
        self.widget().timestepSlider.slider.setPageStep(1)

        #TODO: QRegExp einbinden
        #regExpValidator = QtGui.QRegExpValidator(QRegExp('\d+\s*\-\s*\d+|\d+'), self, "a QRegExpValidator")
        #self.widget().lineEditSelectionString.setValidator(regExpValidator)

        self.animSpeed = 0

        self._reductionFactor = -1
        self._selectionFactor = -1

    def visibilityChangedS(self, visibility):
        if Application.vrpApp.mw:
            Application.vrpApp.mw.windowAnimation_ManagerAction.setChecked(self.isVisible()) # don't use visibility !! (see below)
        # If the DockWidget is displayed tabbed with other DockWidgets and the tab becomes inactive, visiblityChanged(false) is called.
        # Using visibility instead of self.isVisible() this would uncheck the menuentry and hide the DockWidget (including the tab).

    def show(self):
        QtWidgets.QWidget.show(self)

    def hide(self):
        QtWidgets.QWidget.hide(self)

    def setAnimationMgrKey( self, key):
        self._key = key

    def paramChanged(self, key):
        """ params of object key changed"""
        if key==0:
            params = ObjectMgr().getParamsOfObject(0)
            self.widget().NumStepsLabel_2.setText(str(params.numTimeSteps))
            if params.reductionFactor != self._reductionFactor:
                self._reductionFactor = params.reductionFactor
                self.widget().ReductionLE.setText(str(params.reductionFactor))
            if params.selectionString != self._selectionFactor:
                self._selectionFactor = params.selectionString
                self.widget().lineEditSelectionString.setText(str(params.selectionString))
            if params.numTimeSteps > 1:
                self.widget().timestepSlider.setRange([0,params.numTimeSteps-1])
                self.widget().timestepSlider.setValue(params.actTimeStep)

                if params.animationSpeed != self.animSpeed:
                    self.animSpeed = params.animationSpeed
                    self.widget().sliderSpeed.setRange([params.animationSpeedMin, params.animationSpeedMax])
                    self.widget().sliderSpeed.setValue(params.animationSpeed)

                if params.animateOn:
                    self.widget().ToStartButton.setEnabled(False)
                    self.widget().BackButton.setEnabled(False)
                    self.widget().PlayButton.setEnabled(False)
                    self.widget().StopButton.setEnabled(True)
                    self.widget().ForwardButton.setEnabled(False)
                    self.widget().ToEndButton.setEnabled(False)
                else:
                    self.widget().ToStartButton.setEnabled(True)
                    self.widget().BackButton.setEnabled(True)
                    self.widget().PlayButton.setEnabled(True)
                    self.widget().StopButton.setEnabled(False)
                    self.widget().ForwardButton.setEnabled(True)
                    self.widget().ToEndButton.setEnabled(True)



                if params.filterChoice == REDUCTION_FACTOR:
                    self.widget().radioButton15.setChecked(True)
                    self.widget().radioButtonSelectionString.setChecked(False)

                    if not self.widget().ReductionLE.isEnabled():
                        self.widget().ReductionLE.setEnabled(True)
                    if not self.widget().lineEditSelectionString.isEnabled():
                        self.widget().lineEditSelectionString.setEnabled(False)
                    self.widget().pushButtonEditSelectionString.setEnabled(False)
                    self.widget().pushButtonApply.setEnabled(False)
                elif params.filterChoice == SELECTION_STRING:
                    self.widget().radioButton15.setChecked(False)
                    self.widget().radioButtonSelectionString.setChecked(True)

                    if self.widget().ReductionLE.isEnabled():    
                        self.widget().ReductionLE.setEnabled(False)
                    if self.widget().lineEditSelectionString.isEnabled():
                        self.widget().lineEditSelectionString.setEnabled(True)
                    self.widget().pushButtonEditSelectionString.setEnabled(True)
                    self.widget().pushButtonApply.setEnabled(True)

    def __getParams(self):
        data = ObjectMgr().getParamsOfObject(0)
        if data.numTimeSteps > 1:
            data.actTimeStep = self.widget().timestepSlider.getValue()

        data.animationSpeed = self.widget().sliderSpeed.getValue()
        return data

    def changedReductionParams(self):
        rf = getIntInLineEdit(self.widget().ReductionLE)
        Application.vrpApp.mw.spawnPatienceDialog()
        reqId = theGuiMsgHandler().setReductionFactor(rf)
        theGuiMsgHandler().waitforAnswer(reqId)
        Application.vrpApp.mw.unSpawnPatienceDialog()

        # update params of project in ObjectMgr
        params = ObjectMgr().getParamsOfObject(0)
        params.reductionFactor = rf
        ObjectMgr().setParams(0, params)

    def openTimestepSelectorDialog(self):
        params = ObjectMgr().getParamsOfObject(0)

        dialog = TimestepSelectorDialog(self)
        dialog.fillListView(params.numTimeSteps)

        dialog.setFromSelectionString(str(self.widget().lineEditSelectionString.text()))
        decision = dialog.exec_()
        if decision == QtWidgets.QDialog.Accepted:
            selectionString = dialog.getSelectionString()
            self.widget().lineEditSelectionString.setText(selectionString)

    def applyTimestepSelection(self):
        selectionString = str(self.widget().lineEditSelectionString.text())

        Application.vrpApp.mw.spawnPatienceDialog()
        reqId = theGuiMsgHandler().setSelectionString(selectionString)
        theGuiMsgHandler().waitforAnswer(reqId)
        Application.vrpApp.mw.unSpawnPatienceDialog()

        # update params of project in ObjectMgr
        params = ObjectMgr().getParamsOfObject(0)
        params.selectionString = selectionString
        ObjectMgr().setParams(0, params)

    def emitChangedRadioGroup(self):
        if self.widget().radioButton15.isChecked():
            self.widget().ReductionLE.setEnabled(True)
            self.widget().lineEditSelectionString.setEnabled(False)
            self.widget().pushButtonEditSelectionString.setEnabled(False)
            self.widget().pushButtonApply.setEnabled(False)

            filterChoice = REDUCTION_FACTOR

        elif self.widget().radioButtonSelectionString.isChecked():
            self.widget().ReductionLE.setEnabled(False)
            self.widget().lineEditSelectionString.setEnabled(True)
            self.widget().pushButtonEditSelectionString.setEnabled(True)
            self.widget().pushButtonApply.setEnabled(True)

            filterChoice = SELECTION_STRING

        # update params of project in ObjectMgr
        params = ObjectMgr().getParamsOfObject(0)
        params.filterChoice = filterChoice
        ObjectMgr().setParams(0, params)

    def emitAnimationModeOn(self):
        self.widget().ToStartButton.setEnabled(False)
        self.widget().BackButton.setEnabled(False)
        self.widget().PlayButton.setEnabled(False)
        self.widget().StopButton.setEnabled(True)
        self.widget().ForwardButton.setEnabled(False)
        self.widget().ToEndButton.setEnabled(False)
        params = ObjectMgr().getParamsOfObject(0)
        params.animateOn = True
        ObjectMgr().setParams(0, params)
        msg = coGRAnimationOnMsg(True)
        covise.sendRendMsg(msg.c_str())

    def emitAnimationModeOff(self):
        self.widget().ToStartButton.setEnabled(True)
        self.widget().BackButton.setEnabled(True)
        self.widget().PlayButton.setEnabled(True)
        self.widget().StopButton.setEnabled(False)
        self.widget().ForwardButton.setEnabled(True)
        self.widget().ToEndButton.setEnabled(True)
        params = ObjectMgr().getParamsOfObject(0)
        params.animateOn = False
        # enhance timestep because renderer gets message in next frame
        params.actTimeStep = int (self.widget().timestepSlider.getValue()+1)
        self.widget().timestepSlider.setValue(params.actTimeStep)
        ObjectMgr().setParams(0, params)
        msg = coGRAnimationOnMsg( False)
        covise.sendRendMsg(msg.c_str())

    def emitAnimationStepBack(self):
        params = ObjectMgr().getParamsOfObject(0)
        if params == None:
            return 
        params.actTimeStep = int (self.widget().timestepSlider.getValue()-1)
        if params.actTimeStep < 0:
            params.actTimeStep=params.numTimeSteps-1
        ObjectMgr().setParams(0, params)
        msg = coGRSetTimestepMsg( params.actTimeStep, params.numTimeSteps)
        covise.sendRendMsg(msg.c_str())

    def emitAnimationStepForward(self):
        params = ObjectMgr().getParamsOfObject(0)
        if params == None:
            return 
        params.actTimeStep = int (self.widget().timestepSlider.getValue()+1)
        if params.actTimeStep > params.numTimeSteps-1:
            params.actTimeStep=0
        ObjectMgr().setParams(0, params)
        msg = coGRSetTimestepMsg( params.actTimeStep, params.numTimeSteps)
        covise.sendRendMsg(msg.c_str())

    def emitFirstAnimationStep(self):
        params = ObjectMgr().getParamsOfObject(0)
        if params == None:
            return 
        params.actTimeStep = 0
        ObjectMgr().setParams(0, params)
        msg = coGRSetTimestepMsg( params.actTimeStep, params.numTimeSteps)
        covise.sendRendMsg(msg.c_str())

    def emitLastAnimationStep(self):
        params = ObjectMgr().getParamsOfObject(0)
        if params == None:
            return 
        params.actTimeStep = params.numTimeSteps -1 
        ObjectMgr().setParams(0, params)
        msg = coGRSetTimestepMsg( params.actTimeStep, params.numTimeSteps)
        covise.sendRendMsg(msg.c_str())

    def emitTimeStep(self):
        params = ObjectMgr().getParamsOfObject(0)
        if params == None:
            return 
        params.actTimeStep = int (self.widget().timestepSlider.getValue())
        ObjectMgr().setParams(0, params)
        msg = coGRSetTimestepMsg( params.actTimeStep, params.numTimeSteps)
        covise.sendRendMsg(msg.c_str())

    def emitAnimationSpeed(self):
        params = ObjectMgr().getParamsOfObject(0)
        params.animationSpeed = self.widget().sliderSpeed.getValue()
        self.animSpeed = params.animationSpeed
        ObjectMgr().setParams(0, params)
        msg = coGRSetAnimationSpeedMsg( params.animationSpeed, params.animationSpeedMin, params.animationSpeedMax )
        covise.sendRendMsg(msg.c_str())


    def __tr(self,s,c = None):
        return coTranslate(s)


    def __str__(self):
        ret = "Animation Manager\n"
        return ret


# eof
