import sys

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject, pyqtSignal

import Application

from VrmlPanelBase import Ui_VrmlPanelBase
from VrmlPanelConnector import VrmlPanelConnector

from VrmlVis import VrmlVisParams
from Gui2Neg import theGuiMsgHandler
from ObjectMgr import ObjectMgr
from KeydObject import VIS_VRML
from Utils import ParamsDiff, CopyParams
import copy

from printing import InfoPrintCapable

_infoer = InfoPrintCapable()
_infoer.doPrint = False #True #

from vtrans import coTranslate

class VrmlPanel(QtWidgets.QWidget,Ui_VrmlPanelBase):
    def __init__(self, parent=None):
        _infoer.function = str(self.__init__)
        _infoer.write("")
        QtWidgets.QWidget.__init__(self, parent)
        Ui_VrmlPanelBase.__init__(self)
        self.setupUi(self)

        # list of associated keys from same type
        self.__keys = []

        # for multi selection
        self.oldPanelParams = {}

        # designer could not assign a layout to empty widgets
        #
        #self.__dummySensorsLayout = QtWidgets.QVBoxLayout(self.tabSensors)#, 0, 0)
        self.__verticalSpacer = QtWidgets.QSpacerItem(16,16,QtWidgets.QSizePolicy.Preferred,QtWidgets.QSizePolicy.Minimum)
        self.__dummyFrameLayout = QtWidgets.QVBoxLayout(self.tabSensors)
        self.__dummySensorsLayout = QtWidgets.QVBoxLayout()
        self.__dummyBox = QtWidgets.QWidget()
        self.__dummyBox.setLayout(self.__dummySensorsLayout)
        
        #
        # add scroll view
        #
        self.scrollView = QtWidgets.QScrollArea()#self.dummyFrame)   
        self.scrollView.setFrameShadow(QtWidgets.QFrame.Raised)
        self.scrollView.setFrameShape(QtWidgets.QFrame.StyledPanel)        
        
        #self.scrollView.setWidget(self.__dummyBox)
        self.__dummyFrameLayout.addWidget(self.scrollView)

        self.tabWidget.setCurrentIndex(0) # ignore index set by the designer (usually no one cares about the active index when editing ui-files)
        self.tabWidget.setTabEnabled (1, False) # hide sensor tab

        # hold the dynamically created sensor widgets
        self.__sensorButton = {}
        self.__sensorCheckBox = {}

        # catch the pressed sensor button id
        self.__pressedSensorID = None

        # hold the parameters of object. necessary, if not all parameters are stored in Panel widgets
        self.__params = VrmlVisParams()

        self.__firstTime = True

        VrmlPanelConnector(self)


    def paramChanged(self, key):
        """ params of object key changed"""

        _infoer.function = str(self.paramChanged)
        _infoer.write("key %d" %key)

        #update only for single selection
        if len(self.__keys) ==1 :
            if self.__keys[0]==key or (self.__keys[0] in Application.vrpApp.guiKey2visuKey and key==Application.vrpApp.guiKey2visuKey[self.__keys[0]]):
                self.update()

    def update(self):
        _infoer.function = str(self.update)
        _infoer.write("")
        if len(self.__keys)!=0 :
            self.updateForObject( self.__keys )

    def updateForObject( self, keys ):
        """ called from MainWindow to update the content to the choosen object key

        Default params will be shown if there is more than 1 key"""
        _infoer.function = str(self.updateForObject)
        _infoer.write("")


        if isinstance( keys, int ) :
            self.__keys = [keys]
        elif isinstance( keys, list ) and len(keys)>0 :
            self.__keys = keys
        else :
            return

        if len(self.__keys) == 1 :
            params = ObjectMgr().getParamsOfObject(self.__keys[0])
            if isinstance(params, int) or isinstance(params, VrmlVisParams):
                self.__setParams( params )
        elif len(self.__keys) > 1 :
            # multi selection: show default params
            self.oldPanelParams = VrmlVisParams()
            params = CopyParams(self.oldPanelParams)
            params.name = "Multiselection"
            self.__setParams( params )

    def __getParams(self):
        _infoer.function = str(self.__getParams)
        _infoer.write("")

        data = CopyParams(self.__params)    # use parameter values of this object that arent in this panel

        data.filename = str(self.lineEditFilename.text())

#        data.name = str(self.lineEditFilename.text())
        data.sensorIDs = []
        data.autoActiveSensorIDs = []
        for sensorID, checkBox in iter(self.__sensorCheckBox.items()):
            data.sensorIDs.append(sensorID)
            if checkBox.isChecked():
                data.autoActiveSensorIDs.append(sensorID)

        self.__params = CopyParams(data)

        return data

    def __setParams( self, params ):
        _infoer.function = str(self.__setParams)
        _infoer.write("")

        self.__params = CopyParams(params)

#        VrmlPanelBlockSignals(self, True)
        self.blockSignals(self.tabSensors, True)

        if isinstance( params, int):
            self.__keys[0] = params
            return

        self.lineEditFilename.setText(params.filename)

        # create the sensor widgets
        # first time: add all sensors to panel
        if self.__firstTime :
           if params.sensorIDs != []:
               self.tabWidget.setTabEnabled (1, True) # show sensor tab
           for sensorID in params.sensorIDs:
               self.addSensorWidget(self.tabSensors, sensorID, sensorID in params.autoActiveSensorIDs)
           self.__dummySensorsLayout.addItem(self.__verticalSpacer)     # vertical spacer
           self.__firstTime = False
           self.scrollView.setWidget(self.__dummyBox)           
        # other times: if sensor not in list, create (with spacer)
        #              if sensor in list: check checkbox
        else:
           for sensorID in params.sensorIDs:
               if not sensorID in self.__sensorCheckBox:
                  self.__dummySensorsLayout.removeItem(self.__verticalSpacer)
                  self.addSensorWidget(self.tabSensors, sensorID, sensorID in params.autoActiveSensorIDs)
                  self.__dummySensorsLayout.addItem(self.__verticalSpacer)     # vertical spacer
               else:
                   self.__sensorCheckBox[sensorID].setChecked(sensorID in params.autoActiveSensorIDs)

#        VrmlPanelBlockSignals(self, False)
        self.blockSignals(self.tabSensors, False)

        # for multi selection
        if len(self.__keys)>1 :
            self.oldPanelParams = params

    def clearSensorWidgets(self, parent):
        for sensorID, widget in iter(self.__sensorButton.items()):
            widget.setParent(None)
        self.__sensorButton = {}

        for sensorID, widget in iter(self.__sensorCheckBox.items()):
            widget.setParent(None)
        self.__sensorCheckBox = {}

    def addSensorWidget(self, parent, sensorID, autoActive):
        button = QtWidgets.QPushButton(str(sensorID), parent)
        checkBox = QtWidgets.QCheckBox(coTranslate("Auto-activate in Presentation"), parent)
        checkBox.setChecked(autoActive)

        self.__sensorButton[sensorID] = button
        self.__sensorCheckBox[sensorID] = checkBox

        # add widgets to layout
        self.__dummySensorsLayout.addWidget(button)
        self.__dummySensorsLayout.addWidget(checkBox)

        # make signal-slot connections
        checkBox.toggled.connect(self.emitDataChanged)
        button.clicked.connect(self.emitActivateSensor)
        button.pressed.connect(self.__sensorButtonPressed)

        # explicitly show created widgets (doesnt happen automatically)
        button.show()
        checkBox.show()


    def emitNameChange(self, aQString = None):
        _infoer.function = str(self.emitNameChange)
        _infoer.write("")
        # only for single selection
        if len(self.__keys)==1 :
            #MainWindow.globalAccessToTreeView.setItemData(self.__keys[0], str(self.nameWidget.text()))
            # set name of type_2d_part
            params = ObjectMgr().getParamsOfObject(self.__keys[0])
            params.name = str(self.lineEditFilename.text())
            Application.vrpApp.key2params[self.__keys[0]] = params
            ObjectMgr().setParams(self.__keys[0], params)
            self.emitDataChanged()

    def __sensorButtonPressed(self):
        for sensorID, button in iter(self.__sensorButton.items()):
            if button.isDown():
                self.__pressedSensorID = sensorID

    def emitActivateSensor(self):
        params = self.__getParams()
        params.clickedSensorID = self.__pressedSensorID
        Application.vrpApp.key2params[self.__keys[0]] = params
        ObjectMgr().setParams( self.__keys[0], params )

        self.__pressedSensorID = None



    def emitDataChanged(self):
        _infoer.function = str(self.emitDataChanged)
        _infoer.write("")

        params = self.__getParams()
        if len(self.__keys)==1 :
            Application.vrpApp.key2params[self.__keys[0]] = params
            ObjectMgr().setParams( self.__keys[0], params )

        # set params for multi selection
        if len(self.__keys)>1 :
            #find changed params
            originalParams = self.oldPanelParams
            realChange = ParamsDiff( originalParams, params )

            # set params for remaining selected objects
            for i in range(0, len(self.__keys)):
                keyParams = ObjectMgr().getParamsOfObject(self.__keys[i])

                # find the changed param in childKey and replace it with the
                # intended attribut
                for x in realChange :
                    keyParams.__dict__[x] = params.__dict__[x]
                # set the params
                Application.vrpApp.key2params[self.__keys[i]] = keyParams
                ObjectMgr().setParams( self.__keys[i], keyParams )
                #theGuiMsgHandler().runObject( childKeys[i] )
            #save params for multi selection
            self.oldPanelParams = self.__getParams()

    def blockSignals(self, panel, doBlock):
        panel.blockSignals(doBlock)

        for widget in self.__sensorButton.values():
            widget.blockSignals(doBlock)
        for widget in self.__sensorCheckBox.values():
            widget.blockSignals(doBlock)

