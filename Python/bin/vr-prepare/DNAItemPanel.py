
# Part of the vr-prepare program for dc

# Copyright (c) 2007 Visenso GmbH

import sys

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject, pyqtSignal

import Application

from DNAItemPanelBase import Ui_DNAItemPanelBase
from DNAItemPanelConnector import DNAItemPanelBlockSignals, DNAItemPanelConnector
from TransformManager import TransformManager

from coDNAMgr import coDNAItemParams
from Gui2Neg import theGuiMsgHandler
from ObjectMgr import ObjectMgr
from KeydObject import TYPE_DNA_ITEM
from Utils import ParamsDiff, CopyParams
import copy

from printing import InfoPrintCapable


_infoer = InfoPrintCapable()
_infoer.doPrint = False #True #

#id for coloring option
NO_COLOR = 0
RGB_COLOR = 1
MATERIAL = 2
VARIABLE = 3


class DNAItemPanel(QtWidgets.QWidget,Ui_DNAItemPanelBase, TransformManager):
    def __init__(self, parent=None):
        _infoer.function = str(self.__init__)
        _infoer.write("")
        QtWidgets.QWidget.__init__(self, parent)
        Ui_DNAItemPanelBase.__init__(self)
        self.setupUi(self)
        TransformManager.__init__(self, self.emitTransformChange)

        #disable Colorpanel till it is working
        self.TabWidgetGeneralAdvanced.setTabEnabled(2, False)
        #hide all Connections
        self.vrpCheckBoxConn1.hide()
        self.vrpCheckBoxConnEnabled1.hide()        
        self.vrpCheckBoxConn2.hide()
        self.vrpCheckBoxConnEnabled2.hide()        
        self.vrpCheckBoxConn3.hide()
        self.vrpCheckBoxConnEnabled3.hide()        
        self.vrpCheckBoxConn4.hide()
        self.vrpCheckBoxConnEnabled4.hide()        
        self.vrpCheckBoxConn5.hide()        
        self.vrpCheckBoxConnEnabled5.hide()        
        
        self.connections = [self.vrpCheckBoxConn1, self.vrpCheckBoxConn2, self.vrpCheckBoxConn3, self.vrpCheckBoxConn4, self.vrpCheckBoxConn5]
        self.connectionsEnabled = [self.vrpCheckBoxConnEnabled1, self.vrpCheckBoxConnEnabled2, self.vrpCheckBoxConnEnabled3, self.vrpCheckBoxConnEnabled4, self.vrpCheckBoxConnEnabled5]

        self.TabWidgetGeneralAdvanced.setCurrentIndex(0) # ignore index set by the designer (usually no one cares about the active index when editing ui-files)

        # list of associated keys from same type
        self.__keys = []

        # for multi selection
        self.oldPanelParams = {}

        DNAItemPanelConnector(self)

    def paramChanged(self, key):
        """ params of object key changed"""

        _infoer.function = str(self.paramChanged)
        _infoer.write("key %d" %key)

        #update only for single selection
        if len(self.__keys) ==1 :
            if self.__keys[0]==key or (self.__keys[0] in Application.vrpApp.visuKey2GuiKey and key==Application.vrpApp.visuKey2GuiKey[self.__keys[0]]):
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
            if isinstance(params, int) or isinstance(params, coDNAItemParams):
                self.__setParams( params )

        elif len(self.__keys) > 1 :
            # multi selection: show default params
            self.oldPanelParams = coDNAItemParams()
            params = CopyParams(self.oldPanelParams)
            params.name = "Multiselection"
            self.__setParams( params )

    def __getParams(self):
        _infoer.function = str(self.__getParams)
        _infoer.write("")

        data = coDNAItemParams()

        data.name = str(self.nameWidget.text())

        #transform card
        self.TransformManagerGetParams(data)

        #connections
        data.connectionPoints = {}
        data.connectionPointsDisable = {}
        for i in range(5):
            conn = self.connections[i]
            connEnabled = self.connectionsEnabled[i]
            if not conn.isHidden() :
                data.connectionPoints[str(conn.text())] = conn.isChecked()
                data.connectionPointsDisable[str(conn.text())] = connEnabled.isChecked()
        data.needConn = self.vrpCheckBoxNeedToBeConn.isChecked()

        return data

    def __setParams( self, params ):
        _infoer.function = str(self.__setParams)
        _infoer.write("")

        DNAItemPanelBlockSignals(self, True)
        self.TransformManagerBlockSignals(True)

        if isinstance( params, int):
            self.__keys[0] = params
            return

        self.nameWidget.setText(params.name)

        #transform card
        self.TransformManagerSetParams(params)
        
        #connections
        #hide all Connections
        self.vrpCheckBoxConn1.hide()
        self.vrpCheckBoxConnEnabled1.hide()        
        self.vrpCheckBoxConn2.hide()
        self.vrpCheckBoxConnEnabled2.hide()        
        self.vrpCheckBoxConn3.hide()
        self.vrpCheckBoxConnEnabled3.hide()        
        self.vrpCheckBoxConn4.hide()
        self.vrpCheckBoxConnEnabled4.hide()        
        self.vrpCheckBoxConn5.hide()        
        self.vrpCheckBoxConnEnabled5.hide()                
        index = 0
        for conn in params.connectionPoints:
            if index > 4:
                break
            self.connections[index].show()
            self.connectionsEnabled[index].show()
            self.connections[index].setText(conn)
            self.connections[index].setChecked(params.connectionPoints[conn])
            self.connectionsEnabled[index].setChecked(params.connectionPointsDisable[conn])
            # make box unckeckable if there is no connection
            # only possible to disconnect from GUI
            if not self.connections[index].isChecked():
                self.connections[index].setEnabled(False)
            else:
                self.connections[index].setEnabled(True)
            index = index +1
            
        self.vrpCheckBoxNeedToBeConn.setChecked(params.needConn)

        self.TransformManagerBlockSignals(False)
        DNAItemPanelBlockSignals(self, False)

        # for multi selection
        if len(self.__keys)>1 :
            self.oldPanelParams = params


    def emitNameChange(self, aQString = None):
        _infoer.function = str(self.emitNameChange)
        _infoer.write("")
        # only for single selection
        if len(self.__keys)==1 :
            #MainWindow.globalAccessToTreeView.setItemData(self.__keys[0], str(self.nameWidget.text()))
            # set name of type_2d_part
            params = ObjectMgr().getParamsOfObject(self.__keys[0])
            params.name = str(self.nameWidget.text())
            Application.vrpApp.key2params[self.__keys[0]] = params
            ObjectMgr().setParams(self.__keys[0], params)
            self.emitDataChanged()

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

    def emitTransformChange(self, aQString = None):
        _infoer.function = str(self.emitNameChange)
        _infoer.write("")
        # only for single selection
        if len(self.__keys)==1 :
            #MainWindow.globalAccessToTreeView.setItemData(self.__keys[0], str(self.nameWidget.text()))
            # set name of type_2d_part
            params = ObjectMgr().getParamsOfObject(self.__keys[0])
            TransformManagerGetParams(params)
            Application.vrpApp.key2params[self.__keys[0]] = params
            ObjectMgr().setParams(self.__keys[0], params)
            self.emitDataChanged()


