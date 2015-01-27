
# Part of the vr-prepare program for dc

# Copyright (c) 2007 Visenso GmbH

import sys

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject, pyqtSignal

import Application

from SceneGraphItemPanelBase import Ui_SceneGraphItemPanelBase
from SceneGraphItemPanelConnector import SceneGraphItemPanelBlockSignals, SceneGraphItemPanelConnector
from TransformManager import TransformManager

from coSceneGraphMgr import coSceneGraphItemParams
from Gui2Neg import theGuiMsgHandler
from ObjectMgr import ObjectMgr
from KeydObject import TYPE_SCENEGRAPH_ITEM
from Utils import ParamsDiff, fillShaderList, selectInShaderList
import copy
import os

from printing import InfoPrintCapable


_infoer = InfoPrintCapable()
_infoer.doPrint = False #True #

#id for coloring option
NO_COLOR = 0
RGB_COLOR = 1
MATERIAL = 2
VARIABLE = 3


class SceneGraphItemPanel(QtWidgets.QWidget,Ui_SceneGraphItemPanelBase, TransformManager):
    def __init__(self, parent=None):
        _infoer.function = str(self.__init__)
        _infoer.write("")
        QtWidgets.QWidget.__init__(self, parent)
        Ui_SceneGraphItemPanelBase.__init__(self)
        self.setupUi(self)
        TransformManager.__init__(self, self.emitDataChanged, True)

        #remove Tranformpanel till it is working
        #self.TabWidgetGeneralAdvanced.removeTab(2)
        #disable Color (only material coloring is working for VRML)
        #self.vrpRadioButtonColorRGB.setEnabled(False)

        self.floatInRangeTrans.setRange([0.0, 1.0])

        self.TabWidgetGeneralAdvanced.setCurrentIndex(0) # ignore index set by the designer (usually no one cares about the active index when editing ui-files)

        # list of associated keys from same type
        self.__keys = []

        #default setting
        self.floatInRangeShininess_2.setRange([0,40.0])
        self.floatInRangeShininess_2.setValue(16.0)
        self.floatInRangeTrans.setRange([0,1.0])
        self.floatInRangeTrans.setValue(1.0)
        self.__r = 200
        self.__g = 200
        self.__b = 200
        self.__ambient = [180,180,180]
        self.__specular = [255,255,130]
        self.color = NO_COLOR

        # for multi selection
        self.oldPanelParams = {}
        
        fillShaderList(self.shaderList)

        SceneGraphItemPanelConnector(self)

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
            if isinstance(params, int) or isinstance(params, coSceneGraphItemParams):
                self.__setParams( params )

        elif len(self.__keys) > 1 :
            # multi selection: show default params
            self.oldPanelParams = coSceneGraphItemParams()
            params = CopyParams(self.oldPanelParams)
            params.name = "Multiselection"
            self.__setParams( params )

    def __getParams(self):
        _infoer.function = str(self.__getParams)
        _infoer.write("")

        data = coSceneGraphItemParams()

        #get the coloring option
        if self.vrpRadioButtonNoColor.isChecked():
            data.color = NO_COLOR
        elif self.vrpRadioButtonColorRGB.isChecked():
            data.color = RGB_COLOR
        elif self.vrpRadioButtonColorMaterial_2.isChecked():
            data.color = MATERIAL
        #elif self.vrpRadioButtonColorVariable_2.isChecked():
            #data.color = VARIABLE

        # rgb
        data.r = self.__r
        data.g = self.__g
        data.b = self.__b
        # material
        data.ambient = self.__ambient
        data.specular = self.__specular
        data.shininess = self.floatInRangeShininess_2.getValue()

        #transparency
        data.transparency = self.floatInRangeTrans.getValue()
        data.transparencyOn = self.vrpCheckBoxTransparency.isChecked()

        data.isMoveable = self.vrpCheckBoxIsMoveable.isChecked()
        data.isMoveSelected = self.vrpCheckBoxIsMoveSelected.isChecked()

        data.name = str(self.nameWidget.text())

        data.nodeClassName = str(self.nodeTypeWidget.text())
        
        # shader
        data.shader = str(self.shaderList.currentItem().text())

        #transform card
        self.TransformManagerGetParams(data)

        return data

    def __setParams( self, params ):
        _infoer.function = str(self.__setParams)
        _infoer.write("")

        SceneGraphItemPanelBlockSignals(self, True)
        self.TransformManagerBlockSignals(True)

        if isinstance( params, int):
            self.__keys[0] = params
            return

        self.nameWidget.setText(params.name)
        self.nodeTypeWidget.setText(params.nodeClassName)
        # enable coloring-tab only for geodes or multiselection
        if params.nodeClassName == "Geode" or len(self.__keys) > 1:
            self.groupBox.setEnabled(True)
        else:
            self.groupBox.setEnabled(False)

        # TODO: these parameters are currently not used
        # rgb color
        self.__r=params.r
        self.__g=params.g
        self.__b=params.b
        # material
        self.floatInRangeShininess_2.setValue(params.shininess)
        self.__ambient = params.ambient
        self.__specular = params.specular

        # transparency
        self.vrpCheckBoxTransparency.setChecked(params.transparencyOn)
        self.floatInRangeTrans.setEnabled(params.transparencyOn)
        self.floatInRangeTrans.setValue(params.transparency)

        # set the radio buttons
        self.color = params.color
        if self.color == NO_COLOR:
            self.vrpRadioButtonNoColor.setChecked(True)
            self.vrpRadioButtonColorRGB.setChecked(False)
            self.vrpRadioButtonColorMaterial_2.setChecked(False)
            self.__changeNoColor(True)
            self.__changeRGB(False)
            self.__changeMaterial(False)
        elif self.color == RGB_COLOR:
            self.vrpRadioButtonNoColor.setChecked(False)
            self.vrpRadioButtonColorRGB.setChecked(True)
            self.vrpRadioButtonColorMaterial_2.setChecked(False)
            self.__changeNoColor(False)
            self.__changeRGB(True)
            self.__changeMaterial(False)
        elif self.color == MATERIAL:
            self.vrpRadioButtonNoColor.setChecked(False)
            self.vrpRadioButtonColorRGB.setChecked(False)
            self.vrpRadioButtonColorMaterial_2.setChecked(True)
            self.__changeNoColor(False)
            self.__changeRGB(False)
            self.__changeMaterial(True)
        elif self.color == VARIABLE:
            self.vrpRadioButtonNoColor.setChecked(False)
            self.vrpRadioButtonColorRGB.setChecked(False)
            self.vrpRadioButtonColorMaterial_2.setChecked(False)
            self.__changeNoColor(False)
            self.__changeRGB(False)
            self.__changeMaterial(False)

        # set moveable checkbox
        self.vrpCheckBoxIsMoveable.setChecked(params.isMoveable)

        # set checkbox for selection in renderer
        self.vrpCheckBoxIsMoveSelected.setChecked(params.isMoveSelected)
        self.vrpCheckBoxIsMoveSelected.setEnabled(params.isMoveable)

        #transform card
        self.TransformManagerSetParams(params)
        
        # shader
        selectInShaderList(self.shaderList, params.shaderFilename)

        self.TransformManagerBlockSignals(False)
        SceneGraphItemPanelBlockSignals(self, False)

        # for multi selection
        if len(self.__keys)>1 :
            self.oldPanelParams = params

    # select a new rgb color
    def emitColorRGB(self):
        _infoer.function = str(self.emitColorRGB)
        _infoer.write("")
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(self.__r,self.__g,self.__b),self)
        if color.isValid():
            self.__r=color.red()
            self.__g=color.green()
            self.__b =color.blue()
            self.emitDataChanged()

    # select diffuse color of material
    def emitColorDiffuse(self):
        _infoer.function = str(self.emitColorDiffuse)
        _infoer.write("")
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(self.__r,self.__g,self.__b),self)
        if color.isValid():
            self.__r=color.red()
            self.__g=color.green()
            self.__b =color.blue()
            self.emitDataChanged()

    # select ambient color of material
    def emitColorAmbient(self):
        _infoer.function = str(self.emitColorAmbient)
        _infoer.write("")
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(self.__r,self.__g,self.__b),self)
        if color.isValid():
            self.__ambient=(color.red(),color.green(),color.blue())
            self.emitDataChanged()

    # select specular color of material
    def emitColorSpecular(self):
        _infoer.function = str(self.emitColorSpecular)
        _infoer.write("")
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(self.__r,self.__g,self.__b),self)
        if color.isValid():
            self.__specular=(color.red(),color.green(),color.blue())
            self.emitDataChanged()

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

    def emitTransparencyChanged(self):
        _infoer.function = str(self.emitTransparencyChanged)
        _infoer.write("")
        for key in self.__keys:
            self.recursiveTransparency(key)

    def recursiveTransparency(self, key):
        params = Application.vrpApp.key2params[key]
        params.transparency = self.floatInRangeTrans.getValue()
        ObjectMgr().setParams( key, params )
        for childkey in ObjectMgr().getChildrenOfObject(key):
            self.recursiveTransparency(childkey)

    def emitTransChecked(self):
        _infoer.function = str(self.emitTransChecked)
        _infoer.write("")
        for key in self.__keys:
            self.recursiveTransparencyCB(key)
        if self.vrpCheckBoxTransparency.isChecked():
            self.floatInRangeTrans.setEnabled(True)
        else:
            self.floatInRangeTrans.setEnabled(False)

    def recursiveTransparencyCB(self, key):
        params = Application.vrpApp.key2params[key]
        params.transparencyOn = self.vrpCheckBoxTransparency.isChecked()
        ObjectMgr().setParams( key, params )
        for childkey in ObjectMgr().getChildrenOfObject(key):
            self.recursiveTransparencyCB(childkey)

    def emitShaderChanged(self):
        _infoer.function = str(self.emitShaderChanged)
        _infoer.write("")
        for key in self.__keys:
            self.recursiveShader(key)

    def recursiveShader(self, key):
        params = Application.vrpApp.key2params[key]
        params.shaderFilename = str(self.shaderList.currentItem().text())
        ObjectMgr().setParams( key, params )
        for childkey in ObjectMgr().getChildrenOfObject(key):
            self.recursiveShader(childkey)

    # the selected radiobutton has changed (original color/ rgb / material / variable)
    def emitChangedRadioGroup(self):
        _infoer.function = str(self.emitChangedRadioGroup)
        _infoer.write("")
        if self.vrpRadioButtonNoColor.isChecked():
            self.__changeNoColor(True)
            self.__changeRGB(False)
            self.__changeMaterial(False)
            self.emitDataChanged()
        if self.vrpRadioButtonColorRGB.isChecked():
            self.__changeNoColor(False)
            self.__changeRGB(True)
            self.__changeMaterial(False)
            self.emitDataChanged()
        elif self.vrpRadioButtonColorMaterial_2.isChecked():
            self.__changeNoColor(False)
            self.__changeRGB(False)
            self.__changeMaterial(True)
            self.emitDataChanged()

    # enables and disables different coloroptions
    def __changeNoColor(self, on):
        _infoer.function = str(self.__changeNoColor)
        _infoer.write("")
        self.color = NO_COLOR
    def __changeRGB(self, on):
        _infoer.function = str(self.__changeRGB)
        _infoer.write("")
        self.color = RGB_COLOR
        self.vrpPushButtonColorRGB.setEnabled(on)
    def __changeMaterial(self, on):
        _infoer.function = str(self.__changeMaterial)
        _infoer.write("")
        self.color = MATERIAL
        self.vrpPushButtonAmbientRGB_2.setEnabled(on)
        self.vrpPushButtonDiffuseRGB_2.setEnabled(on)
        self.vrpPushButtonSpecularRGB_2.setEnabled(on)
        self.textLabel4_4_3_4_2.setEnabled(on)
        self.floatInRangeShininess_2.setEnabled(on)

