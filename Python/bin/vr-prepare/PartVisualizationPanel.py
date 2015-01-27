
# Part of the vr-prepare program for dc

# Copyright (c) 2007 Visenso GmbH

import sys

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject, pyqtSignal

import Application
import MainWindow
import covise

from PartVisualizationPanelBase import Ui_PartVisualizationPanelBase
from PartVisualizationPanelConnector import PartVisualizationPanelBlockSignals, PartVisualizationPanelConnector
from TransformManager import TransformManager

from Visualization2DPanel import Visualization2DPanel
from Part2DRawVis import Part2DRawVisParams
from co2DPartMgr import co2DPartMgrParams
from co2DCutGeometryPartMgr import co2DCutGeometryPartMgrParams
from Gui2Neg import theGuiMsgHandler
from ObjectMgr import ObjectMgr
from KeydObject import TYPE_2D_PART, TYPE_2D_COMPOSED_PART, TYPE_2D_CUTGEOMETRY_PART
from RectangleManager import RectangleManager, CUTTINGSURFACE
from Utils import ParamsDiff, csvStrToList, fillShaderList, selectInShaderList, CopyParams

from printing import InfoPrintCapable

_infoer = InfoPrintCapable()
_infoer.doPrint = False #True #

#id for coloring option
NO_COLOR = 0
RGB_COLOR = 1
MATERIAL = 2
VARIABLE = 3



class PartVisualizationPanel(QtWidgets.QWidget,Ui_PartVisualizationPanelBase, Visualization2DPanel, TransformManager):
    sigEditColorMap = pyqtSignal()
    def __init__(self, parent=None):
        _infoer.function = str(self.__init__)
        _infoer.write("")
        QtWidgets.QWidget.__init__(self, parent)
        Ui_PartVisualizationPanelBase.__init__(self)
        self.setupUi(self)
        Visualization2DPanel.__init__(self)
        TransformManager.__init__(self, self.emitDataChangedTransform)

        # list of associated keys from same type
        self.__keys = []

        #default setting
        self.floatInRangeShininess_2.setRange([0,40.0])
        self.floatInRangeShininess_2.setValue(16.0)
        self.floatInRangeTrans.setRange([0,1.0])
        self.floatInRangeTrans.setValue(1.0)
        self.__floatParams = ''
        self.__intParams = ''
        self.__boolParams = ''
        self.__vec2Params = ''
        self.__vec3Params = ''
        self.__vec4Params = ''
        self.__mat2Params = ''
        self.__mat3Params = ''
        self.__mat4Params = ''
        self.__r = 200
        self.__g = 200
        self.__b = 200
        self.__ambient = [180,180,180]
        self.__specular = [255,255,130]
        self.color = NO_COLOR
        self.colorCreator = None
        self.vectorVariableNames = []
        self.scalarVariableNames = []
        self.__baseVariable = 'Select a Variable'
        self.variablesSet = False
        self.TabWidgetGeneralAdvanced.setCurrentIndex(0) # ignore index set by the designer (usually no one cares about the active index when editing ui-files)

        # for multi selection
        self.oldPanelParams = {}
        self.oldPanelRealParams = {}

        self.__rectangleManager = RectangleManager( self, self.emitCutChanged, self.emitCutChanged, None, CUTTINGSURFACE )

        fillShaderList(self.shaderList)

        PartVisualizationPanelConnector(self)

    def setSelectedColormap( self, callerKey, key, name ):
        _infoer.function = str(self.setSelectedColormap)
        _infoer.write("")
        # new colormap was selected in color manager
        if len(self.__keys)>0 and self.__keys[0]==callerKey:
            if MainWindow.globalColorManager.setSelectedColormapKey( self.vrpComboboxColorMap_2, key ):
                self.emitVariableChanged()

    def paramChanged(self, key):
        """ params of object key changed"""

        _infoer.function = str(self.paramChanged)
        _infoer.write("key %d" %key)
        #update comes from change within panel

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
        # if a type_2d_composed_part disable the first two tabs (coloring and shader)
        if len(self.__keys)==1 :
            # enable the visualization and the transform tab
            self.TabWidgetGeneralAdvanced.setTabEnabled(0, True)
            self.TabWidgetGeneralAdvanced.setTabEnabled(1, True)
            self.TabWidgetGeneralAdvanced.setTabEnabled(2, True)
            self.TabWidgetGeneralAdvanced.setTabEnabled(3, True)

            if ObjectMgr().getTypeOfObject(self.__keys[0]) == TYPE_2D_COMPOSED_PART:
                self.TabWidgetGeneralAdvanced.setTabEnabled(0, False)
                self.TabWidgetGeneralAdvanced.setTabEnabled(1, False)
            # if not a type_2d_part disable the transform tab
            if ObjectMgr().getTypeOfObject(self.__keys[0]) != TYPE_2D_PART:
                self.TabWidgetGeneralAdvanced.setTabEnabled(3, False)
            # if not a type_2d_cutgeometry_part disable the cut tab
            if ObjectMgr().getTypeOfObject(self.__keys[0]) != TYPE_2D_CUTGEOMETRY_PART:
                self.TabWidgetGeneralAdvanced.setTabEnabled(4, False)

        elif len(self.__keys) > 1 : # multi selection
            # always show first tab
            self.TabWidgetGeneralAdvanced.setCurrentIndex(0)
            # disable the visualization and the transform tab
            self.TabWidgetGeneralAdvanced.setTabEnabled(2, False)
            self.TabWidgetGeneralAdvanced.setTabEnabled(3, False)

        # set the variables of first key
        self.__setScalarVariables(ObjectMgr().getPossibleScalarVariablesForType(self.__keys[0]))
        self.__setVectorVariables(ObjectMgr().getPossibleVectorVariablesForType(self.__keys[0]))

        # apply params
        if len(self.__keys) == 1 :
            realparams = ObjectMgr().getRealParamsOfObject(self.__keys[0])
            params = ObjectMgr().getParamsOfObject(self.__keys[0])
            params.name = realparams.name
            #params.boundingBox = realparams.boundingBox
            if isinstance(params, int) or isinstance(params, Part2DRawVisParams):
                self.__setParams( params )
            self.__setRealParams( realparams )
            # update
            Visualization2DPanel.updateForObject(self, self.__keys[0])
        elif len(self.__keys) > 1 :
            # multi selection: show default params
            self.oldPanelParams = Part2DRawVisParams()
            params = CopyParams(self.oldPanelParams)
            self.__setParams( params )
            if ObjectMgr().getTypeOfObject(self.__keys[0]) == TYPE_2D_PART:
                self.oldPanelRealParams = co2DPartMgrParams()
            if ObjectMgr().getTypeOfObject(self.__keys[0]) == TYPE_2D_COMPOSED_PART:
                self.oldPanelRealParams = co2DComposedPartMgrParams()
            if ObjectMgr().getTypeOfObject(self.__keys[0]) == TYPE_2D_CUTGEOMETRY_PART:
                self.oldPanelRealParams = co2DCutGeometryPartMgrParams()
                #self.oldPanelRealParams.name = coTranslate("Multiselection")
                self.oldPanelRealParams.name = "Multiselection"
            realparams = CopyParams(self.oldPanelRealParams)
            self.__setRealParams( realparams )
            # set ComboBoxVariable on default value
            self.vrpComboBoxVariable_2.setCurrentIndex(0)

    def __getParams(self):
        _infoer.function = str(self.__getParams)
        _infoer.write("")

        data = Part2DRawVisParams()

        #get the coloring option
        if self.vrpRadioButtonNoColor.isChecked():
            data.color = NO_COLOR
        elif self.vrpRadioButtonColorRGB.isChecked():
            data.color = RGB_COLOR
        elif self.vrpRadioButtonColorMaterial_2.isChecked():
            data.color = MATERIAL
        elif self.vrpRadioButtonColorVariable_2.isChecked():
            data.color = VARIABLE

        # rgb
        data.r = self.__r
        data.g = self.__g
        data.b = self.__b
        # material
        data.ambient = self.__ambient
        data.specular = self.__specular
        data.shininess = self.floatInRangeShininess_2.getValue()
        # variable coloring
        data.allVariables = []
        for i in range(self.vrpComboBoxVariable_2.count()):
            data.allVariables.append(str(self.vrpComboBoxVariable_2.itemText(i)))
        data.variable = str(self.vrpComboBoxVariable_2.currentText())
        data.colorTableKey = self.colorCreator
        if self.__baseVariable and self.__baseVariable==data.variable and data.colorTableKey:
            data.colorTableKey[data.variable] = MainWindow.globalColorManager.getSelectedColormapKey( self.vrpComboboxColorMap_2)
        #transparency
        data.transparency = self.floatInRangeTrans.getValue()
        data.transparencyOn = self.vrpCheckBoxTransparency.isChecked()

        #transform card
        self.TransformManagerGetParams(data)
        
        # shader
        data.shaderFilename = str(self.shaderList.currentItem().text())

        data.isVisible = True
        return data

    def __setParams( self, params ):
        _infoer.function = str(self.__setParams)
        _infoer.write("")

        PartVisualizationPanelBlockSignals(self, True)
        self.TransformManagerBlockSignals(True)

        if isinstance( params, int):
            self.__keys[0] = params
            return

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

        if len(params.allVariables)>0:
            self.vrpComboBoxVariable_2.clear()
            index = 0
            for var in params.allVariables:
                self.vrpComboBoxVariable_2.insertItem(index, var)
                if var == params.variable:
                    self.vrpComboBoxVariable_2.setCurrentIndex(index)
                index+= 1

        # colortable
        currentVariable = params.variable
        self.__baseVariable = currentVariable
        currentColorTableKey = None
        if currentVariable!=None and params.colorTableKey!=None and currentVariable in params.colorTableKey:
            currentColorTableKey = params.colorTableKey[currentVariable]
        MainWindow.globalColorManager.update( self.vrpComboboxColorMap_2, currentVariable, currentColorTableKey )

        self.colorCreator=params.colorTableKey

        # set the radio buttons
        self.color = params.color
        if self.color == NO_COLOR:
            self.vrpRadioButtonNoColor.setChecked(True)
            self.vrpRadioButtonColorRGB.setChecked(False)
            self.vrpRadioButtonColorMaterial_2.setChecked(False)
            self.vrpRadioButtonColorVariable_2.setChecked(False)
            self.__changeNoColor(True)
            self.__changeRGB(False)
            self.__changeMaterial(False)
            self.__changeVariable(False)
        elif self.color == RGB_COLOR:
            self.vrpRadioButtonNoColor.setChecked(False)
            self.vrpRadioButtonColorRGB.setChecked(True)
            self.vrpRadioButtonColorMaterial_2.setChecked(False)
            self.vrpRadioButtonColorVariable_2.setChecked(False)
            self.__changeNoColor(False)
            self.__changeRGB(True)
            self.__changeMaterial(False)
            self.__changeVariable(False)
        elif self.color == MATERIAL:
            self.vrpRadioButtonNoColor.setChecked(False)
            self.vrpRadioButtonColorRGB.setChecked(False)
            self.vrpRadioButtonColorMaterial_2.setChecked(True)
            self.vrpRadioButtonColorVariable_2.setChecked(False)
            self.__changeNoColor(False)
            self.__changeRGB(False)
            self.__changeMaterial(True)
            self.__changeVariable(False)
        elif self.color == VARIABLE:
            self.vrpRadioButtonNoColor.setChecked(False)
            self.vrpRadioButtonColorRGB.setChecked(False)
            self.vrpRadioButtonColorMaterial_2.setChecked(False)
            self.vrpRadioButtonColorVariable_2.setChecked(True)
            self.__changeNoColor(False)
            self.__changeRGB(False)
            self.__changeMaterial(False)
            self.__changeVariable(True)

        #transform card
        self.TransformManagerSetParams(params)

        # shader
        selectInShaderList(self.shaderList, params.shaderFilename)

        self.TransformManagerBlockSignals(False)
        PartVisualizationPanelBlockSignals(self, False)

        # for multi selection
        if len(self.__keys)>1 :
            self.oldPanelParams = params

    # Note: RealParams is a special case for PartVisualizationPanel since we have two objects behind the panel.
    #       realparams correlates to co2DPartMgr and params to Part2DRawVis
    #       DON'T COPY THE REALPARAMS-MECHANISM TO OTHER PANELS.
    def __setRealParams( self, params ):
        _infoer.function = str(self.__setRealParams)
        _infoer.write("")
        PartVisualizationPanelBlockSignals(self, True)
        self.nameWidget.setText(params.name)
        if isinstance(params, co2DCutGeometryPartMgrParams):
            self.__rectangleManager.setRectangle( params.alignedRectangle )

        # bounding box is param of part not of visualizer, so set as real param
        if hasattr(params, 'boundingBox') and params.boundingBox:
           self.__rectangleManager.setBoundingBox( params.boundingBox )

        PartVisualizationPanelBlockSignals(self, False)

        # for multi selection
        if len(self.__keys)>1 :
            self.oldPanelRealParams = params

    # emitter for the qt buttons

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
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(self.__ambient[0],self.__ambient[1],self.__ambient[2]),self)
        if color.isValid():
            self.__ambient=(color.red(),color.green(),color.blue())
            self.emitDataChanged()

    # select specular color of material
    def emitColorSpecular(self):
        _infoer.function = str(self.emitColorSpecular)
        _infoer.write("")
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(self.__specular[0],self.__specular[1],self.__specular[2]),self)
        if color.isValid():
            self.__specular=(color.red(),color.green(),color.blue())
            self.emitDataChanged()

    # change the name of the appearance (only for single selection)
    def emitNameChange(self, aQString=None):
        _infoer.function = str(self.emitNameChange)
        _infoer.write("")
        # only for single selection
        if len(self.__keys)==1 :
            #MainWindow.globalAccessToTreeView.setItemData(self.__keys[0], str(self.nameWidget.text()))
            # set name of type_2d_part
            params = ObjectMgr().getRealParamsOfObject(self.__keys[0])
            params.name = str(self.nameWidget.text())
            Application.vrpApp.key2params[self.__keys[0]] = params
            ObjectMgr().setParams(self.__keys[0], params)
            self.emitDataChanged()

    # any data of Part2DRawVis has changed(works on multi-selection)
    def emitDataChanged(self):
        _infoer.function = str(self.emitDataChanged)
        _infoer.write("")

        if len(self.__keys)>0 and not ObjectMgr().getTypeOfObject(self.__keys[0]) == TYPE_2D_COMPOSED_PART:     ##all key types in self.keys should be the same
            # mapping of the keys for the object manager
            childKeys = [Application.vrpApp.guiKey2visuKey[k] for k in self.__keys]

            params = self.__getParams()

            if len(self.__keys)==1 :
                Application.vrpApp.key2params[childKeys[0]] = params
                ObjectMgr().setParams( childKeys[0], params )
                #theGuiMsgHandler().runObject( childKey )

            # set params for multi selection
            if len(self.__keys)>1 :
                #find changed params
                originalParams = self.oldPanelParams
                realChange = ParamsDiff( originalParams, params )

                # set params for remaining selected objects
                for i in range(0, len(self.__keys)):
                    childKeyParams = ObjectMgr().getParamsOfObject(childKeys[i])

                    # find the changed param in childKey and replace it with the
                    # intended attribut
                    for x in realChange :
                        childKeyParams.__dict__[x] = params.__dict__[x]
                    # set the params
                    Application.vrpApp.key2params[childKeys[i]] = childKeyParams
                    ObjectMgr().setParams( childKeys[i], childKeyParams )
                    #theGuiMsgHandler().runObject( childKeys[i] )
                #save params for multi selection
                self.oldPanelParams = self.__getParams()

    def emitDataChangedTransform(self):
        _infoer.function = str(self.emitDataChangedTransform)
        _infoer.write("")
        #TODO multi selection
        if len(self.__keys)==1 :
            if ObjectMgr().getTypeOfObject(self.__keys[0]) == TYPE_2D_PART:
                childKey = Application.vrpApp.guiKey2visuKey[self.__keys[0]]
                params = self.__getParams()
                Application.vrpApp.key2params[childKey] = params
                ObjectMgr().setParams( childKey, params )
                theGuiMsgHandler().runObject( childKey )


    # select a new variable for coloring
    def emitVariableChanged(self):
        _infoer.function = str(self.emitVariableChanged)
        _infoer.write("")

        if len(self.__keys)>0 and not ObjectMgr().getTypeOfObject(self.__keys[0]) == TYPE_2D_COMPOSED_PART:     ##all key types in self.keys should be the same
            # mapping of the keys for the object manager
            childKeys = []
            for i in range(0, len(self.__keys)):
                childKeys.append(Application.vrpApp.guiKey2visuKey[self.__keys[i]])

            params = self.__getParams()

            if len(self.__keys)==1 :
                Application.vrpApp.key2params[childKeys[0]] = self.__getParams()
                ObjectMgr().setParams( childKeys[0], params )
                reqId = theGuiMsgHandler().setParams( childKeys[0], params )
                theGuiMsgHandler().waitforAnswer(reqId)
                theGuiMsgHandler().runObject( childKeys[0] )
                self.__setParams( self.__getParams() )#necessary for allVariablesList

            # set params for multi selection
            if len(self.__keys)>1 :
                # find changed params
                originalParams = self.oldPanelParams
                realChange = ParamsDiff( originalParams, params )

                # set params for remaining selected objects
                for i in range(0, len(self.__keys)):
                    childKeyParams = ObjectMgr().getParamsOfObject(childKeys[i])

                    # find the changed param in childKey and replace it with the
                    # intended attribut
                    for x in realChange :
                        childKeyParams.__dict__[x] = params.__dict__[x]
                    # set the params
                    Application.vrpApp.key2params[childKeys[i]] = childKeyParams
                    ObjectMgr().setParams( childKeys[i], childKeyParams )
                    reqId = theGuiMsgHandler().setParams( childKeys[i], childKeyParams )
                    theGuiMsgHandler().waitforAnswer(reqId)
                    theGuiMsgHandler().runObject( childKeys[i] )
                #necessary for allVariablesList
                self.__setParams( self.__getParams() )
                #save params for multi selection
                self.oldPanelParams = self.__getParams()

    # the cut data has changed
    def emitCutChanged(self):
        _infoer.function = str(self.emitCutChanged)
        _infoer.write("")

        if len(self.__keys)>0 and ObjectMgr().getTypeOfObject(self.__keys[0]) == TYPE_2D_CUTGEOMETRY_PART:

            params = ObjectMgr().getRealParamsOfObject(self.__keys[0])
            params.alignedRectangle = self.__rectangleManager.getParams()

            if len(self.__keys)==1 :
                Application.vrpApp.key2params[self.__keys[0]] = params
                ObjectMgr().setParams(self.__keys[0], params )
                theGuiMsgHandler().runObject(self.__keys[0])

            # set params for multi selection
            if len(self.__keys)>1 :
                #find changed params
                originalParams = self.oldPanelRealParams
                realChange = ParamsDiff( originalParams, params )
                # set params for remaining selected objects
                for i in range(0, len(self.__keys)):
                    childKeyParams = ObjectMgr().getRealParamsOfObject(self.__keys[i])
                    # find the changed param in childKey and replace it with the
                    # intended attribut
                    for x in realChange :
                        childKeyParams.__dict__[x] = params.__dict__[x]
                    # set the params
                    Application.vrpApp.key2params[self.__keys[i]] = childKeyParams
                    ObjectMgr().setParams( self.__keys[i], childKeyParams )
                    theGuiMsgHandler().runObject( self.__keys[i] )
                #save params for multi selection
                self.oldPanelRealParams = CopyParams(params)


    # the selected radiobutton has changed (original color/ rgb / material / variable)
    def emitChangedRadioGroup(self):
        _infoer.function = str(self.emitChangedRadioGroup)
        _infoer.write("")
        if self.vrpRadioButtonNoColor.isChecked():
            self.__changeNoColor(True)
            self.__changeRGB(False)
            self.__changeMaterial(False)
            self.__changeVariable(False)
            self.emitDataChanged()
        if self.vrpRadioButtonColorRGB.isChecked():
            self.__changeNoColor(False)
            self.__changeRGB(True)
            self.__changeMaterial(False)
            self.__changeVariable(False)
            self.emitDataChanged()
        elif self.vrpRadioButtonColorMaterial_2.isChecked():
            self.__changeNoColor(False)
            self.__changeRGB(False)
            self.__changeMaterial(True)
            self.__changeVariable(False)
            self.emitDataChanged()
        elif self.vrpRadioButtonColorVariable_2.isChecked():
            self.__changeNoColor(False)
            self.__changeRGB(False)
            self.__changeMaterial(False)
            self.__changeVariable(True)
            if not self.vrpComboBoxVariable_2.currentText()=="Select a variable" and not self.vrpComboBoxVariable_2.currentText()=="":
                self.emitVariableChanged()


    # transparence checkbox has changed
    def emitTransChecked(self):
        _infoer.function = str(self.emitTransChecked)
        _infoer.write("")
        if self.vrpCheckBoxTransparency.isChecked():
            self.floatInRangeTrans.setEnabled(True)
        else:
            self.floatInRangeTrans.setEnabled(False)
        self.emitDataChanged()

    # colormap changes
    def emitCustomizeColorMap(self):
        _infoer.function = str(self.emitCustomizeColorMap)
        _infoer.write("")
        # only one popup is needed, so call for key[0]
        self.sigEditColorMap.emit(self.__keys[0], MainWindow.globalColorManager.getSelectedColormapKey( self.vrpComboboxColorMap_2 ) )


    # private helper functions for class
    # ----------------------------------#


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
    def __changeVariable(self, on):
        _infoer.function = str(self.__changeVariable)
        _infoer.write("")
        self.color = VARIABLE
        self.vrpComboBoxVariable_2.setEnabled(on)
        self.vrpLabelVariable_2.setEnabled(on)
        if not self.vrpComboBoxVariable_2.currentText()=="Select a variable":
            self.vrpComboboxColorMap_2.setEnabled(on)
            self.vrpLabelColorMap_2.setEnabled(on)
            self.vrpPushButtonCustomizeColorMap_2.setEnabled(on)
        else:
            self.vrpComboboxColorMap_2.setEnabled(False)
            self.vrpLabelColorMap_2.setEnabled(False)
            self.vrpPushButtonCustomizeColorMap_2.setEnabled(False)

    # sets the list of the variables and set the combobox
    def __setVectorVariables(self, aNameList):
        _infoer.function = str(self.__setVectorVariables)
        _infoer.write("")
        self.vectorVariableNames = aNameList
        self.__fillVariables()

    def __setScalarVariables(self, aNameList):
        _infoer.function = str(self.__setScalarVariables)
        _infoer.write("")
        self.scalarVariableNames = aNameList
        self.__fillVariables()

    #fill the combobox with the variables 
    def __fillVariables(self):
        _infoer.function = str(self.__fillVariables)
        _infoer.write("")
        self.variablesSet = True
        self.vrpComboBoxVariable_2.clear()
        self.vrpComboBoxVariable_2.addItem("Select a variable")
        self.varColor = {}
        for aName in self.vectorVariableNames:
            self.vrpComboBoxVariable_2.addItem(aName)
        for aName in self.scalarVariableNames:
            self.vrpComboBoxVariable_2.addItem(aName)
        self.__baseVariable = self.vrpComboBoxVariable_2.currentText()
        self.vrpComboboxColorMap_2.clear()

