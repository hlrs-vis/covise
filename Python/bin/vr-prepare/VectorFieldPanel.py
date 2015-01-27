
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH


import math

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject, pyqtSignal
   
import covise
from printing import InfoPrintCapable

from Utils import SliderForFloatManager, getDoubleInLineEdit

import Application
import MainWindow

from VectorFieldPanelBase import Ui_VectorFieldPanelBase
from VectorFieldPanelConnector import VectorFieldPanelBlockSignals, VectorFieldPanelConnector
from TransformManager import TransformManager
from Gui2Neg import theGuiMsgHandler
from PartVectorFieldVis import PartVectorFieldVisParams, RGB_COLOR, COLOR_MAP

from KeydObject import globalKeyHandler
from ObjectMgr import ObjectMgr

from vtrans import coTranslate

_logger = InfoPrintCapable()
_logger.doPrint = False # True
_logger.startString = '(log)'
_logger.module = __name__

def _log(func):
    def logged_func(*args, **kwargs):
        _logger.function = repr(func)
        _logger.write('')
        return func(*args, **kwargs)
    return logged_func

_infoer = InfoPrintCapable()
_infoer.doPrint = False # False

PLANE = 2
LINE = 3

class VectorFieldPanel(QtWidgets.QWidget,Ui_VectorFieldPanelBase, TransformManager):

    """For controling parameters of a iso interval surface """

    sigVisibiliyToggled = pyqtSignal()
    sigEditColorMap = pyqtSignal()
    
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        Ui_VectorFieldPanelBase.__init__(self)
        self.setupUi(self)
        TransformManager.__init__(self, self.emitTransformChanged)

        #current object key
        self.__key = -1

        # for custom RGB color
        self.__r = None
        self.__g = None
        self.__b = None
        self.__coloringOption = None

        #default setting of the panel
        self.vrpCheckBoxMapVariable.setEnabled(False)
        self.vrpComboBoxVariable.setEnabled(False)
        self.vrpLineEditVariable.hide()

        self.__baseVariable = None

        #temporary storage of this parameter part
        self.colorCreator = None

#        middleFloatInRangeCtrls = [self.floatInRangeScalingValue]
#        self.__boundingBox = ((-1, 1), (-1, 1), (-1, 1))
#        for w, r in zip(middleFloatInRangeCtrls, self.__boundingBox):
#            w.setRange(r)
#        sideRange = 0, self.__heuristicProbeMaxSideLengthFromBox(self.__boundingBox)

        VectorFieldPanelConnector(self)

        self.visibilityCheckBox.setVisible(covise.coConfigIsOn("vr-prepare.AdditionalVisibilityCheckbox", False))

        # set up initial radio group status
        self.emitChangedRadioGroup()
            
    def setSelectedColormap( self, callerKey, key, name):
        # new colormap was selected in color manager
        if (self.__key == callerKey):
            if MainWindow.globalColorManager.setSelectedColormapKey( self.colorMapCombobox, key ):
                self.emitValueChange()
            
    def paramChanged(self, key):
        ''' params of the object with key changed '''
        if self.__key == key:
            self.update()
            
    def update(self):
        if self.__key != -1:
            self.updateForObject( self.__key )
            
    def updateForObject( self, key ):
        """ called from MainWindow to update the content to the choosen object key """
        self.__key = key
        params = ObjectMgr().getParamsOfObject(key)
        self.__set2ndVariable( ObjectMgr().getPossibleScalarVariablesForVisItem(key) )
        self.__setParams( params )
        
    def __getParams(self):
        ''' convert information in the panel into the right negotiator param classes '''
        data = PartVectorFieldVisParams()

        #data.vector = self.__vector
        data.name = str(self.nameWidget.text())
        data.isVisible = self.visibilityCheckBox.isChecked()
        data.variable = str(self.vrpLineEditVariable.text())
        data.colorTableKey = self.colorCreator
        if self.vrpCheckBoxMapVariable.isChecked():
            data.secondVariable = str(self.vrpComboBoxVariable.currentText())
            if data.secondVariable!="" and data.colorTableKey!=None and data.secondVariable in data.colorTableKey and self.__baseVariable and self.__baseVariable==data.secondVariable:
                data.colorTableKey[data.secondVariable] = MainWindow.globalColorManager.getSelectedColormapKey( self.colorMapCombobox)
        else :
            if self.__baseVariable and self.__baseVariable==data.variable and data.colorTableKey:
                data.colorTableKey[data.variable] = MainWindow.globalColorManager.getSelectedColormapKey( self.colorMapCombobox)

        data.scalingValue = getDoubleInLineEdit(self.scalingValue)
        data.scalingType = self.comboBoxScalingType.currentIndex()
        data.arrowHeadFactor = getDoubleInLineEdit(self.arrowHeadFactor)

        data.r = self.__r
        data.g = self.__g
        data.b = self.__b
        data.coloringOption = self.__coloringOption

        data.boundingBox = self.__boundingBox
        
        #transform card
        self.TransformManagerGetParams(data)

        return data

    def __set2ndVariable( self, varlist):
        """ fill the combobox to choose a variable to be mapped on the trace """
        self.vrpComboBoxVariable.clear()
        self.vrpComboBoxVariable.setEnabled( len(varlist)!=0 )
        self.vrpCheckBoxMapVariable.setEnabled( len(varlist)!=0 )
        for v in varlist:
            self.vrpComboBoxVariable.addItem(v)

    def __setParams( self, params ):
        ''' updates the panel with the params of the negotiatior'''
        VectorFieldPanelBlockSignals(self, True)
        self.TransformManagerBlockSignals(True)

        if isinstance( params, int):
            self.__key = params
            return

        #self.__vector = params.vector
        self.nameWidget.setText(params.name)

        if hasattr(params, 'isVisible' ): self.visibilityCheckBox.setChecked(params.isVisible)
        if hasattr(params, 'variable' ): self.vrpLineEditVariable.setText(params.variable)
        if hasattr(params, 'secondVariable') and params.secondVariable!=None:
            self.vrpCheckBoxMapVariable.setChecked(True)
            self.vrpComboBoxVariable.setCurrentIndex(self.vrpComboBoxVariable.findText(params.secondVariable))
            currentVariable = params.secondVariable
        else:
            self.vrpCheckBoxMapVariable.setChecked(False)
            currentVariable = params.variable
        
        currentVariable = params.variable # no second variable in this visualizer right now

        self.__baseVariable = currentVariable
        currentColorTableKey = None
        if currentVariable!=None and params.colorTableKey!=None and currentVariable in params.colorTableKey:
            currentColorTableKey = params.colorTableKey[currentVariable]            
        MainWindow.globalColorManager.update( self.colorMapCombobox, currentVariable, currentColorTableKey)        

        if hasattr(params.boundingBox, 'getXMin' ):
            self.__boundingBox = params.boundingBox
            maxSideLength = self.__boundingBox.getMaxEdgeLength()

        self.colorCreator=params.colorTableKey

        self.__r = params.r
        self.__g = params.g
        self.__b = params.b
        self.__coloringOption = params.coloringOption
        # change radio button checks according to coloring option
        if self.__coloringOption == RGB_COLOR:
            self.__changeRGBColor(True)
            self.__changeColorMap(False)
        elif self.__coloringOption == COLOR_MAP:
            self.__changeRGBColor(False)
            self.__changeColorMap(True)


        self.scalingValue.setText(str(params.scalingValue))
        self.comboBoxScalingType.setCurrentIndex(params.scalingType)
        self.arrowHeadFactor.setText(str(params.arrowHeadFactor))
        #self.floatInRangeIsoValueHigh.setRange([params.high_isomin, params.high_isomax])
        #self.floatInRangeIsoValueHigh.setValue(params.high_isovalue)

        #transform card
        self.TransformManagerSetParams(params)

        self.TransformManagerBlockSignals(False)
        VectorFieldPanelBlockSignals(self, False)

    def emitNameChange(self, aQString=None):
        if self.__key!=-1:
            MainWindow.globalAccessToTreeView.setItemData(self.__key, str(self.nameWidget.text()))
            self.emitValueChange()

    def emitVisibilityToggled(self, b):
        self.sigVisibiliyToggled.emit((b,))
        self.emitValueChange(b)

    def emitCustomizeColorMap(self):
        self.sigEditColorMap.emit( self.__key, MainWindow.globalColorManager.getSelectedColormapKey( self.colorMapCombobox) )

    def emitDataChanged(self):        
        if not self.__key==-1:
            #childKey = Application.App.guiKey2visuKey[self.__key]
            #params = self.__getParams()
            #Application.vrpApp.key2params[childKey] = params
            #ObjectMgr().setParams( childKey, params )
            #theGuiMsgHandler().runObject( childKey )

            #params = self.__getParams()
            Application.vrpApp.key2params[self.__key] = self.__getParams()#params
            ObjectMgr().setParams(self.__key, self.__getParams())#params)
            theGuiMsgHandler().runObject(self.__key)

    def emitTransformChanged(self):
        if not self.__key==-1:
            Application.vrpApp.key2params[self.__key] = self.__getParams()#params
            ObjectMgr().setParams(self.__key, self.__getParams())#params)        

    def emitEditRGBColor(self):
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(self.__r,self.__g,self.__b), self)
        if color.isValid():
            self.__r = color.red()
            self.__g = color.green()
            self.__b = color.blue()
            self.emitChangedColor()
            
    def emitChangedColor(self):
        if not self.__key==-1:
            Application.vrpApp.key2params[self.__key] = self.__getParams()#params
            ObjectMgr().setParams(self.__key, self.__getParams())#params)        

    def emitChangedRadioGroup(self):
        if self.radioButtonRGBColor.isChecked():
            self.__changeRGBColor(True)
            self.__changeColorMap(False)
            self.__coloringOption = RGB_COLOR
            self.emitChangedColor()
        elif self.radioButtonColorMap.isChecked():
            self.__changeRGBColor(False)
            self.__changeColorMap(True)
            self.__coloringOption = COLOR_MAP
            self.emitDataChanged()

    def __changeRGBColor(self, enabled):
        self.radioButtonRGBColor.setChecked(enabled)
        self.pushButtonEditRGBColor.setEnabled(enabled)

    def __changeColorMap(self, enabled):
        self.radioButtonColorMap.setChecked(enabled)
        self.colorMapCombobox.setEnabled(enabled)
        self.PushButtonCustomizeColorMap.setEnabled(enabled)

    def emitValueChange(self, val=False):
        if not self.__key==-1:
            Application.vrpApp.key2params[self.__key] = self.__getParams()
            ObjectMgr().setParams( self.__key, self.__getParams() )
            theGuiMsgHandler().runObject( self.__key )        

    def __heuristicProbeMaxSideLengthFromBox(self, bb):
        return math.sqrt(
            (bb[0][1] - bb[0][0]) * (bb[0][1] - bb[0][0])
            + (bb[1][1] - bb[1][0]) * (bb[1][1] - bb[1][0])
            + (bb[2][1] - bb[2][0]) * (bb[2][1] - bb[2][0]))


    def __tr(self,s,c = None):
        return coTranslate(s)

# eof
