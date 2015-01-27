
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH


import math

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject, pyqtSignal
   
import covise
from printing import InfoPrintCapable

from Utils import SliderForFloatManager

import Application
import MainWindow

from IsoCutterPanelBase import Ui_IsoCutterPanelBase
from IsoCutterPanelConnector import IsoCutterPanelBlockSignals, IsoCutterPanelConnector
from Gui2Neg import theGuiMsgHandler
from PartIsoCutterVis import PartIsoCutterVisParams

from KeydObject import globalKeyHandler
from ObjectMgr import ObjectMgr
from BoundingBox import Box

_logger = InfoPrintCapable()
_logger.doPrint = False # True
_logger.startString = '(log)'
_logger.module = __name__

from vtrans import coTranslate 

def _log(func):
    def logged_func(*args, **kwargs):
        _logger.function = repr(func)
        _logger.write('')
        return func(*args, **kwargs)
    return logged_func

_infoer = InfoPrintCapable()
_infoer.doPrint = False # True#

PLANE = 2
LINE = 3

class IsoCutterPanel(QtWidgets.QWidget,Ui_IsoCutterPanelBase):

    """For controling parameters of a iso surface """

    sigVisibiliyToggled = pyqtSignal()
    sigEditColorMap = pyqtSignal()
    
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        Ui_IsoCutterPanelBase.__init__(self)
        self.setupUi(self)
        #current object key
        self.__key = -1
        
        #default setting of the panel
        self.vrpCheckBoxMapVariable.setEnabled(False)
        self.vrpComboBoxVariable.setEnabled(False)
        self.visibilityCheckBox.setVisible(covise.coConfigIsOn("vr-prepare.AdditionalVisibilityCheckbox", False))
            
        self.__baseVariable = None
        
        #temporary storage of this parameter part
        self.colorCreator = {}

        middleFloatInRangeCtrls = [self.floatInRangeIsoValue]
        self.__boundingBox = Box()
        for w, r in zip(middleFloatInRangeCtrls, self.__boundingBox.getTuple()):
            w.setRange(r)
        sideRange = 0, self.__heuristicProbeMaxSideLengthFromBox(
            self.__boundingBox.getTuple())
            
        IsoCutterPanelConnector(self)
            
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
        data = PartIsoCutterVisParams()

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
        data.isovalue = self.floatInRangeIsoValue.getValue()
        data.cutoff_side = self.checkBoxCutoffSide.isChecked()
        data.isomin = self.floatInRangeIsoValue.getRange()[0]
        data.isomax = self.floatInRangeIsoValue.getRange()[1]
        data.boundingBox = self.__boundingBox
        return data

    def __panelAccordingTaskType( self ):
        pass

    def __set2ndVariable( self, varlist):
        """ fill the combobox to choose a variable to be mapped on the trace """
        self.vrpComboBoxVariable.clear()
        self.vrpComboBoxVariable.setEnabled( len(varlist)!=0 )
        self.vrpCheckBoxMapVariable.setEnabled( len(varlist)!=0 )
        for v in varlist:
            self.vrpComboBoxVariable.addItem(v)

    def __setParams( self, params ):
        ''' updates the panel with the params of the negotiatior'''
        IsoCutterPanelBlockSignals(self, True)

        if isinstance( params, int):
            self.__key = params
            return
            
        #self.__vector = params.vector
        self.__panelAccordingTaskType()
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

        self.floatInRangeIsoValue.setRange([params.isomin, params.isomax])
        self.floatInRangeIsoValue.setValue(params.isovalue)
        self.checkBoxCutoffSide.setChecked(params.cutoff_side)

        IsoCutterPanelBlockSignals(self, False)

    def emitNameChange(self, aQString=None):
        if self.__key!=-1:
            MainWindow.globalAccessToTreeView.setItemData(self.__key, str(self.nameWidget.text()))
            self.emitValueChange()

    def emitVisibilityToggled(self, b):
        self.sigVisibiliyToggled.emit((b,))
        self.emitValueChange(b)

    def emitCustomizeColorMap(self):
        self.sigEditColorMap.emit( self.__key, MainWindow.globalColorManager.getSelectedColormapKey( self.colorMapCombobox) )

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
