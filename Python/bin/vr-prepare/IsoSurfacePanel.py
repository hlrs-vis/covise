
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

from GenericVisualizerPanelBase import Ui_GenericVisualizerPanelBase
from IsoSurfacePanelConnector import IsoSurfacePanelBlockSignals, IsoSurfacePanelConnector
from Gui2Neg import theGuiMsgHandler
from PartIsoSurfaceVis import PartIsoSurfaceVisParams
from VRPCoviseNetAccess import theNet

from ObjectMgr import ObjectMgr

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

from vtrans import coTranslate 

PLANE = 2
LINE = 3

class IsoSurfacePanel(QtWidgets.QWidget,Ui_GenericVisualizerPanelBase):

    """For controling parameters of a iso surface """

    sigVisibiliyToggled = pyqtSignal()
    sigEditColorMap = pyqtSignal()

    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        Ui_GenericVisualizerPanelBase.__init__(self)
        self.setupUi(self)
        #current object key
        self.__key = -1

        # remove unneccessary tabs
        self.TabWidgetGeneralAdvanced.removeTab(self.TabWidgetGeneralAdvanced.indexOf(self.tabAdjustmentStreamline))
        self.TabWidgetGeneralAdvanced.removeTab(self.TabWidgetGeneralAdvanced.indexOf(self.tabAdjustmentCuttingSurface))
        self.TabWidgetGeneralAdvanced.removeTab(self.TabWidgetGeneralAdvanced.indexOf(self.tabAdvanced))
        self.TabWidgetGeneralAdvanced.removeTab(self.TabWidgetGeneralAdvanced.indexOf(self.tabClipPlane))
        self.TabWidgetGeneralAdvanced.setCurrentIndex(0) # ignore index set by the designer (usually no one cares about the active index when editing ui-files)

        #default setting of the panel
        self.vrpCheckBoxMapVariable.setEnabled(True)
        self.vrpComboBoxVariable.setEnabled(True)
        #self.vrpToolButtonChain.hide()
        self.floatInRangeEndPointX.hide()
        self.floatInRangeEndPointY.hide()
        self.floatInRangeEndPointZ.hide()
        self.textEndPointX.hide()
        self.textEndPointY.hide()
        self.textEndPointZ.hide()
        self.checkBoxFreeStartpoints.hide()
        self.vrpPushButtonAdjustPreview.hide()
        self.visibilityCheckBox.setVisible(covise.coConfigIsOn("vr-prepare.AdditionalVisibilityCheckbox", False))

        self.__baseVariable = None

        #temporary storage of this parameter part
        self.colorCreator = None

        middleFloatInRangeCtrls = [self.floatInRangeX]
        self.__boundingBox = ((-1, 1), (-1, 1), (-1, 1))
        for w, r in zip(middleFloatInRangeCtrls, self.__boundingBox):
            w.setRange(r)
        sideRange = 0, self.__heuristicProbeMaxSideLengthFromBox(
            self.__boundingBox)

        IsoSurfacePanelConnector(self)

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
        if self.__vector==PLANE:
            data = PartIsoSurfaceVisParams()

        data.vector = self.__vector
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
        data.isovalue = self.floatInRangeX.getValue()
        data.isomin = self.floatInRangeX.getRange()[0]
        data.isomax = self.floatInRangeX.getRange()[1]
        data.boundingBox = self.__boundingBox
        return data

    def __panelAccordingTaskType( self ):
        self.checkBox5.hide()
        self.vrpGroupBoxPositioning.setTitle(self.__tr('Iso Value'))
        self.textLabel4_4_3.setText(self.__tr('Iso Value:'))
        self.textLabel4_4_2_3.hide()
        self.vrpLabelTitle.setText(self.__tr("Edit IsoSurface"))
        self.floatInRangeY.hide()
        self.textLabel4_4_2_2_2.hide()
        self.floatInRangeZ.hide()
        self.textLabel4_3_3.hide()
        self.floatInRangeWidth.hide()
        self.textLabel4_3_2_3.hide()
        self.floatInRangeHeight.hide()
        self.groupOrientation.hide()
        self.textLabelRotX.hide()
        self.floatInRangeRotX.hide()
        self.textLabelRotY.hide()
        self.floatInRangeRotY.hide()
        self.textLabelRotZ.hide()
        self.floatInRangeRotZ.hide()
        self.checkBoxDomainFromList.hide()
        self.comboBoxDomain.hide()
        self.checkBoxProjectArrows.hide()
        self.vrpCheckBoxShow.hide()
        #self.TabWidgetGeneralAdvanced.changeTab(self.tab, "General")
        #self.TabWidgetGeneralAdvanced.setTabEnabled(self.TabPage, False)
        #self.TabWidgetGeneralAdvanced.setTabEnabled(self.tab_2, False)
        #self.TabWidgetGeneralAdvanced.setTabEnabled(self.TabClipPlane, False)

        self.textLabel4_4_3.setToolTip(self.__tr("Set the isovalue."))
        self.textLabel4_4_3.setWhatsThis(self.__tr("Set the isovalue."))

    def __set2ndVariable( self, varlist):
        """ fill the combobox to choose a variable to be mapped on the trace """
        self.vrpComboBoxVariable.clear()
        self.vrpComboBoxVariable.setEnabled( len(varlist)!=0 )
        self.vrpCheckBoxMapVariable.setEnabled( len(varlist)!=0 )
        for v in varlist:
            self.vrpComboBoxVariable.addItem(v)

    def __setParams( self, params ):
        ''' updates the panel with the params of the negotiatior'''
        IsoSurfacePanelBlockSignals(self, True)

        if isinstance( params, int):
            self.__key = params
            return

        self.__vector = params.vector
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

        self.__baseVariable = currentVariable
        currentColorTableKey = None
        if currentVariable!=None and params.colorTableKey!=None and currentVariable in params.colorTableKey:
            currentColorTableKey = params.colorTableKey[currentVariable]
        MainWindow.globalColorManager.update( self.colorMapCombobox, currentVariable, currentColorTableKey)

        if hasattr(params.boundingBox, 'getXMin' ):
            self.__boundingBox = params.boundingBox
            maxSideLength = self.__boundingBox.getMaxEdgeLength()

        self.colorCreator=params.colorTableKey

        _infoer.write("setRange "+str(params.isomin)+" "+str(params.isomax))
        self.floatInRangeX.setRange([params.isomin, params.isomax])
        self.floatInRangeX.setValue(params.isovalue)

        IsoSurfacePanelBlockSignals(self, False)

    def emitNameChange(self, aQString=None):
        if self.__key!=-1:
            MainWindow.globalAccessToTreeView.setItemData(self.__key, str(self.nameWidget.text()))
            self.emitValueChange()

    def emitVisibilityToggled(self, b):
        self.sigVisibiliyToggled.emit((b,))
        self.emitValueChange(b)

    def emitCustomizeColorMap(self):
        self.sigEditColorMap.emit(self.__key, MainWindow.globalColorManager.getSelectedColormapKey( self.colorMapCombobox) )
        
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
