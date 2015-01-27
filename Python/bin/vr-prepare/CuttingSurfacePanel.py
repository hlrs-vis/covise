
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH


import math

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject, pyqtSignal

import covise
from printing import InfoPrintCapable
_infoer = InfoPrintCapable()
_infoer.doPrint = False # False

from Utils import (
    AxisAlignedRectangleIn3d,
    SliderForFloatManager,
    getDoubleInLineEdit,
    getIntInLineEdit)

import Application
import MainWindow

from GenericVisualizerPanelBase import Ui_GenericVisualizerPanelBase
from CuttingSurfacePanelConnector import CuttingSurfacePanelBlockSignals, CuttingSurfacePanelConnector
from RectangleManager import RectangleManager, CUTTINGSURFACE
from Gui2Neg import theGuiMsgHandler
from PartCuttingSurfaceVis import PartPlaneVisParams, PartVectorVisParams
from VRPCoviseNetAccess import theNet

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

from vtrans import coTranslate 

PLANE = 2
VECTOR = 3

class CuttingSurfacePanel(QtWidgets.QWidget,Ui_GenericVisualizerPanelBase):

    """ For controling parameters of a cutting surface """
    sigVisibiliyToggled  = pyqtSignal()
    sigEditColorMap  = pyqtSignal()
    
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        Ui_GenericVisualizerPanelBase.__init__(self)
        self.setupUi(self)
        #current object
        self.__key = -1
        
        self.vrpPushButtonAdjustPreview.hide()
        self.floatInRangeRotX.setEnabled(False)
        self.floatInRangeRotY.setEnabled(True)
        self.floatInRangeRotZ.setEnabled(True)
        self.floatInRangeEndPointX.hide()
        self.floatInRangeEndPointY.hide()
        self.floatInRangeEndPointZ.hide()
        self.textEndPointX.hide()
        self.textEndPointY.hide()
        self.textEndPointZ.hide()
        self.vrpCheckBoxShow.setText(self.__tr("Show Preview"))
        self.vrpGroupBoxPositioning.setTitle(self.__tr("Plane Positioning")) 
        self.checkBoxFreeStartpoints.hide()
        self.vrpCheckBoxMapVariable.hide()
        self.vrpComboBoxVariable.hide()
        self.textLabel4_3_3.hide()
        self.floatInRangeWidth.hide()
        self.textLabel4_3_2_3.hide()
        self.floatInRangeHeight.hide()
        self.checkBoxDomainFromList.hide()
        self.comboBoxDomain.hide()
        self.checkBoxFreeStartpoints.hide()
        self.groupOrientation.setTitle(self.__tr("Normal:"))

        # change tooltips
        self.textLabel4_4_3.setToolTip(self.__tr("x, y, z define one point on the plane."))
        self.textLabel4_4_3.setWhatsThis(self.__tr("x, y, z define one point on the plane."))
        self.textLabel4_4_2_3.setToolTip(self.__tr("x, y, z define one point on the plane."))
        self.textLabel4_4_2_3.setWhatsThis(self.__tr("x, y, z define one point on the plane."))
        self.textLabel4_4_2_2_2.setToolTip(self.__tr("x, y, z define one point on the plane."))
        self.textLabel4_4_2_2_2.setWhatsThis(self.__tr("x, y, z define one point on the plane."))
        self.xAxisRadioButton.setToolTip(self.__tr("Orientation defines the normal of the plane."))
        self.xAxisRadioButton.setWhatsThis(self.__tr("Orientation defines the normal of the plane."))
        self.yAxisRadioButton.setToolTip(self.__tr("Orientation defines the normal of the plane."))
        self.yAxisRadioButton.setWhatsThis(self.__tr("Orientation defines the normal of the plane."))
        self.zAxisRadioButton.setToolTip(self.__tr("Orientation defines the normal of the plane."))
        self.zAxisRadioButton.setWhatsThis(self.__tr("Orientation defines the normal of the plane."))
        self.textLabelRotX.setToolTip(self.__tr("For finetuning the orientation<br> of the plane, you can add rotations around each axis."))
        self.textLabelRotX.setWhatsThis(self.__tr("For finetuning the orientation<br> of the plane, you can add rotations around each axis."))
        self.textLabelRotY.setToolTip(self.__tr("For finetuning the orientation<br> of the plane, you can add rotations around each axis."))
        self.textLabelRotY.setWhatsThis(self.__tr("For finetuning the orientation<br> of the plane, you can add rotations around each axis."))
        self.textLabelRotZ.setToolTip(self.__tr("For finetuning the orientation<br> of the plane, you can add rotations around each axis."))
        self.textLabelRotZ.setWhatsThis(self.__tr("For finetuning the orientation<br> of the plane, you can add rotations around each axis."))

        self.TabWidgetGeneralAdvanced.removeTab(self.TabWidgetGeneralAdvanced.indexOf(self.tabAdjustmentStreamline))
        self.TabWidgetGeneralAdvanced.removeTab(self.TabWidgetGeneralAdvanced.indexOf(self.tabAdvanced))
        self.TabWidgetGeneralAdvanced.setCurrentIndex(0) # ignore index set by the designer (usually no one cares about the active index when editing ui-files)

        self.visibilityCheckBox.setVisible(covise.coConfigIsOn("vr-prepare.AdditionalVisibilityCheckbox", False))

        # manager handling the starting rectangle
        self.__rectangleManager = RectangleManager( self, self.emitPlaneParametersApply, self.emitPlaneParametersApply, self.emitRectangleChange, CUTTINGSURFACE )

        self.__vector = PLANE
        self.__baseVariable = None
        
        #temporary storage of this parameter part       
        self.colorCreator = None

        #qt connections and settings
        CuttingSurfacePanelConnector(self)
        
    def setSelectedColormap( self, callerKey, key, name):
        # new colormap was selected in color manager
        if (self.__key == callerKey):
            if MainWindow.globalColorManager.setSelectedColormapKey( self.colorMapCombobox, key ):
                self.emitPlaneParametersApply()


    def paramChanged( self, key ):
        """ params of object key changed"""
        if self.__key==key:
            self.update()
            
    def update( self ):
        if self.__key!=-1:
            self.updateForObject( self.__key )
                
    def updateForObject( self, key ):
        """ called from MainWindow to update the content to the choosen object key """
        self.__key = key
        params = ObjectMgr().getParamsOfObject(key)      
        self.__setParams( params )

    def __getParams(self):
        """
            convert information in the panel into the right negotiator param classes
        """    
        if self.__vector==PLANE:
            data = PartPlaneVisParams()
        elif self.__vector==VECTOR:
            data = PartVectorVisParams()
            data.length = self.cuttingSurfaceLength.currentIndex()+1
            data.scale = getDoubleInLineEdit(self.cuttingSurfaceScale)
            data.arrow_head_factor = getDoubleInLineEdit(self.cuttingSurfaceArrowHead)
            data.project_arrows = self.checkBoxProjectArrows.isChecked()
        data.vector = self.__vector
        data.name = str(self.nameWidget.text())
        data.isVisible = self.visibilityCheckBox.isChecked()
        data.variable = str(self.vrpLineEditVariable.text())
        data.colorTableKey = self.colorCreator
        if self.__baseVariable and self.__baseVariable==data.variable and data.colorTableKey:
            data.colorTableKey[data.variable] = MainWindow.globalColorManager.getSelectedColormapKey( self.colorMapCombobox)
        data.alignedRectangle = self.__rectangleManager.getParams()
        data.boundingBox = self.__rectangleManager.getBoundingBox()
        data.showSmoke= self.vrpCheckBoxShow.isChecked()
        data.showInteractor = self.checkBox5.isChecked()
        data.attachedClipPlane_index = self.ClipPlaneIndexCombo.currentIndex()-1
        data.attachedClipPlane_offset = getDoubleInLineEdit(self.vrpLineEditClipPlaneOffset)
        data.attachedClipPlane_flip = self.ClipPlaneFlipCheckbox.isChecked()
        return data

    def __panelAccordingTaskType( self ):
        """
            change the panel according the needs of the visualizer
            NOTE: changes identical for all types should be done in __init__
        """   

        if self.__vector==PLANE:
            self.TabWidgetGeneralAdvanced.setTabEnabled(self.TabWidgetGeneralAdvanced.indexOf(self.tabAdjustmentCuttingSurface), False)
            text = "Edit Plane on Cutting Surface:"
            ColoredCuttingSurfaceText = covise.getCoConfigEntry("vr-prepare.ColoredCuttingSurfaceText")
            if ColoredCuttingSurfaceText:
                text = ColoredCuttingSurfaceText
            self.vrpLabelTitle.setText(self.__tr(text))
            self.vrpCheckBoxShow.show()
        if self.__vector==VECTOR:
            self.vrpCheckBoxShow.hide()
            self.TabWidgetGeneralAdvanced.setTabEnabled(self.TabWidgetGeneralAdvanced.indexOf(self.tabAdjustmentCuttingSurface), True)
            text = "Edit Arrows on Cutting Surface:"
            ArrowsOnCuttingSurfaceText = covise.getCoConfigEntry("vr-prepare.ArrowsOnCuttingSurfaceText")
            if ArrowsOnCuttingSurfaceText:
                text = ArrowsOnCuttingSurfaceText
            self.vrpLabelTitle.setText(self.__tr(text))
            title = "Arrows on Cutting Plane Adjustment"
            ArrowsOnCuttingSurfaceAdjustmentText = covise.getCoConfigEntry("vr-prepare.ArrowsOnCuttingSurfaceAdjustmentText")
            if ArrowsOnCuttingSurfaceAdjustmentText:
                title = ArrowsOnCuttingSurfaceAdjustmentText
            self.vrpGroupBoxAdjustment.setTitle(self.__tr(title))

    def __setParams( self, params ):
        """ set update the panel with the information in the negotiator param class """
        CuttingSurfacePanelBlockSignals( self, True )

        if isinstance( params, int):
            self.__key = params
            return

        self.__vector = params.vector
        self.__panelAccordingTaskType()
        self.nameWidget.setText(params.name)
        if hasattr(params, 'isVisible' ): self.visibilityCheckBox.setChecked(params.isVisible)
        if hasattr(params, 'variable' ): self.vrpLineEditVariable.setText(params.variable)
        currentVariable = params.variable

        self.__baseVariable = currentVariable
        currentColorTableKey = None
        if currentVariable!=None and params.colorTableKey!=None and currentVariable in params.colorTableKey:
            currentColorTableKey = params.colorTableKey[currentVariable]            
        MainWindow.globalColorManager.update( self.colorMapCombobox, currentVariable, currentColorTableKey)        


        if hasattr(params.boundingBox, 'getXMin' ):
            self.__boundingBox = params.boundingBox
            self.__rectangleManager.setBoundingBox( params.boundingBox )

        self.__rectangleManager.setRectangle( params.alignedRectangle )
        
        if hasattr(params, 'scale'):
            self.cuttingSurfaceScale.setText(str(params.scale))

        if hasattr(params, 'arrow_head_factor'):
            self.cuttingSurfaceArrowHead.setText(str(params.arrow_head_factor))
        else:
            self.cuttingSurfaceArrowHead.setText("0.2")

        if hasattr(params, 'project_arrows'):
            self.checkBoxProjectArrows.setChecked(params.project_arrows)

        #set cursor to begin of field
        self.cuttingSurfaceScale.home(False)
        if hasattr(params, 'length'): self.cuttingSurfaceLength.setCurrentIndex(params.length-1)
        self.vrpCheckBoxShow.setChecked(params.showSmoke)
        self.checkBox5.setChecked(params.showInteractor)
        self.colorCreator=params.colorTableKey

        self.ClipPlaneIndexCombo.setCurrentIndex(params.attachedClipPlane_index + 1)
        self.vrpLineEditClipPlaneOffset.setText(str(params.attachedClipPlane_offset))
        self.vrpLineEditClipPlaneOffset.setEnabled(self.ClipPlaneIndexCombo.currentIndex()>0)
        self.ClipPlaneFlipCheckbox.setChecked(params.attachedClipPlane_flip)
        self.ClipPlaneFlipCheckbox.setEnabled(self.ClipPlaneIndexCombo.currentIndex()>0)

        CuttingSurfacePanelBlockSignals( self, False )       

    def emitPlaneParametersApply(self):
        if not self.__key==-1:
            Application.vrpApp.key2params[self.__key] = self.__getParams()
            ObjectMgr().setParams( self.__key, self.__getParams() )
            theGuiMsgHandler().runObject( self.__key )
        _infoer.function = str(self.emitPlaneParametersApply)
        _infoer.write("")

    def emitNameChange(self, aQString=None):
        if self.__key!=-1:
            MainWindow.globalAccessToTreeView.setItemData(self.__key, str(self.nameWidget.text()))
            self.emitPlaneParametersApply()

    def emitVisibilityToggled(self, b):
        self.sigVisibiliyToggled.emit((b,))
        self.emitRectangleChange(b)

    def emitClipPlaneChanged(self):
        if not self.__key==-1:
            self.vrpLineEditClipPlaneOffset.setEnabled(self.ClipPlaneIndexCombo.currentIndex()>0)
            self.ClipPlaneFlipCheckbox.setEnabled(self.ClipPlaneIndexCombo.currentIndex()>0)
            Application.vrpApp.key2params[self.__key] = self.__getParams()
            ObjectMgr().setParams( self.__key, self.__getParams() )

    def emitCustomizeColorMap(self):
        self.sigEditColorMap.emit( self.__key, MainWindow.globalColorManager.getSelectedColormapKey( self.colorMapCombobox ))

    def emitRectangleChange(self, val=False):
        if not self.__key==-1:
            Application.vrpApp.key2params[self.__key] = self.__getParams()
            ObjectMgr().setParams( self.__key, self.__getParams() )

    def __setRightDomainEnabling(self, isPartChoice):
        self.checkBox5.setEnabled(not isPartChoice)
        self.vrpCheckBoxShow.setEnabled(not isPartChoice)

    def setRightDomainEnablingUpdate(self, isPartChoice):
        self.__setRightDomainEnabling( isPartChoice )
        self.emitPlaneParametersApply()

    def __tr(self,s,c = None):
        return coTranslate(s)



# eof
