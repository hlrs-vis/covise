
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject, pyqtSignal
   
from printing import InfoPrintCapable
import covise

from Utils import getDoubleInLineEdit, getIntInLineEdit, ConversionError

import Application
import MainWindow

from GenericVisualizerPanelBase import Ui_GenericVisualizerPanelBase
from StreamlinesPanelConnector import StreamlinesPanelBlockSignals, StreamlinesPanelConnector
from RectangleManager import RectangleManager
from Gui2Neg import theGuiMsgHandler
from PartTracerVis import PartStreamlineVisParams, PartMovingPointsVisParams, PartPathlinesVisParams

from ImportGroupManager import COMPOSED_VELOCITY
from ObjectMgr import ObjectMgr
import coviseCase

_infoer = InfoPrintCapable()
_infoer.doPrint = False #True # 

from vtrans import coTranslate

# the panel handles the following three streamline types
STREAMLINE = 1
MOVING_POINTS = 2
PATHLINES = 3

LINE  = 1
PLANE = 2
FREE  = 3

class StreamlinesPanel(QtWidgets.QWidget,Ui_GenericVisualizerPanelBase):

    """For controling parameters of a streamline.

    """
    sigVisibiliyToggled = pyqtSignal()
    sigEditColorMap = pyqtSignal()

    def __init__(self, parent=None):
        _infoer.function = str(self.__init__)
        _infoer.write("")

        QtWidgets.QWidget.__init__(self, parent)
        Ui_GenericVisualizerPanelBase.__init__(self)
        self.setupUi(self)
        # current object key
        self.__key = -1

        self.TabWidgetGeneralAdvanced.removeTab(self.TabWidgetGeneralAdvanced.indexOf(self.tabAdjustmentCuttingSurface))
        self.TabWidgetGeneralAdvanced.removeTab(self.TabWidgetGeneralAdvanced.indexOf(self.tabClipPlane))
        self.TabWidgetGeneralAdvanced.setCurrentIndex(0) # ignore index set by the designer (usually no one cares about the active index when editing ui-files)

        self.visibilityCheckBox.setVisible(covise.coConfigIsOn("vr-prepare.AdditionalVisibilityCheckbox", False))

        self.vrpPushButtonAdjustPreview.hide()

        # manager handling the starting rectangle
        self.__rectangleManager = RectangleManager( self, self.emitTraceParametersApply, self.emitRadioBttnsTracerApply, self.emitRectangleChange )
        
        # keys of 2D objects in combobox        
        self.__key2UsePart = {}
        self.__UsePart2key = {}
        
        # current task type        
        self.__taskType = STREAMLINE

        # start style of tracer to that the panel will change
        self.__startStyle = PLANE
        self.__baseVariable = None
        
        #temporary storage of this parameter part
        self.colorCreator = None        

        # qt connections and settings
        StreamlinesPanelConnector(self)
               
    def setSelectedColormap( self, callerKey, key, name):
        _infoer.function = str(self.setSelectedColormap)
        _infoer.write("")
        # new colormap was selected in color manager
        if (self.__key == callerKey):
            if MainWindow.globalColorManager.setSelectedColormapKey( self.colorMapCombobox, key ):
                self.emitTraceParametersApply()
    
    def paramChanged( self, key ):
        """ params of object key changed"""
        _infoer.function = str(self.paramChanged)
        _infoer.write(""+str(key))
        if self.__key==key:
            self.update()
            
    def update( self ):
        _infoer.function = str(self.update)
        _infoer.write("")
        if self.__key!=-1:
            self.updateForObject( self.__key )
                
    def updateForObject( self, key ):
        """ called from MainWindow to update the content to the choosen object key """
        _infoer.function = str(self.updateForObject)
        _infoer.write("")
        self.__key = key
        params = ObjectMgr().getParamsOfObject(key)
        self.__setStartDomainNamesList( ObjectMgr().getList2dPartsForVisItem(key) )
        self.__set2ndVariable( ObjectMgr().getPossibleScalarVariablesForVisItem(key) )
        self.__setParams( params )
        
    def __getParams(self):
        """
            convert information in the panel into the right negotiator param classes
        """
        _infoer.function = str(self.__getParams)
        _infoer.write("")
        if self.__taskType==STREAMLINE:
            data = PartStreamlineVisParams()
            data.tubeWidth = getDoubleInLineEdit(self.TubeWidth)
        elif self.__taskType==MOVING_POINTS:
            data = PartMovingPointsVisParams()
            data.numSteps = getIntInLineEdit(self.NumberOfSteps)
            data.duration = getDoubleInLineEdit(self.DurationOfSteps)
            data.sphereRadius = getDoubleInLineEdit(self.RadiusOfSpheres)
        else:
            data = PartPathlinesVisParams()
            data.numSteps = getIntInLineEdit(self.NumberOfSteps)
            data.duration = getDoubleInLineEdit(self.DurationOfSteps)
            data.tubeWidth = getDoubleInLineEdit(self.TubeWidth)
            data.sphereRadius = getDoubleInLineEdit(self.RadiusOfSpheres)
        data.taskType = self.__taskType
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
        if self.__startStyle == PLANE:
            data.alignedRectangle = self.__rectangleManager.getParams()
        elif self.__startStyle == LINE:
            #data.line3D = self.__rectangleManager.getParams(LINE)
            data.alignedRectangle = self.__rectangleManager.getParams(LINE)
        else:
            data.alignedRectangle = self.__rectangleManager.getParams()
        data.boundingBox = self.__rectangleManager.getBoundingBox()
        try:
            data.no_startp = getIntInLineEdit(self.numberStartpoints)
        except ConversionError:
            data.no_startp = 0
        try:
            data.len = getDoubleInLineEdit(self.lengthTraces)
        except ConversionError:
            data.len = 0.0
        data.direction = self.tracingDirectionCB.currentIndex()+1
        try:
            data.eps = getDoubleInLineEdit(self.vrpLineEditRelativeErrors)
        except ConversionError:
            data.eps = 0.0
        try:
            data.abs = getDoubleInLineEdit(self.vrpLineEditAbsoluteErrors)
        except ConversionError:
            data.abs = 0.0
        try:
            data.grid_tol = getDoubleInLineEdit(self.vrpLineEditGridTolerance)
        except ConversionError:
            data.grid_tol = 0.0
        try:
            data.maxOutOfDomain = getDoubleInLineEdit(self.vrpLineEditMinimumValue_2)
        except ConversionError:
            data.maxOutOfDomain = 0.0
        try:
            data.min_vel = getDoubleInLineEdit(self.vrpLineEditMinimumValue)
        except ConversionError:
            data.min_vel = 0.0
        data.showSmoke= self.vrpCheckBoxShow.isChecked()
        data.showInteractor = self.checkBox5.isChecked()
        if self.checkBoxDomainFromList.isChecked():
            if self.comboBoxDomain.currentIndex()>0 :
                data.use2DPartKey=self.__UsePartIdx2key[self.comboBoxDomain.currentIndex()]
            else :
                data.use2DPartKey=-1
        if self.checkBoxFreeStartpoints.isChecked():
            data.start_style = 3
        else:
            data.start_style = self.__startStyle
                
        return data

    def __panelAccordingTaskType( self ):
        """
            change the panel according the needs of streamlines, moving points or pathlines
            NOTE: changes identical for all types should be done in __init__
        """
        _infoer.function = str(self.__panelAccordingTaskType)
        _infoer.write("")

        #default settings
        self.vrpCheckBoxMapVariable.setEnabled(True)
        self.vrpComboBoxVariable.setEnabled(True)
        self.comboBoxDomain.setEnabled(False)
        # widgets for end point not needed
        self.textEndPointX.hide()
        self.textEndPointY.hide()
        self.textEndPointZ.hide()
        self.floatInRangeEndPointX.hide()
        self.floatInRangeEndPointY.hide()
        self.floatInRangeEndPointZ.hide()
        # make sure whole data is shown
        self.checkBoxDomainFromList.show()
        self.comboBoxDomain.show()
        self.floatInRangeWidth.show()
        self.floatInRangeHeight.show()
        self.xAxisRadioButton.show()
        self.yAxisRadioButton.show()
        self.zAxisRadioButton.show()
        self.textLabel4_3_3.show()
        self.textLabel4_3_2_3.show()
        self.floatInRangeRotX.show()
        self.floatInRangeRotY.show()
        self.floatInRangeRotZ.show()
        self.textLabelRotX.show()
        self.textLabelRotY.show()
        self.textLabelRotZ.show()
        self.checkBoxFreeStartpoints.show()
        self.vrpCheckBoxShow.show()

        if self.__taskType==STREAMLINE:
            self.vrpLabelTitle.setText(self.__tr("Edit Streamline:"))
            self.textNumberOfStreamlines.setText(self.__tr("Number of Streamlines:"))
            self.vrpGroupBoxAdjustment.setTitle(self.__tr("Streamline Adjustment"))
            self.textLength.show()
            self.lengthTraces.show()
            self.textDirection.show()
            self.tracingDirectionCB.show()
            self.textNumberOfSteps.hide()
            self.NumberOfSteps.hide()
            self.textRadiusOfSpheres.hide()
            self.RadiusOfSpheres.hide()
            self.textTubeWidth.show()
            self.TubeWidth.show()
            self.textDurationOfSteps.hide()
            self.DurationOfSteps.hide()
            if self.__startStyle == LINE:
                self.textEndPointX.show()
                self.textEndPointY.show()
                self.textEndPointZ.show()
                self.floatInRangeEndPointX.show()
                self.floatInRangeEndPointY.show()
                self.floatInRangeEndPointZ.show()
                self.checkBoxDomainFromList.hide()
                self.comboBoxDomain.hide()
                self.floatInRangeWidth.hide()
                self.floatInRangeHeight.hide()
                self.xAxisRadioButton.hide()
                self.yAxisRadioButton.hide()
                self.zAxisRadioButton.hide()
                self.groupOrientation.hide()
                self.textLabel4_3_3.hide()
                self.textLabel4_3_2_3.hide()
                self.floatInRangeRotX.hide()
                self.floatInRangeRotY.hide()
                self.floatInRangeRotZ.hide()
                self.textLabelRotX.hide()
                self.textLabelRotY.hide()
                self.textLabelRotZ.hide()
                self.checkBoxFreeStartpoints.hide()
                self.vrpCheckBoxShow.hide()

        elif self.__taskType==MOVING_POINTS or self.__taskType==PATHLINES:
            self.textNumberOfStreamlines.setText(self.__tr("Number of Startpoints:"))
            self.vrpGroupBoxAdjustment.setTitle(self.__tr("Points Adjustment"))
            self.textLength.hide()
            self.lengthTraces.hide()
            self.textDirection.hide()
            self.tracingDirectionCB.hide()
            self.textNumberOfSteps.show()
            self.NumberOfSteps.show()
            self.textDurationOfSteps.show()
            self.DurationOfSteps.show()
            if self.__taskType==MOVING_POINTS:
                self.vrpLabelTitle.setText(self.__tr("Edit Moving Points:"))
                self.textRadiusOfSpheres.setText(self.__tr("Radius of Spheres"))
                self.textRadiusOfSpheres.show()
                self.RadiusOfSpheres.show()
                self.textTubeWidth.hide()
                self.TubeWidth.hide()
            elif self.__taskType==PATHLINES:
                self.vrpLabelTitle.setText(self.__tr("Edit Pathlines:"))
                self.textRadiusOfSpheres.setText(self.__tr("Size of head object:"))
                self.textTubeWidth.show()
                self.TubeWidth.show()
    
    def __set2ndVariable( self, varlist):
        """ fill the combobox to choose a variable to be mapped on the trace """
        _infoer.function = str(self.__set2ndVariable)
        _infoer.write("")
        self.vrpComboBoxVariable.clear()
        self.vrpComboBoxVariable.setEnabled( len(varlist)!=0 )
        self.vrpCheckBoxMapVariable.setEnabled( len(varlist)!=0 )
        for v in varlist:
            self.vrpComboBoxVariable.addItem(v)
            
    def __setParams( self, params ):
        """ set update the panel with the information in the negotiator param class """
        _infoer.function = str(self.__setParams)
        _infoer.write("")
        StreamlinesPanelBlockSignals( self, True )

        # int is always the key
        # TODO CHANGE
        if isinstance( params, int):
            self.__key = params
            return

        self.__taskType = params.taskType
        if params.taskType == MOVING_POINTS:
            self.textTubeWidth.hide()
            self.TubeWidth.hide()
        else:
            self.TubeWidth.show()
            self.textTubeWidth.show()
        if hasattr(params, 'start_style'):
            if params.start_style != FREE:
                self.__startStyle = params.start_style
        self.__panelAccordingTaskType()
        if hasattr(params, 'numSteps'): self.NumberOfSteps.setText(str(params.numSteps))
        if hasattr(params, 'duration'): self.DurationOfSteps.setText(str(params.duration))
        if hasattr(params, 'sphereRadius'):
           self.RadiusOfSpheres.setText(str(params.sphereRadius))
        else:
           self.RadiusOfSpheres.setText("0.2")
        if hasattr(params, 'tubeWidth'):
           self.TubeWidth.setText(str(params.tubeWidth))
        else:
           self.TubeWidth.setText("0.0")

        self.nameWidget.setText(params.name)
        if hasattr(params, 'isVisible' ): self.visibilityCheckBox.setChecked(params.isVisible)
        if hasattr(params, 'variable' ):
            self.vrpLineEditVariable.setText(params.variable)

        if params.secondVariable!=None:
            self.vrpCheckBoxMapVariable.setChecked(True)
            self.vrpComboBoxVariable.setCurrentIndex(self.vrpComboBoxVariable.findText(params.secondVariable))
            currentVariable = params.secondVariable
        else:
            self.vrpCheckBoxMapVariable.setChecked(False)
            currentVariable = params.variable
            
        if hasattr(params, 'start_style'):
            if params.start_style==3:
                self.checkBoxFreeStartpoints.setChecked(True)
            else:
                self.checkBoxFreeStartpoints.setChecked(False)

        self.__baseVariable = currentVariable
        currentColorTableKey = None
        if currentVariable!=None and params.colorTableKey!=None and currentVariable in params.colorTableKey:
            currentColorTableKey = params.colorTableKey[currentVariable]            
        MainWindow.globalColorManager.update( self.colorMapCombobox, currentVariable, currentColorTableKey)        
            
        if hasattr(params.boundingBox, 'getXMin' ):
            self.__boundingBox = params.boundingBox
            if self.__startStyle == PLANE:
                self.__rectangleManager.setBoundingBox( params.boundingBox )
            elif self.__startStyle == LINE:
                self.__rectangleManager.setBoundingBox( params.boundingBox, LINE )
            else:
                self.__rectangleManager.setBoundingBox( params.boundingBox )

        if self.__startStyle == PLANE:
            self.__rectangleManager.setRectangle( params.alignedRectangle )
        elif self.__startStyle == LINE:
            #self.__rectangleManager.setLine( params.lines3D )
            self.__rectangleManager.setLine( params.alignedRectangle )
        else:
            self.__rectangleManager.setRectangle( params.alignedRectangle )
        self.numberStartpoints.setText(str(params.no_startp))
        self.lengthTraces.setText(str(params.len))
        self.tracingDirectionCB.setCurrentIndex(params.direction-1)
        self.vrpLineEditRelativeErrors.setText(str(params.eps))
        self.vrpLineEditAbsoluteErrors.setText(str(params.abs))
        self.vrpLineEditGridTolerance.setText(str(params.grid_tol))
        self.vrpLineEditMinimumValue.setText(str(params.min_vel))
        self.vrpLineEditMinimumValue_2.setText(str(params.maxOutOfDomain))
        self.vrpCheckBoxShow.setChecked(params.showSmoke)
        self.checkBox5.setChecked(params.showInteractor)
        self.colorCreator=params.colorTableKey
        if (params.use2DPartKey==None) or (self.comboBoxDomain.count() == 1):
            self.checkBoxDomainFromList.setChecked(False)
        else :
            self.checkBoxDomainFromList.setChecked(True)
            if params.use2DPartKey in self.__key2UsePartIdx:
                self.comboBoxDomain.setCurrentIndex( self.__key2UsePartIdx[params.use2DPartKey] )
            else :
                self.comboBoxDomain.setCurrentIndex( 0 )
        self.__setRightDomainEnabling(self.checkBoxDomainFromList.isChecked())

        if hasattr(params, 'tubeWidth'):
            if params.secondVariable != None and params.tubeWidth <= 0.0:
                self.vrpCheckBoxMapVariable.setEnabled(True)
                self.vrpComboBoxVariable.setEnabled(True)
                self.TubeWidth.setEnabled(False)
            elif params.secondVariable == None and params.tubeWidth > 0.0:
                self.vrpCheckBoxMapVariable.setEnabled(False)
                self.vrpComboBoxVariable.setEnabled(False)
                self.TubeWidth.setEnabled(True)
            elif params.secondVariable == None and params.tubeWidth <= 0.0:
                self.vrpCheckBoxMapVariable.setEnabled(True)
                self.vrpComboBoxVariable.setEnabled(True)
                self.TubeWidth.setEnabled(True)
            else:
                print("ERROR: Tube width > 0 and mapping of 2nd variable not supported")

            # show qLabel RadiusOfSpheres only if COMPLEX_OBJECT_TYPE is BAR_MAGNET or COMPASS
            # and if tube width is changed
            self.textRadiusOfSpheres.hide()
            self.RadiusOfSpheres.hide()

            if self.__taskType==PATHLINES and self.TubeWidth.text() != "0.0":
                complexObjetType = covise.getCoConfigEntry('TRACERConfig.COMPLEX_OBJECT_TYPE')
                if (complexObjetType == 'BAR_MAGNET') or (complexObjetType == 'COMPASS'):
                    self.textRadiusOfSpheres.show()
                    self.RadiusOfSpheres.show()

        StreamlinesPanelBlockSignals( self, False )           

    def __setStartDomainNamesList(self, dL):
        """ fill combobox with 2d domain name list dL"""
        _infoer.function = str(self.__setStartDomainNamesList)
        _infoer.write("")
        self.__key2UsePartIdx = {}
        self.__UsePartIdx2key = {}
        self.checkBoxDomainFromList.setEnabled( len(dL)!=0 )
        self.comboBoxDomain.clear()
        self.comboBoxDomain.blockSignals(True)
        self.comboBoxDomain.addItem("Please select a part")
        cnt=1
        for keydName in dL:
            self.__key2UsePartIdx[keydName[0]] = cnt
            self.__UsePartIdx2key[cnt] = keydName[0]
            self.comboBoxDomain.addItem(keydName[1])
            cnt=cnt+1
        self.comboBoxDomain.blockSignals(False)
        
    def emitTraceParametersApply(self):
        _infoer.function = str(self.emitTraceParametersApply)
        _infoer.write("")
        if not self.__key==-1:
            Application.vrpApp.key2params[self.__key] = self.__getParams()
            ObjectMgr().setParams( self.__key, self.__getParams() )
            theGuiMsgHandler().runObject( self.__key )

    def emitRadioBttnsTracerApply(self):
        _infoer.function = str(self.emitRadioBttnsTracerApply)
        _infoer.write("")
        if not self.__key==-1:
            # send message to cover
            if self.xAxisRadioButton.isChecked():
                theGuiMsgHandler().sendKeyWord("xAxis")
            if self.yAxisRadioButton.isChecked():
                theGuiMsgHandler().sendKeyWord("yAxis")
            if self.zAxisRadioButton.isChecked():
                theGuiMsgHandler().sendKeyWord("zAxis")
            self.emitTraceParametersApply()

    def emitNameChange(self, aQString=None):
        _infoer.function = str(self.emitNameChange)
        _infoer.write("")
        if self.__key!=-1:
            MainWindow.globalAccessToTreeView.setItemData(self.__key, str(self.nameWidget.text()))
            self.emitTraceParametersApply()

    def emitVisibilityToggled(self, b):
        _infoer.function = str(self.emitVisibilityToggled)
        _infoer.write("")
        self.sigVisibiliyToggled.emit( (b,))
        self.emitRectangleChange(b)

    def emitCustomizeColorMap(self):
        _infoer.function = str(self.emitCustomizeColorMap)
        _infoer.write("")
        self.sigEditColorMap.emit( self.__key, MainWindow.globalColorManager.getSelectedColormapKey( self.colorMapCombobox) )
        
    def emitRectangleChange(self, val=False):
        _infoer.function = str(self.emitRectangleChange)
        _infoer.write("")
        if not self.__key==-1:
            Application.vrpApp.key2params[self.__key] = self.__getParams()
            ObjectMgr().setParams( self.__key, self.__getParams() )

    def __setRightDomainEnabling(self, isPartChoice):
        _infoer.function = str(self.__setRightDomainEnabling)
        _infoer.write("")
        self.comboBoxDomain.setEnabled(isPartChoice)
        if not self.checkBoxFreeStartpoints.isChecked():
            self.groupBoxRectPositioning.setEnabled(not isPartChoice)
        if self.checkBoxDomainFromList.isChecked():     #quick fix show/hide tracer
            self.checkBox5.setChecked(False)
        TraceFrom2DStyle = covise.getCoConfigEntry("vr-prepare.TraceFrom2DStyle")
        if TraceFrom2DStyle and TraceFrom2DStyle == "TRACE_FROM_2D_SAMPLE":
            pass        
        else :
            self.numberStartpoints.setEnabled(not isPartChoice)
        # change interactor checkbox
        self.checkBox5.setEnabled(not isPartChoice)
        self.vrpCheckBoxShow.setEnabled((not isPartChoice) and COMPOSED_VELOCITY!=str(self.vrpLineEditVariable.text()))
        if isPartChoice:
            self.checkBoxFreeStartpoints.setChecked(False)
    
    def setRightDomainEnablingUpdate(self, isPartChoice):
        _infoer.function = str(self.setRightDomainEnablingUpdate)
        _infoer.write("")
        self.__setRightDomainEnabling( isPartChoice )
        self.emitTraceParametersApply()    
        
    def emitFreeStartpoint(self):
        _infoer.function = str(self.emitFreeStartpoint)
        _infoer.write("")
        isFreeStartPoints = self.checkBoxFreeStartpoints.isChecked()
        self.checkBoxDomainFromList.setChecked(False)
        self.groupBoxRectPositioning.setEnabled(not isFreeStartPoints)
        self.emitTraceParametersApply()

    def __tr(self,s,c = None):
        _infoer.function = str(self.__tr)
        _infoer.write("")
        return coTranslate(s)

# eof
