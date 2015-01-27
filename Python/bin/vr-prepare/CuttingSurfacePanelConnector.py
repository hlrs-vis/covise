
from PyQt5 import QtCore, QtGui

import ObjectMgr

def CuttingSurfacePanelConnector(panel):

    panel.vrpPushButtonCustomizeColorMap.clicked.connect(panel.emitCustomizeColorMap)
    panel.cuttingSurfaceScale.returnPressed.connect(panel.emitPlaneParametersApply)
    panel.cuttingSurfaceLength.activated.connect(panel.emitPlaneParametersApply)
    panel.vrpCheckBoxShow.toggled.connect(panel.emitPlaneParametersApply)
    panel.vrpComboBoxVariable.activated.connect(panel.emitPlaneParametersApply)
    panel.colorMapCombobox.activated.connect(panel.emitPlaneParametersApply)
    panel.nameWidget.returnPressed.connect(panel.emitNameChange)
    panel.visibilityCheckBox.toggled.connect(panel.emitVisibilityToggled)
    panel.checkBox5.toggled.connect(panel.emitRectangleChange)
    panel.checkBoxProjectArrows.toggled.connect(panel.emitPlaneParametersApply)
    panel.cuttingSurfaceArrowHead.returnPressed.connect(panel.emitPlaneParametersApply)
    panel.ClipPlaneIndexCombo.activated.connect(panel.emitClipPlaneChanged)
    panel.vrpLineEditClipPlaneOffset.returnPressed.connect(panel.emitClipPlaneChanged)
    panel.ClipPlaneFlipCheckbox.toggled.connect(panel.emitClipPlaneChanged)
    panel.visibilityCheckBox.toggled.connect(panel.emitVisibilityToggled)
  
    


    # param changed signal from object manager
    ObjectMgr.ObjectMgr().sigGuiParamChanged.connect(panel.paramChanged)

    #validators
    # allow only positiv double values for changeIndicatedLEs
    doubleValidator = QtGui.QDoubleValidator(panel)
    doubleValidator.setBottom(0.0)
    panel.cuttingSurfaceScale.setValidator(doubleValidator)
    panel.TubeWidth.setValidator(doubleValidator)
    panel.cuttingSurfaceArrowHead.setValidator(doubleValidator)
    panel.vrpLineEditRelativeErrors.setValidator(doubleValidator)
    panel.vrpLineEditAbsoluteErrors.setValidator(doubleValidator)
    panel.vrpLineEditGridTolerance.setValidator(doubleValidator)
    panel.vrpLineEditMinimumValue.setValidator(doubleValidator)
    panel.vrpLineEditMinimumValue_2.setValidator(doubleValidator)

    # allow only positiv int values for changeIndicatedLEs
    intValidator = QtGui.QIntValidator(panel)
    intValidator.setBottom(0)
    panel.numberStartpoints.setValidator(intValidator)
    panel.NumberOfSteps.setValidator(intValidator)
    panel.DurationOfSteps.setValidator(intValidator)

def CuttingSurfacePanelBlockSignals( panel, doBlock ):
        panel.blockSignals(doBlock)
        # block all widgets with apply
        applyWidgets = [
            panel.checkBox5,
            panel.colorMapCombobox,
            panel.floatInRangeRotX,
            panel.floatInRangeRotY,
            panel.floatInRangeRotZ,
            panel.floatInRangeX,
            panel.floatInRangeY,
            panel.floatInRangeZ,
            panel.nameWidget,
            panel.vrpCheckBoxShow,
            panel.visibilityCheckBox,
            panel.vrpLineEditVariable,
            panel.ClipPlaneIndexCombo,
            panel.vrpLineEditClipPlaneOffset,
            panel.ClipPlaneFlipCheckbox,
            panel.checkBoxProjectArrows,
            panel.cuttingSurfaceScale,
            panel.cuttingSurfaceLength,
            panel.cuttingSurfaceArrowHead
            ]

        for widget in applyWidgets:
            widget.blockSignals(doBlock)        
