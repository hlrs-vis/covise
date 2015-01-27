
from PyQt5 import QtCore, QtGui

import ObjectMgr

def StreamlinesPanelConnector(panel):

    panel.vrpPushButtonCustomizeColorMap.clicked.connect(panel.emitCustomizeColorMap)
    panel.numberStartpoints.returnPressed.connect(panel.emitTraceParametersApply)
    panel.lengthTraces.returnPressed.connect(panel.emitTraceParametersApply)
    panel.vrpLineEditRelativeErrors.returnPressed.connect(panel.emitTraceParametersApply)
    panel.vrpLineEditAbsoluteErrors.returnPressed.connect(panel.emitTraceParametersApply)
    panel.vrpLineEditGridTolerance.returnPressed.connect(panel.emitTraceParametersApply)
    panel.vrpLineEditMinimumValue_2.returnPressed.connect(panel.emitTraceParametersApply)
    panel.vrpLineEditMinimumValue.returnPressed.connect(panel.emitTraceParametersApply)
    panel.tracingDirectionCB.activated.connect(panel.emitTraceParametersApply)
    panel.vrpCheckBoxShow.released.connect(panel.emitTraceParametersApply)
    panel.vrpCheckBoxMapVariable.released.connect(panel.emitTraceParametersApply)
    panel.checkBox5.toggled.connect(panel.emitRectangleChange)
    panel.vrpComboBoxVariable.activated.connect(panel.emitTraceParametersApply)
    panel.nameWidget.returnPressed.connect(panel.emitNameChange)
    panel.visibilityCheckBox.toggled.connect(panel.emitVisibilityToggled)
    panel.checkBoxDomainFromList.toggled.connect(panel.setRightDomainEnablingUpdate)
    panel.checkBoxFreeStartpoints.released.connect(panel.emitFreeStartpoint)
    panel.colorMapCombobox.activated.connect(panel.emitTraceParametersApply)
    panel.comboBoxDomain.activated.connect(panel.emitTraceParametersApply)
    panel.NumberOfSteps.returnPressed.connect(panel.emitTraceParametersApply)
    panel.DurationOfSteps.returnPressed.connect(panel.emitTraceParametersApply)
    panel.RadiusOfSpheres.returnPressed.connect(panel.emitTraceParametersApply)
    panel.TubeWidth.returnPressed.connect(panel.emitTraceParametersApply)

    # param changed signal from object manager
    ObjectMgr.ObjectMgr().sigGuiParamChanged.connect(panel.paramChanged)

    #validators
    # allow only positiv double values for changeIndicatedLEs
    doubleValidator = QtGui.QDoubleValidator(panel)
    doubleValidator.setBottom(0.0)
    panel.lengthTraces.setValidator(doubleValidator)
    panel.TubeWidth.setValidator(doubleValidator)
    panel.RadiusOfSpheres.setValidator(doubleValidator)
    panel.vrpLineEditRelativeErrors.setValidator(doubleValidator)
    panel.vrpLineEditAbsoluteErrors.setValidator(doubleValidator)
    panel.vrpLineEditGridTolerance.setValidator(doubleValidator)
    panel.vrpLineEditMinimumValue.setValidator(doubleValidator)
    panel.vrpLineEditMinimumValue_2.setValidator(doubleValidator)
    panel.DurationOfSteps.setValidator(doubleValidator)
    
    # allow only positiv int values for changeIndicatedLEs
    intValidator = QtGui.QIntValidator(panel)
    intValidator.setBottom(0)
    panel.numberStartpoints.setValidator(intValidator)
    panel.NumberOfSteps.setValidator(intValidator)


def StreamlinesPanelBlockSignals( panel, doBlock ):
    panel.blockSignals(doBlock)
    # block all widgets with apply
    applyWidgets = [
    panel.checkBox5,
    panel.colorMapCombobox,
    panel.floatInRangeHeight,
    panel.floatInRangeRotX,
    panel.floatInRangeRotY,
    panel.floatInRangeRotZ,
    panel.floatInRangeWidth,
    panel.floatInRangeX,
    panel.floatInRangeY,
    panel.floatInRangeZ,
    panel.floatInRangeEndPointX,
    panel.floatInRangeEndPointY,
    panel.floatInRangeEndPointZ,
    panel.lengthTraces,
    panel.nameWidget,
    panel.numberStartpoints,
    panel.tracingDirectionCB,
    panel.visibilityCheckBox,
    panel.vrpCheckBoxShow,
    panel.vrpLineEditAbsoluteErrors,
    panel.vrpLineEditGridTolerance,
    panel.vrpLineEditMinimumValue,
    panel.vrpLineEditMinimumValue_2,
    panel.vrpLineEditRelativeErrors,
    panel.vrpLineEditVariable,
    panel.comboBoxDomain,
    panel.checkBoxDomainFromList,
    panel.NumberOfSteps,
    panel.DurationOfSteps,
    panel.RadiusOfSpheres,
    panel.TubeWidth
    ]

    for widget in applyWidgets:
        widget.blockSignals(doBlock)
