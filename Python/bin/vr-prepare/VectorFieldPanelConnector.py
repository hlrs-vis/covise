
from PyQt5 import QtCore, QtGui

import ObjectMgr

def VectorFieldPanelConnector(panel):
    '''Connections for the panel of corresponding visualizer'''
    
    panel.PushButtonCustomizeColorMap.clicked.connect(panel.emitCustomizeColorMap)
    panel.pushButtonEditRGBColor.clicked.connect(panel.emitEditRGBColor)
    panel.colorMapCombobox.activated.connect(panel.emitValueChange)
    panel.nameWidget.returnPressed.connect(panel.emitNameChange)
    panel.visibilityCheckBox.toggled.connect(panel.emitVisibilityToggled)
    panel.vrpCheckBoxMapVariable.toggled.connect(panel.emitValueChange)
    panel.vrpComboBoxVariable.activated.connect(panel.emitValueChange)
    
    # connect radio buttons
    
    panel.radioButtonRGBColor.clicked.connect(panel.emitChangedRadioGroup)
    panel.radioButtonColorMap.clicked.connect(panel.emitChangedRadioGroup)

    # param changed signal from object manager
    ObjectMgr.ObjectMgr().sigGuiParamChanged.connect(panel.paramChanged)

    # make connections for particular widgets of the visualizer
    panel.scalingValue.returnPressed.connect(panel.emitValueChange)
    panel.arrowHeadFactor.returnPressed.connect(panel.emitValueChange)
    panel.comboBoxScalingType.activated.connect(panel.emitValueChange)

    #validators
    # allow only double values for changeIndicatedLEs
    doubleValidator = QtGui.QDoubleValidator(panel)
    panel.floatX.setValidator(doubleValidator)
    panel.floatY.setValidator(doubleValidator)
    panel.floatZ.setValidator(doubleValidator)
    panel.scalingValue.setValidator(doubleValidator)

    # allow only positiv double values for changeIndicatedLEs
    posDoubleValidator = QtGui.QDoubleValidator(panel)
    posDoubleValidator.setBottom(0.0)
    panel.arrowHeadFactor.setValidator(posDoubleValidator)

def VectorFieldPanelBlockSignals( panel, doBlock):
    panel.blockSignals(doBlock)
    # block all widgets with apply
    applyWidgets = [
        # common widgets to all panels
        panel.colorMapCombobox,
        panel.nameWidget,
        panel.visibilityCheckBox,
        panel.vrpLineEditVariable,
        panel.radioButtonRGBColor,
        panel.radioButtonColorMap,

        # my individual widgets
        panel.scalingValue,
        panel.comboBoxScalingType,
        panel.arrowHeadFactor
        ]

    for widget in applyWidgets:
        widget.blockSignals(doBlock)
