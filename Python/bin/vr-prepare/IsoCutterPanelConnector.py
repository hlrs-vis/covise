
from PyQt5 import QtCore, QtGui

import ObjectMgr

def IsoCutterPanelConnector(panel):
    '''Connections for the IsoCutter panel'''
    
    panel.vrpPushButtonCustomizeColorMap.clicked.connect(panel.emitCustomizeColorMap)
    panel.colorMapCombobox.activated.connect(panel.emitValueChange)
    panel.nameWidget.returnPressed.connect(panel.emitNameChange)
    panel.visibilityCheckBox.toggled.connect(panel.emitVisibilityToggled)
    panel.floatInRangeIsoValue.sigSliderReleased.connect(panel.emitValueChange)
    panel.vrpCheckBoxMapVariable.toggled.connect(panel.emitValueChange)
    panel.vrpComboBoxVariable.activated.connect(panel.emitValueChange)
    panel.checkBoxCutoffSide.toggled.connect(panel.emitValueChange)

        
    # param changed signal from object manager
    ObjectMgr.ObjectMgr().sigGuiParamChanged.connect(panel.paramChanged)

def IsoCutterPanelBlockSignals( panel, doBlock):
    panel.blockSignals(doBlock)
    # block all widgets with apply
    applyWidgets = [
        panel.colorMapCombobox,
        panel.nameWidget,
        panel.visibilityCheckBox,
        panel.vrpLineEditVariable,

        # my individual widgets
        panel.floatInRangeIsoValue,
        panel.checkBoxCutoffSide
        ]

    for widget in applyWidgets:
        widget.blockSignals(doBlock)
