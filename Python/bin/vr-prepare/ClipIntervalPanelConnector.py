
from PyQt5 import QtCore, QtGui
import ObjectMgr

def ClipIntervalPanelConnector(panel):
    '''Connections for the IsoCutter panel'''
    
    panel.vrpPushButtonCustomizeColorMap.clicked.connect(panel.emitCustomizeColorMap)
    panel.colorMapCombobox.activated.connect(panel.emitValueChange)
    panel.nameWidget.returnPressed.connect(panel.emitNameChange)
    panel.visibilityCheckBox.toggled.connect(panel.emitVisibilityToggled)
    panel.floatInRangeIsoValueLow.sigSliderReleased.connect(panel.emitValueChange)
    panel.floatInRangeIsoValueHigh.sigSliderReleased.connect(panel.emitValueChange)
    panel.vrpCheckBoxMapVariable.toggled.connect(panel.emitValueChange)
    panel.vrpComboBoxVariable.activated.connect(panel.emitValueChange)

        
    # param changed signal from object manager
    ObjectMgr.ObjectMgr().sigGuiParamChanged.connect(panel.paramChanged)

def ClipIntervalPanelBlockSignals( panel, doBlock):
    panel.blockSignals(doBlock)
    # block all widgets with apply
    applyWidgets = [
        panel.colorMapCombobox,
        panel.nameWidget,
        panel.visibilityCheckBox,
        panel.vrpLineEditVariable,

        # my individual widgets
        panel.floatInRangeIsoValueLow,
        panel.floatInRangeIsoValueHigh
        ]

    for widget in applyWidgets:
        widget.blockSignals(doBlock)
