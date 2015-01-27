
from PyQt5 import QtCore, QtGui

import ObjectMgr

def IsoSurfacePanelConnector(panel):
    '''Connections for the Isosurface panel'''
    panel.vrpPushButtonCustomizeColorMap.clicked.connect(panel.emitCustomizeColorMap)
    panel.colorMapCombobox.activated.connect(panel.emitCustomizeColorMap)
    panel.colorMapCombobox.activated.connect(panel.emitValueChange)
    panel.nameWidget.returnPressed.connect(panel.emitNameChange)
    panel.visibilityCheckBox.toggled.connect(panel.emitVisibilityToggled)
    panel.floatInRangeX.sigSliderReleased.connect(panel.emitValueChange)
    panel.vrpCheckBoxMapVariable.toggled.connect(panel.emitValueChange)
    panel.vrpComboBoxVariable.activated.connect(panel.emitValueChange)
   
   


    # param changed QtCore.SIGNAL from object manager
    ObjectMgr.ObjectMgr().sigGuiParamChanged.connect(panel.paramChanged)

def IsoSurfacePanelBlockSignals( panel, doBlock):
    panel.blockSignals(doBlock)
    # block all widgets with apply
    applyWidgets = [
        panel.colorMapCombobox,
        panel.floatInRangeX,
        panel.nameWidget,
        panel.visibilityCheckBox,
        panel.vrpLineEditVariable,
        panel.vrpCheckBoxMapVariable,
        panel.vrpComboBoxVariable
        ]

    for widget in applyWidgets:
        widget.blockSignals(doBlock)
