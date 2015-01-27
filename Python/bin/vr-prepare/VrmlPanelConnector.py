
from PyQt5 import QtCore, QtGui

import ObjectMgr

def VrmlPanelConnector(panel):
    '''Connections for the panel of corresponding visualizer'''

    # param changed signal from object manager
    ObjectMgr.ObjectMgr().sigGuiParamChanged.connect(panel.paramChanged)


def VrmlPanelBlockSignals( panel, doBlock):
    panel.blockSignals(doBlock)
    # block all widgets with apply
    applyWidgets = [
        # common widgets to all panels

        # my individual widgets
        ]

    for widget in applyWidgets:
        widget.blockSignals(doBlock)
