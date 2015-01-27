
from PyQt5 import QtCore, QtGui

import ObjectMgr

def SceneObjectPanelConnector(panel):
    '''Connections for the panel of corresponding visualizer'''

    # param changed signal from object manager
    ObjectMgr.ObjectMgr().sigGuiParamChanged.connect(panel.paramChanged)
    
    # make connections for particular widgets of the visualizer
    panel.lineEdit_width.returnPressed.connect(panel.emitDataChanged)
    panel.lineEdit_height.returnPressed.connect(panel.emitDataChanged)
    panel.lineEdit_length.returnPressed.connect(panel.emitDataChanged)
    panel.lineEdit_trans_x.returnPressed.connect(panel.emitDataChanged)
    panel.lineEdit_trans_y.returnPressed.connect(panel.emitDataChanged)
    panel.lineEdit_trans_z.returnPressed.connect(panel.emitDataChanged)


    #validators
    # allow only double values for changeIndicatedLEs
    doubleValidator = QtGui.QDoubleValidator(panel)
    panel.lineEdit_width.setValidator(doubleValidator)
    panel.lineEdit_height.setValidator(doubleValidator)
    panel.lineEdit_length.setValidator(doubleValidator)
    panel.lineEdit_trans_x.setValidator(doubleValidator)
    panel.lineEdit_trans_y.setValidator(doubleValidator)
    panel.lineEdit_trans_z.setValidator(doubleValidator)

def SceneObjectPanelBlockSignals( panel, doBlock):
    panel.blockSignals(doBlock)
    # block all widgets with apply
    applyWidgets = [
        panel.lineEdit_width,
        panel.lineEdit_height,
        panel.lineEdit_length,
        panel.lineEdit_trans_x,
        panel.lineEdit_trans_y,
        panel.lineEdit_trans_z,
        ]

    for widget in applyWidgets:
        widget.blockSignals(doBlock)
