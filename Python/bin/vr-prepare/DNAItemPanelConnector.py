
from PyQt5 import QtCore, QtGui

import ObjectMgr

def DNAItemPanelConnector(panel):
    #connections
    
    panel.nameWidget.returnPressed.connect(panel.emitNameChange)
    panel.vrpCheckBoxConn1.clicked.connect(panel.emitDataChanged)
    panel.vrpCheckBoxConn2.clicked.connect(panel.emitDataChanged)
    panel.vrpCheckBoxConn3.clicked.connect(panel.emitDataChanged)
    panel.vrpCheckBoxConn4.clicked.connect(panel.emitDataChanged)
    panel.vrpCheckBoxConn5.clicked.connect(panel.emitDataChanged)
    panel.vrpCheckBoxConnEnabled1.clicked.connect(panel.emitDataChanged)
    panel.vrpCheckBoxConnEnabled2.clicked.connect(panel.emitDataChanged)
    panel.vrpCheckBoxConnEnabled3.clicked.connect(panel.emitDataChanged)
    panel.vrpCheckBoxConnEnabled4.clicked.connect(panel.emitDataChanged)
    panel.vrpCheckBoxConnEnabled5.clicked.connect(panel.emitDataChanged)
    panel.vrpCheckBoxNeedToBeConn.clicked.connect(panel.emitDataChanged)
 
    ObjectMgr.ObjectMgr().sigGuiParamChanged.connect(panel.paramChanged)



def DNAItemPanelBlockSignals( panel, doBlock ):
    panel.blockSignals(doBlock)
    # block all widgets with apply
    applyWidgets = [
        panel.vrpCheckBoxConn1,
        panel.vrpCheckBoxConn2,
        panel.vrpCheckBoxConn3,        
        panel.vrpCheckBoxConn4,        
        panel.vrpCheckBoxConn5,
        panel.vrpCheckBoxConnEnabled1,
        panel.vrpCheckBoxConnEnabled2,
        panel.vrpCheckBoxConnEnabled3,        
        panel.vrpCheckBoxConnEnabled4,        
        panel.vrpCheckBoxConnEnabled5,
        panel.vrpCheckBoxNeedToBeConn                
    ]

    for widget in applyWidgets:
        widget.blockSignals(doBlock)
