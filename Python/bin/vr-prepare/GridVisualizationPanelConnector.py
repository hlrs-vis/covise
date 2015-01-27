
from PyQt5 import QtCore, QtGui
import ObjectMgr

def GridVisualizationPanelConnector(panel):
    
    panel.StreamlinesPushButton.clicked.connect(panel.emitStreamlineRequest)
    panel.MovingPointsPushButton.clicked.connect(panel.emitMovingPointsRequest)
    panel.PathlinesPushButton.clicked.connect(panel.emitPathlinesRequest)
    panel.vrpComboBoxVariable.activated[str].connect(panel._enableMethodButtsForVariableSlot)
    panel.CuttingSurfaceColoredPushButton.clicked.connect(panel.emitPlaneRequest)
    panel.CuttingSurfaceArrowPushButton.clicked.connect(panel.emitArrowsRequest)
    panel.IsoSurfacePushButton.clicked.connect(panel.emitIsoPlaneRequest)
    panel.DomainLinesPushButton.clicked.connect(panel.emitDomainLinesRequest)
    panel.DomainSurfacePushButton.clicked.connect(panel.emitDomainSurfaceRequest)
    panel.StreamlinesPushButton.clicked.connect(panel.emitStreamlineRequest)
    panel.StreamlinesPushButton.clicked.connect(panel.emitStreamlineRequest)
    panel.StreamlinesPushButton.clicked.connect(panel.emitStreamlineRequest)
    panel.StreamlinesPushButton.clicked.connect(panel.emitStreamlineRequest)
    panel.StreamlinesPushButton.clicked.connect(panel.emitStreamlineRequest)

  
    ObjectMgr.ObjectMgr().sigGuiParamChanged.connect(panel.paramChanged)
    #validators:
    # allow only double values for transform lineEdits
    doubleValidator = QtGui.QDoubleValidator(panel)
    panel.floatX.setValidator(doubleValidator)
    panel.floatY.setValidator(doubleValidator)
    panel.floatZ.setValidator(doubleValidator)
