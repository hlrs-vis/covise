
from PyQt5 import QtCore, QtGui

import ObjectMgr

def CoxmlBrowserPanelConnector(panel):
    '''Connections for the panel of corresponding visualizer'''

    
    panel.listView.clicked.connect(panel.listViewClicked)
    panel.listView.doubleClicked.connect(panel.listViewDoubleClicked)

    panel.plButton1.clicked.connect(panel.productLineClicked)
    panel.plButton2.clicked.connect(panel.productLineClicked)
    panel.plButton3.clicked.connect(panel.productLineClicked)
    panel.plButton4.clicked.connect(panel.productLineClicked)
    panel.plButton5.clicked.connect(panel.productLineClicked)
    panel.plButton6.clicked.connect(panel.productLineClicked)
    panel.plButton7.clicked.connect(panel.productLineClicked)

