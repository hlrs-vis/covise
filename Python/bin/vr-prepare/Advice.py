
from PyQt5 import QtCore, QtGui, QtWidgets

from AdviceBase import Ui_AdviceBase
import Application

class AdviceBase(QtWidgets.QWidget, Ui_AdviceBase):
    def __init__(self, parent=None, f=QtCore.Qt.WindowFlags()):
        QtWidgets.QWidget.__init__(self, parent, f)
        self.setupUi(self)
        
class Advice(QtWidgets.QDockWidget):

    def __init__(self, parent):
        QtWidgets.QDockWidget.__init__(self, "Advice", parent)
        
        self.setWidget(AdviceBase(self))

        #connection of the DockWidget visibilityChanged
        self.visibilityChanged.connect(self.visibilityChangedS)

    def visibilityChangedS(self, visibility):
        if Application.vrpApp.mw:
            Application.vrpApp.mw.windowAdviceAction.setChecked(self.isVisible()) # don't use visibility !! (see below)
        # If the DockWidget is displayed tabbed with other DockWidgets and the tab becomes inactive, visiblityChanged(false) is called.
        # Using visibility instead of self.isVisible() this would uncheck the menuentry and hide the DockWidget (including the tab).

    def show(self):
        QtWidgets.QWidget.show(self)

    def hide(self):
        QtWidgets.QWidget.hide(self)

    def addToStack(self, widget):
        self.widget().widgetStackDocu.addWidget(widget)
        
    def raiseInStack(self, widget):
        self.widget().widgetStackDocu.setCurrentWidget(widget)
        
    def getWidgetStack(self):
        return self.widget().widgetStackDocu

