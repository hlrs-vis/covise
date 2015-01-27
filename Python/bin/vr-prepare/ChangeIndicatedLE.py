
# Part of the vr-prepare program for dc

# Copyright (c) 2006-2007 Visenso GmbH

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject, pyqtSignal

class ChangeIndicatedLE(QtWidgets.QLineEdit):
    """For signalling non-finalized input to the user."""
    __signNotCommitedColor = QtGui.QColor('yellow')

    sigEnterPendingMode = pyqtSignal()
    sigLeavePendingMode = pyqtSignal()
    def __init__(self, parent=None, name=None):
        super(QtWidgets.QLineEdit, self).__init__(parent)
        self.__stdBgColor = self.palette().color(QtGui.QPalette.Background)
        self.__textBeforeInPendingMode = self.text()
        self.returnPressed.connect(self.__reactOnLineEditReturnPressed)
        self.textChanged.connect(self.__reactOnTextChange)
        self.editingFinished.connect(self.__undoPending)


    def inPendingMode(self):
        return self.__signNotCommitedColor == self.palette().color(QtGui.QPalette.Background)

    def setText(self, aText):
        """Set in widget and save aText for reset on focus loss."""
        self.__textBeforeInPendingMode = aText
        self.__resetToStoredText()

    def leavePendingMode(self):
        self._setBgColor(self.__stdBgColor)
        self.sigLeavePendingMode.emit()

    def enterPendingMode(self):
        self._setBgColor(self.__signNotCommitedColor)
        self.sigEnterPendingMode.emit()


    def __reactOnTextChange(self):
        self.enterPendingMode()

    def __reactOnLineEditReturnPressed(self):
        self.__textBeforeInPendingMode = self.text()
        self.leavePendingMode()

    def __undoPending(self):
        if not self.inPendingMode(): return
        self.__resetToStoredText()

    def __resetToStoredText(self):
        rem = self.signalsBlocked()
        self.blockSignals(True)
        self._setBgColor(self.__stdBgColor)
        QtWidgets.QLineEdit.setText(self, self.__textBeforeInPendingMode)
        QtWidgets.QLineEdit.home(self, False)
        self.blockSignals(rem)

    def _setBgColor( self, color ):
        palette = QtGui.QPalette();
        palette.setColor(self.backgroundRole(), color);
        self.setPalette(palette);
# eof
