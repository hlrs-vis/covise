
# Part of the vr-prepare program for dc

# Copyright (c) 2007 Visenso GmbH

from PyQt5 import QtCore, QtGui, QtWidgets

import os
import sys

from PatienceDialogBase import Ui_PatienceDialogBase 

from vtrans import coTranslate 

class PatienceDialog(Ui_PatienceDialogBase):
    def __init__(self, patienceText=None):
        Ui_PatienceDialogBase.__init__(self)
        self.setupUi(self)

        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.FramelessWindowHint)

        self.vrpLabelWait.setText(self.__tr("Please wait... %s") % patienceText)

        #progress bar
        self.__progressSteps    = 0
        self.__progressMaxSteps = 20
        self.progressWidget.setMinimum(0)
        self.progressWidget.setMaximum(self.__progressMaxSteps)
        self.progressWidget.setTextVisible(False)

        width=QtWidgets.QApplication.desktop().screenGeometry().width()
        height=QtWidgets.QApplication.desktop().screenGeometry().height()
        self.setGeometry ( int(width/2)-225, int(height/2)-25, self.width(), self.height())

        timer = QtCore.QTimer(self)
        timer.setSingleShot(False)
        timer.timeout.connect(self.incProgress)
        timer.start(200)

    def incProgress(self):
        # the parent process is vr-prepare -> if the parents id is 1, vr-prepare is not running any more
        if (os.getppid() == 1):
            QtWidgets.QApplication.quit()
        # increase process
        self.__progressSteps = self.__progressSteps + 1
        if self.__progressSteps>self.__progressMaxSteps:
            self.__progressSteps = 0
        self.progressWidget.setValue(self.__progressSteps)

    def __tr(self,s,c = None):
        return coTranslate(s)


if __name__ == "__main__":
    a = QtWidgets.QApplication(sys.argv)
    argc = len(sys.argv)
    if argc==2:
        w = PatienceDialog(sys.argv[1])
    else:
        w = PatienceDialog(self.__tr("COVISE is loading"))
    w.show()
    a.setQuitOnLastWindowClosed(True)
    a.exec_()

# eof
