
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH

from PyQt5 import QtCore, QtGui, QtWidgets

from ProjectInformationBase import Ui_ProjectInformationBase
from ProjectSetUp import ProjectSetUp

class ProjectInformation(QtWidgets.QWidget,Ui_ProjectInformationBase):

    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        Ui_ProjectInformationBase.__init__(self)
        self.setupUi(self)

        self.__setUpWidget = ProjectSetUp()
        self.widgetStack.addWidget(self.__setUpWidget)
        self.widgetStack.setCurrentWidget(self.__setUpWidget)
        self.widgetStack.show()

        self.__setUpWidget.showProjectInformation()

    def ok(self):
        self.__setUpWidget.updateProjectInformation()
        self.close()

# eof
