
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH

from PyQt5 import QtCore, QtGui, QtWidgets

from ProjectSetUpBase import Ui_ProjectSetUpBase

import ObjectMgr

from printing import InfoPrintCapable

_infoer = InfoPrintCapable()
_infoer.doPrint = False # True # 

from vtrans import coTranslate 

class ProjectSetUp(QtWidgets.QWidget,Ui_ProjectSetUpBase):

    def __init__(self):
        _infoer.function = str(self.__init__)
        _infoer.write("")
        QtWidgets.QWidget.__init__(self, None)
        Ui_ProjectSetUpBase.__init__(self)
        self.setupUi(self)

        self.vrpLineEditProject.setText(self.__tr("Project-1"))
        self.vrpEditDate.setDate(QtCore.QDate.currentDate())

    def showProjectInformation(self):
        _infoer.function = str(self.showProjectInformation)
        _infoer.write("")
        params = ObjectMgr.ObjectMgr().getParamsOfObject(0)
        self.vrpLineEditProject.setText(params.name)
        date = QtCore.QDate.fromString(params.creation_date, QtCore.Qt.ISODate)
        if date:
            self.vrpEditDate.setDate(date)
        self.vrpLineEditClient.setText(params.designer)
        self.vrpLineEditRegion.setText(params.region)
        self.vrpLineEditDivision.setText(params.division)
        self.vrpTextEditComment.setText(params.comment)

    def updateProjectInformation(self):
        _infoer.function = str(self.updateProjectInformation)
        _infoer.write("")
        # update params of project in ObjectMgr
        params = ObjectMgr.ObjectMgr().getParamsOfObject(0)
        params.name =  str(self.vrpLineEditProject.text())
        params.creation_date =  str(self.vrpEditDate.date().toString(QtCore.Qt.ISODate))
        params.designer =  str(self.vrpLineEditClient.text())
        params.region =  str(self.vrpLineEditRegion.text())
        params.division =  str(self.vrpLineEditDivision.text())
        params.comment =  str(self.vrpTextEditComment.toPlainText())
        ObjectMgr.ObjectMgr().setParams(0, params)

    def __tr(self, s, c = None):
        _infoer.function = str(self.__tr)
        _infoer.write("")
        return coTranslate(s)

# eof
