
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH



from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject, pyqtSignal

from CompositionPanelBase import Ui_CompositionPanelBase
from co3DComposedPartMgr import co3DComposedPartMgrParams
from ObjectMgr import ObjectMgr

import covise

from printing import InfoPrintCapable

from vtrans import coTranslate 

class GridCompositionPanel(QtWidgets.QWidget, Ui_CompositionPanelBase):

    """ Create grid compositions

    """
    sigComposedGridRequest = pyqtSignal()

    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        Ui_CompositionPanelBase.__init__(self)
        self.setupUi(self)
        self.CheckBoxVariable = {}
        self.ComboBoxVariable = {}
        self.__key2name = {}

        self.__key = -1

        # connect button
        self.composedButton.clicked.connect(self.emitComposedGridRequest)

       # designer could not assign a layout to empty widgets
        #
        self.dummyFrameLayout = QtWidgets.QVBoxLayout(self.dummyFrame)

        #
        # add scroll view
        #
        self.scrollView = QtWidgets.QScrollArea(self.dummyFrame)
        self.dummyFrameLayout.addWidget(self.scrollView)
        self.scrollView.setFrameShadow(QtWidgets.QFrame.Raised)
        self.scrollView.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.scrollView.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)

        # add a box for the widgets in the scrollview. 
        self.box = QtWidgets.QWidget(self.scrollView)
        self.box.resize(300,2000) # workaround: large height, because the automatic resize of self.box is not working
        self.scrollView.setWidget(self.box)
        self.boxLayout = QtWidgets.QGridLayout(self.box)

    def getComposedName(self):
        composedName = " "
        for key in self.__key2name:
            if self.CheckBoxVariable[key].isChecked():
                if composedName == " ":
                    composedName = self.__key2name[key]
                elif len(composedName)<30:
                    composedName += " + " + self.__key2name[key]
                else :
                    composedName += " ..."
                    return composedName
        return composedName


    def isUseful(self):
        return len(self.__key2name)>1

    def addGrid(self, key, name, vectorvars):
        if not key in self.__key2name.keys():
            self.__key2name[key] = name
            self.CheckBoxVariable[key] = QtWidgets.QCheckBox(self.box)
            self.CheckBoxVariable[key].setGeometry(QtCore.QRect(10,20,150,24))
            self.CheckBoxVariable[key].setText(name)
            self.CheckBoxVariable[key].setChecked(True)
            self.CheckBoxVariable[key].toggled.connect(self.checkSelection)
            self.ComboBoxVariable[key] = QtWidgets.QComboBox(self.box)
            self.ComboBoxVariable[key].setGeometry(QtCore.QRect(160,20,178,24))
            self.ComboBoxVariable[key].setSizePolicy(QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred,QtWidgets.QSizePolicy.Fixed))
            self.ComboBoxVariable[key].setCursor(QtGui.QCursor(QtCore.Qt.CursorShape(13)))
            for varname in vectorvars:
                self.ComboBoxVariable[key].addItem(varname)
        self.boxLayout.addWidget(self.CheckBoxVariable[key], self.cnt, 0)
        self.boxLayout.addWidget(self.ComboBoxVariable[key], self.cnt, 1)
        self.ComboBoxVariable[key].setVisible(covise.coConfigIsOn("vr-prepare.UseComposedVelocity", False))
        self.cnt=self.cnt+1

    def clearGrids(self):
        if hasattr(self, "groupBox5Layout"):
            for key in self.__key2name.keys():
                self.boxLayout.removeWidget(self.CheckBoxVariable[key])
                self.boxLayout.removeWidget(self.ComboBoxVariable[key])
        self.boxLayout.setSpacing(6)
        self.boxLayout.setContentsMargins(11, 11, 11, 11)
        self.boxLayout.setAlignment(QtCore.Qt.AlignTop)
        self.box.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)

    def getParams(self):
        p = co3DComposedPartMgrParams()
        p.name = self.getComposedName()
        for key in self.__key2name:
            if self.CheckBoxVariable[key].isChecked():
                p.subKeys.append( key )
                p.velDefinitions.append( str(self.ComboBoxVariable[key].currentText()) )
        return p

    def emitComposedGridRequest(self):
        self.sigComposedGridRequest.emit()

    def checkSelection(self, dummy):
        for key in self.__key2name:
            if self.CheckBoxVariable[key].isChecked():
                self.composedButton.setEnabled(True)
                return
        self.composedButton.setEnabled(False)

    def __tr(self,s,c = None):
        return coTranslate(s)

    def updateForObject( self, key):
        self.clearGrids()
        self.__key = key
        toFill = ObjectMgr().getAllChildrenGridsOfObject(self.__key)
        # delete obsolete widgets
        for gridkey in list(self.__key2name.keys()):
            if not gridkey in list(map(lambda x: x[0], toFill)):
                del self.__key2name[gridkey]
                del self.CheckBoxVariable[gridkey]
                del self.ComboBoxVariable[gridkey]
        # add widgets (create when nescessary)
        self.cnt = 0
        for gridkey, gridName, vectorVars in toFill:
            self.addGrid( gridkey, gridName, vectorVars )


# eof
