
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH



from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject, pyqtSignal
   
from ChangeIndicatedLE import ChangeIndicatedLE

from GenericObjectPanelBase import Ui_GenericObjectPanelBase
from coGenericObjectMgr import coGenericObjectParams, PARAM_TYPE_BOOL, PARAM_TYPE_INT, PARAM_TYPE_FLOAT, PARAM_TYPE_STRING, PARAM_TYPE_VEC3, PARAM_TYPE_MATRIX
from ObjectMgr import ObjectMgr, GUI_PARAM_CHANGED_SIGNAL
from Utils import CopyParams, getIntInLineEdit, getDoubleInLineEdit
from Gui2Neg import theGuiMsgHandler
import Application

from printing import InfoPrintCapable

from vtrans import coTranslate 

class GenericObjectPanel(QtWidgets.QWidget,Ui_GenericObjectPanelBase):


    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        Ui_GenericObjectPanelBase.__init__(self)
        self.setupUi(self)
        self.__name2label = {}
        self.__name2input = {}
        self.__key = -1

        # designer could not assign a layout to empty widgets
        self.dummyFrameLayout = QtWidgets.QVBoxLayout(self.dummyFrame)

        # add scroll view
        self.scrollView = QtWidgets.QScrollArea(self.dummyFrame)
        self.dummyFrameLayout.addWidget(self.scrollView)
        self.scrollView.setFrameShadow(QtWidgets.QFrame.Raised)
        self.scrollView.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.scrollView.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)

        # add a box for the widgets in the scrollview.
        self.box = QtWidgets.QWidget(self.scrollView)
        self.box.resize(250,2000)
        self.scrollView.setWidget(self.box)
        self.boxLayout = QtWidgets.QGridLayout(self.box)

        ObjectMgr().sigGuiParamChanged.connect(self.paramChanged)

    def __addGenericParam(self, paramName, paramType, value):
        self.__name2label[paramName] = QtWidgets.QLabel(self.box)
        self.__name2label[paramName].setGeometry(QtCore.QRect(10,20,150,24))
        self.__name2label[paramName].setText(paramName)
        if (paramType == PARAM_TYPE_BOOL):
            self.__name2input[paramName] = QtWidgets.QCheckBox(self.box)
            self.__name2input[paramName].setChecked(value)
            __name2input[paramName].toggled.connect(self.emitChange)
        elif (paramType == PARAM_TYPE_INT):
            self.__name2input[paramName] = ChangeIndicatedLE(self.box)
            self.__name2input[paramName].setText(str(value))
            __name2input[paramName].returnPressed.connect(self.emitChange)
        elif (paramType == PARAM_TYPE_FLOAT):
            self.__name2input[paramName] = ChangeIndicatedLE(self.box)
            self.__name2input[paramName].setText(str(value))
            __name2input[paramName].returnPressed.connect(self.emitChange)
        elif (paramType == PARAM_TYPE_VEC3):
            self.__name2input[paramName] = ChangeIndicatedLE(self.box)
            self.__name2input[paramName].setText(str(value[0]) + "/" + str(value[1]) + "/" + str(value[2]))
            __name2input[paramName].returnPressed.connect(self.emitChange)
        elif (paramType == PARAM_TYPE_MATRIX):
            self.__name2input[paramName] = ChangeIndicatedLE(self.box)
            self.__name2input[paramName].setText(str(value[ 0]) + "/" + str(value[ 1]) + "/" + str(value[ 2]) + "/" + str(value[ 3]) + "/" + \
                                                 str(value[ 4]) + "/" + str(value[ 5]) + "/" + str(value[ 6]) + "/" + str(value[ 7]) + "/" + \
                                                 str(value[ 8]) + "/" + str(value[ 9]) + "/" + str(value[10]) + "/" + str(value[11]) + "/" + \
                                                 str(value[12]) + "/" + str(value[13]) + "/" + str(value[14]) + "/" + str(value[15]))
            __name2input[paramName].returnPressed.connect(self.emitChange)
        else:
            self.__name2input[paramName] = ChangeIndicatedLE(self.box)
            self.__name2input[paramName].setText(value)
            __name2input[paramName].returnPressed.connect(self.emitChange)
        self.__name2input[paramName].setGeometry(QtCore.QRect(160,20,200,24))
        if (paramName == "NextPresStepAllowed"):
            self.__name2input[paramName].setEnabled(False)
        #self.__name2input[paramName].setSizePolicy(QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred,QtWidgets.QSizePolicy.Fixed))
        self.boxLayout.addWidget(self.__name2label[paramName], self.cnt, 0)
        self.boxLayout.addWidget(self.__name2input[paramName], self.cnt, 1)
        self.cnt=self.cnt+1

    def __clearGenericParams(self):
        if hasattr(self, "boxLayout"):
            for paramName in list(self.__name2label.keys()):
                self.__name2label[paramName].setVisible(False) # Does not work without setVisible(false) but should, widget seems not to be removed properly.
                self.__name2input[paramName].setVisible(False) # Does not work without setVisible(false) but should, widget seems not to be removed properly.
                self.boxLayout.removeWidget(self.__name2label[paramName])
                self.boxLayout.removeWidget(self.__name2input[paramName])
                del self.__name2label[paramName]
                del self.__name2input[paramName]
        self.boxLayout.setSpacing(6)
        self.boxLayout.setContentsMargins(11, 11, 11, 11)
        self.boxLayout.setAlignment(QtCore.Qt.AlignTop)
        self.box.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        self.cnt = 0

    def emitChange(self, notNeeded=False):
        if not self.__key==-1:
            Application.vrpApp.key2params[self.__key] = self.__getParams()
            ObjectMgr().setParams( self.__key, self.__getParams() )
            theGuiMsgHandler().runObject( self.__key )

    def __getParams(self):
        params = CopyParams(ObjectMgr().getParamsOfObject(self.__key))
        for paramName, paramType in iter(params.gpTypes.items()):
            if (paramType == PARAM_TYPE_BOOL):
                  params.gpValues[paramName] = self.__name2input[paramName].isChecked()
            elif (paramType == PARAM_TYPE_INT):
                  params.gpValues[paramName] = getIntInLineEdit(self.__name2input[paramName])
            elif (paramType == PARAM_TYPE_FLOAT):
                  params.gpValues[paramName] = getDoubleInLineEdit(self.__name2input[paramName])
            elif (paramType == PARAM_TYPE_VEC3):
                  params.gpValues[paramName] = [float(x) for x in self.__name2input[paramName].text().split("/")]
            elif (paramType == PARAM_TYPE_MATRIX):
                  params.gpValues[paramName] = [float(x) for x in self.__name2input[paramName].text().split("/")]
            else:
                  params.gpValues[paramName] = str(self.__name2input[paramName].text())
        return params

    def __tr(self,s,c = None):
        return coTranslate(s)

    def update( self ):
        if (self.__key == -1):
            return
        self.__clearGenericParams()
        params = ObjectMgr().getParamsOfObject(self.__key)
        self.vrpLabelTitle.setText(params.name)
        # add widgets
        for paramName in sorted(params.gpTypes.keys()):
            self.__addGenericParam( paramName, params.gpTypes[paramName], params.gpValues[paramName] )

    def paramChanged(self, key): 
        ''' params of the object with key changed '''
        if self.__key == key:
            self.update()

    def updateForObject( self, key ):
        self.__key = key
        self.update()
