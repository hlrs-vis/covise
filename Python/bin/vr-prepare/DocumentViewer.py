
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject, pyqtSignal

import Application

from DocumentViewerBase import Ui_DocumentViewerBase
from coDocumentMgr import coDocumentMgrParams
from Gui2Neg import theGuiMsgHandler
from ObjectMgr import ObjectMgr, GUI_PARAM_CHANGED_SIGNAL
from Utils import getDoubleInLineEdit
import covise

from printing import InfoPrintCapable
_infoer = InfoPrintCapable()
_infoer.doPrint = False # True #

from vtrans import coTranslate 

class DocumentViewer(QtWidgets.QWidget,Ui_DocumentViewerBase):

    """For controling parameters of a document.

    """

    def __init__(self, parent=None):
        _infoer.function = str(self.__init__)
        _infoer.write("")
        QtWidgets.QWidget.__init__(self, parent)
        Ui_DocumentViewerBase.__init__(self)
        self.setupUi(self)
        self.__key = -1

        changeIndicatedLEs = [ self.floatX, self.floatY, self.floatZ, self.floatHsize, self.floatVsize, self.floatScaling  ]

        for w in changeIndicatedLEs:
            w.returnPressed.connect(self.emitParamsChange)

        
        self.spinBoxPage.valueChanged.connect(self.emitParamsChange)
        self.ChangeButton.clicked.connect(self.change)

        ObjectMgr().sigGuiParamChanged.connect(self.paramChanged)
        #validators:
        # allow only double values for changeIndicatedLEs
        doubleValidator = QtGui.QDoubleValidator(self)
        self.floatX.setValidator(doubleValidator)
        self.floatY.setValidator(doubleValidator)
        self.floatZ.setValidator(doubleValidator)
        self.floatHsize.setValidator(doubleValidator)
        self.floatVsize.setValidator(doubleValidator)
        self.floatScaling.setValidator(doubleValidator)

        self.spinBoxPage.setMinimum(1)
        self.spinBoxPage.setMaximum(1)
        self.currentImage_ = None
        self.currentPixmap_ = None

        self.documentsInGUI_ = covise.coConfigIsOn("vr-prepare.DocumentsInGUI", False)

    def paramChanged( self, key ):
        _infoer.function = str(self.paramChanged)
        _infoer.write("key %d" %(key))
        """ params of object key changed"""
        if self.__key==key:
            self.update()

    #change document values for all presentationsteps
    def change(self):
        _infoer.function = str(self.change)
        _infoer.write("")

        # add asker
        msgBox = QtWidgets.QMessageBox(Application.vrpApp.mw)
        msgBox.setWindowTitle(self.__tr("Save changes of documents"))
        msgBox.setText(self.__tr("Do you want to save your changes for the documents?"))
        msgBox.setStandardButtons(QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Cancel)
        msgBox.setDefaultButton(QtWidgets.QMessageBox.Save)
        acceptedOrRejected = msgBox.exec_()
        if acceptedOrRejected == QtWidgets.QMessageBox.Save:
            # comunication with negotiator
            params = ObjectMgr().getParamsOfObject(self.__key)
            params.changed = True
            ObjectMgr().setParams( self.__key, params )

    def update( self ):
        _infoer.function = str(self.update)
        _infoer.write("")
        if self.__key!=-1:
            self.updateForObject( self.__key )

    def updateForObject( self, key ):
        """ called from MainWindow to update the content to the choosen object key """
        _infoer.function = str(self.updateForObject)
        _infoer.write("key %d" %(key))
        self.__key = key
        params = ObjectMgr().getParamsOfObject(key)
        self.__setParams( params )

    def __getParams(self):
        _infoer.function = str(self.__getParams)
        _infoer.write("")
        data = coDocumentMgrParams()
        data.documentName = str(self.DocumentName.text())
        data.name = data.documentName
        data.imageName = str(self.ImageName.text())

        data.pageNo = self.spinBoxPage.value()
        data.pos = (getDoubleInLineEdit(self.floatX),
                    getDoubleInLineEdit(self.floatY),
                    getDoubleInLineEdit(self.floatZ))
        data.size = (getDoubleInLineEdit(self.floatHsize),
                    getDoubleInLineEdit(self.floatVsize))
        data.scaling = getDoubleInLineEdit(self.floatScaling)
        data.isVisible = self.__isVisible
        data.minPage = self.spinBoxPage.minimum()
        data.maxPage = self.spinBoxPage.maximum()
        # remember current picture
        data.currentImage = self.currentImage_
        return data


    def __setParams( self, params ):
        # block all widgets with apply
        _infoer.function = str(self.__setParams)
        _infoer.write("")
        applyWidgets = [
            self.floatX,
            self.floatY,
            self.floatZ,
            self.floatHsize,
            self.floatVsize,
            self.floatScaling,
            self.spinBoxPage]
        for widget in applyWidgets:
            widget.blockSignals(True)
        if isinstance( params, int):
            self.__key = params
            return
        if hasattr(params, 'documentName'): self.DocumentName.setText(str(params.documentName))
        if hasattr(params, 'imageName'):
            self.ImageName.setText(str(params.imageName))
        if hasattr(params, 'minPage'):
            self.spinBoxPage.setMinimum(params.minPage)
        if hasattr(params, 'maxPage'):
            self.spinBoxPage.setMaximum(params.maxPage)
        if hasattr(params, 'pageNo'):
            self.spinBoxPage.setValue(params.pageNo)
        if hasattr(params, 'pos'):
            self.floatX.setText(str(params.pos[0]))
            self.floatY.setText(str(params.pos[1]))
            self.floatZ.setText(str(params.pos[2]))
        if hasattr(params, 'size'):
            self.floatHsize.setText(str(params.size[0]))
            self.floatVsize.setText(str(params.size[1]))
        if hasattr(params, 'scaling'):
            self.floatScaling.setText(str(params.scaling))
        if hasattr(params, 'currentImage'):
            if self.documentsInGUI_ :
                self.showDocumentInPanel(params.currentImage)
                self.currentImage_ = params.currentImage

        self.__isVisible = params.isVisible

        for widget in applyWidgets:
            widget.blockSignals(False)

   # shows the document in the panel
    def showDocumentInPanel(self, currentImage ):
        if(currentImage != None):
            # maps currentImage from COVER to Panel
            self.currentPixmap_ = QtGui.QPixmap( str(currentImage) )
            self.resizeEvent(None)

    def resizeEvent(self, event): # implement QWidgets resizeEvent
        if self.currentPixmap_:
            self.document.setPixmap( self.currentPixmap_.scaled(self.document.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation) )

    def emitParamsChange(self):
        _infoer.function = str(self.emitParamsChange)
        _infoer.write("")
        if not self.__key==-1:
            Application.vrpApp.key2params[self.__key] = self.__getParams()
            ObjectMgr().setParams( self.__key, self.__getParams() )
            theGuiMsgHandler().runObject( self.__key )

    def __tr(self,s,c = None):
        return coTranslate(s)

# eof
