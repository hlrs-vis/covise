
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH

from PyQt5 import QtCore, QtGui, QtWidgets

from VideoCaptureManagerBase import Ui_VideoCaptureManagerBase
from coGRMsg import coGRSnapshotMsg, coGRKeyWordMsg, coGRTurnTableAnimationMsg
import covise
from os import getenv, path
from Gui2Neg import theGuiMsgHandler
from Utils import ReallyWantToOverrideAsker
import Application

from vtrans import coTranslate 

class VideoCaptureManagerBase(QtWidgets.QWidget, Ui_VideoCaptureManagerBase):
    def __init__(self, parent=None, f=QtCore.Qt.WindowFlags()):
        QtWidgets.QWidget.__init__(self, parent, f)
        self.setupUi(self)
        
class VideoCaptureManager(QtWidgets.QDockWidget):

    def __init__(self, parent):
        QtWidgets.QDockWidget.__init__(self, coTranslate("Video Capture Manager"), parent)

        self.setWidget(VideoCaptureManagerBase(self))

        #connection of the DockWidget visibilityChanged
        self.visibilityChanged.connect(self.visibilityChangedS)
        
        # connection of buttons
        self.widget().buttonCapture.clicked.connect(self.capture)
        self.widget().buttonPreview.clicked.connect(self.startPreview)
        self.widget().comboBox.activated.connect(self.selectMode)
        
        # connect open
        self.widget().pushButton.clicked.connect(self.openFile)
        
        # connect filename
        self.widget().lineEdit.returnPressed.connect(self.setFilename)
        self.widget().lineEdit.editingFinished.connect(self.checkFilename)

        self.widget().groupSettings.setVisible(True)
        self.widget().groupCapturing.setVisible(True)
        #self.recording = False;
        
        self.filename = "C:\capture.wmv"
        if covise.coConfigIsOn("COVER.Plugin.Video", False):
            filename = covise.getCoConfigEntry("COVER.Plugin.Video.Filename")
            if filename:
                self.filename = filename
        
        self.widget().lineEdit.setText(self.filename)
        self.oldFilename = self.filename
        self.freeCapture = False
        self.mode=0
    
    def checkFilename(self):
        tmpFileName = unicode(self.widget().lineEdit.text()).encode('utf-8')
        if not tmpFileName.strip() == "" and not path.isdir(tmpFileName):
            if not tmpFileName.endswith(".wmv"):
                self.widget().lineEdit.setText(tmpFileName + ".wmv")
            return True
        else:
            return False

    def openFile(self):
        fd = QtWidgets.QFileDialog(self)
        fd.setMinimumWidth(654)
        fd.setMinimumHeight(488)
        fd.setNameFilter(coTranslate("Windows Media Video (*.wmv)"))
        fd.setWindowTitle(coTranslate('Save As...'))
        fd.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        fd.setConfirmOverwrite(False)
        tmpFileName = unicode(self.widget().lineEdit.text()).encode('utf-8')
        if path.exists(tmpFileName):
            fd.setDirectory(tmpFileName)
        else:
            fd.setDirectory(getenv("COVISEDIR"))
        acceptedOrRejected = fd.exec_()
        if acceptedOrRejected != QtWidgets.QDialog.Accepted or fd.selectedFiles().isEmpty():
            return
        tmpFileName = unicode(fd.selectedFiles()[0]).encode('utf-8')
        if not tmpFileName == "":
            if not tmpFileName.endswith(".wmv"):
                tmpFileName += ".wmv"
            self.widget().lineEdit.setText(tmpFileName)
            self.setFilename()
    
    def selectMode(self):
        self.mode =  self.widget().comboBox.currentIndex()
        if self.mode == 2:
            # free capturing, disable preview, rename the capture button
            self.widget().buttonCapture.setText(coTranslate("Start Free Capturing"))
            self.widget().buttonPreview.setVisible(False)
        else:    
            self.widget().buttonPreview.setVisible(True)
            self.widget().buttonCapture.setText(coTranslate("Capture Animation"))
            if self.freeCapture:
                # stop capturing
                msg = coGRSnapshotMsg(self.filename, "stopCapturing")
                covise.sendRendMsg(msg.c_str())
            
    def startPreview(self):
        if self.mode == 0:
            msg = coGRTurnTableAnimationMsg(10.0)
            covise.sendRendMsg(msg.c_str())
        if self.mode == 1:
            msg = coGRTurnTableAnimationMsg(20.0)
            covise.sendRendMsg(msg.c_str())
        
    def visibilityChangedS(self, visibility):
        if Application.vrpApp.mw:
            Application.vrpApp.mw.videoCaptureAction.setChecked(self.isVisible()) # don't use visibility !! (see below)
        # If the DockWidget is displayed tabbed with other DockWidgets and the tab becomes inactive, visiblityChanged(false) is called.
        # Using visibility instead of self.isVisible() this would uncheck the menuentry and hide the DockWidget (including the tab).

    def show(self):
        QtWidgets.QWidget.show(self)

    def hide(self):
        QtWidgets.QWidget.hide(self)
        
    def setFilename(self):
        if self.checkFilename():
            tmpFileName = unicode(self.widget().lineEdit.text()).encode('utf-8')
            if path.abspath(tmpFileName) != path.abspath(self.filename) and path.isfile(tmpFileName):
                asker = ReallyWantToOverrideAsker(self, tmpFileName)
                decicion = asker.exec_()
                if decicion == QtWidgets.QDialog.Rejected:
                    self.widget().lineEdit.setText(self.oldFilename)
            self.filename = tmpFileName
            self.oldFilename = self.filename
         
    def capture(self):    
        if self.mode == 2: # free capturing
            if  not self.freeCapture: # start capture and rename button to stop capture
                self.freeCapture = True
                self.widget().buttonCapture.setText(coTranslate("Stop Free Capturing"))
                msg = coGRSnapshotMsg(self.filename, "startCapturing")
                covise.sendRendMsg(msg.c_str())
            else: # stop capturing
                self.freeCapture = False
                self.widget().buttonCapture.setText(coTranslate("Start Free Capturing"))
                msg = coGRSnapshotMsg(self.filename, "stopCapturing")
                covise.sendRendMsg(msg.c_str())
                
        elif self.mode == 0: # turntable animation 10 sec
            if self.freeCapture:
                self.freeCapture = False
                msg = coGRSnapshotMsg(self.filename, "stopCapturing")
                covise.sendRendMsg(msg.c_str())
            msg = coGRSnapshotMsg(self.filename, "startCapturing")
            covise.sendRendMsg(msg.c_str())
            msg = coGRTurnTableAnimationMsg(10.0)
            covise.sendRendMsg(msg.c_str())
        else:  # turntable animation 20 sec
            if self.freeCapture:
                self.freeCapture = False
                msg = coGRSnapshotMsg(self.filename, "stopCapturing")
                covise.sendRendMsg(msg.c_str())
            msg = coGRSnapshotMsg(self.filename, "startCapturing")
            covise.sendRendMsg(msg.c_str())
            msg = coGRTurnTableAnimationMsg(20.0)
            covise.sendRendMsg(msg.c_str())
