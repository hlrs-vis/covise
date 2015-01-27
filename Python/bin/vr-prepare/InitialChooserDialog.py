# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui

from InitialChooserDialogBase import Ui_InitialChooserDialogBase
import covise

from vtrans import coTranslate 

class InitialChooserDialog(QtWidgets.QDialog, Ui_InitialChooserDialogBase ):
    
    def __init__( self, parent ):
        QtWidgets.QDialog.__init__(self, parent)
        Ui_InitialChooserDialogBase.__init__(self)
        
    def setupDialog(self):
        Ui_InitialChooserDialogBase.setupUi( self, self )
        # if not name: self.setName("InitialChooserDialog")
        self.checkBox12.hide() # hide ShowOnStart-checkbox

    def newProjectSlot(self):
        self.hide()
        self.newProjectRequest.emit(None)
        self.accept()

    def openProjectSlot(self):
        self.hide()
        self.openProjectRequest.emit(None)
        self.accept()

    def __tr(self,s,c = None):
        return coTranslate(s)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QDialog()
    ui = InitialChooserDialog(Form)
    ui.setupDialog()    
    ui.show()
    sys.exit(app.exec_())
