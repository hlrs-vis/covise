
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QApplication

from MainWindow import MainWindow

from Gui2Neg import theGuiMsgHandler

from Application import vrpApp

from negGuiHandlers import initHandlers

import covise

from vtrans import languageLocale
qapp = None

def main():
    import sys
    import os
    

    global qapp 
    qapp = QApplication(sys.argv)
    
    # translation of standard dialogs BEGIN #
    
    if os.getenv('QT_HOME') != None:
        translationsPath = os.getenv('QT_HOME') + os.sep + "translations"
        qtTranslator = QtCore.QTranslator()
        succ = qtTranslator.load("qt_%s" % languageLocale,  translationsPath)
        
        if (succ != True):
            print("Translator for standard dialogs not loaded!")
        else:
            qapp.installTranslator(qtTranslator)
    
    # translation of standard dialogs BEGIN #
    
    ui = MainWindow()
    vrpApp.mw = ui # global hook
    
    # removes the message "event loop already running" when using pdb
    QtCore.pyqtRemoveInputHook()

    # Set up connections. Needs Neg2Gui and Gui2Neg
    initHandlers(vrpApp.mw.coverWidgetId)
    if os.getenv('COVISE_HIDDEN') == "1":
        vrpApp.mw.hide()
    else:
        vrpApp.mw.show()
    acceptedOrRejected = vrpApp.mw.openInitialDialog()
    if acceptedOrRejected == QtWidgets.QDialog.Accepted:
        qapp.exec_()

    covise.clean()
    covise.quit()

# eof
