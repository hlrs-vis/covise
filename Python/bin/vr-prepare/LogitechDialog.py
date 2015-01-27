
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH

from PyQt5 import QtCore, QtGui

from printing import InfoPrintCapable

import Application
from LogitechDialogBase import Ui_LogitechDialogBase

from vtrans import coTranslate 

class LogitechDialog(Ui_LogitechDialogBase):

    """ Handling of events from the logitech presenter

    """

    def __init__(self, parent=None):
        Ui_LogitechDialogBase.__init__(self, parent)
        self.setupUi(self)

    def keyPressEvent(self,e):
        if e.key()==0x01000017: # page down
            if Application.vrpApp.mw.presenterManager:
                Application.vrpApp.mw.presenterManager.forward()
        elif e.key()==0x01000016: # page up
            if Application.vrpApp.mw.presenterManager:
                Application.vrpApp.mw.presenterManager.backward()
        elif e.key()==4096: #esc
            pass
        elif e.key()==4148: #f5
            pass
        elif e.key()==46: #bildschirm
            pass
        #print int(e.key())

    def __tr(self,s,c = None):
        return coTranslate(s)

