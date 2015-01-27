
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH

from PyQt5 import QtCore, QtGui

import os
import covise

from printing import InfoPrintCapable

import Application
from GettingStartedWindowBase import Ui_GettingStartedWindowBase


class GettingStartedWindow(Ui_GettingStartedWindowBase):


    def __init__(self, parent=None):
        Ui_GettingStartedWindowBase.__init__(self, parent)
        self.setupUi(self)
        str = "file:///"
        str = str + os.path.dirname(__file__)
        str = str + "/documents/"
        config = covise.getCoConfigEntry("vr-prepare.GettingStartedDocument")
        if config == None:
            config = "GettingStarted/index.html"
        str = str + config
        self.textBrowser.setSource(QtCore.QUrl(str))

