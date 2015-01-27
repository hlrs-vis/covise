
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH


import os
import sys

from PatienceDialog import PatienceDialog

from PyQt5 import QtCore, QtGui, QtWidgets

class PatienceDialogManager(QtCore.QObject):

    """ Controll of the PatienceDialog

    """

    def __init__(self, parent):
        QtCore.QObject.__init__(self)
        self.__proc=None


    def spawnPatienceDialog(self, patienceText=None):
        if self.__proc and (self.__proc.state() == QtCore.QProcess.Running):
            return

        self.__proc = QtCore.QProcess(self)
        #self.__proc.setProcessChannelMode(QtCore.QProcess.ForwardedChannels) # displays the output of the process

        pythoninterpreter=sys.executable

        #covisedir=os.getenv('COVISEDIR')
        #archsuffix=os.getenv('ARCHSUFFIX')
        #widgetfile=covisedir+'/Python/bin/vr-prepare/PatienceDialog.py'
        widgetfile = os.path.dirname(__file__) + '/PatienceDialog.py'

        arguments = []
        arguments.append(widgetfile)
        if patienceText:
            arguments.append(patienceText)

        self.__proc.start(pythoninterpreter, arguments)
        # For some reason this does not always work properly when vr-prepare is launched with a coProject.
        # In these cases, __proc.state() just stays 1 (STARTING).

        QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))

    def unSpawnPatienceDialog(self):
        if self.__proc:
            self.__proc.kill()
            del self.__proc
            self.__proc=None
            QtWidgets.QApplication.restoreOverrideCursor()

# eof
