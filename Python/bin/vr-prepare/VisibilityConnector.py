
# Part of the vr-prepare program for dc

# Copyright (c) 2006-2007 Visenso GmbH

from PyQt5 import QtCore

from Gui2Neg import theGuiMsgHandler
from VisItem import VisItemParams

class VisibilityConnector(QtCore.QObject):
    def forward(self, key, isOn):
        aux = VisItemParams()
        aux.isVisible = isOn
        theGuiMsgHandler().setParams(key, aux)

# eof
