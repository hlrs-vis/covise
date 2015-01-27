
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH

from PyQt5 import QtCore, QtGui, QtWidgets

from printing import InfoPrintCapable

import Application
from ColorChooserBase import Ui_ColorChooserBase

import covise


# Colors defined as tuple (r,g,b,a) each from 0 to 255
# Returns the new color or None if aborted
# If a nameless palette exists it will always be used (as long as no palette matches the given paletteName)
def getColor(parent=None, color=None, palette=None):
    
    colors = [] # list of (name, (color tuple))

    if (palette != None):
        entries = covise.getCoConfigSubEntries("vr-prepare.ColorPalette:" + palette)
        for entry in entries:
            name = entry.split(":")[-1]
            color = covise.getCoConfigEntry("vr-prepare.ColorPalette:" + palette + "." + entry)
            color = tuple([int(v) for v in color.split()])
            colors.append((name, color))

    if (len(colors) == 0):
        entries = covise.getCoConfigSubEntries("vr-prepare.ColorPalette")
        for entry in entries:
            name = entry.split(":")[-1]
            color = covise.getCoConfigEntry("vr-prepare.ColorPalette." + entry)
            color = tuple([int(v) for v in color.split()])
            colors.append((name, color))

    if (len(colors) > 0):
        cc = ColorChooser(parent)
        for (name, color) in colors:
            cc.addColor(name, color)
        cc.exec_()
        return cc.selectedColor

    # no palette, use default dialog
    if color == None:
        newcolor = QtWidgets.QColorDialog.getColor(parent)
    else:
        newcolor = QtWidgets.QColorDialog.getColor(QtGui.QColor(color[0], color[1], color[2]),parent)
    if newcolor.isValid():
        return (newcolor.red(), newcolor.green(), newcolor.blue(), color[3])
    else:
        return None


class ClickableLabel(QtWidgets.QLabel):
    
    def __init__(self, parent):
        QtWidgets.QLabel.__init__(self, parent)
        
    def mousePressEvent(self, event):
        self.clicked.emit()


class ClickableFrame(QtWidgets.QLabel):
    
    def __init__(self, parent):
        QtWidgets.QLabel.__init__(self, parent)
        
    def mousePressEvent(self, event):
        self.clicked.emit()


class ColorChooser(QtWidgets.QWidget,Ui_ColorChooserBase):


    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        Ui_ColorChooserBase.__init__(self)
        self.setupUi(self)
        self._layout = QtWidgets.QGridLayout(self.frame)
        self._count = 0
        self._widget2color = {}
        self.selectedColor = None

    def addColor(self, name, color):
        colorFrame = ClickableFrame(self.frame)
        colorFrame.setStyleSheet("background-color: rgb(%d,%d,%d)" % (color[0], color[1], color[2]))
        self._layout.addWidget(colorFrame, self._count, 0)
        colorFrame.clicked.connect(self.colorClicked)

        colorLabel = ClickableLabel(self.frame)
        colorLabel.setText(name)
        self._layout.addWidget(colorLabel, self._count, 1)
        colorLabel.clicked.connect(self.colorClicked)

        self._widget2color[colorFrame] = color
        self._widget2color[colorLabel] = color
        self._count = self._count + 1

    def colorClicked(self):
        if (self.sender() in self._widget2color):
            self.selectedColor = self._widget2color[self.sender()] 
        self.close()

    def exec_(self):
        spacer = QtWidgets.QSpacerItem(16, 16, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self._layout.addItem(spacer)
        ColorChooserBase.exec_(self)

