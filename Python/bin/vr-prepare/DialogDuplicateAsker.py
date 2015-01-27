
from PyQt5 import QtCore, QtGui

from DialogDuplicateBase import Ui_DialogDuplicateBase

class DialogDuplicateAsker(Ui_DialogDuplicateBase):

    def __init__(self, parent):
        Ui_DialogDuplicateBase.__init__(self, parent)
        self.setupUi(self)

        #remember values
        self.number = 1
        self.axisX = 0.0
        self.axisY = 0.0
        self.axisZ = 1.0
        self.angle = 180.0
        self.transX = 0.0
        self.transY = 0.0
        self.transZ = 0.0

        #default values
        self.editNumber.setValidator(QtGui.QIntValidator(1, 100, self))
        self.editNumber.setText('1')
        floatInRangeAxis = [self.floatInRangeAxisX, self.floatInRangeAxisY, self.floatInRangeAxisZ]
        for w in floatInRangeAxis:
            w.setRange([-1,1])
            w.setValue(0.0)
        self.floatInRangeAxisZ.setValue(1.0)
        self.floatInRangeAngle.setRange([-180,180])
        self.floatInRangeAngle.setValue(180.0)
        floatInRangeTrans = [self.floatInRangeX, self.floatInRangeY, self.floatInRangeZ]
        for w in floatInRangeTrans:
            w.setRange([-50, 50])
            w.setValue(0.0)
        
        panel.buttonOk.clicked.connect(self.__buttonOkPressed)
        panel.editNumber.returnPressed.connect(self.__numberEdit)
        panel.floatInRangeAxisX.sigSliderReleased.connect(self.__axisEdit)
        panel.floatInRangeAxisY.sigSliderReleased.connect(self.__axisEdit)
        panel.floatInRangeAxisZ.sigSliderReleased.connect(self.__axisEdit)
        panel.floatInRangeAngle.sigSliderReleased.connect(self.__angleEdit)
        panel.floatInRangeX.sigSliderReleased.connect(self.__transEdit)
        panel.floatInRangeY.sigSliderReleased.connect(self.__transEdit)
        panel.floatInRangeZ.sigSliderReleased.connect(self.__transEdit)

        self.buttonOk.setDefault(False)

    def __buttonOkPressed(self):
        self.accept()

    def __numberEdit(self):
        self.number = int(str(self.editNumber.text()))
        self.angle = 360.0 / (self.number+1)
        self.floatInRangeAngle.setValue(self.angle)

    def __axisEdit(self):
        self.axisX = self.floatInRangeAxisX.getValue()
        self.axisY = self.floatInRangeAxisY.getValue()
        self.axisZ = self.floatInRangeAxisZ.getValue()

    def __angleEdit(self):
        self.angle = self.floatInRangeAngle.getValue()

    def __transEdit(self):
        self.transX = self.floatInRangeX.getValue()
        self.transY = self.floatInRangeY.getValue()
        self.transZ = self.floatInRangeZ.getValue()
