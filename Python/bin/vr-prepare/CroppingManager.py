
from PyQt5 import QtCore, QtGui

from CroppingManagerBase import Ui_CroppingManagerBase

class CroppingManager(Ui_CroppingManagerBase):

    def __init__(self, parent):
        Ui_CroppingManagerBase.__init__(self, parent)
        #cursor = self.cursor()
        #cursor.setShape(Qt.ArrowCursor)
        #self.setCursor(cursor)
        self.setupUi(self)
        self.__pressedButton = None

        self.buttonOk.clicked.connect(self.__buttonYesPressed)
        self.buttonCancel.clicked.connect(self.__buttonNoPressed)

    def pressedYes(self):
        return self.__pressedButton == 0

    def pressedNo(self):
        return self.__pressedButton == 1

    def __buttonYesPressed(self):
        self.__pressedButton = 0
        self.accept()

    def __buttonNoPressed(self):
        self.__pressedButton = 1
        self.accept()

    def exec_(self):
        # adjust slider to min/max, if all slider values are set to 0 (means: cropping is called first time)

        tmpMin = [self.minimumX.getValue(), self.minimumY.getValue(), self.minimumZ.getValue()]
        tmpMax = [self.maximumX.getValue(), self.maximumY.getValue(), self.maximumZ.getValue()]

        if tmpMin == [0,0,0] and tmpMax == [0,0,0]:
            self.minimumX.setValue(self.minimumX.getRange()[0])
            self.minimumY.setValue(self.minimumY.getRange()[0])
            self.minimumZ.setValue(self.minimumZ.getRange()[0])

            self.maximumX.setValue(self.maximumX.getRange()[1])
            self.maximumY.setValue(self.maximumY.getRange()[1])
            self.maximumZ.setValue(self.maximumZ.getRange()[1])

        return CroppingManagerBase.exec_(self)
