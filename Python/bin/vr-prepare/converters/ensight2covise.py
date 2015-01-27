from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject, pyqtSignal
import sys
import os
import os.path
from Ensight2CoviseGui import Ensight2CoviseGui
   
a = QtWidgets.QApplication(sys.argv)
w = Ensight2CoviseGui()
w.show()
sys.exit(a.exec_())
