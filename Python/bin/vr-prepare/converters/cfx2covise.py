from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject, pyqtSignal
import sys
import os
import os.path
from Cfx2CoviseGui import Cfx2CoviseGui
   
a = QtWidgets.QApplication(sys.argv)
w = Cfx2CoviseGui()
w.show()
sys.exit(a.exec_())
