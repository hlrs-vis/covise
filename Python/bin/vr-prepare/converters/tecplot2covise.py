from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject, pyqtSignal
import sys
import os
import os.path
from Tecplot2CoviseGui import Tecplot2CoviseGui
   
a = QtWidgets.QApplication(sys.argv)
w = Tecplot2CoviseGui()
w.show()
sys.exit(a.exec_())
