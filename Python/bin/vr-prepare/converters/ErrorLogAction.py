from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject, pyqtSignal

from CoviseMsgLoop import CoviseMsgLoop, CoviseMsgLoopAction

ERROR_EVENT = QtCore.QEvent.Type(QtCore.QEvent.User+2)

class ErrorEvent(QtCore.QEvent):
    def __init__(self,error):
        QtCore.QEvent.__init__(self, ERROR_EVENT )
        self.error = error

class ErrorLogAction(CoviseMsgLoopAction):

    """Action to log covise-errors"""

    def __init__(self):
        CoviseMsgLoopAction.__init__(
            self,
            "ErrorLog",
            36,
            "36 is covise error message" )
        self.error="None"
        self.textBrowser="None"
        self.gui=None
        
    def register( self, gui ):
        #print "register"
        self.gui = gui
        
    def run(self, param):
        #print "ErrorLogAction.run"
        #global logFile
        
        #print("ERROR LOG ACTION")
        #logFile.write("%s: %s_%s@%s %s\n"%(ErrorLogAction.__name__,str(param[0]), str(param[1]), str(param[2]), str(param[3])))
        self.error=param[3]
        if self.gui:
            #print "post event"
            ev=ErrorEvent(self.error)
            QApplication.postEvent( self.gui, ev )
        
    def getError(self):
        return error
