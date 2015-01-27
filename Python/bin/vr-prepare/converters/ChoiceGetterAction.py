from paramAction import NotifyAction
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject, pyqtSignal

NOTIFIER_EVENT = QtCore.QEvent.Type(QtCore.QEvent.User+1)

class NotifierEvent(QtCore.QEvent):
    def __init__(self,param,value):
        QtCore.QEvent.__init__(self, NOTIFIER_EVENT )
        self.param = param
        self.value = value

class ChoiceGetterAction(NotifyAction):

    def __init__(self):
        self.readingDone_ = False
        self.gui = None

    def register( self, gui ):
        self.gui = gui
        
    def run(self):
        self.readingDone_ = True
        if self.gui:
            ev=NotifierEvent(self.param_, self.value_)
            QtWidgets.QApplication.postEvent( self.gui, ev )

    def waitForChoices(self):
        while not self.readingDone_: pass

    def resetWait(self):
        self.readingDone_ = False

    def getChoices(self):
        # we expect the variable list looks like this ["1", "None", "temperature", "pressure"]
        #print "ChoiceGetterAction.getChoices"
        #print self.value_
        #assert 2 < len(self.value_)
        return self.value_[2:] # extract variables ommit 0 (actual choice), 1 (None), start with 2

    def getAllChoices(self):
        # we expect the variable list looks like this ["1", "None", "temperature", "pressure"]
        #print "ChoiceGetterAction.getChoices"
        #print self.value_
        #assert 2 < len(self.value_)
        return self.value_[1:] # extract variables ommit 0 (actual choice), start with 1 (None)
