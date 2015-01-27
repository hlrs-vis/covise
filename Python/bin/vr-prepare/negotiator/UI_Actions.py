
# Part of the vr-prepare program for dc

# Moduleright (c) 2006 Visenso GmbH


from CoviseMsgLoop import *
from paramAction import NotifyAction
from KeydObject import RUN_ALL
from printing import InfoPrintCapable
_infoer = InfoPrintCapable()
_infoer.doPrint = False # True

from PyQt5 import QtCore, QtGui, QtWidgets

# QtCore.QEvent.User == 1000
COPY_EVENT = QtCore.QEvent.Type(QtCore.QEvent.User+1)
DELETE_EVENT = QtCore.QEvent.Type(QtCore.QEvent.User+2)
NOTIFIER_EVENT = QtCore.QEvent.Type(QtCore.QEvent.User+3)
GRMSG_EVENT = QtCore.QEvent.Type(QtCore.QEvent.User+4)
RENDERER_CRASH_EVENT = QtCore.QEvent.Type(QtCore.QEvent.User+5)
EXEC_EVENT = QtCore.QEvent.Type(QtCore.QEvent.User+6)
VRC_EVENT = QtCore.QEvent.Type(QtCore.QEvent.User+7)

class CopyEvent(QtCore.QEvent):
    def __init__(self,key):
        QtCore.QEvent.__init__(self, COPY_EVENT )
        self.key = key

class ExecEvent(QtCore.QEvent):
    def __init__(self,key):
        QtCore.QEvent.__init__(self, EXEC_EVENT )
        self.key = key
        
class DeleteEvent(QtCore.QEvent):
    def __init__(self,key):
        QtCore.QEvent.__init__(self, DELETE_EVENT )
        self.key = key

class NotifierEvent(QtCore.QEvent):
    def __init__(self,key,param,value):
        QtCore.QEvent.__init__(self, NOTIFIER_EVENT )
        self.key = key
        self.param = param
        self.value = value

class GuiRenderMsgEvent(QtCore.QEvent):
    def __init__(self, msgstring):
        QtCore.QEvent.__init__(self, GRMSG_EVENT )
        self.msgstring = msgstring

class RendererCrashEvent(QtCore.QEvent):
    def __init__(self):
        QtCore.QEvent.__init__(self, RENDERER_CRASH_EVENT )
        
class VRCEvent(QtCore.QEvent):
    def __init__(self,param):
        QtCore.QEvent.__init__(self, VRC_EVENT )
        self.param = param    
        


class _StartMessageAction(CoviseMsgLoopAction):

    def __init__(self):
        CoviseMsgLoopAction.__init__(
            self,
            "start",
            35,
            "start module" )
        self.module2Content = {}

        self.__enabled = True

    def register( self, module, obj, negMsgHandler ):
        _infoer.function = str(self.register)
        _infoer.write("module %s,%s,%s " % (module.name_, str(module.nr_), module.host_) )

        if obj==0:
            self.module2Content[ (module.name_, str(module.nr_), module.host_) ] = ( 0, negMsgHandler )
        else:
            self.module2Content[ (module.name_, str(module.nr_), module.host_) ] = ( obj.key, negMsgHandler )

    def run(self, params):
        _infoer.function = str(self.run)
        _infoer.write("params %s" % params[0] )

        if self.__enabled:
            module = (params[0], params[1], params[2])  # params = (module name, module instance, host)
            if module in self.module2Content:
                objKey =  self.module2Content[module][0]
                negMsgHandler = self.module2Content[module][1]
                ev=ExecEvent(objKey)
                QtWidgets.QApplication.postEvent( negMsgHandler, ev )

    def setEnabled(self, b):
        self.__enabled = b
                
class _ModuleMessageAction(CoviseMsgLoopAction):

    def __init__(self):
        CoviseMsgLoopAction.__init__(
            self,
            "UI Module",
            6,
            "UI Module Filter" )
        self.module2Content = {}

    def register( self, module, obj, negMsgHandler ):
        _infoer.function = str(self.register)
        _infoer.write("module %s,%s,%s " % (module.name_, str(module.nr_), module.host_) )
        if obj==0:
            self.module2Content[ (module.name_, str(module.nr_), module.host_) ] = ( 0, negMsgHandler )
        else:
            self.module2Content[ (module.name_, str(module.nr_), module.host_) ] = ( obj.key, negMsgHandler )

    def run(self, params):
        _infoer.function = str(self.run)
        _infoer.write("params %s" % params[0] )

        if 'COPY_MODULE_EXEC'==params[0]:
            module = (params[1], params[2], params[3])
            if module in self.module2Content:
                objKey =  self.module2Content[module][0]
                negMsgHandler = self.module2Content[module][1]
                ev=CopyEvent(objKey)
                QtWidgets.QApplication.postEvent( negMsgHandler, ev )

        if 'DELETE_MODULE'==params[0]:
            module = (params[1], params[2], params[3])
            if module in self.module2Content:
                objKey =  self.module2Content[module][0]
                negMsgHandler = self.module2Content[module][1]
                ev=DeleteEvent(objKey)
                QtWidgets.QApplication.postEvent( negMsgHandler, ev )

        if 'DIED'==params[0]:
            module = (params[1], params[2], params[3])
            if module in self.module2Content:
                if self.module2Content[module][0]==0:
                    negMsgHandler = self.module2Content[module][1]
                    ev=RendererCrashEvent()
                    QtWidgets.QApplication.postEvent( negMsgHandler, ev )
        else: raise NameError
_moduleMsgHandler = None
def theModuleMsgHandler():
    """Assert instance and access to the copy-message-handler."""

    global _moduleMsgHandler
    if None == _moduleMsgHandler:
        _moduleMsgHandler = _ModuleMessageAction()
        CoviseMsgLoop().register( _moduleMsgHandler )
    return _moduleMsgHandler

_startMsgHandler = None
def theStartMsgHandler():
    """Assert instance and access to the message-handler."""

    global _startMsgHandler
    if None == _startMsgHandler:
        _startMsgHandler = _StartMessageAction()
        CoviseMsgLoop().register( _startMsgHandler )
    return _startMsgHandler


class ModuleNotifier(NotifyAction):
    def __init__(self, key):
        self.key = key
        self.negMsgHandler = None

    def register( self, negMsgHandler ):
        self.negMsgHandler = negMsgHandler

    def run(self):
        _infoer.function = str(self.run)
        _infoer.write("** Key: %s StartPositionNotifier param: %s\n **  -> Coordinates: %s" % ( self.key, self.param_,self.value_)  )
        if self.negMsgHandler:
            ev=NotifierEvent(self.key, self.param_, self.value_)
            QtWidgets.QApplication.postEvent( self.negMsgHandler, ev )

_moduleNotifier = {}
def theModuleNotifier(key):
    """Assert instance and access to the module notifier."""
    global _moduleNotifier
    if not key in _moduleNotifier:
        _moduleNotifier[key] = ModuleNotifier(key)
    return _moduleNotifier[key]



class _GuiRenderMessageAction(CoviseMsgLoopAction):
    def __init__(self):
        CoviseMsgLoopAction.__init__(
            self,
            "UI",
            6,
            "UI Msg Filter" )
        self.negMsgHandler = None

    def register( self, negMsgHandler ):
        self.negMsgHandler = negMsgHandler

    def run(self, params):
        _infoer.function = str(self.run)
        _infoer.write("UI msg : %s" % params )

        if params[0]!="GRMSG" : raise NameError
        msgString = "\n".join(params)
        if self.negMsgHandler:
            ev=GuiRenderMsgEvent(msgString)
            QtWidgets.QApplication.postEvent( self.negMsgHandler, ev )

_grMsgAction = None
def theGrMsgAction():
    """Assert instance and access to the gui-render-message-Action."""
    global _grMsgAction
    if None == _grMsgAction:
        _grMsgAction = _GuiRenderMessageAction()
        CoviseMsgLoop().register( _grMsgAction )
    return _grMsgAction


class _VRCEventAction(CoviseMsgLoopAction):
    """Action to log covise-errors"""

    def __init__(self):
        CoviseMsgLoopAction.__init__(
            self, 
            "VRC Log", 
            1000, 
            "1000 is covise error message" )
        self.negMsgHandler = None
        
    def register( self, negMsgHandler ):
        self.negMsgHandler = negMsgHandler     
            
    def run(self, param):
        _infoer.function = str(self.run)
        _infoer.write("%s, %s, %s, %s, %s " %  ( str(param[0]), str(param[1]), str(param[2]), str(param[3]), str(param[4]) ) )
        ev=VRCEvent(param)
        QtWidgets.QApplication.postEvent( self.negMsgHandler, ev )        

_vrcAction = None
def theVRCAction():
    """Assert instance and access to the gui-render-message-Action."""
    global _vrcAction
    if None == _vrcAction:
        _vrcAction = _VRCEventAction()
        CoviseMsgLoop().register( _vrcAction )
    return _vrcAction

# eof
