# This file was created automatically by SWIG.
# Don't modify this file, modify the SWIG interface instead.
# This file is compatible with both classic and new-style classes.

import _covise

def _swig_setattr_nondynamic(self,class_type,name,value,static=1):
    if (name == "this"):
        if isinstance(value, class_type):
            self.__dict__[name] = value.this
            if hasattr(value,"thisown"): self.__dict__["thisown"] = value.thisown
            del value.thisown
            return
    method = class_type.__swig_setmethods__.get(name,None)
    if method: return method(self,value)
    if (not static) or hasattr(self,name) or (name == "thisown"):
        self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)

def _swig_setattr(self,class_type,name,value):
    return _swig_setattr_nondynamic(self,class_type,name,value,0)

def _swig_getattr(self,class_type,name):
    method = class_type.__swig_getmethods__.get(name,None)
    if method: return method(self)
    raise AttributeError,name

import types
try:
    _object = types.ObjectType
    _newclass = 1
except AttributeError:
    class _object : pass
    _newclass = 0
del types


class CoMsg(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, CoMsg, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, CoMsg, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ CoMsg instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    __swig_setmethods__["type"] = _covise.CoMsg_type_set
    __swig_getmethods__["type"] = _covise.CoMsg_type_get
    if _newclass:type = property(_covise.CoMsg_type_get, _covise.CoMsg_type_set)
    __swig_setmethods__["data"] = _covise.CoMsg_data_set
    __swig_getmethods__["data"] = _covise.CoMsg_data_get
    if _newclass:data = property(_covise.CoMsg_data_get, _covise.CoMsg_data_set)
    def __init__(self, *args):
        _swig_setattr(self, CoMsg, 'this', _covise.new_CoMsg(*args))
        _swig_setattr(self, CoMsg, 'thisown', 1)
    def __del__(self, destroy=_covise.delete_CoMsg):
        try:
            if self.thisown: destroy(self)
        except: pass

    def show(*args): return _covise.CoMsg_show(*args)
    def getType(*args): return _covise.CoMsg_getType(*args)

class CoMsgPtr(CoMsg):
    def __init__(self, this):
        _swig_setattr(self, CoMsg, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, CoMsg, 'thisown', 0)
        _swig_setattr(self, CoMsg,self.__class__,CoMsg)
_covise.CoMsg_swigregister(CoMsgPtr)


run_xuif = _covise.run_xuif

openMap = _covise.openMap

runMap = _covise.runMap

clean = _covise.clean

quit = _covise.quit

sendCtrlMsg = _covise.sendCtrlMsg

sendRendMsg = _covise.sendRendMsg

getSingleMsg = _covise.getSingleMsg

