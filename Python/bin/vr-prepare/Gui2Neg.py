
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH

import time
import os
import os.path

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject, pyqtSignal

from printing import InfoPrintCapable
import auxils

from ErrorManager import (
    NO_ERROR,
    WRONG_PATH_ERROR)

from Neg2GuiMessages import (
    initMsg, paramMsg, SESSION_KEY,
    requestObjMsg, runObjMsg, requestParamsMsg, loadObjectMsg, saveObjectMsg, setReductionFactorMsg,
    setSelectionStringMsg, setCropMinMaxMsg, setParamsMsg, setTempParamsMsg, guiOKMsg, guiErrorMsg, guiExitMsg, guiChangedPathMsg, keyWordMsg,
    requestDuplicateObjMsg, finishLoadingMsg, restartRendererMsg, guiAskedTimestepReductionMsg, requestDelObjMsg, moveObjMsg )
from Utils import CopyParams
from KeydObject import RUN_ALL

transactionKeyHandler = auxils.KeyHandler()

_infoer = InfoPrintCapable()
_infoer.doPrint = False # True

class _gui2Neg(QtCore.QObject):
    """Send and receive message between gui and negotiator(gui side)."""
    sigRequestObj = pyqtSignal(int, requestObjMsg )
    sigRequestDelObj = pyqtSignal(int, requestDelObjMsg )
    sigRequestParams = pyqtSignal(int, requestParamsMsg )
    sigRunObj = pyqtSignal(int, runObjMsg )
    sigLoadObject = pyqtSignal(int, loadObjectMsg )
    sigSaveObject = pyqtSignal(int, saveObjectMsg )
    sigSetReductionFactor = pyqtSignal(int, setReductionFactorMsg )
    sigSetSelectionString = pyqtSignal(int, setSelectionStringMsg )
    sigSetCropMinMax = pyqtSignal(int, setCropMinMaxMsg )
    sigDuplicateObj = pyqtSignal(int, requestDuplicateObjMsg )
    sigGuiRestartRenderer = pyqtSignal(int, restartRendererMsg )
    sigGuiChangedPath = pyqtSignal(int, guiChangedPathMsg )
    sigGuiAskedTimestepReduction = pyqtSignal(int, guiAskedTimestepReductionMsg )
    sigGuiOk = pyqtSignal(int, guiOKMsg )
    sigGuiError = pyqtSignal(int, guiErrorMsg )
    sigGuiExit = pyqtSignal(int, guiExitMsg )
    sigGuiParam = pyqtSignal(int, setParamsMsg )
    sigTmpParam = pyqtSignal(int, setTempParamsMsg )
    sigKeyWord = pyqtSignal(int, keyWordMsg )
    sigMoveObj = pyqtSignal(int, moveObjMsg )
    
    def __init__(self):
        QtCore.QObject.__init__(self)
        self.__paramCalls = {}
        self.__addCalls   = {}
        self.__delCalls   = {}
        self.__errorCalls = {}
        self.__finishLoadingCall = None
        self.__reduceTimestepCallback = None
        self.__bboxCallsForChiildren = {}
        self.__keyWordCalls = {}

    def requestObject(self, typeNr, callback=None, parentKey=SESSION_KEY, params=None):
        _infoer.write("_gui2Neg.requestObject, typeNr: %s, parentKey: %s" % ( typeNr, SESSION_KEY))
        if callback==None:
            self.__appBusy()
            requestNr = transactionKeyHandler.registerObject(self.__appReady)
        else :
            requestNr = transactionKeyHandler.registerObject(callback)
        msg = requestObjMsg(requestNr, typeNr, parentKey, params)
        if(msg.getSignal() == 'sigRequestObj'):
            self.sigRequestObj.emit(requestNr, msg )
        else:
            print('wrong message type in requestObject')
        return requestNr

    def requestDelObject(self, objectKey, callback=None):
        _infoer.function = str(self.requestDelObject)
        _infoer.write("key: %s" % ( objectKey) )
        requestNr = transactionKeyHandler.registerObject(callback)
        msg = requestDelObjMsg( requestNr, objectKey )
        self.sigRequestDelObj.emit( requestNr, msg )
        return requestNr

    def requestParams(self, objectKey, callback=None):
        requestNr = transactionKeyHandler.registerObject(callback)
        msg = requestParamsMsg(requestNr, objectKey)
        self.sigRequestParams.emit(requestNr, msg)
        return requestNr

    def runObject( self, objectKey, runmode=RUN_ALL, callback=None ):
        if callback==None:
            self.__appBusy()
            requestNr = transactionKeyHandler.registerObject(self.__appReady)
        else :
            requestNr = transactionKeyHandler.registerObject(callback)
        msg = runObjMsg( requestNr, objectKey, runmode )
        self.sigRunObj.emit( requestNr, msg )
        return requestNr
   
    def loadObject( self, filename, addToCurrent, autoSync=False, callback=None, replaceInPathList=[], oldRequestNr=-1 ):
        # use the same requestNr from a previous call. needed, if loadObject() was previously called
        if oldRequestNr == -1:
            requestNr = transactionKeyHandler.registerObject(callback)
        else:
            requestNr = oldRequestNr
        msg = loadObjectMsg( filename, addToCurrent, autoSync, requestNr, replaceInPathList)
        self.sigLoadObject.emit(requestNr, msg )
        return requestNr
    
    def saveObject( self, objectKey, filename, callback=None ):
        _infoer.function = str(self.saveObject)
        _infoer.write("key: %s, filename: %s" % ( objectKey, filename ) )
        requestNr = transactionKeyHandler.registerObject(callback)
        msg = saveObjectMsg( objectKey, filename, requestNr )
        self.sigSaveObject.emit(requestNr, msg )
        return requestNr
    def setReductionFactor(self, rf, callback = None):
        requestNr = transactionKeyHandler.registerObject(callback)
        msg = setReductionFactorMsg(requestNr, rf)
        self.sigSetReductionFactor.emit(requestNr, msg)
        return requestNr
    
    def setSelectionString(self, selectionString, callback = None):
        requestNr = transactionKeyHandler.registerObject(callback)
        msg = setSelectionStringMsg(requestNr, selectionString)
        self.sigSetSelectionString.emit(requestNr, msg)
        return requestNr
    
    def setCropMinMax(self, minX, minY, minZ, maxX, maxY, maxZ, callback = None):
        requestNr = transactionKeyHandler.registerObject(callback)
        msg = setCropMinMaxMsg(requestNr, minX, minY, minZ, maxX, maxY, maxZ)
        self.sigSetCropMinMax.emit(requestNr, msg)
        return requestNr

    def requestDuplicateObject(self, key, newName, typeNr, callback=None, parentKey=SESSION_KEY, params=None):
        #if callback==None:
        #    self.__appBusy()
        #    requestNr = transactionKeyHandler.registerObject(self.__appReady)
        #else :
        requestNr = transactionKeyHandler.registerObject(callback)
        msg = requestDuplicateObjMsg(requestNr, key, newName, typeNr, parentKey, params)
        self.sigDuplicateObj.emit(requestNr, msg)
        return requestNr

    def restartRenderer(self, callback=None):
        requestNr = transactionKeyHandler.registerObject(callback)
        msg = restartRendererMsg(requestNr)
        self.sigGuiRestartRenderer.emit(requestNr, msg )
        return requestNr
      
    def registerAddCallback(self, key, addCallback):
        """addCallback gets called when a child of object key is created."""
        self.__addCalls[key] = addCallback

    def registerDelCallback(self, key, delCallback):
        """delCallback gets called when a child of object key is deleted."""
        self.__delCalls[key] = delCallback

    def registerParamCallback( self, key, paramCallback ):
        self.__paramCalls[key] = paramCallback

    def registerErrorCallback(self, key, errorCallback):
        """ errorCallback gets called when an error occurs in the negotiator """
        self.__errorCalls[key] = errorCallback
        
    def registerBBoxForChildrenCallback( self, key, bboxCallback):
        self.__bboxCallsForChiildren[key] = bboxCallback
        
    def registerKeyWordCallback(self, keyword, keywordCallback):
        self.__keyWordCalls[keyword] = keywordCallback
        
    def registerFinishLoadingCallback(self, loadingCallback):
        self.__finishLoadingCall = loadingCallback
        
    def registerIsTransientCallback(self, isTransientCallback):
        self.__isTransientCall = isTransientCallback
        
    def registerReduceTimestepCallback(self, reduceCallback):
        self.__reduceTimestepCallback = reduceCallback
        
    def _delRequest( self, requestNr, msg=None, status=NO_ERROR):
        # call request callback if available
        if transactionKeyHandler.hasKey(requestNr) and \
        transactionKeyHandler.getObject(requestNr):
            (transactionKeyHandler.getObject(requestNr))(requestNr, status, msg)
        if transactionKeyHandler.hasKey(requestNr):
            transactionKeyHandler.unregisterObject(requestNr)

    #

    def recvInit(self, requestNr, keyMsg):
        self._delRequest(requestNr, keyMsg)
        # call add callback if avaible
        if keyMsg.getParentKey() in self.__addCalls:
            self.__addCalls[keyMsg.getParentKey()](
                requestNr, keyMsg.typeNr, keyMsg.getKey(), keyMsg.getParentKey(), keyMsg.params)

    def recvDelete(self, requestNr, keyMsg):
        _infoer.function = str(self.recvDelete)
        _infoer.write("RequestNr: %s for object %s" % (requestNr, keyMsg.getKey()))
        self._delRequest(requestNr, keyMsg)
        # call add callback if avaible
        if keyMsg.getParentKey() in self.__delCalls:
            self.__delCalls[keyMsg.getParentKey()](
                requestNr, keyMsg.typeNr, keyMsg.getKey(), keyMsg.getParentKey())

    def recvParam(self, requestNr, pMsg):
        _infoer.function = str(self.recvParam)
        _infoer.write("RequestNr: %s for object %s" % (
            requestNr, pMsg.getKey()))
        self._delRequest(requestNr, pMsg)
        if pMsg.getKey() in self.__paramCalls:
            (self.__paramCalls[pMsg.getKey()])(
                requestNr, pMsg.getKey(), pMsg.params)

    def recvBbox(self, requestNr, pMsg):
        _infoer.function = str(self.recvBbox)
        _infoer.write("RequestNr: %s for object %s" % (
            requestNr, pMsg.getKey()))
        self._delRequest(requestNr, pMsg)
        if pMsg.getKey() in self.__bboxCallsForChiildren:
            (self.__bboxCallsForChiildren[pMsg.getKey()])(
                requestNr, pMsg.getKey(), pMsg.bbox) 
                
    def recvKeyWord(self, requestNr, pMsg):
        _infoer.function = str(self.recvKeyWord)
        self._delRequest(requestNr, pMsg)
        if pMsg.keyWord in self.__keyWordCalls:
            (self.__keyWordCalls[pMsg.keyWord])()
            
    def recvSetPresentationPointID(self, requestNr, pMsg):
        _infoer.function = str(self.recvSetPresentationPointID)
        self._delRequest(requestNr, pMsg)
        if 'PRESENTATION_SET_ID' in self.__keyWordCalls:
            (self.__keyWordCalls['PRESENTATION_SET_ID'])(pMsg.pid)
            
    def recvFinishLoading(self, requestNr, pMsg):
        _infoer.function = str(self.recvFinishLoading)
        self._delRequest(requestNr, pMsg)
        (self.__finishLoadingCall)()

    def recvIsTransient(self, requestNr, pMsg):
        _infoer.function = str(self.recvIsTransient)
        self._delRequest(requestNr, pMsg)
        (self.__isTransientCall)(pMsg.isTransient)

    def recvOk(self, requestNr):
        _infoer.function = str(self.recvOk)
        _infoer.write("gui2Neg::recvOK, RequestNr: %s" % requestNr)
        self._delRequest(requestNr)

    def recvError(self, requestNr):
        _infoer.function = str(self.recvError)
        _infoer.write("gui2Neg::recvError, RequestNr: %s" % requestNr)
        self._delRequest(requestNr)

    def recvChangePath(self, requestNr, pMsg):
        """ Requests a new path and sends it back to the negotiator """
        _infoer.function = str(self.recvChangePath)
        _infoer.write("gui2Neg::recvChangePath, RequestNr: %s" % requestNr)

        newPath = transactionKeyHandler.getObject(pMsg.getRequestNr())(pMsg.getRequestNr(), WRONG_PATH_ERROR, pMsg)

        # got path, now send it to negotiator
        msg = guiChangedPathMsg(pMsg.filename, pMsg.addToCurrent, pMsg.autoSync, pMsg.getRequestNr(), newPath, pMsg.wrongFileName)
        self.sigGuiChangedPath.emit(requestNr, msg)
        
    def recvVarNotFound(self, requestNr, pMsg):
        """ variable not found just raise dialog """
        text = "Did not find variable: "+ pMsg.wrongFile
        self.__errorCalls['okay'] (text)
        
    def recvAskTimestepReduction(self, requestNr, pMsg):
        """Request reduction of timesteps and sends it back to negotiator"""
        #if self.__reduceTimestepCallback == None:
            #print "ERRRROEEOEOEOEOEOEOEO"
        reduction = self.__reduceTimestepCallback(pMsg)
        
        msg = guiAskedTimestepReductionMsg(reduction, pMsg.oldMsg, pMsg.getRequestNr())
        self.sigGuiAskedTimestepReduction.emit(requestNr, msg)

    def recvSelect(self, requestNr, pMsg):
        #tell treeview which item to select
        if 'SET_SELECTION' in self.__keyWordCalls:
            (self.__keyWordCalls['SET_SELECTION'])(pMsg.getKey(), pMsg.select)        
            
    def recvDeleteFromCOVER(self, requestNr, pMsg):
        if 'DELETE_OBJECT' in self.__keyWordCalls:
            (self.__keyWordCalls['DELETE_OBJECT'])(pMsg.getKey())           

    def waitforAnswer(self, requestNr):
        num = 0
        while transactionKeyHandler.hasKey(requestNr) and num < 3:
            num = num + 1
            time.sleep(1.)
        if(num >= 3):
            print('Timeout in waitforAnswer')

    def answerOk(self, requestNr):
        msg = guiOKMsg(requestNr)
        self.sigGuiOk.emit(requestNr, msg)
        transactionKeyHandler.unregisterObject(requestNr)

    def answerError(self, requestNr):
        msg = guiErrorMsg(requestNr)
        self.sigGuiError.emit(requestNr, msg)
        transactionKeyHandler.unregisterObject(requestNr)

    def sendExit(self):
        msg = guiExitMsg(-1)
        self.sigGuiExit.emit(-1, msg)

    def setParams(self, objectKey, params, callback=None):
        """Set params for object with key objectKey."""            
        requestNr = transactionKeyHandler.registerObject(callback)
        _infoer.write( "setParams: reqNr: "+str(requestNr)+" objectKey: "+str(objectKey))
        msg = setParamsMsg(requestNr, objectKey, CopyParams(params))
        self.sigGuiParam.emit(requestNr, msg)
        return requestNr

    def setTempParams(self, objectKey, params, callback=None):
        """Set params for object with key objectKey."""            
        requestNr = transactionKeyHandler.registerObject(callback)
        msg = setTempParamsMsg(requestNr, objectKey, CopyParams(params))
        self.sigTmpParam.emit(requestNr, msg)
        return requestNr
        
    def sendKeyWord(self, keyWord, callback=None):
        requestNr = transactionKeyHandler.registerObject(callback)
        msg = keyWordMsg(keyWord, requestNr)
        self.sigKeyWord.emit(requestNr, msg)
        
    def sendMoveObj(self, move, x, y, z, callback=None):
        requestNr = transactionKeyHandler.registerObject(callback)
        msg = moveObjMsg(move, x, y, z, requestNr)
        self.sigMoveObj.emit(msg.getSignal(), requestNr, msg)        
        
    def __appBusy(self):
        QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))
        
    def __appReady(self, requestNr, status, key ):
        QtWidgets.QApplication.restoreOverrideCursor()
        
_theGuiMsgHandler = None
def theGuiMsgHandler():

    """Access the singleton."""

    global _theGuiMsgHandler
    if None == _theGuiMsgHandler: _theGuiMsgHandler = _gui2Neg()
    return _theGuiMsgHandler

# eof
