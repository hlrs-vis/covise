
# Part of the vr-prepare program for dc

# Copyright (c) 2006-2007 Visenso GmbH

from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import QObject, pyqtSignal

from KeydObject import RUN_ALL

# signals and their string represantation
INIT_SIGNAL        = 'sigInit'
DELETE_SIGNAL      = 'sigDelObj'
PARAM_SIGNAL       = 'sigParam'
BBOX_SIGNAL        = 'sigBbox'
KEYWORD_SIGNAL     = 'sigKeyWord'
SET_PARAMS_SIGNAL  = 'sigGuiParam'
SET_TEMP_PARAMS_SIGNAL  = 'sigTmpParam'
REQUEST_OBJ_SIGNAL = 'sigRequestObj'
REQUEST_DEL_OBJ_SIGNAL = 'sigRequestDelObj'
REQUEST_PARAMS_SIGNAL  = 'sigRequestParams'
RUN_OBJ_SIGNAL     = 'sigRunObj'
GUI_OK_SIGNAL      = 'sigGuiOk'
GUI_ERROR_SIGNAL   = 'sigGuiError'
GUI_EXIT_SIGNAL   = 'sigGuiExit'
NEG_OK_SIGNAL      = 'sigNegOk'
NEG_ERROR_SIGNAL   = 'sigNegError'
NEG_CHANGE_PATH_SIGNAL = 'sigChangePath'
LOAD_OBJECT_SIGNAL = 'sigLoadObject'
SAVE_OBJECT_SIGNAL = 'sigSaveObject'
REQUEST_DUPLICATE_OBJ_SIGNAL = 'sigDuplicateObj'
FINISH_LOADING_SIGNAL = 'sigFinishLoading'
IS_TRANSIENT_SIGNAL = 'sigIsTransient'
GUI_CHANGED_PATH_SIGNAL = 'sigGuiChangedPath'
RESTART_RENDERER_SIGNAL = 'sigGuiRestartRenderer'
SET_REDUCTION_FACTOR_SIGNAL = 'sigSetReductionFactor'
SET_CROP_MIN_MAX_SIGNAL = 'sigSetCropMinMax'
SET_SELECTION_STRING_SIGNAL = 'sigSetSelectionString'
NEG_ASK_TIMESTEP_REDUCTION = 'sigAskTimestepReduction'
GUI_ASKED_TIMESTEP_REDUCTION = 'sigGuiAskedTimestepReduction'
PRESENTATIONPOINTID_SIGNAL = 'sigSetPresentationPoint'
MOVEOBJ_SIGNAL = 'sigMoveObj'
VARIABLE_NOT_FOUND_SIGNAL = 'sigVarNotFound'
SELECT_SIGNAL = 'sigSelectObj'
DELETE_FROM_COVER_SIGNAL = 'sigDelFromCoverObj'

# no request nr existing
NO_REQ_NR = -1

# object key of the base session. Used as parentKey for top-level objects
SESSION_KEY = -1

# no parent key information is needed
DUMMY_PARENT = -2

class negGuiMsg(QtCore.QObject):
    """ base message class """
    sigInit = pyqtSignal()
    sigDelObj = pyqtSignal()
    def __init__(self, signal, requestNr, parentKey, qtParent=None):
        QtCore.QObject.__init__( self, qtParent)
        self.__signal    = signal
        self.__requestNr = requestNr
        self.__parentKey = parentKey

    def sendObj( self, obj ):
        self.__sendMsg( (self.__requestNr, obj) )

    def sendMsg( self, msg ):
        if self.__signal == 'sigInit':
            self.sigInit.emit( msg[0], msg[1] )
        if self.__signal == 'sigDelObj':
            self.sigDelObj.emit( msg[0], msg[1] )
        

    def getSignal( self ):
        return self.__signal

    def getParentKey( self ):
        return self.__parentKey

    def getRequestNr( self ):
        return self.__requestNr

class negGuiObjMsg(negGuiMsg ):
    """ message depending on an object: objectKey stored """
    def __init__(self, signal, requestNr, key, parentKey, qtParent=None):
        negGuiMsg.__init__(self, signal, requestNr, parentKey, qtParent)
        self.__key = key

    def getKey(self):
        return self.__key

class initMsg( negGuiObjMsg ):
    """ msg from negotiator to gui to init an object """
    def __init__(self, key, typeNr, requestNr, parentKey, params=None):
        negGuiObjMsg.__init__(self, INIT_SIGNAL, requestNr, key,  parentKey)
        self.typeNr = typeNr
        self.params = params

class deleteMsg( negGuiObjMsg ):
    """ msg from negotiator to gui to delete an object """
    def __init__(self, key, typeNr, requestNr, parentKey ):
        negGuiObjMsg.__init__(self, DELETE_SIGNAL, requestNr, key,  parentKey)
        self.typeNr = typeNr

class paramMsg( negGuiObjMsg ):
    """ msg from negotiator to gui to update params of an object """
    def __init__(self, key, params, requestNr ):
        negGuiObjMsg.__init__(self, PARAM_SIGNAL, requestNr, key, DUMMY_PARENT )
        self.params = params
        
class bboxMsg( negGuiObjMsg ):
    """ msg from negotiator to gui to update all bboxes of children from key """
    def __init__(self, key, bbox, requestNr ):
        negGuiObjMsg.__init__(self, BBOX_SIGNAL, requestNr, key, DUMMY_PARENT )
        self.bbox = bbox

class negOKMsg( negGuiMsg ):
    """ msg from negotiator to gui to send OK status """
    def __init__(self, requestNr ):
        negGuiMsg.__init__(self, NEG_OK_SIGNAL, requestNr, DUMMY_PARENT )

class negErrorMsg( negGuiMsg ):
    """ msg from negotiator to gui to send error status """
    def __init__(self, requestNr ):
        negGuiMsg.__init__(self, NEG_ERROR_SIGNAL, requestNr, DUMMY_PARENT )

class restartRendererMsg( negGuiMsg ):
    """ msg from negotiator to gui to request a restart of the renderer """
    def __init__(self, requestNr ):
        negGuiMsg.__init__(self, RESTART_RENDERER_SIGNAL, requestNr, DUMMY_PARENT )
        
class negChangePathMsg( negGuiMsg ):
    """ msg from negotiator to gui to request a new path from the user """
    def __init__(self, filename, addToCurrent, autoSync, requestNr, replaceInPathList, wrongFileName):
        negGuiMsg.__init__(self, NEG_CHANGE_PATH_SIGNAL, requestNr, DUMMY_PARENT)
        self.filename = filename
        self.wrongFileName = wrongFileName
        self.addToCurrent = addToCurrent
        self.replaceInPathList = replaceInPathList
        self.autoSync = autoSync

class negAskTimestepReductionMsg( negGuiMsg ):
    """ msg from negotiator to gui to request the reduction of timesteps from the user"""
    def __init__(self, oldMsg, requestNr):
         negGuiMsg.__init__(self, NEG_ASK_TIMESTEP_REDUCTION, requestNr, DUMMY_PARENT)
         self.oldMsg = oldMsg
        
class loadObjectMsg( negGuiMsg ):
    """ msg from gui to negotiator to load an object for file """
    def __init__(self, filename, addToCurrent, autoSync, requestNr, replaceInPathList ):
        negGuiMsg.__init__(self, LOAD_OBJECT_SIGNAL, requestNr, DUMMY_PARENT )
        self.filename = filename
        self.addToCurrent = addToCurrent
        self.replaceInPathList = replaceInPathList
        self.autoSync = autoSync

class saveObjectMsg( negGuiObjMsg ):
    """ msg from gui to negotiator to save an object for file """
    def __init__(self, key, filename, requestNr ):
        negGuiObjMsg.__init__(self, SAVE_OBJECT_SIGNAL, requestNr, key, DUMMY_PARENT )
        self.filename = filename

class requestObjMsg( negGuiObjMsg ):
    """ msg from gui to negotiator to request a new object """
    def __init__(self, requestNr, typeNr, parentKey, params =None):
        negGuiMsg.__init__(self, REQUEST_OBJ_SIGNAL, requestNr, parentKey)
        self.typeNr = typeNr
        self.params = params
    def send(self):
        self.sendObj( self )

class requestDelObjMsg( negGuiObjMsg ):
    """ msg from gui to negotiator to delete an object """
    def __init__(self, requestNr, key ):
        negGuiObjMsg.__init__(self, REQUEST_DEL_OBJ_SIGNAL, requestNr, key, DUMMY_PARENT )

class requestParamsMsg( negGuiObjMsg ):
    """ msg from gui to negotiator to request params of an object """
    def __init__(self, requestNr, objectKey ):
        negGuiObjMsg.__init__(self, REQUEST_PARAMS_SIGNAL, requestNr, objectKey, DUMMY_PARENT )

class runObjMsg( negGuiObjMsg ):
    """ msg from gui to negotiator to run an object """
    def __init__(self, requestNr, objectKey, runmode=RUN_ALL ):
        negGuiObjMsg.__init__(self, RUN_OBJ_SIGNAL, requestNr, objectKey, DUMMY_PARENT )
        self.runmode = runmode
        
class requestDuplicateObjMsg( negGuiObjMsg ):
    """msg from gui to negotiator to duplicate an object"""
    def __init__(self, requestNr, key, newName, typeNr, parentKey, params):
        negGuiObjMsg.__init__(self, REQUEST_DUPLICATE_OBJ_SIGNAL, requestNr, key, parentKey)
        self.typeNr = typeNr
        self.newName = newName
        self.params = params

class setReductionFactorMsg( negGuiMsg ):
    """ msg from gui to neg to set the time step reduction for the project """
    def __init__(self, requestNr, reductionFactor):
        negGuiMsg.__init__(self, SET_REDUCTION_FACTOR_SIGNAL, requestNr, DUMMY_PARENT)
        self.reductionFactor = reductionFactor

class setSelectionStringMsg( negGuiMsg ):
    """ msg from gui to neg to set the selection string for the GetSubset modules of all import managers """
    def __init__(self, requestNr, selectionString):
        negGuiMsg.__init__(self, SET_SELECTION_STRING_SIGNAL, requestNr, DUMMY_PARENT)
        self.selectionString = selectionString

class setCropMinMaxMsg( negGuiMsg ):
    """ msg from gui to neg to set the cropping settings for the project """
    def __init__(self, requestNr, minX, minY, minZ, maxX, maxY, maxZ):
        negGuiMsg.__init__(self, SET_CROP_MIN_MAX_SIGNAL, requestNr, DUMMY_PARENT)
        self.cropMin = [minX, minY, minZ]
        self.cropMax = [maxX, maxY, maxZ]

class setParamsMsg( negGuiObjMsg ):
    """ msg from gui to megotiator to update params of an object """
    def __init__(self, requestNr, objectKey, params ):
        negGuiObjMsg.__init__(self, SET_PARAMS_SIGNAL, requestNr, objectKey, DUMMY_PARENT )
        self.params = params

class setTempParamsMsg( negGuiObjMsg ):
    """ msg from gui to megotiator to update params of an object """
    def __init__(self, requestNr, objectKey, params ):
        negGuiObjMsg.__init__(self, SET_TEMP_PARAMS_SIGNAL, requestNr, objectKey, DUMMY_PARENT )
        self.params = params
        
class guiOKMsg( negGuiMsg ):
    """ msg from gui to negotiator to send OK status """
    def __init__(self, requestNr ):
        negGuiMsg.__init__(self, GUI_OK_SIGNAL, requestNr, DUMMY_PARENT )

class guiErrorMsg( negGuiMsg ):
    """ msg from negotiator to gui to send error status """
    def __init__(self, requestNr ):
        negGuiMsg.__init__(self, GUI_ERROR_SIGNAL, requestNr, DUMMY_PARENT )

class guiExitMsg( negGuiMsg ):
    """ msg from negotiator to gui to send exit status """
    def __init__(self, requestNr ):
        negGuiMsg.__init__(self, GUI_EXIT_SIGNAL, requestNr, DUMMY_PARENT )

class guiChangedPathMsg( negGuiMsg ):
    """ msg from gui to negotiator to inform about the new changed path from the user """
    def __init__(self, filename, addToCurrent, autoSync, requestNr, correctedPath, wrongFileName):
        negGuiMsg.__init__(self, GUI_CHANGED_PATH_SIGNAL, requestNr, DUMMY_PARENT)
        self.filename = filename
        self.wrongFileName = wrongFileName
        self.addToCurrent = addToCurrent
        self.correctedPath = correctedPath
        self.autoSync = autoSync
        
class guiAskedTimestepReductionMsg( negGuiMsg):
    """ msg from gui to negotiator to inform about the new changed path from the user """
    def __init__(self, reduction, oldMsg, requestNr):
        negGuiMsg.__init__(self, GUI_ASKED_TIMESTEP_REDUCTION, requestNr, DUMMY_PARENT)
        self.reduction = reduction
        self.oldMsg = oldMsg
        
class keyWordMsg( negGuiMsg ):
    def __init__(self, keyWord, requestNr ):
        negGuiMsg.__init__(self, KEYWORD_SIGNAL, requestNr, DUMMY_PARENT )
        self.keyWord = keyWord
        
class finishLoadingMsg( negGuiMsg ):
    def __init__(self, requestNr):
        negGuiMsg.__init__(self, FINISH_LOADING_SIGNAL, requestNr, DUMMY_PARENT )
        
class isTransientMsg( negGuiMsg ):
    """ msg from neg to gui to inform, if we have transient data so the main window will show appropriate panels """
    def __init__(self, requestNr, isTransient):
        negGuiMsg.__init__(self, IS_TRANSIENT_SIGNAL, requestNr, DUMMY_PARENT )
        self.isTransient = isTransient

class setPresentationPointMsg ( negGuiMsg ):
   def __init__(self, pid, requestNr):
        negGuiMsg.__init__(self, PRESENTATIONPOINTID_SIGNAL, requestNr, DUMMY_PARENT)
        self.pid = pid
        
class moveObjMsg( negGuiMsg ):
    def __init__(self, move, x, y, z, requestNr ):
        negGuiMsg.__init__(self, MOVEOBJ_SIGNAL, requestNr, DUMMY_PARENT )
        self.move = move
        self.x = x
        self.y = y
        self.z = z
        
class varNotFoundMsg( negGuiMsg ):
    def __init__(self, filename, regusetNr):
        negGuiMsg.__init__(self, VARIABLE_NOT_FOUND_SIGNAL, regusetNr, DUMMY_PARENT )
        self.wrongFile = filename
    
class selectMsg( negGuiObjMsg ):
    """ msg from negotiator to gui to delete an object """
    def __init__(self, key, select, requestNr ):
        negGuiObjMsg.__init__(self, SELECT_SIGNAL, requestNr, key,  DUMMY_PARENT)
        self.select = select 

class deleteFromCOVERMsg( negGuiObjMsg ):
    """ msg from negotiator to gui to delete an object """
    def __init__(self, key, requestNr ):
        negGuiObjMsg.__init__(self, DELETE_FROM_COVER_SIGNAL, requestNr, key,  DUMMY_PARENT)

# eof
