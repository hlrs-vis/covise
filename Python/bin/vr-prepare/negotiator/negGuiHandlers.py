
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH

from PyQt5 import QtCore, QtGui

# import singleton class instances
from Gui2Neg import theGuiMsgHandler
from Neg2Gui import theNegMsgHandler
from Neg2GuiMessages import (
    INIT_SIGNAL, DELETE_SIGNAL, PARAM_SIGNAL, BBOX_SIGNAL, KEYWORD_SIGNAL, NEG_OK_SIGNAL, NEG_ERROR_SIGNAL, NEG_CHANGE_PATH_SIGNAL,
    REQUEST_OBJ_SIGNAL, REQUEST_DEL_OBJ_SIGNAL, RUN_OBJ_SIGNAL, REQUEST_PARAMS_SIGNAL, SET_PARAMS_SIGNAL, SET_TEMP_PARAMS_SIGNAL, SET_REDUCTION_FACTOR_SIGNAL, SET_CROP_MIN_MAX_SIGNAL,
    SET_SELECTION_STRING_SIGNAL, GUI_OK_SIGNAL, GUI_ERROR_SIGNAL, GUI_EXIT_SIGNAL, GUI_CHANGED_PATH_SIGNAL, LOAD_OBJECT_SIGNAL, SAVE_OBJECT_SIGNAL, KEYWORD_SIGNAL,
    REQUEST_DUPLICATE_OBJ_SIGNAL, FINISH_LOADING_SIGNAL, IS_TRANSIENT_SIGNAL, RESTART_RENDERER_SIGNAL, NEG_ASK_TIMESTEP_REDUCTION, GUI_ASKED_TIMESTEP_REDUCTION, PRESENTATIONPOINTID_SIGNAL, MOVEOBJ_SIGNAL, VARIABLE_NOT_FOUND_SIGNAL, SELECT_SIGNAL, DELETE_FROM_COVER_SIGNAL )


def initHandlers(coverWidgetId=None):
    """Connect the theGuiMsgHandler and theNegMsgHandler."""

    g = theGuiMsgHandler()
    
    
    n = theNegMsgHandler(coverWidgetId)
    
    n.sigInit.connect(g.recvInit)
    n.sigDelObj.connect(g.recvDelete)
    n.sigParam.connect(g.recvParam)
    n.sigNegOk.connect(g.recvOk)
    n.sigNegError.connect(g.recvError)
    n.sigChangePath.connect(g.recvChangePath)
    n.sigBbox.connect(g.recvBbox)
    n.sigKeyWord.connect(g.recvKeyWord)
    n.sigFinishLoading.connect(g.recvFinishLoading)
    n.sigIsTransient.connect(g.recvIsTransient)
    n.sigAskTimestepReduction.connect(g.recvAskTimestepReduction)
    n.sigSetPresentationPoint.connect(g.recvSetPresentationPointID)
    n.sigVarNotFound.connect(g.recvVarNotFound)
    n.sigSelectObj.connect(g.recvSelect)
    n.sigDelFromCoverObj.connect(g.recvDeleteFromCOVER)
    g.sigRequestObj.connect(n.requestObject)
    g.sigRequestDelObj.connect(n.requestDelObject)
    g.sigDuplicateObj.connect(n.requestDuplicateObject)
    g.sigRunObj.connect(n.runObject)
    g.sigLoadObject.connect(n.loadObject)
    g.sigSaveObject.connect(n.saveObject)
    g.sigRequestParams.connect(n.requestParams)
    g.sigGuiParam.connect(n.recvParams)
    g.sigTmpParam.connect(n.recvTempParams)
    g.sigGuiError.connect(n.recvError)
    g.sigGuiExit.connect(n.recvExit)
    g.sigKeyWord.connect(n.keyWord)
    g.sigGuiChangedPath.connect(n.recvChangedPath)
    g.sigGuiRestartRenderer.connect(n.restartRenderer)
    g.sigSetReductionFactor.connect(n.recvReductionFactor)
    g.sigSetCropMinMax.connect(n.recvCropMinMax)
    g.sigSetSelectionString.connect(n.recvSelectionString)
    g.sigGuiAskedTimestepReduction.connect(n.recvAskedReductionFactor)
    g.sigMoveObj.connect(n.moveObj)
# eof
