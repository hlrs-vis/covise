
# Part of the vr-prepare program for dc

# Copyright (c) 2006-2007 Visenso GmbH

import pdb
import sys

from qt import *

import auxils

from KeydObject import (
    globalKeyHandler,
    TYPE_PROJECT,
    TYPE_CASE,
    TYPE_2D_GROUP,
    TYPE_3D_GROUP,
    TYPE_2D_PART,
    TYPE_3D_PART)

from KeyedTreeView import KeyedTreeView
from Neg2GuiMessages import SESSION_KEY
from coCaseMgr import coCaseMgrParams
from negGuiHandlers import (theGuiMsgHandler, initHandlers)
from coviseCase import (coviseCase2DimensionSeperatedCase, NameAndCoviseCase )

def childInitCB(requestNr, typeNr, key, parentKey):
    theGuiMsgHandler().registerAddCallback(key, childInitCB)
    if TYPE_PROJECT == typeNr:
        global project_key
        project_key = key
        pass
    elif TYPE_CASE == typeNr:
        global case_key
        case_key = key
        theGuiMsgHandler().registerParamCallback(key, setCaseParams)
        global_tree.insert(None, key, 'casefile-entry')
    elif TYPE_2D_GROUP == typeNr:
        theGuiMsgHandler().registerParamCallback(key, set2d_groupParams)
        global_tree.insert(parentKey, key, 'sub-casefile-entry')
    elif TYPE_3D_GROUP == typeNr:
        theGuiMsgHandler().registerParamCallback(key, set3d_groupParams)
        global_tree.insert(parentKey, key, 'sub-casefile-entry')
    elif TYPE_2D_PART == typeNr:
        theGuiMsgHandler().registerParamCallback(key, set2d_partParams)
        global_tree.insert(parentKey, key, 'sub-casefile-entry')
        # TODO  insert check box
    elif TYPE_3D_PART == typeNr:
        theGuiMsgHandler().registerParamCallback(key, set3d_partParams)
        global_tree.insert(parentKey, key, 'sub-casefile-entry')
        # TODO  insert check box
    else:
##         assert False, 'unknown type'
        print('HERE IS AN UNHANDLED OR UNKNOW TYPE.  (AND THIS IS IGNORED FOR NOW.)')
    theGuiMsgHandler().answerOk(requestNr)

def setNameForObjectInTree(key, aName):
    global_tree.setItemData(key, aName)

def setCaseParams(requestNr, key, params):
    setNameForObjectInTree(key, params.name)
    theGuiMsgHandler().answerOk(requestNr)

def set2d_groupParams(requestNr, key, params):
    global_tree.setItemData(key, params.name)
    theGuiMsgHandler().answerOk(requestNr)

def set3d_groupParams(requestNr, key, params):
    global_tree.setItemData(key, params.name)
    theGuiMsgHandler().answerOk(requestNr)

def set2d_partParams(requestNr, key, params):
    global_tree.setItemData(key, params.name)
    theGuiMsgHandler().answerOk(requestNr)

def set3d_partParams(requestNr, key, params):
    global_tree.setItemData(key, params.name)
    theGuiMsgHandler().answerOk(requestNr)


def testGuiForObjectTree():
    keepReference = QApplication(sys.argv)

    global global_tree
    global_tree = KeyedTreeView()

    initHandlers()

    g = theGuiMsgHandler()

    g.registerAddCallback(SESSION_KEY, childInitCB)
    reqId = g.requestObject(
        typeNr = TYPE_PROJECT, callback = None, parentKey = SESSION_KEY)
    g.waitforAnswer(reqId)

    reqId = g.requestObject(
        typeNr = TYPE_CASE, callback = None, parentKey = project_key)
    g.waitforAnswer(reqId)

    caseP = coCaseMgrParams()
    nameAndCase = NameAndCoviseCase()
    nameAndCase.setFromFile('/work/common/Projekte/DC-CFDGui/datasets/'
                            'TINY/CoviseDaten/TINY.cocase')
    caseP.dsc = coviseCase2DimensionSeperatedCase(
        nameAndCase.case, nameAndCase.name)
    g.setParams(case_key, caseP)

    qApp.setMainWidget(global_tree)
    global_tree.show()
    qApp.exec_loop()


if __name__ == "__main__":
##    pdb.run('testGuiForObjectTree()')
    testGuiForObjectTree()

# eof
