
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH

from PyQt5 import QtCore, QtGui

from printing import InfoPrintCapable

from KeyedTreeView import KeyedTreeView
from KeydObject import (
    VIS_3D_BOUNDING_BOX,
    VIS_STREAMLINE,
    VIS_MOVING_POINTS,
    VIS_PATHLINES,
    VIS_2D_RAW,
    VIS_COVISE,
    VIS_VRML,
    VIS_PLANE,
    VIS_ISOPLANE,
    VIS_ISOCUTTER,
    VIS_VECTOR,
    VIS_DOCUMENT,
    TYPE_PROJECT,
    TYPE_CASE,
    TYPE_2D_GROUP,
    TYPE_3D_GROUP,
    TYPE_2D_PART,
    TYPE_3D_PART,
    TYPE_3D_COMPOSED_PART,
    TYPE_2D_COMPOSED_PART,
    TYPE_COLOR_CREATOR,
    TYPE_COLOR_TABLE,
    TYPE_COLOR_MGR,
    TYPE_PRESENTATION,
    TYPE_PRESENTATION_STEP,
    TYPE_JOURNAL,
    TYPE_JOURNAL_STEP,
    TYPE_VIEWPOINT,
    TYPE_VIEWPOINT_MGR,
    TYPE_TRACKING_MGR,
    TYPE_CAD_PRODUCT,
    TYPE_CAD_PART,
    TYPE_SCENEGRAPH_MGR,
    TYPE_DNA_MGR,
    TYPE_GENERIC_OBJECT_MGR,
    TYPE_GENERIC_OBJECT
    )

import ObjectMgr, Application
from auxils import NamedCheckable
from KeyedTreeFilterProxy import KeyedTreeFilterProxy


typeNotInTree = [ TYPE_PROJECT, TYPE_PRESENTATION, TYPE_PRESENTATION_STEP, TYPE_JOURNAL, TYPE_JOURNAL_STEP, TYPE_CAD_PRODUCT, 
    TYPE_COLOR_CREATOR, TYPE_COLOR_TABLE, TYPE_COLOR_MGR, VIS_3D_BOUNDING_BOX, VIS_2D_RAW, TYPE_VIEWPOINT_MGR, 
    TYPE_VIEWPOINT, TYPE_SCENEGRAPH_MGR, TYPE_TRACKING_MGR, TYPE_DNA_MGR, TYPE_GENERIC_OBJECT_MGR ]
typeWithoutVisibleCheckbox = [ TYPE_PROJECT, TYPE_CASE, TYPE_2D_GROUP, TYPE_3D_GROUP, TYPE_3D_COMPOSED_PART, 
    TYPE_VIEWPOINT_MGR, TYPE_2D_COMPOSED_PART, VIS_VRML, TYPE_GENERIC_OBJECT ]

class KeyedTreeMgr(KeyedTreeView):

    """ Handling of KeyedTreeView panel 

    """

    def __init__(self, parent):
        KeyedTreeView.__init__(self, parent, KeyedTreeFilterProxy(parent), "TreeView")
        
        ObjectMgr.ObjectMgr().sigGuiObjectAdded.connect(self.addObject)
        ObjectMgr.ObjectMgr().sigGuiObjectDeleted.connect(self.deleteObject)
        ObjectMgr.ObjectMgr().sigGuiParamChanged.connect(self.paramChanged)

    def addObject( self, key, parentKey, typeNr ):

        # dont add already existing objects
        if KeyedTreeView.has_key(self, key):
            return False

        if typeNr in typeNotInTree:
            return False

        if KeyedTreeView.has_key(self, parentKey) :
            parent = parentKey
        else :
            parent = None
        if typeNr in typeWithoutVisibleCheckbox: 
            KeyedTreeView.insert(self, parent, key, 'newentry')
        else :
            checkable = NamedCheckable('default', False)
            KeyedTreeView.insert(self, parent, key, checkable)

    def deleteObject( self, key, parentKey, typeNr):
        if not typeNr in typeNotInTree:
            KeyedTreeView.delete(self, key)

    def paramChanged( self, key):
        if key in Application.vrpApp.visuKey2GuiKey:
            key = Application.vrpApp.visuKey2GuiKey[key]

        typeNr = ObjectMgr.ObjectMgr().getTypeOfObject(key)
        params = ObjectMgr.ObjectMgr().getParamsOfObject(key)
        realparams = ObjectMgr.ObjectMgr().getRealParamsOfObject(key)
        if not hasattr(realparams, 'name'):
            return

        if not typeNr in typeNotInTree:
            if typeNr in typeWithoutVisibleCheckbox or ( not hasattr( params, 'isVisible') ):
                KeyedTreeView.setItemData(self, key, realparams.name)
            else :
                checkable = NamedCheckable(realparams.name, params.isVisible)
                KeyedTreeView.setItemData(self, key, checkable)
# eof
