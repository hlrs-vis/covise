
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH


import sys

import covise

from PyQt5 import QtCore, QtGui

from printing import InfoPrintCapable

from VisualizationPanel import VisualizationPanel
from negGuiHandlers import theGuiMsgHandler

from KeydObject import TYPE_2D_CUTGEOMETRY_PART, VIS_STREAMLINE_2D, VIS_MOVING_POINTS, VIS_PATHLINES, VIS_PLANE, VIS_VECTOR, VIS_ISOPLANE,  VIS_ISOCUTTER, VIS_CLIPINTERVAL, VIS_VECTORFIELD, VIS_MAGMATRACE, globalKeyHandler# VIS_POINTPROBING,
from KeydObject import TYPE_2D_COMPOSED_PART
from ObjectMgr import ObjectMgr
import Application
from ImportGroupManager import COMPOSED_VELOCITY


from VRPCoviseNetAccess import theNet
class Visualization2DPanel(VisualizationPanel):

    def __init__(self):
        VisualizationPanel.__init__(self)
        self.vrpLocalisationLabel.hide()
        self.vrpLabelLocalisation.hide()

        #dictionaries to enable the buttons
        self._enableDictComposedMode = {
                self.CuttingSurfaceColoredPushButton: False,
                self.IsoSurfacePushButton: False,
                self.IsoCutterPushButton: False,
                self.ClipIntervalPushButton: False,
                self.MovingPointsPushButton: False,
                self.PathlinesPushButton: False,
                self.ProbingPointPushButton: False, #True
                self.ProbingSquarePushButton: False,
                self.StreaklinesPushButton: False,
                self.Streamlines2DPushButton: False,
                self.SurfaceVectorsPushButton: False,
                self.MagmaTracePushButton: False,
                }
        self._enableDictVectorVariable = {
                self.CuttingSurfaceColoredPushButton: True,
                self.IsoSurfacePushButton: False,
                self.IsoCutterPushButton: False,
                self.ClipIntervalPushButton: False,
                self.MovingPointsPushButton: True,
                self.PathlinesPushButton: True,
                self.ProbingPointPushButton: False,
                self.ProbingSquarePushButton: False,
                self.StreaklinesPushButton: True,
                self.Streamlines2DPushButton: True,
                self.SurfaceVectorsPushButton: True,
                self.MagmaTracePushButton: False,
                }
        self._enableDictScalarVariable = {
                self.CuttingSurfaceColoredPushButton: True,
                self.IsoSurfacePushButton: True,
                self.IsoCutterPushButton: True,
                self.ClipIntervalPushButton: True,
                self.MovingPointsPushButton: False,
                self.PathlinesPushButton: False,
                self.ProbingPointPushButton: False, #True
                self.ProbingSquarePushButton: False, #True
                self.StreaklinesPushButton: False,
                self.Streamlines2DPushButton: False,
                self.SurfaceVectorsPushButton: False,
                self.MagmaTracePushButton: True,
                }
        #disabled buttons
        self._disablees = [
            self.CuttingSurfaceColoredPushButton,
            self.IsoSurfacePushButton,
            self.ProbingPointPushButton,
            self.ProbingSquarePushButton,
            self.MovingPointsPushButton,
            self.PathlinesPushButton,
            self.StreaklinesPushButton,
            ]
        # hide buttons
        if not covise.coConfigIsOn("CFDGui.VISUALIZER_MAGMATRACE"):
            self.MagmaTracePushButton.hide()
        # current key
        self.__key = None

        self._disableBrokenParts()

        
        self.Streamlines2DPushButton.clicked.connect(self.emitStreamline2DRequest)
        self.MovingPointsPushButton.clicked.connect(self.emitMovingPointsRequest)
        self.PathlinesPushButton.clicked.connect(self.emitPathlinesRequest)
        self.vrpComboBoxVariable.activated[str].connect(self._enableMethodButtsForVariableSlot)
        self.CuttingSurfaceColoredPushButton.clicked.connect(self.emitPlaneRequest)
        self.IsoSurfacePushButton.clicked.connect(self.emitIsoPlaneRequest)
        self.IsoCutterPushButton.clicked.connect(self.emitIsoCutterRequest)
        self.ClipIntervalPushButton.clicked.connect(self.emitClipIntervalRequest)
        self.SurfaceVectorsPushButton.clicked.connect(self.emitVectorFieldRequest)
        self.MagmaTracePushButton.clicked.connect(self.emitMagmaTraceRequest)

    def emitStreamline2DRequest(self):
        ObjectMgr().requestObjForVariable( VIS_STREAMLINE_2D, self.__key, self.currentVariable() )

    def emitMovingPointsRequest(self):
        ObjectMgr().requestObjForVariable( VIS_MOVING_POINTS, self.__key, self.currentVariable() )

    def emitPathlinesRequest(self):
        ObjectMgr().requestObjForVariable( VIS_PATHLINES, self.__key, self.currentVariable() )

    def emitPlaneRequest(self):
        ObjectMgr().requestObjForVariable( VIS_PLANE, self.__key, self.currentVariable() )

    def emitIsoPlaneRequest(self):
        ObjectMgr().requestObjForVariable( VIS_ISOPLANE, self.__key, self.currentVariable() )

    def emitIsoCutterRequest(self):
        ObjectMgr().requestObjForVariable( VIS_ISOCUTTER, self.__key, self.currentVariable() )

    def emitClipIntervalRequest(self):
        ObjectMgr().requestObjForVariable( VIS_CLIPINTERVAL, self.__key, self.currentVariable() )

    def emitVectorFieldRequest(self):
        ObjectMgr().requestObjForVariable( VIS_VECTORFIELD, self.__key, self.currentVariable() )

    #def emitPointProbingRequest(self):
    #    ObjectMgr().requestObjForVariable( VIS_POINTPROBING, self.__key, self.currentVariable() )

    def emitMagmaTraceRequest(self):
        ObjectMgr().requestObjForVariable( VIS_MAGMATRACE, self.__key, self.currentVariable() )

    def updateForObject( self, key):
        self.__key = key
        scalar = ObjectMgr().getPossibleScalarVariablesForType(self.__key)
        vector = ObjectMgr().getPossibleVectorVariablesForType(self.__key)
        if covise.coConfigIsOn("vr-prepare.UseComposedVelocity", False):
            myObject = globalKeyHandler().getObject(key)
            while (myObject.typeNr == TYPE_2D_CUTGEOMETRY_PART):
                myObject = globalKeyHandler().getObject(myObject.parentKey)
            if (myObject.typeNr == TYPE_2D_COMPOSED_PART):
                scalar.append(COMPOSED_VELOCITY)
        self._setScalarVariables(scalar)
        self._setVectorVariables(vector)
        self.setGridName(ObjectMgr().getNameOfType(self.__key))

# eof
