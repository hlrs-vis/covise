
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH


import sys

from PyQt5 import QtCore, QtGui, QtWidgets

from GridVisualizationPanelBase import Ui_GridVisualizationPanelBase
from VisualizationPanel import VisualizationPanel
from GridVisualizationPanelConnector import GridVisualizationPanelConnector
from TransformManager import TransformManager

from Part3DBoundingBoxVis import Part3DBoundingBoxVisParams
from Gui2Neg import theGuiMsgHandler

from KeydObject import VIS_STREAMLINE, VIS_MOVING_POINTS, VIS_PATHLINES, VIS_PLANE, VIS_VECTOR, VIS_ISOPLANE, \
        TYPE_3D_PART, TYPE_3D_COMPOSED_PART, VIS_DOMAINLINES, VIS_DOMAINSURFACE, globalKeyHandler #VIS_POINTPROBING,

from ObjectMgr import ObjectMgr
import Application
from Utils import ParamsDiff
from ImportGroupManager import COMPOSED_VELOCITY
import covise

from printing import InfoPrintCapable

class GridVisualizationPanel(QtWidgets.QWidget,Ui_GridVisualizationPanelBase, VisualizationPanel, TransformManager):

    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        Ui_GridVisualizationPanelBase.__init__(self)
        self.setupUi(self)
        VisualizationPanel.__init__(self)
        TransformManager.__init__(self, self.emitDataChanged)

        #dictionaries to enable the buttons
        self._enableDictComposedMode = {
                self.CuttingSurfaceArrowPushButton: False,
                self.CuttingSurfaceColoredPushButton: False,
                self.IsoSurfacePushButton: False,
                self.MovingPointsPushButton: True,
                self.PathlinesPushButton: True,
                self.ProbingPointPushButton: False,
                self.ProbingSquarePushButton: False,
                self.StreaklinesPushButton: False,
                self.StreamlinesPushButton: True,
                self.DomainLinesPushButton: False,
                self.DomainSurfacePushButton: False,
                }
        self._enableDictVectorVariable = {
                self.CuttingSurfaceArrowPushButton: True,
                self.CuttingSurfaceColoredPushButton: True,
                self.IsoSurfacePushButton: False,
                self.MovingPointsPushButton: True,
                self.PathlinesPushButton: True,
                self.ProbingPointPushButton: False,
                self.ProbingSquarePushButton: False,
                self.StreaklinesPushButton: True,
                self.StreamlinesPushButton: True,
                self.DomainLinesPushButton: False,
                self.DomainSurfacePushButton: False,
                }
        self._enableDictScalarVariable = {
                self.CuttingSurfaceArrowPushButton: False,
                self.CuttingSurfaceColoredPushButton: True,
                self.IsoSurfacePushButton: True,
                self.MovingPointsPushButton: False,
                self.PathlinesPushButton: False,
                self.ProbingPointPushButton: False, #True,
                self.ProbingSquarePushButton: False, #True,
                self.StreaklinesPushButton: False,
                self.StreamlinesPushButton: False,
                self.DomainLinesPushButton: False,
                self.DomainSurfacePushButton: True,
                }
        self._enableDictUnsetVariable = {
                self.CuttingSurfaceArrowPushButton: False,
                self.CuttingSurfaceColoredPushButton: False,
                self.IsoSurfacePushButton: False,
                self.MovingPointsPushButton: False,
                self.PathlinesPushButton: False,
                self.ProbingPointPushButton: False, #True,
                self.ProbingSquarePushButton: False, #True,
                self.StreaklinesPushButton: False,
                self.StreamlinesPushButton: False,
                self.DomainLinesPushButton: True,
                self.DomainSurfacePushButton: True,
                }

        #disabled buttons
        self._disablees = [
            self.ProbingPointPushButton,
            self.ProbingSquarePushButton,
            self.StreaklinesPushButton,
            self.DescriptionCheckBox,
            ]
        # list of associated keys
        self.__keys = []
        self.__visible = False

        self.__inFixedGridMode = True
        self._disableBrokenParts()
        # temporary restriction for composed grids

        self.tabWidget.setCurrentIndex(0) # ignore index set by the designer (usually no one cares about the active index when editing ui-files)

        GridVisualizationPanelConnector(self)

    def emitStreamlineRequest(self):
        ObjectMgr().requestObjForVariable( VIS_STREAMLINE, self.__keys[0], self.currentVariable())

    def emitMovingPointsRequest(self):
        ObjectMgr().requestObjForVariable( VIS_MOVING_POINTS, self.__keys[0], self.currentVariable())

    def emitPathlinesRequest(self):
        ObjectMgr().requestObjForVariable( VIS_PATHLINES, self.__keys[0], self.currentVariable())

    def emitPlaneRequest(self):
        ObjectMgr().requestObjForVariable( VIS_PLANE, self.__keys[0], self.currentVariable())

    def emitArrowsRequest(self):
        ObjectMgr().requestObjForVariable( VIS_VECTOR, self.__keys[0], self.currentVariable())

    def emitIsoPlaneRequest(self):
        ObjectMgr().requestObjForVariable( VIS_ISOPLANE, self.__keys[0], self.currentVariable())

    #def emitPointProbingRequest(self):
    #    ObjectMgr().requestObjForVariable( VIS_POINTPROBING, self.__keys[0], self.currentVariable())

    def emitDomainLinesRequest(self):
        ObjectMgr().requestObjForVariable( VIS_DOMAINLINES, self.__keys[0], self.currentVariable())  

    def emitDomainSurfaceRequest(self):
       ObjectMgr().requestObjForVariable( VIS_DOMAINSURFACE, self.__keys[0], self.currentVariable())  
        

    def paramChanged(self, key):
        """ params of object key changed"""
        if key in self.__keys :
            self.update()
        for k in self.__keys :
            if k in Application.vrpApp.guiKey2visuKey and key == Application.vrpApp.guiKey2visuKey[k]:
                self.update()

    def update(self):
        if len(self.__keys)!=0 :
            self.updateForObject( self.__keys )

    def updateForObject( self, keys ):
        if isinstance( keys, int):
            self.__keys = [keys]
        else:
            self.__keys = keys

        # set the variables
        if len(self.__keys)==1 :
            # enable the visualization and the transform tab
            self.tabWidget.setTabEnabled(0, True)
            self.tabWidget.setTabEnabled(1, True)

            scalar = ObjectMgr().getPossibleScalarVariablesForType(self.__keys[0])
            vector = ObjectMgr().getPossibleVectorVariablesForType(self.__keys[0])
            if covise.coConfigIsOn("vr-prepare.UseComposedVelocity", False):
                myObject = globalKeyHandler().getObject(self.__keys[0])
                if (myObject.typeNr == TYPE_3D_COMPOSED_PART):
                    vector.append(COMPOSED_VELOCITY)
            self._setUnsetVariable(True)        
            self._setScalarVariables(scalar)
            self._setVectorVariables(vector)
            self.setGridName(ObjectMgr().getNameOfType(self.__keys[0]) )
        else : # multi selection
            # disable the visualization and the transform tab
            self.tabWidget.setTabEnabled(0, False)
            self.tabWidget.setTabEnabled(1, False)
        # apply params
        params = ObjectMgr().getParamsOfObject(self.__keys[0])
        params.name = ObjectMgr().getNameOfType(self.__keys[0]) 
        #if self.__key in Application.vrpApp.guiKey2visuKey:
        #    params.isVisible = ObjectMgr().getParamsOfObject(Application.vrpApp.guiKey2visuKey[self.__key]).isVisible
        self.__setParams( params )

    def __getParams(self):
        data = Part3DBoundingBoxVisParams()
        if not ObjectMgr().getTypeOfObject(self.__keys[0]) == TYPE_3D_COMPOSED_PART:
            #transform card
            self.TransformManagerGetParams(data)

        data.isVisible = self.__visible
        return data


    def __setParams( self, params ):

        if isinstance( params, int):
            self.__keys[0] = params
            return

        if not ObjectMgr().getTypeOfObject(self.__keys[0]) == TYPE_3D_COMPOSED_PART:
            self.TransformManagerBlockSignals(True)
            self.TransformManagerSetParams(params)
            if hasattr(params, 'isVisible'):
                self.__visible = params.isVisible
            self.TransformManagerBlockSignals(False)

    # any data has changed
    def emitDataChanged(self): 
        if not len(self.__keys)==0 and not ObjectMgr().getTypeOfObject(self.__keys[0]) == TYPE_3D_COMPOSED_PART:
            params = self.__getParams()

            # mapping of the keys for the object manager
            childKeys = []
            for i in range(0, len(self.__keys)):
                childKeys.append(Application.vrpApp.guiKey2visuKey[self.__keys[i]])
            # set params for first object
            if len(self.__keys)>0 :
                #save original params
                originalParams = ObjectMgr().getParamsOfObject( childKeys[0] )
                ObjectMgr().setParams( childKeys[0], params )
                theGuiMsgHandler().runObject( childKeys[0] )
            # set params for multi selection
            if len(self.__keys)>1 : 
                #find changed params
                realChange = ParamsDiff( originalParams, params )
                # set params for remaining selected objects
                for i in range(1, len(self.__keys)):
                    childKeyParams = ObjectMgr().getParamsOfObject(childKeys[i])
                    # find the changed param in childKey and replace it with the
                    # intended attribut
                    for x in realChange :
                        childKeyParams.__dict__[x] = params.__dict__[x]
                    # set the params
                    ObjectMgr().setParams( childKeys[i], childKeyParams )
                    theGuiMsgHandler().runObject( childKeys[i] )
                #ObjectMgr().setParamsMultiSelect( parts[i], self.__getParams() )

    #axis radio button is clicked
    def emitAxisChanged(self):
        self.floatInRangeAxisX.setValue(0.0)
        self.floatInRangeAxisY.setValue(0.0)
        self.floatInRangeAxisZ.setValue(0.0)
        self.emitDataChanged()


#eof
