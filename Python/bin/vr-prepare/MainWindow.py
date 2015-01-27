# Part of the vr-prepare program for dc

# Copyright (c) 2006-2007 Visenso GmbH


import sys
import os.path
import codecs

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QGridLayout, QLabel, QLineEdit, QMenu
from PyQt5.QtWidgets import QTextEdit, QWidget, QDialog, QApplication
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

import covise

from coGRMsg import coGRKeyWordMsg

import KeydObject

# disabled local testing for testNegotiator.py
from negGuiHandlers import initHandlers

from printing import InfoPrintCapable
#REM from FileBrowser import Preview

import ccNotifier

from vtrans import coTranslate

from KeydObject import (
    VIS_3D_BOUNDING_BOX,
    VIS_STREAMLINE,
    VIS_STREAMLINE_2D,
    VIS_MOVING_POINTS,
    VIS_PATHLINES,
    VIS_2D_RAW,
    VIS_DOCUMENT,
    VIS_PLANE,
    VIS_VECTOR,
    VIS_ISOPLANE,
    VIS_ISOCUTTER,
    VIS_CLIPINTERVAL,
    VIS_VECTORFIELD,
    #VIS_POINTPROBING,
    VIS_COVISE,
    VIS_VRML,
    VIS_DOMAINLINES,
    VIS_DOMAINSURFACE,
    VIS_MAGMATRACE,
    VIS_SCENE_OBJECT,
    TYPE_PROJECT,
    TYPE_CASE,
    TYPE_2D_GROUP,
    TYPE_3D_GROUP,
    TYPE_2D_PART,
    TYPE_2D_COMPOSED_PART,
    TYPE_2D_CUTGEOMETRY_PART,
    TYPE_3D_PART,
    TYPE_3D_COMPOSED_PART,
    TYPE_CAD_PART,
    TYPE_PRESENTATION,
    TYPE_PRESENTATION_STEP,
    TYPE_SCENEGRAPH_ITEM,
    TYPE_DNA_ITEM,
    TYPE_GENERIC_OBJECT,
    RUN_GEO,
    RUN_OCT,
    RUN_ALL,
    globalKeyHandler,
    globalProjectKey)

from ErrorManager import (
    NO_ERROR,
    WRONG_PATH_ERROR, 
    TIMESTEP_ERROR)

from ColorComboBoxManager import ColorComboBoxManager
from PresenterManager import PresenterManager
from AnimationManager import AnimationManager
from TrackingManager import TrackingManager
#from coJournalMgr import coJournalMgrParams
from DatasetInformationPanel import DatasetInformationPanel
#from DialogChangeObjectSettingsBase import DialogChangeObjectSettingsBase
#from GeometryObjectsGlobalBase import GeometryObjectsGlobalBase
#from InitialChooserDialog import InitialChooserDialog
#from NamedWithFlagListView import NamedWithFlagListView
#from PreferencesBase import PreferencesBase
#from ProbingSquareBase import ProbingSquareBase
from StartPreparation import StartPreparation
from ProjectInformation import ProjectInformation
from StreamlinesPanel import StreamlinesPanel
from IsoSurfacePanel import IsoSurfacePanel
from IsoCutterPanel import IsoCutterPanel
from VectorFieldPanel import VectorFieldPanel
from ClipIntervalPanel import ClipIntervalPanel
from CuttingSurfacePanel import CuttingSurfacePanel
#from PointProbingPanel import PointProbingPanel
#from MagmaTracePanel import MagmaTracePanel
from Utils import SaveBeforeExitAsker, ReallyWantToOverrideAsker, ChangePathAsker, getIntInLineEdit, getDoubleInLineEdit, ReduceTimestepAsker, OkayAsker, getImportFileTypes, FileNewAsker #, ConversionError
from DialogDuplicateAsker import DialogDuplicateAsker
#from VisensoLogoBase import VisensoLogoBase
from PartVisualizationPanel import PartVisualizationPanel
from GridVisualizationPanel import GridVisualizationPanel
#from Visualization2DPanel import Visualization2DPanel
from PicturePanelBase import Ui_PicturePanelBase
class PicturePanelBase(QtWidgets.QWidget, Ui_PicturePanelBase):
    def __init__(self, parent=None, f=QtCore.Qt.WindowFlags()):
        QtWidgets.QWidget.__init__(self, parent, f)
        self.setupUi(self)
from CoxmlBrowserPanel import CoxmlBrowserPanel
from DocumentViewer import DocumentViewer
from SceneGraphItemPanel import SceneGraphItemPanel
from DNAItemPanel import DNAItemPanel
from VrmlPanel import VrmlPanel
from SceneObjectPanel import SceneObjectPanel
from GridCompositionPanel import GridCompositionPanel
from PartCompositionPanel import PartCompositionPanel
from GenericObjectPanel import GenericObjectPanel
import Application
from Gui2Neg import theGuiMsgHandler

from MainWindowBase import Ui_MainWindowBase
class MainWindowBase(QtWidgets.QMainWindow, Ui_MainWindowBase):
    def __init__(self, parent=None, f=QtCore.Qt.WindowFlags()):
        QtWidgets.QMainWindow.__init__(self, parent, f)
        self.setupUi(self)
import MainWindowBase
from Advice import Advice
from MainWindowHelpBase import Ui_MainWindowHelpBase
class MainWindowHelpBase(QtWidgets.QWidget, Ui_MainWindowHelpBase):
    def __init__(self, parent=None, f=QtCore.Qt.WindowFlags()):
        QtWidgets.QWidget.__init__(self, parent, f)
        self.setupUi(self)
from SceneGraphItemPanelHelpBase import Ui_SceneGraphItemPanelHelpBase
class SceneGraphItemPanelHelpBase(QtWidgets.QWidget, Ui_SceneGraphItemPanelHelpBase):
    def __init__(self, parent=None, f=QtCore.Qt.WindowFlags()):
        QtWidgets.QWidget.__init__(self, parent, f)
        self.setupUi(self)
from DatasetInformationHelpBase import Ui_DatasetInformationHelpBase
class DatasetInformationHelpBase(QtWidgets.QWidget, Ui_DatasetInformationHelpBase):
    def __init__(self, parent=None, f=QtCore.Qt.WindowFlags()):
        QtWidgets.QWidget.__init__(self, parent, f)
        self.setupUi(self)
from GridCompositionPanelHelpBase import Ui_GridCompositionPanelHelpBase
class GridCompositionPanelHelpBase(QtWidgets.QWidget, Ui_GridCompositionPanelHelpBase):
    def __init__(self, parent=None, f=QtCore.Qt.WindowFlags()):
        QtWidgets.QWidget.__init__(self, parent, f)
        self.setupUi(self)
from PartCompositionPanelHelpBase import Ui_PartCompositionPanelHelpBase
class PartCompositionPanelHelpBase(QtWidgets.QWidget, Ui_PartCompositionPanelHelpBase):
    def __init__(self, parent=None, f=QtCore.Qt.WindowFlags()):
        QtWidgets.QWidget.__init__(self, parent, f)
        self.setupUi(self)
#from ColoringGroupHelpBase import ColoringGroupHelpBase
from PartVisualizationPanelHelpBase import Ui_PartVisualizationPanelHelpBase
class PartVisualizationPanelHelpBase(QtWidgets.QWidget, Ui_PartVisualizationPanelHelpBase):
    def __init__(self, parent=None, f=QtCore.Qt.WindowFlags()):
        QtWidgets.QWidget.__init__(self, parent, f)
        self.setupUi(self)
from GridVisualizationPanelHelpBase import Ui_GridVisualizationPanelHelpBase
class GridVisualizationPanelHelpBase(QtWidgets.QWidget, Ui_GridVisualizationPanelHelpBase):
    def __init__(self, parent=None, f=QtCore.Qt.WindowFlags()):
        QtWidgets.QWidget.__init__(self, parent, f)
        self.setupUi(self)
from CuttingSurfaceContoursHelpBase import Ui_CuttingSurfaceContoursHelpBase
class CuttingSurfaceContoursHelpBase(QtWidgets.QWidget, Ui_CuttingSurfaceContoursHelpBase):
    def __init__(self, parent=None, f=QtCore.Qt.WindowFlags()):
        QtWidgets.QWidget.__init__(self, parent, f)
        self.setupUi(self)
from CuttingSurfaceVectorsHelpBase import Ui_CuttingSurfaceVectorsHelpBase
class CuttingSurfaceVectorsHelpBase(QtWidgets.QWidget, Ui_CuttingSurfaceVectorsHelpBase):
    def __init__(self, parent=None, f=QtCore.Qt.WindowFlags()):
        QtWidgets.QWidget.__init__(self, parent, f)
        self.setupUi(self)
from StreamlinesHelpBase import Ui_StreamlinesHelpBase
class StreamlinesHelpBase(QtWidgets.QWidget, Ui_StreamlinesHelpBase):
    def __init__(self, parent=None, f=QtCore.Qt.WindowFlags()):
        QtWidgets.QWidget.__init__(self, parent, f)
        self.setupUi(self)
#from DomainLinesHelpBase import DomainLinesHelpBase
from IsoSurfaceHelpBase import Ui_IsoSurfaceHelpBase
class IsoSurfaceHelpBase(QtWidgets.QWidget, Ui_IsoSurfaceHelpBase):
    def __init__(self, parent=None, f=QtCore.Qt.WindowFlags()):
        QtWidgets.QWidget.__init__(self, parent, f)
        self.setupUi(self)
from IsoCutterHelpBase import Ui_IsoCutterHelpBase
class IsoCutterHelpBase(QtWidgets.QWidget, Ui_IsoCutterHelpBase):
    def __init__(self, parent=None, f=QtCore.Qt.WindowFlags()):
        QtWidgets.QWidget.__init__(self, parent, f)
        self.setupUi(self)
from VectorFieldHelpBase import Ui_VectorFieldHelpBase
class VectorFieldHelpBase(QtWidgets.QWidget, Ui_VectorFieldHelpBase):
    def __init__(self, parent=None, f=QtCore.Qt.WindowFlags()):
        QtWidgets.QWidget.__init__(self, parent, f)
        self.setupUi(self)
from ClipIntervalHelpBase import Ui_ClipIntervalHelpBase
class ClipIntervalHelpBase(QtWidgets.QWidget, Ui_ClipIntervalHelpBase):
    def __init__(self, parent=None, f=QtCore.Qt.WindowFlags()):
        QtWidgets.QWidget.__init__(self, parent, f)
        self.setupUi(self)
#from ProbingPointHelpBase import ProbingPointHelpBase
#from MagmaTraceHelpBase import MagmaTraceHelpBase
from DocumentViewerHelpBase import Ui_DocumentViewerHelpBase
class DocumentViewerHelpBase(QtWidgets.QWidget, Ui_DocumentViewerHelpBase):
    def __init__(self, parent=None, f=QtCore.Qt.WindowFlags()):
        QtWidgets.QWidget.__init__(self, parent, f)
        self.setupUi(self)
from NoHelpBase import Ui_NoHelpBase
class NoHelpBase(QtWidgets.QWidget, Ui_NoHelpBase):
    def __init__(self, parent=None, f=QtCore.Qt.WindowFlags()):
        QtWidgets.QWidget.__init__(self, parent, f)
        self.setupUi(self)
from MultiSelectionHelpBase import Ui_MultiSelectionHelpBase

class MultiSelectionHelpBase(QtWidgets.QWidget, Ui_MultiSelectionHelpBase):
    def __init__(self, parent=None, f=QtCore.Qt.WindowFlags()):
        QtWidgets.QWidget.__init__(self, parent, f)
        self.setupUi(self)
        
from PatienceDialogManager import PatienceDialogManager
from ViewpointManager import ViewpointManager
from VideoCaptureManager import VideoCaptureManager
#from SyncManager import SyncManager
from CroppingManager import CroppingManager
from VRPCoviseNetAccess import theNet
from CoverWidget import CoverWidget
from DomainSurfacePanel import DomainSurfacePanel
from DomainSurfaceHelpBase import Ui_DomainSurfaceHelpBase
class DomainSurfaceHelpBase(QtWidgets.QWidget, Ui_DomainSurfaceHelpBase):
    def __init__(self, parent=None, f=QtCore.Qt.WindowFlags()):
        QtWidgets.QWidget.__init__(self, parent, f)
        self.setupUi(self)
from GettingStartedWindow import GettingStartedWindow
from SceneObjectListWindow import SceneObjectListWindow

from CasesFilter import CasesFilter
import coviseCase

import ObjectMgr
from KeyedTreeMgr import KeyedTreeMgr
import vtrans

_infoer = InfoPrintCapable()
_infoer.doPrint =  False #True #

globalAccessToTreeView = None
globalPdmForOpen = None
globalColorManager = None
globalSyncManager = None


class MainWindow(QMainWindow,Ui_MainWindowBase):

    PART2D = 0
    PART2DVARIABLE = 1
    PART3D = 2
    PART3DVARIABLE = 3

    def __init__(self, parent=None):
        _infoer.function = str(self.__init__)
        _infoer.write("")

        # load custom images first
        imageStyle = covise.getCoConfigEntry("vr-prepare.ImageStyle")
        if imageStyle:
            QtCore.QResource.registerResource(os.path.dirname(__file__) + "/" + imageStyle)
        # then load default images
        QtCore.QResource.registerResource(os.path.dirname(__file__) + "/" + "StaticImages.rcc")
        
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        
        iconimage = covise.getCoConfigEntry("vr-prepare.ApplicationIcon")
        if iconimage:
            icon = QIcon(iconimage)
            if icon:
                self.setWindowIcon(icon)
        
        # set stylesheet
        qssStyleSheet = covise.getCoConfigEntry("vr-prepare.StyleSheet")
        if qssStyleSheet:
            fstyle = QtCore.QFile(qssStyleSheet)
            fstyle.open(QtCore.QFile.ReadOnly)
            styleSheet = fstyle.readAll()
            self.setStyleSheet(str(styleSheet))


        
        self.__disableBrokenParts()
        self.__forceClose = False
        self.setWindowTitle(self.__tr('<untitled>'))
        #self.__initProcessInfoDockWindow()

            
        # panel for every object type
        self.__panelForType = {}
        self.__docuForType = {}

        self.FrameRightTopWidget.setVisible(False) # will be made visible when needed
        self.clearFilterButton.hide()
        self.statusBar().hide() # hide the status bar (to save space -> no save-confirmations)

        # hide buttons for missing default viewpoints
        self.actionFront.setVisible( covise.coConfigIsOn("COVER.Plugin.ViewPoint.Viewpoints:front") )
        self.actionBack.setVisible( covise.coConfigIsOn("COVER.Plugin.ViewPoint.Viewpoints:back") )
        self.actionLeft.setVisible( covise.coConfigIsOn("COVER.Plugin.ViewPoint.Viewpoints:left") )
        self.actionRight.setVisible( covise.coConfigIsOn("COVER.Plugin.ViewPoint.Viewpoints:right") )
        self.actionTop.setVisible( covise.coConfigIsOn("COVER.Plugin.ViewPoint.Viewpoints:top") )
        self.actionBottom.setVisible( covise.coConfigIsOn("COVER.Plugin.ViewPoint.Viewpoints:bottom") )

        self.setupFeatures()
        
        #open the PresenterManager
        self.presenterManager = PresenterManager( self )
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.presenterManager )
        if covise.coConfigIsOn("vr-prepare.Features.PresentationManager", True):
            if not covise.coConfigIsOn("vr-prepare.Panels.PresentationManager", "visible", True):
                self.presenterManager.hide()
                self.windowPresenter_ManagerAction.setChecked(False)
        else:
            self.presenterManager.hide()

        #open the ViewpointManager
        self.viewpointManager = ViewpointManager(self)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.viewpointManager)
        if covise.coConfigIsOn("vr-prepare.Features.ViewpointManager", True):
            if not covise.coConfigIsOn("vr-prepare.Panels.ViewpointManager", "visible", True):
                self.viewpointManager.hide()
                self.windowViewpoint_ManagerAction.setChecked(False)
        else:
            self.viewpointManager.hide()

        #open the AnimationManager
        self.animationManager = AnimationManager(self)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.animationManager)
        self.animationManager.hide()

        #open the TrackingManager
        self.trackingManager = TrackingManager(self)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.trackingManager)
        if covise.coConfigIsOn("vr-prepare.Features.TrackingManager", True):
            if not covise.coConfigIsOn("vr-prepare.Panels.TrackingManager", "visible", False):
                self.trackingManager.hide()
                self.windowTracking_ManagerAction.setChecked(False)
        else:
            self.trackingManager.hide()

        #videoCapture
        self.videoCaptureManager = VideoCaptureManager(self)
        self.addDockWidget(QtCore.Qt.TopDockWidgetArea, self.videoCaptureManager)
        if not covise.coConfigIsOn("vr-prepare.Panels.VideoCaptureManager", "docked", True):
            self.videoCaptureManager.setFloating(True)
        self.videoCaptureManager.hide()

        #open the advice
        self.advice = Advice(self)
        # if window size is big enough, open as internal dock window
        if covise.coConfigIsOn("vr-prepare.Panels.Advice", "docked", True):
            self.addDockWidget(QtCore.Qt.TopDockWidgetArea, self.advice)
        # else open outside
        else :
            self.addDockWidget(QtCore.Qt.TopDockWidgetArea, self.advice)
            self.advice.setFloating(True)
            self.advice.resize(300,500)
        if covise.coConfigIsOn("vr-prepare.Features.Advice", True):
            if not covise.coConfigIsOn("vr-prepare.Panels.Advice", "visible", True):
                self.advice.hide()
                self.windowAdviceAction.setChecked(False)
        else:
            self.advice.hide()

        global globalColorManager
        globalColorManager = ColorComboBoxManager(self)

        """
        global globalSyncManager
        globalSyncManager = SyncManager(self)

        global tesselationPanel
        tesselationPanel=None
        # tesselationPanel = TesselationPanel(self)
        # self.WidgetStackRight.addWidget(tesselationPanel)
        """

        # add tree view 
        global globalAccessToTreeView
        self.coverWidget = None # widget for embedded
        self.coverWidgetId = None
        if covise.coConfigIsOn("vr-prepare.Panels.SceneView", "dockable", False) or (not covise.coConfigIsOn("vr-prepare.Features.SceneView", True)):
            #we want the treeview hidden - start it as a DockWidget
            self.QDockWidgetTreeView = QtWidgets.QDockWidget(self)
            self.QDockWidgetTreeView.setMinimumSize(360,200)
            globalAccessToTreeView = KeyedTreeMgr(self.QDockWidgetTreeView)
            self.QDockWidgetTreeView.setWidget(globalAccessToTreeView)
            self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.QDockWidgetTreeView)
            if not covise.coConfigIsOn("vr-prepare.Panels.SceneView", "visible", True):
                self.QDockWidgetTreeView.hide()
            else:
                self.windowScene_ViewAction.setChecked(True)
            #hide the bar for filtering # TODO: move to SceneViewPanel so the can be used if the SceneView is dockable
            self.frameButtons.hide()
            # if the treeview is not in the left widget stack we can start the cover there
            # start an empty window for cover
            if covise.coConfigIsOn("vr-prepare.EmbeddedOpenCOVER"):
                self.FrameRightTopWidget.setVisible(True)
                self.coverWidget = CoverWidget(self.FrameLeft, self)
                #self.coverWidget.setEnabled(False)
                self.WidgetStackLeft.addWidget(self.coverWidget)
                self.WidgetStackLeft.setCurrentWidget(self.coverWidget)
                #self.WidgetStackLeft.setEnabled(False)
                self.coverWidgetId = self.coverWidget.winId()
                self.WidgetStackLeft.setUpdatesEnabled(False)
                self.FrameLeft.setUpdatesEnabled(False)
                #self.WidgetStackLeft.setEnabled(False)
                #self.splitter1.setUpdatesEnabled(False)
                #self.FrameLeft.setAutoFillBackground(False)
                #self.splitter1.setAutoFillBackground(False)
                #self.setUpdatesEnabled(False)
                #self.toolBar_Standard.setUpdatesEnabled(False)
                #self.toolBar_Viewpoints.setUpdatesEnabled(False)
                #self.toolBar_snap.setUpdatesEnabled(False)
                #self.toolBar_Presenter.setUpdatesEnabled(False)
                #self.WidgetStackRight.setParent(None)
                #self.FrameRight.setParent(None)
                self.WidgetStackRight.setMaximumWidth(350)
                self.WidgetStackRight.setUpdatesEnabled(True)
                self.FrameRight.setUpdatesEnabled(True)
                self.FrameRight.setFixedWidth(350)
                self.FrameRight.hide()
                # workaround for embedded cover with stylesheets
                #self.widget.setUpdatesEnabled(False)
            else:
                self.FrameRightPinButton.setVisible(False)
        else:
            self.FrameRightPinButton.setVisible(False)
            globalAccessToTreeView = KeyedTreeMgr(self.WidgetStackLeft)
            self.WidgetStackLeft.addWidget(globalAccessToTreeView)
            self.WidgetStackLeft.setCurrentWidget(globalAccessToTreeView)
            self.QDockWidgetTreeView = None
            # remove action for show/hide sceneview (not nescessary since SceneView is not dockable)
            self.windowScene_ViewAction.setVisible(False)

        globalAccessToTreeView.sigItemChecked.connect(self._checkboxToggleSlot)
        globalAccessToTreeView.sigItemClicked.connect(self._itemClickedInTreeSlot)
        globalAccessToTreeView.sigSelectionChanged.connect(self._selectionChangedInTreeSlot)
        globalAccessToTreeView.sigNoItemClicked.connect(self._noItemClicked)
        globalAccessToTreeView.sigRemoveFilter.connect(self._clearTreeViewFilter)
        
        self.filterLineEdit.textChanged.connect(self._changeTreeViewFilter)
        self.filterSceneGraphItems.stateChanged.connect(self._changeTreeViewFilter)
        self.clearFilterButton.clicked.connect(self._clearTreeViewFilter)

        # advice widget stack
        self.mainWindowHelp = MainWindowHelpBase(self.advice.getWidgetStack())
        self.noHelp = NoHelpBase(self.advice.getWidgetStack())
        self.advice.addToStack(self.noHelp)

        self.multiSelectHelp = MultiSelectionHelpBase(self.advice.getWidgetStack())
        self.advice.addToStack(self.multiSelectHelp)

        self.__panelForType[TYPE_CASE] = DatasetInformationPanel(self.WidgetStackRight)
        self.__docuForType[TYPE_CASE] = DatasetInformationHelpBase(self.advice.getWidgetStack())
        self.datasetInformation = self.__panelForType[TYPE_CASE]

        self.__panelForType[TYPE_3D_GROUP] = GridCompositionPanel(self.WidgetStackRight)
        self.__panelForType[TYPE_3D_GROUP].sigComposedGridRequest.connect(self.composedGridRequest)
        self.__docuForType[TYPE_3D_GROUP] = GridCompositionPanelHelpBase(self.advice.getWidgetStack())

        self.__panelForType[TYPE_3D_PART] = GridVisualizationPanel(self.WidgetStackRight)
        self.__docuForType[TYPE_3D_PART] = GridVisualizationPanelHelpBase(self.advice.getWidgetStack())
        self.__panelForType[TYPE_3D_COMPOSED_PART] = GridVisualizationPanel(self.WidgetStackRight)

        self.__docuForType[TYPE_3D_COMPOSED_PART] = GridVisualizationPanelHelpBase(self.advice.getWidgetStack())

        self.__panelForType[TYPE_2D_GROUP] = PartCompositionPanel(self.WidgetStackRight)
        self.__panelForType[TYPE_2D_GROUP].sigComposedPartRequest.connect(self.composedPartRequest)
        self.__docuForType[TYPE_2D_GROUP] = PartCompositionPanelHelpBase(self.advice.getWidgetStack())

        self.__panelForType[TYPE_2D_PART] = PartVisualizationPanel(self.WidgetStackRight)
        self.__docuForType[TYPE_2D_PART] = PartVisualizationPanelHelpBase(self.advice.getWidgetStack())
        self.__panelForType[TYPE_2D_COMPOSED_PART] = PartVisualizationPanel(self.WidgetStackRight)
        self.__docuForType[TYPE_2D_COMPOSED_PART] = PartVisualizationPanelHelpBase(self.advice.getWidgetStack())
        self.__panelForType[TYPE_2D_CUTGEOMETRY_PART] = PartVisualizationPanel(self.WidgetStackRight)
        self.__docuForType[TYPE_2D_CUTGEOMETRY_PART] = PartVisualizationPanelHelpBase(self.advice.getWidgetStack())

        self.__panelForType[VIS_STREAMLINE] = StreamlinesPanel(self.WidgetStackRight)
        self.__panelForType[VIS_MOVING_POINTS] = self.__panelForType[VIS_STREAMLINE]
        self.__panelForType[VIS_PATHLINES] = self.__panelForType[VIS_STREAMLINE]
        self.__panelForType[VIS_STREAMLINE_2D] = self.__panelForType[VIS_STREAMLINE]

        self.__docuForType[VIS_STREAMLINE] = StreamlinesHelpBase(self.advice.getWidgetStack())
        self.__docuForType[VIS_MOVING_POINTS] = self.__docuForType[VIS_STREAMLINE]
        self.__docuForType[VIS_PATHLINES] = self.__docuForType[VIS_STREAMLINE]
        self.__docuForType[VIS_STREAMLINE_2D] = self.__docuForType[VIS_STREAMLINE]

        self.__panelForType[VIS_PLANE] = CuttingSurfacePanel(self.WidgetStackRight)
        self.__panelForType[VIS_PLANE] = self.__panelForType[VIS_PLANE]

        self.__docuForType[VIS_PLANE] = CuttingSurfaceContoursHelpBase(self.advice.getWidgetStack())
        self.__docuForType[VIS_VECTOR] = CuttingSurfaceVectorsHelpBase(self.advice.getWidgetStack())

        self.__panelForType[VIS_ISOPLANE] = IsoSurfacePanel(self.WidgetStackRight)
        self.__docuForType[VIS_ISOPLANE] = IsoSurfaceHelpBase(self.advice.getWidgetStack())

        self.__panelForType[VIS_ISOCUTTER] = IsoCutterPanel(self.WidgetStackRight)
        self.__docuForType[VIS_ISOCUTTER] = IsoCutterHelpBase(self.advice.getWidgetStack())

        self.__panelForType[VIS_VECTORFIELD] = VectorFieldPanel(self.WidgetStackRight)
        self.__docuForType[VIS_VECTORFIELD] = VectorFieldHelpBase(self.advice.getWidgetStack())

        self.__panelForType[VIS_CLIPINTERVAL] = ClipIntervalPanel(self.WidgetStackRight)
        self.__docuForType[VIS_CLIPINTERVAL] = ClipIntervalHelpBase(self.advice.getWidgetStack())

        self.__panelForType[TYPE_SCENEGRAPH_ITEM] = SceneGraphItemPanel(self.WidgetStackRight)
        self.__docuForType[TYPE_SCENEGRAPH_ITEM] = SceneGraphItemPanelHelpBase(self.advice.getWidgetStack())

        self.__panelForType[TYPE_DNA_ITEM] = DNAItemPanel(self.WidgetStackRight)

        self.__panelForType[VIS_VRML] = VrmlPanel(self.WidgetStackRight)
        self.__panelForType[VIS_SCENE_OBJECT] = SceneObjectPanel(self.WidgetStackRight)

        self.__panelForType[TYPE_GENERIC_OBJECT] = GenericObjectPanel(self.WidgetStackRight)

        self.__panelForType[VIS_DOCUMENT]  = DocumentViewer(self)
        self.__docuForType[VIS_DOCUMENT] = DocumentViewerHelpBase(self.advice.getWidgetStack())

        self.__panelForType[VIS_DOMAINSURFACE] = DomainSurfacePanel(self.WidgetStackRight)
        self.__docuForType[VIS_DOMAINSURFACE] = DomainSurfaceHelpBase(self.advice.getWidgetStack())
        

        if covise.coConfigIsOn("vr-prepare.Features.CoxmlBrowser", False):
            self.noSelectionPanel = CoxmlBrowserPanel(self.WidgetStackRight)
            # workaround for embedded cover with stylesheets
            if self.coverWidget:
                self.noSelectionPanel.setDependingWidgets(self.widget)
            self.noSelectionPanel.importCoxml.connect(self.importRequest)
            self.FrameRightTopWidget.setVisible(True)
        else:
            self.noSelectionPanel = PicturePanelBase(self.WidgetStackRight)
            self.CoxmlButton.setVisible(False)
            self.PropertiesButton.setVisible(False)
        self.WidgetStackRight.addWidget(self.noSelectionPanel)
        

        #self.__panelForType[VIS_POINTPROBING] = PointProbingPanel(self.WidgetStackRight)
        #self.__docuForType[VIS_POINTPROBING] = ProbingPointHelpBase(self.advice.getWidgetStack())

        #self.probingSquareBase = ProbingSquareBase(self.WidgetStackRight)
        #self.WidgetStackRight.addWidget(self.probingSquareBase)

        #self.__panelForType[VIS_MAGMATRACE] = MagmaTracePanel(self.WidgetStackRight)
        #self.__docuForType[VIS_MAGMATRACE] = MagmaTraceHelpBase(self.advice.getWidgetStack())
        #self.__docuForType[VIS_DOMAINLINES] = DomainLinesHelpBase(self.advice.getWidgetStack())

        # add all panels
        for typeNr in self.__panelForType:
            self.WidgetStackRight.addWidget( self.__panelForType[typeNr] )
        for typeNr in self.__docuForType:
            self.advice.addToStack( self.__docuForType[typeNr] )

        self.advice.addToStack(self.mainWindowHelp)
        self.advice.raiseInStack(self.mainWindowHelp)

        # connect color manager if necessary
        panelsRaisingColorManager = [
            VIS_STREAMLINE,
            VIS_PLANE,
            VIS_ISOPLANE,
            VIS_ISOCUTTER,
            VIS_CLIPINTERVAL,
            VIS_VECTORFIELD,
            VIS_DOMAINSURFACE,
            #VIS_POINTPROBING,
            TYPE_2D_PART,
            TYPE_2D_CUTGEOMETRY_PART,
            #VIS_MAGMATRACE,
            ]
        for typeNr in  panelsRaisingColorManager :
            self.__panelForType[typeNr].sigEditColorMap.connect(self.editColorMaps)
            # immediately change colormap selection in panel if selection changed in color manager window
            globalColorManager.sigSelectColorMap.connect(self.__panelForType[typeNr].setSelectedColormap)

        self.lastClickedKey = None
        # ignore item signal if check button was toggled
        self.__ignoreClick = False

        # NOTE: some additional connections are realized via the designer
        #connect presenter manager
        self.windowPresenter_ManagerAction.toggled.connect(self.showPresenterWindow)
        self.createPresentation_StepAction.triggered.connect(self.presenterManager.new)
        self.presentationNew_Presentation_StepAction.triggered.connect(self.presenterManager.new)
        self.presentationDelete_Presentation_StepAction.triggered.connect(self.presenterManager.delete)
        self.presentationNavigationPlayAction.triggered.connect(self.presenterManager.play)
        self.presentationNavigationForwardAction.triggered.connect(self.presenterManager.forward)
        self.presentationNavigationGo_To_StartAction.triggered.connect(self.presenterManager.goToStart)
        self.presentationNavigationGo_To_EndAction.triggered.connect(self.presenterManager.goToEnd)
        self.presentationNavigationBackAction.triggered.connect(self.presenterManager.backward)
        self.presentationNavigationStopAction.triggered.connect(self.presenterManager.stop)
        self.presentationNewViewpointAction.toggled.connect(self.presenterManager.newViewpoint)
        self.enablePresenter(False)
        #connect Viewpoint manager
        self.windowViewpoint_ManagerAction.toggled.connect(self.showViewpointWindow)
        self.createViewpointAction.triggered.connect(self.addViewpoint)
        self.viewpointsNew_ViewpointAction.triggered.connect(self.addViewpoint)
        self.viewpointsDelete_ViewpointAction.triggered.connect(self.viewpointManager.delete)
        self.viewpointsChange_ViewpointAction.triggered.connect(self.viewpointManager.change)
        self.viewpointsView_AllAction.triggered.connect(self.viewpointManager.viewAll)
        self.actionOrthographic_Projection.triggered.connect(self.viewpointManager.orthographicProjection)
        self.actionTurntable_animation.triggered.connect(self.viewpointManager.turntableAnimation)
        self.actionTurntable_rotate45.triggered.connect(self.viewpointManager.turntableRotate45)
        self.actionFront.triggered.connect(self.viewpointManager.showViewpointFront)
        self.actionBack.triggered.connect(self.viewpointManager.showViewpointBack)
        self.actionLeft.triggered.connect(self.viewpointManager.showViewpointLeft)
        self.actionRight.triggered.connect(self.viewpointManager.showViewpointRight)
        self.actionTop.triggered.connect(self.viewpointManager.showViewpointTop)
        self.actionBottom.triggered.connect(self.viewpointManager.showViewpointBottom)
        self.viewpointsFlying_ModeAction.toggled.connect(self.viewpointManager.setFlyMode)
        self.snapallAction.triggered.connect(self.viewpointManager.snapAll)
        self.snapshotAction.triggered.connect(self.viewpointManager.takeSnapshot)
        self.videoCaptureAction.triggered.connect(self.showVideoCaptureWindow)
        self.actionSceneEditorGrid.triggered.connect(self.toggleSceneEditorGrid)
        self.actionSceneEditorOperatingRange.triggered.connect(self.toggleSceneEditorOperatingRange)      
        self.enableViewpoint(False)
        #connect advice
        self.windowAdviceAction.toggled.connect(self.showAdviceWindow)  
        self.showAdvice.triggered.connect(self.windowAdviceAction.toggle)  
        #connect help
        self.helpGettingStartedAction.triggered.connect(self.showGettingStartedWindow)  
        #connect animation manager
        self.windowAnimation_ManagerAction.toggled.connect(self.showAnimationWindow)
        #connect tracking manager
        self.windowAnimation_ManagerAction.toggled.connect(self.showTrackingWindow)
        #connect scene view
        if self.QDockWidgetTreeView:
            self.QDockWidgetTreeView.visibilityChanged.connect(self.setShowSceneView)
        self.windowScene_ViewAction.toggled.connect(self.showSceneView)
        #connect part manipulation
        self.editPartsCut_Selected_PartAction.triggered.connect(self.cutParts)
        self.editPartsDelete_Selected_PartAction.triggered.connect(self.deleteParts)
        self.editPartsDuplicate_Selected_PartAction.triggered.connect(self.duplicateParts)
        self.editPartsSynchronization_Options_Action.triggered.connect(self.syncPartOptions)
        self.editPartsCrop_all_parts_Action.triggered.connect(self.cropAllParts)
        #connect visibility manipulation
        self.editPartsShow_Selected_PartAction.triggered.connect(self.showParts)
        self.editPartsHide_Selected_PartAction.triggered.connect(self.hideParts)
        #connect documentation (list of all scene objects)
        self.actionScene_Object_List.triggered.connect(self.createSceneObjectList)
        # connect navigation
        self.navigationMode_NoneAction.triggered.connect(self.navigationModeNone)
        self.navigationMode_TransformAction.triggered.connect(self.navigationModeTransform)
        self.navigationMode_MeasureAction.triggered.connect(self.navigationModeMeasure)
        #connect coxml/properties button
        self.CoxmlButton.clicked.connect(self._coxmlButtonClicked)
        self.PropertiesButton.clicked.connect(self._propertiesButtonClicked)


        # Popup Menu (we use the actions from the menu bar)
        self.__contextMenuParts = QMenu()
        self.__contextMenuParts.addAction(self.editPartsDuplicate_Selected_PartAction)
        self.__contextMenuParts.addAction(self.editPartsCut_Selected_PartAction)
        self.__contextMenuParts.addAction(self.editPartsDelete_Selected_PartAction)
        self.__contextMenuParts.addSeparator()
        self.__contextMenuParts.addAction(self.editPartsShow_Selected_PartAction)
        self.__contextMenuParts.addAction(self.editPartsHide_Selected_PartAction)
        
        globalAccessToTreeView.sigContextMenuRequest.connect(self._contextMenuRequest)

        self.WidgetStackRight.setCurrentWidget(self.noSelectionPanel)

        self.__projectFilenameSuggestion = covise.getCoConfigEntry("vr-prepare.InitialDatasetSearchPath")
        #print(".-.-.-.-.InitialDatasetSearchPath", self.__projectFilenameSuggestion)
        self.__currentProjectFilename = ''

        # register okayAsker
        theGuiMsgHandler().registerErrorCallback('okay', self.raiseOkayDialog)
        theGuiMsgHandler().registerKeyWordCallback('SET_SELECTION', self.setSelectionOfTreeView)


    def setupFeatures(self):
        # SceneView
        isOn = covise.coConfigIsOn("vr-prepare.Features.SceneView", True)
        self.windowScene_ViewAction.setVisible(isOn) # [Window / Scene View] (might be hidden later if SceneView is embedded)
        # PresentationManager
        isOn = covise.coConfigIsOn("vr-prepare.Features.PresentationManager", True)
        self.windowPresenter_ManagerAction.setVisible(isOn) # [Window / Presentation Manager]
        self.createPresentation_StepAction.setVisible(isOn) # [Create / Presentation Step]
        self.presentationNew_Presentation_StepAction.setVisible(isOn) # [Presentation / New Presentation Step]
        self.presentationDelete_Presentation_StepAction.setVisible(isOn) # [Presentation / Delete Presentation Step]
        self.presentationNavigationBackAction.setVisible(isOn) # [Presentation / Navigation / Back]
        self.presentationNavigationForwardAction.setVisible(isOn) # [Presentation / Navigation / Forward]
        self.presentationNavigationGo_To_EndAction.setVisible(isOn) # [Presentation / Navigation / Go to end]
        self.presentationNavigationGo_To_StartAction.setVisible(isOn) # [Presentation / Navigation / Go to start]
        self.presentationNavigationPlayAction.setVisible(isOn) # [Presentation / Navigation / Play]
        self.presentationNavigationStopAction.setVisible(isOn) # [Presentation / Navigation / Stop]
        self.presentationNewViewpointAction.setVisible(isOn) # [Presentation / Always create new Viewpoint]
        # ViewpointManager
        isOn = covise.coConfigIsOn("vr-prepare.Features.ViewpointManager", True)
        self.windowViewpoint_ManagerAction.setVisible(isOn) # [Window / Viewpoint Manager]
        self.createViewpointAction.setVisible(isOn) # [Create / Viewpoint]
        self.viewpointsNew_ViewpointAction.setVisible(isOn) # [Viewpoints / New Viewpoint]
        self.viewpointsDelete_ViewpointAction.setVisible(isOn) # [Viewpoints / Delete Viewpoint]
        self.viewpointsChange_ViewpointAction.setVisible(isOn) # [Viewpoints / Change Viewpoint]
        # TrackingManager
        isOn = covise.coConfigIsOn("vr-prepare.Features.TrackingManager", True)
        self.windowTracking_ManagerAction.setVisible(isOn) # [Window / Tracking Manager]
        # AnimationManager
        isOn = covise.coConfigIsOn("vr-prepare.Features.AnimationManager", True)
        self.windowAnimation_ManagerAction.setVisible(isOn) # [Window / Animation Manager]
        # RestartRenderer
        isOn = covise.coConfigIsOn("vr-prepare.Features.RestartRenderer", True)
        self.RendererRestartAction.setVisible(isOn) # [Window / Restart Renderer]
        # Advice
        isOn = covise.coConfigIsOn("vr-prepare.Features.Advice", True)
        self.windowAdviceAction.setVisible(isOn) # [Window / Advice]
        self.showAdvice.setVisible(isOn) # [Help / Advice]
        # GettingStarted
        isOn = covise.coConfigIsOn("vr-prepare.Features.GettingStarted", False)
        self.helpGettingStartedAction.setVisible(isOn) # [Window / Getting Started]
        # OrthographicProjection
        isOn = covise.coConfigIsOn("vr-prepare.Features.OrthographicProjection", True)
        self.actionOrthographic_Projection.setVisible(isOn) # [Viewpoints / Orthographic Projection]
        # TurntableAnimation
        isOn = covise.coConfigIsOn("vr-prepare.Features.TurntableAnimation", True)
        self.actionTurntable_animation.setVisible(isOn) # [Viewpoints / Turntable Animation]
        # TurntableRotate45
        isOn = covise.coConfigIsOn("vr-prepare.Features.TurntableRotate45", True)
        self.actionTurntable_rotate45.setVisible(isOn) # [Viewpoints / Turntable Roatate 45]
        # FylingMode
        isOn = covise.coConfigIsOn("vr-prepare.Features.FlyingMode", True)
        self.viewpointsFlying_ModeAction.setVisible(isOn) # [Viewpoints / Flying Mode]
        # FileNewWizard
        isOn = covise.coConfigIsOn("vr-prepare.Features.FileNewWizard", True)
        self.fileWizardAction.setVisible(isOn) # [File / New Project (Wizard) ...]
        # AddProject
        isOn = covise.coConfigIsOn("vr-prepare.Features.AddProject", False)
        self.fileAddAction.setVisible(isOn) # [File / Add Project]
        # CompareProject
        isOn = covise.coConfigIsOn("vr-prepare.Features.CompareProject", False)
        self.fileCompare_Project_Action.setVisible(isOn) # [File / Compare Project]
        # ExportToMap
        isOn = covise.coConfigIsOn("vr-prepare.Features.ExportToMap", False)
        self.fileExport_to_mapAction.setVisible(isOn) # [File / Export to Map]
        # EditPartsMenu
        isOn = covise.coConfigIsOn("vr-prepare.Features.EditPartsMenu", True)
        self.editPartsDuplicate_Selected_PartAction.setVisible(isOn) # [Edit / Parts / ...]
        self.editPartsCut_Selected_PartAction.setVisible(isOn) # [Edit / Parts / ...]
        self.editPartsDelete_Selected_PartAction.setVisible(isOn) # [Edit / Parts / ...]
        self.editPartsShow_Selected_PartAction.setVisible(isOn) # [Edit / Parts / ...]
        self.editPartsHide_Selected_PartAction.setVisible(isOn) # [Edit / Parts / ...]
        self.editPartsSynchronization_Options_Action.setVisible(isOn) # [Edit / Parts / ...]
        self.editPartsCrop_all_parts_Action.setVisible(isOn) # [Edit / Parts / ...]
        # Snapshot
        isOn = covise.coConfigIsOn("vr-prepare.Features.Snapshot", True)
        self.snapshotAction.setVisible(isOn) # [File / Snapshot]
        # SnapAll
        isOn = covise.coConfigIsOn("vr-prepare.Features.SnapAll", False)
        self.snapallAction.setVisible(isOn) # [File / Snap all]
        # VideoCapture
        isOn = covise.coConfigIsOn("vr-prepare.Features.VideoCapture", False)
        self.videoCaptureAction.setVisible(isOn) # [File / Capture video]
        # NavigationMode
        isOn = covise.coConfigIsOn("vr-prepare.Features.NavigationMode", False)
        self.navigationMode_NoneAction.setVisible(isOn) # [Navigation / None]
        self.navigationMode_TransformAction.setVisible(isOn) # [Navigation / Transform]
        self.navigationMode_MeasureAction.setVisible(isOn) # [Navigation / Measure]
        # Grid (SceneEditor)
        isOn = covise.coConfigIsOn("vr-prepare.Features.SceneEditor_Grid", False)
        self.actionSceneEditorGrid.setVisible(isOn) # [Viewpoints / Grid]
        # OperatingRange (SceneEditor)
        isOn = covise.coConfigIsOn("vr-prepare.Features.SceneEditor_OperatingRange", False)
        self.actionSceneEditorOperatingRange.setVisible(isOn) # [Viewpoints / Operating Range]
        # CreateSceneObjectList
        isOn = covise.coConfigIsOn("vr-prepare.Features.CreateSceneObjectList", False)
        self.actionScene_Object_List.setVisible(isOn) # [Create / Scene Object List]
        # Help ...

        # hide empty menus and double separators
        for menu in self.MenuBar.children():
            if (type(menu) is QMenu):
                self.__hideEmptyMenus(menu) # hide empty toolbars
        self.__hideToolbarIfEmpty(self.toolBar_Standard)
        self.__hideToolbarIfEmpty(self.toolBar_Viewpoints)
        self.__hideToolbarIfEmpty(self.toolBar_snap)
        self.__hideToolbarIfEmpty(self.toolBar_Presenter)


    def openInitialDialog(self):
        _infoer.function = str(self.openInitialDialog)
        _infoer.write("")
        acceptedOrRejected = QDialog.Accepted

        # check language license     
        languageID = vtrans.languageLocale.upper().partition(".")[0]       

        projectToLoad = os.getenv('VR_PREPARE_PROJECT')
        if projectToLoad :
            projectToLoad = projectToLoad.replace("\"", "")
            # project file from command line
            if projectToLoad.lower().endswith(".coprj") :
                projectName = str(QtCore.QFileInfo(projectToLoad).baseName())
                # cyberclassroom license check
                if projectName[0:3]=="CC_":
                    project = projectName.upper()
                self.__fileOpen( projectToLoad, False, False)
                if os.getenv('VR_PREPARE_RUN_PRESENTATION'):
                    self.presenterManager.play()
                return QDialog.Accepted
            # cocase file from command line
            elif projectToLoad.lower().endswith(".cocase") :
                self.fileWizard(projectToLoad)
                return QDialog.Accepted
            # vrml file from command line
            elif projectToLoad.lower().endswith(".wrl") :
                if os.access(projectToLoad, os.R_OK):
                    # create new empty project
                    ObjectMgr.ObjectMgr().initProject()
                    # load vrml
                    ObjectMgr.ObjectMgr().importFile(projectToLoad)
                    return QDialog.Accepted
            # covise file from command line
            elif projectToLoad.lower().endswith(".covise") :
                if os.access(projectToLoad, os.R_OK):
                    # create new empty project
                    ObjectMgr.ObjectMgr().initProject()
                    # load covise
                    ObjectMgr.ObjectMgr().importFile(projectToLoad)
                    return QDialog.Accepted
            else :
                #print("\nWARNING: Can not open file", projectToLoad)
                # pop up warning dialog
                dialog = QErrorMessage(Application.vrpApp.mw)
                string = self.__tr("WARNING: Can not open file %s") % projectToLoad
                dialog.showMessage(string)
                dialog.setWindowTitle(self.__tr("Can not open file"))
                acceptedOrRejected = dialog.exec_()
        # No coprj, cocase, .. given
        # Create empty project
        ObjectMgr.ObjectMgr().initProject()
        return acceptedOrRejected

    #def __sigFileNew(self):
        #_infoer.function = str(self.__sigFileNew)
        #_infoer.write("")
        #self.fileNew() # we cant connect fileNew directly to SIGNAL(newProjectRequest) because the parameter cocaseFile will get some value from the qt mechanism

    def fileNew(self):
        _infoer.function = str(self.fileNew)
        _infoer.write("")
        asker = FileNewAsker(self)
        decicion = asker.exec_()
        if decicion == QDialog.Rejected:
            return
        self.__currentProjectFilename = ''
        ObjectMgr.ObjectMgr().initProject()

    def fileWizard(self, cocaseFile = None):
        _infoer.function = str(self.fileNew)
        _infoer.write("cocaseFile %s" %(cocaseFile))
        dialog = StartPreparation(self, cocaseFile)
        ignore = dialog.exec_()

    def fileProjectInformation(self):
        _infoer.function = str(self.fileProjectInformation)
        _infoer.write("")
        dialog = ProjectInformation(self)
        ignore = dialog.exec_()

    def exportMap(self):
        _infoer.function = str(self.exportMap)
        _infoer.write("")
        theNet().save("test.net")
        self.statusBar().showMessage(self.__tr('Saved Covise Map as test.net'))

    def fileOpen(self):
        _infoer.function = str(self.fileOpen)
        _infoer.write("")
#        asker = FileNewAsker(self)
#        decicion = asker.exec_()
#        if decicion == QtWidgets.QDialog.Rejected:
#            return
        self.__fileOpen( None, False, False)

    def fileAdd(self):
        _infoer.function = str(self.fileAdd)
        _infoer.write("")
        self.__fileOpen( None, True, False)

    def fileCompare(self):
        _infoer.function = str(self.fileCompare)
        _infoer.write("")
        self.__fileOpen( None, True, True)

    def __fileOpen(self, fn=None, addToCurrent=False, autoSync=False):
        _infoer.function = str(self.__fileOpen)
        _infoer.write("fileName %s addToCurrent %s autoSync %s" %(fn, str(addToCurrent), str(autoSync)))
        filename = " "
        if fn:
            filename = fn
        else:
            # REM pre = Preview(self)
            fd = QFileDialog(self)
            fd.setMinimumWidth(1050)
            fd.setMinimumHeight(700)
            # REM fd.setContentsPreviewEnabled(True)
            # REM fd.setContentsPreview( pre, pre.preview)
            # REM fd.setPreviewMode( QFileDialog.Contents)
            fd.setNameFilter(self.__tr('Project-Files (*.coprj)'))
            fd.setWindowTitle(self.__tr('Open Project'))
            if self.__projectFilenameSuggestion:
                fd.selectFile(self.__projectFilenameSuggestion)

            acceptedOrRejected = fd.exec_()
            if acceptedOrRejected != QDialog.Accepted :
                return

            filenamesQt = fd.selectedFiles()
            if filenamesQt[0] == "":
                return

            filename = str(filenamesQt[0])
            ProductName = covise.getCoConfigEntry("vr-prepare.ProductName")
            if not ProductName:
                ProductName = ""
            if not os.access(filename, os.R_OK):
                QMessageBox.information(
                    self,
                    ProductName,
                    self.__tr("The file \"%s\" is not accessable.\nYou may check the permissions.") % filenamesQt,
                    self.__tr("&Ok"),
                    "",
                    "",
                    0,
                    0)
                return
                
        self.spawnPatienceDialog()
        # delete old project

        if not addToCurrent :
            ObjectMgr.ObjectMgr().deleteProject()
        # load project
        reqId = theGuiMsgHandler().loadObject(filename, addToCurrent, autoSync, self.errorCallback, [], -1)
        theGuiMsgHandler().waitforAnswer(reqId)

        #self.unSpawnPatienceDialog()
        #if os.getenv('VR_PREPARE_CLOSE_AFTER_LOADING') and os.getenv('VR_PREPARE_DEBUG_VISITEMS_DIR'):
        #    theGuiMsgHandler().sendExit()

        if not addToCurrent:
            self.__setCaptionFromFilename(filename)
            self.__projectFilenameSuggestion = filename
            self.__currentProjectFilename = filename
        self.WidgetStackRight.setCurrentWidget(self.noSelectionPanel) 
        self.advice.raiseInStack(self.mainWindowHelp)

    def _changeTreeViewFilter(self, param=None):
        globalAccessToTreeView.setFilter(str(self.filterLineEdit.text()), self.filterSceneGraphItems.isChecked())
        self.clearFilterButton.setVisible(str(self.filterLineEdit.text()) != "" or self.filterSceneGraphItems.isChecked())

    def _clearTreeViewFilter(self):
        self.filterLineEdit.setText("")
        self.filterSceneGraphItems.setChecked(False)

    def errorCallback(self, requestNr, status, msg):
        _infoer.function = str(self.errorCallback)
        _infoer.write("requestNr %s status %s msg %s" %(str(requestNr), str(status), msg))
        if status == NO_ERROR:
            return
        elif status == WRONG_PATH_ERROR:
            print("ERROR: The file", msg.wrongFileName, "was not found. Asking user for new location.")
            correctedPath = None

            # hide patience dialog
            self.unSpawnPatienceDialog()

            # ask if user wants to change path
            asker = ChangePathAsker(msg.wrongFileName, None)
            decision = asker.exec_()
            if decision == QDialog.Accepted:
                if asker.pressedYes():
                    # call path browser
                    correctedPath = QFileDialog.getExistingDirectory(self, self.__tr('Choose path'), os.path.dirname(msg.wrongFileName))
                    if correctedPath == "":
                        # close the application if no path was chosen
                        self.close()
                    self.spawnPatienceDialog()
                else:
                    # close the application if user does not want to choose a new path
                    self.close()
            else:
                self.close()

            # just return the new path
            return str(correctedPath)
        elif status == TIMESTEP_ERROR:
            # hide patience dialog
            self.unSpawnPatienceDialog()

            # raise the asker dialogue
            asker = ReduceTimestepAsker()
            decision = asker.exec_()
            # no cancel pressed
            if decision == QDialog.Accepted:
                if asker.pressedYes():
                    self.spawnPatienceDialog()
                    return True
                self.spawnPatienceDialog()
                return False
        else:
            return


    def importDataset(self):
        _infoer.function = str(self.importDataset)
        _infoer.write("")
        filetypes = getImportFileTypes()
        initialName = self.__projectFilenameSuggestion
        if not initialName:
            initialName = ""
        filenamesQt = [QFileDialog.getOpenFileName(
            self,
            self.__tr('Import Dataset'),
            initialName,
            filetypes
            )]
        # Disabled multi-selection for now because it causes problems with the vrml startIndex.
        # The startIndex for the next vrml files arrives too late (the modules are already initialized).
#        filenamesQt = QtWidgets.QFileDialog.getOpenFileNames(
#            self,
#            self.__tr('Import Dataset'),
#            initialName,
#            filetypes
#            )
        for filenameQt in filenamesQt:
            filename = str(filenameQt[0])
            print(filename)
            load = True
            ProductName = covise.getCoConfigEntry("vr-prepare.ProductName")
            if not ProductName:
                ProductName = ""
            if not os.access(filename, os.R_OK):
                QMessageBox.information(
                    self,
                    ProductName,
                    self.__tr("The file \"%s\" is not accessable.\nYou may check the permissions.") % filenameQt,
                    self.__tr("&Ok"),
                    "",
                    "",
                    0,
                    0)
                load = False
            if load:
                self.importRequest(filename)

    def importRequest(self, filename):
        if filename.lower().endswith('.cocase'):
            filterDatasetsWidget = CasesFilter(self)
            namedCase = coviseCase.NameAndCoviseCase()
            namedCase.setFromFile(filename)
            dsc = coviseCase.coviseCase2DimensionSeperatedCase(
                namedCase.case, namedCase.name, os.path.dirname(filename))
            filterDatasetsWidget.addDimensionSeperatedCase(dsc)
            filterDatasetsWidget.pushButtonOK.show()
            filterDatasetsWidget.pushButtonCancel.show()
            decision = filterDatasetsWidget.exec_()
            if decision == QDialog.Accepted:
                ObjectMgr.ObjectMgr().importCases(filterDatasetsWidget.getChoice())
        else:
            ObjectMgr.ObjectMgr().importFile(filename)

    def fileSaveAs(self):
        _infoer.function = str(self.fileSaveAs)
        _infoer.write("")
        filenameQt = QFileDialog.getSaveFileName(
            self,
            self.__tr('Save Project'),
            self.__projectFilenameSuggestion,
            self.__tr('Project-Files (*.coprj)'),
            None,
            QFileDialog.DontConfirmOverwrite)
        if filenameQt == "":
            return False
        filename = str(filenameQt[0])
        if not filename.lower().endswith(".coprj"):
            filename += ".coprj"
            _infoer.function = str(self.fileSaveAs)
            _infoer.write(
                'Using filename "' + filename + '" for save.  Appended "'
                + ".coprj" + '".')
            _infoer.reset()
        if  os.path.exists(filename):
            asker = ReallyWantToOverrideAsker(self, filename)
            decicion = asker.exec_()
            if decicion == QDialog.Rejected:
                self.statusBar().showMessage(self.__tr('Cancelled overwrite of "%s"') % filename)
                return False
        self.save( filename )
        self.statusBar().showMessage(self.__tr('Wrote "%s"') % filename)
        self.__setCaptionFromFilename(filename)
        self.__projectFilenameSuggestion = filename
        self.__currentProjectFilename = filename
        return True

    def fileSave(self):
        _infoer.function = str(self.fileSave)
        _infoer.write("")
        if self.__currentProjectFilename:
            self.save(self.__currentProjectFilename)
            self.statusBar().showMessage(
                self.__tr('Wrote "%s"') % self.__currentProjectFilename)
            return True
        else:
            return self.fileSaveAs()

    def save( self, filename ):
        _infoer.function = str(self.save)
        _infoer.write("filename %s" %(filename))
        assert filename.lower().endswith('')
        # 0 is the project_key
        theGuiMsgHandler().saveObject(0, filename)

    def fileExit(self):
        _infoer.function = str(self.fileExit)
        _infoer.write("")
        
        import Start
        asker = SaveBeforeExitAsker(self)
        decision = asker.exec_()
        if decision == QDialog.Accepted:
            if asker.pressedSave():
                if self.fileSave():
                    Start.qapp.exit()
            else :
                Start.qapp.exit()

    def editUndo(self):
        _infoer.function = str(self.editUndo)
        _infoer.write("")
        params = Application.vrpApp.globalJournalMgrParams
        params.currentIdx = params.currentIdx-1
        if params.currentIdx==params.maxIdx:
            self.editRedoAction.setEnabled(False)
            self.editUndoAction.setEnabled(True)
        else:
            self.editUndoAction.setEnabled(True)
            self.editRedoAction.setEnabled(True)
        if params.currentIdx<=0:
            self.editUndoAction.setEnabled(False)
        theGuiMsgHandler().setParams( Application.vrpApp.globalJournalMgrKey, params)

    def editRedo(self):
        _infoer.function = str(self.editRedo)
        _infoer.write("")
        params = Application.vrpApp.globalJournalMgrParams
        params.currentIdx = params.currentIdx+1

        if params.currentIdx==params.maxIdx:
            self.editRedoAction.setEnabled(False)
            self.editUndoAction.setEnabled(True)
        else:
            self.editUndoAction.setEnabled(True)
            self.editRedoAction.setEnabled(True)
        if params.currentIdx<=0:
            self.editUndoAction.setEnabled(False)
        theGuiMsgHandler().setParams( Application.vrpApp.globalJournalMgrKey, params)

    def restartRenderer(self):
        _infoer.function = str(self.restartRenderer)
        _infoer.write("")
        theGuiMsgHandler().restartRenderer()

    def navigationModeNone(self):
        self.navigationMode_NoneAction.setChecked(True)       # NOTE: This should not be the final solution.
        self.navigationMode_TransformAction.setChecked(False) #       We might get out of sync with OpenCOVER.  
        self.navigationMode_MeasureAction.setChecked(False)   #       OpenCOVER should send the NavMode to the GUI.
        self.trackingManager.setNavigationMode("NavNone")

    def navigationModeTransform(self):
        self.navigationMode_NoneAction.setChecked(False)      # NOTE: This should not be the final solution.
        self.navigationMode_TransformAction.setChecked(True)  #       We might get out of sync with OpenCOVER. 
        self.navigationMode_MeasureAction.setChecked(False)   #       OpenCOVER should send the NavMode to the GUI.
        self.trackingManager.setNavigationMode("XForm")

    def navigationModeMeasure(self):
        self.navigationMode_NoneAction.setChecked(False)      # NOTE: This should not be the final solution.
        self.navigationMode_TransformAction.setChecked(False) #       We might get out of sync with OpenCOVER. 
        self.navigationMode_MeasureAction.setChecked(True)    #       OpenCOVER should send the NavMode to the GUI.
        self.trackingManager.setNavigationMode("Measure")

    def toggleSceneEditorGrid(self):
        msg = coGRKeyWordMsg("toggleGrid", True)
        covise.sendRendMsg(msg.c_str())

    def toggleSceneEditorOperatingRange(self):
        msg = coGRKeyWordMsg("toggleOperatingRange", True)
        covise.sendRendMsg(msg.c_str())

    def enablePresenter(self, b):
        _infoer.function = str(self.enablePresenter)
        _infoer.write("%s" %(b))
        self.presentationDelete_Presentation_StepAction.setEnabled(b)
        #Play will crash vr-prepare, so hide it
        self.presentationNavigationPlayAction.setEnabled(False)
        self.presentationNavigationStopAction.setEnabled(b)
        self.presentationNavigationForwardAction.setEnabled(b)
        self.presentationNavigationBackAction.setEnabled(b)
        self.presentationNavigationGo_To_EndAction.setEnabled(b)
        self.presentationNavigationGo_To_StartAction.setEnabled(b)

    def addViewpoint(self, fromPresentation=False):
        _infoer.function = str(self.addViewpoint)
        _infoer.write("fromPresentation %s" %(str(fromPresentation)))
        self.viewpointManager.new(fromPresentation)

    def enableViewpoint(self, b):
        _infoer.function = str(self.enableViewpoint)
        _infoer.write("%s" %(b))
        self.viewpointsDelete_ViewpointAction.setEnabled(b)
        self.viewpointsChange_ViewpointAction.setEnabled(b)

    def requestForGeometryObjectsGlobal(self):
        _infoer.function = str(self.requestForGeometryObjectsGlobal)
        _infoer.write("")
#        dialog = DialogChangeObjectSettingsBase(
#            self, 'DialogChangeObjectSettingsBase-dialog')
#        dialogReturnValue = dialog.exec_()
        # TODO: Do the right things when dialog has
        # been closed.  E.g. cancel, accept.

    def editColorMaps(self, callerKey, colorTableKey):
        _infoer.function = str(self.editColorMaps)
        _infoer.write("callerKey %s colorTableKey %s" %(str(callerKey), str(colorTableKey)))
        globalColorManager.setCallerKey(callerKey)
        globalColorManager.setColorTableKey(colorTableKey)
        dialogReturnValue = globalColorManager.exec_()
        # TODO: Do the right things when dialog has
        # been closed.  E.g. cancel, accept.

    def showGettingStartedWindow(self):
        dialog = GettingStartedWindow(self)
        dialog.exec_()

    def showPresenterWindow(self, show):
        _infoer.function = str(self.showPresenterWindow)
        _infoer.write("show %s" %(str(show)))
        if show:
            self.presenterManager.show()
            self.presenterManager.activateWindow()
        else:
            self.presenterManager.hide()

    def showViewpointWindow(self, show):
        _infoer.function = str(self.showViewpointWindow)
        _infoer.write("show %s" %(str(show)))
        if show:
            self.viewpointManager.show()
            self.viewpointManager.activateWindow()
        else:
            self.viewpointManager.hide()

    def showAdviceWindow(self, show):
        _infoer.function = str(self.showAdviceWindow)
        _infoer.write("show %s" %(str(show)))
        if show:
            self.advice.show()
            self.advice.activateWindow()
        else:
            self.advice.hide()
    
    def showVideoCaptureWindow(self, show):
        _infoer.function = str(self.showVideoCaptureWindow)
        _infoer.write("show %s" %(str(show)))
        if show:
            self.videoCaptureManager.show()
            self.videoCaptureManager.activateWindow()
        else:
            self.videoCaptureManager.hide()

    def showAnimationWindow(self, show):
        _infoer.function = str(self.showAnimationWindow)
        _infoer.write("show %s" %(str(show)))
        if show:
            self.animationManager.show()
            self.animationManager.activateWindow()
        else:
            self.animationManager.hide()

    def showTrackingWindow(self, show):
        _infoer.function = str(self.showTrackingWindow)
        _infoer.write("show %s" %(str(show)))
        if show:
            self.trackingManager.show()
            self.trackingManager.activateWindow()
        else:
            self.trackingManager.hide()
            
    def showSceneView(self, show):
        _infoer.function = str(self.showSceneView)
        _infoer.write("show %s" %(str(show)))
        if not self.QDockWidgetTreeView: 
            return
        if show:
            self.QDockWidgetTreeView.show()
            self.QDockWidgetTreeView.activateWindow()
        else:
            self.QDockWidgetTreeView.hide()
            
    def setShowSceneView(self, show):
        _infoer.function = str(self.setShowSceneView)
        _infoer.write("show %s" %(str(show)))
        if not self.QDockWidgetTreeView: 
            return
        self.windowScene_ViewAction.setChecked(self.QDockWidgetTreeView.isVisible())

    def syncPartOptions(self):
        _infoer.function = str(self.syncPartOptions)
        _infoer.write("")
        globalSyncManager.setCallerKeys(globalAccessToTreeView.getSelectedKeys()) 
        dialogReturnValue = globalSyncManager.exec_()
        
    def setSelectionOfTreeView(self, key, selection):
        globalAccessToTreeView.setItemSelected(key, False, selection)
        globalAccessToTreeView.sigSelectionChanged.emit()

    def cropAllParts(self):
        _infoer.function = str(self.cropAllParts)
        _infoer.write("")
#        cropMin = globalKeyHandler().getObject(globalProjectKey).getCropMin()
#        cropMax = globalKeyHandler().getObject(globalProjectKey).getCropMax()
        params = ObjectMgr.ObjectMgr().getParamsOfObject(0)

        box = globalKeyHandler().getObject(globalProjectKey).getPartsBoundingBox()

        asker = CroppingManager(self)

        # set slider values
        asker.minimumX.setRange(box.getXMinMax())
        asker.minimumY.setRange(box.getYMinMax())
        asker.minimumZ.setRange(box.getZMinMax())
        asker.maximumX.setRange(box.getXMinMax())
        asker.maximumY.setRange(box.getYMinMax())
        asker.maximumZ.setRange(box.getZMinMax())

        asker.minimumX.setValue(params.cropMin[0])
        asker.minimumY.setValue(params.cropMin[1])
        asker.minimumZ.setValue(params.cropMin[2])
        asker.maximumX.setValue(params.cropMax[0])
        asker.maximumY.setValue(params.cropMax[1])
        asker.maximumZ.setValue(params.cropMax[2])

        decision = asker.exec_()

        if decision == QDialog.Accepted and asker.pressedYes():
            newMin = [asker.minimumX.getValue(), asker.minimumY.getValue(), asker.minimumZ.getValue()]
            newMax = [asker.maximumX.getValue(), asker.maximumY.getValue(), asker.maximumZ.getValue()]

            reqId = theGuiMsgHandler().setCropMinMax(newMin[0], newMin[1], newMin[2], newMax[0], newMax[1], newMax[2], None)
            theGuiMsgHandler().waitforAnswer(reqId)

            params.cropMin = newMin
            params.cropMax = newMax
            ObjectMgr.ObjectMgr().setParams(0, params)

    def duplicateParts(self):
        _infoer.function = str(self.duplicateParts)
        _infoer.write("")
        selectedParts = globalAccessToTreeView.getSelectedKeys()
        if (len(selectedParts) == 0):
            return
        typeNr = Application.vrpApp.key2type[selectedParts[0]]
        if (typeNr in [VIS_STREAMLINE, VIS_MOVING_POINTS, VIS_PATHLINES, VIS_ISOPLANE, VIS_VECTORFIELD, VIS_PLANE, VIS_VECTOR]):
            for partKey in selectedParts:
                ObjectMgr.ObjectMgr().duplicateVisualizer(partKey)
        elif (typeNr in [TYPE_2D_PART, TYPE_3D_PART]):
            # raise the asker dialogue
            asker = DialogDuplicateAsker(self)
            decision = asker.exec_()
            # no cancel pressed
            if decision == QDialog.Accepted:
                for partKey in selectedParts:
                    angle = 0.0
                    transX = 0.0
                    transY = 0.0
                    transZ = 0.0
                    for i in range(asker.number):        
                        angle = angle + asker.angle
                        transX = transX + asker.transX
                        transY = transY + asker.transY
                        transZ = transZ + asker.transZ
                        ObjectMgr.ObjectMgr().duplicatePart(partKey, asker.axisX, asker.axisY, asker.axisZ, angle, transX, transY, transZ)

    def cutParts(self):
        _infoer.function = str(self.cutParts)
        _infoer.write("")
        parts = globalAccessToTreeView.getSelectedKeys()
        for partKey in parts:
            theGuiMsgHandler().requestObject( TYPE_2D_CUTGEOMETRY_PART, None, partKey )
            # hide original part
            params = ObjectMgr.ObjectMgr().getParamsOfObject(partKey)
            params.isVisible = False
            if partKey in Application.vrpApp.guiKey2visuKey:
                key = Application.vrpApp.guiKey2visuKey[partKey]
            else:
                key = partKey
            ObjectMgr.ObjectMgr().setParams(key, params)

    def deleteParts(self):
        _infoer.function = str(self.deleteParts)
        _infoer.write("")
        parts = globalAccessToTreeView.getSelectedKeys()
        for partKey in parts:
            ObjectMgr.ObjectMgr().deleteObject(partKey)

    def showParts(self):
        _infoer.function = str(self.showParts)
        _infoer.write("")
        parts = globalAccessToTreeView.getSelectedKeys()
        for partKey in parts:
            params = ObjectMgr.ObjectMgr().getParamsOfObject(partKey)
            params.isVisible = True
            if partKey in Application.vrpApp.guiKey2visuKey:
                key = Application.vrpApp.guiKey2visuKey[partKey]
            else:
                key = partKey
            ObjectMgr.ObjectMgr().setParams(key, params)

    def hideParts(self):
        _infoer.function = str(self.hideParts)
        _infoer.write("")
        parts = globalAccessToTreeView.getSelectedKeys()
        for partKey in parts:
            params = ObjectMgr.ObjectMgr().getParamsOfObject(partKey)
            params.isVisible = False
            if partKey in Application.vrpApp.guiKey2visuKey:
                key = Application.vrpApp.guiKey2visuKey[partKey]
            else:
                key = partKey
            ObjectMgr.ObjectMgr().setParams(key, params)
            
    def createSceneObjectList(self):
        _infoer.function = str(self.createSceneObjectList)
        _infoer.write("")
        dialog = SceneObjectListWindow(self)
        dialog.exec_()


    def composedGridRequest(self):
        _infoer.function = str(self.composedGridRequest)
        _infoer.write("")
        assert self.lastClickedKey
        # params ready for immediate setting after construction.
        reqId = theGuiMsgHandler().requestObject( TYPE_3D_COMPOSED_PART, parentKey = self.lastClickedKey)
        theGuiMsgHandler().waitforAnswer(reqId)

    def composedPartRequest(self):
        _infoer.function = str(self.composedPartRequest)
        _infoer.write("")
        assert self.lastClickedKey
        # params ready for immediate setting after construction.
        reqId = theGuiMsgHandler().requestObject( TYPE_2D_COMPOSED_PART, parentKey = self.lastClickedKey)
        theGuiMsgHandler().waitforAnswer(reqId) 

    def setCaption(self, aString):
        _infoer.function = str(self.setCaption)
        _infoer.write("%s" %(aString))

        ProductName = covise.getCoConfigEntry("vr-prepare.ProductName")
        if not ProductName:
            ProductName = ""

        MainWindowBase.setWindowTitle(
            self, str(aString) + ' - ' + ProductName)

    def raisePanelForKey(self, key, showHidden=False, selectInTree=True):
        _infoer.function = str(self.raisePanelForKey)
        _infoer.write("key %s" %(str(key)))
        if (not showHidden) and self.isPanelHiddenForObject(key):
            return
        typeNr = Application.vrpApp.key2type[key]
        if typeNr in self.__panelForType:
            self.__panelForType[typeNr].updateForObject(key)
        #elif typeNr == TYPE_PRESENTATION:
        #    globalPresenterManager.updateForObject(key)
        self.__raisePanelForType(typeNr)
        # todo check
        if selectInTree:
            globalAccessToTreeView.setItemSelected(key)
            self.lastClickedKey = key

    def getPanelForKey(self, key):
        _infoer.function = str(self.getPanelForKey)
        _infoer.write("key %s" %(str(key)))
        # returns the panel of the key
        typeNr = Application.vrpApp.key2type[key]
        if typeNr in self.__panelForType:
            self.__panelForType[typeNr].updateForObject(key)
        if typeNr in self.__panelForType:
            return self.__panelForType[typeNr]
#        elif TYPE_CAD_PART==typeNr:
#            return tesselationPanel
        return None
        
    def setStyleSheet(self, sheet):
        #print("**style sheet")
        #self.splitter1.setStyleSheet(sheet)
        #self.FrameRight.setStyleSheet(sheet)
        #self.FrameLeft.setStyleSheet(sheet)
        #self.widget.setStyleSheet(sheet)
        #self.setUpdatesEnabled(False)
        #self.toolBar_Standard.setStyleSheet(sheet)
        #self.toolBar_Viewpoints.setStyleSheet(sheet)
        #self.toolBar_snap.setStyleSheet(sheet)
        #self.toolBar_Presenter.setStyleSheet(sheet)
        QWidget.setStyleSheet(self, sheet)

    def update(self):
        print("update maninwindow")

    #def update(self, rect):
    #    print("update maninwindow 2")

    #def update(self, region):
    #    print("update maninwindow 3"        )
        
    #def update(self, ax, ay, aw, ah):
    #    print("update maninwindow 4"        )
        
    #def updateGeometry(self):
    #    print("updateGeo maninwindow")
        
    #def updateMicroFocus(self):
    #    print("updateMicro maninwindow"        )
        
    #def render(self, target, offset, region, flags):
    #    print("render mw 1")
        
    #def render(self, painter, offset, region, flags):
    #    print("render mw 2")
    
    #def repaint(self):
    #    print("repaint mw 1")
        
    #def repaint(self, x,y,w,h):
    #    print("repaint mw 2")

    #def repaint(self, rec):
    #    print("repaint mw 3"        )
        
    #def resetInputContext(self):
    #    print("reset")
        
    #def event(self, event):
    #    print("event", event)
    #    return QtWidgets.QWidget.event(self, event)
        
    #def paintEvent(self, paintEvent):
    #    print("----- paintEvent mw")
    #    #return QtWidgets.QWidget.paintEvent(self, paintEvent)        

    def updatePanel(self, key):
        _infoer.function = str(self.updatePanel)
        _infoer.write("key %s" %(str(key)))
        #if key == 0:
            # check if dataset is transient --> hide probing point
            #if ObjectMgr.ObjectMgr().getParamsOfObject(0).numTimeSteps > 1:
                #self.__panelForType[TYPE_3D_PART].ProbingPointPushButton.hide()
                #self.__panelForType[TYPE_3D_COMPOSED_PART].ProbingPointPushButton.hide()
                #self.__panelForType[TYPE_2D_PART].ProbingPointPushButton.hide()
                #self.__panelForType[TYPE_2D_COMPOSED_PART].ProbingPointPushButton.hide()
#                self.showAnimationWindow(True)
#            else:
#                self.showAnimationWindow(False)


    def _noItemClicked(self):
        _infoer.function = str(self._noItemClicked)
        _infoer.write("")
        """ If no item is clicked in tree picturePanel is shown"""
        self.WidgetStackRight.setCurrentWidget(self.noSelectionPanel)
        self.advice.raiseInStack(self.mainWindowHelp)

    def _selectionChangedInTreeSlot(self):
        _infoer.function = str(self._selectionChangedInTreeSlot)
        _infoer.write("")
        """
        Slot thought for doing the right after selection changed in tree.
        Workaround for selection with mouse without click.
        """

        if not self.__selectionChangedOrItemClicked():
            return

    def _itemClickedInTreeSlot(self, aKey):
        _infoer.function = str(self._itemClickedInTreeSlot)
        _infoer.write("key %s" %(str(aKey)))
        """
        Slot thought for doing the right after click on an item aKey.
        """

        self.lastClickedKey = aKey

        if not self.__selectionChangedOrItemClicked():
            return

        #use old style for cad-panel (IS THIS REALLY NESCESSARY?)
        #typeNr = Application.vrpApp.key2type[aKey]
        #if typeNr in [ TYPE_CAD_PART ]:
            #theGuiMsgHandler().requestParams(aKey)

    def __selectionChangedOrItemClicked(self):
        _infoer.function = str(self.__selectionChangedOrItemClicked)
        _infoer.write("")
        selectedKeys = globalAccessToTreeView.getSelectedKeys()

        # check if if no part is selected
        if len(selectedKeys) == 0:
            self.WidgetStackRight.setCurrentWidget(self.noSelectionPanel)
            self.advice.raiseInStack(self.mainWindowHelp)
            return False

        firstKey = selectedKeys[0]
        firstType = ObjectMgr.ObjectMgr().getTypeOfObject(firstKey)

        if (len(selectedKeys) > 0) and self.isPanelHiddenForObject(firstKey):
            self.WidgetStackRight.setCurrentWidget(self.noSelectionPanel)
            self.advice.raiseInStack(self.mainWindowHelp)
            return

        # if more than one part is selected, show help for multi selection
        if len(selectedKeys) > 1:
            # check if parts have different types, then show picture panel
            typesAreEqual = True
            for key in selectedKeys:
                if ObjectMgr.ObjectMgr().getTypeOfObject(key) != firstType:
                    typesAreEqual = False
                    break
            if not typesAreEqual:
                self.WidgetStackRight.setCurrentWidget(self.noSelectionPanel)
                self.advice.raiseInStack(self.multiSelectHelp)
                return False

        if len(selectedKeys) == 1 and firstType in [ TYPE_3D_PART, VIS_STREAMLINE, VIS_STREAMLINE_2D, VIS_MOVING_POINTS, \
             VIS_PATHLINES, VIS_PLANE, VIS_VECTOR, VIS_ISOPLANE, VIS_ISOCUTTER, VIS_CLIPINTERVAL, VIS_VECTORFIELD, \
             VIS_MAGMATRACE, TYPE_2D_PART, VIS_DOCUMENT, TYPE_CASE, TYPE_3D_GROUP , TYPE_2D_GROUP, \
             TYPE_3D_COMPOSED_PART, TYPE_2D_COMPOSED_PART, TYPE_2D_CUTGEOMETRY_PART, TYPE_SCENEGRAPH_ITEM, VIS_VRML, \
             TYPE_DNA_ITEM, TYPE_GENERIC_OBJECT, VIS_SCENE_OBJECT, VIS_DOMAINSURFACE] :#VIS_POINTPROBING,
            # one object selected
            self.__panelForType[firstType].updateForObject(firstKey)
            self.__raisePanelForType(firstType)
        elif len(selectedKeys) > 1 and firstType in [ TYPE_3D_PART, TYPE_2D_PART, TYPE_2D_CUTGEOMETRY_PART, TYPE_SCENEGRAPH_ITEM, TYPE_DNA_ITEM ] :
            # multiple objects selected
            self.__panelForType[firstType].updateForObject(selectedKeys)
            self.__raisePanelForType(firstType)
        else :
            self.WidgetStackRight.setCurrentWidget(self.noSelectionPanel)
            self.advice.raiseInStack(self.mainWindowHelp)

        # set parts actions enabled or disabled
        allow_duplicate_1 = True
        allow_duplicate_2 = True
        allow_cut = True
        allow_delete = True
        for key in selectedKeys:
            typeNr = Application.vrpApp.key2type[key]
            allow_duplicate_1 = allow_duplicate_1 and (typeNr in [TYPE_2D_PART, TYPE_3D_PART])
            allow_duplicate_2 = allow_duplicate_2 and (typeNr in [VIS_STREAMLINE, VIS_MOVING_POINTS, VIS_PATHLINES, VIS_ISOPLANE, VIS_VECTORFIELD, VIS_PLANE, VIS_VECTOR])
            allow_cut = allow_cut and (typeNr in [TYPE_2D_PART, TYPE_2D_CUTGEOMETRY_PART, TYPE_2D_COMPOSED_PART])
            allow_delete = allow_delete and (typeNr != TYPE_SCENEGRAPH_ITEM)
        self.editPartsDuplicate_Selected_PartAction.setEnabled(allow_duplicate_1 ^ allow_duplicate_2)
        self.editPartsCut_Selected_PartAction.setEnabled(allow_cut)
        self.editPartsDelete_Selected_PartAction.setEnabled(allow_delete)

        return True

    def _checkboxToggleSlot(self, aKey, v_state):
        _infoer.function = str(self._checkboxToggleSlot)
        _infoer.write("key %s v_state %s" %(str(aKey), str(v_state)))
        params = ObjectMgr.ObjectMgr().getParamsOfObject(aKey)
        params.isVisible = v_state
        if aKey in Application.vrpApp.guiKey2visuKey:
            key = Application.vrpApp.guiKey2visuKey[aKey]
        else:
            key = aKey
        ObjectMgr.ObjectMgr().setParams(key, params)

    def __raisePanelForType(self, typeNr):
        _infoer.function = str(self.__raisePanelForType)
        _infoer.write("")
        if TYPE_3D_GROUP == typeNr:
            if self.__panelForType[TYPE_3D_GROUP].isUseful():
                panelToRaise = self.__panelForType[TYPE_3D_GROUP]
                adviceToRaise = self.__docuForType[TYPE_3D_GROUP]
            else :
                panelToRaise = self.noSelectionPanel
                adviceToRaise = self.noHelp
        elif TYPE_2D_GROUP == typeNr:
            if self.__panelForType[TYPE_2D_GROUP].isUseful():
                panelToRaise = self.__panelForType[TYPE_2D_GROUP]
                adviceToRaise = self.__docuForType[TYPE_2D_GROUP]
            else :
                panelToRaise = self.noSelectionPanel
                adviceToRaise = self.noHelp 
        elif TYPE_CAD_PART==typeNr:
            panelToRaise = self.noSelectionPanel #tesselationPanel
            adviceToRaise = self.noHelp
        else :
            if typeNr in self.__panelForType:
                panelToRaise = self.__panelForType[typeNr]
            else :
                _infoer.function = str(self.__raisePanelForType)
                _infoer.write(
                    'No functional panel to raise for type %s _or_ '
                    'the type is wrong.  '
                    'Raising a non-functional panel.' %
                    KeydObject.nameOfCOType[typeNr])
                _infoer.function = ''
                panelToRaise = self.noSelectionPanel
            if typeNr in self.__docuForType:
                adviceToRaise = self.__docuForType[typeNr]
            else:
                adviceToRaise = self.noHelp

        self.WidgetStackRight.setCurrentWidget(panelToRaise)
        self.advice.raiseInStack(adviceToRaise)

    def __disableBrokenParts(self):
        _infoer.function = str(self.__disableBrokenParts)
        _infoer.write("")
        # Grouping
        self.vrpToolButtonUngroup.hide()
        self.vrpToolButtonInterlink.hide()
        self.vrpToolButtonRemove.hide()
        self.vrpToolButtonGroup.hide()
        # Undo / Redo
        self.editUndoAction.setVisible(False)
        self.editRedoAction.setVisible(False)
        # Sync
        self.editPartsSynchronization_Options_Action.setEnabled(False)

    # hides menu (including submenus) if empty and removes more than one separator in a row
    def __hideEmptyMenus(self, menu):
        totalCount = 0
        sectionCount = 0
        lastSeparator = None
        childMenus = dict([(element.menuAction(), element) for element in menu.children() if type(element) is QMenu])
        for action in menu.actions():
            if (action in childMenus):
                # submenu
                if self.__hideEmptyMenus(childMenus[action]):
                    totalCount = totalCount + 1
                    sectionCount = sectionCount + 1
            elif (action.text() == ""):
                # separator
                if (sectionCount == 0):
                    action.setVisible(False)
                sectionCount = 0
                lastSeparator = action
            else:
                # action
                if action.isVisible():
                    totalCount = totalCount + 1
                    sectionCount = sectionCount + 1
        # remove last separator
        if lastSeparator and (sectionCount == 0):
            lastSeparator.setVisible(False)
        # set menu visible
        menu.menuAction().setVisible(totalCount > 0)
        return menu.menuAction().isVisible()

    # hides toolbar if empty and removes more than one separator in a row
    def __hideToolbarIfEmpty(self, toolbar):
        totalCount = 0
        sectionCount = 0
        lastSeparator = None
        for element in toolbar.children():
            if (type(element) is QToolButton) and (element.objectName() != "qt_toolbar_ext_button"): # action
                action = element.defaultAction()
                if action.isVisible():
                    totalCount = totalCount + 1
                    sectionCount = sectionCount + 1
            elif (type(element) is QWidget): # separator
                if (sectionCount == 0):
                    element.setVisible(False)
                sectionCount = 0
                lastSeparator = element
        # remove last separator
        if lastSeparator and (sectionCount == 0):
            lastSeparator.setVisible(False)
        # set toolbar visible
        if (totalCount == 0):
            toolbar.close()

    def _coxmlButtonClicked(self):
        self.WidgetStackRight.setCurrentWidget(self.noSelectionPanel)

    def _propertiesButtonClicked(self):
        keys = globalAccessToTreeView.getSelectedKeys()
        if len(keys) > 0:
            self.raisePanelForKey(keys[0], showHidden=True)

    def __initProcessInfoDockWindow(self):
        _infoer.function = str(self.__initProcessInfoDockWindow)
        _infoer.write("")
#        self.dockProcessInfo = QtWidgets.QDockWindow(self, 'MainWindow.dockProcessInfo')
#        self.dockProcessInfo.setCaption(self.__tr('Process Information'))
#        self.dockProcessInfo.setResizeEnabled(True)
#        self.dockProcessInfo.setCloseMode(QtWidgets.QDockWindow.Always)
#        self.processInfo = QtWidgets.QTextEdit(self.dockProcessInfo)
#        self.dockProcessInfo.setWidget(self.processInfo)
#        self.addDockWindow(self.dockProcessInfo, QtCore.Qt.DockBottom)
#        self.dockProcessInfo.hide()

    def _contextMenuRequest(self, point):
        _infoer.function = str(self._contextMenuRequest)
        _infoer.write("")
        self.__contextMenuParts.popup(point)

    def __setCaptionFromFilename(self, filename):
        _infoer.function = str(self.__setCaptionFromFilename)
        _infoer.write("")
        assert filename.lower().endswith(".coprj")
        s = os.path.basename(filename)
        self.setWindowTitle(s[0:s.find(".coprj")])

    def __tr(self,s,c = None):
        return coTranslate(s)

    def spawnPatienceDialog(self):
        _infoer.function = str(self.spawnPatienceDialog)
        _infoer.write("")
        global globalPdmForOpen
        if globalPdmForOpen==None:
            globalPdmForOpen = PatienceDialogManager(self)
        globalPdmForOpen.spawnPatienceDialog()

    def unSpawnPatienceDialog(self):
        _infoer.function = str(self.unSpawnPatienceDialog)
        _infoer.write("")
        if globalPdmForOpen:
            #if os.getenv('VR_PREPARE_CLOSE_AFTER_LOADING')!=None:
            #    theGuiMsgHandler().sendExit()
            globalPdmForOpen.unSpawnPatienceDialog()
            
    def raiseOkayDialog(self, text):
        # raise the asker dialogue
        asker = OkayAsker(self, text)
        decision = asker.exec_()

    def close(self): # closes the application without showing a warning (save dialog)
        _infoer.function = str(self.close)
        _infoer.write("")
        self.unSpawnPatienceDialog()
        self.__forceClose = True # prevents MainWindow.closeEvent to show the Save-Dialog
        ret = MainWindowBase.close(self)
        theGuiMsgHandler().sendExit()
        return ret

    def closeEvent(self, closeEvent):
        if not self.__forceClose:
            closeEvent.ignore()
            self.fileExit()

    def globalAccessToTreeView(self):
        global globalAccessToTreeView
        return globalAccessToTreeView

    def globalColorManager(self):
        global globalColorManager
        return globalColorManager

    def globalSyncManager(self):
        global globalSyncManager
        return globalSyncManager

    def globalPdmForOpen(self):
        global globalPdmForOpen
        return globalPdmForOpen

    def isPanelHiddenForObject(self, key):
        typeNr = Application.vrpApp.key2type[key]
        if VIS_SCENE_OBJECT==typeNr:
            params = ObjectMgr.ObjectMgr().getParamsOfObject(key)
            if covise.getCoConfigEntry("vr-prepare.Coxml.InitialFile") != None and covise.getCoConfigEntry("vr-prepare.Coxml.InitialFile") in params.filename:
                # dont show properties dialog for InitialFile (Room)
                return True
        return False


def close():
    global globalPdmForOpen
    if globalPdmForOpen:
        globalPdmForOpen.unSpawnPatienceDialog()
    Application.vrpApp.mw.close()
    theGuiMsgHandler().sendExit()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QMainWindow()
    ui = MainWindow(Form)
    # ui.setupUi(Form)
    ui.show()
    sys.exit(app.exec_())

# eof
