
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH


import copy, math

from VisItem import VisItem
from PartModuleVis import PartModuleVis, PartModuleVisParams
from PartColoredVis import PartColoredVis, PartColoredVisParams
from PartInteractorVis import PartInteractorVis, PartInteractorVisParams

from coPyModules import TracerComp

from Utils import convertAlignedRectangleToGeneral, convertGeneralToAlignedRectangle, RectangleIn3d2Ps1Dir, ParamsDiff, CopyParams, Line3D
from KeydObject import globalKeyHandler, RUN_ALL, VIS_STREAMLINE
from ImportSampleManager import USER_DEFINED, MAX_FLT
from coColorTable import coColorTableParams

from printing import InfoPrintCapable
import os

_infoer = InfoPrintCapable()
_infoer.doPrint =  False # True

class PartTracerVis(PartModuleVis, PartColoredVis, PartInteractorVis):
    """ VisItem to compute traces for 2d or 3d objects """
    def __init__(self):
        if os.getenv('VR_PREPARE_DEBUG_RUN'):
            print("PartTracerVis.__init__")

        PartModuleVis.__init__(self, TracerComp, VIS_STREAMLINE, self.__class__.__name__, ['meshIn'],['octtreesIn'],['SampleGeom'],['SampleData'],['dataIn'],['fieldIn'],['pointsIn'], USER_DEFINED)
        PartColoredVis.__init__(self)
        PartInteractorVis.__init__(self)

        self.params = PartStreamlineVisParams()
        self.__initBase()
        self.oldFreeStartPoints = ''

    def __initBase(self):
        """ __initBase is called from the constructor and after the class was unpickled """
        # store last startpoint from COVER
        self.__startpoint1 = (0,0,0)#None
        self.__startpoint2 = (0,0,0)#None
        self.__firstTime = True

    def PartTracerVisinitBase(self):
        self.__initBase()

    def __init(self, negMsgHandler):
        if os.getenv('VR_PREPARE_DEBUG_RUN'):
            print("PartTracerVis.__init")
            
        """ __init is called from __update

            + start TracerComp module if it does not already exist and connect output to COVER
            + set default parameters of Tracer
            + set default boundingBox which is also used in the GUI to set borders of the float sliders
            + send params to the gui
        """
        if self.__firstTime==True:
            self.__firstTime = False

            # disable sampling if import is transient
            if self.importModule.getIsTransient():
                self._geoSampleNames  = []
                self._dataSampleNames = []

            PartModuleVis._init(self, negMsgHandler, USER_DEFINED)
            PartColoredVis._init(self, negMsgHandler, self._module, 'colorMapIn')
            PartInteractorVis._init(self, negMsgHandler)
            self.__register(negMsgHandler)

            #init params
            self._module.set_taskType(self.params.taskType)

            # spread initial line in between bounding box
            # but not when unpickled or values will be overwritten
            if hasattr(self.params, 'start_style') and self.params.start_style == 1 and not self.fromRecreation:
                self.params.alignedRectangle.setStartEndPoint(self.params.boundingBox.getXMin() / 2.0,
                                                              self.params.boundingBox.getYMin() / 2.0,
                                                              self.params.boundingBox.getZMax(),
                                                              self.params.boundingBox.getXMax() / 2.0,
                                                              self.params.boundingBox.getYMax() / 2.0,
                                                              self.params.boundingBox.getZMax())

            # send params to gui
            self.sendParams()

    def __update(self, negMsgHandler):
        """ __update is called from the run method to update the module parameter before execution
            + do init if the tracer module if necessary
            + update module parameters """
        self.__init(negMsgHandler)
        PartModuleVis._update(self, negMsgHandler)
        PartColoredVis._update(self, negMsgHandler)
        PartInteractorVis._update(self, negMsgHandler)

        if not hasattr(self, 'importModule'):
            return

        #update params
        # no of starting points
        if self.params.no_startp<12:
            min_start = 1
            max_start = 20
        elif self.params.no_startp<100:
            min_start = 1
            max_start = 2*self.params.no_startp
        else:
            min_start = 0.5 * self.params.no_startp
            max_start = 4 * self.params.no_startp
        self._module.set_no_startp( min_start, max_start, self.params.no_startp )
        # other parameters
        if self.params.alignedRectangle.orthogonalAxis=='line':
            s0 = self.params.alignedRectangle.getStartPoint()
            s1 = self.params.alignedRectangle.getEndPoint()
            self._module.set_startpoint1(s0[0], s0[1], s0[2] )
            self._module.set_startpoint2(s1[0], s1[1], s1[2] )
        else :
            aRectangleIn3d2Ps1Dir = convertAlignedRectangleToGeneral( self.params.alignedRectangle)
            self._module.set_startpoint1(*aRectangleIn3d2Ps1Dir.pointA)
            self._module.set_startpoint2(*aRectangleIn3d2Ps1Dir.pointC)
            self._module.set_direction(*aRectangleIn3d2Ps1Dir.direction)
        self._module.set_trace_len(self.params.len)
        self._module.set_trace_eps(self.params.eps)
        self._module.set_trace_abs(self.params.abs)
        self._module.set_min_vel(self.params.min_vel)
        self._module.set_tdirection(self.params.direction)
        self._module.set_grid_tol(self.params.grid_tol)
        self._module.set_maxOutOfDomain(self.params.maxOutOfDomain)

        if PartColoredVis.currentVariable(self) and PartColoredVis.currentVariable(self) in self.params.colorTableKey and globalKeyHandler().getObject(self.params.colorTableKey[self.currentVariable()]).params.mode==coColorTableParams.LOCAL :
            self._module.set_autoScales('TRUE')
        else :
            self._module.set_autoScales('FALSE')

        self._module.setTitle( self.params.name )
        # init params in case of moving points or pathlines
        if hasattr(self.params, 'duration' ):     self._module.set_stepDuration(self.params.duration)
        if hasattr(self.params, 'sphereRadius' ): self._module.set_SphereRadius(self.params.sphereRadius)
        if hasattr(self.params, 'tubeWidth' ): self._module.set_TubeWidth(self.params.tubeWidth)
        if hasattr(self.params, 'numSteps' ): self._module.set_MaxPoints(self.params.numSteps)
        if hasattr(self.params, 'start_style'): self._module.set_startStyle(self.params.start_style)
        if hasattr(self.params, 'freeStartPoints'): 
            if self.params.freeStartPoints == '':
                if hasattr(self, 'oldFreeStartPoints'):
                    self.params.freeStartPoints = self.oldFreeStartPoints
            self._module.set_FreeStartPoints(self.params.freeStartPoints)

    def connectionPoint(self):
        """ return the object to be displayed
            called by the class VisItem """
        return PartModuleVis.connectionPoint(self, 'geometry')

    def getCoObjName(self):
        """ return the generated covise object name
            called by the class VisItem to check if an object name registered by the COVER was created by the tracer
        """
        return PartModuleVis.getCoObjName(self,'geometry')

    def run(self, runmode, negMsgHandler):
        if os.getenv('VR_PREPARE_DEBUG_RUN'):
            print("PartTracerVis.run")
        """ create a new visulisation
            + register for events from Covise if not done yet
            + runmode RUN_GEO and RUN_OCT are ignored
            + update module parameter
            + exec the tracer module
        """
        assert negMsgHandler

        if runmode==RUN_ALL:

            _infoer.function = str(self.run)
            _infoer.write("go")

            if not hasattr(self, 'importModule'):
                return

            self.__update(negMsgHandler)

            PartColoredVis.run(self, runmode, negMsgHandler, self._module, self.fromRecreation)
            PartModuleVis.run(self, runmode, negMsgHandler)
            PartInteractorVis.run(self, runmode, negMsgHandler)



    def __register(self, negMsgHandler):
        """ register to receive events from covise """

        PartModuleVis.register( self, negMsgHandler, [ 'startpoint1', 'startpoint2', 'direction', 'trace_len', 'no_startp',
                             'Min/Max', 'taskType', 'startStyle', 'FreeStartPoints' ] )

    def registerCOVISEkey( self, covise_key):
        """ called during registration if key received from COVER
            + update states in COVER
        """
        (registered, firstTime) = VisItem.registerCOVISEkey( self, covise_key)
        if registered:
            self.sendInteractorStatus()
            self.sendSmokeStatus()
            return (True, firstTime)
        return (False, False)

    def recreate(self, negMsgHandler, parentKey, offset):
        """ recreate is called after all classes of the session have been unpickled """
        self.__initBase()
        self.__correctParams()
        PartInteractorVis.setFormat(self, 0)
        PartModuleVis.recreate(self, negMsgHandler, parentKey, offset, TracerComp, 
                              ['meshIn'],['octtreesIn'],['SampleGeom'],['SampleData'],['dataIn'],['fieldIn'],['pointsIn'] )
        PartColoredVis.recreate(self, negMsgHandler, parentKey, offset)
        PartColoredVis.setColorMap(self,True)

    def __correctParams(self):
        if hasattr(self.params, "freeStartPoints") and type(self.params.freeStartPoints) == tuple:
            self.params.freeStartPoints = '[0.01, 0.01, 0.01]'

    def setParamsByModule( self, mparam, mvalue):
        """ receives parameter changes from the tracer module
            return a list of objKey and their parameters to be set by the Neg2Gui class
        """
        _infoer.function = str(self.setParamsByModule)
        _infoer.write("setParamsByModule ")
        pChangeList = []
        newparams = CopyParams(self.params)
        if mparam == 'startpoint1':
            self.__startpoint1 = mvalue
            if self.params.alignedRectangle.orthogonalAxis=='line':
                newparams.alignedRectangle.setStartEndPoint( float(self.__startpoint1[0]),\
                                                             float(self.__startpoint1[1]),\
                                                             float(self.__startpoint1[2]),\
                                                             float(self.__startpoint2[0]),\
                                                             float(self.__startpoint2[1]),\
                                                             float(self.__startpoint2[2]) )
            else : return pChangeList

            #return pChangeList
        if mparam == 'startpoint2':
            self.__startpoint2 = mvalue
            if self.params.alignedRectangle.orthogonalAxis=='line':
                newparams.alignedRectangle.setStartEndPoint( float(self.__startpoint1[0]),\
                                                             float(self.__startpoint1[1]),\
                                                             float(self.__startpoint1[2]),\
                                                             float(self.__startpoint2[0]),\
                                                             float(self.__startpoint2[1]),\
                                                             float(self.__startpoint2[2]) )
            else : return pChangeList

        if mparam == 'direction':
            if not self.params.alignedRectangle.orthogonalAxis=='line':
                aaa = RectangleIn3d2Ps1Dir()#convertAlignedRectangleToGeneral(self.params.alignedRectangle)
                if self.__startpoint1:
                    aaa.pointA=(float(self.__startpoint1[0]), float(self.__startpoint1[1]), float(self.__startpoint1[2]) )
                if self.__startpoint2:
                    aaa.pointC=(float(self.__startpoint2[0]), float(self.__startpoint2[1]), float(self.__startpoint2[2]) )
                aaa.direction = (float(mvalue[0]), float(mvalue[1]), float(mvalue[2]) )
                xxx=convertGeneralToAlignedRectangle(aaa, self.params.alignedRectangle.orthogonalAxis)
                newparams.alignedRectangle=xxx
                self.__startpoint1=None
                self.__startpoint2=None
        if mparam == 'trace_len':
            newparams.len=float(mvalue[0])
        if mparam == 'no_startp':
            try:
                newparams.no_startp= int(mvalue[2])
            except ValueError:
                newparams.no_startp= int(float(mvalue[2]))
        if mparam == 'taskType':
            newparams.taskType= int(mvalue[0])
        if mparam == 'Min/Max':
            cTableObject = globalKeyHandler().getObject(self.params.colorTableKey[self.currentVariable()])
            if cTableObject.params.baseMin != float(mvalue[0]) or cTableObject.params.baseMax != float(mvalue[1]):
                cTableObject.params.baseObjectName = self.params.name
                cTableObject.params.baseMin = float(mvalue[0])
                cTableObject.params.baseMax = float(mvalue[1])
                if not hasattr(cTableObject.params, 'dependantKeys'):
                    cTableObject.params.dependantKeys = []
                if self.key not in cTableObject.params.dependantKeys:
                    cTableObject.params.dependantKeys.append(self.key)
                pChangeList.append( (cTableObject.key, cTableObject.params) )
        if mparam == 'startStyle':
            newparams.start_style = int(mvalue[0])
        if mparam == 'FreeStartPoints':
            newparams.freeStartPoints = mvalue
            self.oldFreeStartPoints = mvalue

        pChangeList.append( (self.key, newparams) )

        return pChangeList

    def setParams( self, params, negMsgHandler=None, sendToCover=True):
        """ set parameters from outside
            + init tracer module if necessary
            + mainly receive parameter changes from Gui
            + send status messages to COVER if state has changed
        """
        _infoer.function = str(self.setParams)
        _infoer.write("setParams")

        realChange = ParamsDiff( self.params, params )

        PartModuleVis.setParams(self, params, negMsgHandler, sendToCover, realChange)
        PartColoredVis.setParams(self, params, negMsgHandler, sendToCover, realChange)
        PartInteractorVis.setParams(self, params, negMsgHandler, sendToCover, realChange)

        if 'use2DPartKey' in realChange: PartInteractorVis.sendInteractorPosibility(self)


class PartTracerVisParams(PartModuleVisParams, PartColoredVisParams, PartInteractorVisParams):
    def __init__(self):
        PartModuleVisParams.__init__(self)
        PartColoredVisParams.__init__(self)
        PartInteractorVisParams.__init__(self)
        self.isVisible = True
        self.taskType = 1
        self.name = 'PartTracerVisParams'

        #start points from other 2D Part
        self.use2DPartKey = None
        self.len = 1.
        self.no_startp = 50
        self.direction = 1
        self.eps = 0.0000001
        self.grid_tol = 0.0001
        self.maxOutOfDomain = 0.25
        self.abs = 0.0001
        self.min_vel = 0.001
        self.start_style = 2
        self.freeStartPoints = '[0.01, 0.01, 0.01]'

class PartStreamlineVis(PartTracerVis):
    """ VisItem to compute streamlines for 2d or 3d objects """
    def __init__(self):
        PartTracerVis.__init__(self)
        self.params = PartStreamlineVisParams()

class PartStreamlineVisParams(PartTracerVisParams):
    def __init__(self):
        PartTracerVisParams.__init__(self)
        self.name = 'Streamline'
        self.taskType = 1
        self.tubeWidth = 0.0

class PartStreamline2DVis(PartStreamlineVis):
    """ VisItem to compute streamlines for 2d objects """
    def __init__(self):
        PartStreamlineVis.__init__(self)
        #disable sampling
        self._geoSampleNames  = []
        self._dataSampleNames = []
        self.params = PartStreamline2DVisParams()

    def recreate(self, negMsgHandler, parentKey, offset):
        """ overload recreate() since Sampling inputs arent used """
        self.PartTracerVisinitBase()    # call __initBase from PartTracerVis
        PartModuleVis.recreate(self, negMsgHandler, parentKey, offset, TracerComp,
                              ['meshIn'],['octtreesIn'],[],[],['dataIn'],['fieldIn'],['pointsIn'] )
        PartColoredVis.recreate(self, negMsgHandler, parentKey, offset)
        PartColoredVis.setColorMap(self,True)


class PartStreamline2DVisParams(PartStreamlineVisParams):
    def __init__(self):
        PartStreamlineVisParams.__init__(self)
        self.start_style=1
        self.alignedRectangle = Line3D()

class PartMovingPointsVis(PartTracerVis):
    """ VisItem to compute moving particles for 2d or 3d objects """
    def __init__(self):
        PartTracerVis.__init__(self)
        self.params = PartMovingPointsVisParams()
        self.params.name = 'Moving Points'
        self.params.taskType = 2

class PartMovingPointsVisParams(PartTracerVisParams):
    def __init__(self):
        PartTracerVisParams.__init__(self)
        self.numSteps = 50
        self.duration = 0.05
        self.sphereRadius = 0.1

class PartPathlinesVis(PartTracerVis):
    """ VisItem to compute pathlines for 2d or 3d objects """
    def __init__(self):
        PartTracerVis.__init__(self)
        self.params = PartPathlinesVisParams()
        self.params.name = 'Pathlines'
        self.params.taskType = 3


class PartPathlinesVisParams(PartTracerVisParams):
    def __init__(self):
        PartTracerVisParams.__init__(self)
        self.numSteps = 50
        self.duration = 0.05
        self.tubeWidth = 0.0
        self.sphereRadius = 0.2
