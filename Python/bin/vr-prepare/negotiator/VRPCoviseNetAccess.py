
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH

import math
import os
import os.path

from coviseModuleBase import net, CoviseModule
from ErrorManager import CoviseFileNotFoundError
import covise
from Utils import getExistingFilename

from coPyModules import                         \
     OpenCOVER,                                 \
     RWCovise,                                  \
     TracerComp,                                \
     BlockCollect,                              \
     AssembleUsg,                               \
     Transform,                                \
     ReduceSet,                                 \
     CropUsg,                                   \
     GetSubset,                                 \
     CutGeometry

_net = None
_globalRenderer = None
_newPath = None

def theNet():
    global _net
    if None == _net: _net = net()
    return _net

def saveExecute(aCoviseModule):
    # TODO: clarify: is this function really needed?
    assert isinstance(aCoviseModule,CoviseModule), 'saveExecute called with no CoviseModule'
    aCoviseModule.execute()
    if( globalRenderer().isAlive() ): 
        if not theNet().finishedBarrier():
            globalRenderer().died()
    else: print("saveExecute without a renderer -> normal execute")
    
def connect(ptA, ptB):
    # print("connect a: %s b: %s " % (ptA.port, ptB.port))
    theNet().connect(ptA.module, ptA.port,
                     ptB.module, ptB.port)

def disconnect(ptA, ptB):
    # print("disconnect a: %s b: %s " % (ptA.port, ptB.port))
    theNet().disconnect(ptA.module, ptA.port,
                        ptB.module, ptB.port)


def prependBounds(value):
    return value - 1, value + 1, value


#------------------------------
# function is called if the path in coproject-file is not correct
# calls dialog and filebrowser
#------------------------------
"""
def changePath(filename, name):
    global _newPath
    # if widget for openining is active -> close 
    if VRPMainWindow.globalPdmForOpen:
        VRPMainWindow.globalPdmForOpen.unSpawnPatienceDialog()
    # asker asks user if to choose a new path or close the application
    asker = ChangePathAsker(filename,None)
    decision = asker.exec_loop()
    if decision == QDialog.Accepted:
        if asker.pressedYes():
            # call file chooser
            fname = QFileDialog.getExistingDirectory(
                filename, None, 'Choose path', 'Choose Path')  
            if fname == QString.null:
                # close the application if no path is choosed
                VRPMainWindow.close()
            else:
                # save the new selected path to user later
                _newPath = str(fname)
                if VRPMainWindow.globalPdmForOpen==None:
                    VRPMainWindow.globalPdmForOpen = PatienceDialogManager()
                VRPMainWindow.globalPdmForOpen.spawnPatienceDialog()
            # the new path is the selected path of the filechooser and the old filename
            filename = str(fname)+name
        else:
            # close the application if user does not want to choose a new path
            VRPMainWindow.close()
        return filename
    return ""
"""

class ConnectionPoint(object):

    """An entity in a covise-net you can connect with.

    Two ConnectionPoints can lead a connection if
    further constraints are fullfilled.

    """

    def __init__(self, module, portname):
        self.__modu = module
        self.__port = portname

    def __str__(self):
        return 'ConnectionPoint(%s, %s)' % (
            self.module.name_ + str(self.module.nr_), self.port)

    def getModule(self):
        return self.__modu

    def getPortname(self):
        return self.__port

    module = property(getModule)
    port = property(getPortname)


class CoviseNetAccessModule(object):

    def __init__(self, module):
        self._module = module
        theNet().add(self._module)

    def getCovModule(self):
        return self._module

    def execute(self):
        saveExecute(self._module)

    def remove(self):
        if hasattr(self, "_module"):
            theNet().remove(self._module)


class RWCoviseModule(CoviseNetAccessModule):

    def __init__(self, filename, doWrite = False):
        CoviseNetAccessModule.__init__(self, RWCovise())
        self.setGridPath(filename, doWrite)

    def setGridPath(self, filename, doWrite = False):
        # check if filename exists
        if not doWrite and (getExistingFilename(filename) == None):
            #raise IOError(filename)
            raise CoviseFileNotFoundError(filename)
            """
            fn = os.path.basename(filename)
            # if a path allready was selected
            if not _newPath == None:
                fname = _newPath + fn
                # test new path
                if not os.access(fname, os.R_OK):
                    filename = changePath(filename, fn)
                else:
                    filename = fname
            else:
                filename = changePath(filename , fn)
            """
        self._module.set_grid_path(filename)
        self.__gridPath = filename

    def gridPath(self):
        return self.__gridPath

    def connectionPoint(self):
        return ConnectionPoint(self._module, 'mesh')

    def inConnectionPoint(self):
        return ConnectionPoint(self._module, 'mesh_in')

    def getCoObjName(self):
        return self._module.getCoObjName('mesh')

class TracerModule(CoviseNetAccessModule):

    def __init__(self):
         CoviseNetAccessModule.__init__(self, TracerComp())

    def gridConnectionPoint(self):
        return ConnectionPoint(self._module, 'meshIn')

    def variableConnectionPoint(self):
        return ConnectionPoint(self._module, 'dataIn')

    def octTreeConnectionPoint(self):
        return ConnectionPoint(self._module, 'octtreesIn')

    def geometryConnectionPoint(self):
        return ConnectionPoint(self._module, 'geometry')

    def set_no_startp(self, value):
        self._module.set_no_startp(*prependBounds(value))

    def set_startpoint1(self, triple):
        self._module.set_startpoint1(*triple)

    def set_startpoint2(self, triple):
        self._module.set_startpoint2(*triple)

    def set_direction(self, triple):
        self._module.set_direction(*triple)

    def set_tdirection(self, value):
        self._module.set_tdirection(value)

    def set_taskType(self, value):
        self._module.set_taskType(value)

    def set_startStyle(self, value):
        self._module.set_startStyle(value)

    def set_trace_len(self, value):
        self._module.set_trace_len(value)

class BlockCollectModule(CoviseNetAccessModule):
    def __init__(self, isTransient=False):
        CoviseNetAccessModule.__init__(self, BlockCollect())
        # set "cat blocks" option
        if not isTransient:
            self._module.set_mode(6)
        else :
            self._module.set_mode(3)

    def objInConnectionPoint(self, i):
        assert ( i>=0 and i<15 ), 'BlockCollectModule::objInConnectionPoint  called with wrong port number %s ' %i
        port = "inport_%s" % i
        return ConnectionPoint(self._module, port)

    def objOutConnectionPoint(self):
        return ConnectionPoint(self._module, 'set_out')

    def getCoObjName(self):
        return self._module.getCoObjName('set_out')

class AssembleUsgModule(CoviseNetAccessModule):
    def __init__(self):
        CoviseNetAccessModule.__init__(self, AssembleUsg())
    def geoInConnectionPoint(self):
        return ConnectionPoint(self._module, 'GridIn0')
    def dataInConnectionPoint(self):
        return ConnectionPoint(self._module, 'DataIn0')
    def geoOutConnectionPoint(self):
        return ConnectionPoint(self._module, 'GridOut0')
    def dataOutConnectionPoint(self):
        return ConnectionPoint(self._module, 'DataOut0')

class CropUsgModule(CoviseNetAccessModule):
    def __init__(self, dataDimension):
        """ dataDimension: 1=SCALAR, 3=VECTOR """
        if not dataDimension in [1,3]:
            assert False, "CropUsgModule: given dataDimension doesn't match"
        self.__dataDimension = dataDimension
        CoviseNetAccessModule.__init__(self, CropUsg())
    def geoInConnectionPoint(self):
        return ConnectionPoint(self._module, 'GridIn0')
    def dataInConnectionPoint(self):
        if self.__dataDimension == 1:
            return ConnectionPoint(self._module, 'DataIn1')
        elif self.__dataDimension == 3:
            return ConnectionPoint(self._module, 'DataIn0')
    def geoOutConnectionPoint(self):
        return ConnectionPoint(self._module, 'GridOut0')
    def dataOutConnectionPoint(self):
        if self.__dataDimension == 1:
            return ConnectionPoint(self._module, 'DataOut1')
        elif self.__dataDimension == 3:
            return ConnectionPoint(self._module, 'DataOut0')

    def setCropMin(self, x, y, z):
        self._module.set_xMin(x)
        self._module.set_yMin(y)
        self._module.set_zMin(z)
    def setCropMax(self, x, y, z):
        self._module.set_xMax(x)
        self._module.set_yMax(y)
        self._module.set_zMax(z)
    def setCrop(self, xMin, yMin, zMin, xMax, yMax, zMax):
        self.setCropMin(xMin, yMin, zMin)
        self.setCropMax(xMax, yMax, zMax)
    def invertCropSide(self, invert):
        self._module.set_invert_crop(str(invert))

class ReduceSetModule(CoviseNetAccessModule):
    def __init__(self):
        CoviseNetAccessModule.__init__(self, ReduceSet())
    def geoInConnectionPoint(self):
        return ConnectionPoint(self._module, 'input_0')
    def dataInConnectionPoint(self):
        return ConnectionPoint(self._module, 'input_1')
    def geoOutConnectionPoint(self):
        return ConnectionPoint(self._module, 'output_0')
    def dataOutConnectionPoint(self):
        return ConnectionPoint(self._module, 'output_1')

    def data0InConnectionPoint(self):
        return ConnectionPoint(self._module, 'input_0')
    def data1InConnectionPoint(self):
        return ConnectionPoint(self._module, 'input_1')
    def data2InConnectionPoint(self):
        return ConnectionPoint(self._module, 'input_2')
    def data3InConnectionPoint(self):
        return ConnectionPoint(self._module, 'input_3')
    def data4InConnectionPoint(self):
        return ConnectionPoint(self._module, 'input_4')
    def data5InConnectionPoint(self):
        return ConnectionPoint(self._module, 'input_5')
    def data6InConnectionPoint(self):
        return ConnectionPoint(self._module, 'input_6')
    def data7InConnectionPoint(self):
        return ConnectionPoint(self._module, 'input_7')
    def data0OutConnectionPoint(self):
        return ConnectionPoint(self._module, 'output_0')
    def data1OutConnectionPoint(self):
        return ConnectionPoint(self._module, 'output_1')
    def data2OutConnectionPoint(self):
        return ConnectionPoint(self._module, 'output_2')
    def data3OutConnectionPoint(self):
        return ConnectionPoint(self._module, 'output_3')
    def data4OutConnectionPoint(self):
        return ConnectionPoint(self._module, 'output_4')
    def data5OutConnectionPoint(self):
        return ConnectionPoint(self._module, 'output_5')
    def data6OutConnectionPoint(self):
        return ConnectionPoint(self._module, 'output_6')
    def data7OutConnectionPoint(self):
        return ConnectionPoint(self._module, 'output_7')

    def setReductionFactor( self, factor):
        self._module.set_factor( factor-1, factor+1, factor)

class GetSubsetModule(CoviseNetAccessModule):
    def __init__(self):
        CoviseNetAccessModule.__init__(self, GetSubset())
    def geoInConnectionPoint(self):
        return ConnectionPoint(self._module, 'DataIn0')
    def dataInConnectionPoint(self):
        return ConnectionPoint(self._module, 'DataIn1')
    def geoOutConnectionPoint(self):
        return ConnectionPoint(self._module, 'DataOut0')
    def dataOutConnectionPoint(self):
        return ConnectionPoint(self._module, 'DataOut1')

    def data0InConnectionPoint(self):
        return ConnectionPoint(self._module, 'DataIn0')
    def data1InConnectionPoint(self):
        return ConnectionPoint(self._module, 'DataIn1')
    def data2InConnectionPoint(self):
        return ConnectionPoint(self._module, 'DataIn2')
    def data3InConnectionPoint(self):
        return ConnectionPoint(self._module, 'DataIn3')
    def data4InConnectionPoint(self):
        return ConnectionPoint(self._module, 'DataIn4')
    def data5InConnectionPoint(self):
        return ConnectionPoint(self._module, 'DataIn5')
    def data6InConnectionPoint(self):
        return ConnectionPoint(self._module, 'DataIn6')
    def data7InConnectionPoint(self):
        return ConnectionPoint(self._module, 'DataIn7')
    def data0OutConnectionPoint(self):
        return ConnectionPoint(self._module, 'DataIn0')
    def data1OutConnectionPoint(self):
        return ConnectionPoint(self._module, 'DataOut1')
    def data2OutConnectionPoint(self):
        return ConnectionPoint(self._module, 'DataOut2')
    def data3OutConnectionPoint(self):
        return ConnectionPoint(self._module, 'DataOut3')
    def data4OutConnectionPoint(self):
        return ConnectionPoint(self._module, 'DataOut4')
    def data5OutConnectionPoint(self):
        return ConnectionPoint(self._module, 'DataOut5')
    def data6OutConnectionPoint(self):
        return ConnectionPoint(self._module, 'DataOut6')
    def data7OutConnectionPoint(self):
        return ConnectionPoint(self._module, 'DataOut7')

    def setSelectionString( self, selection):
        self._module.set_selection(selection)

class CutGeometryModule(CoviseNetAccessModule):
    def __init__(self):
        CoviseNetAccessModule.__init__(self, CutGeometry())
    def geoInConnectionPoint(self):
        return ConnectionPoint(self._module, 'GridIn0')
    def dataInConnectionPoint(self):
        return ConnectionPoint(self._module, 'DataIn0')
    def geoOutConnectionPoint(self):
        return ConnectionPoint(self._module, 'GridOut0')
    def dataOutConnectionPoint(self):
        return ConnectionPoint(self._module, 'DataOut0')
    def setPlane( self, nx, ny, nz, d):
        self._module.set_normal( nx, ny, nz)
        self._module.set_distance(d)

class TransformModule(CoviseNetAccessModule):
    def __init__(self):
        CoviseNetAccessModule.__init__(self, Transform())
    def geoInConnectionPoint(self):
        return ConnectionPoint(self._module, 'geo_in')
    def data0InConnectionPoint(self):
        return ConnectionPoint(self._module, 'data_in0')
    def data1InConnectionPoint(self):
        return ConnectionPoint(self._module, 'data_in1')
    def data2InConnectionPoint(self):
        return ConnectionPoint(self._module, 'data_in2')
    def data3InConnectionPoint(self):
        return ConnectionPoint(self._module, 'data_in3')
    def geoOutConnectionPoint(self):
        return ConnectionPoint(self._module, 'geo_out')
    def data0OutConnectionPoint(self):
        return ConnectionPoint(self._module, 'data_out0')
    def data1OutConnectionPoint(self):
        return ConnectionPoint(self._module, 'data_out1')
    def data2OutConnectionPoint(self):
        return ConnectionPoint(self._module, 'data_out2')
    def data3OutConnectionPoint(self):
        return ConnectionPoint(self._module, 'data_out3')
    def setRotation(self, alpha, x,y,z):
        self._module.set_Transform(4)
        self._module.set_axis_of_rotation(x,y,z)
        self._module.set_angle_of_rotation(alpha)
    def setTranslation(self, x,y,z):
        self._module.set_Transform(3)
        self._module.set_vector_of_translation(x,y,z)

class InventorRendererModule(CoviseNetAccessModule):

    def __init__(self):
        self._module = Renderer()
        theNet().add(self._module, theNet().getLocalHost())

    def connectionPoint(self):
        return ConnectionPoint(self._module, 'RenderData')

class COVERModule(CoviseNetAccessModule):

    def __init__(self, negMsgHandler=None, winId=None):
        self.__winId = winId
        self.__start()
        self.__alive = True
        self.__msgHandler = negMsgHandler
        
    def restart(self):
        rendererRestartPreCommand = covise.getCoConfigEntry("vr-prepare.RendererRestartPreCommand")
        if rendererRestartPreCommand:
            os.system(rendererRestartPreCommand)
        self.__start()
        
    def __start(self):
        self._module = OpenCOVER()
        if covise.coConfigIsOn("vr-prepare.UseCOVERRenderer"):
            from coPyModules import COVER
            self._module = COVER()
                
        theNet().add(self._module, theNet().getLocalHost())
        if not self.__winId==None:
            self._module.set_WindowID(self.__winId)
            #print("********************* set WinId", self.__winId)
        
    def died(self):
        self.setAlive(False)
        if self.__msgHandler:
            self.__msgHandler.sendFinishLoading()
        
    def setAlive( self, isAlive):
        self.__alive = isAlive
        
    def isAlive(self):
        return self.__alive

    def connectionPoint(self):
        return ConnectionPoint(self._module, 'RenderData')


def globalRenderer(msgHandler=None, winId=None):
    global _globalRenderer
    if None == _globalRenderer:
        _globalRenderer = COVERModule(msgHandler, winId)#InventorRendererModule() #
    return _globalRenderer

# eof
