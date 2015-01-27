
from PyQt5 import QtCore, QtGui

import math

from Utils import AxisAlignedRectangleIn3d, Line3D
from BoundingBox import Box

#type to use
TRACER = 0
CUTTINGSURFACE = 1
#POINTPROBING = 2

LINE = 1
PLANE = 2
FREE = 3

class RectangleManager(QtCore.QObject):
    """
        general gui class to handle the rectangle in VRPStreamlineBase
        used by different VisItem panels
    """

    def __init__(self, parent, doApply, doApplyRadioBttn, doParamChange, typeRec=TRACER, startStyle = PLANE):
        QtCore.QObject.__init__(self)

        assert parent
        assert doApply

        #slots to be registered by parent
        self.__parent = parent
        self.__apply = doApply
        self.__applyRadioBttn = doApplyRadioBttn
        self.__paramChange = doParamChange
        self.__type = typeRec

        middleFloatInRangeCtrls = [ self.__parent.floatInRangeX, self.__parent.floatInRangeY, self.__parent.floatInRangeZ  ]
        sideFloatInRangeCtrls = []
        rotFloatInRangeCtrls = []
        endPointFloatInRangeCtrls = []

        # a streamlines panel will have these widgets so they will get connected
        if typeRec == TRACER:
            endPointFloatInRangeCtrls = [ self.__parent.floatInRangeEndPointX, self.__parent.floatInRangeEndPointY, self.__parent.floatInRangeEndPointZ  ]

        self.__boundingBox = ((-1, 1), (-1, 1), (-1, 1))
        for w, r in zip(middleFloatInRangeCtrls, self.__boundingBox):
            w.setRange(r)
        #if not typeRec == POINTPROBING:
        if startStyle == PLANE:
            sideRange = 0, self.__heuristicProbeMaxSideLengthFromBox(self.__boundingBox)
            if self.__type == TRACER:
                sideFloatInRangeCtrls = [self.__parent.floatInRangeWidth, self.__parent.floatInRangeHeight]
                for w in sideFloatInRangeCtrls:
                    w.setRange(sideRange)
            rotFloatInRangeCtrls = [ self.__parent.floatInRangeRotX, self.__parent.floatInRangeRotY, self.__parent.floatInRangeRotZ  ]
            for w in rotFloatInRangeCtrls:
                w.setRange((-180.,180.))
        elif startStyle == LINE:
            endPointFloatInRangeCtrls = [ self.__parent.floatInRangeEndPointX, self.__parent.floatInRangeEndPointY, self.__parent.floatInRangeEndPointZ  ]
            for w, r in zip(endPointFloatInRangeCtrls, self.__boundingBox):
                w.setRange(r)
            sideFloatInRangeCtrls = []
            rotFloatInRangeCtrls = []
        #else:
        #    sideFloatInRangeCtrls = []
        #    rotFloatInRangeCtrls = []

        for w in middleFloatInRangeCtrls + sideFloatInRangeCtrls + rotFloatInRangeCtrls + endPointFloatInRangeCtrls:
            w.sigSliderReleased.connect(self.__apply)
            if self.__paramChange:
                w.sigValueChanged.connect(self.__paramChange)

        #if not typeRec == POINTPROBING:
        for w in [self.__parent.xAxisRadioButton,
                self.__parent.yAxisRadioButton,
                self.__parent.zAxisRadioButton]:
            w.clicked.connect(self.__radioButtonClick)

        self.__boundingBox = Box((-1, 1), (-1, 1), (-1, 1))


    def getParams(self, startStyle = PLANE):
        if startStyle == PLANE:
            aar = AxisAlignedRectangleIn3d()
            aar.middle = self.__getMiddle()
            if self.__parent.zAxisRadioButton.isChecked():
                aar.orthogonalAxis = 'z'
            elif self.__parent.yAxisRadioButton.isChecked():
                aar.orthogonalAxis = 'y'
            elif self.__parent.xAxisRadioButton.isChecked():
                aar.orthogonalAxis = 'x'
            else:
                text = (
                    'Invalid orthogonalAxis.  Invalid value is "%s".  '
                    'Expected one out of {"x", "y", "z"}'
                    % str(aar.orthogonalAxis))
                assert False, text
            if self.__type == TRACER:
                aar.lengthA = self.__parent.floatInRangeWidth.getValue()
                aar.lengthB = self.__parent.floatInRangeHeight.getValue()
            elif self.__type == CUTTINGSURFACE:
                aar.lengthA = 1.0
                aar.lengthB = 1.0
            aar.rotX = self.__parent.floatInRangeRotX.getValue()
            aar.rotY = self.__parent.floatInRangeRotY.getValue()
            aar.rotZ = self.__parent.floatInRangeRotZ.getValue()
            
            return aar
        elif startStyle == LINE:
            line = Line3D()

            line.setStartEndPoint(self.__parent.floatInRangeX.getValue(),
                                  self.__parent.floatInRangeY.getValue(),
                                  self.__parent.floatInRangeZ.getValue(),
                                  self.__parent.floatInRangeEndPointX.getValue(),
                                  self.__parent.floatInRangeEndPointY.getValue(),
                                  self.__parent.floatInRangeEndPointZ.getValue())

            return line

    def getBoundingBox(self):
        return self.__boundingBox

    def setBoundingBox( self, box, startStyle = PLANE):
        if self.__boundingBox == box:
            return
        self.__boundingBox = box
        #self.__parent.floatInRangeY.setRange(self.__boundingBox.getYMinMax())
        #self.__parent.floatInRangeZ.setRange(self.__boundingBox.getZMinMax())
        if self.__type==TRACER:
            maxSideLength = self.__boundingBox.getMaxEdgeLength()
            self.__parent.floatInRangeHeight.setRange((0.0, maxSideLength))
            self.__parent.floatInRangeWidth.setRange((0.0, maxSideLength))
        center = self.__boundingBox.getCenter()

        self.__parent.floatInRangeX.setValue(center[0])
        #print "setXRange: "+str(self.__boundingBox.getXMinMax())
        self.__parent.floatInRangeX.setRange(self.__boundingBox.getXMinMax())
        self.__parent.floatInRangeY.setValue(center[1])
        #print "setYRange: "+str(self.__boundingBox.getYMinMax())
        self.__parent.floatInRangeY.setRange(self.__boundingBox.getYMinMax())
        self.__parent.floatInRangeZ.setValue(center[2])
        #print "setZRange: "+str(self.__boundingBox.getZMinMax())
        self.__parent.floatInRangeZ.setRange(self.__boundingBox.getZMinMax())

        if startStyle == LINE:
            self.__parent.floatInRangeEndPointX.setValue(self.__boundingBox.getXMinMax()[0])#center[0])
            #print "setXRange: "+str(self.__boundingBox.getXMinMax())
            self.__parent.floatInRangeEndPointX.setRange(self.__boundingBox.getXMinMax())
            self.__parent.floatInRangeEndPointY.setValue(self.__boundingBox.getYMinMax()[0])#center[1])
            #print "setYRange: "+str(self.__boundingBox.getYMinMax())
            self.__parent.floatInRangeEndPointY.setRange(self.__boundingBox.getYMinMax())
            self.__parent.floatInRangeEndPointZ.setValue(self.__boundingBox.getZMinMax()[0])#center[2])
            #print "setZRange: "+str(self.__boundingBox.getZMinMax())
            self.__parent.floatInRangeEndPointZ.setRange(self.__boundingBox.getZMinMax())


    def setRectangle(self, aar):
        controls = [
            self.__parent.floatInRangeX,
            self.__parent.floatInRangeY,
            self.__parent.floatInRangeZ,
            self.__parent.floatInRangeRotX,
            self.__parent.floatInRangeRotY,
            self.__parent.floatInRangeRotZ,
            self.__parent.xAxisRadioButton,
            self.__parent.yAxisRadioButton,
            self.__parent.zAxisRadioButton]
        for c in controls: c.blockSignals(True)
        self.__parent.floatInRangeX.setValue(aar.middle[0])
        self.__parent.floatInRangeY.setValue(aar.middle[1])
        self.__parent.floatInRangeZ.setValue(aar.middle[2])
        self.__parent.floatInRangeRotX.setValue(aar.rotX)
        self.__parent.floatInRangeRotY.setValue(aar.rotY)
        self.__parent.floatInRangeRotZ.setValue(aar.rotZ)
        if aar.orthogonalAxis == 'z':
            self.__parent.zAxisRadioButton.setChecked(True)
            self.__parent.xAxisRadioButton.setChecked(False)
            self.__parent.yAxisRadioButton.setChecked(False)
        elif aar.orthogonalAxis == 'y':
            self.__parent.yAxisRadioButton.setChecked(True)
            self.__parent.xAxisRadioButton.setChecked(False)
            self.__parent.zAxisRadioButton.setChecked(False)
        elif aar.orthogonalAxis == 'x':
            self.__parent.xAxisRadioButton.setChecked(True)
            self.__parent.zAxisRadioButton.setChecked(False)
            self.__parent.yAxisRadioButton.setChecked(False)
        elif aar.orthogonalAxis == 'line':
            pass
        else:
            text = ('Invalid orthogonalAxis.  Invalid value is "%s"'
                    % str(aar.orthogonalAxis))
            assert False, text
        self.__setElementsEnabled()
        for c in controls: c.blockSignals(False)

        if self.__type==TRACER:
            controls = [self.__parent.floatInRangeHeight,
                        self.__parent.floatInRangeWidth]
            for c in controls: c.blockSignals(True)
            self.__parent.floatInRangeWidth.setValue(aar.lengthA)
            self.__parent.floatInRangeHeight.setValue(aar.lengthB)
            for c in controls: c.blockSignals(False)


    def setLine(self, line):
        controls = [
            #self.__parent.floatInRangeHeight,
            #self.__parent.floatInRangeWidth,
            #self.__parent.floatInRangeRotX,
            #self.__parent.floatInRangeRotY,
            #self.__parent.floatInRangeRotZ,
            #self.__parent.xAxisRadioButton,
            #self.__parent.yAxisRadioButton,
            #self.__parent.zAxisRadioButton,
            self.__parent.floatInRangeX,
            self.__parent.floatInRangeY,
            self.__parent.floatInRangeZ,
            self.__parent.floatInRangeEndPointX,
            self.__parent.floatInRangeEndPointY,
            self.__parent.floatInRangeEndPointZ]
        for c in controls: c.blockSignals(True)

        self.__parent.floatInRangeX.setValue(line.getStartPoint()[0])
        self.__parent.floatInRangeY.setValue(line.getStartPoint()[1])
        self.__parent.floatInRangeZ.setValue(line.getStartPoint()[2])

        self.__parent.floatInRangeEndPointX.setValue(line.getEndPoint()[0])
        self.__parent.floatInRangeEndPointY.setValue(line.getEndPoint()[1])
        self.__parent.floatInRangeEndPointZ.setValue(line.getEndPoint()[2])

        for c in controls: c.blockSignals(False)


    def __heuristicProbeMaxSideLengthFromBox(self, bb):
        return math.sqrt(
            (bb[0][1] - bb[0][0]) * (bb[0][1] - bb[0][0])
            + (bb[1][1] - bb[1][0]) * (bb[1][1] - bb[1][0])
            + (bb[2][1] - bb[2][0]) * (bb[2][1] - bb[2][0]))

    def __getMiddle(self):
        return (self.__parent.floatInRangeX.getValue(),
                self.__parent.floatInRangeY.getValue(),
                self.__parent.floatInRangeZ.getValue())

    def __radioButtonClick(self):
        self.__parent.floatInRangeRotX.setValue(0.0)
        self.__parent.floatInRangeRotY.setValue(0.0)
        self.__parent.floatInRangeRotZ.setValue(0.0)
        self.__setElementsEnabled()
        if self.__applyRadioBttn != None:
            self.__applyRadioBttn()

    def __setElementsEnabled(self):
        if (self.__type == CUTTINGSURFACE):
            if self.__parent.xAxisRadioButton.isChecked():
                self.__parent.floatInRangeRotX.setEnabled(False)
                self.__parent.floatInRangeX.setEnabled(True)
                self.__parent.floatInRangeRotY.setEnabled(True)
                self.__parent.floatInRangeY.setEnabled(True)
                self.__parent.floatInRangeRotZ.setEnabled(True)            
                self.__parent.floatInRangeZ.setEnabled(True)
            elif self.__parent.yAxisRadioButton.isChecked():
                self.__parent.floatInRangeRotX.setEnabled(True)
                self.__parent.floatInRangeX.setEnabled(True)
                self.__parent.floatInRangeRotY.setEnabled(False)
                self.__parent.floatInRangeY.setEnabled(True)
                self.__parent.floatInRangeRotZ.setEnabled(True)
                self.__parent.floatInRangeZ.setEnabled(True)
            else:
                self.__parent.floatInRangeRotX.setEnabled(True)
                self.__parent.floatInRangeX.setEnabled(True)
                self.__parent.floatInRangeRotY.setEnabled(True)
                self.__parent.floatInRangeY.setEnabled(True)
                self.__parent.floatInRangeRotZ.setEnabled(False)
                self.__parent.floatInRangeZ.setEnabled(True)

