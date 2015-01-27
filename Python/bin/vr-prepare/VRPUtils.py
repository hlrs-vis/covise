
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH


import math
import numpy
import copy
from numpy.linalg import inv
import os
import re


# pickle and deepcopy
try:
    import cPickle as pickle
    def fastDeepcopy(obj):
        try:
           return pickle.loads(pickle.dumps(obj, -1)) # cPickle is much faster than deepcopy but might have problems with some data structures
        except PicklingError:
           return copy.deepcopy(obj)
except:
    import pickle
    fastDeepcopy = copy.deepcopy

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject, pyqtSignal

from printing import InfoPrintCapable

import covise
import VRPCoviseNetAccess

from FloatInRangeControlBase import Ui_FloatInRangeControlBase
from ExitConfirmationBase import Ui_ExitConfirmationBase
from DialogOverwriteFileBase import Ui_DialogOverwriteFileBase
from DialogFileNewBase import Ui_DialogFileNewBase
from NewViewpointConfirmationBase import Ui_NewViewpointConfirmationBase
from DialogReduceTimeStepAskerBase import Ui_DialogReduceTimeStepAskerBase
from DialogChangePathBase import Ui_DialogChangePathBase
from DialogOkayAskerBase import Ui_DialogOkayAskerBase
from ErrorManager import ConversionError, OutOfDomainError

_infoer = InfoPrintCapable()
_infoer.doPrint = False # True

_logger = InfoPrintCapable()
_logger.doPrint = False # True
_logger.startString = '(log)'
_logger.module = __name__

_replaceList = []

def _log(func):
    def logged_func(*args, **kwargs):
        _logger.function = repr(func)
        _logger.write('')
        return func(*args, **kwargs)
    return logged_func


# Generals

def xor(a, b):
    return (a and not b) or (b and not a)

def inRange(number, aRange):
    return aRange[0] <= number <= aRange[1]

def roundToZero(value):
    if (abs(value) < 1.0e-20):
        return 0.0
    else:
        return value

def multMatVec(mat, vec):
    ret = [0.0,0.0,0.0]
    mat = numpy.array(mat)
    ret[0] = mat[0][0]*vec[0] + mat[1][0]*vec[1] + mat[2][0]*vec[2]
    ret[1] = mat[0][1]*vec[0] + mat[1][1]*vec[1] + mat[2][1]*vec[2]
    ret[2] = mat[0][2]*vec[0] + mat[1][2]*vec[1] + mat[2][2]*vec[2]
    return ret

def transformationMatrix(params):
    # translation
    trans = numpy.identity(4)
    trans[0,3] = params.transX
    trans[1,3] = params.transY
    trans[2,3] = params.transZ
    # rotation
    rot = numpy.identity(4)
    if not (params.rotX==0.0 and params.rotY==0.0 and params.rotZ==0.0):
        angle = params.rotAngle * math.pi/180.0
        s = numpy.sin(-angle)
        c = numpy.cos(-angle)
        t = 1 - c
        axis = numpy.array([params.rotX, params.rotY, params.rotZ])
        axis = axis / numpy.sqrt(numpy.dot(axis, axis))
        x = axis[0]
        y = axis[1]
        z = axis[2]
        rot[0,0] = t*x**2 + c
        rot[0,1] = t*x*y - s*z
        rot[0,2] = t*x*z + s*y
        rot[1,0] = t*x*y + s*z
        rot[1,1] = t*y**2 + c
        rot[1,2] = t*y*z - s*x
        rot[2,0] = t*x*z - s*y
        rot[2,1] = t*y*z + s*x
        rot[2,2] = t*z**2 + c
    # scale
    scale = numpy.identity(4)
    scale[0,0] = params.scaleX
    scale[1,1] = params.scaleY
    scale[2,2] = params.scaleZ
    # combine
    mat = numpy.identity(4)
    mat = numpy.dot(mat, trans)
    mat = numpy.dot(mat, rot)
    mat = numpy.dot(mat, scale)
    return mat

def fitAngle(angle):
    if angle > math.pi*2.0 or angle < -math.pi*2.0:
        print("---------angle bigger 360")
        anlge = angle % 360.0
    if (angle > math.pi and angle < math.pi*2.0):
        print("---------angle bigger 180")
        angle = math.pi*2.0  - angle
    elif (angle < -math.pi and angle > -math.pi*2.0):
        print("---------angle smaller -180")
        angle = -math.pi*2.0 - angle

    return angle


# Geometric entities

class AxisAlignedRectangleIn3d(object):

    """A rectangle with sides aligned to x, y or z-direction.

    An object of class AxisAlignedRectangle essentially
    stores data that defines a rectangle aligned to one
    of the axis.

    Meaning of lengthA--and lengthB respectively--is
    dependent on the orthogonal-direction d.  The table
    shows which side is used for the lengthA
    determination.  The side is referenced in the table
    simply by the direction, i.e. either x, y or z.

    d, lengthA, lengthB

    x, y, z
    y, x, z
    z, x, y

    Remarks

    . The meaning of the A and B sides is somewhat arbitrary.

    . Values lengthA == 0 or lengthB == 0 are possible.

    """

    def __init__(self):
        self.__middle = 0, 0, 0
        self.__orthoAxis = 'x'
        self.__sideA = 0
        self.__sideB = 0
        self.__orthoAxisPool = ['x', 'y', 'z', 'free', 'line']
        self.__rotX = 0
        self.__rotY = 0
        self.__rotZ = 0

    def setMiddle(self, middle):
        self.__middle = middle

    def setOrthogonalAxis(self, candidate):
        if not candidate in self.__orthoAxisPool:
            raise OutOfDomainError(candidate, self.__orthoAxisPool)
        self.__orthoAxis = candidate

    def setLengthA(self, value):
        assert 0 <= value
        self.__sideA = value

    def setLengthB(self, value):
        assert 0 <= value
        self.__sideB = value

    def setRotX(self, value):
        assert value<=180. and value>=-180, "%f" % value
        self.__rotX = value

    def setRotY(self, value):
        assert value<=180. and value>=-180, "%f" % value
        self.__rotY = value

    def setRotZ(self, value):
        assert value<=180. and value>=-180, "%f" % value
        self.__rotZ = value

    def getMiddle(self):
        return self.__middle

    def getOrthogonalAxis(self):
        return self.__orthoAxis

    def getLengthA(self):
        return self.__sideA

    def getLengthB(self):
        return self.__sideB

    def getRotX(self):
        return self.__rotX

    def getRotY(self):
        return self.__rotY

    def getRotZ(self):
        return self.__rotZ

    def __eq__(self, other):
    # dr: die Umrechnung von Startpoint1, startpoint2, direction auf Rectangle fuehrt oft zu Rundungsfehlern
    # die fuehren dann zu unnoetigen executes
    # wenn im config.vr-prepare.xml epsRectangleLength fuer Mittelpunkt/Laengen und epsRectangleAngle fuer Winkel drin stehen
    # wird auf < eps verglichen statt auf ==
    # Winkel geben erfahrungsgemaess den groeseren Fehler, daher wurden zwei unterschiedliche eps eingefuehrt
    
        if not other:
            return False
        EpsRectangleLength = covise.getCoConfigEntry("vr-prepare.EpsRectangleLength")
        if EpsRectangleLength:
            EpsRectangleLength = float(EpsRectangleLength)
            if math.fabs(self.middle[0] - other.middle[0]) < EpsRectangleLength \
            and math.fabs(self.middle[1] - other.middle[1]) < EpsRectangleLength \
            and math.fabs(self.middle[2] - other.middle[2]) < EpsRectangleLength \
            and math.fabs(self.lengthA - other.lengthA) < EpsRectangleLength \
            and math.fabs(self.lengthB - other.lengthB) < EpsRectangleLength:
                l=True
            else:
                l=False       
        else:
            if self.middle[0]==other.middle[0] \
            and self.middle[1]==other.middle[1] \
            and self.middle[2]==other.middle[2] \
            and self.lengthA == other.lengthA \
            and self.lengthB == other.lengthB:
                l=True
            else:
                l=False
                
        EpsRectangleAngle = covise.getCoConfigEntry("vr-prepare.EpsRectangleAngle")
        if EpsRectangleAngle:
            EpsRectangleAngle = float(EpsRectangleAngle)
            if math.fabs(self.rotX - other.rotX) < EpsRectangleAngle \
            and  math.fabs(self.rotY - other.rotY) < EpsRectangleAngle \
            and math.fabs(self.rotZ - other.rotZ) < EpsRectangleAngle:
                a=True
            else:
                a=False
        else:
            if self.rotX == other.rotX  and   self.rotY == other.rotY and self.rotZ == other.rotZ:
                a=True
            else:
                 a=False
                                 
        if self.orthogonalAxis == other.orthogonalAxis:
            axis=True
        else:
            axis=False 
        return l and a and axis
         

    def __ne__(self, other):
        return not self == other

    def __str__(self):
        return 'middle == %s\northogonalAxis == %s\n' \
               'lengthA == %f\nlengthB == %f\n' \
                'rotX == %f\nrotY == %f\nrotZ == %f' % \
               (str(self.middle), self.orthogonalAxis,
                self.lengthA, self.lengthB,
                self.rotX, self.rotY, self.rotZ )

    middle = property(getMiddle, setMiddle)
    orthogonalAxis = property(
        getOrthogonalAxis, setOrthogonalAxis)
    lengthA = property(getLengthA, setLengthA)
    lengthB = property(getLengthB, setLengthB)
    rotX = property(getRotX, setRotX)
    rotY = property(getRotY, setRotY)
    rotZ = property(getRotZ, setRotZ)

class AxisAlignedRectangleWithBounds(AxisAlignedRectangleIn3d):

    def __init__(self):
        super(AxisAlignedRectangleWithBounds, self).__init__()
        m = self.middle
        self.__bounds = (
            (m[0], m[0] + 1),
            (m[1], m[1] + 1),
            (m[2], m[2] + 1))

    def __eq__(self, other):
        return (
            self.__bounds == other.__bounds and
            super(AxisAlignedRectangleWithBounds, self).__eq__(other))

    def getBounds(self):
        return self.__bounds

    def setBounds(self, theBounds):
        """Set bounds for middle.

        When a middle coordinate does not fit into the
        respective interval then change this
        middle-coordinate to the minimum-value of the
        intervall.

        """
        for i in range(3): assert theBounds[i][0] <= theBounds[i][1]
        validMiddle = list(self.middle)
        for i in range(3):
            if not inRange(self.middle[i], theBounds[i]):
                validMiddle[i] = theBounds[i][0]
        # first set new valid bounds
        self.__bounds = theBounds
        # then the middle, since now we are having the new bound!
        self.middle = tuple(validMiddle)

    def setMiddle(self, middle):
        for i in range(3):
            if not inRange(middle[i], self.__bounds[i]):
                # middle not within bounds, dont set and return
                return

        # middle is between bounds, so set
        #super(AxisAlignedRectangleWithBounds, self).middle = middle
        super(AxisAlignedRectangleWithBounds, self).setMiddle(middle)
        #self.setMiddle(middle)

    def getMiddle(self):
        #return super(AxisAlignedRectangleWithBounds, self).middle
        return super(AxisAlignedRectangleWithBounds, self).getMiddle()
        #return self.getMiddle()


    bounds = property(getBounds, setBounds)
    middle = property(getMiddle, setMiddle)

class Line3D(AxisAlignedRectangleIn3d):

    def __init__(self):
        AxisAlignedRectangleIn3d.__init__(self)
        AxisAlignedRectangleIn3d.setOrthogonalAxis(self, 'line')
        self.__startpoint = ( 0., 0., 0. )
        self.__endpoint = ( 0., 0., 0. )
        AxisAlignedRectangleIn3d.setLengthB(self, 0.)
        
    def setStartEndPoint(self, x0, y0, z0, x1, y1, z1):
        self.__startpoint = (x0,y0,z0)
        self.__endpoint = (x1,y1,z1)
        AxisAlignedRectangleIn3d.setMiddle(self, (0.5*(x1-x0),0.5*(y1-y0),0.5*(z1-z0)) )
        AxisAlignedRectangleIn3d.setLengthA(self, 0.5*math.sqrt( (x1-x0)*(x1-x0) + \
                                                            (y1-y0)*(y1-y0) + \
                                                            (z1-z0)*(z1-z0) ) )

    def getStartPoint(self):
        return self.__startpoint
    
    def getEndPoint(self):
        return self.__endpoint

class RectangleIn3d2Ps1Dir(object):

    """A rectangle in space.

    The rectangle is defined by two points A and C plus
    a direction d.

    This class is written to fit for the
    'plane'-definition for Covise-modul Tracer.

    Quote: 'plane: rectangle created with the two starting
    points as opposite corners and direction as the
    direction of one side.'

    """

    def __init__(self):
        self.__pointA = 0, 0, 0
        self.__pointC = 0, 0, 0
        self.__d = 0, 0, 0

    def setPointA(self, point): self.__pointA = point
    def setPointC(self, point): self.__pointC = point
    def setDirection(self, direction): self.__d = direction
    def getPointA(self): return self.__pointA
    def getPointC(self): return self.__pointC
    def getDirection(self): return self.__d

    pointA = property(getPointA, setPointA)
    pointC = property(getPointC, setPointC)
    direction = property(getDirection, setDirection)

    def __eq__(self, other):
        return other.getPointA() == self.__pointA and \
               other.getPointC() == self.__pointC and \
               other.getDirection() == self.__d

    def __str__(self):
        ret =  ' pointA:    '+str(self.__pointA) +'\n'
        ret += ' pointC:    '+str(self.__pointC) +'\n'
        ret += ' direction: '+str(self.__d)
        return ret


class RectangleIn3d1Mid1Norm(object):
    """ A rectangle in space.

    The rectangle is defined by a point and a
    normal. Instead of the point, the distance
    to the origin can be used.

    This class is written to fit for the
    'plane'-definition for Covise-modul CuttingSurface.
    """

    def __init__(self):
        self.__point = 0, 0, 0
        self.__normal = 0, 0, 0

    def setPoint(self, point): self.__point = point
    def setNormal(self, normal): self.__normal = normal
    def getPoint(self): return self.__point
    def getNormal(self): return self.__normal

    def setDistance(self, distance):
        normal = numpy.array(self.__normal)
        lennormal = math.sqrt(numpy.dot(normal, normal))
        self.__point = normal * distance / lennormal
    def getDistance(self):
        normal = numpy.array(self.__normal)
        lennormal = math.sqrt(numpy.dot(normal, normal))
        normal = normal / lennormal
        return numpy.dot(self.__point, normal)

    point = property(getPoint, setPoint)
    normal = property(getNormal, setNormal)

    def __eq__(self, other):
        return other.getPoint() == self.__point and \
               other.getNormal() == self.__normal

    def __str__(self):
        ret =  ' point:    '+str(self.__point) +'\n'
        ret += ' normal: '+str(self.__normal)
        return ret

def convertAlignedRectangleToGeneral(aligned):
    """Return a AxisAlignedRectangleIn3d as a RectangleIn3d2Ps1Dir."""
    #print("++++****")
    #print("convertAlignedRectangleToGeneral")
    #print(aligned.lengthA)
    #print(aligned.lengthB)
    #print(aligned.orthogonalAxis)
    #print(aligned.rotX)
    #print(aligned.rotY)
    #print(aligned.rotZ)
    #print(aligned.middle)

    p1 = 0.0,0.0,0.0
    p2 = 0.0,0.0,0.0
    halfA = aligned.lengthA/2.0
    halfB = aligned.lengthB/2.0
    d = 0, 0, 1
    if aligned.orthogonalAxis == 'x':
        p1 = 0.0, -halfA, -halfB
        p2 = 0.0, halfA,  halfB
        d = 0, 0, 1
    elif aligned.orthogonalAxis == 'y':
        p1 = -halfA, 0, -halfB
        p2 = halfA, 0, halfB
        d = 1, 0, 0
    elif aligned.orthogonalAxis == 'z':
        p1 = -halfA, -halfB, 0
        p2 = halfA, halfB, 0
        d = 0, 1, 0

    alpha = aligned.rotX * math.pi/180.0
    betha = aligned.rotY * math.pi/180.0
    gamma = aligned.rotZ * math.pi/180.0
    rotXMatrix = numpy.matrix([[1,0,0], [0, numpy.cos(alpha), numpy.sin(alpha)], [0, -numpy.sin(alpha), numpy.cos(alpha)]])
    rotYMatrix = numpy.matrix([[numpy.cos(betha), 0, -numpy.sin(betha)], [0,1,0], [numpy.sin(betha), 0, numpy.cos(betha)]])
    rotZMatrix = numpy.matrix([[numpy.cos(gamma), numpy.sin(gamma), 0], [-numpy.sin(gamma), numpy.cos(gamma), 0], [0,0,1]])
    RotMatrix = rotXMatrix * rotYMatrix * rotZMatrix
    p1 = multMatVec(RotMatrix, p1)
    p2 = multMatVec(RotMatrix, p2)
    d = multMatVec(RotMatrix, d)
    dlen = math.sqrt(numpy.dot(d,d))

    ret = RectangleIn3d2Ps1Dir()
    ret.pointA = p1[0]+aligned.middle[0], p1[1]+aligned.middle[1],p1[2]+aligned.middle[2]
    ret.pointC = p2[0]+aligned.middle[0], p2[1]+aligned.middle[1],p2[2]+aligned.middle[2]
    ret.direction = d[0]/dlen, d[1]/dlen, d[2]/dlen

    #print("return "+str(ret.pointA))
    #print("return "+str(ret.pointC))
    #print("return "+str(ret.direction))
    #print("++++****")

    return ret


def convertAlignedRectangleToCutRectangle(aligned):
    """Return a AxisAlignedRectangleIn3d as a RectangleIn3d1Mid1Norm."""
    n = 1, 0, 0
    if aligned.orthogonalAxis == 'y':
        n = 0, 1, 0
    elif aligned.orthogonalAxis == 'z':
        n = 0, 0, 1

    alpha = aligned.rotX * math.pi/180.0
    betha = aligned.rotY * math.pi/180.0
    gamma = aligned.rotZ * math.pi/180.0
    rotXMatrix = numpy.matrix([[1,0,0], [0, numpy.cos(alpha), numpy.sin(alpha)], [0, -numpy.sin(alpha), numpy.cos(alpha)]])
    rotYMatrix = numpy.matrix([[numpy.cos(betha), 0, -numpy.sin(betha)], [0,1,0], [numpy.sin(betha), 0, numpy.cos(betha)]])
    rotZMatrix = numpy.matrix([[numpy.cos(gamma), numpy.sin(gamma), 0], [-numpy.sin(gamma), numpy.cos(gamma), 0], [0,0,1]])
    RotMatrix = rotXMatrix * rotYMatrix * rotZMatrix
    n = multMatVec(RotMatrix, n)
    nlen = math.sqrt(numpy.dot(n,n))

    ret = RectangleIn3d1Mid1Norm()
    ret.point = aligned.middle
    ret.normal = n[0]/nlen, n[1]/nlen, n[2]/nlen

    return ret


def convertGeneralToAlignedRectangle(gen, setAxis):
    #print("++++****")
    #print("convertGeneralToAlignedRectangle")
    #print(gen.pointA)
    #print(gen.pointC)
    #print(gen.direction)

    EPS = 0.00000001
    retRec = AxisAlignedRectangleIn3d()
    dia = numpy.array(gen.pointC)-numpy.array(gen.pointA)
    retRec.middle = (numpy.array(gen.pointA)+numpy.array(gen.pointC))*0.5
    normal = numpy.cross(dia, gen.direction)
    lennormal = math.sqrt(numpy.dot(normal, normal))
    normal = normal / lennormal
    lendia = math.sqrt(numpy.dot(dia, dia))
    uniqueDia = dia / lendia
    lendirection = math.sqrt(numpy.dot(gen.direction, gen.direction))
    uniqueDir = numpy.array(gen.direction)/lendirection
    lenA = numpy.dot(uniqueDia, uniqueDir) * lendia
    lenB = math.sqrt(lendia**2 - lenA**2)
    #dotX = numpy.dot(normal, numpy.array([1,0,0]))
    #angleX = 180.0 / math.pi * numpy.arccos( dotX )
    #dotY = numpy.dot(normal, numpy.array([0,1,0]))
    #angleY = 180.0 / math.pi * numpy.arccos( dotY )
    #dotZ = numpy.dot(normal, numpy.array([0,0,1]))
    #angleZ = 180.0 / math.pi * numpy.arccos( dotZ )
    #dotX2 = numpy.dot(normal, numpy.array([-1,0,0]))
    #angleX2 = 180.0 / math.pi * numpy.arccos( dotX2 )
    #dotY2 = numpy.dot(normal, numpy.array([0,-1,0]))
    #angleY2 = 180.0 / math.pi * numpy.arccos( dotY2 )
    #dotZ2 = numpy.dot(normal, numpy.array([0,0,-1]))
    #angleZ2 = 180.0 / math.pi * numpy.arccos( dotZ2 )
    #angles = {angleX:'x', angleY:'y', angleZ:'z', angleX2:'x', angleY2:'y', angleZ2:'z'}
    #minAngle = min(angles)
    #retRec.orthogonalAxis = angles[minAngle]
    retRec.lengthA=math.fabs(lenB)
    retRec.lengthB=math.fabs(lenA)
    retRec.orthogonalAxis = setAxis
    if retRec.orthogonalAxis == 'x':
        axis = numpy.array([1.0,0.0,0.0])
        if normal[0] == 0 and normal[1] == 0:
            #if normal is z axis use angle bisector
            normProjZ = numpy.array([1.0, 1.0, 0.0])
        else:
            normProjZ = numpy.array([normal[0], normal[1], 0.0])
        #normProjZ = numpy.array([normal[0], normal[1], 0.0])
        normProjZ = normProjZ / math.sqrt( numpy.dot(normProjZ, normProjZ) )
        rotZ = fitAngle(numpy.arccos (numpy.dot(normProjZ, axis)))
        if normProjZ[1] < 0.0:
            rotZ = -rotZ
        retRec.rotZ = 180.0/math.pi * rotZ
        rotZMatrix = numpy.matrix([[numpy.cos(rotZ), numpy.sin(rotZ), 0.0], [-numpy.sin(rotZ), numpy.cos(rotZ), 0.0], [0.0,0.0,1.0]])
        rotZMatrix = inv(rotZMatrix)
        normal = multMatVec(rotZMatrix, normal)
        direction = multMatVec(rotZMatrix, gen.direction)
        lennormal = math.sqrt(numpy.dot(normal, normal))
        normal = normal[0] / lennormal, normal[1]/lennormal, normal[2]/lennormal
        if normal[0] == 0 and normal[2] == 0:
            #if normal is y axis use angle bisector
            normProjY = numpy.array([1.0, 0.0, 1.0])
        else:
            normProjY = numpy.array([normal[0], 0.0, normal[2]])
        #normProjY = numpy.array([normal[0], 0.0, normal[2]])
        normProjY = normProjY / math.sqrt( numpy.dot(normProjY, normProjY) )
        rotY = fitAngle(numpy.arccos( numpy.dot(normProjY, axis) ))
        if normProjY[2] > 0.0:
            rotY = -rotY
        retRec.rotY =180.0/math.pi * rotY
        rotYMatrix = numpy.matrix([[numpy.cos(rotY), 0.0, -numpy.sin(rotY)], [0.0,1.0,0.0],[numpy.sin(rotY), 0.0, numpy.cos(rotY)]])
        rotYMatrix = inv(rotYMatrix)
        direction = multMatVec(rotYMatrix, direction)
        lenDir = math.sqrt(numpy.dot(direction, direction))
        direction = numpy.array(direction) /lenDir
        rotX = fitAngle(numpy.arccos(numpy.dot(numpy.array([0.0,0.0,1.0]), direction)))
        if direction[1] > 0.0:
            rotX = -rotX
        retRec.rotX = 180.0/math.pi * rotX
    elif retRec.orthogonalAxis == 'y':
        axis = numpy.array([0.0,1.0,0.0])
        if normal[0] == 0 and normal[1] == 0:
            #if normal is z axis use angle bisector
            normProjZ = numpy.array([1.0, 1.0, 0.0])
        else:
            normProjZ = numpy.array([normal[0], normal[1], 0.0])
        #normProjZ = numpy.array([normal[0], normal[1], 0.0])
        normProjZ = normProjZ / math.sqrt( numpy.dot(normProjZ, normProjZ) )
        rotZ = fitAngle(numpy.arccos(normProjZ[1]))#fitAngle(numpy.arccos (numpy.dot(normProjZ, axis)))
        if normal[0] > 0.0:
            rotZ = -rotZ
        retRec.rotZ = 180.0/math.pi * rotZ
        rotZMatrix = numpy.matrix([[numpy.cos(rotZ), numpy.sin(rotZ), 0.0], [-numpy.sin(rotZ), numpy.cos(rotZ), 0.0], [0.0,0.0,1.0]])
        rotZMatrix = inv(rotZMatrix)
        normal = multMatVec(rotZMatrix, normal)
        direction = multMatVec(rotZMatrix, gen.direction)
        direction = numpy.array( [direction[0],0.0,direction[2]])
        direction = numpy.array(direction) /math.sqrt(numpy.dot(direction, direction))
        rotY = fitAngle(numpy.arccos(numpy.dot(numpy.array([1.0,0.0,0.0]),direction)))
        if direction[2] > 0.0:
            rotY=-rotY
        retRec.rotY = 180.0/math.pi * rotY
        rotYMatrix = numpy.matrix([[numpy.cos(rotY), 0.0, -numpy.sin(rotY)], [0.0,1.0,0.0], [numpy.sin(rotY), 0.0, numpy.cos(rotY)]])
        rotYMatrix = inv(rotYMatrix)
        normal = multMatVec(rotYMatrix, normal) 
        if normal[1] == 0 and normal[2] == 0:
            #if normal is x axis use angle bisector
            normProjX = numpy.array([0.0, 1.0, 1.0])
        else:
            normProjX = numpy.array([0.0, normal[1], normal[2]])
        #normProjX = numpy.array([0.0, normal[1], normal[2]])
        normProjX = normProjX / math.sqrt(numpy.dot(normProjX, normProjX))
        rotX = fitAngle(numpy.arccos( numpy.dot(normProjX, axis) ))
        if normProjX[2] < 0.0:
            rotX=-rotX
        retRec.rotX = 180.0/math.pi * rotX
        retRec.lengthA=math.fabs(lenA)
        retRec.lengthB=math.fabs(lenB)
    elif retRec.orthogonalAxis == 'z':
        axis = numpy.array([0.0,0.0,1.0])
        direction = numpy.array( [gen.direction[0],gen.direction[1],0.0])
        direction = direction /math.sqrt(numpy.dot(direction, direction))
        rotZ = fitAngle(numpy.arccos(numpy.dot(numpy.array([0.0,1.0,0.0]),direction)))
        if direction[0] > 0.0:
            rotZ=-rotZ
        retRec.rotZ = 180.0/math.pi * rotZ
        rotZMatrix = numpy.matrix([[numpy.cos(rotZ), numpy.sin(rotZ), 0.0], [-numpy.sin(rotZ), numpy.cos(rotZ), 0.0], [0.0,0.0,1.0]])
        rotZMatrix = inv(rotZMatrix)
        normal = multMatVec(rotZMatrix, normal)
        if normal[0] == 0 and normal[2] == 0:
            #if normal is y axis use angle bisector
            normProjY = numpy.array([1.0, 0.0, 1.0])
        else:
            normProjY = numpy.array([normal[0], 0.0, normal[2]])
        #normProjY = numpy.array([normal[0], 0.0, normal[2]])
        normProjY = normProjY / math.sqrt( numpy.dot(normProjY, normProjY) )
        rotY = fitAngle(numpy.arccos (numpy.dot(normProjY, axis)))
        if normProjY[0] <0.0:
            rotY=-rotY
        retRec.rotY = 180.0/math.pi * rotY
        rotYMatrix = numpy.matrix([[numpy.cos(rotY), 0.0, -numpy.sin(rotY)], [0.0,1.0,0.0], [numpy.sin(rotY), 0.0, numpy.cos(rotY)]])
        rotYMatrix = inv(rotYMatrix)
        normal = multMatVec(rotYMatrix, normal)
        if normal[1] == 0 and normal[2] == 0:
            #if normal is x axis use angle bisector
            normProjX = numpy.array([0.0, 1.0, 1.0])
        else:
            normProjX = numpy.array([0.0, normal[1], normal[2]])
        #normProjX = numpy.array([0.0, normal[1], normal[2]])
        normProjX = normProjX / math.sqrt(numpy.dot(normProjX, normProjX))
        rotX = fitAngle(numpy.arccos( numpy.dot(normProjX, axis) ))
        if normProjX[1] > 0.0:
            rotX=-rotX
        retRec.rotX = 180.0/math.pi * rotX

    #print("return "+str(retRec.lengthA))
    #print("return "+str(retRec.lengthB))
    #print("return "+str(retRec.orthogonalAxis))
    #print("return "+str(retRec.rotX))
    #print("return "+str(retRec.rotY))
    #print("return "+str(retRec.rotZ))
    #print("return "+str(retRec.middle))
    #print("++++****")

    return retRec


def convertCutRectangleToAlignedRectangle(gen, setAxis):
    EPS = 0.00000001
    retRec = AxisAlignedRectangleIn3d()
    retRec.middle = gen.point
    retRec.lengthA=math.fabs(1.0)
    retRec.lengthB=math.fabs(1.0)
    retRec.orthogonalAxis = setAxis
    normal = gen.normal
    if retRec.orthogonalAxis == 'x':
        axis = numpy.array([1.0,0.0,0.0])
        if normal[0] == 0 and normal[1] == 0:
            #if normal is z axis use angle bisector
            normProjZ = numpy.array([1.0, 1.0, 0.0])
        else:
            normProjZ = numpy.array([normal[0], normal[1], 0.0])
        #normProjZ = numpy.array([normal[0], normal[1], 0.0])
        normProjZ = normProjZ / math.sqrt( numpy.dot(normProjZ, normProjZ) )
        rotZ = fitAngle(numpy.arccos (numpy.dot(normProjZ, axis)))
        if normProjZ[1] < 0.0:
            rotZ = -rotZ
        retRec.rotZ = 180.0/math.pi * rotZ
        rotZMatrix = numpy.matrix([[numpy.cos(rotZ), numpy.sin(rotZ), 0.0], [-numpy.sin(rotZ), numpy.cos(rotZ), 0.0], [0.0,0.0,1.0]])
        rotZMatrix = inv(rotZMatrix)
        normal = multMatVec(rotZMatrix, normal)
        lennormal = math.sqrt(numpy.dot(normal, normal))
        normal = normal[0] / lennormal, normal[1]/lennormal, normal[2]/lennormal
        normProjY = numpy.array([normal[0], 0.0, normal[2]])
        normProjY = normProjY / math.sqrt( numpy.dot(normProjY, normProjY) )
        rotY = fitAngle(numpy.arccos( numpy.dot(normProjY, axis) ))
        if normProjY[2] > 0.0:
            rotY = -rotY
        retRec.rotY =180.0/math.pi * rotY
        rotX = 0.0
    elif retRec.orthogonalAxis == 'y':
        axis = numpy.array([0.0,1.0,0.0])
        if normal[0] == 0 and normal[1] == 0:
            #if normal is z axis use angle bisector
            normProjZ = numpy.array([1.0, 1.0, 0.0])
        else:
            normProjZ = numpy.array([normal[0], normal[1], 0.0])
        #normProjZ = numpy.array([normal[0], normal[1], 0.0])
        normProjZ = normProjZ / math.sqrt( numpy.dot(normProjZ, normProjZ) )
        rotZ = fitAngle(numpy.arccos(normProjZ[1]))#fitAngle(numpy.arccos (numpy.dot(normProjZ, axis)))
        if normal[0] > 0.0:
            rotZ = -rotZ
        retRec.rotZ = 180.0/math.pi * rotZ
        rotZMatrix = numpy.matrix([[numpy.cos(rotZ), numpy.sin(rotZ), 0.0], [-numpy.sin(rotZ), numpy.cos(rotZ), 0.0], [0.0,0.0,1.0]])
        rotZMatrix = inv(rotZMatrix)
        normal = multMatVec(rotZMatrix, normal)
        retRec.rotY = 0.0
        if normal[1] == 0 and normal[2] == 0:
            #if normal is x axis use angle bisector
            normProjX = numpy.array([0.0, 1.0, 1.0])
        else:
            normProjX = numpy.array([0.0, normal[1], normal[2]])
        #normProjX = numpy.array([0.0, normal[1], normal[2]])
        normProjX = normProjX / math.sqrt(numpy.dot(normProjX, normProjX))
        rotX = fitAngle(numpy.arccos( numpy.dot(normProjX, axis) ))
        if normProjX[2] < 0.0:
            rotX=-rotX
        retRec.rotX = 180.0/math.pi * rotX
    elif retRec.orthogonalAxis == 'z':
        axis = numpy.array([0.0,0.0,1.0])
        retRec.rotZ = 0.0
        if normal[0] == 0 and normal[2] == 0:
            #if normal is y axis use angle bisector
            normProjY = numpy.array([1.0, 0.0, 1.0])
        else:
            normProjY = numpy.array([normal[0], 0.0, normal[2]])
        #normProjY = numpy.array([normal[0], 0.0, normal[2]])
        normProjY = normProjY / math.sqrt( numpy.dot(normProjY, normProjY) )
        rotY = fitAngle(numpy.arccos (numpy.dot(normProjY, axis)))
        if normProjY[0] <0.0:
            rotY=-rotY
        retRec.rotY = 180.0/math.pi * rotY
        rotYMatrix = numpy.matrix([[numpy.cos(rotY), 0.0, -numpy.sin(rotY)], [0.0,1.0,0.0], [numpy.sin(rotY), 0.0, numpy.cos(rotY)]])
        rotYMatrix = inv(rotYMatrix)
        normal = multMatVec(rotYMatrix, normal)
        if normal[1] == 0 and normal[2] == 0:
            #if normal is x axis use angle bisector
            normProjX = numpy.array([0.0, 1.0, 1.0])
        else:
            normProjX = numpy.array([0.0, normal[1], normal[2]])
        #normProjX = numpy.array([0.0, normal[1], normal[2]])
        normProjX = normProjX / math.sqrt(numpy.dot(normProjX, normProjX))
        rotX = fitAngle(numpy.arccos( numpy.dot(normProjX, axis) ))
        if normProjX[1] > 0.0:
            rotX=-rotX
        retRec.rotX = 180.0/math.pi * rotX

    return retRec



# qt stuff

def getDoubleInLineEdit(lE):

    """Return the double from the line-edit.

    If the string in lE is empty return 0.

    If neither of the above is possible then raise an
    exception.

    """

    if '' == str(lE.text()).strip():
        return 0
    val, ok = lE.text().toDouble()
    if not ok:
        raise ConversionError(
            str(lE.text()), float)
    return val

def getIntInLineEdit(lE):

    """Return the int from the line-edit.

    If this is not possible raise an exception.

    """

    val, ok = lE.text().toInt()
    if not ok:
        val, ok = lE.text().toDouble()
        if ok:
           val = int(val)
           return val
        raise ConversionError(str(lE.text()), int)
    return val


class SliderForFloatManager(QtCore.QObject):

    """To control a float value within a range.

    This class brings together a QSlider and a float interval.

    Emit sigValueChanged when the slider changed position.

    """
    sigValueChanged = pyqtSignal()
    sigSliderReleased = pyqtSignal()
    sigSliderPressed = pyqtSignal()

    def __init__(self, slider, aRange=(0, 1)):

        """Default values setting.

        Set the value to aRange[0].

        The number of steps for the slider is set to an
        arbitrary but big number.  So the
        SliderForFloatManager can provide only that
        number of floats.  Hopefully this is sufficient
        for the user.

        """

        QtCore.QObject.__init__(self)
        self.__min, self.__max = aRange
        assert self.__min < self.__max
        self.__slider = slider
        self.__arbitraryButBig = arbitraryButBig = 4242
        self.__slider.setMinimum(-arbitraryButBig)
        self.__slider.setMaximum(arbitraryButBig)
        self.__slider.setValue(self.__slider.minimum())
        
        self.__slider.valueChanged.connect(self.__emitValueChanged)
        self.__slider.sliderReleased.connect(self.__emitSliderReleased)
        self.__slider.sliderPressed.connect(self.__emitSliderPressed)

    def getValue(self):

        """Return the value according to the slider-position."""

        return self.__calculateFloatValue(self.__slider.value())

    def getRange(self):
        return self.__min, self.__max

    def getNumberDiscretisation(self):

        """Return number of subintervals discretising the range.

        The subintervals are eqidistributed and fill the range.

        """

        return 2 * self.__arbitraryButBig


    def setValue(self, floatValue):

        """Position the slider to floatValue.

        Attention.  There is typically information loss
        when using this function because there is only
        a certain number of values representable by
        this class.

        Precondition: floatValue is contained in the
        currently stored range.

        """

        assert self.__min <= floatValue,                        \
               'Refusing to set a value beyond '                \
               'the bounds.  floatValue==' + str(floatValue) +  \
               ' is not contained in [' + str(self.__min) +     \
               ', ' + str(self.__max) + ']'
        assert floatValue <= self.__max,                        \
               'Refusing to set a value beyond '                \
               'the bounds.  floatValue==' + str(floatValue) +  \
               ' is not contained in [' + str(self.__min) +     \
               ', ' + str(self.__max) + ']'

        if self.__max - self.__min != 0:
            percentage = (floatValue - self.__min) / (self.__max - self.__min)
        else:
            percentage = 1.0
        n = self.getNumberDiscretisation()
        self.__slider.setValue(self.__slider.minimum() + int(n * percentage))

    def setRange(self, anInterval, value=None):

        """Set range anInterval.  Set value to value.

        Precondition: value is contained in anInterval.

        Recent settings get lost.

        Setting of value gets emitted.

        """
        _infoer.function = str(self.setRange)
        _infoer.write(
            'setRange: [[[' + str(anInterval) + ', ' + str(value) + ']]]')

        if value == None:
            value = (anInterval[1] + anInterval[0])/2.0

        assert anInterval[0] <= value
        assert value <= anInterval[1]

        self.__min, self.__max = anInterval
        self.setValue(value)


    def __emitValueChanged(self, intValue):
        self.sigValueChanged.emit((self.__calculateFloatValue(intValue),))

    def __emitSliderReleased(self):
        self.sigSliderReleased.emit()
            
    def __emitSliderPressed(self):
        self.sigSliderPressed.emit()

    def __calculateFloatValue(self, intValue):
        percentage = float(intValue - self.__slider.minimum()) / \
            (self.__slider.maximum() - self.__slider.minimum())
        return self.__min + percentage * (self.__max - self.__min)


class FloatInRangeControl(Ui_FloatInRangeControlBase):

    sigValueChanged = pyqtSignal()
    sigSliderReleased = pyqtSignal()
    def __init__(self,parent = None,name = None,fl = 0):
        Ui_FloatInRangeControlBase.__init__(self, parent)
        self.setupUi(self)
        self.__range = 0.0, 1.0
        self.__value = 0.0
        validator = QtGui.QDoubleValidator(self.lineEdit)
        self.lineEdit.setValidator(validator)
        self.__outOfRangeColor = QtGui.QColor('red')
        PrecisionFloatInRangeControls = covise.getCoConfigEntry("vr-prepare.PrecisionFloatInRangeControls")
        if PrecisionFloatInRangeControls:
            self.precision = int(PrecisionFloatInRangeControls)
        else:
            self.precision = 6
        self.__setRepresentationInLineEdit()
        self.__mgr = SliderForFloatManager(self.slider)
        self.__mgr.setRange(self.__range, self.__value)

        self.slider.sliderReleased.connect(self.reactOnSliderReleased)
        self.__mgr.sigValueChanged.connect(self.__reactOnSliderChange)
        self.lineEdit.returnPressed.connect(self.__reactOnLineEditReturnPressed)
        

    def getRange(self):
        return self.__range

    def getValue(self):
        return self.__value


    def setRange(self, aRange):
        self.__range = aRange
        self.__mgr.setRange(self.__range)
        self.setValue(aRange[0])

    def setValue(self, aValue):
        value = aValue
        if aValue<self.__range[0]: value = self.__range[0]
        if aValue>self.__range[1]: value = self.__range[1]
        self.__value = value
        self.__setRepresentationInLineEdit()
        self.__setRepresentationInSlider()
        self.__emitValueChanged()

    def __setRepresentationInLineEdit(self):
        self.lineEdit.blockSignals(True)
        formattedValue = '%.*g' % (self.precision, self.__value)
        self.lineEdit.setMaximumWidth( self.precision*10 )
        self.lineEdit.setText(formattedValue)
        self.lineEdit.home(False)
        self.lineEdit.blockSignals(False)

    def __setRepresentationInSlider(self):
        self.__mgr.blockSignals(True)
        self.__mgr.setValue(self.__value)
        self.__mgr.blockSignals(False)

    def __reactOnSliderChange(self, aValue):
        #print("**Slider", self.__range[0] , aValue , self.__range[1])
        assert self.__range[0] <= aValue[0] <= self.__range[1]
        self.__value = aValue[0]
        self.__setRepresentationInLineEdit()
        self.lineEdit.leavePendingMode()
        self.__emitValueChanged()

    def reactOnSliderReleased(self):
        self.__emitSliderReleased()

    def __reactOnLineEditReturnPressed(self):
        try:
            aValue = float(str(self.lineEdit.text()))
        except:
            assert False, ('The text-input-control is expected to '
                           'provide always numbers.')
        if not self.__range[0] <= aValue <= self.__range[1]:
            self.lineEdit._setBgColor(self.__outOfRangeColor)
            return
        self.__value = aValue
        self.__setRepresentationInSlider()
        self.lineEdit.leavePendingMode()
        self.__emitSliderReleased()

    @_log
    def __emitValueChanged(self):
        self.sigValueChanged.emit((self.__value,))

    def __emitSliderReleased(self):
        self.sigSliderReleased.emit()


class IntInRangeControl(Ui_FloatInRangeControlBase):

    sigValueChanged = pyqtSignal()
    sigSliderReleased = pyqtSignal()
    
    def __init__(self,parent = None,name = None,fl = 0):
        Ui_FloatInRangeControlBase.__init__(self, parent)
        self.setupUi(self)
        self.__range = 0, 1
        self.__value = 0
        validator = QtGui.QIntValidator(parent)
        self.lineEdit.setValidator(validator)
        self.__outOfRangeColor = QtGui.QColor('red')
        self.precision = 1
        self.__setRepresentationInLineEdit()
        self.slider.setRange( self.__range[0], self.__range[1])
        self.slider.setValue(self.__value)

        self.slider.sliderReleased.connect(self.__reactOnSliderReleased)
        self.slider.valueChanged.connect(self.__reactOnSliderChange)
        self.slider.returnPressed.connect(self.__reactOnLineEditReturnPressed)

    def getRange(self):
        return self.__range

    def getValue(self):
        return self.__value


    def setRange(self, aRange):
        self.__range = aRange
        self.slider.setRange( self.__range[0], self.__range[1])
        self.setValue(aRange[0])

    def setValue(self, aValue):
        value = aValue
        if aValue<self.__range[0]: value = self.__range[0]
        if aValue>self.__range[1]: value = self.__range[1]
        self.__value = value
        self.__setRepresentationInLineEdit(True)
        self.__setRepresentationInSlider()
        self.__emitValueChanged()

    def __setRepresentationInLineEdit(self, fromSetValue=False):
        self.lineEdit.blockSignals(True)
        formattedValue = str(self.__value)
        self.lineEdit.setMaximumWidth( 100 )
        self.lineEdit.setText(formattedValue)
        self.lineEdit.home(False)
        self.lineEdit.blockSignals(False)
        if not fromSetValue:
            self.__emitSliderReleased()
            
    def __setRepresentationInSlider(self):
        self.slider.blockSignals(True)
        self.slider.setValue(self.__value)
        self.slider.blockSignals(False)

    def __reactOnSliderChange(self, aValue):
        assert self.__range[0] <= aValue <= self.__range[1]
        self.__value = aValue
        self.__setRepresentationInLineEdit()
        self.lineEdit.leavePendingMode()
        self.__emitValueChanged()

    def __reactOnSliderReleased(self):
        self.__emitSliderReleased()

    @_log
    def __reactOnLineEditReturnPressed(self):
        try:
            aValue = int(str(self.lineEdit.text()))
        except:
            assert False, ('The text-input-control is expected to '
                           'provide always numbers.')
        if not self.__range[0] <= aValue <= self.__range[1]:
            self.lineEdit.setPaletteBackgroundColor(self.__outOfRangeColor)
            return
        self.__value = aValue
        self.__setRepresentationInSlider()
        self.lineEdit.leavePendingMode()
        self.__emitSliderReleased()

    @_log
    def __emitValueChanged(self):
        self.sigValueChanged.emit((self.__value,))

    def __emitSliderReleased(self):
        self.sigSliderReleased.emit()


class PopUpper(QtCore.QObject):
    def __init__(self):
        self.__init__(self, None) # no parent
        self.__itemList = None
        self.__lastChoice = None
        self.__idItemDict = {}

    def setItems(self, aList):
        self.__itemList = aList

    def popUp(self, parent, pt):
        menu = QPopupMenu(parent)
        self.__idItemDict.clear()
        for name in self.__itemList:
            id = menu.insertItem(
                name, self.__popupItemSelected)
            self.__idItemDict[id] = name
        menu.popup(pt)

    def getChoice(self):
        return self.__lastChoice

    def __popupItemSelected(self, id):
        _infoer.write('__popupItemSelected id == %s' % id)
        self.__lastChoice = self.__idItemDict[id]
        _infoer.write('\'' + self.getChoice() + '\' chosen')


class ReallyWantToOverrideAsker(Ui_DialogOverwriteFileBase):

    def __init__(self, parent, filename):
        Ui_DialogOverwriteFileBase.__init__(self, parent)
        self.setupUi(self)
        self.textLabel_fileName.setText(filename)

class FileNewAsker(Ui_DialogFileNewBase):

    def __init__(self, parent):
        Ui_DialogFileNewBase.__init__(self, parent)
        self.setupUi(self)

class SaveBeforeExitAsker(Ui_ExitConfirmationBase):
    
    def __init__(self, parent):
        Ui_ExitConfirmationBase.__init__(self, parent)
        self.setupUi(self)
        self.__pressedButton = None
        
        self.buttonSave.clicked.connect(self.__buttonSavePressed)
        self.buttonDiscard.clicked.connect(self.__buttonDiscardPressed)
        
    def pressedSave(self):
        return  self.__pressedButton==0
        
    def pressedDiscard(self):
        return  self.__pressedButton==1
        
    def __buttonSavePressed(self):
        self.__pressedButton=0
        self.accept()
        
    def __buttonDiscardPressed(self):
        self.__pressedButton=1    
        self.accept()
        
class NewViewpointAsker(Ui_NewViewpointConfirmationBase):

    def __init__(self, parent):
         Ui_NewViewpointConfirmationBase.__init__(self, parent)
         self.setupUi(self)
         self.__pressedButton = None
         self.__saveDecision = False
         
         self.buttonYes.clicked.connect(self.__buttonYesPressed)
         self.buttonNo.clicked.connect(self.__buttonNoPressed)
         self.checkBoxRemember.clicked.connect(self.__buttonCheckRemember)
         
    def pressedYes(self):
        return self.__pressedButton == 0

    def pressedNo(self):
        return self.__pressedButton == 1
        
    def getDecision(self):
        return self.__saveDecision
        
    def __buttonYesPressed(self):
        self.__pressedButton = 0
        self.accept()
        
    def __buttonNoPressed(self):
        self.__pressedButton = 1
        self.accept()
        
    def __buttonCheckRemember(self):
        self.__saveDecision = self.checkBoxRemember.isChecked()

_reduceDecision = None
class ReduceTimestepAsker(Ui_DialogReduceTimeStepAskerBase):

    def __init__(self):
        global _reduceDecision
        Ui_DialogReduceTimeStepAskerBase.__init__(self, None)
        self.setupUi(self)
        if _reduceDecision != None:
            self.hide()
            return 
         
        self.buttonYes.clicked.connect(self.__buttonYesPressed)
        self.buttonNo.clicked.connect(self.__buttonNoPressed)

    def exec_loop(self):
        global _reduceDecision
        if _reduceDecision == None:
            return Ui_DialogReduceTimeStepAskerBase.exec_loop(self)
        else:
            return self.accept()
         
    def pressedYes(self):
        global _reduceDecision  
        return _reduceDecision == 0

    def pressedNo(self):
        global _reduceDecision
        return _reduceDecision == 1

    def __buttonYesPressed(self):
        global _reduceDecision
        _reduceDecision = 0
        self.accept()
        
    def __buttonNoPressed(self):
        global _reduceDecision
        _reduceDecision = 1
        self.accept()
        
class OkayAsker(Ui_DialogOkayAskerBase):

    def __init__(self, parent, text):
        Ui_DialogOkayAskerBase.__init__(self, parent)
        self.setupUi(self)
        self.textLabel.setText(text)
        
        self.buttonOkay.clicked.connect(self.__buttonYesPressed)
    
    def __buttonYesPressed(self):
        self.accept()
        
                
class ChangePathAsker(Ui_DialogChangePathBase):                

    def __init__(self, nameDataset, parent):
        Ui_DialogChangePathBase.__init__(self, parent)
        self.setupUi(self)
        self.datasetLabel.setText(nameDataset)
        cursor = self.cursor()
        cursor.setShape(QtCore.Qt.ArrowCursor)
        self.setCursor(cursor)
        self.__pressedButton = None
        
        self.buttonOk.clicked.connect(self.__buttonYesPressed)
        self.buttonCancel.clicked.connect(self.__buttonNoPressed)
        
    def pressedYes(self):
        return self.__pressedButton == 0
        
    def pressedNo(self):
        return self.__pressedButton == 1
        
    def __buttonYesPressed(self):
        self.__pressedButton = 0
        self.accept()
        
    def __buttonNoPressed(self):
        self.__pressedButton = 1
        self.accept()
                 
def unpickleProjectFile(filename, replaceInPathList = []):
    inputFile = open(filename, 'rb')
    content = inputFile.read()
    # conversions for version 1
    replaceList = [ ('(cImportManager\nImportSample2DModule', '(cImportSampleManager\nImportSample2DModule'),
                    ('(cImportManager\nImportSample3DModule', '(cImportSampleManager\nImportSample3DModule'),
                    ('(cImportManager\nImportGroup3DModule', '(cImportGroupManager\nImportGroup3DModule'),
                    ('(cImportManager\nImportGroup2DModule', '(cImportGroupManager\nImportGroup2DModule'),
                    ('(cVRPImportSampleManager\n', '(cImportSampleManager\n'),
                    ('(cVRPImportGroupManager\n', '(cImportGroupManager\n'),
                    ('(cVRPImportManager\n', '(cImportManager\n'),
                    ('PartIsoPlaneVis', 'PartIsoSurfaceVis' ),
                    (' (scalar)', ''),
                    ('Part2DRawVisParamsMaterialsAlu','Part2DRawVisParams') ,
                    ('flyMode','flyingMode')]
    for rep in replaceList :
        content = content.replace( rep[0], rep[1] )
    for rep in replaceInPathList:
        content = content.replace(rep[0], rep[1])
        if not rep in _replaceList:
            _replaceList.append(rep)
    project = pickle.loads(content)
    inputFile.close()
    return project

def ParamsDiff( oldp, newp ):
    """ return a list of changed parameter names """
    return [key for key in newp.__dict__.keys() \
        if (key in oldp.__dict__) \
        and (newp.__dict__[key] != oldp.__dict__[key]) \
        and ( not ( isinstance(newp.__dict__[key], float) and isinstance(oldp.__dict__[key], float) ) \
            or (abs(newp.__dict__[key] - oldp.__dict__[key]) > abs(0.000001*oldp.__dict__[key])) \
            ) \
        ]


def CopyParams( params ):
    """ return a copied parameter class """

    ### To improve speed, we don't copy the status dict of coPresentationStepParams.
    ### The problem is the structure in which the viewpoints are stored and the deepcopy of the status-dict in coPresentationStepParams (which holds the params of all VisItems).
    ### Each coPresentationMgr.changeUnconfirmedViewpointID (called for each viewpoint) results in changes of the viewpoint in all presentation steps.
    ### So we have num_viewpoints*num_presentationsteps calls of coPresentationStep.changeUnconfirmedViewpointID which uses at least 3 CopyParams.
    ### As a result, loading a project takes a considerable amount of time.

    if hasattr(params, "status"):
        ret = params.__class__()
        for key in params.__dict__:
            if (key == "status"):
                ret.__dict__[key] = None
            else:
                ret.__dict__[key] = fastDeepcopy(params.__dict__[key])
        return ret

    return fastDeepcopy(params)


def TestCopyParams( params ):
    # the function is not usable at the moment
    # the right import has to be generated
    return

    newparams = getattr(__import__(__name__), params.__class__.__name__) ()
    for key in params.__dict__:
        newparams.__dict__[key] = params.__dict__[key]


def getExistingFilename(filename):
    # check filename directly
    try:
        if os.access(filename, os.R_OK):
            return filename
    except:
        pass
    
    # check filename at COVISE_PATH
    try:
        tmp = os.getenv("COVISE_PATH") + "/" + filename
        if os.access(tmp, os.R_OK):
            return tmp
    except:
        pass

    # check filename at COVISEDIR
    try:
        tmp = os.getenv("COVISEDIR") + "/" + filename
        if os.access(tmp, os.R_OK):
            return tmp
    except:
        pass
     
    for rep in _replaceList:
        f = filename.replace(rep[0], rep[1])
        try:
            if os.access(tmp, os.R_OK):
                return tmp
        except:
            pass
        
        
    # nothing worked
    return None

def mergeGivenParams(paramsObject, defaultParams):
    for key in defaultParams.keys():
        if not hasattr(paramsObject, key):
            if (type(defaultParams[key]).__name__ == 'list'):
                paramsObject.__dict__[key] = list(defaultParams[key])
            elif (type(defaultParams[key]).__name__ == 'dict'):
                paramsObject.__dict__[key] = dict(defaultParams[key])
            else:
                paramsObject.__dict__[key] = defaultParams[key]

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    ExitConfirmationBase = QtWidgets.QDialog()
    FloatInRangeControl(ExitConfirmationBase)
    ExitConfirmationBase.show()
    sys.exit(app.exec_())
    

def csvStrToList(csvStr):
    """ separate comma separated values into list, eg. "aaa, bbb, ccc" -> ['aaa', 'bbb', 'ccc']
    """
    if csvStr:
        return [str.strip(x) for x in csvStr.split(",")]
    else:
        return []

def getImportFileTypes():
    # define possible import file types
    from vtrans import coTranslate

    importFileTypes = coTranslate('All files (%s)\n') % '*.*'
    importFileTypes += coTranslate('COCASE (%s)\n') % '*.cocase'
    importFileTypes += coTranslate('COVISE (%s)\n') % '*.covise'

    if covise.coConfigIsOn("vr-prepare.Features.ImportVRML", True):
        importFileTypes += coTranslate('VRML (%s)\n') % '*.wrl *.vrml'
    if covise.coConfigIsOn("vr-prepare.Features.ImportOBJ", True):
        importFileTypes += coTranslate('OBJ (%s)\n') % '*.obj'
    if covise.coConfigIsOn("vr-prepare.Features.ImportSTL", True):
        importFileTypes += coTranslate('STL (%s)\n') % '*.stl'
    if covise.coConfigIsOn("vr-prepare.Features.Import3DS", True):
        importFileTypes += coTranslate('3DS (%s)\n') % '*.3ds'
    if covise.coConfigIsOn("vr-prepare.Features.ImportIV", True):
        importFileTypes += coTranslate('Inventor (%s)\n') % '*.iv'
    if covise.coConfigIsOn("vr-prepare.Features.ImportOSG", True):
        importFileTypes += coTranslate('OpenSceneGraph (%s)\n') % '*.osg'
    if covise.coConfigIsOn("vr-prepare.Features.ImportJT", False):
        importFileTypes += coTranslate('Jupiter Tesselation (%s)\n') % '*.jt'
    if covise.coConfigIsOn("vr-prepare.Features.ImportPLMXML", False):
        importFileTypes += coTranslate('PLMXML (%s)\n') % '*.plmxml'

    if covise.coConfigIsOn("vr-prepare.Features.ImportDocument", True):
        importFileTypes += coTranslate('Document (%s)\n') % '*.tif *.tiff *.png'
                    
    return importFileTypes

def getImportFileTypesFlat():
    importFileTypesFlat = list(set([x.lower() for x in re.findall("\*(\.\w+)", getImportFileTypes().replace('\n', ' '))]))
    
    return importFileTypesFlat

def addServerHostFromConfig():
    ServerHost = covise.getCoConfigEntry("vr-prepare.ServerConfig.Host")
    if ServerHost:
        serverHost = ServerHost
        # TODO get Covise Username as default
        ServerUser = covise.getCoConfigEntry("vr-prepare.ServerConfig.User")
        if ServerUser:
            serverUser = ServerUser
        else:
            serverUser = "demo60"
        # ssh connection
        serverConnection = "3"
        # timeout
        ServerTimeout = covise.getCoConfigEntry("vr-prepare.ServerConfig.Timeout")
        if ServerTimeout:
            serverTimeout = ServerTimeout
        else:
            serverTimeout = "3"

        VRPCoviseNetAccess.theNet().addHost( serverHost, \
                                             serverUser, \
                                             " ", \
                                             serverConnection, \
                                             serverTimeout )
        VRPCoviseNetAccess.theNet().setDefaultHost( serverHost )


def fillShaderList(listWidget):
    listWidget.clear()
    listWidget.addItem("")
    try:
        shaderDir = os.getenv("COVISEDIR")
        if (shaderDir == None):
            shaderDir = os.getenv("COVISE_PATH")
        shaderDir = shaderDir + "/share/covise/materials/"
        tmpList = os.listdir(shaderDir)
        xmlList = [s for s in tmpList if os.path.isfile(shaderDir + s) and s.endswith(".xml")]
        xmlList.sort()
        for s in xmlList:
            listWidget.addItem(s.replace(".xml", ""))
    except:
        pass

def selectInShaderList(listWidget, selection):
    items = listWidget.findItems(selection, QtCore.Qt.MatchCaseSensitive)
    if (items == None) or (len(items) == 0):
        listWidget.addItem(selection)
        items = listWidget.findItems(selection, QtCore.Qt.MatchCaseSensitive)
    listWidget.setCurrentItem(items[0])

# eof
