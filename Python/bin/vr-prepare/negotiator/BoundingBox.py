"""File: goodies.py

Various auxillary entities.


(C) 2004-2006 VISENSO GmbH, Stuttgart   info@visenso.de

"""
import re
from printing import InfoPrintCapable

class Box(object):

    """An axisaligned box in 3d-space.

    Invariant: min <= max in every coordinate direction.

    With this invariant it is clear, that there is no
    use of an interval like (42.0, -42.0).

    """

    def __init__(self, xminmax=(0, 1), yminmax=(0, 1), zminmax=(0, 1)):
        self.__xMinMax = xminmax
        self.__yMinMax = yminmax
        self.__zMinMax = zminmax
        assert self.__invariantHolds()

    def getXMinMax(self): return self.__xMinMax
    def getYMinMax(self): return self.__yMinMax
    def getZMinMax(self): return self.__zMinMax

    def getXMin(self): return self.__xMinMax[0]
    def getYMin(self): return self.__yMinMax[0]
    def getZMin(self): return self.__zMinMax[0]
    def getXMax(self): return self.__xMinMax[1]
    def getYMax(self): return self.__yMinMax[1]
    def getZMax(self): return self.__zMinMax[1]

    def setXMinMax(self, value):
        self.__xMinMax = value
        assert self.__invariantHolds()

    def setYMinMax(self, value):
        self.__yMinMax = value
        assert self.__invariantHolds()

    def setZMinMax(self, value):
        self.__zMinMax = value
        assert self.__invariantHolds()

    def getCenter(self):
        return ( 0.5*(self.__xMinMax[1]+self.__xMinMax[0]),
                 0.5*(self.__yMinMax[1]+self.__yMinMax[0]),
                 0.5*(self.__zMinMax[1]+self.__zMinMax[0]) )

    def getTuple(self):
        return (  (self.__xMinMax[0], self.__xMinMax[1]),
                  (self.__yMinMax[0], self.__yMinMax[1]),
                  (self.__zMinMax[0], self.__zMinMax[0]) )

    def getMaxEdgeLength(self):
        return max( self.__xMinMax[1]-self.__xMinMax[0],
                    self.__yMinMax[1]-self.__yMinMax[0],
                    self.__zMinMax[1]-self.__zMinMax[0] )
    xMinMax = property(getXMinMax, setXMinMax)
    yMinMax = property(getYMinMax, setYMinMax)
    zMinMax = property(getZMinMax, setZMinMax)

    def __invariantHolds(self):

        """Return if in every dimension the relation min < max holds."""

        return self.__xMinMax[0] <= self.__xMinMax[1] and \
               self.__yMinMax[0] <= self.__yMinMax[1] and \
               self.__zMinMax[0] <= self.__zMinMax[1]

    def __eq__(self, other):
        if not other:
            return False
        return other.xMinMax == self.xMinMax and \
               other.yMinMax == self.yMinMax and \
               other.zMinMax == self.zMinMax

    def __add__(self,other):
        if other:
            self.__xMinMax = ( min( self.__xMinMax[0], other.getXMinMax()[0] ), max( self.__xMinMax[1], other.getXMinMax()[1] ) )
            self.__yMinMax = ( min( self.__yMinMax[0], other.getYMinMax()[0] ), max( self.__yMinMax[1], other.getYMinMax()[1] ) )
            self.__zMinMax = ( min( self.__zMinMax[0], other.getZMinMax()[0] ), max( self.__zMinMax[1], other.getZMinMax()[1] ) )
            assert self.__invariantHolds()
        return self

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return str(self.xMinMax) + str(self.yMinMax) + str(self.zMinMax)


def boundPointToBox(box, point):

    """Return point restricted to the box.

    Identity if point is inside box.  Else choose
    nearest borders of the box as point-coordinates.

    """

    px, py, pz = point
    xmin, xmax = box.xMinMax
    ymin, ymax = box.yMinMax
    zmin, zmax = box.zMinMax
    if px > xmax: px = xmax
    if px < xmin: px = xmin
    if py > ymax: py = ymax
    if py < ymin: py = ymin
    if pz > zmax: pz = zmax
    if pz < zmin: pz = zmin

    assert box.xMinMax[0] <= px and px <= box.xMinMax[1]
    assert box.yMinMax[0] <= py and py <= box.yMinMax[1]
    assert box.zMinMax[0] <= pz and pz <= box.zMinMax[1]

    return px, py, pz



class BoundingBoxParser(object):

    """Extraction of bounding-box coordinates from lines in a special format.

    Parse bounding-box information out of covise info msg
    and fill a list with iot.

    For each coordinate three values get extracted.
    These are typically min, max and center of a box.

    History:
    Project: EdF 2003
    Status: under dev UNRELEASED

    (C) 2003 VirCinity GmbH, Stuttgart   info@vircinity.com
    31.10.2002 [RM@visenso.de]

    """

    def __init__(self):
        # place for the bounding-box and center
        self.__bb = ['', '', '',
                     '', '', '',
                     '', '', '']
        self.__numTimeSteps = 0

    def parseQueue(self, queue):
        line = queue.get()
        while line != None:
            if self.fill(line): break
            line = queue.get()

    def fill(self, line):
        dispatch = {
            'x': self.__fillBBx,
            'y': self.__fillBBy,
            'z': self.__fillBBz}
        if line.find("TimeSteps =") > -1:
            self.__fillTimeSteps(line)
        if line.find("center =") > -1:
            aLine = line.split(' ')
            f = dispatch[str(aLine[1])]
            f(aLine)
            if (aLine[1] == 'z'): return True # last line to be parsed
        return False

    def getBoundingBoxAndCenter(self):
        """Return the list x-minimum, x-maximum, x-center,...,z-center."""
        return self.__bb

    def getBox(self):
        box = Box()
        if self.__bb[0] and self.__bb[1] : box.xMinMax = float(self.__bb[0]), float(self.__bb[1])
        if self.__bb[3] and self.__bb[4] : box.yMinMax = float(self.__bb[3]), float(self.__bb[4])
        if self.__bb[6] and self.__bb[7] : box.zMinMax = float(self.__bb[6]), float(self.__bb[7])
        return box

    def getCenter(self):

        """Return 'center'-coordinates as float-triple."""

        return float(self.__bb[2]), float(self.__bb[5]), float(self.__bb[8])
        
    def getNumTimeSteps(self):
        return self.__numTimeSteps


    def __fillBB(self, startIdx, line):
        if startIdx < 9:
            i = 0
            num = re.compile("[(0-9)+-][(0-9).e+-]*")
            for xx in line:
                nn = num.search(xx)
                if nn != None:
                    self.__bb[int(startIdx)+i] = xx
                    i = i + 1

    def __fillBBx(self, line):
        self.__fillBB(0, line)

    def __fillBBy(self, line):
        self.__fillBB(3, line)

    def __fillBBz(self, line):
        self.__fillBB(6, line)
        
        
    def __fillTimeSteps(self, line):
        self.__numTimeSteps = (int)(line.split()[2])
#eof
