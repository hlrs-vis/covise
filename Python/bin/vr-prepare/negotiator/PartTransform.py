
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH

import math
import numpy
from Utils import multMatVec, transformationMatrix, ParamsDiff, mergeGivenParams

import covise
import Neg2Gui

from coGRMsg import coGRObjTransformMsg

from printing import InfoPrintCapable
_infoer = InfoPrintCapable()
_infoer.doPrint = False # True

class PartTransform(object):
    """ VisItem to color an object with an rgba color """
    def __init__(self, sendToCOVER=False):
        object.__init__(self)
        self._sendToCOVER = sendToCOVER
        self._box = None

    def setParams( self, params, negMsgHandler=None, sendToCover=True, realChange=None):
        """ set parameters from outside
            + init tracer module if necessary
            + mainly receive parameter changes from Gui
            + send status messages to COVER if state has changed
        """
        _infoer.function = str(self.setParams)
        _infoer.write(" ")

        if not hasattr(self.params, 'rotAngle'):
            return
        if realChange==None:
            realChange = ParamsDiff( self.params, params )
        if (   'rotAngle' in realChange \
            or (self.params.rotAngle > 0 and ('rotX' in realChange or 'rotY' in realChange or 'rotZ' in realChange)) \
            or ('transX' in realChange or 'transY' in realChange or 'transZ' in realChange) \
            or ('scaleX' in realChange or 'scaleY' in realChange or 'scaleZ' in realChange) ):
            if not hasattr(self, '_sendToCOVER') or self._sendToCOVER:
                self._sendMatrix()
            self._setTransform()


    def recreate(self, negMsgHandler, parentKey, offset):
        PartTransformParams.mergeDefaultParams(self.params) # explicitly call mergeDefaultParams of this class
        self._box = None
        return

    #send transform matrix to cover
    def _sendMatrix(self):
        return
        if self.covise_key == 'No key':
            return
        if not hasattr( self.params, 'rotX'):
            return
        if self.params.rotX==0.0 and self.params.rotY==0.0 and self.params.rotZ==0.0 and \
            self.params.transX==0.0 and self.params.transY==0.0 and self.params.transZ==0.0 and \
            self.params.scaleX==0.0 and self.params.scaleY==0.0 and self.params.scaleZ==0.0:
            return

        matrix = transformationMatrix(self.params)

        # transform the bbox and send to children
        bbox = self.importModule.getBox()
        bboxVec1 = [bbox.getXMin(), bbox.getYMin(), bbox.getZMin()]
        bboxVec2 = [bbox.getXMax(), bbox.getYMax(), bbox.getZMax()]
        if not (self.params.rotX==0.0 and self.params.rotY==0.0 and self.params.rotZ==0.0):
            mat = numpy.array(matrix[0:3,0:3])
            bboxVec1 = multMatVec(mat, bboxVec1)
            bboxVec2 = multMatVec(mat, bboxVec2)
        #sort vecs min/max
        if bboxVec1[0] > bboxVec2[0]: 
            tmp = bboxVec1[0]
            bboxVec1[0] = bboxVec2[0]
            bboxVec2[0] = tmp
        if bboxVec1[1] > bboxVec2[1]: 
            tmp = bboxVec1[1]
            bboxVec1[1] = bboxVec2[1]
            bboxVec2[1] = tmp
        if bboxVec1[2] > bboxVec2[2]: 
            tmp = bboxVec1[2]
            bboxVec1[2] = bboxVec2[2]
            bboxVec2[2] = tmp

        if not (self.params.transX==0.0 and self.params.transY==0.0 and self.params.transZ==0.0):
            bboxVec1[0] = bboxVec1[0] + self.params.transX
            bboxVec2[0] = bboxVec2[0] + self.params.transX
            bboxVec1[1] = bboxVec1[1] + self.params.transY
            bboxVec2[1] = bboxVec2[1] + self.params.transY
            bboxVec1[2] = bboxVec1[2] + self.params.transZ
            bboxVec2[2] = bboxVec2[2] + self.params.transZ

        bbox.setXMinMax([bboxVec1[0], bboxVec2[0]])
        bbox.setYMinMax([bboxVec1[1], bboxVec2[1]])
        bbox.setZMinMax([bboxVec1[2], bboxVec2[2]])

        Neg2Gui.theNegMsgHandler().sendBBox(self.key, bbox)

        # send transformation to COVER
        msg = coGRObjTransformMsg(self.covise_key, *numpy.reshape(matrix, (16,)))
        covise.sendRendMsg(msg.c_str())

    # set translation of the transformModule
    def _setTransform(self):
        if self.covise_key == 'No key':
            # set transformation, so the modules will be created if necessary even if theres no covise_key
            if hasattr(self, 'importModule') and self.importModule:
                self.importModule.setRotation(self.params.rotAngle, self.params.rotX, self.params.rotY, self.params.rotZ)
                self.importModule.setTranslation(self.params.transX, self.params.transY, self.params.transZ)
            return False
        if not hasattr( self.params, 'rotX'):
            return False
        self.importModule.setTranslation(self.params.transX, self.params.transY, self.params.transZ)
        self.importModule.setRotation(self.params.rotAngle, self.params.rotX, self.params.rotY, self.params.rotZ)
        if not hasattr(self, '_sendToCOVER') or  not self._sendToCOVER:
            #Neg2Gui.theNegMsgHandler().sendBBox(self.key, self.importModule.getBox(True))
            self._sendBox()


    def _sendBox(self):
        if self._box != self.importModule.getBox(True):
            self._box = self.importModule.getBox(False)
            Neg2Gui.theNegMsgHandler().sendBBox(self.key, self._box)

class PartTransformParams(object):
    def __init__(self):
        PartTransformParams.mergeDefaultParams(self) # explicitly call mergeDefaultParams of this class

    def mergeDefaultParams(self):
        defaultParams = {
            'rotX' : 1.0,
            'rotY' : 0.0,
            'rotZ' : 0.0,
            'rotAngle' : 0.0,
            'transX' : 0.0,
            'transY' : 0.0,
            'transZ' : 0.0,
            'scaleX' : 1.0,
            'scaleY' : 1.0,
            'scaleZ' : 1.0
        }
        mergeGivenParams(self, defaultParams)

