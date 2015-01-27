
# Part of the vr-prepare program for dc

# Copyright (c) 2006-2007 Visenso GmbH
import time
import covise
import numpy
from KeydObject import globalKeyHandler, coKeydObject, TYPE_2D_GROUP, TYPE_3D_GROUP, TYPE_2D_PART, TYPE_3D_PART, VIS_2D_STATIC_COLOR, TYPE_CASE
from coGroupMgr import coGroupMgrParams
from co3DPartMgr import co3DPartMgrParams
from co2DPartMgr import co2DPartMgrParams
from PartTransform import PartTransformParams
from Utils import ParamsDiff, transformationMatrix
from coGRMsg import coGRObjTransformCaseMsg

class coCaseMgr(coKeydObject):
    """ class handling cocase files """
    def __init__(self):
        coKeydObject.__init__(self, TYPE_CASE, 'case')
        self.params = coCaseMgrParams()
        self.group2d = None

    def initContent( self, negMsgHandler ):
        self.group2d = negMsgHandler.internalRequestObject(
            TYPE_2D_GROUP, self.key)
        gP = coGroupMgrParams()
        gP.name = 'Geometry (2D Parts)'
        self.group2d.setParams(gP)
        negMsgHandler.sendParams(self.group2d.key, gP)
        for part in self.params.filteredDsc.parts2d:
            part2d = negMsgHandler.internalRequestObject(
                TYPE_2D_PART, self.group2d.key, part)
            pP = co2DPartMgrParams()
            pP.name = part.name
            pP.partcase = part
            part2d.setParams(pP)
            negMsgHandler.sendParams(part2d.key, pP)

        group3d = negMsgHandler.internalRequestObject( TYPE_3D_GROUP, self.key)
        gP = coGroupMgrParams()
        gP.name = 'Grids (3D Parts)'
        group3d.setParams(gP)
        negMsgHandler.sendParams( group3d.key, gP)

        for part in self.params.filteredDsc.parts3d:
            part3d = negMsgHandler.internalRequestObject(
                TYPE_3D_PART, group3d.key, part)
            pP = co3DPartMgrParams()
            pP.name = part.name
            pP.partcase = part
            part3d.setParams(pP)
            negMsgHandler.sendParams(part3d.key, pP)

    def setParams( self, params, negMsgHandler):
        realChange = ParamsDiff( self.params, params )
        coKeydObject.setParams( self, params)
        if self.group2d==None:
            self.params.name = self.params.filteredDsc.name
            negMsgHandler.sendParams( self.key, self.params )
            self.initContent(negMsgHandler)
        # transformation matrix
        if 'rotAngle' in realChange or \
            (self.params.rotAngle != 0 and ('rotX' in realChange or \
                                            'rotY' in realChange or \
                                            'rotZ' in realChange)) or \
            ('transX' in realChange or 'transY' in realChange or 'transZ' in realChange) or \
            ('scaleX' in realChange or 'scaleY' in realChange or 'scaleZ' in realChange):
            self._sendMatrix()

    # send transform matrix to cover
    def _sendMatrix(self):
        if not hasattr( self.params, 'rotX'):
            return
        matrix = transformationMatrix(self.params)
        msg = coGRObjTransformCaseMsg(self.params.name, *numpy.reshape(matrix, (16,)))
        covise.sendRendMsg(msg.c_str())

    def recreate(self, negMsgHandler, parentKey, offset):
        coCaseMgrParams.mergeDefaultParams(self.params) # explicitly call mergeDefaultParams of this class
        coKeydObject.recreate(self, negMsgHandler, parentKey, offset)
        if offset>0 :
            globalKeyHandler().getObject(0).addObject(self)
        # send matrix in VisItem.py after case is registered in COVER
        #self._sendMatrix()

class coCaseMgrParams(PartTransformParams):
    def __init__(self):
        PartTransformParams.__init__(self)
        self.name = 'testCase'
        self.filteredDsc = None
        self.origDsc = None
        coCaseMgrParams.mergeDefaultParams(self) # explicitly call mergeDefaultParams of this class

    def mergeDefaultParams(self):
        PartTransformParams.mergeDefaultParams(self) # call mergeDefaultParams of PartTransformParams since coCaseMgr does not inherit from PartTransform


# eof
