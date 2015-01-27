# Part of the vr-prepare program
# Copyright (c) 2007 Visenso GmbH

# parent class for visItems with Interactor
#
# every visItem, which uses an interactor should include this class
# this class implemts all necessary functions to execute, update, etc. an interactor


from VRPCoviseNetAccess import (
    connect,
    disconnect,
    ConnectionPoint,
    theNet)

from VisItem import VisItem, VisItemParams
from KeydObject import RUN_ALL
import covise
from Utils import  ParamsDiff, convertAlignedRectangleToCutRectangle, convertAlignedRectangleToGeneral, AxisAlignedRectangleIn3d
from coGRMsg import coGRMsg, coGRObjMoveInterMsg, coGRObjVisMsg, coGRObjRestrictAxisMsg
from printing import InfoPrintCapable
_infoer = InfoPrintCapable()
_infoer.doPrint =False # True # 

TRACER = 0
CUT = 1


class PartInteractorVis(object):
    '''VisItem for an interactor'''
    def __init__(self, format=TRACER):
        '''typeNr of the visitem and name of the visitem'''
        _infoer.function = str(self.__init__)
        _infoer.write("")

        self.__format = format

    def _init(self, negMsgHandler):
        '''called from _update '''
        _infoer.function = str(self._init)
        _infoer.write("")

        if not hasattr(self.params, 'boundingBox'):
            return 
        if self.params.alignedRectangle==AxisAlignedRectangleIn3d() and not self.fromRecreation and self.params.boundingBox!=None:
            bb = self.params.boundingBox
            aar = self.params.alignedRectangle
            aar.middle = 0.05*bb.getXMinMax()[1] + (1-0.05)*bb.getXMinMax()[0], \
                              0.5 * (bb.getYMinMax()[1] + bb.getYMinMax()[0]), \
                              0.5 * (bb.getZMinMax()[1] + bb.getZMinMax()[0])
            # don't overwrite parameter, because e.g. 2d-streamlines have 'line' not 'x'
            #aar.orthogonalAxis = 'x'
            goodSize = 0.4 * (bb.getXMinMax()[1] - bb.getXMinMax()[0])
            aar.lengthA = goodSize
            aar.lengthB = goodSize

    def _update(self, negMsgHandler):
        '''_update is called from the run method to update the module parameter before execution'''
        _infoer.function = str(self._update)
        _infoer.write("")

        PartInteractorVis._init(self, negMsgHandler)

    def setFormat( self, format ):
        """ called after recreation to be sure that also project files from version 1 are supported """
        _infoer.function = str(self.setFormat)
        _infoer.write("")

        self.__format = format

    def run(self, runmode, negMsgHandler):
        _infoer.function = str(self.run)
        _infoer.write("")

        if runmode==RUN_ALL:
            _infoer.function = str(self.run)
            _infoer.write("go")

            if not hasattr(self, 'importModule'):
                return

            PartInteractorVis._update(self, negMsgHandler)

    def setParams( self, params, negMsgHandler=None, sendToCover=True, realChange=None):
        """ set parameters from outside
            + init cutting surface module if necessary
            + mainly receive parameter changes from Gui
            + send status messages to COVER if state has changed
        """
        _infoer.function = str(self.setParams)
        _infoer.write("")

        if realChange==None:
            realChange = ParamsDiff( self.params, params )

        if realChange == []:
            return

        if sendToCover and \
            not 'showInteractor' in realChange and \
            not 'showSmoke' in realChange and \
            not 'isVisible' in realChange and \
            not 'boundingBox' in realChange and \
            not 'attachedClipPlane_index' in realChange and \
            not 'attachedClipPlane_offset' in realChange and \
            not 'attachedClipPlane_flip' in realChange:
                self.sendInteractor()

        if 'showInteractor' in realChange and sendToCover: self.sendInteractorStatus()
        if 'showSmoke' in realChange and sendToCover: self.sendSmokeStatus()
        if 'alignedRectangle' in realChange and sendToCover: self.sendInteractorAxis()


    def sendInteractor(self):
        """ send interactor geometry to cover """
        _infoer.function = str(self.sendInteractor)
        _infoer.write("")

        if self.keyRegistered():
            _infoer.function = str(self.sendInteractor)
            _infoer.write("sendInteractor for key %s in mode %s" % ( self.covise_key, self.__format ) )
            if self.__format==TRACER:
                if self.params.alignedRectangle.orthogonalAxis == 'line':
                    rec = convertAlignedRectangleToGeneral( self.params.alignedRectangle )
                    startPoint = self.params.alignedRectangle.getStartPoint()
                    msg = coGRObjMoveInterMsg( coGRMsg.MOVE_INTERACTOR, self.covise_key, "s1", startPoint[0], startPoint[1], startPoint[2] )
                    covise.sendRendMsg(msg.c_str())
                    endPoint = self.params.alignedRectangle.getEndPoint()
                    msg = coGRObjMoveInterMsg( coGRMsg.MOVE_INTERACTOR, self.covise_key, "s2", endPoint[0], endPoint[1], endPoint[2] )
                    covise.sendRendMsg(msg.c_str())
                    msg = coGRObjMoveInterMsg( coGRMsg.MOVE_INTERACTOR,self.covise_key, "direction", rec.direction[0], rec.direction[1], rec.direction[2] )
                    covise.sendRendMsg(msg.c_str())
                else:
                    rec = convertAlignedRectangleToGeneral( self.params.alignedRectangle )
                    msg = coGRObjMoveInterMsg( coGRMsg.MOVE_INTERACTOR, self.covise_key, "s1", rec.pointA[0], rec.pointA[1], rec.pointA[2] )
                    covise.sendRendMsg(msg.c_str())
                    msg = coGRObjMoveInterMsg( coGRMsg.MOVE_INTERACTOR,self.covise_key, "s2", rec.pointC[0], rec.pointC[1], rec.pointC[2] )
                    covise.sendRendMsg(msg.c_str())
                    msg = coGRObjMoveInterMsg( coGRMsg.MOVE_INTERACTOR,self.covise_key, "direction", rec.direction[0], rec.direction[1], rec.direction[2] )
                    covise.sendRendMsg(msg.c_str())
            elif self.__format==CUT:
                rec = convertAlignedRectangleToCutRectangle( self.params.alignedRectangle )
                msg = coGRObjMoveInterMsg( coGRMsg.MOVE_INTERACTOR,self.covise_key, "normal", rec.normal[0], rec.normal[1], rec.normal[2] )
                covise.sendRendMsg(msg.c_str())
                msg = coGRObjMoveInterMsg( coGRMsg.MOVE_INTERACTOR, self.covise_key, "point", rec.point[0], rec.point[1], rec.point[2] )
                covise.sendRendMsg(msg.c_str())
            else:
                print("unknown format")
        #else:
        #    print("!self.keyRegistered")

    def sendInteractorStatus(self):
        """ send visibility of interactor msg to cover """
        _infoer.function = str(self.sendInteractorStatus)
        _infoer.write("")

        if self.keyRegistered():
            _infoer.function = str(self.sendInteractorStatus)
            _infoer.write("send")
            msg = coGRObjVisMsg( coGRMsg.INTERACTOR_VISIBLE, self.covise_key, self.params.showInteractor )
            covise.sendRendMsg(msg.c_str())

    def sendInteractorPosibility(self):
        """ send usability of interactor msg to cover """
        _infoer.function = str(self.sendInteractorPosibility)
        _infoer.write("")

        if self.keyRegistered():
            _infoer.function = str(self.sendInteractorPosibility)
            _infoer.write("send")
            msg = coGRObjVisMsg( coGRMsg.INTERACTOR_USED, self.covise_key, self.params.use2DPartKey==None )
            covise.sendRendMsg(msg.c_str())

    def sendSmokeStatus(self):
        """ send status of smoke to cover """
        _infoer.function = str(self.sendSmokeStatus)
        _infoer.write("")

        if self.keyRegistered():
            _infoer.function = str(self.sendSmokeStatus)
            _infoer.write("send")
            msg = coGRObjVisMsg( coGRMsg.SMOKE_VISIBLE, self.covise_key, self.params.showSmoke )
            covise.sendRendMsg(msg.c_str())

    def sendInteractorAxis(self):
        """ send axis of the interactor to cover """
        _infoer.function = str(self.sendInteractorAxis)
        _infoer.write("")

        if self.keyRegistered():
            _infoer.function = str(self.sendInteractorAxis)
            _infoer.write("send")
            if self.params.alignedRectangle.orthogonalAxis == 'x' and self.params.alignedRectangle.rotY==0 \
                and self.params.alignedRectangle.rotZ==0:
                axis = 'xAxis'
            elif self.params.alignedRectangle.orthogonalAxis == 'y' and self.params.alignedRectangle.rotX==0 \
                and self.params.alignedRectangle.rotZ==0:
                axis = 'yAxis'
            elif self.params.alignedRectangle.orthogonalAxis == 'z' and self.params.alignedRectangle.rotX==0 \
                and self.params.alignedRectangle.rotY==0:
                axis = 'zAxis'
            else:
                axis = 'freeAxis'
            msg = coGRObjRestrictAxisMsg( coGRMsg.RESTRICT_AXIS, self.covise_key, axis )
            covise.sendRendMsg(msg.c_str())

class PartInteractorVisParams(object):
    '''Params for helper class for interactor'''
    def __init__(self):
        self.showInteractor = True
        self.smokeBox = None # of type Box
        self.showSmoke = False
        self.alignedRectangle = AxisAlignedRectangleIn3d()

