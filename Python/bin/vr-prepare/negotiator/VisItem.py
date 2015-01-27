
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH



import covise
import os
from printing import InfoPrintCapable

from coGRMsg import coGRMsg, coGRObjVisMsg, coGRObjSetCaseMsg, coGRObjSetNameMsg
from KeydObject import coKeydObject, globalKeyHandler
from Utils import mergeGivenParams
from VRPCoviseNetAccess import theNet, connect, globalRenderer, RWCoviseModule
from coPyModules import PerformerScene, CoverDocument
import coCaseMgr
import Neg2Gui
import VRPCoviseNetAccess
import Utils
import covise

_infoer = InfoPrintCapable()
_infoer.doPrint = False # True


class VisItem(coKeydObject):
    """ a visitem gets its input from an importModule and
        gives a geometry object out which is connected to the COVER """

    def __init__(self, typeNr, name='VisItem' ):
        coKeydObject.__init__(self, typeNr, name)
        self.__init()
        self.importModule = None

    def __init(self):
        # key which is defined by COVER
        self.covise_key = 'No key'
        # the module that can check if a covise_key is created by this VisItem
        self.__creationModule = None
        # the module that writes the covise object into a file for testing
        self.__debugRwModule = None

        # register ui action
        Neg2Gui.theNegMsgHandler().registerVisItem(self)

    def recreate(self, negMsgHandler, parentKey, offset):
        VisItemParams.mergeDefaultParams(self.params) # explicitly call mergeDefaultParams of this class
        self.__init()
        coKeydObject.recreate(self, negMsgHandler, parentKey, offset)

    def registerCOVISEkey( self, covise_key):
        """ check if object name was created by this visItem
            and if yes store it """
        if not self.__creationModule==None:
            if self.createdKey( covise_key ):
                firstTime = not self.keyRegistered()
                self.covise_key = covise_key
                self.sendVisibility()
                self.sendCaseName()
                self.sendCaseTransform()
                self.sendName()
                return (True, firstTime)
        return (False, False)

    def keyRegistered(self):
        if self.covise_key=='No key':
            return False
        return True

    def __getCaseName(self):
        caseName = None
        key = self.parentKey
        #for i in range(3):
        while key >= 0 and caseName==None:
            parent = globalKeyHandler().getObject(key)
            if isinstance(parent,coCaseMgr.coCaseMgr ):
                caseName = parent.params.name
            else :
                key = parent.parentKey
        return caseName

    def getCoObjName(self):
        return self.importModule.getCoObjName()

    def createdKey(self, key):
        importKey = self.getCoObjName()
        _infoer.function = str(self.createdKey)
        _infoer.write("check my key %s against COVER key %s" % (importKey, key) )

        posCover = key.find("(")
        posImport = importKey.find("OUT")
        # if the beginning of my key and COVER key are equal, dann the key exists
        return ( importKey[0:posImport-1]==key[0:posCover] )

    def sendVisibility(self):
        """ send visibility msg to cover """
        if not self.covise_key=='No key':
            msg = coGRObjVisMsg( coGRMsg.GEO_VISIBLE, self.covise_key, self.params.isVisible )
            covise.sendRendMsg(msg.c_str())

    def sendName(self):
        """ send name of the VisItem to cover """
        if not self.covise_key=='No key':
            msg = coGRObjSetNameMsg( coGRMsg.SET_NAME, self.covise_key, str(self.params.name) )
            covise.sendRendMsg(msg.c_str())

    def sendCaseName(self):
        """ send visibility msg to cover """
        if not self.covise_key=='No key':
            caseName = self.__getCaseName()
            if caseName:
                msg = coGRObjSetCaseMsg( coGRMsg.SET_CASE, self.covise_key, caseName)
                covise.sendRendMsg(msg.c_str())

    def sendCaseTransform(self):
        """ send case transformation to COVER """
        key = self.parentKey
        sentMatrix = False
        while key >= 0 and not sentMatrix:
            parent = globalKeyHandler().getObject(key)
            if isinstance(parent,coCaseMgr.coCaseMgr ):
                # send transformation matrix for this visItem
                parent._sendMatrix()
                sentMatrix = True
            else :
                key = parent.parentKey


    def reconnect(self):
        coKeydObject.reconnect(self)
        if self.__creationModule :
            self.connectToCover(self.__creationModule)

    def connectToCover( self, visConnectionModule ):
        self.__creationModule = visConnectionModule
        VRPCoviseNetAccess.connect(
            visConnectionModule.connectionPoint(),
            globalRenderer().connectionPoint() )

        if os.getenv('VR_PREPARE_DEBUG_VISITEMS_DIR'):
            #filename = os.getenv('VR_PREPARE_DEBUG_VISITEMS_DIR') + '/' + self.params.name + "_%s.covise" % self.key
            #rw = RWCoviseModule(filename, True)
            #filename = rw.gridPath()
            #connect( visConnectionModule.connectionPoint(), rw.inConnectionPoint() )
            filename = os.getenv('VR_PREPARE_DEBUG_VISITEMS_DIR') + '/' \
                       + self.params.name + "_" + str(self.key) + ".covise"
            self.__debugRwModule = RWCoviseModule(filename, True)
            connect( visConnectionModule.connectionPoint(), self.__debugRwModule.inConnectionPoint() )

    def disconnectFromCover(self, visConnectionModule):
        coverModule = globalRenderer().connectionPoint().module
        visItemModule = visConnectionModule.connectionPoint().module
        theNet().disconnectModuleModule(coverModule, visItemModule)

    def delete(self, isInitialized, negMsgHandler=None):
        if isInitialized:
            if os.getenv('VR_PREPARE_DEBUG_VISITEMS_DIR') and hasattr(self, '_VisItem__debugRwModule') and self.__debugRwModule:
                self.__debugRwModule.remove()

        return coKeydObject.delete(self, isInitialized, negMsgHandler)

    def setImport(self, group):
        self.importModule = group

    def setParams(self, params, negMsgHandler=None, sendToCover=True):
        visibility_changed = (params.isVisible!=self.params.isVisible)
        self.params.isVisible = params.isVisible

        # check if params contains more than isVisible
        if issubclass( params.__class__, VisItemParams):

            name_changed = (params.name!=self.params.name)

            coKeydObject.setParams(self, params)

            if name_changed:
                self.sendName()

        if visibility_changed:
            self.sendVisibility()

    def sendParams(self):
        """ send Params to Gui """
        Neg2Gui.theNegMsgHandler().sendParams(self.key, self.params )

    def recvParams( self, isVisible):
        self.params.isVisible = isVisible
        self.sendParams()

    def updateDebugFilename(self, PresentationStepKey):
        if hasattr(self, '_VisItem__debugRwModule') and self.__debugRwModule:
            filename = os.getenv('VR_PREPARE_DEBUG_VISITEMS_DIR') + '/' \
                       + 'STEP' + str(PresentationStepKey) + "_" \
                       + self.params.name + "_" + str(self.key) + ".covise"
            self.__debugRwModule.setGridPath(filename, True)

class VisItemParams(object):
    def __init__(self):
        self.name      = 'VisItemParam'
        self.isVisible = False
        VisItemParams.mergeDefaultParams(self) # explicitly call mergeDefaultParams of this class

    def mergeDefaultParams(self):
        defaultParams = {
            'nextPresStep' : True
        }
        mergeGivenParams(self, defaultParams)

    def isStaticParam(self, paramname):
        return paramname in ["name"]

# eof
