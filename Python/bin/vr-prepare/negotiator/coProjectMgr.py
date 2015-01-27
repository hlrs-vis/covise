
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH


from KeydObject import coKeydObject, globalKeyHandler, RUN_ALL, TYPE_PROJECT, TYPE_3D_PART, TYPE_2D_PART, VIS_2D_RAW, TYPE_DNA_MGR, TYPE_TRACKING_MGR, TYPE_GENERIC_OBJECT_MGR
from Utils import ParamsDiff, mergeGivenParams
from BoundingBox import Box
import coprjVersion
import covise
from coGRMsg import coGRMsg, coGRAnimationOnMsg, coGRSetTimestepMsg, coGRSetAnimationSpeedMsg

from vrpconstants import REDUCTION_FACTOR, SELECTION_STRING

class coProjectMgr(coKeydObject):
    """ class to handle project files """
    def __init__(self):
        #only for saving (has to be before coKeydObject.recreate because the KeyHandler will be deleted)
        self.__keyHandler = globalKeyHandler(None, True)

        coKeydObject.__init__(self, TYPE_PROJECT, 'Project')

        self.params = coProjectMgrParams()
        self.name = self.params.name

        # new projects already have latest coprj file format -> overwrite default coprjVersion=0
        self.params.coprjVersion = coprjVersion.version
        self.params.originalCoprjVersion = coprjVersion.version


    def recreate(self, negMsgHandler, parentKey, offset):
        coProjectMgrParams.mergeDefaultParams(self.params) # explicitly call mergeDefaultParams of this class
        # force overwriting of old globalKeyHandler
        globalKeyHandler(self.__keyHandler, (offset==0) )

        # work around
        """
        Utils.addServerHostFromConfig()
        """
        coKeydObject.recreate(self, negMsgHandler, parentKey, offset)

        existingTrackingManager = False
        existingDNAManager = False
        existingGenericObjectManager = False

        for key in globalKeyHandler().getAllElements():
            obj = globalKeyHandler().getObject(key)
            if obj:
                # set crop min/max and selectionString (GetSubset) for all recreated parts of project
                if obj.typeNr in [TYPE_3D_PART, TYPE_2D_PART]:
                    # set params for import
                    globalKeyHandler().getObject(key).importModule.setCropMin(self.params.cropMin[0], self.params.cropMin[1], self.params.cropMin[2])
                    globalKeyHandler().getObject(key).importModule.setCropMax(self.params.cropMax[0], self.params.cropMax[1], self.params.cropMax[2])

                    if self.params.filterChoice == SELECTION_STRING:
                        globalKeyHandler().getObject(key).importModule.setSelectionString(self.params.selectionString)
                    if self.params.filterChoice == REDUCTION_FACTOR:
                        globalKeyHandler().getObject(key).importModule.setReductionFactor(self.params.reductionFactor)
                if obj.typeNr == TYPE_TRACKING_MGR:
                    existingTrackingManager = True
                if obj.typeNr == TYPE_DNA_MGR:
                    existingDNAManager = True
                if obj.typeNr == TYPE_GENERIC_OBJECT_MGR:
                    existingGenericObjectManager = True

        if not existingTrackingManager: # create TrackingManager if it's an old project without TrackingManager
            negMsgHandler.internalRequestObjectDuringRecreate(TYPE_TRACKING_MGR, self.key)

        if not existingDNAManager: # create DNAManager if it's an old project without TrackingManager
            negMsgHandler.internalRequestObjectDuringRecreate(TYPE_DNA_MGR, self.key)

        if not existingGenericObjectManager: # create GenericObjectManager if it's an old project without GenericObjectManager
            negMsgHandler.internalRequestObjectDuringRecreate(TYPE_GENERIC_OBJECT_MGR, self.key)

        self.sendMessages()

    def run(self, runmode, negMsgHandler=None):
        coKeydObject.run(self, runmode, negMsgHandler)
        self.sendMessages()

    def sendMessages(self):
        msg = coGRAnimationOnMsg( self.params.animateOn)
        covise.sendRendMsg(msg.c_str())
        msg = coGRSetAnimationSpeedMsg( self.params.animationSpeed, self.params.animationSpeedMin, self.params.animationSpeedMax )
        covise.sendRendMsg(msg.c_str())
        if (not self.params.animateOn):
            msg = coGRSetTimestepMsg( self.params.actTimeStep, self.params.numTimeSteps)
            covise.sendRendMsg(msg.c_str())

    def setParams( self, params, negMsgHandler=None, sendToCover=True):
        coKeydObject.setParams(self, params)
        realChange = ParamsDiff( self.params, params )
        self.params.sync = params.sync
        self.params.reductionFactor = params.reductionFactor
        self.params.selectionString = params.selectionString
        self.params.filterChoice = params.filterChoice
        self.params.numTimeSteps = params.numTimeSteps
        self.params.cropMin = params.cropMin
        self.params.cropMax = params.cropMax
        self.params.animateOn = params.animateOn
        self.params.animationSpeed = params.animationSpeed
        self.params.actTimeStep= params.actTimeStep

    def __getSync( self, key, param ):
        """ get the directly synced keys/params (distance = 1 in sync-graph) """
        result = []
        for k,v in iter(self.params.sync.items()):
            if (key,param) == k:
                result.extend(v)
            if (key,param) in v:
                result.append(k)
        return list(set(result))

    def getSync(self, key, param):
        """ get all synced keys/params (distance >= 1 in sync-graph) """
        result = self.__getSync(key, param)
        appended = True

        while appended:
            appended = False
            for pair in result:
                r = self.__getSync(pair[0], pair[1])
                for p in r:
                    if p not in result and p != (key,param):
                        result.append(p)
                        appended = True

        return result

    def __getSyncKeys( self, key):
        """ get the directly synced keys (distance = 1 in sync-graph) """
        result = []
        for keypair in filter(lambda x: x[0]==key, self.params.sync.keys()):
            synclist = self.params.sync[keypair]
            for pair in synclist:
                if pair[0] not in result:
                    result.append(pair[0])
        return result

    def getSyncKeys(self, key):
        """ get all synced keys (distance >= 1 in sync-graph) """
        result = self.__getSyncKeys(key)
        appended = True

        while appended:
            appended = False
            for k in result:
                r = self.__getSyncKeys(k)
                for i in r:
                    if i not in result and i != key:
                        result.append(i)
                        appended = True

        return result

    def setReductionFactor(self, rf, negMsgHandler):
        if rf != self.params.reductionFactor:
            self.params.reductionFactor = rf

            for key in globalKeyHandler().getAllElements():
                if globalKeyHandler().getObject(key).typeNr in [TYPE_3D_PART, TYPE_2D_PART]:
                    # set params for import, but dont execute
                    globalKeyHandler().getObject(key).importModule.setReductionFactor(rf)

            #for key in globalKeyHandler().getAllElements():
                #if globalKeyHandler().getObject(key).typeNr in [TYPE_3D_PART, TYPE_2D_PART]:
                    #num = globalKeyHandler().getObject(key).importModule.getNumReductionModules()
                    ## need to add modules for reduce
                    #globalKeyHandler().getObject(key).importModule.setReductionFactor(rf)
                    ## need to reconnect the importmodules
                    #if num < globalKeyHandler().getObject(key).importModule.getNumReductionModules():
                        #globalKeyHandler().getObject(key).run(RUN_ALL, negMsgHandler)
                    ## execute the reduceSet Modules
                    #globalKeyHandler().getObject(key).importModule.setReductionFactor(rf, True)
                    ## set the color of 2d parts
                    #if globalKeyHandler().getObject(key).typeNr == TYPE_2D_PART:
                        #globalKeyHandler().getObject(key).run(RUN_ALL, negMsgHandler)

    def setSelectionString(self, selectionString):
        if selectionString != self.params.selectionString:
            self.selectionString = selectionString

            for key in globalKeyHandler().getAllElements():
                if globalKeyHandler().getObject(key).typeNr in [TYPE_3D_PART, TYPE_2D_PART]:
                    # set params for import, but dont execute
                    globalKeyHandler().getObject(key).importModule.setSelectionString(selectionString)

    def setCropMin(self, x, y, z):
        if self.params.cropMin != [x, y, z]:
            self.params.cropMin = [x, y, z]

            for key in globalKeyHandler().getAllElements():
                if globalKeyHandler().getObject(key).typeNr in [TYPE_3D_PART, TYPE_2D_PART]:
                    # set params for import, but dont execute
                    globalKeyHandler().getObject(key).importModule.setCropMin(x, y, z)

    def setCropMax(self, x, y, z):
        if self.params.cropMax != [x, y, z]:
            self.params.cropMax = [x, y, z]

            for key in globalKeyHandler().getAllElements():
                if globalKeyHandler().getObject(key).typeNr in [TYPE_3D_PART, TYPE_2D_PART]:
                    # set params for import, but dont execute
                    globalKeyHandler().getObject(key).importModule.setCropMax(x, y, z)

    def getPartsBoundingBox(self):
        """ get the bounding box from all originally unfiltered/untransformed/unreduced/un... import modules """

        box = Box( (0,0),(0,0),(0,0) )
        for key in globalKeyHandler().getAllElements():
            # get bounding box of all parts
            if globalKeyHandler().getObject(key) and globalKeyHandler().getObject(key).typeNr in [TYPE_3D_PART, TYPE_2D_PART]:
                tmpBB = globalKeyHandler().getObject(key).importModule.getBoxFromGeoRWCovise()
                box = box + tmpBB
        return box

    def getSelectionString(self):
        return self.params.selectionString

    def getCropMin(self):
        return self.params.cropMin

    def getCropMax(self):
        return self.params.cropMax

    def getReductionFactor(self):
        return self.params.reductionFactor

    def getNumTimeSteps(self):
        return self.params.numTimeSteps

class coProjectMgrParams(object):
    def __init__(self):
        self.name          = 'New Project'
        self.filename      = '.'
        self.creation_date = 'today'
        self.comment       = 'No comment'
        self.designer      = ' '
        self.region        = ' '
        self.division      = ' '
        self.serverHost    = None
        self.serverUser    = None
        self.serverTimeout = 0
        self.version = 7
        coProjectMgrParams.mergeDefaultParams(self) # explicitly call mergeDefaultParams of this class

    def mergeDefaultParams(self):
        defaultParams = {
            'sync' : {},
            'reductionFactor' : 1,
            'numTimeSteps' : 1,
            'cropMin' : [0, 0, 0],
            'cropMax' : [0, 0, 0],
            'selectionString' : "",
            'filterChoice' : REDUCTION_FACTOR,
            'animateOn' : True,
            'animationSpeed' : 25,
            'animationSpeedMin' : -25,
            'animationSpeedMax' : 25,
            'actTimeStep' : 1,
            'coprjVersion' : 0, # will be updated to the current version after loading
            'originalCoprjVersion' : 0 # will never be updated
        }
        mergeGivenParams(self, defaultParams)

    def isStaticParam(self, paramname):
        return paramname in ["name", "filename", "creation_date", "comment", "designer", "region", "division", "serverHost", "serverUser", "serverTimeout", "version", "coprjVersion", "originalCoprjVersion"]
