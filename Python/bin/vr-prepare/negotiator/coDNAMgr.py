

# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH
import numpy

from KeydObject import TYPE_DNA_ITEM, TYPE_DNA_MGR, globalKeyHandler, RUN_ALL
from VisItem import VisItem, VisItemParams
from PartTransform import PartTransform, PartTransformParams
from Utils import multMatVec, transformationMatrix, ParamsDiff, mergeGivenParams
from coGRMsg import coGRMsg
from coGRMsg import coGRObjTransformMsg, coGRObjSetConnectionMsg, coGRObjMovedMsg
import covise
from printing import InfoPrintCapable
import Neg2Gui

import re

### global prints
_infoer = InfoPrintCapable()
_infoer.doPrint =   False#True #

#ID String
DNA_ID_STRING = 'DNA:'
DNA_APPENDIX_ID_STRING = 'DNA'

# defines for coloring choice
NO_COLOR = 0
RGB_COLOR = 1
MATERIAL = 2
VARIABLE = 3


class coDNAMgr(VisItem):
    """ class to handle covise keys for visability """
    def __init__(self):
        VisItem.__init__(self, TYPE_DNA_MGR, 'DNAMgr')
        globalKeyHandler().globalDNAMgrKey = self.key
        self.params = coDNAMgrParams()

    def recreate(self, negMsgHandler, parentKey, offset):
        coDNAMgrParams.mergeDefaultParams(self.params) # explicitly call mergeDefaultParams of this class
        # discard mapping from coprj, since we get different registered scene graph items from COVER
        #self.params.openCOVER2key = {}
        VisItem.recreate(self, negMsgHandler, parentKey, offset)
        globalKeyHandler().globalDNAMgrKey = self.key

        self.sendVisibility()

    def registerCOVISEkey( self, covise_keys):
        """ called during registration if key received from COVER """
        _infoer.function = str(self.registerCOVISEkey)
        _infoer.write("")#for printout
        #_infoer.write("%s" % covise_key)
        
        """ looking for the prefix DNA_ID_STRING """
        coviseKeyFound = covise_keys.find(DNA_ID_STRING)

        if coviseKeyFound >-1:
            """ Separating the message into a list of tuples [key ;; parentkey] """
            key_list = covise_keys.split('\t')
            del key_list[0]

            for relation in key_list:
                """ split key from parent key """
                keys = relation.split(';;')

                covise_key = keys[0]

                if len(keys) > 1:
                    if not covise_key in self.params.openCOVER2key:
                        parent_key = keys[1]
                        """ Attach to parent if found """
                        if parent_key in self.params.openCOVER2key:
                            obj = Neg2Gui.theNegMsgHandler().internalRequestObject( TYPE_DNA_ITEM, self.params.openCOVER2key[parent_key], None)
                        else:
                            obj = Neg2Gui.theNegMsgHandler().internalRequestObject( TYPE_DNA_ITEM, self.key, None)

                        obj.covise_key = covise_key
                        self.params.openCOVER2key[ covise_key ] = obj.key

                        """ cutting of suffix string for GUI """
                        #suffix_found = covise_key.find("_OUT");
                        obj.params.name = re.sub(r'_' + DNA_APPENDIX_ID_STRING + '_\d+$', '', covise_key) # covise_key
                        obj.sendParams()
                    else:
                        #### for recreate
                        obj = globalKeyHandler().getObject(self.params.openCOVER2key[covise_key])
                        obj.covise_key = covise_key

                        # send to COVER
                        obj.sendTransformation()
                        obj.sendVisibility()
                        for conn in obj.params.connectionPoints:
                                            obj.sendConnections(conn, [obj.params.connectionPoints[conn],obj.params.connectionPointsDisable[conn]])
                        # send to GUI
                        obj.sendParams()
                        

            return (True, True)

        return (False, False)

    def getItem(self, covise_keys):
        """ looking for the prefix DNA_ID_STRING """
        coviseKeyFound = covise_keys.find(DNA_ID_STRING)

        if coviseKeyFound >-1:
            """ Separating the message into a list of tuples [key ;; parentkey] """
            key_list = covise_keys.split('\t')
            del key_list[0]

            for relation in key_list:
                """ split key from parent key """
                keys = relation.split(';;')
                covise_key = keys[0]
                
                if len(keys) > 1:
                    if covise_key in self.params.openCOVER2key:
                        return globalKeyHandler().getObject(self.params.openCOVER2key[covise_key])
        return None
        


    def deleteKey(self, key):
       """ delete the covise_key from the openCOVER2key dictionary"""
       obj = globalKeyHandler().getObject(key)
       if obj.covise_key in self.params.openCOVER2key:
          del self.params.openCOVER2key[obj.covise_key]

class coDNAMgrParams(VisItemParams):
    def __init__(self):
        VisItemParams.__init__(self)
        self.name         = 'DNAMgrParams'
        coDNAMgrParams.mergeDefaultParams(self) # explicitly call mergeDefaultParams of this class

    def mergeDefaultParams(self):
        defaultParams = {
            'openCOVER2key' : {}
        }
        mergeGivenParams(self, defaultParams)

class coDNAItem(VisItem, PartTransform):
    """ class to handle viewpoints """
    def __init__(self):
        VisItem.__init__(self, TYPE_DNA_ITEM, 'DNA')
        PartTransform.__init__(self)
        self.params = coDNAItemParams()

    def setParams(self, params, negMsgHandler = None, sendToCover = True):
        print("setParams ", self.params.name)
        realChange = ParamsDiff(self.params, params)
        VisItem.setParams(self, params, negMsgHandler, sendToCover)

        #if ('rotAngle' in realChange or (self.params.rotAngle > 0 and ('rotX' in realChange or 'rotY' in realChange or 'rotZ' in realChange)) or \
        #    ('transX' in realChange or 'transY' in realChange or 'transZ' in realChange)): 
        self.sendTransformation()        
        
        #send all connections
        for conn in self.params.connectionPoints:
            self.sendConnections(conn, [self.params.connectionPoints[conn], self.params.connectionPointsDisable[conn]])
                        
        # check if need connections is on and set variable nextPresStep
        self.params.nextPresStep = True
        if self.params.needConn:
            for cp in self.params.connectionPoints:
                if self.params.connectionPointsDisable[cp] and not self.params.connectionPoints[cp]:
                    self.params.nextPresStep = False
                   
        
    def sendTransformation(self):
        if not self.covise_key=='No key':        
            matrix = transformationMatrix(self.params)
            # send transformation to COVER
            self.params.matrix[3] = self.params.transX
            self.params.matrix[7] = self.params.transY
            self.params.matrix[11] = self.params.transZ            
            msg = coGRObjTransformMsg(self.covise_key, self.params.matrix[0], self.params.matrix[1], self.params.matrix[2], self.params.matrix[3], self.params.matrix[4], self.params.matrix[5], self.params.matrix[6], self.params.matrix[7], self.params.matrix[8], self.params.matrix[9], self.params.matrix[10], self.params.matrix[11], self.params.matrix[12], self.params.matrix[13], self.params.matrix[14], self.params.matrix[15])
            #msg = coGRObjMovedMsg(self.covise_key, self.params.transX, self.params.transY, self.params.transZ, self.params.rotX, self.params.rotY, self.params.rotZ, self.params.rotAngle )
            covise.sendRendMsg(msg.c_str())
            return True
        return False

    def sendConnections(self, name, stateList):
        if not self.covise_key=='No key':        
            conn = name.split(" - ")
            # send Connection to COVER
            if len(conn) == 2:
                if str(name) in self.params.connectedUnit: connUnit = str(self.params.connectedUnit[str(name)])
                else: connUnit=" "
                #print("send Connection ", name, connUnit)
                if not(stateList[0]==1 and connUnit==" "):
                    msg = coGRObjSetConnectionMsg( self.covise_key, str(conn[0]), str(conn[1]), int(stateList[0]), int(stateList[1]),  connUnit)
                    covise.sendRendMsg(msg.c_str())
                    return True
        return False

    def recreate(self, negMsgHandler, parentKey, offset):
        coDNAItemParams.mergeDefaultParams(self.params) # explicitly call mergeDefaultParams of this class
        PartTransform.recreate(self, negMsgHandler, parentKey, offset)   
        VisItem.recreate(self, negMsgHandler, parentKey, offset)
        for conn in self.params.connectionPoints:
            self.sendConnections(conn, [self.params.connectionPoints[conn],self.params.connectionPointsDisable[conn]])
        self.sendTransformation()

    def __update(self, negMsgHandler):
        """ __update is called from the run method to update the module parameter before execution
            + do init 
            + update module parameters """
        VisItem.__update(negMsgHandler)

    def delete(self, isInitialized, negMsgHandler=None):
        globalKeyHandler().getObject(globalKeyHandler().globalDNAMgrKey).deleteKey(self.key)
        return VisItem.delete(self, isInitialized, negMsgHandler)

class coDNAItemParams(VisItemParams, PartTransformParams):
    def __init__(self):
        VisItemParams.__init__(self)
        PartTransformParams.__init__(self)
        self.name   = 'DNAItem'
        self.isVisible = True
        self.connectionPoints = {}
        self.connectionPointsDisable = {}            
        self.connectedUnit = {}
        self.matrix = [1,0,0,0 ,0,1,0,0 ,0,0,1,0 ,0,0,0,1]
        coDNAItemParams.mergeDefaultParams(self) # explicitly call mergeDefaultParams of this class

    def mergeDefaultParams(self):
        defaultParams = {
            'needConn' : False
        }
        mergeGivenParams(self, defaultParams)
