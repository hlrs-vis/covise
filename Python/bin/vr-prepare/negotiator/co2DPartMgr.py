
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH
from KeydObject import coKeydObject, globalKeyHandler, RUN_ALL, TYPE_2D_PART, TYPE_2D_COMPOSED_PART, VIS_STREAMLINE, VIS_MOVING_POINTS, VIS_PATHLINES
import Neg2Gui
from ImportSampleManager import ImportSample2DModule
#from Utils import ReduceTimestepAsker
from ErrorManager import TimestepFoundError
from Utils import CopyParams

class co2DPartMgr(coKeydObject):
    """ class handling 2D Parts """
    def __init__(self):
        coKeydObject.__init__(self, TYPE_2D_PART, '2D')
        self.params = co2DPartMgrParams()
        self.name = self.params.name
        self.redFactor = False

    def init( self, partcase, reductionFactor=None ):
        self.importModule = ImportSample2DModule( partcase )
        varNotFound = self.importModule.readPartcase()
        for var in varNotFound:
            Neg2Gui.theNegMsgHandler().raiseVariableNotFound(var)     
        self.params.partcase = partcase
        if reductionFactor!=None:
            self.setReductionFactor(reductionFactor)
        if self.importModule.getIsTransient():
            Neg2Gui.theNegMsgHandler().sendIsTransient(True)
            if not self.redFactor:
                return False
        #else:
            #Neg2Gui.theNegMsgHandler().sendIsTransient(False)
        return True

    def addObject( self, visItem):
        visItem.setImport( self.importModule )
        coKeydObject.addObject( self, visItem )

    def setParams( self, params, negMsgHandler=None, sendToCover=True):
        coKeydObject.setParams( self, params)
        self.name = self.params.name

    def recreate(self, negMsgHandler, parentKey, offset):
        self.redFactor = False
        if not negMsgHandler.setReductionFactor:
            if not self.init( self.params.partcase ):
                raise  TimestepFoundError()
        else:
            self.init(self.params.partcase, negMsgHandler.reductionFactor)
        for visItem in self.objects:
            visItem.setImport( self.importModule )
        coKeydObject.recreate(self, negMsgHandler, parentKey, offset)
        
    def setImport(self, group):
        if not self.init(group.getPartCase()):
            if not negMsgHandler.setReductionFactor:
                raise  TimestepFoundError()
            else:
                self.setReductionFactor(negMsgHandler.reductionFactor)
                
    def setReductionFactor(self, reduction):
        self.redFactor = True
        if reduction:
            self.importModule.setReductionFactor(-1)
            Neg2Gui.theNegMsgHandler().internalRecvReductionFactor(self.importModule.getReductionFactor(), self.importModule.getNumTimeSteps())
        Neg2Gui.theNegMsgHandler().internalRecvReductionFactor(None, self.importModule.getNumTimeSteps())

    def run(self, runmode, negMsgHandler=None):
        coKeydObject.run(self, runmode, negMsgHandler)

    def delete(self, isInitialized, negMsgHandler=None):
        if isInitialized:
            for key in list(globalKeyHandler().getAllElements().keys()):
                obj = globalKeyHandler().getObject(key)
                if obj:
                    # remove composed parts containing this part as well
                    if (obj.typeNr == TYPE_2D_COMPOSED_PART) and (self.key in obj.params.subKeys):
                        obj.delete(isInitialized, negMsgHandler)
                    # remove startFrom2DPart association
                    if obj.typeNr in [VIS_STREAMLINE, VIS_MOVING_POINTS, VIS_PATHLINES] and (obj.params.use2DPartKey == self.key):
                        newParams = CopyParams(obj.params)
                        newParams.use2DPartKey = None
                        obj.setParams(newParams)
                        if negMsgHandler:
                            negMsgHandler.sendParams(obj.key, newParams)
                            obj.run(RUN_ALL, negMsgHandler)
            if hasattr(self, "importModule") and self.importModule:
                self.importModule.delete()
        coKeydObject.delete(self, isInitialized, negMsgHandler)

class co2DPartMgrParams(object):
    """ parameters of class co2DPartMgr """
    def __init__(self):
        self.name      = 'test.2DPart'
        self.partcase  = None
