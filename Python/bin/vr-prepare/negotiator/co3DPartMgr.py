
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH

from KeydObject import coKeydObject, globalKeyHandler, RUN_GEO, RUN_ALL, RUN_OCT, TYPE_3D_PART, TYPE_3D_COMPOSED_PART
from ImportSampleManager import ImportSample3DModule
import Part3DBoundingBoxVis
import Neg2Gui
import covise

from ErrorManager import TimestepFoundError

class co3DPartMgr(coKeydObject):
    """ class handling 3D Parts """
    def __init__(self):
        coKeydObject.__init__(self, TYPE_3D_PART, '3D')
        self.params = co3DPartMgrParams()
        self.name = self.params.name
        self.redFactor = False

    def init( self, partcase, reductionFactor=None ):
        self.importModule = ImportSample3DModule( partcase )
        varNotFound = self.importModule.readPartcase()
        for var in varNotFound:
            Neg2Gui.theNegMsgHandler().raiseVariableNotFound(var)
        if reductionFactor!=None:
            self.setReductionFactor(reductionFactor)
        # need for composed grid - otherwise the order of modules is wrong
        self.importModule.geoConnectionPoint()
        self.params.partcase = partcase
        if self.importModule.getIsTransient():
            Neg2Gui.theNegMsgHandler().sendIsTransient(True)
            if not self.redFactor:
                return False
        #else:
            #Neg2Gui.theNegMsgHandler().sendIsTransient(False)
        return True
                        
    def addObject( self, visItem):
        # check if visItem is a copy
        if ((not visItem.name == self.name) or self.name == 'test.3DPart') and type(visItem) == Part3DBoundingBoxVis.Part3DBoundingBoxVis:
            visItem.setImport( self.importModule )
        elif not type(visItem) == Part3DBoundingBoxVis.Part3DBoundingBoxVis and not type(visItem) == co3DPartMgr:
            visItem.setImport( self.importModule )
        coKeydObject.addObject( self, visItem )

    def setParams( self, params, negMsgHandler=None, sendToCover=True):
        coKeydObject.setParams( self, params)
        self.name = self.params.name

    def recreate(self, negMsgHandler, parentKey, offset):
        self.redFactor = False
        if not self.init( self.params.partcase ):
            if not negMsgHandler.setReductionFactor:
                raise  TimestepFoundError()
            else:
                self.setReductionFactor(negMsgHandler.reductionFactor)
        for visItem in self.objects:
            #check if visItem is a copy
            if not visItem.name == self.name and type(visItem) == Part3DBoundingBoxVis.Part3DBoundingBoxVis:
                visItem.setImport( self.importModule )
            elif  not type(visItem) == Part3DBoundingBoxVis.Part3DBoundingBoxVis and not type(visItem) == co3DPartMgr:
                visItem.setImport( self.importModule )
        coKeydObject.recreate(self, negMsgHandler, parentKey, offset)
                
    def setReductionFactor(self, reduction):
        self.redFactor = reduction
        if reduction:
            self.importModule.setReductionFactor(-1)
            Neg2Gui.theNegMsgHandler().internalRecvReductionFactor(self.importModule.getReductionFactor(), self.importModule.getNumTimeSteps())
        Neg2Gui.theNegMsgHandler().internalRecvReductionFactor(None, self.importModule.getNumTimeSteps())
        
    def run(self, runmode, negMsgHandler=None):
        if runmode==RUN_GEO:
            self.importModule.executeGeo()
        elif runmode==RUN_OCT:
            self.importModule.executeOct()
            # sample over first vel value
            if covise.coConfigIsOn("vr-prepare.UseSamplingModules", False):
                if not self.importModule.getIsTransient():
                    for v in self.params.partcase.variables:
                        if v.variableDimension==3:
                            self.importModule.executeSampleData(v.name)
        else:
            coKeydObject.run(self, runmode, negMsgHandler)
        
    def setImport(self, group):
        if not self.init(group.getPartCase()):
            if not negMsgHandler.setReductionFactor:
                raise  TimestepFoundError()
            else:
                self.setReductionFactor(negMsgHandler.reductionFactor)

    def delete(self, isInitialized, negMsgHandler=None):
        if isInitialized:
            for key in list(globalKeyHandler().getAllElements().keys()):
                obj = globalKeyHandler().getObject(key)
                if obj:
                    # remove composed parts containing this part as well
                    if (obj.typeNr == TYPE_3D_COMPOSED_PART) and (self.key in obj.params.subKeys):
                        obj.delete(isInitialized, negMsgHandler)
            if hasattr(self, "importModule") and self.importModule:
                self.importModule.delete()
        coKeydObject.delete(self, isInitialized, negMsgHandler)


class co3DPartMgrParams(object):
    """ parameters of class co3DPartMgr """
    def __init__(self):
        self.name = 'test.3DPart'
        self.partcase = None
