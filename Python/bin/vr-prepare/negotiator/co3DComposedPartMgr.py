
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH

from KeydObject import coKeydObject, globalKeyHandler, RUN_GEO, RUN_ALL, RUN_OCT, TYPE_3D_COMPOSED_PART
from ImportGroupManager import COMPOSED_VELOCITY, ImportGroup3DModule
import covise

from printing import InfoPrintCapable
_infoer = InfoPrintCapable()
_infoer.doPrint = False # True

class co3DComposedPartMgr(coKeydObject):
    """ class handling composed 3D Parts """
    def __init__(self):
        _infoer.function = str(self.__init__)
        coKeydObject.__init__(self, TYPE_3D_COMPOSED_PART, '3D')
        self.params = co3DComposedPartMgrParams()
        self.importModule = None

    def __addSubPart( self, key, velValue=None ):
        _infoer.function = str(self.setParams)
        _infoer.write("__addSubPart %s" % key)
        if self.importModule==None:
            self.importModule = ImportGroup3DModule()
        self.importModule.addImport( globalKeyHandler().getObject(key).importModule, velValue )

    def __initSubParts( self ):
        cnt=0
        for key in self.params.subKeys:
            self.__addSubPart( key, self.params.velDefinitions[cnt] )
            cnt = cnt + 1

    def addObject( self, visItem):
        _infoer.function = str(self.setParams)
        _infoer.write("addObject")
        visItem.setImport( self.importModule )
        coKeydObject.addObject( self, visItem )

    def recreate(self, negMsgHandler, parentKey, offset):
        self.importModule = None
        self.__initSubParts()
        coKeydObject.recreate(self, negMsgHandler, parentKey, offset)
        for visItem in self.objects:
            visItem.setImport( self.importModule )

    def setParams( self, params, negMsgHandler=None, sendToCover=True):
        _infoer.function = str(self.setParams)
        _infoer.write("setParams")
        if len(self.params.subKeys)==0:
            coKeydObject.setParams(self, params)
            self.__initSubParts()

    def run(self, runmode, negMsgHandler=None):
        if runmode==RUN_GEO:
            self.importModule.executeGeo()
        elif runmode==RUN_OCT:
            self.importModule.executeOct()
            if covise.coConfigIsOn("vr-prepare.UseSamplingModules", False):
                if not self.importModule.getIsTransient():
                    self.importModule.executeSampleData(COMPOSED_VELOCITY)
        else:
           coKeydObject.run(self, runmode, negMsgHandler)

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

class co3DComposedPartMgrParams(object):
    """ parameters of class co3DPartMgr """
    def __init__(self):
        self.name = 'Composed.3DPart'
        self.subKeys = []
        # velocity variable in each sub part
        self.velDefinitions = []
