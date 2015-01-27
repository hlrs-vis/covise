
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH

from KeydObject import coKeydObject, globalKeyHandler, RUN_GEO, RUN_ALL, RUN_OCT, TYPE_2D_CUTGEOMETRY_PART, TYPE_2D_COMPOSED_PART
from ImportGroupCutGeometry import ImportGroupCutGeometry
from Utils import AxisAlignedRectangleIn3d, convertAlignedRectangleToCutRectangle

from printing import InfoPrintCapable
_infoer = InfoPrintCapable()
_infoer.doPrint = False # True

class co2DCutGeometryPartMgr(coKeydObject):
    """ class handling composed 2D Parts """
    def __init__(self):
        _infoer.function = str(self.__init__)
        coKeydObject.__init__(self, TYPE_2D_CUTGEOMETRY_PART, '2D')
        self.params = co2DCutGeometryPartMgrParams()
        self.importModule = None
        
    def setImport(self, importModule):
        if not hasattr(self, "importModule") or (importModule != self.importModule):
            self.importModule = importModule
            self.cut = ImportGroupCutGeometry(self.importModule) 
            for visItem in self.objects:
                visItem.setImport( self.cut )
            self.__update()

    def addObject( self, visItem):
        _infoer.function = str(self.setParams)
        _infoer.write("addObject")
        visItem.setImport( self.cut )
        coKeydObject.addObject( self, visItem )
        
    def recreate(self, negMsgHandler, parentKey, offset):
        coKeydObject.recreate(self, negMsgHandler, parentKey, offset)

    def setParams( self, params, negMsgHandler=None, sendToCover=True):
        _infoer.function = str(self.setParams)
        _infoer.write("setParams")
        coKeydObject.setParams(self, params)
        self.__update()

    def __update(self):
        rect = convertAlignedRectangleToCutRectangle(self.params.alignedRectangle)
        self.cut.setNormal( *rect.getNormal() )
        self.cut.setDistance( rect.getDistance() )

    def run(self, runmode, negMsgHandler=None):
        if runmode==RUN_GEO:
            self.cut.executeGeo()
        elif runmode==RUN_OCT:
            self.cut.executeOct()
        else:
            coKeydObject.run(self, runmode, negMsgHandler)

    def delete(self, isInitialized, negMsgHandler=None):
        if isInitialized:
            for key in list(globalKeyHandler().getAllElements().keys()):
                obj = globalKeyHandler().getObject(key)
                if obj:
                    # remove composed parts containing this part as well
                    if (obj.typeNr == TYPE_2D_COMPOSED_PART) and (self.key in obj.params.subKeys):
                        obj.delete(isInitialized, negMsgHandler)
            if hasattr(self, "cut") and self.cut:
                self.cut.delete()
        coKeydObject.delete(self, isInitialized, negMsgHandler)

class co2DCutGeometryPartMgrParams(object):
    """ parameters of class co2DPartMgr """
    def __init__(self):
        self.name = 'CutGeometry.2DPart'
        self.partcase = None
        self.alignedRectangle = AxisAlignedRectangleIn3d()
