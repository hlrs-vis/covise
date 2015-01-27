
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH

from KeydObject import coKeydObject, TYPE_CAD_PART
from CADImport import ImportCADModule
import copy

class coCADPartMgr(coKeydObject):
    """ class handling CAD Parts """
    def __init__(self):
        coKeydObject.__init__(self, TYPE_CAD_PART, 'CAD_PART')
        self.params = coCADPartMgrParams()
        self.importModule = None

    def init( self, helper ):
        self.importModule = ImportCADModule( helper.filename, helper.index )

    def addObject( self, visItem):
        visItem.setImport( self.importModule )
        coKeydObject.addObject( self, visItem )

    def setParams( self, params, negMsgHandler):
        coKeydObject.setParams( self, params)
        if self.params.featureAngleDefault:
            featureAngle = -1
        else:
            featureAngle = self.params.featureAngle
        if self.params.max_Dev_mm_Default:
            max_Dev_mm = -1
        else:
            max_Dev_mm = self.params.max_Dev_mm
        if self.params.max_Size_mm_Default:
            max_Size_mm = -1
        else:
            max_Size_mm = self.params.max_Size_mm
        self.importModule.setTesselationParams( featureAngle, max_Dev_mm, max_Size_mm )

    def __getstate__(self):
        """ __getstate__ returns a cleaned dictionary
            only called while class is pickled
        """
        mycontent = copy.copy(self.__dict__)
        del mycontent['importModule']
        return mycontent

    def recreate(self, negMsgHandler, parentKey, offset):
        class Helper:
            pass
        helper = Helper()
        helper.filename = self.params.filename
        helper.index = self.params.index

        self.init( helper )
        for visItem in self.objects:
            visItem.setImport( self.importModule )
        coKeydObject.recreate(self, negMsgHandler, parentKey, offset)

class coCADPartMgrParams(object):
    """ parameters of class coCADPartMgr """
    def __init__(self):
        self.name      = 'coCADPartMgrParams'
        self.filename  = None
        self.featureAngleDefault = True
        self.featureAngle = 30
        self.max_Dev_mm_Default = True
        self.max_Dev_mm   = 20
        self.max_Size_mm_Default = True
        self.max_Size_mm  = 400

        # body index in product structure
        self.index = 2
