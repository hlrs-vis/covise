
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH

from KeydObject import coKeydObject, RUN_ALL, RUN_OCT, RUN_GEO, TYPE_2D_GROUP, TYPE_3D_GROUP



from ImportGroupManager import ImportGroup2DModule, ImportGroup3DModule
from co2DPartMgr import co2DPartMgr
from co3DPartMgr import co3DPartMgr
from VRPCoviseNetAccess import theNet
from Part2DStaticColorVis import  Part2DStaticColorVis
from Part3DBoundingBoxVis import  Part3DBoundingBoxVis

class coGroupMgr(coKeydObject):
    """ class handling import groups """
    def __init__(self, typeNr, dimension):
        coKeydObject.__init__(self, typeNr, 'group')
        self.params = coGroupMgrParams()
        self.name = self.params.name
        self.dimension = dimension

    def setParams( self, params, negMsgHandler=None):
        coKeydObject.setParams( self, params)
        self.name = self.params.name

class coGroupMgrParams(object):
    def __init__(self):
        self.name = 'test.Group'

class co2DGroupMgr(coGroupMgr):
    """ group of 2D elements """
    def __init__(self):
        coGroupMgr.__init__(self, TYPE_2D_GROUP, 2)
        self.importGroup = ImportGroup2DModule()#ImportSimpleGroup2DModule()

    def addObject( self, obj ):
        #assert isinstance( obj, co2DPartMgr), 'co2DGroupMgr::addObject called with wrong obj type'
        if isinstance( obj, Part2DStaticColorVis):
            obj.setImport( self.importGroup)
        #elif isinstance( obj,co2DPartMgr):
            #self.importGroup.addImport( obj.importModule )
        coGroupMgr.addObject( self, obj)

    """ comment code which is not used

    def run(self, runmode):
        print "co2DGroupMgr::run"
        coGroupMgr.run(self, runmode)

        if len(self.objects)==0:
            print "co2DGroupMgr::noChild"
            return
        if runmode==RUN_GEO or runmode==RUN_ALL:
            print "co2DGroupMgr::runGeo"
            self.importGroup.executeGeo()

        #if runmode==RUN_OCT or runmode==RUN_ALL:
        #    print "co2DGroupMgr::runOct"
        #    self.importGroup.executeOct()
    """

class co3DGroupMgr(coGroupMgr):
    """ group of 2D elements """
    def __init__(self):
        coGroupMgr.__init__(self, TYPE_3D_GROUP, 3)
        self.importGroup = ImportGroup3DModule()

    def addObject( self, obj ):
        #assert isinstance( obj, co2DPartMgr), 'co2DGroupMgr::addObject called with wrong obj type'
        if isinstance( obj, Part3DBoundingBoxVis):
            obj.setImport( self.importGroup)
        #elif isinstance( obj,co3DPartMgr):
            #self.importGroup.addImport( obj.importModule )
        coGroupMgr.addObject( self, obj)

    """ comment code which is not used
    def run(self, runmode):
        print "co3DGroupMgr::run"
        coGroupMgr.run(self, runmode)

        if len(self.objects)==0:
            print "co3DGroupMgr::noChild"
            return
        if runmode==RUN_GEO or runmode==RUN_ALL:
            print "co3DGroupMgr::runGeo"
            self.importGroup.executeGeo()
        if runmode==RUN_OCT or runmode==RUN_ALL:
            print "co3DGroupMgr::runOct"
            self.importGroup.executeOct()
    """
