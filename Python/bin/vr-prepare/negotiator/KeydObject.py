
# Part of the vr-prepare program for dc

# Copyright (c) 2006-2007 Visenso GmbH

import copy

import auxils

from ImportManager import ImportModule
from ImportGroupManager import ImportGroupModule
from ImportGroupFilter import ImportGroupFilter


# object types
TYPE_SESSION    = -1
TYPE_PROJECT    = 0
TYPE_CASE       = 1
TYPE_2D_GROUP   = 2
TYPE_3D_GROUP   = 3
TYPE_2D_PART    = 4
TYPE_3D_PART    = 5
TYPE_3D_COMPOSED_PART    = 29
TYPE_COLOR_TABLE = 12
TYPE_COLOR_CREATOR = 13
TYPE_COLOR_MGR = 14
TYPE_PRESENTATION = 17
TYPE_PRESENTATION_STEP = 18
TYPE_VIEWPOINT_MGR = 19
TYPE_VIEWPOINT = 20
TYPE_CAD_PRODUCT = 27
TYPE_CAD_PART = 28
TYPE_JOURNAL = 31
TYPE_JOURNAL_STEP = 32
TYPE_2D_COMPOSED_PART = 33
TYPE_2D_CUTGEOMETRY_PART = 39
TYPE_SCENEGRAPH_MGR = 40
TYPE_SCENEGRAPH_ITEM = 41
TYPE_TRACKING_MGR = 43
TYPE_DNA_MGR = 44
TYPE_DNA_ITEM = 45
TYPE_GENERIC_OBJECT_MGR = 46
TYPE_GENERIC_OBJECT = 47

#visualization types
VIS_2D_STATIC_COLOR = 6
VIS_2D_SCALAR_COLOR = 7
VIS_2D_MATERIAL     = 8
VIS_2D_RAW          = 9
VIS_3D_BOUNDING_BOX = 10
VIS_STREAMLINE      = 11
VIS_MOVING_POINTS   = 15
VIS_PATHLINES       = 16
VIS_VRML = 21
VIS_COVISE = 22
VIS_PLANE = 23
VIS_VECTOR = 24
VIS_ISOPLANE = 25
VIS_DOCUMENT = 26
VIS_POINTPROBING = 30
VIS_ISOCUTTER = 34
VIS_CLIPINTERVAL = 35
VIS_VECTORFIELD = 36
VIS_STREAMLINE_2D = 37
VIS_DOMAINLINES = 38
VIS_MAGMATRACE = 42
VIS_DOMAINSURFACE = 48
VIS_SCENE_OBJECT = 49

nameOfCOType = {}
nameOfCOType[TYPE_PROJECT] = 'TYPE_PROJECT'
nameOfCOType[TYPE_CASE] = 'TYPE_CASE'
nameOfCOType[TYPE_2D_GROUP] = 'TYPE_2D_GROUP'
nameOfCOType[TYPE_3D_GROUP] = 'TYPE_3D_GROUP'
nameOfCOType[TYPE_2D_PART] = 'TYPE_2D_PART'
nameOfCOType[TYPE_3D_PART] = 'TYPE_3D_PART'
nameOfCOType[TYPE_COLOR_TABLE] = 'TYPE_COLOR_TABLE'
nameOfCOType[TYPE_COLOR_CREATOR] = 'TYPE_COLOR_CREATOR'
nameOfCOType[TYPE_COLOR_MGR] = 'TYPE_COLOR_MGR'
nameOfCOType[VIS_2D_STATIC_COLOR] = 'VIS_2D_STATIC_COLOR'
nameOfCOType[VIS_2D_SCALAR_COLOR] = 'VIS_2D_SCALAR_COLOR'
nameOfCOType[VIS_2D_MATERIAL] = 'VIS_2D_MATERIAL'
nameOfCOType[VIS_2D_RAW] = 'VIS_2D_RAW'
nameOfCOType[VIS_3D_BOUNDING_BOX] = 'VIS_3D_BOUNDING_BOX'
nameOfCOType[VIS_STREAMLINE] = 'VIS_STREAMLINE'
nameOfCOType[VIS_MOVING_POINTS] = 'VIS_MOVING_POINTS'
nameOfCOType[VIS_PATHLINES] = 'VIS_PATHLINES'
nameOfCOType[TYPE_PRESENTATION] = 'TYPE_PRESENTATION'
nameOfCOType[TYPE_PRESENTATION_STEP] = 'TYPE_PRESENTATION_STEP'
nameOfCOType[TYPE_VIEWPOINT_MGR] = 'TYPE_VIEWPOINT_MGR'
nameOfCOType[TYPE_VIEWPOINT] = 'TYPE_VIEWPOINT'
nameOfCOType[VIS_VRML] = 'VIS_VRML'
nameOfCOType[VIS_COVISE] = 'VIS_COVISE'
nameOfCOType[VIS_PLANE] = 'VIS_PLANE'
nameOfCOType[VIS_VECTOR] = 'VIS_VECTOR'
nameOfCOType[VIS_ISOPLANE] = 'VIS_ISOPLANE'
nameOfCOType[VIS_ISOCUTTER] = 'VIS_ISOCUTTER'
nameOfCOType[VIS_CLIPINTERVAL] = 'VIS_CLIPINTERVAL'
nameOfCOType[VIS_VECTORFIELD] = 'VIS_VECTORFIELD'
nameOfCOType[VIS_DOCUMENT] = 'VIS_DOCUMENT'
nameOfCOType[VIS_SCENE_OBJECT] = 'VIS_SCENE_OBJECT'
nameOfCOType[TYPE_CAD_PRODUCT] = 'TYPE_CAD_PRODUCT'
nameOfCOType[TYPE_CAD_PART] = 'TYPE_CAD_PART'
nameOfCOType[TYPE_3D_COMPOSED_PART] = 'TYPE_3D_COMPOSED_PART'
nameOfCOType[VIS_POINTPROBING] = 'VIS_POINTPROBING'
nameOfCOType[TYPE_JOURNAL] = 'TYPE_JOURNAL'
nameOfCOType[TYPE_JOURNAL_STEP] = 'TYPE_JOURNAL_STEP'
nameOfCOType[TYPE_2D_COMPOSED_PART] = 'TYPE_2D_COMPOSED_PART'
nameOfCOType[VIS_STREAMLINE_2D] = 'VIS_STREAMLINE_2D'
nameOfCOType[VIS_DOMAINLINES] = 'VIS_DOMAINLINES'
nameOfCOType[VIS_DOMAINSURFACE] = 'VIS_DOMAINSURFACE'
nameOfCOType[TYPE_2D_CUTGEOMETRY_PART] = 'TYPE_CUTGEOMETRY_PART'
nameOfCOType[TYPE_SCENEGRAPH_MGR] = 'TYPE_SCENEGRAPH_MGR'
nameOfCOType[TYPE_SCENEGRAPH_ITEM] = 'TYPE_SCENEGRAPH_ITEM'
nameOfCOType[VIS_MAGMATRACE] = 'VIS_MAGMATRACE'
nameOfCOType[TYPE_TRACKING_MGR] = 'TYPE_TRACKING_MGR'
nameOfCOType[TYPE_DNA_MGR] = 'TYPE_DNA_MGR'
nameOfCOType[TYPE_DNA_ITEM] = 'TYPE_DNA_ITEM'
nameOfCOType[TYPE_GENERIC_OBJECT_MGR] = 'TYPE_GENERIC_OBJECT_MGR'
nameOfCOType[TYPE_GENERIC_OBJECT] = 'TYPE_GENERIC_OBJECT'

#running methods
RUN_ALL = 0
RUN_GEO = 1
RUN_OCT = 2

globalProjectKey = 0

# central colorMap Manager
# is always the second node of our session. First node is the project itself
globalColorMgrKey = 1

# central presentation manager
# is always the third node of our session.
globalPresentationMgrKey = 2

# central viewpoint Manager
# is always the fourth node of our session.
globalViewpointMgrKey = 3

_globalKeyHandler = None
def globalKeyHandler(newHandler=None, forceOverwrite=False):

    """Assert instance and access to the gui-message-handler (in auxils)."""

    global _globalKeyHandler
    if (None == _globalKeyHandler) or forceOverwrite:
        if (None == newHandler):
            _globalKeyHandler = auxils.KeyHandler()
        else:
            _globalKeyHandler = newHandler

        # The following keys are not known (Since the managers were not present in older projects, the respective key is already used).
        # The keys will be changed in init and recreate of *Mgr

        # central Journal Manager
        _globalKeyHandler.globalJournalMgrKey = 4

        # central SceneGraph Manager
        _globalKeyHandler.globalSceneGraphMgrKey = 5

        # central Tracking Manager
        _globalKeyHandler.globalTrackingMgrKey = 6

        # central DNA Manager
        _globalKeyHandler.globalDNAMgrKey = 7

        # central Generic Object Manager
        _globalKeyHandler.globalGenericObjectMgrKey = 8

    return _globalKeyHandler



# root node
NO_PARENT = -1

class coKeydObject(object):
    """Base object class in vr-prepare.

    Stores parameters and name of the object. Stores children of the object.

    """

    def __init__(self, typeNr=TYPE_SESSION, name='object' ):
        """ init """
        self.name = name
        self.typeNr = typeNr
        self.objects = []
        self.key = globalKeyHandler().registerObject(self)
        self.parentKey = NO_PARENT
        # true if class was created by a saved project
        self.fromRecreation = False

        # the following does not work due to import problems
        #
        # automatically construct name of the according parameter class
        self.param_name =  self.__class__.__name__ + "Params"
        # initparam = "self.params = " + self.param_name + "() "
        # exec initparam
        self.params = None

    def addObject( self, new_object):
        self.objects.append( new_object)
        new_object.parentKey = self.key

    def delete(self, isInitialized, negMsgHandler=None):
        # if overriding this function, make sure you call coKeyedObject.delete(self) at the end !!!!!
        # delete children
        while len(self.objects) > 0:
            obj = self.objects[0]
            del self.objects[0] # remove from objects list because the removal in obj.delete() doesn't work if not in globalKeyHandler
            obj.delete(isInitialized, negMsgHandler)
        # remove from parents object-list
        if self.parentKey != NO_PARENT:
            parent = globalKeyHandler().getObject(self.parentKey)
            if (parent != None) and self in parent.objects:
                i = 0
                for obj in parent.objects:
                    if obj == self:
                        del parent.objects[i]
                        break
                    i = i+1
        if isInitialized:
            # perform actions in negotiator with the deleted object (i.e. send message to GUI)
            if negMsgHandler: negMsgHandler.deletingObject(self.key, self.typeNr, -1, self.parentKey)
            # unregister
            globalKeyHandler().unregisterObject(self.key)
        # delete
        del self

    def setParams( self, params, negMsgHandler=None, sendToCover=False):
        #imCall = "from %s import %s" % (self.__class__.__name__,self.param_name)
        #exec imCall
        #assertCall = "assert isinstance(params, %s), 'setParams called with wrong parameter type'" % self.param_name
        #exec  assertCall
        self.params = params

    def getParams( self ):
        return self.params

    def run( self, runmode=RUN_ALL, negMsgHandler=None):
        for obj in self.objects:
            if (runmode==RUN_ALL):
                print("Starting " + obj.name)
            elif (runmode==RUN_OCT):
                print("Starting Octtree of " + obj.name)
            elif (runmode==RUN_GEO):
                print("Starting Geometry of " + obj.name)
            obj.run(runmode, negMsgHandler)

    def recreate(self, negMsgHandler, parentKey, offset=0):
        if offset>0:
            self.key += offset
            self.parentKey += offset
            globalKeyHandler().registerKeydObject( self.key, self)
        negMsgHandler.initObj( self.key, self.typeNr, -1, parentKey)
        negMsgHandler.sendParams( self.key, self.params )
        for obj in list(self.objects): # iterate over copy so loop doesn't get confused when viewpoints are deleted
            obj.fromRecreation = True
            obj.recreate(negMsgHandler, self.key, offset)

    def merge( self, obj ):
        # merging by adding all objects of obj to my object list
        for objs in obj.objects:
            self.addObject( objs ) 

    def reconnect( self ):
        # reconnect to renderer if necessary, only implemented for VisItems
        for obj in self.objects:
            obj.reconnect()

    def __getstate__(self):
        ''' __getstate__ returns a cleaned dictionary
            only called while class is pickled
        '''
        mycontent = copy.copy(self.__dict__)
        for key, value in iter(self.__dict__.items()):
            # do not pickle Import Modules
            if isinstance(value, ImportModule) or isinstance(value, ImportGroupModule) or isinstance(value, ImportGroupFilter):
                del mycontent[key]
        return mycontent

    def __str__(self):
        if len(self.objects)>0:
            status = "Object '%s'(key:%s) contains:" % (self.name, self.key)
            for obj in  self.objects:
                status = status + "\n- %s" % str(obj)
            return status
        else:
            return "Object '%s'(key:%s) " % (self.name, self.key)


# eof
