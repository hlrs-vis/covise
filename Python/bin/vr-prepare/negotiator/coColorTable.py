
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH


from KeydObject import globalKeyHandler, RUN_ALL, globalColorMgrKey, coKeydObject, TYPE_COLOR_TABLE
from Utils import mergeGivenParams

from printing import InfoPrintCapable
_infoer = InfoPrintCapable()
_infoer.doPrint = False

import covise

class coColorTable(coKeydObject):
    def __init__(self):
        coKeydObject.__init__(self, TYPE_COLOR_TABLE, 'ColorTable')
        self.params = coColorTableParams()
        self.__initColorMapsFromConfig()
        
    def __initColorMapsFromConfig(self):
        # the Colormaps section has to be like
        #  Colormaps
        #  {
        #     <name of colormap>  <anyText>
        #  }
        #
        if not hasattr( covise, "getCoConfigSubEntries" ):
            self.params.colorMapList = coColorTableParams.defaultColorMapList ######["Standard", "Star", "ITMS", "Rainbow"]
            return

        colorMaps = covise.getCoConfigSubEntries("Colormaps")
        self.params.colorMapList = coColorTableParams.defaultColorMapList ######["Standard", "Star", "ITMS", "Rainbow"]
        self.params.colorMapList.extend(colorMaps)

    def recreate(self, negMsgHandler, parentKey, offset):
        coColorTableParams.mergeDefaultParams(self.params) # explicitly call mergeDefaultParams of this class
        if (self.params.dependantKeys == None): # There are projects with dependantKeys==None instead of an empty list.
            self.params.dependantKeys = []      # We fix that here.
        savedList = self.params.colorMapList
        if len(savedList)==0 :
            savedList = coColorTableParams.defaultColorMapList ######["Standard", "Star", "ITMS", "Rainbow"]
        self.__initColorMapsFromConfig()
        if savedList and (self.params.colorMapIdx-1 >= 0) and (self.params.colorMapIdx-1 <= len(savedList)-1):
            if savedList[self.params.colorMapIdx-1] in self.params.colorMapList:
                self.params.colorMapIdx = self.params.colorMapList.index( savedList[self.params.colorMapIdx-1] )+1
        if offset>0 :
            for i in range( len(self.params.dependantKeys)):
                self.params.dependantKeys[i] += offset
            #check if species already exists
            self.params.name = globalKeyHandler().getObject(globalColorMgrKey).getRightName(self.params.species)
        coKeydObject.recreate(self, negMsgHandler, parentKey, offset)

    def run(self, runmode, negMsgHandler=None):
        if runmode==RUN_ALL and not negMsgHandler==None:
            for key in self.params.dependantKeys:
                if globalKeyHandler().hasKey(key):
                    negMsgHandler.run( key )
        coKeydObject.run( self, runmode, negMsgHandler )

class coColorTableParams(object):
    FREE   = 1
    LOCAL  = 2
    GLOBAL = 3
    defaultColorMapList = ["Default"]
    
    def __init__(self):
        self.name  = 'ColorTableParams'
        #self.mode  = coColorTableParams.GLOBAL
        self.min   = 0.
        self.max   = 1.
        self.steps = 16
        self.cont  = True
        self.colorMapIdx  = 2

        # 1st entry in colormap choice is the "Editable" (here: "Default"), which isn't in config-colormaps.xml
        self.colorMapList = coColorTableParams.defaultColorMapList ######["Standard", "Star", "ITMS", "Rainbow"]

        self.species = 'variable'
        # last used visItem properties
        self.baseObjectName = 'default'
        self.baseMin = 0.
        self.baseMax = 1.
        #self.globalMin = -1.
        #self.globalMax = 1.
        # objects that use this colortable
        #self.dependantKeys = []
        coColorTableParams.mergeDefaultParams(self) # explicitly call mergeDefaultParams of this class

    def mergeDefaultParams(self):
        defaultParams = {
            'dependantKeys' : [],
            'mode' : 3,
            'globalMin' : -1.,
            'globalMax' : 1.
        }
        mergeGivenParams(self, defaultParams)
