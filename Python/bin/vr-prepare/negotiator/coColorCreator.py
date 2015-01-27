
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH

import copy
from VRPCoviseNetAccess import (
    connect,
    disconnect,
    ConnectionPoint,
    saveExecute,
    theNet)

from coPyModules import Colors
from KeydObject import coKeydObject, TYPE_COLOR_CREATOR, RUN_ALL, globalKeyHandler
from coColorTable import coColorTable, coColorTableParams

from types import *

from printing import InfoPrintCapable
import os

_infoer = InfoPrintCapable()
_infoer.doPrint = False

class coColorCreator(coKeydObject):
    """ class creating colors and colormap"""
    def __init__(self):
        coKeydObject.__init__(self, TYPE_COLOR_CREATOR, self.__class__.__name__)
        self.params = coColorCreatorParams()
        self.__colors = None
        self.__lastDataConnection = None

    def __init(self):
        """ start COVISE Module """
        if self.__colors==None:
            self.__colors = Colors()
            theNet().add(self.__colors)
            self.__colors.set_autoScales( 'FALSE' )
            self.__colorsIn = ConnectionPoint( self.__colors, 'DataIn0' )
            self.__colorsOut = ConnectionPoint( self.__colors, 'DataOut0' )

            self.__previousColorTableParams = None      # used to indicate a param change

    def __update(self):
        self.__init()
        _infoer.function = str(self.__update)
        _infoer.write("update %s" % self.params.colorTableKey )

        changed = False

        #update input
        if not self.__lastDataConnection==None :
            disconnect( self.__lastDataConnection,  self.__colorsIn )
            changed = True

        if not self.params.variable==None:
            dataInConnect = self.params.importModule.dataConnectionPoint(self.params.variable)
            if dataInConnect:
                connect( self.params.importModule.dataConnectionPoint(self.params.variable),  self.__colorsIn )
                self.__lastDataConnection=self.params.importModule.dataConnectionPoint(self.params.variable)
                changed = True

        #update parameters
        if not self.params.variable==None:
            self.__colors.set_annotation( self.params.variable )
        if self.params.colorTableKey>=0:
            colorTable = globalKeyHandler().getObject(self.params.colorTableKey)

            if self.__previousColorTableParams == colorTable.params:
                return changed or False        # no params were actually updated, but a connection might have changed

            self.__colors.set_numSteps( colorTable.params.steps )
            if colorTable.params.mode==coColorTableParams.LOCAL:
                self.__colors.set_MinMax( colorTable.params.baseMin, colorTable.params.baseMax)
            elif colorTable.params.mode==coColorTableParams.FREE:
                self.__colors.set_MinMax( colorTable.params.min, colorTable.params.max)
            else :
                self.__colors.set_MinMax( colorTable.params.globalMin, colorTable.params.globalMax)

            # Colormap cannot (yet) be set by simply passing the index of the selected choice
            # read the Colormap parameter from the module and replace the corresponding choice selection (1st number in string)
            __colorMap = self.__colors.get_Colormap()
            __colorMapSplit = __colorMap.split(' ')
            __colorMapSplit[0] = str(colorTable.params.colorMapIdx)
            __colorMap = str.join(" ", __colorMapSplit)
            self.__colors.set_Colormap(__colorMap)

            self.__colors.set_annotation( colorTable.params.name )

            self.__previousColorTableParams = colorTable.params
        return True

    def colorContConnectionPoint(self):
        self.__init()
        return self.__colorsOut

    def colorMapConnectionPoint(self):
        self.__init()
        return ConnectionPoint( self.__colors, 'ColormapOut0' )

    def colorTextureConnectionPoint(self):
        self.__init()
        return ConnectionPoint( self.__colors, 'TextureOut0' )

    def __getstate__(self):
        mycontent = copy.copy(self.__dict__)
        del mycontent['_coColorCreator__colors']
        return mycontent

    def recreate(self, negMsgHandler, parentKey, offset):
        self.__colors = None
        self.__lastDataConnection = None
        self.__init()
        coKeydObject.recreate(self, negMsgHandler, parentKey, offset)
        self.params.colorTableKey += offset
        
    def run(self, runmode, negMsgHandler=None):
        if os.getenv('VR_PREPARE_DEBUG_RUN'):
            print("coColorCreator.run")
        if runmode==RUN_ALL:
            _infoer.function = str(self.run)
            _infoer.write("go")
            if self.__update():
                if not self.params.variable==None:
                    if not self.params.importModule.executeData(self.params.variable):
                        saveExecute( self.__colors)
                        return True
                    else:
                        return True
                else:
                    saveExecute( self.__colors)
                    return True
            return False

    def delete(self, isInitialized, negMsgHandler=None):
        if isInitialized:
            if hasattr(self, '_coColorCreator__colors') and self.__colors: theNet().remove(self.__colors)
        coKeydObject.delete(self, isInitialized, negMsgHandler)

class coColorCreatorParams(object):
    def __init__(self):
        self.name = 'ColorCreatorParams'
        self.importModule = None
        self.variable = None
        self.colorTableKey = -2
