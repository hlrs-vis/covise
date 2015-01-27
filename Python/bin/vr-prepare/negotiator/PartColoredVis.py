# Part of the vr-prepare program
# Copyright (c) 2007 Visenso GmbH

# parent class for visItems with Color
#
# every visItem, which includes coloring should inherit this class
# this class implemts all necessary functions for coloring 
# creates color creator...

from VRPCoviseNetAccess import (
    connect,
    disconnect,
    ConnectionPoint,
    theNet)

import KeydObject
import Neg2Gui
from Utils import  ParamsDiff
from KeydObject import globalKeyHandler, RUN_ALL, TYPE_COLOR_TABLE, TYPE_COLOR_CREATOR
from coColorCreator import coColorCreatorParams
from coColorTable import coColorTableParams

from printing import InfoPrintCapable
import os

_infoer = InfoPrintCapable()
_infoer.doPrint = False # True#


class PartColoredVis(object):
    '''helper class for coloring'''
    def __init__(self, colorMap=True):
        '''typeNr of the visItem, name of the visitem and module of the visitem'''        
        self.__colorMap = colorMap
        self.__initBase()

    def __initBase(self):
        """ __initBase is called from the constructor and after the class was unpickled """
        #self.__firstTime = True
        #self.__needExecute = True
        self._visualizerColorCP = None # deleted in PartModuleVis before the class gets pickled

    def _init(self, negMsgHandler, module, colorMapPortName = 'ColormapIn0'):
        '''_init is called from _update
           connect color map
        '''
        # check if ColorCreator already exists
        if len(self.objects)==0:
            self._initColor(negMsgHandler)
        if len(self.objects)>0 and self.__colorMap:
            self._visualizerColorCP = ConnectionPoint(module, colorMapPortName)
            connect( (self.objects[0]).colorMapConnectionPoint(), self._visualizerColorCP )

    def setColorMap( self, colorMap ):
        # for compatibility with version 1
        #if colorMap != self.__colorMap:
        #    self.__needExecute = True
        self.__colorMap = colorMap 

    def currentVariable(self):
        """ return the currently variable that is used to map colors on the tracer 
            return None if nothing was set
        """
        if hasattr(self.params, 'secondVariable') and self.params.secondVariable:
            return self.params.secondVariable
        else:
            if self.params.variable=='unset' or self.params.variable=='Select a variable':
                return None
            else :
                return self.params.variable

    def _initColor(self, negMsgHandler):
        ''' + start a ColorCreator if it does not exist
            + set current color table according to the currently used variable
            + if no color table for that variable exists : create one and set min/max accoring current variable
            + set colorTable param in the ColorCreator
            + set dependantKey in the color table
        '''
        #assert negMsgHandler
        if self.importModule==None or self.currentVariable()==None:
            return

        if not self.currentVariable() in self.params.colorTableKey:
            colorTableKey = (globalKeyHandler().getObject(KeydObject.globalColorMgrKey)).getKeyOfTable( self.currentVariable() )
            if not colorTableKey==None:
                _infoer.function = str(self._initColor)
                _infoer.write("using existing table for %s" % self.currentVariable())
                self.params.colorTableKey[self.currentVariable()] = colorTableKey
                colorTable = globalKeyHandler().getObject(colorTableKey)
            elif negMsgHandler :
                _infoer.function = str(self._initColor)
                _infoer.write("creating color table for %s" % self.currentVariable())
                colorTable = negMsgHandler.internalRequestObject( TYPE_COLOR_TABLE )
                if colorTable==None:
                    # request was called during recreation
                    return
                self.params.colorTableKey[self.currentVariable()] = colorTable.key
                tP = coColorTableParams()
                dataRange = self.importModule.getDataMinMax(self.currentVariable())
                if dataRange:
                    tP.globalMin = float(dataRange[0])
                    tP.globalMax = float(dataRange[1])
                    tP.min = float(dataRange[0])
                    tP.max = float(dataRange[1])
                    tP.species = self.currentVariable()
                    tP.name = (globalKeyHandler().getObject(KeydObject.globalColorMgrKey)).getRightName(tP.species)
                    if not self.__colorMap:
                        tP.baseObjectName = "Part"
                    colorTable.setParams(tP)
                    negMsgHandler.sendParams(colorTable.key, tP)
            # send the now known colorTableKey back to the GUI
            if negMsgHandler:
                negMsgHandler.sendParams( self.key, self.params )

        # create or update colorCreator
        _infoer.function = str(self._initColor)
        _infoer.write("updating color creator for %s" % self.currentVariable())
        cP = coColorCreatorParams()
        cP.colorTableKey = self.params.colorTableKey[self.currentVariable()]
        if not self.__colorMap:
            cP.variable = self.currentVariable()
            cP.importModule = self.importModule
        if len(self.objects)>0:
            (self.objects[0]).setParams(cP)
            #(self.objects[0]).run(RUN_ALL) # This is nescessary because CuttingSurfaceComp can't be executed without a colormap if sampled data is present. Because of PartColoredVis called after PartModuleVis, the colors module is not executed before.
            #print("PartColoredVis._initColor(): omitting explicit (self.objects[0]).run(RUN_ALL)")
        else:
            if negMsgHandler:
                _infoer.function = str(self._initColor)
                _infoer.write("creating color creator for %s" % self.currentVariable())
                colorCreator=negMsgHandler.internalRequestObject( TYPE_COLOR_CREATOR, self.key)
                colorCreator.setParams(cP)
                colorCreator.run(RUN_ALL)

        # Make sure the objects key is contained in the colormaps dependantKeys
        colorTable = globalKeyHandler().getObject(self.params.colorTableKey[self.currentVariable()])
        if self.key not in colorTable.params.dependantKeys:
            colorTable.params.dependantKeys.append(self.key)
            if negMsgHandler:
                negMsgHandler.sendParams( colorTable.key, colorTable.params )


    def _update(self, negMsgHandler):
        '''_update is called from the run method to update the module parameter before execution'''
        self._initColor(negMsgHandler)

    def recreate(self, negMsgHandler, parentKey, offset ):
        self.__initBase()
        if offset>0:
            for obj in self.params.colorTableKey:
                self.params.colorTableKey[obj]+=offset

    def run(self, runmode, negMsgHandler, module, afterRecreation=False):
        if os.getenv('VR_PREPARE_DEBUG_RUN'):
            print("PartColoredVis.run")
        '''create a new visulisation'''
        #assert negMsgHandler

        if runmode==RUN_ALL:
            if not hasattr(self, 'importModule'):
                return False

            _infoer.function = str(self.run)
            _infoer.write("go colored vis for %s" % self.params.variable)


            PartColoredVis._update(self, negMsgHandler)

            #update colortable at the first time after recreation
            #if afterRecreation and len(self.objects)>0:
            #    colorCreator = self.objects[0]
            #    return colorCreator.run(RUN_ALL)

            #if not self.__needExecute:
            #    # TODO: self.__needExecute zu oft auf False -> geaenderte color map wird nicht uebernommen
            #    #return False
            #self.__needExecute = False

            """
            # to be sure that the import module loaded all data
            self.importModule.executeOct()

            if not self.importModule.executeData(self.currentVariable()):
                if len(self.objects)>0:
                    colorCreator = self.objects[0]
                    # the colorCreator contains a colors module which is connected to the import of the tracer
                    # implicit execution of the tracer module
                    colorCreator.run(RUN_ALL)
                    return True
                #else:
                #    module.execute()
            """

            if len(self.objects)>0:
                colorCreator = self.objects[0]
                if self._visualizerColorCP:
                    # Since we dont know if the visualizers module is actually ready to execute,
                    # we dont want it to be executed along with the colors module.
                    # Therefore the disconnect.
                    disconnect(colorCreator.colorMapConnectionPoint(), self._visualizerColorCP)
                colorCreator.run(RUN_ALL)
                if self._visualizerColorCP:
                    connect(colorCreator.colorMapConnectionPoint(), self._visualizerColorCP)

            return False # always return False because the visualizers module is disconnected and wont be executed

    def setParams( self, params, negMsgHandler=None, sendToCover=True, realChange=None):
        ''' set parameters from outside'''

        if realChange==None:
            realChange = ParamsDiff( self.params, params )

        #self.__needExecute = False

        if not hasattr(params, 'variable') or params.variable == 'Select a variable':
            return

        if 'variable' in realChange:
            self._initColor( negMsgHandler )
            #self.__needExecute = True

        if 'secondVariable' in realChange:
            self._initColor( negMsgHandler )  
            # update base name of color table
            colorTable = globalKeyHandler().getObject(self.params.colorTableKey[self.currentVariable()])
            colorTable.params.baseObjectName = self.params.name
            if negMsgHandler:
                negMsgHandler.sendParams( colorTable.key, colorTable.params, -1 )
            #self.__needExecute = True

class PartColoredVisParams(object):
    '''Params for Colored helper of VisItem'''
    def __init__(self):
        self.colorTableKey = {}
        self.variable = 'unset'
        self.secondVariable = None
