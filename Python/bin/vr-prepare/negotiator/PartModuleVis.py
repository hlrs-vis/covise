
# Part of the vr-prepare program
# Copyright (c) 2007 Visenso GmbH

# parent class for visItems with Module
#
# every visItem, which includes a module should inherit this class
# this class implemts all necessary functions to execute, update, etc. a module 

import copy
import ImportGroupManager
import covise

from VRPCoviseNetAccess import (
    connect,
    disconnect,
    ConnectionPoint,
    theNet,
    saveExecute)
import traceback
from VisItem import VisItem, VisItemParams
from coPyModules import *
from KeydObject import coKeydObject, globalKeyHandler, RUN_ALL
from Utils import  ParamsDiff, mergeGivenParams
from printing import InfoPrintCapable
import os

_infoer = InfoPrintCapable()
_infoer.doPrint = False # True

from vrpconstants import NO_COLOR, RGB_COLOR, MATERIAL, VARIABLE

class PartModuleVis(VisItem):
    '''VisItem for one module'''
    def __init__(self,modName, typeNr, name='VisItem', geoInputNames=['GridIn0'] , \
        octtreeInputNames=[], geoSampleNames=[], dataSampleNames=[], \
        dataInputNames=[], scalarInputNames=[], twoDInputNames=[], sampleType=1, bbox = True, octtree= True):
        '''name of the module, list of the geoInputName (meshIn) of the module, 
           typeNr of the visItem, name of the visitem and module of the visitem
           octtreeInputNames of the module, geoSampleNames of the module, dataSamplenames of the module
           dataInputNames of the module, scalarInputNames of the module, 2DPartInputNames of the module'''
        
        VisItem.__init__(self, typeNr, name)
        self.params = PartModuleVisParams()
        self._initBase( modName, geoInputNames, octtreeInputNames, geoSampleNames, dataSampleNames, \
                                dataInputNames, scalarInputNames, twoDInputNames, sampleType, bbox, octtree)
        
    def _initBase(self, modName, geoInputNames=['GridIn0'], octtreeInputNames=[], geoSampleNames=[], dataSampleNames=[], \
        dataInputNames=[], scalarInputNames=[], twoDInputNames=[], sampleType=1, bbox = True, octtree= True):
        '''called from the constructor and after the class was unpickled '''
        self._module = None
        self._lastDataConnection = None
        self._lastScalarConnection = None
        self._last2DPartConnection = None
        
        self._moduleName = modName
        self._geoInputNames = geoInputNames
        self._octtreeInputNames = octtreeInputNames
        self._geoSampleNames = geoSampleNames
        self._dataSampleNames = dataSampleNames
        self._sampleType = sampleType
        self._dataInputNames = dataInputNames
        self._scalarInputNames = scalarInputNames
        self._2DPartInputNames = twoDInputNames
        self.__createBBox = bbox
        self.__createOcttree = octtree
        self.__firstTime = True
        self.__connectedToCOVER = False
        
        #register for UI Actions
        self._registered = False
        self._firstGeoForVars = True

        if not covise.coConfigIsOn("vr-prepare.UseSamplingModules", False):
            self._geoSampleNames  = []
            self._dataSampleNames = []

    def _init(self, negMsgHandler, sampleType=None):
        '''called from _update
            if module exists connect output of it to COVER
        '''
        if not sampleType == None:
            self._sampleType = sampleType
        if self._module == None and hasattr(self, 'importModule') and self.__firstTime:
            self.__firstTime = False
            self._module = self._moduleName()
            theNet().add(self._module)
            # need to execute blockcollect for geo
            if type(self.importModule) == ImportGroupManager.ImportGroup3DModule:
                self.importModule.executeGeo()

            for geoInput in self._geoInputNames:
                connect( self.importModule.geoConnectionPoint(), ConnectionPoint(self._module, geoInput) )

            if self.__createOcttree:
                if len(self._octtreeInputNames) > 0: 
                    for octtree in self._octtreeInputNames:
                        connect( self.importModule.octTreeConnectionPoint(), ConnectionPoint(self._module, octtree) )
                    self.importModule.executeOct()

            if (len(self._dataInputNames)>0):
                if hasattr(self.params, 'variable') and self.params.variable and self.params.variable!='Select a variable' and self.params.variable != 'unset':
                    for dataInput in self._dataInputNames:
                        connect( self.importModule.dataConnectionPoint(self.params.variable), ConnectionPoint(self._module, dataInput) )

            if len(self._geoSampleNames) > 0 and not self.importModule.getIsTransient():
                if hasattr(self.params, 'variable') and self.params.variable and self.params.variable!='Select a variable' and self.params.variable != 'unset':
                   assert self.params.variable!='Select a variable'
                   self.importModule.executeSampleData(self.params.variable, None, self._sampleType)
                   for geoSample in self._geoSampleNames:
                       connect( self.importModule.geoSampleConnectionPoint(self.params.variable, self._sampleType), ConnectionPoint(self._module, geoSample) )

            if len(self._dataSampleNames) > 0 and not self.importModule.getIsTransient():
                if hasattr(self.params, 'variable') and self.params.variable and self.params.variable!='Select a variable' and self.params.variable != 'unset':
                   assert self.params.variable!='Select a variable' 
                   for dataSample in self._dataSampleNames:
                       connect( self.importModule.dataSampleConnectionPoint(self.params.variable, self._sampleType), ConnectionPoint(self._module, dataSample) )
            self.reloadBBox()

            # optionally connect to COVER
            if not covise.coConfigIsOn("vr-prepare.InvisibleConnectToRenderer", True):
                # only connect to COVER if this visualizer really is visible
                if self.params.isVisible and not self.__connectedToCOVER:
                    VisItem.connectToCover( self, self )
                    self.__connectedToCOVER = True
            else:
                VisItem.connectToCover( self, self )
                self.__connectedToCOVER = True

        # refresh all connections as they might have changed
        if self._module != None and hasattr(self, 'importModule'):

            for geoInput in self._geoInputNames:
                theNet().disconnectAllFromModulePort(self._module, geoInput)     # remove all connections at the inport
                connect( self.importModule.geoConnectionPoint(), ConnectionPoint(self._module, geoInput) )

            if self.__createOcttree:
                for octtree in self._octtreeInputNames:
                    theNet().disconnectAllFromModulePort(self._module, octtree)     # remove all connections at the inport
                    connect( self.importModule.octTreeConnectionPoint(), ConnectionPoint(self._module, octtree) )

            if (len(self._dataInputNames)>0):
                if hasattr(self.params, 'variable') and self.params.variable and self.params.variable!='Select a variable' and self.params.variable != 'unset':
                    for dataInput in self._dataInputNames:
                        theNet().disconnectAllFromModulePort(self._module, dataInput)
                        connect( self.importModule.dataConnectionPoint(self.params.variable), ConnectionPoint(self._module, dataInput) )

            if hasattr(self.params, 'variable') and self.params.variable and self.params.variable!='Select a variable' and self.params.variable != 'unset':
                if self.params.variable != 'Select a variable' and not self.importModule.getIsTransient():
                    for geoSample in self._geoSampleNames:
                        theNet().disconnectAllFromModulePort(self._module, geoSample)
                        connect( self.importModule.geoSampleConnectionPoint(self.params.variable, self._sampleType), ConnectionPoint(self._module, geoSample) )

            if hasattr(self.params, 'variable') and self.params.variable and self.params.variable!='Select a variable' and self.params.variable != 'unset':
                if self.params.variable != 'Select a variable' and not self.importModule.getIsTransient():
                    for dataSample in self._dataSampleNames:
                        theNet().disconnectAllFromModulePort(self._module, dataSample)
                        connect( self.importModule.dataSampleConnectionPoint(self.params.variable, self._sampleType), ConnectionPoint(self._module, dataSample) )

            self.reloadBBox()

            # connect to COVER if visible and wasnt done initially
            if self.params.isVisible and not self.__connectedToCOVER:
                VisItem.connectToCover( self, self )
                self.__connectedToCOVER = True



    def reloadBBox(self):
        if self.__createBBox and not self.fromRecreation:
            #bb should be defined by the user
            if not hasattr(self.params.boundingBox, 'getXMin'):
                """
                if hasattr( InternalConfig, 'boundingBox' ):
                    self.params.boundingBox = Box( InternalConfig.boundingBox[0],
                                                  InternalConfig.boundingBox[1],
                                                  InternalConfig.boundingBox[2] )
                    self.params.smokeBox = self.params.boundingBox
                else:
                    self.params.boundingBox = self.importModule.getBox()
                """
                self.params.boundingBox = self.importModule.getBox()

    def reloadParams(self, negMsgHandler=None):
        self.reloadBBox()
        return self.params               
    
    def delete(self, isInitialized, negMsgHandler=None):
        ''' delete this VisItem: remove the module '''
        _infoer.function = str(self.delete)
        _infoer.write(" ")
        if isInitialized:
            if hasattr(self, '_module') and self._module: theNet().remove(self._module)
        VisItem.delete(self, isInitialized, negMsgHandler)
   
    def _update(self, negMsgHandler):
        ''' _update is called from the run method to update the module parameter before execution
            + do init the module if necessary
            update module parameters should be realized in the parent class'''
        self._init(negMsgHandler)

        if not hasattr(self, 'importModule'):
            return
            
        # update input
        # vec variable
        # vec variable is not changable from the Gui at the moment

        #scalar variable
        if not self._lastScalarConnection==None :
            for scalarInput in self._scalarInputNames:
                disconnect( self._lastScalarConnection,  ConnectionPoint(self._module, scalarInput))
        if hasattr(self.params, 'secondVariable') and len(self._scalarInputNames)>0 and not self.params.secondVariable==None:
            self.importModule.executeData(self.params.secondVariable)
            scalarInConnect = self.importModule.dataConnectionPoint(self.params.secondVariable)
            if scalarInConnect:
                for scalarInput in self._scalarInputNames:
                    connect( scalarInConnect, ConnectionPoint(self._module, scalarInput) )
                self._lastScalarConnection=scalarInConnect

        # starting points from 2d part     
        if not self._last2DPartConnection==None:
            part2D = globalKeyHandler().getObject(self.params.use2DPartKey)
            if not part2D or not self._last2DPartConnection==part2D.importModule.geoConnectionPoint():
                for partInput in self._2DPartInputNames:
                    disconnect( self._last2DPartConnection,  ConnectionPoint(self._module, partInput))
                self._last2DPartConnection = None
        if len(self._2DPartInputNames)>0 and self.params.use2DPartKey!=None and self.params.use2DPartKey>=0 and self._last2DPartConnection==None:
            # sampling removed at the moment
            part2D = globalKeyHandler().getObject(self.params.use2DPartKey)
            part2D.importModule.executeGeo()         
            for partInput in self._2DPartInputNames:
                connect( part2D.importModule.geoConnectionPoint(), ConnectionPoint(self._module, partInput) )
                if self._firstGeoForVars:
                    self._firstGeoForVars = False
                    #setTransform to module
                    negMsgHandler.run(self.params.use2DPartKey)
            self._last2DPartConnection=part2D.importModule.geoConnectionPoint()

        self._module.setTitle( self.params.name )

    def register(self, negMsgHandler, paramList):
        """ register to receive events from covise """
        if negMsgHandler and self._module:
            if not self._registered:
                mL = []
                mL.append( self._module )
                negMsgHandler.registerCopyModules( mL, self )
                negMsgHandler.registerStartNotifier(self._module, self)
                negMsgHandler.registerParamsNotfier( self._module, self.key, paramList )
                self._registered=True
                
    def connectionPoint(self, outputName='GeometryOut0'):
        ''' return the object to be displayed
            the parent class is called by the class VisItem 
                parentClass connectionPoint should call this funtion'''
        if self._module:
            return ConnectionPoint(self._module, outputName)

    def getCoObjName(self, outputName='GeometryOut0'):
        ''' return the generated covise object name
            parentclass function is called by the class VisItem to check if an 
                object name registered by the COVER 
        '''
        if self._module:
            return self._module.getCoObjName(outputName)

    def run(self, runmode, negMsgHandler):
        if os.getenv('VR_PREPARE_DEBUG_RUN'):
            print("PartModuleVis.run")
        ''' create a new visulisation
            + register for events from Covise if not done yet
            + runmode RUN_GEO and RUN_OCT are ignored
            + update module parameter should be done in parentClass
            + exec the module
        '''
        #assert negMsgHandler

        if runmode==RUN_ALL:
            _infoer.function = str(self.run)
            _infoer.write("go")

            if not hasattr(self, 'importModule'):
                return

            self._update(negMsgHandler)

            # to be sure that the import module loaded all data
            if (self.__createOcttree):
                self.importModule.executeOct()

            geoExecuted = self.importModule.executeGeo()

            colorExecute = (hasattr(self.params, 'color') and self.params.color == VARIABLE) or (not hasattr(self.params, 'color'))
            if hasattr(self.params, 'variable') and self.params.variable and self.params.variable!='Select a variable' and self.params.variable != 'unset' and colorExecute:
                dataExecuted = self.importModule.executeData(self.params.variable)
                if not dataExecuted and not geoExecuted:
                    saveExecute(self._module)
            elif self._dataInputNames==[] and not geoExecuted:
                saveExecute(self._module)

    def __getstate__(self):
        mycontent = coKeydObject.__getstate__(self)
        del mycontent['_module']
        del mycontent['_moduleName']
        del mycontent['_lastDataConnection']        
        del mycontent['_lastScalarConnection']
        del mycontent['_last2DPartConnection']
        if ('_visualizerColorCP' in mycontent.keys()): # PartColoredVis
            del mycontent['_visualizerColorCP']
        return mycontent

    def recreate(self, negMsgHandler, parentKey, offset,
                 modName, geoInputNames=['meshIn'] , \
        octtreeInputNames=[], geoSampleNames=[], dataSampleNames=[],\
        dataInputNames=[], scalarInputNames=[], twoDInputNames=[], sampleType=1, bbox = True, octtree= True ) :
        #geoInputNames, octtreeInputNames, geoSampleNames, dataSampleNames,\
        #                dataInputNames, scalarInputNames, twoDInputNames, bbox, octtree   ):
        ''' recreate is called after all classes of the session have been unpickled '''
        PartModuleVisParams.mergeDefaultParams(self.params) # explicitly call mergeDefaultParams of this class
        #self.params.mergeDefaultParams()
        self._initBase(modName, geoInputNames, octtreeInputNames, geoSampleNames, dataSampleNames,\
                        dataInputNames, scalarInputNames, twoDInputNames, sampleType, bbox, octtree )
        VisItem.recreate(self, negMsgHandler, parentKey, offset)


    def setParams( self, params, negMsgHandler=None, sendToCover=True, realChange=None):
        ''' set parameters from outside
            + init module if necessary
            + mainly receive parameter changes from Gui
            + send status messages to COVER if state has changed
        '''
        _infoer.function = str(self.setParams)
        _infoer.write("setParams")
        
        if realChange==None:
            realChange = ParamsDiff( self.params, params )
        VisItem.setParams(self, params)

        # only connect to COVER if this visualizer really is visible and wasnt connected before
        if self._module:
            if self.params.isVisible and not self.__connectedToCOVER:
                VisItem.connectToCover( self, self )
                self.__connectedToCOVER = True

        
class PartModuleVisParams(VisItemParams):
    '''Params for ColoredVisItem'''
    def __init__(self):
        VisItemParams.__init__(self)
        self.boundingBox = None
        self.variable = None
        PartModuleVisParams.mergeDefaultParams(self) # explicitly call mergeDefaultParams of this class

    def mergeDefaultParams(self):
        defaultParams = {
            'variable' : None,
            'boundingBox' : None
        }
        mergeGivenParams(self, defaultParams)
