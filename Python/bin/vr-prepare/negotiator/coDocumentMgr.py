
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH
#
# connects Modul CoverDocuments and communicates with it
#
#

import VRPCoviseNetAccess

from VRPCoviseNetAccess import (
    connect,
    disconnect,
    ConnectionPoint,
    globalRenderer,
    theNet,
    saveExecute)

from KeydObject import globalKeyHandler, globalPresentationMgrKey, RUN_ALL, VIS_DOCUMENT
from Utils import ParamsDiff, getDoubleInLineEdit, mergeGivenParams
from VisItem import VisItem, VisItemParams
from coPyModules import CoverDocument
from coGRMsg import coGRAddDocMsg, coGRDocVisibleMsg, coGRSetDocPageMsg, coGRSetDocPageSizeMsg, coGRSetDocPositionMsg, coGRSetDocScaleMsg
import covise
from Utils import getExistingFilename
import socket

from ErrorManager import CoviseFileNotFoundError

from printing import InfoPrintCapable
_infoer = InfoPrintCapable()
_infoer.doPrint = False # True #

DOCUMENT_ID_STRING = 'Document'

class coDocumentMgr(VisItem):
    """ class to handle project files """
    def __init__(self):
        _infoer.function = str(self.__init__)
        _infoer.write("")
        VisItem.__init__(self, VIS_DOCUMENT, 'DocumentMgr')
        self.params = coDocumentMgrParams()
        self.name = self.params.name
        self.params.isVisible = True
        self.__initBase()

    def __initBase(self):
        _infoer.function = str(self.__initBase)
        _infoer.write("")
        self.__connected = False
        self.coverDocument = None
        self.__loaded = False

    def __update(self):
        """ __update is called from the run method to update the module parameter before execution
            + update module parameters """
        _infoer.function = str(self.__update)
        _infoer.write("")
        if self.coverDocument==None:
            self.coverDocument = CoverDocument()
            theNet().add(self.coverDocument)

        #update params
        self.coverDocument.set_Filename(self.params.imageName)
        self.coverDocument.set_Title(self.params.documentName)
        #if self.params.singlePage==True:
        #     self.coverDocument.set_SinglePage("TRUE")
        #else:
        #     self.coverDocument.set_SinglePage("FALSE")

    def connectionPoint(self):
        """ return the object to be displayed
            called by the class VisItem """
        _infoer.function = str(self.connectionPoint)
        _infoer.write("")
        if self.coverDocument:
            return ConnectionPoint(self.coverDocument, 'Document')

    def getCoObjName(self):
        _infoer.function = str(self.getCoObjName)
        if self.coverDocument:
            _infoer.write("%s" %(self.coverDocument.getCoObjName('Document')))
            return self.coverDocument.getCoObjName('Document')


    def setParams( self, params, negMsgHandler=None, sendToCover=True):
        _infoer.function = str(self.setParams)
        _infoer.write(" ")
        
        self.sendImagePathToWebserver()

        realChange = ParamsDiff( self.params, params )

        # make sure, variable change is removed from params again
        if hasattr(self.params, 'changed'):
            oldChanged = self.params.changed
        else:
            oldChanged = False

        VisItem.setParams(self, params)
        self.params.changed = oldChanged

        #if 'imageName' in realChange and sendToCover:
        #    self.sendImageName()
        #    self.sendPosition()
        #    self.sendVisibility()
        if 'pageNo' in realChange and sendToCover:
            self.sendPageNo()
        if 'size' in realChange and sendToCover:
            self.sendSize()
        if 'scaling' in realChange and sendToCover:
           self.sendScaling()
        if 'pos' in realChange and sendToCover:
            self.sendPosition()

        # changed in realChange happens when doc is changed in gui
        if 'changed' in realChange:
            # tell the coPresentationMgr that doc has changed
            globalKeyHandler().getObject(globalPresentationMgrKey).documentChanged(self.params.documentName, self.params.pos, self.params.isVisible, self.params.scaling, self.params.size, negMsgHandler)

    
    def sendImagePathToWebserver(self):
        if covise.coConfigIsOn("vr-prepare.RemoteDeviceControll", False) and self.params.imageName != None:
            
            import os
            import vtrans
            import PathTranslator
            
            # starting point of 
            # image path processing
            imageName = self.params.imageName
            sNum = '1'
            
            # compute the image corresponding to step number 
            imageName = imageName.replace('\\', '/')
            ## just to be shure
            imageName = imageName.replace(os.sep, '/')
            
            pos = imageName.rfind('/')
            if pos >= 0:
                dirName = imageName[:pos]
                baseName = imageName[pos+1:]
                
                imageNumber = int(self.params.pageNo)
                imageSuffix = ''
                
                nPos = baseName.rfind('.')
                if nPos >= 0:
                    sNum = baseName[:nPos]
                    imageSuffix = baseName[nPos+1:]
                    iNum = int(sNum)
                    
                    # for backward compatibility
                    # in cc versions lesser than 3.2
                    # images were alloed to start with zero (0.png) 
                    if iNum == 0:
                        imageNumber = int(self.params.pageNo) - 1
            
                #compose all back togerther
                imageName = dirName + '/' + str(imageNumber) + '.' + imageSuffix             
                
            
            # get the language environment settings
            coPath = vtrans.covisePath
            localePrefix = covise.getCoConfigEntry("COVER.Localization.LocalePrefix")
            languageLocale = vtrans.languageLocale
            loc = localePrefix + "/" + languageLocale
            fullPathToImage = coPath + "/" +imageName
            
            # retrieve localized path
            fullPathToImage = PathTranslator.translate_path(loc, fullPathToImage)
            
            # retrieve relative path again
            imageName = fullPathToImage[len(coPath)+1:]
            
            path_to_current_image = imageName
            
            # again...
            # this time for the browser
            # compute the image corresponding to step number 
            path_to_current_image = path_to_current_image.replace('\\', '/')
            ## just to be shure
            path_to_current_image = path_to_current_image.replace(os.sep, '/')
            
            # pack the message
            msg = 'VRT IMAGE ' + path_to_current_image + '.' + str(self.params.maxPage)+ '.' + sNum
            
            # send it away
            destinationHost = covise.getCoConfigEntry("vr-prepare.RemoteDeviceHost")
            if not destinationHost:
               destinationHost = "127.0.0.1"
            destinationPort = covise.getCoConfigEntry("vr-prepare.RemoteDevicePort")
            if not destinationPort:
               destinationPort = "44142"
            sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
            sock.sendto( str(msg),(destinationHost,int(destinationPort)) )



    def recreate(self, negMsgHandler, parentKey, offset):
        """ recreate is called after all classes of the session have been unpickled """
        _infoer.function = str(self.recreate)
        _infoer.write(" ")
        coDocumentMgrParams.mergeDefaultParams(self.params) # explicitly call mergeDefaultParams of this class
        self.__initBase()
        VisItem.recreate(self, negMsgHandler, parentKey, offset)
        self.params.changed = False

        if getExistingFilename(self.params.imageName) == None:
            raise CoviseFileNotFoundError(self.params.imageName)


    def registerCOVISEkey( self, covise_key):
        """ check if object name was created by this visItem
            and if yes store it """
        if not self.coverDocument==None:
            if self.createdKey( covise_key ):
                firstTime = not self.keyRegistered()
                self.covise_key = covise_key
                self.sendCaseName()
                self.sendCaseTransform()
                self.sendName()
                self.sendSize()
                self.sendScaling()
                self.sendPosition()
                self.sendPageNo()
                self.sendVisibility()
                return (True, firstTime)
        return (False, False)

    def run(self, runmode, negMsgHandler=None):
        if runmode==RUN_ALL:
            _infoer.function = str(self.run)
            _infoer.write("go")
            self.__update()
            if not self.__connected :#and self.params.isVisible:
                VisItem.connectToCover( self, self )
                self.__connected=True
            if not self.__loaded:
                saveExecute(self.coverDocument)
                self.__loaded=True

    def sendVisibility(self):
        """ send visible msg to cover """
        _infoer.function = str(self.sendVisibility)
        _infoer.write("visible %d %s" %(self.params.isVisible, str(self.params.documentName)))
        if (self.params.documentName!=None and self.params.imageName!=None and self.covise_key!='No key'):
            if self.params.isVisible:
                visible = 1
            else:
                visible = 0
            msg = coGRDocVisibleMsg( self.params.documentName, visible )
            covise.sendRendMsg(msg.c_str())

    def sendPosition(self):
        """ send position msg to cover """
        _infoer.function = str(self.sendPosition)
        _infoer.write("position [%f %f %f]" %(self.params.pos[0], self.params.pos[1], self.params.pos[2]))
        if (self.params.documentName!=None and self.params.imageName!=None and self.covise_key!='No key'):
            msg = coGRSetDocPositionMsg( self.params.documentName, self.params.pos[0], self.params.pos[1], self.params.pos[2] )
            covise.sendRendMsg(msg.c_str())

    def sendSize(self):
        """ send size msg to cover """
        _infoer.function = str(self.sendSize)
        _infoer.write("size [%f %f]" %(self.params.size[0], self.params.size[1]))
        if (self.params.size!=(-1,-1) and self.params.documentName!=None and self.params.imageName!=None and self.covise_key!='No key'):
            msg = coGRSetDocPageSizeMsg( self.params.documentName, self.params.pageNo, self.params.size[0], self.params.size[1] )
            covise.sendRendMsg(msg.c_str())

    def sendScaling(self):
        """ send scaling msg to cover """
        _infoer.function = str(self.sendScaling)
        _infoer.write("scaling %s" %(self.params.scaling))
        if (self.params.documentName!=None and self.params.imageName!=None and self.covise_key!='No key'):
            msg = coGRSetDocScaleMsg( self.params.documentName, self.params.scaling )
            covise.sendRendMsg(msg.c_str())

    def sendPageNo(self):
        """ send pageNo msg to cover """
        _infoer.function = str(self.sendPageNo)
        _infoer.write("pageno %d" %(self.params.pageNo))
        if (self.params.documentName!=None and self.params.imageName!=None and self.covise_key!='No key'):
            msg = coGRSetDocPageMsg( self.params.documentName, self.params.pageNo )
            covise.sendRendMsg(msg.c_str())
            
    def delete(self, isInitialized, negMsgHandler=None):
        ''' delete this CoviseVis: remove the module '''
        _infoer.function = str(self.delete)
        _infoer.write(" ")
        if isInitialized:
            theNet().remove(self.coverDocument)
        VisItem.delete(self, isInitialized, negMsgHandler)

class coDocumentMgrParams(VisItemParams):
    def __init__(self):
        VisItemParams.__init__(self)
        self.name         = 'DocumentMgrParams'
        self.singlePage = False
        self.changed = False
        coDocumentMgrParams.mergeDefaultParams(self) # explicitly call mergeDefaultParams of this class

    def mergeDefaultParams(self):
        defaultParams = {
            'documentName' : None,
            'imageName' : None,
            'singlePage' : True,
            'pageNo' : 1,
            'minPage': 1,
            'maxPage': 1,
            'pos' : ( 0, 0, 0 ),
            'scaling' : 1.0,
            'size' : ( -1, -1 ),
            'currentImage' : None
        }
        mergeGivenParams(self, defaultParams)
