
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH


from VRPCoviseNetAccess import (
    connect,
    disconnect,
    ConnectionPoint,
    globalRenderer,
    theNet,
    saveExecute)

from VisItem import VisItem, VisItemParams
from coPyModules import RWCovise
from KeydObject import RUN_ALL, VIS_COVISE
from Utils import getExistingFilename

import os

from printing import InfoPrintCapable
_infoer = InfoPrintCapable()
_infoer.doPrint = False # True

class CoviseVis(VisItem):
    """ VisItem to show an covise object """
    def __init__(self):
        VisItem.__init__(self, VIS_COVISE, self.__class__.__name__)
        self.params = CoviseVisParams()
        self.params.isVisible = True
        self.__initBase()

    def __initBase(self):
        _infoer.function = str(self.__initBase)
        _infoer.write(" ")
        self.rwCovise = None
        self.__connected = False
        self.__registered=False
        self.__loaded = None

    def __init(self, negMsgHandler):
        """called from __update
           start rwCovise and connects to COVER
           send params to GUI"""
        _infoer.function = str(self.__init)
        _infoer.write(" ")
        if self.rwCovise==None:
            self.rwCovise = RWCovise()
            theNet().add(self.rwCovise)
            VisItem.connectToCover( self, self )
            self.__connected=True

    def __update(self, negMsgHandler):
        """ __update is called from the run method to update the module parameter before execution
            + update module parameters """
        _infoer.function = str(self.__update)
        _infoer.write(" ")
        self.__init(negMsgHandler)
        #update params
        # check if filename exists
        if getExistingFilename(self.params.filename) == None:
            raise IOError(self.params.filename)
            """
            fn = os.path.basename(filename)
            # if a path allready was selected
            if not _newPath == None:
                fname = VRPCoviseNetAccess._newPath + fn
                # test new path
                if not os.access(fname, os.R_OK):
                    filename = VRPCoviseNetAccess.changePath(filename, fn)
                else:
                    filename = fname
            else:
                filename = VRPCoviseNetAccess.changePath(filename , fn)
            """
  
        self.rwCovise.set_grid_path( self.params.filename )
        self.rwCovise.set_stepNo( self.params.stepNo )
        if (self.params.rotate_output==True):
            self.rwCovise.set_rotate_output('TRUE')
        else:
            self.rwCovise.set_rotate_output('FALSE')
        self.rwCovise.set_rotation_axis( self.params.rotation_axis )
        self.rwCovise.set_rot_speed( self.params.rot_speed )

    def createdKey(self, key):
        """ called during registration if key received from COVER """
        _infoer.function = str(self.createdKey)
        _infoer.write("%s, %s" % ( key, self.rwCovise.getCoObjName('mesh') ))
        importKey = self.rwCovise.getCoObjName('mesh')
        posCover = key.find("(")
        posImport = importKey.find("OUT")
        return ( importKey[0:posImport-1]==key[0:posCover] )

    def connectionPoint(self):
        """ return the object to be displayed
            called by the class VisItem """
        if self.rwCovise:
            return ConnectionPoint(self.rwCovise, 'mesh')

    def recreate(self, negMsgHandler, parentKey, offset):
        self.__initBase()
        VisItem.recreate(self, negMsgHandler, parentKey, offset)

    def run(self, runmode, negMsgHandler=None):
        if runmode==RUN_ALL:
            _infoer.function = str(self.run)
            _infoer.write("go")
            self.__update(negMsgHandler)
            if not self.__connected:
                VisItem.connectToCover( self, self )
                self.__connected=True
            if self.__loaded==None or self.__loaded!=self.params.filename:
                saveExecute(self.rwCovise)
                self.__loaded=self.params.filename

    def __register(self, negMsgHandler):
        """ register to receive events from covise """
        if negMsgHandler and self.rwCovise:
            if not self.__registered:
                mL = []
                mL.append( self.rwCovise )
                negMsgHandler.registerCopyModules( mL, self )
                paramList = [ 'grid_path', 'stepNo', 'rotate_output', 'rot_speed' ]
                negMsgHandler.registerParamsNotfier( self.rwCovise, self.key, paramList )
                self.__registered=True

    def setParams( self, params, negMsgHandler=None, sendToCover=True):
        """ set parameters from outside """
        _infoer.function = str(self.setParams)
        _infoer.write("setParams")

        VisItem.setParams(self, params)
        
    def delete(self, isInitialized, negMsgHandler=None):
        ''' delete this CoviseVis: remove the module '''
        _infoer.function = str(self.delete)
        _infoer.write(" ")
        if isInitialized:
            theNet().remove(self.rwCovise)
        VisItem.delete(self, isInitialized, negMsgHandler)        


class CoviseVisParams(VisItemParams):
    def __init__(self):
        VisItemParams.__init__(self)
        self.name = 'CoviseVisParams'
        self.filename = ''
        self.stepNo = 0
        self.rotate_output = False
        self.rotation_axis = 3
        self.rot_speed = 2.0
