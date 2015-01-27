
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH
try:
    import cPickle as pickle
except:
    import pickle

import os

from PyQt5 import QtCore, QtGui, QtWidgets

import Application
from KeydObject import TYPE_CASE, TYPE_VIEWPOINT_MGR, TYPE_PRESENTATION, RUN_ALL, RUN_OCT, globalKeyHandler, globalProjectKey
import coCaseMgr
import Neg2Gui
from Utils import unpickleProjectFile
import coprjVersion

class coSessionMgr(object):
    """ class to handle project files """

    def __init__(self):
        """Replace instance-data with case-data from file.

        The name of the instance will be set to the
        filename without path and extension.

        """
        self.__project = None

    def setProject( self, project):
        self.__project = project

    def getProject(self):
        return self.__project

    def recreate( self, msgHandler, aFilename, offset=0, replaceInPathList = [], reductionFactor=None, autoSync=False):

        auxName= os.path.basename(aFilename)
        self.name = auxName[0: auxName.rfind('.')]
        self.pathToCaseFile = os.path.dirname(aFilename)
        project = unpickleProjectFile(aFilename, replaceInPathList)

        msgHandler.setInRecreation(True)
        if offset==0:
            self.__project = project
            try:
                self.__project.recreate(msgHandler,-1, offset)
            except:     # halt on every exception
                msgHandler.setInRecreation(False)
                self.__project.delete(False, msgHandler)
                # key handler should be deleted while recreate
                globalKeyHandler().delete()
                del self.__project
                raise

            self.__project.run(RUN_OCT, msgHandler)
            self.__project.run(RUN_ALL, msgHandler)
        else :
            self.__checkDuplicateCaseNames(project)
            project.recreate(msgHandler,-1, offset)
            if autoSync: self.__autoSync( self.__project, self.__project, offset)
            project.run(RUN_OCT, msgHandler)
            project.run(RUN_ALL, msgHandler)

        msgHandler.setInRecreation(False)
        
    def __autoSync( self, project, baseobj, offset ):
        # synchronize two similiar project files
        for obj in baseobj.objects:
            self.__autoSync( project, obj, offset )

        if baseobj.typeNr in [TYPE_CASE, TYPE_VIEWPOINT_MGR, TYPE_PRESENTATION]:
            # don't synchronize these objects
            return

        for paramname in baseobj.params.__dict__:
            if not paramname in ["name", "colorTableKey"]:
                if( globalKeyHandler().hasKey(baseobj.key+offset) ): 
                    if hasattr( globalKeyHandler().getObject(baseobj.key+offset).params, paramname):
                        if not (baseobj.key, paramname) in project.params.sync:
                            project.params.sync[ (baseobj.key,paramname) ] = []
                        project.params.sync[ (baseobj.key,paramname) ].append( (baseobj.key+offset,paramname) )
                        if not (baseobj.key+offset, paramname) in project.params.sync:
                            project.params.sync[ (baseobj.key+offset,paramname) ] = []
                        project.params.sync[ (baseobj.key+offset,paramname) ].append( (baseobj.key,paramname) )

                        # set synchronized parameters of appended project to parameters of "master" project
                        globalKeyHandler().getObject(baseobj.key+offset).params.__dict__[paramname] = globalKeyHandler().getObject(baseobj.key).params.__dict__[paramname]

        # call sendParams to inform the GUI of a parameter change
        if( globalKeyHandler().hasKey(baseobj.key+offset) ):
            Neg2Gui.theNegMsgHandler().sendParams(baseobj.key + offset, globalKeyHandler().getObject(baseobj.key+offset).params)

    def __checkDuplicateCaseNames(self, project):
        origNames = {}
        self.__getOrChangeAllCaseNames( globalKeyHandler().getObject(globalProjectKey), origNames )
        newNames =  {}
        self.__getOrChangeAllCaseNames( project, newNames, origNames )

    def __getOrChangeAllCaseNames(self, projectObj, caseList, changeList=None):
        # return list with pairs list[name] = key
        for obj in projectObj.objects:
            self.__getOrChangeAllCaseNames( obj, caseList, changeList )
            if isinstance(obj,coCaseMgr.coCaseMgr ):
                if changeList==None:
                    caseList[ obj.params.name] = obj.key
                else :
                    if obj.params.name in changeList:
                        newName = obj.params.name+"*"
                        while newName in changeList:
                            newName += "*"
                        obj.params.name = newName
                    caseList[ obj.params.name] = obj.key

    def save( self, aFilename):
        if not self.__project==None:
            try:
                output = open(aFilename, 'wb')
                pickle.dump(self.__project, output)
                output.close()
            except IOError:
                # pop up warning dialog
                dialog = QtWidgets.QErrorMessage(Application.vrpApp.mw)
                dialog.showMessage("Permission denied. There is a problem saving the file. Maybe you have no permission in the directory or you ran out of disk space.")
                dialog.setWindowTitle("Permission Denied Error")
                acceptedOrRejected = dialog.exec_()
                return
            print("saving " , aFilename, self.__project.key)
