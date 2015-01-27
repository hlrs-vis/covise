from Ensight2CoviseGuiBase import Ensight2CoviseGuiBase
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject, pyqtSignal
import sys
import os
import os.path
import time
import re
import covise

try:
    import cPickle as pickle
except:
    import pickle

from ErrorLogAction import ErrorLogAction, ERROR_EVENT
from ChoiceGetterAction import ChoiceGetterAction, NOTIFIER_EVENT
from IntGetterAction import IntGetterAction
from CoviseMsgLoop import CoviseMsgLoop, CoviseMsgLoopAction
from coPyModules import ReadEnsight, RWCovise, Transform, GetSubset, FixUsg, GetSetelem
from PatienceDialogManager import PatienceDialogManager
from coviseModuleBase import net
from coviseCase import (
    CoviseCaseFileItem,
    CoviseCaseFile,
    GEOMETRY_2D,
    GEOMETRY_3D,
    SCALARVARIABLE,
    VECTOR3DVARIABLE)


class PartDescriptionAnalyzer(object):

    def __init__(self, candidateString):
        # Regexp fitting for a 2d-part-description that comes with a covise-message.
        self.regexp2dPartDescription = r'(.*)\|(.*)\|(.*)\|(.*)\|.*(2D|d).*'
        self.regexp3dPartDescription = r'(.*)\|(.*)\|(.*)\|(.*)\|.*(3D|d).*'
        self.regexp2dOr3dPartDescription = r'(.*)\|(.*)\|(.*)\|(.*)\|.*(2|3D|d).*'
        self.candidateString = candidateString

    def is2dPartDescription(self):

        """Return if self.candidateString is a part-2d-description.

        For to know what a part-description is use the
        source here and in ReadEnsight until there is a
        better documentation.

        """

        return bool(
            re.match(
            self.regexp2dPartDescription, self.candidateString))

    def is3dPartDescription(self):

        """Return if self.candidateString is a part-3d-description.

        For to know what a part-description is use the
        source here and in ReadEnsight until there is a
        better documentation.

        """

        return bool(
            re.match(
            self.regexp3dPartDescription, self.candidateString))

    def pullOutPartIdAndName(self):

        """Return part-id and name from self.candidateString."""

        assert self.is2dPartDescription() or self.is3dPartDescription()
        matchie = re.match(
            self.regexp2dOr3dPartDescription, self.candidateString)
        coviseId = int(matchie.group(1).strip())
        ensightId = int(matchie.group(2).strip())
        name = matchie.group(3).strip()
        return coviseId, ensightId, name


class PartsCollectorAction(CoviseMsgLoopAction):

    """Action to collect part information when using
    covise with an ReadEnsight.

    Gets the information from covise-info-messages.

    """

    def __init__ (self):
        CoviseMsgLoopAction.__init__(
            self,
            self.__class__.__module__ + '.' + self.__class__.__name__,
            56, # Magic number 56 is covise-msg type INFO
            'Collect parts-names and numbers from covise-info-messages.')

        self.__partsinfoFinished = False

        self.__refAndNameDict2d = {}
        self.__refAndNameDict3d = {}


    def run(self, param):
        assert 4 == len(param)
        # assert param[0] is a modulename
        # assert param[1] is a number
        # assert param[2] is an ip
        # assert param[3] is a string

#       print str(self.run)

        msgText = param[3]

        #print str(msgText)
        analyzer = PartDescriptionAnalyzer(msgText)
        if analyzer.is2dPartDescription():
            coviseId, ensightId, name = analyzer.pullOutPartIdAndName()
            self.__refAndNameDict2d[ensightId] = name
            #logFile.write("Part: id = %d name = %s\n"%(partid, name))
            #logFile.flush()
            #print "CoviseId = %d\tEnsightId = %d\tName = %s"%(coviseId, ensightId, name)

        if analyzer.is3dPartDescription():
            coviseId, ensightId, name = analyzer.pullOutPartIdAndName()
            self.__refAndNameDict3d[ensightId] = name
            #logFile.write("Part: id = %d name = %s\n"%(coviseId, name))
            #logFile.flush()
            #print "CoviseId = %d\tEnsightId = %d\tName = %s"%(coviseId, ensightId, name)
            
        if msgText == "...Finished: List of Ensight Parts":
            #logFile.write("\n")
            #logFile.flush()
            self.__partsinfoFinished = True
#        else:
#            infoer.function = str(self.run)
#            infoer.write(
#                'Covise-message "%s" doesn\'t look like a parts-description.'
#                % str(msgText))

    def getRefNameDict2dParts(self):
        return self.__refAndNameDict2d

    def getRefNameDict3dParts(self):
        return self.__refAndNameDict3d

    def waitForPartsinfoFinished(self):
        #print "Ensight2CoviseBegin.py PartsCollectorAction.waitForPartsinfoFinished"
        while not self.__partsinfoFinished: 
            pass



class Ensight2CoviseGui(Ensight2CoviseGuiBase):

    def __init__(self):
        #print "Ensight2CoviseGui.__init__"
        
        # init base class
        Ensight2CoviseGuiBase.__init__(self, None)

        # connect buttons
        self.outputDirLineEdit.returnPressed.connect(self.setOutputDir)
        self.byteswapped.stateChanged.connect(self.setByteswap)
        self.startConversionPushButton.clicked.connect(self.startConversion)
        
        # initialize output directory
        InitialDatasetSearchPath = covise.getCoConfigEntry("vr-prepare.InitialDatasetSearchPath")
        if not InitialDatasetSearchPath:
            InitialDatasetSearchPath = os.getcwd()
        self.currentFilePath = InitialDatasetSearchPath

        self.scale = 1.0

        # disable all buttons at beginning
        self.settingsFrame.setEnabled(False)
        self.startConversionFrame.setEnabled(False)
        self.outputDirFrame.setEnabled(False)
        self.isByteSwapped = True

    def closeEvent(self, event):
        covise.clean()
        covise.quit()

    def fileOpen(self):
        #print "Ensight2CoviseGui.fileOpen"
        fd = QtWidgets.QFileDialog(self)
        fd.setMinimumWidth(1050)
        fd.setMinimumHeight(700)
        fd.setNameFilter('Ensight case File (*.case *.encas *.CASE *.ENCAS)')
        fd.setWindowTitle('Open Ensight Case File')
        fd.setDirectory(self.currentFilePath)

        acceptedOrRejected = fd.exec_()
        if acceptedOrRejected != QtWidgets.QDialog.Accepted :
            return
        filenamesQt = fd.selectedFiles()
        if filenamesQt.isEmpty():
            return
        self.currentFilePath = os.path.dirname(str(filenamesQt[0]))
        self.fullEnsightCaseName = str(filenamesQt[0])
        
        
        # try to open file
        if not os.access(self.fullEnsightCaseName, os.R_OK):
            self.statusText.append("ERROR: Could not open file "+self.fullEnsightCaseName+ " - not readable")  
        else:

            # start modules
            self.startModules()
       
            # disable file open
            self.fileOpenAction.setEnabled(False) 
        
            # set output File path
            self.outputFilePath = self.currentFilePath + "/CoviseDaten/"
            if os.path.isdir(self.outputFilePath):
                pass
            else:
                try:
                    os.mkdir(self.outputFilePath)
                except(OSError):
                    self.statusText.append("ERROR: Could not create directory "+str(self.outputFilePath)+" check permissions and enter again or select another directory")
                    self.outputFilePath = None
                
            if (self.outputFilePath):
                self.outputDirLineEdit.setText(self.outputFilePath)
                # enable all buttons           
                self.outputDirFrame.setEnabled(True)
                self.startConversionFrame.setEnabled(True)
                self.settingsFrame.setEnabled(True)


    def fileExit(self):
        #print "Ensight2CoviseGui.fileExit"
        self.close()
        sys.exit()

    def setByteswap(self, i):
        self.isByteSwapped = self.byteswapped.isChecked()       

    def setOutputDir(self):
        #print "Ensight2CoviseGui.setOutputDir"
        self.outputFilePath=str(self.outputDirLineEdit.text())
        if not os.path.isdir(self.outputFilePath):
            try:
                os.mkdir(self.outputFilePath)
                self.statusText.append("\nINFO: created directory "+self.outputFilePath)
            except(OSError):
                self.statusText.append("\nERROR: Could not create directory "+self.outputFilePath+" check permissions and enter again or select another directory")
                self.outputFilePath = None
        else:
            self.statusText.append("\nINFO: directory "+self.outputFilePath+" exists already")        
         
        
                    
    def customEvent(self,e):
        pass

   
    
    def startModules(self):
        #print "Ensight2CoviseGui.startModules"
        self.aErrorLogAction = ErrorLogAction()
        CoviseMsgLoop().register(self.aErrorLogAction)

        global theNet
        theNet = net()
        
        self.scalarVariables3DGetterAction = ChoiceGetterAction()
        self.vectorVariables3DGetterAction = ChoiceGetterAction()
        self.scalarVariables2DGetterAction = ChoiceGetterAction()
        self.vectorVariables2DGetterAction = ChoiceGetterAction()

 

        # MODULE: ReadEnsight
        self.ReadEnsight_1 = ReadEnsight()
        theNet.add( self.ReadEnsight_1 )

        self.aPartsCollectorAction = PartsCollectorAction()
        CoviseMsgLoop().register(self.aPartsCollectorAction)


        # hang in variable-getters
        self.ReadEnsight_1.addNotifier('data_for_sdata1_3D', self.scalarVariables3DGetterAction)
        self.ReadEnsight_1.addNotifier('data_for_vdata1_3D', self.vectorVariables3DGetterAction)
        self.ReadEnsight_1.addNotifier('data_for_sdata1_2D', self.scalarVariables2DGetterAction)
        self.ReadEnsight_1.addNotifier('data_for_vdata1_2D', self.vectorVariables2DGetterAction)


        # set parameter values
        if self.isByteSwapped == True:
            self.ReadEnsight_1.set_data_byte_swap( "TRUE" )
        else:
            self.ReadEnsight_1.set_data_byte_swap( "FALSE" )
        self.ReadEnsight_1.set_case_file( self.fullEnsightCaseName )
        self.ReadEnsight_1.set_include_polyhedra( "TRUE")
        self.ReadEnsight_1.set_enable_autocoloring( "FALSE" )
       
        # wait for choices to be updated
        self.scalarVariables3DGetterAction.waitForChoices()
        self.vectorVariables3DGetterAction.waitForChoices()
        self.scalarVariables2DGetterAction.waitForChoices()
        self.vectorVariables2DGetterAction.waitForChoices()

        # wait for the part info message
        self.aPartsCollectorAction.waitForPartsinfoFinished()

                
        # get variables
        self.scalarVariables3D=self.scalarVariables3DGetterAction.getChoices()
        self.vectorVariables3D=self.vectorVariables3DGetterAction.getChoices()
        self.scalarVariables2D=self.scalarVariables2DGetterAction.getChoices()
        self.vectorVariables2D=self.vectorVariables2DGetterAction.getChoices()


        text = "Ensight Case File: %s"%(self.fullEnsightCaseName)
        self.statusText.append(text)

        self.statusText.append("\n3D parts:")
        for partid in self.aPartsCollectorAction.getRefNameDict3dParts().keys():
            partname = self.aPartsCollectorAction.getRefNameDict3dParts()[partid]
            text = "\tPart %d = %s"%(partid, partname)
            self.statusText.append(text)

        self.statusText.append("3D Part Variables:")
        for svar in self.scalarVariables3D:
           text= "\t%s(scalar)"%(svar)   
           self.statusText.append(text)
        for vvar in self.vectorVariables3D:
           text= "\t%s(vector)"%(vvar)   
           self.statusText.append(text)              
                                  
        self.statusText.append("\n2D parts:")
        for partid in self.aPartsCollectorAction.getRefNameDict2dParts().keys():
            partname = self.aPartsCollectorAction.getRefNameDict2dParts()[partid]
            text = "\tPart %d = %s"%(partid, partname)
            self.statusText.append(text)

        self.statusText.append("2D Part Variables:")
        for svar in self.scalarVariables2D:
           text= "\t%s(scalar)"%(svar)   
           self.statusText.append(text)
        for vvar in self.vectorVariables2D:
           text= "\t%s(vector)"%(vvar)   
           self.statusText.append(text) 
           
        QtWidgets.QApplication.processEvents()
       
         

        # MODULE: RWCovise
        self.RWCovise_1 = RWCovise()
        theNet.add( self.RWCovise_1 )
        self.RWCovise_1.set_stepNo( 0 )
        self.RWCovise_1.set_rotate_output( "FALSE" )
        self.RWCovise_1.set_rotation_axis( 3 )
        self.RWCovise_1.set_rot_speed( 2.000000 )

        self.cocase = CoviseCaseFile()
        cn = os.path.basename(str(self.fullEnsightCaseName))
        self.cocasename = cn[0: cn.rfind('.')]
        
        
    def startConversion(self):
        #print "Ensight2CoviseGui.startConversion"

        # Module Transform
        self.scale = float(str(self.leScale.text()))
        if self.scale != 1:
            if not hasattr(self, "Transform_1"):
                self.Transform_1 = Transform()
                theNet.add( self.Transform_1 )
                self.Transform_1.set_Transform( 5 )
                self.Transform_1.set_createSet( "FALSE" )
            self.Transform_1.set_scaling_factor( self.scale )

        self.outputDirFrame.setEnabled(False)
        self.startConversionFrame.setEnabled(False)
        self.settingsFrame.setEnabled(False)
        #self.statusText.clear()
        text="\nStarting Conversion in %s..."%(self.outputFilePath)
        self.statusText.append(text)
        QtWidgets.QApplication.processEvents()
        self.startConversionOf3DParts()
        self.startConversionOf2DParts()

        QtWidgets.QApplication.processEvents()
        CoviseMsgLoop().unregister(self.aPartsCollectorAction)
        CoviseMsgLoop().unregister(self.aErrorLogAction)

        theNet.remove( self.ReadEnsight_1 )

        text="\nWriting cocase to file..."
        self.statusText.append(text)
        pickleFile = self.outputFilePath + self.cocasename + '.cocase'
        counter=0
        while os.path.isfile(pickleFile):
            text= "! A file named %s is already available ... trying a new name"%(pickleFile)
            self.statusText.append(text)
            QtWidgets.QApplication.processEvents()            
            counter=counter+1
            pickleFile = self.outputFilePath + self.cocasename + "_"+str(counter)     
        output = open(pickleFile, 'wb')
        pickle.dump(self.cocase,output)
        output.close()

        text="cocasefile written to %s\n"%(pickleFile,)
        self.statusText.append(text)
        QtWidgets.QApplication.processEvents()


        text="\nConversion finished!"
        self.statusText.append(text)


    def startConversionOf3DParts(self):     
        for partid in self.aPartsCollectorAction.getRefNameDict3dParts().keys():
            #if int(partid) >= int(self.comboStartId.getCurrentIndex()):
            # get partname
            partname = self.aPartsCollectorAction.getRefNameDict3dParts()[partid]
            text = "\nConverting grid of part %s please be patient..."%(partname)
            self.statusText.append(text)            
            QtWidgets.QApplication.processEvents() 
            # connect modules
            if self.scale != 1:
                theNet.connect( self.ReadEnsight_1, "geoOut_3D", self.Transform_1, "geo_in" )
                theNet.connect( self.Transform_1, "geo_out", self.RWCovise_1, "mesh_in" )
            else:
                theNet.connect( self.ReadEnsight_1, "geoOut_3D", self.RWCovise_1, "mesh_in" )
                
            # select part
            self.ReadEnsight_1.set_choose_parts( str(partid) )
            
            # clean partname
            if "/" in partname:
                text="! Removing the / in partname = %s\n"%(partname)
                self.statusText.append(text) 
                QtWidgets.QApplication.processEvents()            
                partname=partname.replace("/","")
                
            # create RW Covise name
            covisename = self.outputFilePath +partname + "-3D.covise"
            # check if file is already available
            #print "rwcovisename=", covisename
            counter=0
            while os.path.isfile(covisename):
                text= "! A file named %s is already available ... trying a new name"%(covisename)
                self.statusText.append(text)
                QtWidgets.QApplication.processEvents()            
                counter=counter+1
                covisename = self.outputFilePath + partname + str(counter) + "-3D.covise"            

            QtWidgets.QApplication.processEvents()
                                     
            self.RWCovise_1.set_grid_path( covisename )
            QtWidgets.QApplication.processEvents() 
            # execute
            self.ReadEnsight_1.execute()
            theNet.finishedBarrier()
        
            #theNet.save( "grid.net" )

            # write logfile
            text="Converted grid of part %s to covise file %s"%(partname, covisename)
            self.statusText.append(text)            
            QtWidgets.QApplication.processEvents() 
            # create cocase item
            item3D = CoviseCaseFileItem(partname, GEOMETRY_3D, os.path.basename(covisename))
           
            # disconnect modules
            if self.scale !=1:
                theNet.disconnect( self.ReadEnsight_1, "geoOut_3D", self.Transform_1, "geo_in" )
                theNet.disconnect( self.Transform_1, "geo_out", self.RWCovise_1, "mesh_in" )
            else:
                theNet.disconnect(self. ReadEnsight_1, "geoOut_3D", self.RWCovise_1, "mesh_in" )
                
        
            QtWidgets.QApplication.processEvents() 

            # scalar variables
        
            # connect modules
            if self.scale!=1:
                theNet.connect( self.ReadEnsight_1, "geoOut_3D", self.Transform_1, "geo_in" )
                theNet.connect( self.ReadEnsight_1, "sdata1_3D", self.Transform_1, "data_in0" )
                theNet.connect( self.Transform_1, "data_out0", self.RWCovise_1, "mesh_in" )
            else:
                theNet.connect( self.ReadEnsight_1, "sdata1_3D", self.RWCovise_1, "mesh_in" )
       
            # loop over scalar variables
            choice=1
            for svar in self.scalarVariables3D:
                text= "\nConverting scalar variable %s of part %s, please be patient..."%(svar, partname)   
                self.statusText.append(text)
                QtWidgets.QApplication.processEvents() 
                    
                # select variable
                choice+=1
                self.ReadEnsight_1.set_data_for_sdata1_3D( choice )
                # clean variablename
                if "/" in svar:
                    text="! Removing the / in svar = %s\n"%(svar,)
                    statusText.append(text)
                    QtWidgets.QApplication.processEvents() 
                    svar=svar.replace("/","")
                # create RWCovise name
                covisename = self.outputFilePath + partname + "-" + svar + "-3D.covise"
                # check if file is already available
                counter=0
                while os.path.isfile(covisename):
                    text="! A file named %s is already available ... trying a new name"%(covisename)
                    self.statusText.append(text)
                    QtWidgets.QApplication.processEvents() 
                    counter=counter+1
                    covisename = self.outputFilePath + partname + str(counter) + "-" + svar + "-3D.covise"
                    
                
                
                self.RWCovise_1.set_grid_path( covisename )
                QtWidgets.QApplication.processEvents()     
                
                # execute
                self.ReadEnsight_1.execute()
                theNet.finishedBarrier()
                # write logfile
                text="Converted scalar variable %s of part %s to file %s"%(svar, partname, covisename)
                self.statusText.append(text)
                QtWidgets.QApplication.processEvents() 
                
                # add variable to cacase item
                item3D.addVariableAndFilename(svar, covisename, SCALARVARIABLE)
                
                #theNet.save( "scalar.net" )

            # disconnect modules
            if self.scale!=1:
                theNet.disconnect( self.ReadEnsight_1, "geoOut_3D", self.Transform_1, "geo_in" )
                theNet.disconnect( self.ReadEnsight_1, "sdata1_3D", self.Transform_1, "data_in0" )
                theNet.disconnect( self.Transform_1, "data_out0", self.RWCovise_1, "mesh_in" )
            else:
                theNet.disconnect( self.ReadEnsight_1, "sdata1_3D", self.RWCovise_1, "mesh_in" )
 
            QtWidgets.QApplication.processEvents() 
            
            

            # vector variables

            # connect modules
            if self.scale!=1:
                theNet.connect( self.ReadEnsight_1, "geoOut_3D", self.Transform_1, "geo_in" )
                theNet.connect( self.ReadEnsight_1, "vdata1_3D", self.Transform_1, "data_in0" )
                theNet.connect( self.Transform_1, "data_out0", self.RWCovise_1, "mesh_in" )
            else:
                theNet.connect( self.ReadEnsight_1, "vdata1_3D", self.RWCovise_1, "mesh_in" )

            # loop over variables
            choice=1
            for vvar in self.vectorVariables3D:
                text= "\nConverting vector variable %s of part %s, please be patient..."%(vvar, partname)   
                self.statusText.append(text)
                QtWidgets.QApplication.processEvents() 

                # select variable
                choice+=1
                self.ReadEnsight_1.set_data_for_vdata1_3D( choice )
                # clean variablename
                if "/" in vvar:
                    text="! Removing the / in vvar = %s\n"%(vvar,)
                    self.statusText.append(text)
                    QtWidgets.QApplication.processEvents() 
                    partname=partname.replace("/","")
                # create covisename
                covisename = self.outputFilePath + partname + "-" + vvar + "-3D.covise"
                # check if file is already available
                counter=0
                while os.path.isfile(covisename):
                    text="! A file named %s is already available trying a new name"%(covisename)
                    self.statusText.append(text)
                    QtWidgets.QApplication.processEvents() 
                    counter=counter+1
                    covisename = self.outputFilePath + partname + str(counter) + "-" + vvar + "-3D.covise"
                self.RWCovise_1.set_grid_path( covisename )
                # execute
                self.ReadEnsight_1.execute()
                theNet.finishedBarrier()
                # write logfile
                text="Converted vector variable%s of part %s to file %s"%(vvar,partname, covisename)
                self.statusText.append(text)
                QtWidgets.QApplication.processEvents() 
                # add variable to cocase item
                item3D.addVariableAndFilename(vvar, covisename, VECTOR3DVARIABLE)
                 

            # disconnect modules
            if self.scale!=1:
                theNet.disconnect( self.ReadEnsight_1, "geoOut_3D", self.Transform_1, "geo_in" )
                theNet.disconnect( self.ReadEnsight_1, "vdata1_3D", self.Transform_1, "data_in0" )
                theNet.disconnect( self.Transform_1, "data_out0", self.RWCovise_1, "mesh_in" )
            else:
                theNet.disconnect( self.ReadEnsight_1, "vdata1_3D",self. RWCovise_1, "mesh_in" )

            # add the cocase item to the case file
            self.cocase.add(item3D)
            QtWidgets.QApplication.processEvents()    

    def startConversionOf2DParts(self):
        #print "__________START 2D---------" 
        for partid in self.aPartsCollectorAction.getRefNameDict2dParts().keys():
            partname = self.aPartsCollectorAction.getRefNameDict2dParts()[partid]
             # write logfile
            text="\nConverting surface of part %s, please be patient..."%(partname)
            self.statusText.append(text)
            QtWidgets.QApplication.processEvents() 
            # connect modules
            if self.scale!=1:
                theNet.connect( self.ReadEnsight_1, "geoOut_2D", self.Transform_1, "geo_in" )
                theNet.connect( self.Transform_1, "geo_out", self.RWCovise_1, "mesh_in" )
            else:
                theNet.connect( self.ReadEnsight_1, "geoOut_2D", self.RWCovise_1, "mesh_in" )
            # select part
            self.ReadEnsight_1.set_choose_parts( str(partid) )
            # clean partname
            if "/" in partname:
                text="! Removing the / in partname = %s\n"%(partname)
                self.statusText.append(text)
                QtWidgets.QApplication.processEvents() 
                partname=partname.replace("/","")
            # create RWCovise name
            covisename = self.outputFilePath + partname + "-2D.covise"
            counter=0
            while os.path.isfile(covisename):
                text="! A file named %s is already available trying a new name"%(covisename)
                self.statusText.append(text)
                QtWidgets.QApplication.processEvents() 
                counter=counter+1
                covisename = self.outputFilePath + partname + str(counter) + "-2D.covise"
            self.RWCovise_1.set_grid_path( covisename )
            
            # execute
            self.ReadEnsight_1.execute()
            theNet.finishedBarrier()
            
            # write logfile
            text="Converted surface of part %s to file %s"%(partname, covisename)
            self.statusText.append(text)
            QtWidgets.QApplication.processEvents() 
            # create cocase item
            item2D = CoviseCaseFileItem(partname, GEOMETRY_2D, os.path.basename(covisename))
            

            # disconnect the modules
            if self.scale !=1:
                theNet.disconnect( self.ReadEnsight_1, "geoOut_2D", self.Transform_1, "geo_in" )
                theNet.disconnect( self.Transform_1, "geo_out", self.RWCovise_1, "mesh_in" )
            else:
                theNet.disconnect( self.ReadEnsight_1, "geoOut_2D", self.RWCovise_1, "mesh_in" )

            #
            # scalar variables
            #

            # connect modules
            if self.scale!=1:
                theNet.connect( self.ReadEnsight_1, "geoOut_2D", self.Transform_1, "geo_in" )
                theNet.connect( self.ReadEnsight_1, "sdata1_2D", self.Transform_1, "data_in0" )
                theNet.connect( self.Transform_1, "data_out0", self.RWCovise_1, "mesh_in" )
            else:
                theNet.connect( self.ReadEnsight_1, "sdata1_2D", self.RWCovise_1, "mesh_in" )

            choice=1
            for svar in self.scalarVariables2D:
                # select variable
                choice+=1
                text="\nConverting scalar variable %s of part %s, please be patient..."%(svar, partname)
                self.statusText.append(text)
                QtWidgets.QApplication.processEvents() 
                self.ReadEnsight_1.set_data_for_sdata1_2D( choice )
                # clean variablename
                if "/" in partname:
                    text="! Removing the / in partname = %s\n"%(partname)
                    self.statusText.append(text)
                    QtWidgets.QApplication.processEvents() 
                    partname=partname.replace("/","")
                # create RWCovise name
                covisename = self.outputFilePath + partname + "-" + svar + "-2D.covise"
                counter=0
                while os.path.isfile(covisename):
                    text="! A file named %s is already available ... trying a new name"%(covisename)
                    self.statusText.append(text)
                    QtWidgets.QApplication.processEvents() 
                    counter=counter+1
                    covisename = self.outputFilePath + partname + str(counter) + "-" + svar + "-2D.covise"
                self.RWCovise_1.set_grid_path( covisename )
                
                # execute
                self.ReadEnsight_1.execute()
                theNet.finishedBarrier()
                # write logfile
                text="Converted scalar variable %s of part %s to file %s"%(svar, partname, covisename,)
                self.statusText.append(text)
                QtWidgets.QApplication.processEvents() 
                # add variable to cicase item
                item2D.addVariableAndFilename(svar, covisename, SCALARVARIABLE)
                # print memory usage of module
                #os.system('ps aux | grep ReadEnsight_1')

            # disconnect modules
            if self.scale!=1:
                theNet.disconnect( self.ReadEnsight_1, "geoOut_2D", self.Transform_1, "geo_in" )
                theNet.disconnect( self.ReadEnsight_1, "sdata1_2D", self.Transform_1, "data_in0" )
                theNet.disconnect( self.Transform_1, "data_out0", self.RWCovise_1, "mesh_in" )
            else:
                theNet.disconnect( self.ReadEnsight_1, "sdata1_2D", self.RWCovise_1, "mesh_in" )

            
            #  vector variables

            # connect modules
            if self.scale!=1:
                theNet.connect( self.ReadEnsight_1, "geoOut_2D", self.Transform_1, "geo_in" )
                theNet.connect( self.ReadEnsight_1, "vdata1_2D", self.Transform_1, "data_in0" )
                theNet.connect( self.Transform_1, "data_out0", self.RWCovise_1, "mesh_in" )
            else:
                theNet.connect( self.ReadEnsight_1, "vdata1_2D", self.RWCovise_1, "mesh_in" )
            choice=1
            for vvar in self.vectorVariables2D:
                # select variable
                choice+=1
                text="\nConverting vector variable %s of part %s, please be patient..."%(vvar, partname)
                self.statusText.append(text)
                QtWidgets.QApplication.processEvents() 
                self.ReadEnsight_1.set_data_for_vdata1_2D( choice )
                # clean partname
                if "/" in partname:
                    text="! Removing the / in partname = %s\n"%(partname,)
                    self.statusText.append(text)
                    QtWidgets.QApplication.processEvents() 
                    partname=partname.replace("/","")
                # create RWCovise name
                covisename = self.outputFilePath + partname + "-" + vvar + "-2D.covise"
                counter=0
                while os.path.isfile(covisename):
                    text="! A file named %s is already available ... trying a new name"%(covisename)
                    self.statusText.append(text)
                    QtWidgets.QApplication.processEvents() 
                    counter=counter+1
                    covisename = self.outputFilePath + partname + str(counter) + "-" + vvar + "-2D.covise"
                self.RWCovise_1.set_grid_path( covisename )
                # execute
                self.ReadEnsight_1.execute()
                theNet.finishedBarrier()
                # write logfile
                text="Converted vector variable%s of part %s to file %s"%(vvar, partname, covisename)
                self.statusText.append(text)
                QtWidgets.QApplication.processEvents() 
                # add varibale to coscase item
                item2D.addVariableAndFilename(vvar, covisename, VECTOR3DVARIABLE)
                

            # disconnect modules
            if self.scale!=1:
                theNet.disconnect( self.ReadEnsight_1, "geoOut_2D", self.Transform_1, "geo_in" )
                theNet.disconnect( self.ReadEnsight_1, "vdata1_2D", self.Transform_1, "data_in0" )
                theNet.disconnect( self.Transform_1, "data_out0", self.RWCovise_1, "mesh_in" )
            else:
                theNet.disconnect( self.ReadEnsight_1, "vdata1_2D", self.RWCovise_1, "mesh_in" )

            # add the cocase item to the case file
            self.cocase.add(item2D)
