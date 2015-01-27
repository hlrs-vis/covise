from Cfx2CoviseGuiBase import Cfx2CoviseGuiBase
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
from coPyModules import ReadCFX, RWCovise, Transform, GetSubset, FixUsg, SimplifySurface
from coviseModuleBase import net
from coviseCase import (
    CoviseCaseFileItem,
    CoviseCaseFile,
    GEOMETRY_2D,
    GEOMETRY_3D,
    SCALARVARIABLE,
    VECTOR3DVARIABLE)


class BoundariesCollectorAction(CoviseMsgLoopAction):

    """Action to collect part information when using
    covise with ReadCFX.

    Gets the information from covise-info-messages.

    """

    def __init__ (self):
        CoviseMsgLoopAction.__init__(
            self,
            self.__class__.__module__ + '.' + self.__class__.__name__,
            56, # Magic number 56 is covise-msg type INFO
            'Collect number of boundaries and names from covise-info-messages.')

        # start with true and set to false in first message
        self.__boundariesFinished = False
        self.__nboundaries = {}
        self.__boundaries = {}


    def run(self, param):
        if self.__boundariesFinished:
            return
        
        msgText = param[3]

        # test for number of boundaries string
        matchie = re.match(r'Zone \d+ (.*) (\d*) boundaries', msgText)
        if (matchie):
           self.__boundariesFinished = False
           self.__nboundaries[matchie.group(1).strip()] = int(matchie.group(2).strip())
              
        # test for boundaries name string
        matchie = re.match(r'(.*) - (\d*) (.*)', str(msgText)) 
        if (matchie):
           self.__boundariesFinished = False
           zone = matchie.group(1).strip()
           index = matchie.group(2).strip()
           name = matchie.group(3).strip()
           if  not zone in self.__boundaries: 
               self.__boundaries[zone] = {}
           self.__boundaries[zone][name]=index
           
           
        # test for Boundaries Info Finished string
        if msgText == 'Finished read boundaries':
           self.__boundariesFinished = True


    def getBoundaryNames(self, zone):
        if not zone in self.__boundaries:
            if "\x7f" in zone:
                zone=zone.replace("\x7f"," ")
        return self.__boundaries[zone]

    def getNumBoundaries(self, zone):
        return self.__nboundaries[zone]

    def waitForBoundaryInfoFinished(self):
        while not self.__boundariesFinished:
            pass

class Cfx2CoviseGui(Cfx2CoviseGuiBase):

    def __init__(self):
        
        InitialDatasetSearchPath = covise.getCoConfigEntry("vr-prepare.InitialDatasetSearchPath")
        if not InitialDatasetSearchPath:
            InitialDatasetSearchPath = os.getcwd()
        self.currentFilePath = InitialDatasetSearchPath
        self.ReadCfx_1 = 'None'
        self.scale=1.0
        self.mirror=0 # 0=none, 1=X, 2=Y, 3=Z
        self.rotAxisX=1.0
        self.rotAxisY=1.0
        self.rotAxisZ=1.0
        self.rotAngle=0.0
        self.composedGrid = False
        self.noGrid = False
        self.processBoundaries = True
        self.numVariables = 0
        self.fixdomain = "None"
        self.calculatePDYNFlag = False
        self.domains=[]
        self.coCaseFile = "None"
        self.reduce = False
        self.reductionFactor = 40.0

        
        #
        # init base class
        #
        Cfx2CoviseGuiBase.__init__(self, None)

        #
        # remove unused menubar items 
        #
        self.fileNewAction.setVisible(False)
        #
        # disable all buttons
        #
        self.gridFrame.setEnabled(False)
        self.reduceFrame.setEnabled(False)
        self.variableFrame.setEnabled(False)
        self.settingsFrame.setEnabled(False)
        self.startConversionFrame.setEnabled(False)
        self.outputDirFrame.setEnabled(False)
             
         
        #
        # register error log action
        #
        self.aErrorLogAction = ErrorLogAction()
        CoviseMsgLoop().register(self.aErrorLogAction)
        self.aErrorLogAction.register(self)
        
        # 
        # register action to read message and boundaries
        self.aBoundaryCollectorAction = BoundariesCollectorAction()
        CoviseMsgLoop().register(self.aBoundaryCollectorAction)
        
        
        #
        # register choice action
        #
        self.domainsGetterAction = ChoiceGetterAction()
        self.RegionsSelectionGetterAction = ChoiceGetterAction()        
        self.BoundarySelectionGetterAction = ChoiceGetterAction()        
        self.scalar_dataGetterAction = ChoiceGetterAction()
        self.vector_dataGetterAction = ChoiceGetterAction()
        self.timestepsGetterAction = IntGetterAction()
        #self.first_timestepGetterAction = ChoiceGetterAction()
        #self.readGridGetterAction = ChoiceGetterAction()
        #self.readRegionsGetterAction = ChoiceGetterAction()
        #self.readBoundariesGetterAction = ChoiceGetterAction()
        self.boundary_scalar_dataGetterAction = ChoiceGetterAction()
        self.boundary_vector_dataGetterAction = ChoiceGetterAction()
        #self.grid_is_time_dependentGetterAction = ChoiceGetterAction()
        #self.zone_with_time_dependent_gridGetterAction = ChoiceGetterAction()
        #self.rotAxisGetterAction = ChoiceGetterAction()
        #self.point_on_rotAxisGetterAction = ChoiceGetterAction()
        #self.rot_Angle_pre_timestepGetterAction = ChoiceGetterAction()
        #self.transform_velocityGetterAction = ChoiceGetterAction()
        #self.transform_directionGetterAction = ChoiceGetterAction()
        #self.rotation_axisGetterAction = ChoiceGetterAction()
        #self.zone_to_transform_velocityGetterAction = ChoiceGetterAction()
        #self.angular_velocityGetterAction = ChoiceGetterAction()
        #self.rotate_velocityGetterAction = ChoiceGetterAction()
        
        self.domainsGetterAction.register(self)
        self.RegionsSelectionGetterAction.register(self)
        self.BoundarySelectionGetterAction.register(self)
        self.scalar_dataGetterAction.register(self)
        self.vector_dataGetterAction.register(self)
        #self.timestepsGetterAction.register(self)
        #self.first_timestepGetterAction.register(self)
        #self.readGridGetterAction.register(self)
        #self.readRegionsGetterAction.register(self)
        #self.readBoundariesGetterAction.register(self)
        self.boundary_scalar_dataGetterAction.register(self)
        self.boundary_vector_dataGetterAction.register(self)
        #self.grid_is_time_dependentGetterAction.register(self)
        #self.zone_with_time_dependent_gridGetterAction.register(self)
        #self.rotAxisGetterAction.register(self)
        #self.point_on_rotAxisGetterAction.register(self)
        #self.rot_Angle_pre_timestepGetterAction.register(self)
        #self.transform_velocityGetterAction.register(self)
        #self.transform_directionGetterAction.register(self)
        #self.rotation_axisGetterAction.register(self)
        #self.zone_to_transform_velocityGetterAction.register(self)
        #self.angular_velocityGetterAction.register(self)
        #self.rotate_velocityGetterAction.register(self)
        
        global theNet
        theNet = net() 
        #
        # MODULE: ReadCfx
        #
        self.ReadCfx_1 = ReadCFX()
        theNet.add( self.ReadCfx_1 )
        #
        # hang in variable-getters
        #
        self.ReadCfx_1.addNotifier('domains',self.domainsGetterAction)
        self.ReadCfx_1.addNotifier('RegionsSelection',self.RegionsSelectionGetterAction)
        self.ReadCfx_1.addNotifier('BoundarySelection',self.BoundarySelectionGetterAction)
        self.ReadCfx_1.addNotifier('scalar_data',self.scalar_dataGetterAction)
        self.ReadCfx_1.addNotifier('vector_data',self.vector_dataGetterAction)
        self.ReadCfx_1.addNotifier('timesteps',self.timestepsGetterAction)
        #self.ReadCfx_1.addNotifier('first_timestep',self.first_timestepGetterAction)
        #self.ReadCfx_1.addNotifier('readGrid',self.readGridGetterAction)
        #self.ReadCfx_1.addNotifier('readRegions',self.readRegionsGetterAction)
        #self.ReadCfx_1.addNotifier('readBoundaries',self.readBoundariesGetterAction)
        self.ReadCfx_1.addNotifier('boundary_scalar_data',self.boundary_scalar_dataGetterAction)
        self.ReadCfx_1.addNotifier('boundary_vector_data',self.boundary_vector_dataGetterAction)
        #self.ReadCfx_1.addNotifier('grid_is_time_dependent',self.grid_is_time_dependentGetterAction)
        #self.ReadCfx_1.addNotifier('zone_with_time_dependent_grid',self.zone_with_time_dependent_gridGetterAction)
        #self.ReadCfx_1.addNotifier('rotAxis',self.rotAxisGetterAction)
        #self.ReadCfx_1.addNotifier('point_on_rotAxis',self.point_on_rotAxisGetterAction)
        #self.ReadCfx_1.addNotifier('rot_Angle_pre_timestep',self.rot_Angle_pre_timestepGetterAction)
        #self.ReadCfx_1.addNotifier('transform_velocity',self.transform_velocityGetterAction)
        #self.ReadCfx_1.addNotifier('transform_direction',self.transform_directionGetterAction)
        #self.ReadCfx_1.addNotifier('rotation_axis',self.rotation_axisGetterAction)
        #self.ReadCfx_1.addNotifier('zone_to_transform_velocity',self.zone_to_transform_velocityGetterAction)
        #self.ReadCfx_1.addNotifier('angular_velocity',self.angular_velocityGetterAction)
        #self.ReadCfx_1.addNotifier('rotate_velocity',self.rotate_velocityGetterAction)

        #
        # connect buttons
        #
        
        self.cbNoGrid.stateChanged.connect(self.setNoGrid)
        self.cbNoGrid.stateChanged.connect(self.setComposedGrid)
        self.cbNoGrid.stateChanged.connect(self.setTransientGrid)
        self.cbNoGrid.currentIndexChanged.connect(self.setNumVar)
        self.cbNoGrid.stateChanged.connect(self.setPdyn)
        self.cbNoGrid.stateChanged.connect(self.setNoBound)
        self.cbNoGrid.currentIndexChanged.connect(self.setDomain)
        self.cbNoGrid.returnPressed.connect(self.setOutputDir)
        self.cbNoGrid.clicked.connect(self.addToCoCase)
        self.cbNoGrid.clicked.connect(self.startConversion)


    def closeEvent(self, event):
        covise.clean()
        covise.quit()


    def addToCoCase(self ):
        fd = QtWidgets.QFileDialog(self)#, 'Open Cfx File',self.currentFilePath,'cfx file (*.plt *.dat)')
        fd.setMinimumWidth(1050)
        fd.setMinimumHeight(700)
        fd.setNameFilter('cocase file (*.cocase)')
        fd.setWindowTitle('Open coCase File')
        fd.setDirectory(self.currentFilePath)

        acceptedOrRejected = fd.exec_()
        if acceptedOrRejected != QtWidgets.QDialog.Accepted :
            return
        filenamesQt = fd.selectedFiles()
        if filenamesQt.isEmpty():
            return
        self.coCaseFile = str(filenamesQt[0])
        self.coCaseLabel.setText(self.coCaseFile)
        
        
    def setNoGrid(self, i):
        self.noGrid = self.cbNoGrid.isChecked()       
        
    def setComposedGrid(self, i):
        self.composedGrid = self.cbComposedGrid.isChecked()       
                
    def setTransientGrid(self, i):
        if self.cbTransientGrid.isChecked():
            self.ReadCfx_1.set_grid_is_time_dependent("True")
        else:
            self.ReadCfx_1.set_grid_is_time_dependent("False")
        
    def setNumVar(self, i):
        self.numVariables = i+1

    def setPdyn(self, i):
        self.calculatePDYNFlag = self.cbPdyn.isChecked()

    def setNoBound(self, i):
        # set the boundaries to all
        # choice param is "None", "All", "Affenfelsen",....
        self.processBoundaries = not self.cbNoBound.isChecked()

    def setDomain(self, i):
        if i!=0:
            self.fixdomain = str(self.comboDomain.currentText ())
        self.domains=self.domainsGetterAction.getChoices()

    def setOutputDir(self):
        self.outputFilePath=str(self.outputDirLineEdit.text())
        

    def customEvent(self,e):
        if e.type() == NOTIFIER_EVENT:
            self.variables=e.value[1:]
            
            if e.param == "domains":
                self.comboDomain.clear()
                for v in self.variables:
                    self.comboDomain.addItem(v)  
                self.domains=self.domainsGetterAction.getChoices()              
                        
            if e.param == "scalar_data":
                    # first is none
                numVar = len(self.variables)-1
                if self.comboNumVar.count()==0 or numVar >= self.comboNumVar.count():
                    self.comboNumVar.clear()
                    for i in range(numVar):
                        self.comboNumVar.addItem(str(i+1))
                    self.numVariables = self.comboNumVar.count()-1
                    self.comboNumVar.setCurrentIndex (self.numVariables)
            if e.param == "vector_data":
                    # first is none
                numVar = len(self.variables)-1
                if self.comboNumVar.count()==0 or numVar >= self.comboNumVar.count():
                    self.comboNumVar.clear()
                    for i in range(numVar):
                        self.comboNumVar.addItem(str(i+1))
                    self.numVariables = self.comboNumVar.count()-1
                    self.comboNumVar.setCurrentIndex (self.numVariables)
            if e.param == "boundary_scalar_data":
                    # first is none
                numVar = len(self.variables)-1
                if self.comboNumVar.count()==0 or numVar >= self.comboNumVar.count():
                    self.comboNumVar.clear()
                    for i in range(numVar):
                        self.comboNumVar.addItem(str(i+1))
                    self.numVariables = self.comboNumVar.count()-1
                    self.comboNumVar.setCurrentIndex (self.numVariables)
            if e.param == "boundary_vector_data":
            # first is none
                numVar = len(self.variables)-1
                if self.comboNumVar.count()==0 or numVar >= self.comboNumVar.count():
                    self.comboNumVar.clear()
                    for i in range(numVar):
                        self.comboNumVar.addItem(str(i+1))
                    self.numVariables = self.comboNumVar.count()-1
                    self.comboNumVar.setCurrentIndex (self.numVariables)
                   
  
        if e.type() == ERROR_EVENT:
            if self.outputFilePath:           
                text="ERROR: "+e.error+"\n"            
                self.statusText.append(text)  
                  
    def fileOpen(self):
    
    
        fd = QtWidgets.QFileDialog(self)#, 'Open Cfx File',self.currentFilePath,'cfx file (*.plt *.dat)')
        fd.setMinimumWidth(1050)
        fd.setMinimumHeight(700)
        fd.setNameFilter('cfx file (*.res)')
        fd.setWindowTitle('Open Cfx File')
        fd.setDirectory(self.currentFilePath)

        acceptedOrRejected = fd.exec_()
        if acceptedOrRejected != QtWidgets.QDialog.Accepted :
            return
        filenamesQt = fd.selectedFiles()
        if filenamesQt.isEmpty():
            return
        self.currentFilePath = os.path.dirname(str(filenamesQt[0]))
        self.fileName = (os.path.basename(str(filenamesQt[0])))[0:-4]
        fullCfxCaseName = str(filenamesQt[0])
        
        
        #
        # try to open file
        #
        if not os.access(fullCfxCaseName, os.R_OK):
            self.statusText.append("ERROR: Could not open file "+fullCfxCaseName+ " - not readable")  
        else:
            #self.setCaption(os.path.basename(fullCfxCaseName))

            #
            # set filename
            #
            self.ReadCfx_1.set_result( fullCfxCaseName )
            #self.ReadCFX_1.set_read_grid("false")
       
            #
            # disable file open
            #
            self.fileOpenAction.setEnabled(False) 
        
            #
            # set output File path
            #
            self.outputDirFrame.setEnabled(True)
            self.outputFilePath = self.currentFilePath + "/CoviseDaten"
            self.outputDirLineEdit.setText(self.outputFilePath)           
                #
                # enable format
                #
                
            #enable all frames
            self.gridFrame.setEnabled(True)
            self.reduceFrame.setEnabled(True)
            self.variableFrame.setEnabled(True)
            self.settingsFrame.setEnabled(True)
            self.startConversionFrame.setEnabled(True)
            self.outputDirFrame.setEnabled(True)  
        
        self.aBoundaryCollectorAction.waitForBoundaryInfoFinished()
        

       

    def fileExit(self):
        sys.exit()

    def calculatePDYN(self, caseFileItem):
        if not self.calculatePDYNFlag:
            return
        ptot_filename = ""
        pres_filename = ""
        for (varName, varFile, varDimension) in caseFileItem.variables_:
            if (varName == "PTOT") and (varDimension == SCALARVARIABLE):
                ptot_filename = varFile
            if (varName == "PRES") and (varDimension == SCALARVARIABLE):
                pres_filename = varFile
        if (ptot_filename != "") and (pres_filename != ""):
            pdyn_filename = ptot_filename.replace("-PTOT-", "-PDYN-")
            # calculate
            os.spawnlp(os.P_WAIT, "calcCovise", "calcCovise", pdyn_filename, "s1-s2", ptot_filename, pres_filename)
            # add
            caseFileItem.addVariableAndFilename("PDYN", pdyn_filename, SCALARVARIABLE)
            
    def convertBegin(self):
        self.cocase = CoviseCaseFile()
        if self.coCaseFile != "None":
            inputFile = open(self.coCaseFile, 'rb')
            self.cocase = pickle.load(inputFile)
            inputFile.close()
            self.cocasename = (os.path.basename(self.coCaseFile))[0:-7]
        else:
            self.cocasename=self.fileName
        logFileName = self.outputFilePath + '/' + self.cocasename + '.log'
        self.logFile = open(logFileName, 'w')
        self.logFile.write("Options:\n")
        self.logFile.write("Covise Case Name = %s\n"%(self.cocasename,))
        self.logFile.write("Covise Data Directory = %s\n"%(self.outputFilePath,))
        self.logFile.write("scale = %s\n"%(self.scale))
        self.logFile.write("\n")
        self.logFile.flush()



    def convertEnd(self):
        self.pickleFile = self.outputFilePath + '/' + self.cocasename + '.cocase'
        output = open(self.pickleFile, 'wb')
        pickle.dump(self.cocase,output)
        output.close()

        self.logFile.write("\ncocasefile = %s\n"%(self.pickleFile,))
        self.logFile.flush()

        self.logFile.write("\nConversion finished\n")
        self.logFile.flush()
        self.logFile.close()    


    def convert(self, domainname, domainchoice, processGrid, processBoundaries):
        # write logfile
        self.logFile.write("\nprocessing %s\n"%(domainname))
        self.logFile.flush()
        text = "\nprocessing %s"%(domainname)
        self.statusText.append(text)

        # reset wait variable    
        self.scalar_dataGetterAction.resetWait()
        self.vector_dataGetterAction.resetWait()
        self.boundary_scalar_dataGetterAction.resetWait()
        
        # select the domain
        self.ReadCfx_1.set_domains( domainchoice )

        # wait for choices to be updated by the module
        self.scalar_dataGetterAction.waitForChoices()
        self.vector_dataGetterAction.waitForChoices()
        self.boundary_scalar_dataGetterAction.waitForChoices()

        # read chocies
        scalarVariablesTuple = self.scalar_dataGetterAction.getChoices()
        vectorVariablesTuple = self.vector_dataGetterAction.getChoices()
        scalarVariables = {}
        vectorVariables = {}
        
        #first choice is none
        varChoice = 2
        for svar in scalarVariablesTuple:
            if  not 'boundary' in svar:
                scalarVariables[varChoice] = svar
            varChoice += 1
        varChoice = 2
        for vvar in vectorVariablesTuple:
            if not 'boundary' in vvar:
                vectorVariables[varChoice] = vvar
            varChoice += 1
       
        
        #get boundaries from message
        if processBoundaries:
            boundaries = self.aBoundaryCollectorAction.getBoundaryNames(domainname)
        else:
            boundaries = []


        #
        # CONVERT GRID
        #

        # connect grid
        gridPort = (self.ReadCfx_1, "GridOut0")
        if (self.scale != 1.0):
            theNet.connect( gridPort[0], gridPort[1], self.Transform_1, "geo_in" )
            gridPort = (self.Transform_1, "geo_out")
        if (self.mirror != 0):
            theNet.connect( gridPort[0], gridPort[1], self.Transform_2, "geo_in" )
            gridPort = (self.Transform_2, "geo_out")
        if (self.rotAngle != 0.0):
            theNet.connect( gridPort[0], gridPort[1], self.Transform_3, "geo_in" )
            gridPort = (self.Transform_3, "geo_out")
        theNet.connect( gridPort[0], gridPort[1], self.RWCovise_1, "mesh_in" )

        # set variables to None
        self.ReadCfx_1.set_scalar_data( 1 )
        self.ReadCfx_1.set_vector_data( 1 )    
        self.ReadCfx_1.set_boundary_scalar_data( 1 )

        covisename = domainname + "-3D.covise"         
        # create the cocase file item
        item3D = None
        if processGrid:
            self.ReadCfx_1.set_readGrid('True')
            item3D = CoviseCaseFileItem(domainname, GEOMETRY_3D, covisename)  
            # write logfile
            self.logFile.write("\n\tconverting grid %s ...\n"%(domainname))
            self.logFile.flush()
            text = "\n\tconverting grid "+ domainname + " ..."
            self.statusText.append( text )
            # clean the domainname
            if "/" in domainname:
                self.logFile.write("\t! Attention: Replacing the / in domainname = %s\n"%(domainname,))
                self.logFile.flush()
                text =  "\t! Attention: Replacing the / in domainname = "+ domainname
                self.statusText.append(text)
                domainname=domainname.replace("/","per")
            if "\x7f" in domainname:
                self.logFile.write("\t! Attention: Replacing a special character in domainname = %s\n"%(domainname,))
                self.logFile.flush()
                text =  "\t! Attention: Removing a special character in domainname = "+ domainname
                self.statusText.append(text)
                domainname=domainname.replace("\x7f","_")
            # create the RWCovise name
            self.RWCovise_1.set_grid_path( self.outputFilePath + '/' +covisename )
            # execute
            self.ReadCfx_1.execute()     
            theNet.finishedBarrier()
            # write logfile
            self.logFile.write("\t... conversion successful! File: %s\n"%(covisename,))
            self.logFile.flush()
            text =  "\t... conversion successful! File: "+covisename
            self.statusText.append(text)
        
            # disconnect grid
            theNet.disconnectAllFromModule(self.ReadCfx_1)
            theNet.disconnectAllFromModule(self.RWCovise_1)
            if (self.scale != 1.0):
                theNet.disconnectAllFromModule(self.Transform_1)
            if (self.mirror != 0):
                theNet.disconnectAllFromModule(self.Transform_2)
            if (self.rotAngle != 0.0):
                theNet.disconnectAllFromModule(self.Transform_3)

            self.ReadCfx_1.set_readGrid("False")

        else:
            self.ReadCfx_1.set_readGrid('False')
            item3D = CoviseCaseFileItem(domainname, GEOMETRY_3D, covisename)  
            for item in self.cocase.items_:
                text = "name "+item.name_
                self.statusText.append( text)
                text = "covisename "+domainname
                self.statusText.append( text)
                if item.name_ == domainname:
                    item3D = item
                    break             
            self.ReadCfx_1.execute()
            theNet.finishedBarrier()            

        timesteps = self.timestepsGetterAction.getInt()
        if timesteps == 0:
            timesteps = None
             

        #
        # CONVERT BOUNDARIES
        #

        # workaround for bug in readCFX, some parts are not convertes correctly,
        # but "all" parts are allways correct, therefore we use all parts
        # and extract the parts with GetSubset
        item2D={}
        if processBoundaries:
            self.ReadCfx_1.set_readBoundaries("True")
            # 
            # RWCovise
            #
            self.RWCovise_2.set_stepNo( 0 )
            self.RWCovise_2.set_rotate_output( "FALSE" )
            self.RWCovise_2.set_rotation_axis( 3 )
            self.RWCovise_2.set_rot_speed( 2.000000 )
    
           
            # connect boundaries
            outPort = (self.ReadCfx_1, "GridOut2")
            if timesteps==None:
                theNet.connect( outPort[0], outPort[1], self.GetSubset_1, "DataIn0" )      
                theNet.connect( self.GetSubset_1, "DataOut0", self.FixUsg_1, "GridIn0" )
                outPort = (self.FixUsg_1, "GridIn0")
            if self.reduce:
                theNet.connect( outPort[0], outPort[1], self.SimplifySurface_1, "meshIn" )
                outPort = (self.SimplifySurface_1, "meshOut")
            if (self.scale != 1.0):
                theNet.connect( outPort[0], outPort[1], self.Transform_1, "geo_in" )
                outPort = (self.Transform_1, "geo_out")
            if (self.mirror != 0):
                theNet.connect( outPort[0], outPort[1], self.Transform_2, "geo_in" )
                outPort = (self.Transform_2, "geo_out")
            if (self.rotAngle != 0.0):
                theNet.connect( outPort[0], outPort[1], self.Transform_3, "geo_in" )
                outPort = (self.Transform_3, "geo_out")
            theNet.connect( outPort[0], outPort[1], self.RWCovise_2, "mesh_in" )

            subset=0
            boundchoice=1
            for boundname in boundaries:
                boundchoice = int(boundaries[boundname])
                self.ReadCfx_1.set_BoundarySelection(str(boundchoice))
                # ommit names "None" "all"
                #boundchoice+=1            
                if int(boundchoice) > 0 :
                    # write logfile
                    self.logFile.write("\n\tconverting surface %s ...\n"%(boundname,))
                    self.logFile.flush()
                    text =  "\n\tconverting surface %s ..."%(boundname,)
                    self.statusText.append( text)
                    bname=boundname
                    # clean boundname
                    if "/" in boundname:
                        self.logFile.write("\t! Attention: Replacing the / in boundname = %s\n"%(boundname,))
                        self.logFile.flush()
                        text =  "\t! Attention: Replacing the / in boundname = "+ boundname
                        self.statusText.append( text)
                        bname=boundname.replace("/","per")
                    if "\x7f" in boundname:
                        self.logFile.write("\t! Attention: Replacing a special character in boundname = %s\n"%(boundname,))
                        self.logFile.flush()
                        text =  "\t! Attention: Replacing a special character in boundname = ", boundname
                        self.statusText.append( text)
                        bname=boundname.replace("\x7f","_")
                    # set the subset
                    self.GetSubset_1.set_selection( str(0) )
                    # create RWCovise name
                    bname = bname.replace(' ','_')
                    covisename = domainname + "-boundary-" + bname + "-2D.covise"
                    self.RWCovise_2.set_grid_path( self.outputFilePath + '/' +covisename )
                    # execute
                    self.ReadCfx_1.execute()
                    #if timesteps==None:
                    #    GetSubset_1.execute()
                    #else :
                    #    GetSetelem_1.set_stepNo(1)
                    #    GetSetelem_1.execute()    
                    theNet.finishedBarrier()
                    # write logfile
                    self.logFile.write("\t... conversion successful! Filen: %s\n"%(covisename,))
                    self.logFile.flush()
                    text =  "\t... conversion successful! File: "+covisename
                    self.statusText.append( text)
                    # create cocase item
                    item2D[boundname] = CoviseCaseFileItem(domainname + "-" + bname, GEOMETRY_2D, covisename)
                    subset+=1
            
            # disconnect boundaries
            theNet.disconnectAllFromModule(self.ReadCfx_1)
            theNet.disconnectAllFromModule(self.RWCovise_2)
            if self.reduce:
                theNet.disconnectAllFromModule(self.SimplifySurface_1)
            if timesteps==None:
                theNet.disconnectAllFromModule(self.GetSubset_1)
                theNet.disconnectAllFromModule(self.FixUsg_1)
            if (self.scale != 1.0):
                theNet.disconnectAllFromModule(self.Transform_1)
            if (self.mirror != 0):
                theNet.disconnectAllFromModule(self.Transform_2)
            if (self.rotAngle != 0.0):
                theNet.disconnectAllFromModule(self.Transform_3)
    
            #theNet.remove( RWCovise_2 )
            #theNet.remove( GetSubset_1 )
            #theNet.remove( FixUsg_1 )
    
            #if not timesteps==None:
            #    theNet.remove( GetSetelem_1 )
            #    theNet.remove( self.PipelineCollect_1 )
            

        #
        # CONVERT SCALAR VARIABLES
        #

        # connect the 3D scalar data
        theNet.connect( self.ReadCfx_1, "DataOut0", self.RWCovise_1, "mesh_in" )


       # loop through the scalar variables
        # select the variable, ommit variable "none"
        countVar = 0
        for varchoice in scalarVariables:
            svar = scalarVariables[varchoice]
            #bsvar = boundScalarVariables[varchoice]
            # convert only the first numVariables variables
            countVar += 1
            #if ( fixresult=="None" and int(varchoice) < (int(self.numVariables)+2) ) or svar==fixresult :
            if countVar <= self.numVariables :
        
                #
                # CONVERT SCALAR VARIABLES OF GRID
                #

                self.logFile.write("\n\tconverting scalar variable %s on grid %s...\n"%(svar,domainname))
                self.logFile.flush()
                text = "\n\tconverting scalar variable "+ svar + " on grid " + domainname + "..."
                self.statusText.append( text)
                # clean variablename
                if "/" in svar:
                    self.logFile.write("\t! Attention: Replacing the / in svar = %s\n"%(svar,))
                    self.logFile.flush()
                    text = "\t! Attention: Replacing the / in svar = "+ svar
                    self.statusText.append( text)
                    svar=svar.replace("/","per")
                if "\x7f" in svar:
                    self.logFile.write("\t! Attention: Replacing a special character in svar = %s\n"%(svar,))
                    self.logFile.flush()
                    text = "\t! Attention: Replacing a special character in svar = "+ svar
                    self.statusText.append( text)
                    svar=svar.replace("\x7f","_")
                # select variable
                self.ReadCfx_1.set_scalar_data( varchoice )
                self.ReadCfx_1.set_boundary_scalar_data( varchoice )
                self.ReadCfx_1.set_readGrid("false")
                covisename = domainname + "-" + svar + "-3D.covise"
                #create RWCovise name
                self.RWCovise_1.set_grid_path( self.outputFilePath + '/' +covisename )
                # execute
                self.ReadCfx_1.execute()
                theNet.finishedBarrier()
                # write logile
                self.logFile.write("\t... conversion successful! Filename: %s\n"%(covisename,))
                self.logFile.flush()
                text = "\t... conversion successful! Filename: "+covisename
                self.statusText.append( text)
                # add variable to cocase item
                item3D.addVariableAndFilename(svar, covisename, SCALARVARIABLE)
            

                #
                # CONVERT SCALAR VARIABLES OF BOUNDARIES
                #

                if processBoundaries:
                    # 
                    # RWCovise
                    #
                    self.RWCovise_2.set_stepNo( 0 )
                    self.RWCovise_2.set_rotate_output( "FALSE" )
                    self.RWCovise_2.set_rotation_axis( 3 )
                    self.RWCovise_2.set_rot_speed( 2.000000 )

                    
                    # connect the boundary scalar data
                    gridPort = (self.ReadCfx_1, "GridOut2")
                    dataPort = (self.ReadCfx_1, "DataOut3")
                    if timesteps==None:
                        theNet.connect( gridPort[0], gridPort[1], self.GetSubset_1, "DataIn0" )      
                        theNet.connect( dataPort[0], dataPort[1], self.GetSubset_1, "DataIn1" )
                        theNet.connect( self.GetSubset_1, "DataOut0", self.FixUsg_1, "GridIn0" )      
                        theNet.connect( self.GetSubset_1, "DataOut1", self.FixUsg_1, "DataIn0" )  
                        gridPort = (self.FixUsg_1, "GridOut0")
                        dataPort = (self.FixUsg_1, "DataOut0")
                    if self.reduce:
                        theNet.connect( gridPort[0], gridPort[1], self.SimplifySurface_1, "meshIn")
                        theNet.connect( dataPort[0], dataPort[1], self.SimplifySurface_1, "dataIn_0")
                        gridPort = (self.SimplifySurface_1, "meshOut")
                        dataPort = (self.SimplifySurface_1, "dataOut_0")
                    theNet.connect( dataPort[0], dataPort[1], self.RWCovise_2, "mesh_in" )
                       
   
                    subset=0
                    boundchoice=1
                    for boundname in boundaries:
                        boundchoice = int(boundaries[boundname])
                        self.ReadCfx_1.set_BoundarySelection(str(boundchoice))
                        # ommit names "None" "all"
                        #boundchoice+=1            
                        if int(boundchoice) > 0 :
                            self.logFile.write("\n\tconverting scalar variable %s on surface %s %d...\n"%(svar,boundname, boundchoice))
                            self.logFile.flush()
                            text =  "\n\tconverting scalar variable "+ svar + " on surface " + boundname + "..."
                            self.statusText.append( text)
                            bname=boundname
                            # clean boundname
                            if "/" in boundname:
                                self.logFile.write("\t! Attention: Replacing the / in boundname = %s\n"%(boundname,))
                                self.logFile.flush()
                                text =  "\t! Attention: Replacing the / in boundname = "+ boundname
                                self.statusText.append( text)
                                bname=boundname.replace("/","per")
                            if "\x7f" in boundname:
                                self.logFile.write("\t! Attention: Replacing a special character in boundname = %s\n"%(boundname,))
                                self.logFile.flush()
                                text = "\t! Attention: Replacing a special character in boundname = "+ boundname
                                self.statusText.append( text)
                                bname=boundname.replace("\x7f","_")
                            # clean variablename
                            if "/" in svar:
                                self.logFile.write("\t! Attention: Replacing the / in svar = %s\n"%(svar,))
                                self.logFile.flush()
                                text = "\t! Attention: Replacing the / in svar = "+ svar
                                self.statusText.append( text)
                                svar=svar.replace("/","per")
                            if "\x7f" in svar:
                                self.logFile.write("\t! Attention: Replacing a special character in svar = %s\n"%(svar,))
                                self.logFile.flush()
                                text = "\t! Attention: Replacing a special character in svar = "+ svar
                                self.statusText.append( text)
                                svar=svar.replace("\x7f","_")
                                text = "\t! new svar = ", svar
                                self.statusText.append( text)
                            # set the subset
                            self.GetSubset_1.set_selection( str(0) )
                            # create RWCovise name
                            bname = bname.replace(' ', '_')
                            svar = svar.replace(' ', '_')
                            covisename = domainname + "-boundary-" + bname + "-" + svar + "-2D.covise"
                            self.RWCovise_2.set_grid_path( self.outputFilePath + '/' +covisename )                         
                            # execute  
                            self.ReadCfx_1.execute()
                            #if timesteps==None:
                            #    self.ReadCfx_1.execute()
                            #else :
                            #    self.GetSetelem_1.set_stepNo(1)
                            #    self.GetSetelem_1.execute()                    
                            theNet.finishedBarrier()
                            # write logfile
                            self.logFile.write("\t... conversion successful: Filename %s\n"%(covisename,))
                            self.logFile.flush()
                            text =  "\t... conversion successful: "+covisename
                            self.statusText.append( text)
                            # append variable to cocase item
                            if boundname in item2D:
                                item2D[boundname].addVariableAndFilename(svar, covisename, SCALARVARIABLE)
                            subset+=1
                                           

                    # disconnect the boundaries
                    gridPort = (self.ReadCfx_1, "GridOut2")
                    dataPort = (self.ReadCfx_1, "DataOut3")
                    if timesteps==None:
                        theNet.disconnect( gridPort[0], gridPort[1], self.GetSubset_1, "DataIn0" )      
                        theNet.disconnect( dataPort[0], dataPort[1], self.GetSubset_1, "DataIn1" )
                        theNet.disconnect( self.GetSubset_1, "DataOut0", self.FixUsg_1, "GridIn0" )      
                        theNet.disconnect( self.GetSubset_1, "DataOut1", self.FixUsg_1, "DataIn0" )  
                        gridPort = (self.FixUsg_1, "GridOut0")
                        dataPort = (self.FixUsg_1, "DataOut0")
                    if self.reduce:
                        theNet.disconnect( gridPort[0], gridPort[1], self.SimplifySurface_1, "meshIn")
                        theNet.disconnect( dataPort[0], dataPort[1], self.SimplifySurface_1, "dataIn_0")
                        gridPort = (self.SimplifySurface_1, "meshOut")
                        dataPort = (self.SimplifySurface_1, "dataOut_0")
                    theNet.disconnect( dataPort[0], dataPort[1], self.RWCovise_2, "mesh_in" )
                          
                        
                    #theNet.remove( RWCovise_2 )
                    #theNet.remove( GetSubset_1 )            
                    #theNet.remove( FixUsg_1 )            
                    #if not timesteps==None:
                    #    theNet.remove( GetSetelem_1 )
                    #    theNet.remove( self.PipelineCollect_1 ) 
    
                
        # disconnect the grid
        theNet.disconnect( self.ReadCfx_1, "DataOut0", self.RWCovise_1, "mesh_in" )            



        #
        # convert the vector variables
        #

        # read no boundaries
        self.ReadCfx_1.set_readBoundaries( "False" )  

        if (self.scale == 1.0) and (self.mirror == 0) and (self.rotAngle == 0.0):
            self.ReadCfx_1.set_readGrid("false")
        else:
            self.ReadCfx_1.set_readGrid("true")

        # connect the modules
        gridPort = (self.ReadCfx_1, "GridOut0")
        dataPort = (self.ReadCfx_1, "DataOut1")
        if (self.scale != 1.0):
            theNet.connect( gridPort[0], gridPort[1], self.Transform_1, "geo_in" )
            theNet.connect( dataPort[0], dataPort[1], self.Transform_1, "data_in0" )
            gridPort = (self.Transform_1, "geo_out")
            dataPort = (self.Transform_1, "data_out0")
        if (self.mirror != 0):
            theNet.connect( gridPort[0], gridPort[1], self.Transform_2, "geo_in" )
            theNet.connect( dataPort[0], dataPort[1], self.Transform_2, "data_in0" )
            gridPort = (self.Transform_2, "geo_out")
            dataPort = (self.Transform_2, "data_out0")
        if (self.rotAngle != 0.0):
            theNet.connect( gridPort[0], gridPort[1], self.Transform_3, "geo_in" )
            theNet.connect( dataPort[0], dataPort[1], self.Transform_3, "data_in0" )
            gridPort = (self.Transform_3, "geo_out")
            dataPort = (self.Transform_3, "data_out0")
        theNet.connect( dataPort[0], dataPort[1], self.RWCovise_1, "mesh_in" )

        # loop though the variables
        # ommit variable "none"
        countVar = 0
        for varchoice in vectorVariables:
            vvar = vectorVariables[varchoice]
            # select only the first numVariables variables
            #if ( fixresult=="None" and int(varchoice) < (int(numVariables)+2) ) or vvar==fixresult :
            countVar +=1
            if countVar <= self.numVariables :
                self.logFile.write("\n\tconverting vector variable %s ...\n"%(vvar,))
                self.logFile.flush()
                text = "\n\tconverting vector variable "+ vvar + " ..."
                self.statusText.append( text)
                # clean variablename
                if "/" in vvar:
                    self.logFile.write("\t! Attention: Replacing the / in vvar = %s\n"%(vvar,))
                    self.logFile.flush()
                    text =  "\t! Attention: Replacing the / in vvar = "+ vvar
                    self.statusText.append( text)
                    vvar=vvar.replace("/","per")
                if "\x7f" in vvar:
                    self.logFile.write("\t! Attention: Replacing a special character in vvar = %s\n"%(vvar,))
                    self.logFile.flush()
                    text = "\t! Attention: Replacing a special character in vvar = "+ vvar
                    self.statusText.append( text)
                    vvar=vvar.replace("\x7f","_")
                    text = "\t! new vvar = ", vvar
                    self.statusText.append( text)
                # select variable
                self.ReadCfx_1.set_vector_data( varchoice )
                # create RWCovise name
                covisename = domainname + "-" + vvar + "-3D.covise"
                self.RWCovise_1.set_grid_path(self.outputFilePath + '/' + covisename )
                # execute
                self.ReadCfx_1.execute()
                theNet.finishedBarrier()
                # write logfile
                self.logFile.write("\t... conversion successful! Filename: %s\n"%(covisename,))
                self.logFile.flush()
                text = "\t... conversion successful! Filename: "+covisename
                self.statusText.append( text)
                # append variable to cocase item
                item3D.addVariableAndFilename(vvar, covisename, VECTOR3DVARIABLE)
        # disconnect the modules
        theNet.disconnectAllFromModule(self.ReadCfx_1)
        theNet.disconnectAllFromModule(self.RWCovise_1)
        if (self.scale != 1.0):
            theNet.disconnectAllFromModule(self.Transform_1)
        if (self.mirror != 0):
            theNet.disconnectAllFromModule(self.Transform_2)
        if (self.rotAngle != 0.0):
            theNet.disconnectAllFromModule(self.Transform_3)

        #
        # add Grid to coCase
        #
        if processGrid:
            # add to cocase
            self.cocase.add(item3D)
        # add the boundary item do the case file
        if processBoundaries:
            for boundname in boundaries:
                # ommit names "None" "all"
                boundchoice = int(boundaries[boundname])            
                if int(boundchoice) > 0 :
                    self.calculatePDYN(item2D[boundname])
                    self.cocase.add(item2D[boundname])
            


    def startModules(self):
        global theNet
        
        
        # Module Transform (scaling)
        #
        if (self.scale != 1.0):
            self.Transform_1 = Transform()
            theNet.add( self.Transform_1 )
            self.Transform_1.set_Transform( 5 )
            self.Transform_1.set_scaling_factor( self.scale )
            self.Transform_1.set_createSet( "FALSE" ) 

        # Module Transform (mirror)
        #
        if (self.mirror != 0):
            self.Transform_2 = Transform()
            theNet.add( self.Transform_2 )
            self.Transform_2.set_Transform( 2 )
            if (self.mirror == 1):
                self.Transform_2.set_normal_of_mirror_plane(1.0, 0.0, 0.0)
            elif (self.mirror == 2):
                self.Transform_2.set_normal_of_mirror_plane(0.0, 1.0, 0.0)
            elif (self.mirror == 3):
                self.Transform_2.set_normal_of_mirror_plane(0.0, 0.0, 1.0)
            self.Transform_2.set_MirroredAndOriginal( "FALSE" )
            self.Transform_2.set_createSet( "FALSE" )

        # Module Transform (rotation)
        #
        if (self.rotAngle != 0.0):
            self.Transform_3 = Transform()
            theNet.add( self.Transform_3 )
            self.Transform_3.set_Transform( 4 )
            self.Transform_3.set_axis_of_rotation(self.rotAxisX, self.rotAxisY, self.rotAxisZ)
            self.Transform_3.set_angle_of_rotation( self.rotAngle )
            self.Transform_3.set_createSet( "FALSE" ) 


        # Module SimplifySurface
        #
        if self.reduce:
            self.SimplifySurface_1 = SimplifySurface()
            theNet.add( self.SimplifySurface_1 )
            self.SimplifySurface_1.set_ignore_data( "TRUE" )
            self.SimplifySurface_1.set_percent( self.reductionFactor )

        #
        # MODULE: RWCovise
        #
        self.RWCovise_1 = RWCovise()
        theNet.add( self.RWCovise_1 )
        self.RWCovise_1.set_stepNo( 0 )
        self.RWCovise_1.set_rotate_output( "FALSE" )
        self.RWCovise_1.set_rotation_axis( 3 )
        self.RWCovise_1.set_rot_speed( 2.000000 )
        
        self.RWCovise_2 = RWCovise()
        theNet.add( self.RWCovise_2 )
        
        #
        # Module GetSubset
        #
        self.GetSubset_1 = GetSubset()
        theNet.add( self.GetSubset_1 )      
          
        #
        # Module FixUsg
        #                 
        self.FixUsg_1 = FixUsg()
        theNet.add( self.FixUsg_1 )      


#        self.GetSetelem_1 = GetSetelem()
#        theNet.add( self.GetSetelem_1 )
#        self.PipelineCollect_1 = PipelineCollect()
#        theNet.add( self.PipelineCollect_1 )


    def startConversion(self):
       
        global theNet
        
        if os.path.isdir(self.outputFilePath):
            pass
        else:
            try:
                os.mkdir(self.outputFilePath)
            except(OSError):
                self.statusText.append("ERROR: Could not create directory "+str(self.outputFilePath)+" check permissions and enter again or select another directory")
                return


        # reduction
        self.reduce = self.reduceCheckBox.isChecked()
        self.reductionFactor=float(str(self.leReductionFactor.text()))
        # scale
        self.scale=float(str(self.leScaleX.text()))
        # mirror
        self.mirror=self.comboMirror.currentIndex()
        # rotation
        self.rotAxisX=float(str(self.leRotAxisX.text()))
        self.rotAxisY=float(str(self.leRotAxisY.text()))
        self.rotAxisZ=float(str(self.leRotAxisZ.text()))
        self.rotAngle=float(str(self.leRotAngle.text()))


        self.statusText.append("Starting conversion...")
        self.startModules()
        self.convertBegin()

        #
        # loop through the domains parts
        # ('all', 'Domain\x7f1', '')
        #
        domainchoice=1
        #
        # read the domain values
        #
        for domainname in self.domains:
            domainchoice+=1
            if self.fixdomain=="None" or self.fixdomain==domainname:            
                self.convert(domainname, domainchoice, (not self.noGrid), self.processBoundaries)

        #
        # create composed grid
        #
        if (not self.noGrid) and (self.composedGrid):
            self.convert("all", 1, True, False)



        self.convertEnd()
        self.repaint()
        

        
        text="\n\n...conversion finished!\n"
        self.statusText.append(text)

        #theNet.save("cfx2covise.net")

        #pdM.unSpawnPatienceDialog()
