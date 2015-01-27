
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject, pyqtSignal

from Tecplot2CoviseGuiBase import Ui_Tecplot2CoviseGuiBase
class Tecplot2CoviseGuiBase(QtWidgets.QMainWindow, Ui_Tecplot2CoviseGuiBase):
    def __init__(self, parent=None, f=QtCore.Qt.WindowFlags()):
        QtWidgets.QMainWindow.__init__(self, parent, f)
        self.setupUi(self)

import sys
import os
import os.path
import time
import re
import covise

from ErrorLogAction import ErrorLogAction, ERROR_EVENT
from ChoiceGetterAction import ChoiceGetterAction, NOTIFIER_EVENT
from CoviseMsgLoop import CoviseMsgLoop, CoviseMsgLoopAction
from coPyModules import ReadTecplot, RWCovise, Transform, GridSurface, Calc, GetSubset
from PatienceDialogManager import PatienceDialogManager
from coviseModuleBase import net

from Basic2DGridBase import Ui_Basic2DGridBase
class Basic2DGridBase(QtWidgets.QWidget, Ui_Basic2DGridBase):
    def __init__(self, parent=None, f=QtCore.Qt.WindowFlags()):
        QtWidgets.QWidget.__init__(self, parent, f)
        self.setupUi(self)

from Basic3DGridBase import Ui_Basic3DGridBase
class Basic3DGridBase(QtWidgets.QWidget, Ui_Basic3DGridBase):
    def __init__(self, parent=None, f=QtCore.Qt.WindowFlags()):
        QtWidgets.QWidget.__init__(self, parent, f)
        self.setupUi(self)

from BottomBase import Ui_BottomBase
class BottomBase(QtWidgets.QWidget, Ui_BottomBase):
    def __init__(self, parent=None, f=QtCore.Qt.WindowFlags()):
        QtWidgets.QWidget.__init__(self, parent, f)
        self.setupUi(self)

from WatersurfaceBase import Ui_WatersurfaceBase
class WatersurfaceBase(QtWidgets.QWidget, Ui_WatersurfaceBase):
    def __init__(self, parent=None, f=QtCore.Qt.WindowFlags()):
        QtWidgets.QWidget.__init__(self, parent, f)
        self.setupUi(self)
        
from Vector2DVariableBase import Ui_Vector2DVariableBase
class Vector2DVariableBase(QtWidgets.QWidget, Ui_Vector2DVariableBase):
    def __init__(self, parent=None, f=QtCore.Qt.WindowFlags()):
        QtWidgets.QWidget.__init__(self, parent, f)
        self.setupUi(self)
        
from Vector3DVariableBase import Ui_Vector3DVariableBase
class Vector3DVariableBase(QtWidgets.QWidget, Ui_Vector3DVariableBase):
    def __init__(self, parent=None, f=QtCore.Qt.WindowFlags()):
        QtWidgets.QWidget.__init__(self, parent, f)
        self.setupUi(self)
        
from ScalarVariableBase import Ui_ScalarVariableBase
class ScalarVariableBase(QtWidgets.QWidget, Ui_ScalarVariableBase):
    def __init__(self, parent=None, f=QtCore.Qt.WindowFlags()):
        QtWidgets.QWidget.__init__(self, parent, f)
        self.setupUi(self)

class ZonesCollectorAction(CoviseMsgLoopAction):

    """Action to collect part information when using
    covise with an ReadEnsight.

    Gets the information from covise-info-messages.

    """

    def __init__ (self):
        CoviseMsgLoopAction.__init__(
            self,
            self.__class__.__module__ + '.' + self.__class__.__name__,
            56, # Magic number 56 is covise-msg type INFO
            'Collect number of zones and zone names from covise-info-messages.')

        # start with true and set to false in first message
        self.__zonesFinished = True
        self.__nzones = 0
        self.__zones = {}


    def run(self, param):
        assert 4 == len(param)
        # assert param[0] is a modulename
        # assert param[1] is a number
        # assert param[2] is an ip
        # assert param[3] is a string

#       print(str(self.run))

        msgText = param[3]
        #print("\nZonesCollectorAction.run for msgText=", str(msgText))

        #print("msgText=", str(msgText))

        # test for Zones Info Finished string
        if msgText == 'Finished zones info':
           self.__zonesFinished = True
           return

        # test for number of zones string
        #print("testing for Number of Zones")
        matchie = re.match(r'Number of zones (\d+)', msgText)
        if (matchie):
           self.__zonesFinished = False
           self.__nzones = int(matchie.group(1).strip())
           print("Number of zones: ", self.__nzones)

        #print("testing for Zone"      )
        # test for zone name string
        matchie = re.match(r'Zone (\d+) (\S+)', str(msgText))
        if (matchie):
           idx = int(matchie.group(1).strip())
           name = matchie.group(2).strip()
           self.__zones[idx] = name
           print("Zone idx=", idx, " name=", name)

#        else:
#            infoer.function = str(self.run)
#            infoer.write(
#                'Covise-message "%s" doesn\'t look like a zones-description.'
#                % str(msgText))



    def getZoneNames(self):
        return self.__zones

    def getNumZones(self):
        return self.__nzones

    def waitForZonesInfoFinished(self):
        #print("ZonesCollectorAction.waitForPartsinfoFinished")
        while not self.__zonesFinished:
            #print("wait" )
            pass
            
            
class Tecplot2CoviseGui(QtWidgets.QMainWindow, Ui_Tecplot2CoviseGuiBase):

    def __init__(self, parent=None):
        
        InitialDatasetSearchPath = covise.getCoConfigEntry("vr-prepare.InitialDatasetSearchPath")
        if not InitialDatasetSearchPath:
            InitialDatasetSearchPath = os.getcwd()
        self.currentFilePath = InitialDatasetSearchPath

        self.ReadTecplot_1 = 'None'
        self.format='None'
        self.dimension=2
        self.waterSurfaceOffset=0.0
        self.scale=10.0
        
         
        # as long as the module didn't update it's parameters they are false
        self.grid2DXDone='False'
        self.grid2DYDone='False'
        self.grid2DZ0Done='False'
        self.grid2DZ1Done='False'
        self.vec2DXDone='False'
        self.vec2DYDone='False'
        self.scalar0Done='False'
        self.scalar1Done='False'
        self.scalar22DDone='False'
        
        self.grid3DXDone='False'
        self.grid3DYDone='False'
        self.grid3DZDone='False'
        
        self.vec2DXDone='False'
        self.vec2DYDone='False'

        self.vec3DXDone='False'
        self.vec3DYDone='False'
        self.vec3DZDone='False'
        self.scalar23DDone='False'
        
        # false as long as the user didn't select from the combo boxes
        self.grid2DXSelected='False'
        self.grid2DYSelected='False'
        self.grid2DZ0Selected='False'
        self.grid2DZ1Selected='False'
        self.vel2DUSelected='False'
        self.vel2DVSelected='False'
        self.scalar2DSelected='False'

        self.grid3DXSelected='False'
        self.grid3DYSelected='False'
        self.grid3DZSelected='False'
        self.vel3DUSelected='False'
        self.vel3DVSelected='False'
        self.vel3DWSelected='False'
        self.scalar3DSelected='False'
         
        # None as long as the selections are not completed
        self.bottomGridFileName = None
        self.watersurfaceGridFileName = None
        self.bottomVariableFileName = None
        self.watersurfaceVariableFileName = None        
        self.vector2DVariableFileName = None
        self.scalar2DVariableFileName = None

        self.grid3DFileName = None
        self.vector3DVariableFileName = None
        self.scalar3DVariableFileName = None
        
        #
        # init base class
        #
        
        super(Tecplot2CoviseGui, self).__init__(parent)
        self.setupUi(self)
        
        self.scaleZLineEdit.setText(str(self.scale))
        #
        # remove unused menubar items 
        #
        self.fileNewAction.setVisible(False)
        """
        self.fileMenu.setItemVisible(self.fileMenu.idAt(0), False )            
        self.fileMenu.setItemVisible(self.fileMenu.idAt(2), False )            
        self.fileMenu.setItemVisible(self.fileMenu.idAt(3), False )            
        self.fileMenu.setItemVisible(self.fileMenu.idAt(4), False )            
        self.fileMenu.setItemVisible(self.fileMenu.idAt(5), False )            
        self.fileMenu.setItemVisible(self.fileMenu.idAt(6), False )
        """
        #
        # disable all buttons
        #
        self.formatFrame.setEnabled(False)
        self.outputDirFrame.setEnabled(False)
        self.settingsFrame.setEnabled(False)
        self.startConversionFrame.setEnabled(False)

        #
        # designer could not assign a layout to empty widgets
        #
        self.dummyFrameLayout = QtWidgets.QVBoxLayout()
        self.dummyFrame.setLayout(self.dummyFrameLayout)

        #
        # add scroll view
        #
        self.scrollView = QtWidgets.QScrollArea()#self.dummyFrame)   
        self.scrollView.setFrameShadow(QtWidgets.QFrame.Raised)
        self.scrollView.setFrameShape(QtWidgets.QFrame.StyledPanel)
        
                       
        #
        # add a box for the widgets in the scrollview. 
        # If the widgets are direct childs of the scrollview, it doesn't scroll
        #
        self.box = QtWidgets.QWidget()
        self.vboxLayout = QtWidgets.QVBoxLayout()
        self.box.setLayout(self.vboxLayout)
                

        #
        # add grid widget
        #
        self.grid2DWidget = Basic2DGridBase()     
        self.grid2DWidget.setEnabled(False)
        self.grid2DWidget.show()
        self.vboxLayout.addWidget(self.grid2DWidget)

        self.grid3DWidget = Basic3DGridBase()    
        self.grid3DWidget.setEnabled(False)
        self.grid3DWidget.hide()
        self.vboxLayout.addWidget(self.grid3DWidget)

        self.bottomWidget = BottomBase()          
        self.bottomWidget.setEnabled(False)
        self.bottomWidget.show()
        self.vboxLayout.addWidget(self.bottomWidget)

        self.watersurfaceWidget = WatersurfaceBase()      
        self.watersurfaceWidget.setEnabled(False)
        self.watersurfaceWidget.watersurfaceOffsetLineEdit.setText(str(self.waterSurfaceOffset))
        self.watersurfaceWidget.show()
        self.vboxLayout.addWidget(self.watersurfaceWidget)

        #
        # add vector widget
        #
        self.velocity2DWidget = Vector2DVariableBase()  
        self.velocity2DWidget.setEnabled(False)
        self.velocity2DWidget.show()
        self.vboxLayout.addWidget(self.velocity2DWidget)

        self.velocity3DWidget = Vector3DVariableBase()  
        self.velocity3DWidget.setEnabled(False)
        self.velocity3DWidget.hide()
        self.vboxLayout.addWidget(self.velocity3DWidget)

        #
        # scalar widgets (modue has 3 scalar parameters, 2 are already used for bottom and watersurface
        #
        self.scalar2DWidget = ScalarVariableBase()    
        self.scalar2DWidget.setEnabled(False)
        self.scalar2DWidget.show()
        self.vboxLayout.addWidget(self.scalar2DWidget)
 
        #
        # scalar3D widget
        #
        self.scalar3DWidget = ScalarVariableBase() 
        self.scalar3DWidget.setEnabled(False)
        self.scalar3DWidget.hide()
        self.vboxLayout.addWidget(self.scalar3DWidget)
         
        self.scrollView.setWidget(self.box)
        self.dummyFrameLayout.addWidget(self.scrollView)
               
         
        #
        # register error log action
        #
        self.aErrorLogAction = ErrorLogAction()
        CoviseMsgLoop().register(self.aErrorLogAction)
        self.aErrorLogAction.register(self)
        #
        # register choice action
        #
        self.gridXGetterAction = ChoiceGetterAction()
        self.gridYGetterAction = ChoiceGetterAction()
        self.gridZ0GetterAction = ChoiceGetterAction()
        self.gridZ1GetterAction = ChoiceGetterAction()
        self.vecXGetterAction = ChoiceGetterAction()
        self.vecYGetterAction = ChoiceGetterAction()
        self.vecZGetterAction = ChoiceGetterAction()
        self.bottomGetterAction = ChoiceGetterAction()
        self.watersurfaceGetterAction = ChoiceGetterAction()
        self.scalarGetterAction = ChoiceGetterAction()
        
        self.gridXGetterAction.register(self)
        self.gridYGetterAction.register(self)
        self.gridZ0GetterAction.register(self)
        self.gridZ1GetterAction.register(self)
        self.vecXGetterAction.register(self)
        self.vecYGetterAction.register(self)
        self.vecZGetterAction.register(self)
        self.bottomGetterAction.register(self)
        self.watersurfaceGetterAction.register(self)
        self.scalarGetterAction.register(self)
        
        
        self.aZonesCollectorAction = ZonesCollectorAction()
        CoviseMsgLoop().register(self.aZonesCollectorAction)
        
        global theNet
        theNet = net() 
        #
        # MODULE: ReadTecplot
        #
        self.ReadTecplot_1 = ReadTecplot()
        theNet.add( self.ReadTecplot_1 )
        #
        # hang in variable-getters
        #
        self.ReadTecplot_1.addNotifier('grid_x', self.gridXGetterAction)
        self.ReadTecplot_1.addNotifier('grid_y', self.gridYGetterAction)
        self.ReadTecplot_1.addNotifier('grid_z0', self.gridZ0GetterAction)
        self.ReadTecplot_1.addNotifier('grid_z1', self.gridZ1GetterAction)
        self.ReadTecplot_1.addNotifier('vec_x', self.vecXGetterAction)
        self.ReadTecplot_1.addNotifier('vec_y', self.vecYGetterAction)
        self.ReadTecplot_1.addNotifier('vec_z', self.vecZGetterAction)
        self.ReadTecplot_1.addNotifier('scalar_0', self.bottomGetterAction)
        self.ReadTecplot_1.addNotifier('scalar_1', self.watersurfaceGetterAction)
        self.ReadTecplot_1.addNotifier('scalar_2', self.scalarGetterAction)
        #
        # set format to autodetect
        #
        self.ReadTecplot_1.set_format_of_file( 1 )
        
        
        #
        # set scale
        #
        self.ReadTecplot_1.set_scale_z( 1,2*self.scale, self.scale )

        #
        #
        #
        self.Transform_bottomGrid=None
        self.GridSurface_bottomGrid=None
        self.RWCovise_bottomGrid=None
        self.RWCovise_bottom=None
        
        self.GridSurface_watersurfaceGrid=None
        self.RWCovise_watersurfaceGrid=None
        self.RWCovise_watersurface=None
        self.RWCovise_velocity2D=None
        self.RWCovise_scalar2D=None
        
        self.RWCovise_grid3D=None
        self.RWCovise_velocity3D=None
        self.RWCovise_scalar3D=None


        #
        # connect buttons
        #
        self.dimensionComboBox.activated.connect(self.setDimension)
        self.dimensionComboBox.currentIndexChanged.connect(self.setDimension)
        self.formatComboBox.activated.connect(self.setFormat)
        self.formatComboBox.currentIndexChanged.connect(self.setFormat)
        self.scaleZLineEdit.returnPressed.connect(self.setScale)
        self.startConversionPushButton.clicked.connect(self.startConversion)
        self.outputDirLineEdit.returnPressed.connect(self.setOutputDir)
        self.grid2DWidget.gridXComboBox.activated.connect(self.setGrid2DX)
        self.grid2DWidget.gridYComboBox.activated.connect(self.setGrid2DY)
        self.bottomWidget.gridZ0ComboBox.activated.connect(self.setGrid2DZ0)
        self.bottomWidget.bottomGridNameLineEdit.returnPressed.connect(self.setBottomGridName)
        self.bottomWidget.bottomVariableNameLineEdit.returnPressed.connect(self.setBottomVariableName)
        self.watersurfaceWidget.gridZ1ComboBox.activated.connect(self.setGrid2DZ1)
        self.watersurfaceWidget.watersurfaceOffsetLineEdit.returnPressed.connect(self.setOffset)
        self.watersurfaceWidget.watersurfaceGridNameLineEdit.returnPressed.connect(self.setWatersurfaceGridName)
        self.watersurfaceWidget.watersurfaceVariableNameLineEdit.returnPressed.connect(self.setWatersurfaceVariableName)
        
        self.grid3DWidget.gridXComboBox.activated.connect(self.setGrid3DX)
        self.grid3DWidget.gridYComboBox.activated.connect(self.setGrid3DY)
        self.grid3DWidget.gridZComboBox.activated.connect(self.setGrid3DZ)   
        self.grid3DWidget.gridNameLineEdit.returnPressed.connect(self.setGrid3DName)           
          
        self.velocity2DWidget.vecXComboBox.activated.connect(self.setVelocity2DU) 
        self.velocity2DWidget.vecYComboBox.activated.connect(self.setVelocity2DV) 
        self.velocity2DWidget.vectorVariableNameLineEdit.returnPressed.connect(self.setVector2DVariableName) 
        
        self.velocity3DWidget.vecXComboBox.activated.connect(self.setVelocity3DU) 
        self.velocity3DWidget.vecYComboBox.activated.connect(self.setVelocity3DV) 
        self.velocity3DWidget.vecZComboBox.activated.connect(self.setVelocity3DW) 
        self.velocity3DWidget.vectorVariableNameLineEdit.returnPressed.connect(self.setVector3DVariableName) 
       
        self.scalar2DWidget.scalarComboBox.activated.connect(self.setScalar2D) 
        self.scalar2DWidget.scalarVariableNameLineEdit.returnPressed.connect(self.setScalar2DVariableName) 
        
        self.scalar3DWidget.scalarComboBox.activated.connect(self.setScalar3D) 
        self.scalar3DWidget.scalarVariableNameLineEdit.returnPressed.connect(self.setScalar3DVariableName) 


    def closeEvent(self, event):
        covise.clean()
        covise.quit()


    def customEvent(self,e):
        if e.type() == NOTIFIER_EVENT:
            self.variables=e.value[1:]
            
            if e.param == "grid_x":
                if self.grid2DXDone == 'False':
                    self.statusText.append("INFO: File format selection is correct")
                    self.grid2DWidget.gridXComboBox.clear()
                    for v in self.variables:
                        self.grid2DWidget.gridXComboBox.addItem(v)                
                    self.grid2DXDone='True'
                if self.grid3DXDone == 'False':
                    self.grid3DWidget.gridXComboBox.clear()
                    for v in self.variables:
                        self.grid3DWidget.gridXComboBox.addItem(v)                
                    self.grid3DXDone='True'
                        
            if e.param == "grid_y":
                if self.grid2DYDone == 'False':   
                    self.grid2DWidget.gridYComboBox.clear()
                    for v in self.variables:
                        self.grid2DWidget.gridYComboBox.addItem(v)
                    self.grid2DYDone='True'
                if self.grid3DYDone == 'False':   
                    self.grid3DWidget.gridYComboBox.clear()
                    for v in self.variables:
                        self.grid3DWidget.gridYComboBox.addItem(v)
                    self.grid3DYDone='True'    
                        
            if e.param == "grid_z0":
                if self.grid2DZ0Done == 'False':
                    self.bottomWidget.gridZ0ComboBox.clear()
                    for v in self.variables:
                        self.bottomWidget.gridZ0ComboBox.addItem(v)
                    self.grid2DZ0Done='True'
                if self.grid3DZDone == 'False':
                    self.grid3DWidget.gridZComboBox.clear()
                    for v in self.variables:
                        self.grid3DWidget.gridZComboBox.addItem(v)
                    self.grid3DZDone='True'
                   
            if e.param == "grid_z1"and self.grid2DZ1Done == 'False':  
                self.watersurfaceWidget.gridZ1ComboBox.clear()
                for v in self.variables:
                    self.watersurfaceWidget.gridZ1ComboBox.addItem(v)
                self.grid2DZ1Done='True'    
            
            if e.param == "vec_x":
                if self.vec2DXDone == 'False':
                    self.velocity2DWidget.vecXComboBox.clear()
                    for v in self.variables:
                        self.velocity2DWidget.vecXComboBox.addItem(v)
                    self.vec2DXDone='True'
                if self.vec3DXDone == 'False':
                    self.velocity3DWidget.vecXComboBox.clear()
                    for v in self.variables:
                        self.velocity3DWidget.vecXComboBox.addItem(v)
                    self.vec3DXDone='True'
                    
            if e.param == "vec_y":
                if self.vec2DYDone == 'False':
                    self.velocity2DWidget.vecYComboBox.clear()
                    for v in self.variables:
                        self.velocity2DWidget.vecYComboBox.addItem(v)
                    self.vec2DYDone='True'
                if self.vec3DYDone == 'False':
                    self.velocity3DWidget.vecYComboBox.clear()
                    for v in self.variables:
                        self.velocity3DWidget.vecYComboBox.addItem(v)
                    self.vec3DYDone='True'
 
            if e.param == "vec_z":
                if self.vec3DZDone == 'False':
                    self.velocity3DWidget.vecZComboBox.clear()
                    for v in self.variables:
                        self.velocity3DWidget.vecZComboBox.addItem(v)
                    self.vec3DZDone='True'
                   
            if e.param == "scalar_0" and self.scalar0Done == 'False':
                self.scalarDone='True'
                
            if e.param == "scalar_1" and self.scalar1Done == 'False':
                self.scalar1Done='True'
                
            if e.param == "scalar_2":
                if self.scalar22DDone == 'False':
                    self.scalar2DWidget.scalarComboBox.clear()
                    for v in self.variables:
                        self.scalar2DWidget.scalarComboBox.addItem(v)
                    self.scalar22DDone='True'
                    
                if self.scalar23DDone == 'False':
                    self.scalar3DWidget.scalarComboBox.clear()
                    for v in self.variables:
                        self.scalar3DWidget.scalarComboBox.addItem(v)
                    self.scalar23DDone='True'
            
            if self.grid2DXDone == "True" and self.grid2DYDone=='True':
                self.grid2DWidget.setEnabled(True)
                #self.grid2DWidget.show()

            if self.grid3DXDone == "True" and self.grid3DYDone=='True' and self.grid3DZDone=='True':
                self.grid3DWidget.setEnabled(True)
                #self.grid3DWidget.show()
                
   
        if e.type() == ERROR_EVENT:
            if self.outputFilePath:           
                text="ERROR: "+e.error+"\n"            
                self.statusText.append(text)  
                  
    def fileOpen(self):
    
    
        fd = QtWidgets.QFileDialog(self)#, 'Open Tecplot File',self.currentFilePath,'tecplot file (*.plt *.dat)')
        fd.setMinimumWidth(1050)
        fd.setMinimumHeight(700)
        fd.setNameFilter('tecplot file (*.plt *.dat)')
        fd.setWindowTitle('Open Tecplot File')
        fd.setDirectory(self.currentFilePath)

        acceptedOrRejected = fd.exec_()
        if acceptedOrRejected != QtWidgets.QDialog.Accepted :
            return
        filenamesQt = fd.selectedFiles()
        if not len(filenamesQt):
            return
        self.currentFilePath = os.path.dirname(str(filenamesQt[0]))
        fullTecplotCaseName = str(filenamesQt[0])
        
        
        #
        # try to open file
        #
        if not os.access(fullTecplotCaseName, os.R_OK):
            self.statusText.append("ERROR: Could not open file "+fullTecplotCaseName+ " - not readable")  
        else:
            #self.setCaption(os.path.basename(fullTecplotCaseName))

            #
            # set filename
            #
            self.ReadTecplot_1.set_fullpath( fullTecplotCaseName )
       
            #
            # disable file open
            #
            self.fileOpenAction.setEnabled(False) 
        
            #
            # set output File path
            #
            self.outputDirFrame.setEnabled(True)
            self.outputFilePath = self.currentFilePath + "/CoviseDaten"
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
                #
                # enable format
                #
                self.formatFrame.setEnabled(True)
                self.settingsFrame.setEnabled(True)
    def setDimension(self, iii):
        index=self.dimensionComboBox.currentIndex()
        self.dimension=index+2  
        
        if self.dimension==2:
            self.grid2DWidget.show()
            self.bottomWidget.show()
            self.watersurfaceWidget.show()
            self.velocity2DWidget.show()
            self.scalar2DWidget.show()
            self.grid3DWidget.hide()
            self.velocity3DWidget.hide()
            self.scalar3DWidget.hide()

        if self.dimension==3:
            self.grid2DWidget.hide()
            self.bottomWidget.hide()
            self.watersurfaceWidget.hide()
            self.velocity2DWidget.hide()
            self.scalar2DWidget.hide()
            self.grid3DWidget.show()
            self.velocity3DWidget.show()
            self.scalar3DWidget.show()
            
    def setFormat(self, iii):
        index=self.formatComboBox.currentIndex()
        # selfainbyteswapped
        if index==2:
            i=7
        else:
            # +1 fuer covise choice 
            i=index+1    
            #print("covise index=", i)
        self.format=i    
                
        self.startConversionPushButton.setEnabled(False)
        self.grid2DXDone='False'
        self.grid2DYDone='False'
        self.grid2DZ0Done='False'
        self.grid2DZ1Done='False'
        self.vec2DXDone='False'
        self.vec2DYDone='False'
        
        self.grid3DXDone='False'
        self.grid3DYDone='False'
        self.grid3DZDone='False'
        self.vec3DXDone='False'
        self.vec3DYDone='False'
        self.vec3DZDone='False'
        
        self.scalar0Done='False'
        self.scalar1Done='False'
        self.scalar22DDone='False'
        self.scalar23DDone='False'
        
        self.ReadTecplot_1.set_format_of_file( i )
        #print("aZonesCollectorAction.waitForZonesInfoFinished")
        self.aZonesCollectorAction.waitForZonesInfoFinished()
        

    def checkLineEdits(self):
    
        if self.grid2DXSelected=="True" and self.grid2DYSelected=="True":
            self.bottomWidget.setEnabled(True)
            self.watersurfaceWidget.setEnabled(True)
            self.velocity2DWidget.setEnabled(True)
            self.scalar2DWidget.setEnabled(True)
         
        # bottom grid   
        if self.grid2DXSelected=="True" and self.grid2DYSelected=="True" and self.grid2DZ0Selected=="True":
            self.bottomGridFileName = self.outputFilePath+"/bottom_grid2D.covise"
            self.bottomWidget.bottomGridNameLineEdit.setText(self.bottomGridFileName)                
        else:
            self.bottomGridFileName = None
            self.bottomWidget.bottomGridNameLineEdit.clear()
                  
        # watersurface grid          
        if self.grid2DXSelected=="True" and self.grid2DYSelected=="True" and self.grid2DZ1Selected=="True":
            self.watersurfaceGridFileName = self.outputFilePath+"/watersurface_grid2D.covise" 
            self.watersurfaceWidget.watersurfaceGridNameLineEdit.setText(self.watersurfaceGridFileName)               
        else:
            self.watersurfaceGridFileName=None
            self.watersurfaceWidget.watersurfaceGridNameLineEdit.clear()
                        
        # velocity 2D variable        
        if self.vel2DUSelected=="True" and self.vel2DVSelected=="True":
            self.vector2DVariableFileName = self.outputFilePath+"/vector2D.covise"
            self.velocity2DWidget.vectorVariableNameLineEdit.setText(self.vector2DVariableFileName)
        else:
            self.vector2DVariableFileName=None
            self.velocity2DWidget.vectorVariableNameLineEdit.clear()
            self.velocity2DWidget.vectorVariableNameLineEdit.leavePendingMode()
 
        # bottom variable        
        if self.grid2DZ0Selected=="True":
            self.bottomVariableFileName = self.outputFilePath+"/bottom2D.covise" 
            self.bottomWidget.bottomVariableNameLineEdit.setText(self.bottomVariableFileName)
                        
        else:
            self.bottomVariableFileName=None
            self.bottomWidget.bottomVariableNameLineEdit.clear()
 
        
        # watersurface variable
        if self.grid2DZ1Selected=="True":
            self.watersurfaceVariableFileName = self.outputFilePath+"/watersurface2D.covise" 
            self.watersurfaceWidget.watersurfaceVariableNameLineEdit.setText(self.watersurfaceVariableFileName) 
        else:
            self.watersurfaceVariableFileName=None
            self.watersurfaceWidget.watersurfaceVariableNameLineEdit.clear()
                               
        # 3D grid   
        if self.grid3DXSelected=="True" and self.grid3DYSelected=="True" and self.grid3DZSelected=="True":
            self.velocity3DWidget.setEnabled(True)
            self.scalar3DWidget.setEnabled(True)
            self.grid3DFileName = self.outputFilePath+"/grid3D.covise"
            self.grid3DWidget.gridNameLineEdit.setText(self.grid3DFileName)
        else:
            self.grid3DFileName = None
            self.grid3DWidget.gridNameLineEdit.clear()
    
        # velocity 3D variable        
        if self.vel3DUSelected=="True" and self.vel3DVSelected=="True" and self.vel3DWSelected=="True":
            self.vector3DVariableFileName = self.outputFilePath+"/vector3D.covise"
            self.velocity3DWidget.vectorVariableNameLineEdit.setText(self.vector3DVariableFileName)
        else:
            self.vector3DVariableFileName=None
            self.velocity3DWidget.vectorVariableNameLineEdit.clear()
            self.velocity3DWidget.vectorVariableNameLineEdit.leavePendingMode()
             

        # scalar2 variable
        if self.scalar2DSelected=="True":
            self.scalar2DVariableFileName = self.outputFilePath+"/scalar2D.covise"
            self.scalar2DWidget.scalarVariableNameLineEdit.setText(self.scalar2DVariableFileName)
        else:
            self.scalar2DVariableFileName=None
            self.scalar2DWidget.scalarVariableNameLineEdit.clear()
            self.scalar2DWidget.scalarVariableNameLineEdit.leavePendingMode()
  
        # scalar3 variable
        if self.scalar3DSelected=="True":
            self.scalar3DVariableFileName = self.outputFilePath+"/scalar3D.covise"
            self.scalar3DWidget.scalarVariableNameLineEdit.setText(self.scalar3DVariableFileName)
        else:
            self.scalar3DVariableFileName=None
            self.scalar3DWidget.scalarVariableNameLineEdit.clear()
            self.scalar3DWidget.scalarVariableNameLineEdit.leavePendingMode()            

        if self.dimension==2:
            if self.bottomGridFileName or self.watersurfaceGridFileName or self.bottomVariableFileName or self.watersurfaceVariableFileName or self.vector2DVariableFileName or self.scalar2DVariableFileName:      
                self.startConversionFrame.setEnabled(True)  
                self.startConversionPushButton.setEnabled(True)      
            else:
                self.startConversionFrame.setEnabled(False)      
                self.startConversionPushButton.setEnabled(False)      

        if self.dimension==3:
            if self.grid3DFileName or self.vector3DVariableFileName or self.scalar3DVariableFileName:      
                self.startConversionFrame.setEnabled(True)  
                self.startConversionPushButton.setEnabled(True)      
            else:
                self.startConversionFrame.setEnabled(False)      
                self.startConversionPushButton.setEnabled(False)      
        
    def setGrid2DX(self):
        index=self.grid2DWidget.gridXComboBox.currentIndex()
        if index==0:
            self.grid2DXSelected="False"
        else:
            self.grid2DXSelected="True"
        
        self.checkLineEdits()
               
        self.ReadTecplot_1.set_grid_x( index+1 )

    def setGrid3DX(self):
        index=self.grid3DWidget.gridXComboBox.currentIndex()
        if index==0:
            self.grid3DXSelected="False"
        else:
            self.grid3DXSelected="True"
        
        self.checkLineEdits()
               
        self.ReadTecplot_1.set_grid_x( index+1 )

    def setGrid2DY(self):
        index=self.grid2DWidget.gridYComboBox.currentIndex()
        if index==0:
            self.grid2DYSelected="False"
        else:
            self.grid2DYSelected="True"

        self.checkLineEdits()

        self.ReadTecplot_1.set_grid_y( index+1 )

    def setGrid3DY(self):
        index=self.grid3DWidget.gridYComboBox.currentIndex()
        if index==0:
            self.grid3DYSelected="False"
        else:
            self.grid3DYSelected="True"

        self.checkLineEdits()

        self.ReadTecplot_1.set_grid_y( index+1 )

    def setGrid2DZ0(self):
        index=self.bottomWidget.gridZ0ComboBox.currentIndex()
        if index==0:
            self.grid2DZ0Selected="False"
        else:
            self.grid2DZ0Selected="True"

        self.checkLineEdits()

        self.ReadTecplot_1.set_grid_z0( index+1 )
        self.ReadTecplot_1.set_scalar_0( index+1 )
        


    def setGrid2DZ1(self):
        index=self.watersurfaceWidget.gridZ1ComboBox.currentIndex()
        if index==0:
            self.grid2DZ1Selected="False"
        else:
            self.grid2DZ1Selected="True"

        self.checkLineEdits()

        self.ReadTecplot_1.set_grid_z1( index+1 )
        self.ReadTecplot_1.set_scalar_1( index+1 )

    def setGrid3DZ(self):
        index=self.grid3DWidget.gridZComboBox.currentIndex()
        if index==0:
            self.grid3DZSelected="False"
        else:
            self.grid3DZSelected="True"

        self.checkLineEdits()

        self.ReadTecplot_1.set_grid_z0( index+1 )
        self.ReadTecplot_1.set_scalar_0( index+1 )

    def setVelocity2DU(self):
        index=self.velocity2DWidget.vecXComboBox.currentIndex()
        if index==0:
            self.vel2DUSelected="False"
        else:
            self.vel2DUSelected="True"

        self.checkLineEdits()

        self.ReadTecplot_1.set_vec_x( index+1 )
        self.ReadTecplot_1.set_vec_z( 1 )
   
    def setVelocity3DU(self):
        
        index=self.velocity3DWidget.vecXComboBox.currentIndex()
        if index==0:
            self.vel3DUSelected="False"
        else:
            self.vel3DUSelected="True"

        self.checkLineEdits()

        self.ReadTecplot_1.set_vec_x( index+1 )
        self.ReadTecplot_1.set_vec_z( 1 )
          
    def setVelocity2DV(self):
        index=self.velocity2DWidget.vecYComboBox.currentIndex()
        if index==0:
            self.vel2DVSelected="False"
        else:
            self.vel2DVSelected="True"

        self.checkLineEdits()

        self.ReadTecplot_1.set_vec_y( index+1 )
        self.ReadTecplot_1.set_vec_z( 1 )

    def setVelocity3DV(self):
        index=self.velocity3DWidget.vecYComboBox.currentIndex()
        if index==0:
            self.vel3DVSelected="False"
        else:
            self.vel3DVSelected="True"

        self.checkLineEdits()

        self.ReadTecplot_1.set_vec_y( index+1 )
        self.ReadTecplot_1.set_vec_z( 1 )

    def setVelocity3DW(self):
        index=self.velocity3DWidget.vecZComboBox.currentIndex()
        if index==0:
            self.vel3DWSelected="False"
        else:
            self.vel3DWSelected="True"

        self.checkLineEdits()

        self.ReadTecplot_1.set_vec_z( index+1  )

    def setScalar2D(self):
        index=self.scalar2DWidget.scalarComboBox.currentIndex()
        if index==0:
            self.scalar2DSelected="False"
        else:
            self.scalar2DSelected="True"

        self.checkLineEdits()
        self.ReadTecplot_1.set_scalar_2( index+1 )

    def setScalar3D(self):
        index=self.scalar3DWidget.scalarComboBox.currentIndex()
        if index==0:
            self.scalar3DSelected="False"
        else:
            self.scalar3DSelected="True"

        self.checkLineEdits()

        self.ReadTecplot_1.set_scalar_2( index+1 )
        
    def setBottomGridName(self):
        text=self.bottomWidget.bottomGridNameLineEdit.text()
        self.bottomGridFileName=str(text)

    def setBottomVariableName(self):
        text=self.bottomWidget.bottomVariableNameLineEdit.text()
        self.bottomVariableFileName=str(text)


    def setWatersurfaceGridName(self):
        text=self.watersurfaceWidget.watersurfaceGridNameLineEdit.text()
        self.watersurfaceGridFileName=str(text)

    def setWatersurfaceVariableName(self):
        text=self.watersurfaceWidget.watersurfaceVariableNameLineEdit.text()
        self.watersurfaceVariableFileName=str(text)

    def setVector2DVariableName(self):
        text=self.velocity2DWidget.vectorVariableNameLineEdit.text()
        self.vector2DVariableFileName=str(text)

    def setScalar2DVariableName(self):
        text=self.scalar2DWidget.scalarVariableNameLineEdit.text()
        self.scalar2DVariableFileName=str(text)        

    def setGrid3DName(self):
        text=self.grid3DWidget.gridNameLineEdit.text()
        self.grid3DFileName=str(text)

    def setVector3DVariableName(self):
        text=self.velocity3DWidget.vectorVariableNameLineEdit.text()
        self.vector3DVariableFileName=str(text)
                
    def setScalar3DVariableName(self):
        text=self.scalar3DWidget.scalarVariableNameLineEdit.text()
        self.scalar3DVariableFileName=str(text)        
        
    def setScale(self):
        text=self.scaleZLineEdit.text()
        scale=float(str(text))        
        self.ReadTecplot_1.set_scale_z( 1,2*scale, scale )

    def setOffset(self):
        text=self.watersurfaceWidget.watersurfaceOffsetLineEdit.text()
        self.waterSurfaceOffset=float(str(text))        
        
    def setOutputDir(self):
        self.outputFilePath=str(self.outputDirLineEdit.text())
        if os.path.isdir(self.outputFilePath):
            
            #
            # enable format
            #
            self.formatFrame.setEnabled(True)
            self.ReadTecplot_1.set_format_of_file( 1 )
            
        else:
            try:
                os.mkdir(self.outputFilePath)
            except(OSError):
                self.statusText.append("ERROR: Could not create directory "+self.outputFilePath+" check permissions and enter again or select another directory")
                self.outputFilePath = None
        
                
    def fileExit(self):
        sys.exit()



    def startModules(self):
        global theNet
        
        nz=self.aZonesCollectorAction.getNumZones()
        names=self.aZonesCollectorAction.getZoneNames()


        zoneInterpretation=self.zonesComboBox.currentIndex()
        if zoneInterpretation==0: # timesteps
             self.ReadTecplot_1.set_data_has_timesteps("TRUE")
             nz=1
             names[0]="timesteps"
        else: #blocks
             self.ReadTecplot_1.set_data_has_timesteps("FALSE")
             
                  
        # bottom grid   
        if self.grid2DXSelected=="True" and self.grid2DYSelected=="True" and self.grid2DZ0Selected=="True":
            self.RWCovise_bottomGrid = {}
            self.GetSubset_bottomGrid = {}
            
            for i in range(0, nz):
                self.RWCovise_bottomGrid[i] = RWCovise()
                theNet.add( self.RWCovise_bottomGrid[i] )
                self.bottomGridFileName = self.outputFilePath+"/bottom_grid2D_"+names[i]+".covise"
                self.RWCovise_bottomGrid[i].set_grid_path( self.bottomGridFileName )
                if zoneInterpretation==1:
                    GetSubset_bottomGrid[i] = GetSubset()
                    theNet.add( GetSubset_bottomGrid[i] )
                    GetSubset_bottomGrid[i].set_selection( str(i) )
            if self.untrimCheckBox.isChecked():
                self.Transform_bottomGrid = Transform()
                theNet.add(self.Transform_bottomGrid)
                self.Transform_bottomGrid.set_Transform(5)
                self.Transform_bottomGrid.set_scale_type(4)
                self.Transform_bottomGrid.set_scaling_factor(-1.0)
                self.Transform_bottomGrid.set_createSet( "FALSE" )
                self.GridSurface_bottomGrid  = GridSurface()
                theNet.add(self.GridSurface_bottomGrid)
                theNet.connect( self.ReadTecplot_1, "grid", self.Transform_bottomGrid, "geo_in" ) 
                theNet.connect( self.Transform_bottomGrid, "geo_out", self.GridSurface_bottomGrid, "gridIn0" ) 
                if zoneInterpretation==0: # timesteps
                    for i in range(0, nz):
                        theNet.connect( self.GridSurface_bottomGrid, "GridOut0", self.RWCovise_bottomGrid[i], "mesh_in" )
                else:
                    for i in range(0, nz):
                        theNet.connect( self.GridSurface_bottomGrid, "GridOut0", self.GetSubset_bottomGrid[i], "DataIn0" )
                        theNet.connect( self.GetSubset_bottomGrid[i], "DataOut0", self.RWCovise_bottomGrid[i], "mesh_in" )
            else: 
                for i in range(0, nz):
                    if zoneInterpretation==0: # timesteps
                        theNet.connect( self.ReadTecplot_1, "grid", self.RWCovise_bottomGrid[i], "mesh_in" )
                    else:
                        theNet.connect( self.ReadTecplot_1, "grid", self.GetSubset_bottomGrid[i], "DataIn0" )
                        theNet.connect( self.GetSubset_bottomGrid, "DataOut0", self.RWCovise_bottomGrid[i], "mesh_in" )
                  
        # watersurface grid          
        if self.grid2DXSelected=="True" and self.grid2DYSelected=="True" and self.grid2DZ1Selected=="True":
            self.RWCovise_watersurfaceGrid = {}
            self.GetSubset_watersurfaceGrid = {}
            self.Transform_1 = Transform()
            theNet.add( self.Transform_1 )
            self.Transform_1.set_Transform( 3 )
            self.Transform_1.set_vector_of_translation( 0,0, float(self.waterSurfaceOffset) )
            self.Transform_1.set_createSet( "FALSE" )
            theNet.connect( self.ReadTecplot_1, "grid2", self.Transform_1, "geo_in" )  
            for i in range(0, nz):
                self.RWCovise_watersurfaceGrid = RWCovise()
                theNet.add( self.RWCovise_watersurfaceGrid )
                self.watersurfaceGridFileName = self.outputFilePath+"/watersurface_grid2D_"+names[i]+".covise"
                self.RWCovise_watersurfaceGrid.set_grid_path( self.watersurfaceGridFileName)
                if zoneInterpretation==1: #blocks
                    GetSubset_watersurfaceGrid[i] = GetSubset()
                    theNet.add( GetSubset_watersurfaceGrid[i] )
                    GetSubset_watersurfaceGrid[i].set_selection( str(i) )                              
            if self.untrimCheckBox.isChecked():
                self.GridSurface_watersurfaceGrid  = GridSurface()
                theNet.add(self.GridSurface_watersurfaceGrid)
                theNet.connect( self.Transform_1, "geo_out", self.GridSurface_watersurfaceGrid, "gridIn0" )
                if zoneInterpretation==0: # timesteps
                    theNet.connect( self.GridSurface_watersurfaceGrid, "GridOut0", self.RWCovise_watersurfaceGrid[i], "mesh_in" )   
                else:
                    theNet.connect( self.GridSurface_watersurfaceGrid, "GridOut0", self.GetSubset_watersurfaceGrid[i], "DataIn0" )
                    theNet.connect( self.GetSubset_watersurfaceGrid[i], "DataOut0", self.RWCovise_watersurfaceGrid[i], "mesh_in" )      
            else:
                for i in range(0, nz):
                    if zoneInterpretation==0: # timesteps
                        theNet.connect( self.Transform_1, "geo_out", self.GetSubset_watersurfaceGrid, "mesh_in" )
                    else: #blocks
                        theNet.connect( self.Transform_1, "geo_out", self.GetSubset_watersurfaceGrid[i], "DataIn0" )
                        theNet.connect( self.GetSubset_watersurfaceGrid[i], "DataOut0", self.RWCovise_watersurfaceGrid[i], "mesh_in" )
                      
        # velocity 2D variable        
        if self.vel2DUSelected=="True" and self.vel2DVSelected=="True":
            self.RWCovise_velocity2D = {}
            self.GetSubset_velocity2D = {}
            for i in range(0, nz):
                self.RWCovise_velocity2D[i] = RWCovise()
                theNet.add( self.RWCovise_velocity2D[i] )
                self.vector2DVariableFileName = self.outputFilePath+"/vector2D_"+names[i]+".covise"
                self.RWCovise_velocity2D[i].set_grid_path( self.vector2DVariableFileName )
                if zoneInterpretation==0: # timesteps
                    theNet.connect( self.ReadTecplot_1, "vector", self.RWCovise_velocity2D[i], "mesh_in" )
                else: #blocks
                    self.GetSubset_velocity2D[i] = GetSubset()
                    theNet.add( self.GetSubset_velocity2D[i] )
                    self.GetSubset_velocity2D[i].set_selection( str(i) )
                    theNet.connect( self.ReadTecplot_1, "vector", self.GetSubset_velocity2D[i], "DataIn0" )
                    theNet.connect( self.GetSubset_velocity2D, "DataOut0", self.RWCovise_velocity2D[i], "mesh_in" )     

        # bottom variable        
        if self.grid2DZ0Selected=="True":
            self.RWCovise_bottom = {}
            self.GetSubset_bottom = {}
            for i in range(0, nz):
                self.RWCovise_bottom[i] = RWCovise()
                theNet.add( self.RWCovise_bottom[i] )
                self.bottomVariableFileName = self.outputFilePath+"/bottom2D_"+names[i]+".covise"
                self.RWCovise_bottom[i].set_grid_path( self.bottomVariableFileName )
                if zoneInterpretation==1: #blocks
                    self.GetSubset_bottom[i] = GetSubset()
                    theNet.add( self.GetSubset_bottom[i] )
                    self.GetSubset_bottom[i].set_selection( str(i) )
                    
            if self.untrimCheckBox.isChecked():
                self.Calc = Calc()
                theNet.add( self.Calc )
                self.Calc.set_expression("0-s1")
                theNet.connect( self.ReadTecplot_1, "dataout0", self.Calc, "DataIn0" )
                if zoneInterpretation==0: # timesteps
                    theNet.connect( self.Calc, "DataOut0", self.RWCovise_bottom[i], "mesh_in" )
                else: #blocks                    
                    theNet.connect( self.Calc, "DataOut0", self.GetSubset_bottom[i], "DataIn0" )
                    theNet.connect( self.GetSubset_bottom, "DataOut0", self.RWCovise_bottom[i], "mesh_in" )      
            else:
                if zoneInterpretation==0: # timesteps
                    theNet.connect( self.ReadTecplot_1, "dataout0", self.RWCovise_bottom, "mesh_in"  )
                else: #blocks
                    theNet.connect( self.ReadTecplot_1, "dataout0", self.GetSubset_bottom, "DataIn0"  )
                    theNet.connect( self.GetSubset_bottom, "DataOut0", self.RWCovise_bottom, "mesh_in"  )
                
        # watersurface variable
        if self.grid2DZ1Selected=="True":
            self.RWCovise_watersurface = {}
            self.GetSubset_watersurface = {}
            for i in range(0, nz):
                self.RWCovise_watersurface[i] = RWCovise()
                theNet.add( self.RWCovise_watersurface[i] )
                self.watersurfaceVariableFileName = self.outputFilePath+"/watersurface2D"+names[i]+".covise"
                self.RWCovise_watersurface[i].set_grid_path( self.watersurfaceVariableFileName)
                if zoneInterpretation==1:  #blocks
                    self.GetSubset_watersurface[i] = GetSubset()
                    theNet.add( self.GetSubset_watersurface[i] )
                    self.GetSubset_watersurface[i].set_selection( str(i) )
                    theNet.connect( self.ReadTecplot_1, "dataout1", self.GetSubset_watersurface[i], "DataIn0" )
                    theNet.connect( self.GetSubset_watersurface, "DataOut0", self.RWCovise_watersurface[i], "mesh_in" )
                else:
                    theNet.connect( self.ReadTecplot_1, "dataout1", self.RWCovise_watersurface[i], "mesh_in" )
            
        # scalar2 variable
        if self.scalar2DSelected=="True":
            self.RWCovise_scalar2D = {}
            self.GetSubset_scalar2D = {}
            for i in range(0, nz):
                self.RWCovise_scalar2D[i] = RWCovise()
                theNet.add( self.RWCovise_scalar2D[i] )
                self.scalar2DVariableFileName = self.outputFilePath+"/scalar2D_"+names[i]+".covise"
                self.RWCovise_scalar2D[i].set_grid_path( self.scalar2DVariableFileName)
                if zoneInterpretation==1:  #blocks
                    self.GetSubset_scalar2D[i] = GetSubset()
                    theNet.add( self.GetSubset_scalar2D[i] )
                    self.GetSubset_scalar2D[i].set_selection( str(i) )
                    theNet.connect( self.ReadTecplot_1, "dataout2", self.GetSubset_scalar2D[i], "DataIn0" )
                    theNet.connect( self.GetSubset_scalar2D, "DataOut0", self.RWCovise_scalar2D[i], "mesh_in" )
                else:    
                    theNet.connect( self.ReadTecplot_1, "dataout2", self.RWCovise_scalar2D[i], "mesh_in" )

                               
        # 3D grid   
        if self.grid3DXSelected=="True" and self.grid3DYSelected=="True" and self.grid3DZSelected=="True":
            self.RWCovise_grid3D = {}
            self.GetSubset_grid3D = {}
            for i in range(0, nz):
               self.RWCovise_grid3D[i] = RWCovise()
               theNet.add( self.RWCovise_grid3D[i] )
               self.grid3DFileName = self.outputFilePath+"/grid3D_"+names[i]+".covise"
               self.RWCovise_grid3D[i].set_grid_path( self.grid3DFileName )
               if zoneInterpretation==1:  #blocks
                    self.GetSubset_grid3D[i] = GetSubset()
                    theNet.add( self.GetSubset_grid3D[i] )
                    self.GetSubset_grid3D[i].set_selection( str(i) )
                    theNet.connect( self.ReadTecplot_1, "grid", self.GetSubset_grid3D[i], "DataIn0" )
                    theNet.connect( self.GetSubset_grid3D[i], "DataOut0", self.RWCovise_grid3D[i], "mesh_in" )
               else:
                   theNet.connect( self.ReadTecplot_1, "grid", self.RWCovise_grid3D[i], "mesh_in" )
   
        # velocity 3D variable        
        if self.vel3DUSelected=="True" and self.vel3DVSelected=="True" and self.vel3DWSelected=="True":
            self.RWCovise_velocity3D = {}
            self.GetSubset_velocity3D = {}
            for i in range(0, nz):
                self.RWCovise_velocity3D[i] = RWCovise()
                theNet.add( self.RWCovise_velocity3D[i] )
                self.vector3DVariableFileName = self.outputFilePath+"/vector3D_"+names[i]+".covise"
                self.RWCovise_velocity3D[i].set_grid_path( self.vector3DVariableFileName )
                if zoneInterpretation==1:  #blocks
                    self.GetSubset_velocity3D[i] = GetSubset()
                    theNet.add( self.GetSubset_velocity3D[i] )
                    self.GetSubset_velocity3D[i].set_selection( str(i) )
                    theNet.connect( self.ReadTecplot_1, "vector", self.GetSubset_velocity3D[i], "DataIn0" )
                    theNet.connect( self.GetSubset_velocity3D[i], "DataOut0", self.RWCovise_velocity3D[i], "mesh_in" )
                else:
                    theNet.connect( self.ReadTecplot_1, "vector", self.RWCovise_velocity3D[i], "mesh_in" )
          

  
        # scalar3 variable
        if self.scalar3DSelected=="True":
            self.RWCovise_scalar3D = {}
            self.GetSubset_scalar3D = {}
            for i in range(0, nz):
                self.RWCovise_scalar3D[i] = RWCovise()
                theNet.add( self.RWCovise_scalar3D[i] )
                self.scalar3DVariableFileName = self.outputFilePath+"/scalar3D"+names[i]+".covise"
                self.RWCovise_scalar3D[i].set_grid_path( self.scalar3DVariableFileName)
                if zoneInterpretation==1:  #blocks
                    self.GetSubset_scalar3D[i] = GetSubset()
                    theNet.add( self.GetSubset_scalar3D[i] )
                    self.GetSubset_scalar3D[i].set_selection( str(i) )
                    theNet.connect( self.ReadTecplot_1, "dataout2", self.GetSubset_scalar3D[i], "DataIn0" )
                    theNet.connect( self.GetSubset_scalar3D[i], "DataOut0", self.RWCovise_scalar3D[i], "mesh_in" )
                else:    
                    theNet.connect( self.ReadTecplot_1, "dataout2", self.RWCovise_scalar3D[i], "mesh_in" )
           

    def startConversion(self):
       
        global theNet
              
        self.statusText.append("Starting conversion...")
        self.startModules()
        self.repaint()
        
        #pdM = PatienceDialogManager("starting conversion, please be patient...")
        #pdM.spawnPatienceDialog("converting data")                

        #
        # enable translation in origin, this executes the module        #      
        if self.translateCheckBox.isChecked(): 
            self.ReadTecplot_1.set_auto_trans( "TRUE" )
        else:
            self.ReadTecplot_1.execute()
        theNet.finishedBarrier()

        
        text="...conversion finished!\n"
        self.statusText.append(text)

        #theNet.save("tecplot2covise.net")

        #pdM.unSpawnPatienceDialog()
