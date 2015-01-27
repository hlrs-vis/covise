
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject, pyqtSignal

from printing import InfoPrintCapable

from Gui2Neg import theGuiMsgHandler
from ColorMapCustomBase import Ui_ColorMapCustomBase
from coColorTable import coColorTableParams
from KeydObject import TYPE_COLOR_TABLE, TYPE_2D_PART
from ObjectMgr import ObjectMgr

from Utils import (
    getDoubleInLineEdit,
    getIntInLineEdit,
    roundToZero)
    
from vtrans import coTranslate 

class ColorManager(QtWidgets.QDialog, Ui_ColorMapCustomBase):

    """ Handling of color definitions ( ColorTables )

    """
    sigSelectColorMap = pyqtSignal()

    def __init__(self, parent):
        QtWidgets.QDialog.__init__(self, parent )
        Ui_ColorMapCustomBase.__init__(self)
        self.__key2Params = {}
        self.__key2Species = {}
        self.__currentKey = None
        self.__dependantKeys = {}
        self.__callerKey = None
        self.__colorMgrKey = 0

        # key of new create color table
        self.__newkey = -1

        # store params
        self.__baseObjectName = 'Stream'
        self.__baseMin = 0.
        self.__baseMax = 1.
        self.__globalMin = -1.
        self.__globalMax = 2.
        self.__colorMapList = []
        
    def setupManagerUi(self):         
        Ui_ColorMapCustomBase.setupUi( self, self )
        # temporary disabled
        self.vrpPushButtonDelete.setEnabled(False)
        self.groupBox_2.hide()

        self.vrpCheckBoxShow.clicked.connect(self.applyParams)
        self.vrpLineEditName.returnPressed.connect(self.applyParams)
        self.vrpLineEditMin.returnPressed.connect(self.applyParams)
        self.vrpLineEditMax.returnPressed.connect(self.applyParams)
        self.vrpSpinBoxSteps.valueChanged.connect(self.applyParams)
        self.vrpComboBoxMap.activated.connect(self.applyParams)
        self.vrpListViewColorMaps.itemSelectionChanged.connect(self.listViewClick)
        self.vrpPushButtonCopy.clicked.connect(self.copy)
       
        for w in [self.radioButtonManual,
                  self.radioButtonLocal,
                  self.radioButtonGlobal]:
            w.clicked.connect(self.changedRadioButtonGroup)
            

        # validators
        # allow only double values for changeIndicatedLEs
        doubleValidator = QtGui.QDoubleValidator(self.vrpLineEditMin)
        self.vrpLineEditMin.setValidator(doubleValidator)
        self.vrpLineEditMax.setValidator(doubleValidator)


    def setCallerKey(self, callerKey):
        self.__callerKey = callerKey


    def listViewClick(self):
        if (self.vrpListViewColorMaps.selectedItems() == []):
            return
        item = self.vrpListViewColorMaps.selectedItems()[0]
        itemName = str(item.text())
        for key in self.__key2Params:
            if self.__key2Params[key].name==itemName:
                if not key == self.__currentKey:
                    self.setParams( key, self.__key2Params[key] )
                    self.emitSelectColormap( key )

    def emitSelectColormap( self, key ):
        self.sigSelectColorMap.emit(self.__callerKey, key, self.__key2Params[key].name )

#    def accept( self ):
#        pass
    
    def getColorTables(self, variable):
        # returns pairs( key, name of colormap )
        colorList = []
        for key in self.__key2Params:
            if self.__key2Params[key].species==variable:
                colorList.append( (key, self.__key2Params[key].name) )
        return colorList

    def setColorTableKey(self, key):
        if not self.__currentKey == key:
            self.setParams( key, self.__key2Params[key] )

    def setColorMangagerKey( self, key ):
        self.__colorMgrKey = key

    def __getParams(self):
        data = coColorTableParams()
        data.name = str(self.vrpLineEditName.text())
        data.species = self.__key2Species[self.__currentKey]
        data.min = getDoubleInLineEdit(self.vrpLineEditMin)
        data.max = getDoubleInLineEdit(self.vrpLineEditMax)
        data.steps = self.vrpSpinBoxSteps.value()
        data.colorMapIdx = self.vrpComboBoxMap.currentIndex()+1
        data.colorMapList = self.__colorMapList
        if self.radioButtonManual.isChecked():
            data.mode = coColorTableParams.FREE
        elif self.radioButtonLocal.isChecked():
            data.mode = coColorTableParams.LOCAL
        else:
            data.mode = coColorTableParams.GLOBAL
        data.baseObjectName = self.__baseObjectName
        data.baseMin = self.__baseMin
        data.baseMax = self.__baseMax
        data.globalMin = self.__globalMin
        data.globalMax = self.__globalMax
        data.dependantKeys = []
        if self.__currentKey in self.__dependantKeys:
            for key in self.__dependantKeys[self.__currentKey]:
                if key not in data.dependantKeys: # ignore duplicates
                    # check if the object still exists and uses this colormap
                    params = ObjectMgr().getParamsOfObject(key)
                    if params != None:
                        if params.secondVariable!=None:
                            currentVariable = params.secondVariable
                        else:
                            currentVariable = params.variable
                        if currentVariable in params.colorTableKey and params.colorTableKey[currentVariable] == self.__currentKey:
                            data.dependantKeys.append( key )
        return data


    def setParams( self, key, params ):
        self.blockSignals(True)
        # block all widgets with apply
        applyWidgets = [ self.vrpLineEditName,
                         self.vrpLineEditMin,
                         self.vrpLineEditMax,
                         self.radioButtonManual,
                         self.radioButtonLocal,
                         self.radioButtonGlobal,
                         self.vrpSpinBoxSteps,
                         self.vrpComboBoxMap,
                         self.vrpRadioButtonColorGradient]                         
        for widget in applyWidgets:
            widget.blockSignals(True)

        if not params:
            print("empty params in ColorManager.setParams()")
            return

        self.vrpListViewColorMaps.clear()
        self.__key2Params[key] = params
        for lkey in self.__key2Params:
            if self.__key2Params[lkey].species == self.__key2Params[key].species:
                newEntry = QtWidgets.QListWidgetItem(self.__key2Params[lkey].name, self.vrpListViewColorMaps)
                self.vrpListViewColorMaps.addItem(newEntry)

        self.__currentKey = key
        self.__key2Species[key] = params.species
        self.vrpLineEditName.setText(params.name)
        self.vrpLineEditMin.setText(str(roundToZero(params.min)))
        self.vrpLineEditMin.home(False)
        self.vrpLineEditMax.setText(str(roundToZero(params.max)))
        self.vrpLineEditMax.home(False)
        self.vrpSpinBoxSteps.setValue( params.steps )
        if len(params.colorMapList)>4:
            self.__colorMapList = params.colorMapList
            self.vrpComboBoxMap.clear()
            for colorMap in params.colorMapList:
                self.vrpComboBoxMap.addItem(colorMap)
        self.vrpComboBoxMap.setCurrentIndex( params.colorMapIdx-1 )
        if params.mode==coColorTableParams.FREE:
            self.radioButtonManual.setChecked(True)
            self.radioButtonLocal.setChecked(False)
            self.radioButtonGlobal.setChecked(False)
        elif params.mode==coColorTableParams.LOCAL:
            self.radioButtonManual.setChecked(False)
            self.radioButtonLocal.setChecked(True)
            self.radioButtonGlobal.setChecked(False)
        else:
            self.radioButtonManual.setChecked(False)
            self.radioButtonLocal.setChecked(False)
            self.radioButtonGlobal.setChecked(True)
        if params.baseObjectName!="default":
            if (params.baseMin > params.baseMax):
                self.radioButtonLocal.setText(self.__tr("Auto Adjust to %s\nmin: undefined, max: undefined")
                    % params.baseObjectName )
            else:
                self.radioButtonLocal.setText(self.__tr("Auto Adjust to %s\nmin: %1.4f, max: %1.4f")
                    % (params.baseObjectName, params.baseMin, params.baseMax) )
            self.radioButtonLocal.show()
        else :
            self.radioButtonLocal.hide()
        self.__baseObjectName = params.baseObjectName
        self.__baseMin = params.baseMin
        self.__baseMax = params.baseMax
        self.radioButtonGlobal.setText(self.__tr("Auto Adjust to Global\nmin: %1.4f, max: %1.4f")
             % ( params.globalMin, params.globalMax) )
        self.__globalMin = params.globalMin
        self.__globalMax = params.globalMax
        self.vrpLineEditMin.setEnabled(self.radioButtonManual.isChecked())
        self.vrpLineEditMax.setEnabled(self.radioButtonManual.isChecked())
        self.__dependantKeys[key] = params.dependantKeys
        self.blockSignals(False)

        for widget in applyWidgets:
            widget.blockSignals(False)
                    
        # REM ColorMapCustomBase.update(self)

    def changedRadioButtonGroup(self):
        if self.__callerKey == None:
            return

        oldParams = ObjectMgr().getParamsOfObject(self.__currentKey)
        currentParams = self.__getParams()
        
        # if clicked on same radio button
        if oldParams.mode == currentParams.mode:
            return
        
        # set params in negotiator, object manager and own panel
        self.setParams(self.__currentKey, currentParams)
        ObjectMgr().setParams(self.__currentKey, currentParams)
        theGuiMsgHandler().setParams( self.__currentKey, currentParams)

        oldMin = 0
        oldMax = 0
        currentMin = 0
        currentMax = 0

        if oldParams.mode == coColorTableParams.FREE:
            oldMin = oldParams.min
            oldMax = oldParams.max
        elif oldParams.mode == coColorTableParams.LOCAL:
            oldMin = oldParams.baseMin
            oldMax = oldParams.baseMax
        elif oldParams.mode == coColorTableParams.GLOBAL:
            oldMin = oldParams.globalMin
            oldMax = oldParams.globalMax
        else:
            print("Error: Unknown radio button in ColorManager!")
            
        if currentParams.mode == coColorTableParams.FREE:
            currentMin = currentParams.min
            currentMax = currentParams.max
        elif currentParams.mode == coColorTableParams.LOCAL:
            currentMin = currentParams.baseMin
            currentMax = currentParams.baseMax
        elif currentParams.mode == coColorTableParams.GLOBAL:
            currentMin = currentParams.globalMin
            currentMax = currentParams.globalMax
        else:
            print("Error: Unknown radio button in ColorManager!")
        
        # only execute in case of different min/max values
        if oldMin != currentMin or oldMax != currentMax:
            theGuiMsgHandler().runObject(self.__currentKey)
        
    def applyParams(self, val=0):
        #store current params
        if self.__callerKey==None:
            return
        
        self.setParams( self.__currentKey, self.__getParams() )
        ObjectMgr().setParams(self.__currentKey, self.__getParams())
        theGuiMsgHandler().setParams( self.__currentKey, self.__getParams())
        if self.vrpCheckBoxShow.isChecked(): theGuiMsgHandler().runObject( self.__currentKey )
        
    def setNewKey( self, key):
        self.__newkey = key

    def copy( self ):
        params = self.__getParams()
        reqid = theGuiMsgHandler().requestObject( TYPE_COLOR_TABLE, self.__colorMgrKey )
        theGuiMsgHandler().waitforAnswer(reqid)
        oldname = params.name
        params.name = self.__tr('Copy of ') + params.name
        self.setParams( self.__newkey, params )
        theGuiMsgHandler().setParams( self.__newkey, params )
        for key in self.__dependantKeys[self.__currentKey]:
            if not key==self.__callerKey:
                reqid = theGuiMsgHandler().requestParams( key )
                theGuiMsgHandler().waitforAnswer(reqid)
        reqid = theGuiMsgHandler().requestParams( self.__callerKey )
        theGuiMsgHandler().waitforAnswer(reqid)
        self.emitSelectColormap( self.__newkey )

    def __tr(self,s,c = None):
        return coTranslate(s)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QDialog()
    ui = ColorManager()
    ui.setupManagerUi(Form)
    Form.show()
    sys.exit(app.exec_())
# eof
