
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH
#
# Base class for the two visualization panels

from PyQt5 import QtCore, QtGui

from ImportGroupManager import COMPOSED_VELOCITY
from vrpconstants import VECTOR3DVARIABLE
import covise

class VisualizationPanel:
    
    def __init__(self):
        #list of the variables
        self.__vectorVariableNames = []
        self.__scalarVariableNames = []
        self.__scalarDecoration = '(scalar)'
        self.__vectorDecoration = '(vector)'
        self.__unsetDecoration = ''
        self.__postfixSeperator = ' '
        #dictionaries to enable the buttons
        self._enableDictComposedMode = {}
        self._enableDictVectorVariable = {}
        self._enableDictScalarVariable = {}
        self._enableDictUnsetVariable = {}
        
        #list of disabled buttons
        self._disablees = []
        self.__inFixedGridMode = False
        self.__useUnset = False
                
        self.vrpComboBoxGrid.hide()
        self.vrpLocalisationLabel.show()        

        #set texts of push buttons
        NewArrowsCuttingSurfaceText = covise.getCoConfigEntry("vr-prepare.NewArrowsCuttingSurfaceText")
        if NewArrowsCuttingSurfaceText and hasattr(self, 'CuttingSurfaceArrowPushButton') :
            self.CuttingSurfaceArrowPushButton.setText(NewArrowsCuttingSurfaceText)
        NewColoredCuttingSurfaceText = covise.getCoConfigEntry("vr-prepare.NewColoredCuttingSurfaceText")
        if NewColoredCuttingSurfaceText:
            self.CuttingSurfaceColoredPushButton.setText(NewColoredCuttingSurfaceText)
           
        self._disableMethodButts()

    def currentVariable(self):
        return self.__undecorateName(str(self.vrpComboBoxVariable.currentText()))
        
    def getParams(self):
        class A(object):
            pass
        a = A()
        a.variableName = self.currentVariable()
        if self.__isVectorVariable(a.variableName):
            a.variableDimension = VECTOR3DVARIABLE
        elif self.__isScalarVariable(a.variableName):
            a.variableDimension = SCALARVARIABLE
        return a

    def _setVectorVariables(self, aNameList):
        self.__vectorVariableNames = aNameList
        self.__fillVariables()

    def _setScalarVariables(self, aNameList):
        self.__scalarVariableNames = aNameList
        self.__fillVariables()

    def _setUnsetVariable(self, useUnset):
        self.__useUnset = useUnset
        self.__fillVariables()    

    def __isVectorVariable(self, var):
        return var in self.__vectorVariableNames

    def __isScalarVariable(self, var):
        return var in self.__scalarVariableNames

    def __isUnsetVariable(self, var):
        return var == 'unset'

    def __fillVariables(self):
        self.vrpComboBoxVariable.clear() 
        if self.__useUnset:
            name = self.__decorateName('unset', self.__unsetDecoration)
            self.vrpComboBoxVariable.addItem(name)      
        for aName in self.__vectorVariableNames:
            aName = self.__decorateName(aName, self.__vectorDecoration)            
            self.vrpComboBoxVariable.addItem(aName)
        for aName in self.__scalarVariableNames:
            aName = self.__decorateName(aName, self.__scalarDecoration)            
            self.vrpComboBoxVariable.addItem(aName)
        if len(self.__vectorVariableNames)+len(self.__scalarVariableNames)>0:
            self.__enableMethodButtsForVariable(self.currentVariable())
        else:
            self._disableMethodButts()

    def _enableMethodButtsForVariableSlot(self, decoratedVar):
        var = str(decoratedVar)
        var = self.__undecorateName(var)
        self.__enableMethodButtsForVariable(var)

    def __enableMethodButtsForVariable(self, var):
        if COMPOSED_VELOCITY == var:
            for butt in self._enableDictComposedMode: butt.setEnabled(self._enableDictComposedMode[butt])
        elif self.__isVectorVariable(var):
            for butt in self._enableDictVectorVariable: butt.setEnabled(self._enableDictVectorVariable[butt])
        elif self.__isScalarVariable(var):
            for butt in self._enableDictScalarVariable: butt.setEnabled(self._enableDictScalarVariable[butt])
        elif self.__isUnsetVariable(var):
            for butt in self._enableDictUnsetVariable: butt.setEnabled(self._enableDictUnsetVariable[butt])
        else:
            assert False, 'unexpected variable "%s"' % var
        self._disableBrokenParts()

    def _disableMethodButts(self):
        for butt in self._enableDictComposedMode: butt.setEnabled(False)
        for butt in self._enableDictVectorVariable: butt.setEnabled(False)
        for butt in self._enableDictScalarVariable: butt.setEnabled(False)
        self._disableBrokenParts()

    def _disableBrokenParts(self):
        """TODO: This function must vanish.

        Enable step by step by adding wanted functionality.
        """
        for disablee in self._disablees:
            disablee.setEnabled(False)
            disablee.hide()


    def __decorateName(self, aString, postfix):
        if aString == 'unset':
            return 'None'
        else: 
            return aString + self.__postfixSeperator + postfix
        
    def __undecorateName(self, aString):
        if aString == 'None':
            return 'unset'
        else:
            return aString[0:aString.rfind(self.__postfixSeperator)]

    def isGridFixed(self):
        return self.__inFixedGridMode

    def gridName(self):
        """Name of the grid that is the base for the visualization object."""
        return str(self.vrpLocalisationLabel.text())

    def setGridFixed(self, b):
        if b == self.isGridFixed(): return
        if b:
            self.vrpComboBoxGrid.hide()
            self.vrpLocalisationLabel.show()
        else:
            self.vrpComboBoxGrid.show()
            self.vrpLocalisationLabel.hide()
        self.__inFixedGridMode = b

    def setGridName(self, aName):
        self.vrpLocalisationLabel.setText(aName)
        
 
        

