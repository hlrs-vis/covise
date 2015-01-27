import sys

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject, pyqtSignal

import sip

import Application

from SceneObjectPanelBase import Ui_SceneObjectPanelBase
from SceneObjectPanelConnector import SceneObjectPanelConnector, SceneObjectPanelBlockSignals
from ColorChooser import getColor

from SceneObjectVis import SceneObjectVisParams
from Gui2Neg import theGuiMsgHandler
from ObjectMgr import ObjectMgr
from KeydObject import VIS_VRML
from Utils import ParamsDiff
import copy

from printing import InfoPrintCapable


_infoer = InfoPrintCapable()
_infoer.doPrint = False #True #

class SceneObjectPanel(QtWidgets.QWidget,Ui_SceneObjectPanelBase):
    def __init__(self, parent=None):
        _infoer.function = str(self.__init__)
        _infoer.write("")
        QtWidgets.QWidget.__init__(self, parent)
        Ui_SceneObjectPanelBase.__init__(self)
        self.setupUi(self)

        # list of associated keys from same type
        self.__keys = []

        # list of hidden tabs
        self.__hiddenTabs = {}

        # variants
        self._variantCombos = {}
        self._previous_variant_groups = {}

        # appearance
        self._colorButtons = {}
        self._storedColors = {} # colors have to be stored manually because they are not displayed anywhere

        # for multi selection
        self.oldPanelParams = {}

        # hold the parameters of object. necessary, because not all parameters are stored in Panel widgets
        self.__params = SceneObjectVisParams()

        self.__firstTime = True

        SceneObjectPanelConnector(self)


    def paramChanged(self, key):
        """ params of object key changed"""

        _infoer.function = str(self.paramChanged)
        _infoer.write("key %d" %key)

        #update only for single selection
        if len(self.__keys) ==1 :
            if self.__keys[0]==key or (self.__keys[0] in Application.vrpApp.guiKey2visuKey and key==Application.vrpApp.guiKey2visuKey[self.__keys[0]]):
                self.update()

    def update(self):
        _infoer.function = str(self.update)
        _infoer.write("")
        if len(self.__keys)!=0 :
            self.updateForObject( self.__keys )

    def updateForObject( self, keys ):
        """ called from MainWindow to update the content to the choosen object key

        Default params will be shown if there is more than 1 key"""
        _infoer.function = str(self.updateForObject)
        _infoer.write("")


        if isinstance( keys, int ) :
            self.__keys = [keys]
        elif isinstance( keys, list ) and len(keys)>0 :
            self.__keys = keys
        else :
            return

        if len(self.__keys) == 1 :
            params = ObjectMgr().getParamsOfObject(self.__keys[0])
            if isinstance(params, int) or isinstance(params, SceneObjectVisParams):
                self.__setParams( params )
        elif len(self.__keys) > 1 :
            # multi selection: show default params
            self.oldPanelParams = SceneObjectVisParams()
            params = CopyParams(self.oldPanelParams)
            params.name = "Multiselection"
            self.__setParams( params )

    def __getParams(self):
        _infoer.function = str(self.__getParams)
        _infoer.write("")

        data = CopyParams(self.__params)    # use parameter values of this object that arent in this panel

        # height / width / length
        try:
            data.width = float(str(self.lineEdit_width.text()))*10.0
        except:
            pass
        try:
            data.height = float(str(self.lineEdit_height.text()))*10.0
        except:
            pass
        try:
            data.length = float(str(self.lineEdit_length.text()))*10.0
        except:
            pass


        # transX / transY / transZ
        try:
            data.transX = float(str(self.lineEdit_trans_x.text()))*10.0
        except:
            pass
        try:
            data.transY = float(str(self.lineEdit_trans_y.text()))*10.0
        except:
            pass
        try:
            data.transZ = float(str(self.lineEdit_trans_z.text()))*10.0
        except:
            pass

        # variants
        for (groupName, (label, combo)) in iter(self._variantCombos.items()):
            data.variant_selected[groupName] = unicode(combo.currentText())

        # colors
        data.appearance_colors = copy.deepcopy(self._storedColors)

        self.__params = CopyParams(data)

        return data


    def __setParams( self, params ):
        _infoer.function = str(self.__setParams)
        _infoer.write("")

        self.__params = CopyParams(params)

        if isinstance( params, int):
            self.__keys[0] = params
            return

        SceneObjectPanelBlockSignals(self, True)

        # name

        self.label_name.setText(params.name)
        
        # description
        self.groupDescription.setVisible(params.description != "")
        self.editDescription.setPlainText(params.description)

        # width / height / length

        self.label_width.setVisible(params.width != None)
        self.label_width_unit.setVisible(params.width != None)
        self.lineEdit_width.setVisible(params.width != None)
        if (params.width == None):
            self.lineEdit_width.setText("")
        else:
            self.lineEdit_width.setText(str(params.width*0.1))

        self.label_height.setVisible(params.height != None)
        self.label_height_unit.setVisible(params.height != None)
        self.lineEdit_height.setVisible(params.height != None)
        if (params.height == None):
            self.lineEdit_height.setText("")
        else:
            self.lineEdit_height.setText(str(params.height*0.1))

        self.label_length.setVisible(params.length != None)
        self.label_length_unit.setVisible(params.length != None)
        self.lineEdit_length.setVisible(params.length != None)
        if (params.length == None):
            self.lineEdit_length.setText("")
        else:
            self.lineEdit_length.setText(str(params.length*0.1))

        # transX / transY / transZ

        self.lineEdit_trans_x.setText(str(params.transX*0.1))
        self.lineEdit_trans_y.setText(str(params.transY*0.1))
        self.lineEdit_trans_z.setText(str(params.transZ*0.1))

        # VariantBehavior
        
        if ("VariantBehavior" in params.behaviors):
            self.groupVariant.setVisible(True)
            # check if variant groups changed
            set1 = tuple([(a, tuple(b)) for a,b in iter(params.variant_groups.items())])
            set2 = tuple([(a, tuple(b)) for a,b in iter(self._previous_variant_groups.items())])
            if (set1 != set2):
                # delete old combos and layout (if present)
                if hasattr(self, "_variantsLayout"):
                    for label, combo in self._variantCombos.values():
                        self._variantsLayout.removeWidget(label)
                        label.setParent(None)
                        sip.delete(label)
                        self._variantsLayout.removeWidget(combo)
                        combo.setParent(None)
                        sip.delete(combo)
                    sip.delete(self._variantsLayout) # sip properly removes the underlying C object so another layout can be added to the widget
                self._variantCombos = {}
                # create new layout
                self._variantsLayout = QtWidgets.QGridLayout(self.groupVariant)
                # create all combos
                cnt = 0
                for groupName, variants in iter(params.variant_groups.items()):
                    label = QtWidgets.QLabel(self.groupVariant)
                    label.setText(groupName)
                    self._variantsLayout.addWidget(label, cnt, 0)
                    combo = QtWidgets.QComboBox(self.groupVariant)
                    for var in variants:
                        combo.addItem(var)
                    self._variantsLayout.addWidget(combo, cnt, 1)
                    combo.activated.connect(self.variantChanged)
                    combo.activated.connect(self.variantChanged)
                    self._variantCombos[groupName] = (label, combo)
                    cnt = cnt + 1
                # add spacer
                spacer = QtWidgets.QSpacerItem(16, 16, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
                self._variantsLayout.addItem(spacer)
                # set previous
                self._previous_variant_groups = copy.deepcopy(params.variant_groups)
            # set value
            for (groupName, (label, combo)) in iter(self._variantCombos.items()):
                if (groupName in params.variant_selected):
                    index = combo.findText(params.variant_selected[groupName])
                    if (index > -1):
                        combo.blockSignals(True)
                        combo.setCurrentIndex(index)
                        combo.blockSignals(False)
        else:
            self.groupVariant.setVisible(False)

        # AppearanceBehavior

        self._storedColors = copy.deepcopy(params.appearance_colors) # colors have to be stored manually because there is no gui element displaying the color
        if ("AppearanceBehavior" in params.behaviors):
            # check if scopes changed
            set1 = tuple(params.appearance_colors.keys())
            set2 = tuple(self._colorButtons.keys())
            if (set1 != set2):
                # delete old buttons and layout (if present)
                if hasattr(self, "_appearanceLayout"):
                    for button in self._colorButtons.values():
                        self._appearanceLayout.removeWidget(button)
                        button.setParent(None)
                        sip.delete(button)
                    sip.delete(self._appearanceLayout) # sip properly removes the underlying C object so another layout can be added to the widget
                self._colorButtons = {}
                # create new layout
                self._appearanceLayout = QtWidgets.QGridLayout(self.groupAppearance)
                # create all buttons
                cnt = 0
                for scopeName in params.appearance_colors.keys():
                    button = QtWidgets.QPushButton(self.groupAppearance)
                    if (scopeName == ""):
                        button.setText("Color")
                    else:
                        button.setText(scopeName)
                    self._appearanceLayout.addWidget(button, cnt, 0)
                    button.clicked.connect(self.colorClicked)
                    self._colorButtons[scopeName] = button
                    cnt = cnt + 1
                # add spacer
                spacer = QtWidgets.QSpacerItem(16, 16, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
                self._appearanceLayout.addItem(spacer)
                # set group box visible
                print("setVisible", len(self._colorButtons))
            self.groupAppearance.setVisible(len(self._colorButtons) > 0)
        else:
            self.groupAppearance.setVisible(False)

        SceneObjectPanelBlockSignals(self, False)

        # for multi selection
        if len(self.__keys)>1 :
            self.oldPanelParams = params


    def variantChanged(self, text):
        self.emitDataChanged()

    def colorClicked(self):
        inverse = dict(zip(self._colorButtons.values(), self._colorButtons.keys()))
        if (self.sender() not in inverse.keys()):
            return
        scopeName = inverse[self.sender()]
        if (scopeName not in self._storedColors):
            return
        
        palette = ""
        if (scopeName in self.__params.appearance_palettes):
            palette = self.__params.appearance_palettes[scopeName]
            
        color = getColor(parent=self, color=self._storedColors[scopeName], palette=palette)
        if (color == None):
            return

        self._storedColors[scopeName] = color
        self.emitDataChanged()

    def emitDataChanged(self):
        _infoer.function = str(self.emitDataChanged)
        _infoer.write("")

        params = self.__getParams()
        if len(self.__keys)==1 :
            Application.vrpApp.key2params[self.__keys[0]] = params
            ObjectMgr().setParams( self.__keys[0], params )

        # set params for multi selection
        if len(self.__keys)>1 :
            #find changed params
            originalParams = self.oldPanelParams
            realChange = ParamsDiff( originalParams, params )

            # set params for remaining selected objects
            for i in range(0, len(self.__keys)):
                keyParams = ObjectMgr().getParamsOfObject(self.__keys[i])

                # find the changed param in childKey and replace it with the
                # intended attribut
                for x in realChange :
                    keyParams.__dict__[x] = params.__dict__[x]
                # set the params
                Application.vrpApp.key2params[self.__keys[i]] = keyParams
                ObjectMgr().setParams( self.__keys[i], keyParams )
                #theGuiMsgHandler().runObject( childKeys[i] )
            #save params for multi selection
            self.oldPanelParams = self.__getParams()




