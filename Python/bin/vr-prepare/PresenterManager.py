
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH

import time

from PyQt5 import QtCore, QtGui, QtWidgets

from printing import InfoPrintCapable
import MainWindow
import Application
from coPresentationMgr import coPresentationMgrParams

from Gui2Neg import theGuiMsgHandler
from PresenterManagerBase import Ui_PresenterManagerBase
from KeydObject import TYPE_COLOR_TABLE
from Utils import CopyParams, NewViewpointAsker, CopyParams
from ObjectMgr import ObjectMgr, GUI_PARAM_CHANGED_SIGNAL
from PatienceDialogManager import PatienceDialogManager
from LogitechDialog import LogitechDialog

from KeydObject import TYPE_PRESENTATION_STEP, TYPE_VIEWPOINT

from ListManager import ListManager

_infoer = InfoPrintCapable()
_infoer.doPrint =  False #True #

from vtrans import coTranslate 

class PresenterManagerBase(QtWidgets.QWidget, Ui_PresenterManagerBase):
    def __init__(self, parent=None, f=QtCore.Qt.WindowFlags()):
        QtWidgets.QWidget.__init__(self, parent, f)
        self.setupUi(self)


class PresenterManager(ListManager):

    """ Handling of presentation steps """
    def __init__(self, parent ):
        _infoer.function = str(self.__init__)
        _infoer.write("")
        ListManager.__init__(self, parent)
        self.setWidget( PresenterManagerBase() )
        self.setWindowTitle(self.__tr("Presentation Manager"))

        #connection of the DockWidget visibilityChanged
        self.visibilityChanged.connect(self.visibilityChangedx)

        #connections of the buttons
        self.widget().NameLE.returnPressed.connect(self.changedParams)
        self.widget().TimeoutSpinBox.valueChanged.connect(self.changedParams)
        self.widget().listBox2.itemClicked.connect(self.listViewClick)
        self.widget().UpButton.clicked.connect(self.up)
        self.widget().DownButton.clicked.connect(self.down)
        self.widget().BackButton.clicked.connect(self.backward)
        self.widget().ForwardButton.clicked.connect(self.forward)
        self.widget().ToEndButton.clicked.connect(self.goToEnd)
        self.widget().ToStartButton.clicked.connect(self.goToStart)
        self.widget().StopButton.clicked.connect(self.stop)
        self.widget().PlayButton.clicked.connect(self.play)
        self.widget().NewButton.clicked.connect(self.new) 
        self.widget().DeleteButton.clicked.connect(self.delete)
        self.widget().ChangeButton.clicked.connect(self.change)
        self.widget().LogitechButton.clicked.connect(self.startLogitech)

        #connect to cover
        theGuiMsgHandler().registerKeyWordCallback('PRESENTATION_PLAY', self.play)
        theGuiMsgHandler().registerKeyWordCallback('PRESENTATION_STOP', self.stop)
        theGuiMsgHandler().registerKeyWordCallback('PRESENTATION_RELOAD', self.reload)
        theGuiMsgHandler().registerKeyWordCallback('PRESENTATION_BACKWARD', self.backward)
        theGuiMsgHandler().registerKeyWordCallback('PRESENTATION_FORWARD', self.forward)
        theGuiMsgHandler().registerKeyWordCallback('PRESENTATION_GO_TO_END', self.goToEnd)
        theGuiMsgHandler().registerKeyWordCallback('PRESENTATION_GO_TO_START', self.goToStart)
        theGuiMsgHandler().registerKeyWordCallback('PRESENTATION_SET_ID', self.goToPresentationPoint)

        #default settings
        #disable the name, timouot - no step is selected or even created
        self.widget().NameLE.setEnabled(False)
        self.widget().TimeoutSpinBox.setEnabled(False)
        self._enablePresenter(False)
        self.widget().textRunning.hide()

        # flags if new viewpoint should be created every time new step is created
        self._decisionSaved = False
        self._decision = True

        # param changed signal from object manager
        ObjectMgr().sigGuiParamChanged.connect( self.paramChangedFromNeg)

    def visibilityChangedx(self, visibility):
        _infoer.function = str(self.visibilityChangedx)
        _infoer.write("")
        if Application.vrpApp.mw:
            Application.vrpApp.mw.windowPresenter_ManagerAction.setChecked(self.isVisible()) # don't use visibility !! (see below)
        # If the DockWidget is displayed tabbed with other DockWidgets and the tab becomes inactive, visiblityChanged(false) is called.
        # Using visibility instead of self.isVisible() this would uncheck the menuentry and hide the DockWidget (including the tab).

    def newViewpoint(self, new):
        _infoer.function = str(self.newViewpoint)
        _infoer.write("")
        self._decision = new
        self._decisionSaved = True

    #change document values for all presentationsteps
    def change(self):
        _infoer.function = str(self.change)
        _infoer.write("")
        # add asker
        msgBox = QtWidgets.QMessageBox(Application.vrpApp.mw)
        msgBox.setWindowTitle(self.__tr("Save changes of presentation step"))
        msgBox.setText(self.__tr("Do you want to save your changes for this presentation step?"))
        msgBox.setStandardButtons(QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Cancel)
        msgBox.setDefaultButton(QtWidgets.QMessageBox.Save)
        acceptedOrRejected = msgBox.exec_()
        if acceptedOrRejected == QtWidgets.QMessageBox.Save:
            # comunication with negotiator
            theGuiMsgHandler().sendKeyWord("changePresentationStep")

    def _getParams(self):
        _infoer.function = str(self._getParams)
        _infoer.write("")
        params = coPresentationMgrParams()
        params.currentKey = self._currentKey
        params.reloadStep = self._reloadStep
        params.currentStep = self.widget().listBox2.currentRow()
        return params

    def _setParams(self, params):
        _infoer.function = str(self._setParams)
        _infoer.write("")
        self._currentKey = params.currentKey

    def setPresentationMgrKey( self, key):
        _infoer.function = str(self.setPresentationMgrKey)
        _infoer.write("")
        self._key = key

    #updates the name and timeout of the current step
    def _updateParamEdits( self ):
        _infoer.function = str(self._updateParamEdits)
        _infoer.write("")
        ListManager._updateParamEdits(self)
        self.widget().TimeoutSpinBox.setEnabled(True)
        self.widget().TimeoutSpinBox.setValue( self._key2Params[self._currentKey].timeout )

    #select the step before the current one
    def backward(self):
        _infoer.function = str(self.backward)
        _infoer.write("")
        index = self.widget().listBox2.currentRow()
        if index>0:
            self.selectItem(index-1)
            self.sendMgrParams()

    #select the step after the current one
    def forward(self):
        _infoer.function = str(self.forward)
        _infoer.write("")
        index = self.widget().listBox2.currentRow()
        if index<self.widget().listBox2.count()-1:
            self.selectItem(index+1)
            self.sendMgrParams()
            return True
        elif (index == self.widget().listBox2.count()-1) and (self.widget().listBox2.count() > 0):
            self.selectItem(0)
            self.sendMgrParams()
            return True
        return False

    def reload(self):
        _infoer.function = str(self.reload)
        _infoer.write("")
        index = self.widget().listBox2.currentRow()
        self._reloadStep = not self._reloadStep
        self.selectItem(index)
        self.sendMgrParams()
        return True

    def goToEnd(self):
        _infoer.function = str(self.goToEnd)
        _infoer.write("")
        if self.widget().listBox2.count() > 0:
            self.selectItem(self.widget().listBox2.count()-1)
            self.sendMgrParams()

    def goToStart(self):
        _infoer.function = str(self.goToStart)
        _infoer.write("")
        if self.widget().listBox2.count() > 0:
            self.selectItem(0)
            self.sendMgrParams()

    def goToPresentationPoint(self, pid):
        _infoer.function = str(self.goToPresentationPoint)
        _infoer.write("")
        if self.widget().listBox2.count() >=pid:
            self.selectItem(pid)
            self.sendMgrParams()

    #stop the animation of steps and return to the first step
    def stop(self):
        _infoer.function = str(self.stop)
        _infoer.write("")
        self._animate = False
        if self._currentKey in self._key2ListBoxItemIdx:
            self.selectItem( self._key2ListBoxItemIdx[self._currentKey] )
        self.widget().textRunning.hide()

    def animate(self):
        _infoer.function = str(self.animate)
        _infoer.write("")
        if self._animate :
            if self.forward():
                self._timer.start( self._key2Params[self._currentKey].timeout*1000 )

    #start the animation of the steps
    def play(self):
        _infoer.function = str(self.play)
        _infoer.write("")
        if len(self._key2ListBoxItemIdx) > 0:
            if not self._currentKey in self._key2ListBoxItemIdx:
                self.selectItem(0)
                self.sendMgrParams()
            self._animate = True
            self._timer = QtCore.QTimer(self)
            self._timer.timeout.connect(self.animate)
            self._timer.start( self._key2Params[self._currentKey].timeout*1000 )
            self.widget().textRunning.show()
        else:
            self._animate = False
            self.widget().textRunning.hide()

    #create new presentation point
    def new(self):
        _infoer.function = str(self.new)
        _infoer.write("")
        createViewpoint = False
        # open asker viewpoint if not decission saved
        if not self._decisionSaved:
            self._decision = False
            # raise the asker dialogue
            asker = NewViewpointAsker(Application.vrpApp.mw) # parent has to be MainWindow because PresenterManager can be invisible (will crash upon return)
            decision = asker.exec_()
            # no cancel pressed
            if decision == QtWidgets.QDialog.Accepted:
                self._decisionSaved = asker.getDecision()
                if asker.pressedYes():
                    createViewpoint = True
                    if self._decisionSaved:
                        self._decision = True
                        Application.vrpApp.mw.presentationNewViewpointAction.setChecked(True)
                else :
                    if self._decisionSaved:
                        self._decision = False
                        Application.vrpApp.mw.presentationNewViewpointAction.setChecked(False)
            else:
                return
        # decision already saved
        else:
            if self._decision:
                createViewpoint = True

        if createViewpoint:
            Application.vrpApp.mw.addViewpoint(True) # in this case, the presentation step will atomatically be created after the viewpoint was created
        else:
            self.requestPresentationStep()

    def requestPresentationStep(self):
        reqid = theGuiMsgHandler().requestObject( TYPE_PRESENTATION_STEP, None, self._key)
        theGuiMsgHandler().waitforAnswer(reqid)

    #clear the list and rebuild it
    def _updateList( self ):
        _infoer.function = str(self._updateList)
        _infoer.write("")
        #clear the listbox
        self.widget().listBox2.clear()
        self.widget().TimeoutSpinBox.blockSignals(True)
        #put all steps into the listbox
        for index in self._ListBoxItemIdx2Key:
            key = self._ListBoxItemIdx2Key[index]
            #if not presentationMgr and not already deleted
            if not key==self._key:
                params = self._key2Params[key]
                # add a number to the name if it is a new point
                # remember this key to select the new point
                if not params.nameChanged:
                    params.name = params.name+str(self._nextIdx)
                    params.nameChanged = True
                    self._key2Params[key] = params
                    self._currentKey = key
                #insert into listview
                self.widget().listBox2.addItem( params.name )
        #select the current step
        if self._currentKey in self._key2ListBoxItemIdx:
            self.selectItem( self._key2ListBoxItemIdx[self._currentKey] )
        else:
            self.widget().NameLE.setEnabled(False)
            self.widget().NameLE.setText('')
            self.widget().TimeoutSpinBox.setEnabled(False)
            self.widget().TimeoutSpinBox.setValue(0)
        # if no point, disable all buttons
        if len(self._key2ListBoxItemIdx) == 0:
            self._enablePresenter(False)
            Application.vrpApp.mw.enablePresenter(False)
        else:
            self._enablePresenter(True)
            Application.vrpApp.mw.enablePresenter(True)
        self.widget().TimeoutSpinBox.blockSignals(False)

    #set the params of a step
    def setParams( self, key, params ):
        if params:
            p = CopyParams(params)
            if not hasattr(p, 'index') or p.index == -1:
                #save index form list
                p.index = self._key2ListBoxItemIdx[key]
                self._ListBoxItemIdx2Key[p.index] = key
            else:
                # Note: The following does not work properly if the key is already present in the list and the index changed.
                #       However, this shouldn't be a problem since the index doesn't change outside the ListManager at the moment.
                #set the index in the list ... insert at this point
                # if key is allready in list, change the order
                if p.index in self._ListBoxItemIdx2Key and key!= self._ListBoxItemIdx2Key[p.index]:
                    # remeber the old key of this index
                    tmpKey = self._ListBoxItemIdx2Key[p.index]
                    # set the new key
                    self._ListBoxItemIdx2Key[p.index] = key
                    self._key2ListBoxItemIdx[key] = p.index
                    # go through the following and delete the old index of this key
                    for i in range(p.index+1, self.widget().listBox2.count()+1):
                        if i in self._ListBoxItemIdx2Key:
                            tmp2Key = self._ListBoxItemIdx2Key[i]
                            if tmp2Key == key:
                                del self._ListBoxItemIdx2Key[i]
                                break
                # key is not in list
                else:
                    self._key2ListBoxItemIdx[key] = p.index
                    self._ListBoxItemIdx2Key[p.index] = key
            self._key2Params[key] = p
            self._updateList()
            ObjectMgr().setParams( key, p )

    def paramChanged(self, key):
        """ params of object key changed"""
        _infoer.function = str(self.paramChanged)
        _infoer.write("")
        if self._key==key:
            self.update()
        if key in self._key2Params:
            self._key2Params[key] = ObjectMgr().getParamsOfObject(key)
            self._updateList()

    # presentation step params changed
    def changedParams(self, val=0):
        _infoer.function = str(self.changedParams)
        _infoer.write("")
        param = self._key2Params[self._currentKey]
        param.name = str( self.widget().NameLE.text() )
        param.timeout = self.widget().TimeoutSpinBox.value()
        param.index = self._key2ListBoxItemIdx[self._currentKey]
        self.setParams( self._currentKey, param )
        theGuiMsgHandler().setParams( self._currentKey, self._key2Params[self._currentKey] )


    def paramChangedFromNeg(self, key):
        _infoer.function = str(self.paramChangedFromNeg)
        _infoer.write("key %s, selfKey %s" % (key, self._key))
        if self._key==key:
            if self._currentKey in self._key2ListBoxItemIdx:
                self.widget().listBox2.blockSignals(True)
                self.widget().listBox2.setCurrentRow( self._key2ListBoxItemIdx[self._currentKey] )
                self._updateParamEdits()
                self.widget().listBox2.blockSignals(False)

    def startLogitech(self):
        _infoer.function = str(self.startLogitech)
        _infoer.write("")
        dialog = LogitechDialog(self)
        dialog.exec_()

    def _enablePresenter(self, b):
        _infoer.function = str(self._enablePresenter)
        _infoer.write("")
        self.widget().DeleteButton.setEnabled(b)
        self.widget().UpButton.setEnabled(b)
        self.widget().DownButton.setEnabled(b)
        self.widget().BackButton.setEnabled(b)
        self.widget().PlayButton.setEnabled(b)
        self.widget().StopButton.setEnabled(b)
        self.widget().ForwardButton.setEnabled(b)
        self.widget().ToEndButton.setEnabled(b)
        self.widget().ToStartButton.setEnabled(b)

    def __tr(self,s,c = None):
        return coTranslate(s)


    def __str__(self):
        ret = "Presenter\n"
        ret = ret + str(self._key2Params[self._key].__dict__)
        ret = ret + '\n' + str(self._key2Params)
        return ret

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QDockWidget()
    ui = PresenterManager(Form)
    # ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())

# eof
