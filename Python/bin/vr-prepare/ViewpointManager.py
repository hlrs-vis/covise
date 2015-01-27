
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH

from PyQt5 import QtCore, QtGui, QtWidgets

import os

from printing import InfoPrintCapable
import MainWindow
import Application
from coViewpointMgr import coViewpointMgrParams, coViewpointParams

from Gui2Neg import theGuiMsgHandler
from ViewpointManagerBase import Ui_ViewpointManagerBase
from ListManager import ListManager
from Utils import CopyParams, ReallyWantToOverrideAsker
from ObjectMgr import ObjectMgr, GUI_PARAM_CHANGED_SIGNAL
from coGRMsg import coGRSnapshotMsg, coGRChangeViewpointMsg, coGRTurnTableAnimationMsg
import covise

from KeydObject import globalKeyHandler, TYPE_VIEWPOINT

from printing import InfoPrintCapable
_infoer = InfoPrintCapable()
_infoer.doPrint =  False #True #

from vtrans import coTranslate


class ViewpointManagerBase(QtWidgets.QWidget, Ui_ViewpointManagerBase):
    def __init__(self, parent=None, f=QtCore.Qt.WindowFlags()):
        QtWidgets.QWidget.__init__(self, parent, f)
        self.setupUi(self)

class ViewpointManager(ListManager):

    """ Handling of viewpoints """
    def __init__(self, parent):
        _infoer.function = str(self.__init__)
        _infoer.write("")
        ListManager.__init__(self, parent )
        self.setWidget(ViewpointManagerBase(parent))
        self.setWindowTitle(self.__tr("Viewpoint Manager"))

        #default settings
        #disable the name, timouot - no step is selected or even created
        self.widget().NameLE.setEnabled(False)

        # hide checkbox for clipplaneMode
        self.widget().checkClipplaneMode.hide()

        #connection of the DockWidget visibilityChanged
        self.visibilityChanged.connect(self.visibilityChangedS);
        #connections of the buttons
        self.widget().NameLE.returnPressed.connect(self.changedParams)
        self.widget().listBox2.itemClicked.connect(self.listViewClick)
        self.widget().NewButton.clicked.connect(self.new)
        self.widget().DeleteButton.clicked.connect(self.delete)
        self.widget().ChangeButton.clicked.connect(self.change)
        self.widget().checkFlyingMode.clicked.connect(self.toggleFlyMode)
        self.widget().checkClipplaneMode.clicked.connect(self.toggleClipplaneMode)
        #connection if params changed
        ObjectMgr().sigGuiParamChanged.connect(self.paramChangedFromNeg)

        self._enableViewpoints(False)
        self._addedKey = None
        self._fromPresentation = False
        # for change viewpoint
        self.lastKey = -1 # last chosen viewpoint
        # self._currentKey is the current viewpoint
        self._currentKey = -1
        
        #list of icon-pixmaps for default viewpoints
        self.__iconOfViewpoints = {}
        self.__functions = {}
        self.__iconOfViewpoints['back'] = ":/backView.png"
        self.__functions['back'] = self.showViewpointBack
        self.__iconOfViewpoints['bottom'] = ":/bottomView.png"
        self.__functions['bottom'] = self.showViewpointBottom
        self.__iconOfViewpoints['front'] = ":/frontView.png"
        self.__functions['front'] = self.showViewpointFront        
        self.__iconOfViewpoints['left'] = ":/leftView.png"
        self.__functions['left'] = self.showViewpointLeft      
        self.__iconOfViewpoints['right'] = ":/rightView.png"
        self.__functions['right'] = self.showViewpointRight        
        self.__iconOfViewpoints['halftop'] = ":/someView.png"
        self.__iconOfViewpoints['top'] = ":/topView.png"
        self.__functions['top'] = self.showViewpointTop
        self.__iconOfViewpoints['center'] = ":/center.png"
        #list of buttons
        self.__buttons = {}
        self.__buttonIds = {}

        #grid layout for iconBar
        iconBarLayout = QtWidgets.QGridLayout(self.widget().iconBar) #REM,1,1,5,-1,"iconBarLayout")
        # add iconGroup to iconBar
        self.iconGroup = QtWidgets.QGroupBox(self.widget().iconBar)
        # REM self.iconGroup.setLineWidth(0)
        iconBarLayout.addWidget(self.iconGroup,0,0)

        theGuiMsgHandler().registerKeyWordCallback('VIEW_CHANGED', self.viewChanged)

    def visibilityChangedS(self, visibility):
        if Application.vrpApp.mw:
            Application.vrpApp.mw.windowViewpoint_ManagerAction.setChecked(self.isVisible()) # don't use visibility !! (see below)
        # If the DockWidget is displayed tabbed with other DockWidgets and the tab becomes inactive, visiblityChanged(false) is called.
        # Using visibility instead of self.isVisible() this would uncheck the menuentry and hide the DockWidget (including the tab).

    def viewChanged(self):
        self.setDefaultVPActionsChecked(-1)
        _infoer.function = str(self.viewChanged)
        _infoer.write("")
        
        # deselect a selected key in list view
        if (self._currentKey > -1) and (self._currentKey in self._key2Params):
            self._updateParamEdits()
            params = self._key2Params[self._currentKey]
            params.isVisible = False
            params.selectedKey = None
            ObjectMgr().setParams( self._currentKey, params )
            self.sendMgrParams()

        if self._currentKey != -1:
            # if selection in list is disabled keep the last key (for change viewpoint)
            self.lastKey = self._currentKey
        self._currentKey = -1
        
        self.widget().listBox2.setCurrentRow(-1)
        self.widget().listBox2.clearSelection()

    def _getParams(self):
        _infoer.function = str(self._getParams)
        _infoer.write("%s " %str(self._currentKey) )
        params = coViewpointMgrParams()
        params.currentKey = self._currentKey
        params.flyingMode = self.widget().checkFlyingMode.isChecked()
        params.clipplaneMode = True#self.widget().checkClipplaneMode.isChecked()
        return params

    def _setParams(self, params):
        _infoer.function = str(self._setParams)
        _infoer.write("%s " %(str(params)))
        if hasattr(params, 'currentKey'):
            if params.currentKey in self._key2ListBoxItemIdx:
                self._currentKey = params.currentKey
            else:
                self.setDefaultVPActionsChecked(params.currentKey)
            #if self._currentKey in self._key2ListBoxItemIdx:
            #    self.widget().listBox2.setCurrentItem(self._key2ListBoxItemIdx[self._currentKey])
        if hasattr(params, 'flyingMode'):
            self.widget().checkFlyingMode.setChecked(params.flyingMode)
        #if hasattr(params, 'clipplaneMode'):
        #    self.widget().checkClipplaneMode.setChecked(params.clipplaneMode)

    def getNumViewpoints(self):
        return len(self._key2ListBoxItemIdx)

    def setViewpointMgrKey( self, key):
        _infoer.function = str(self.__init__)
        _infoer.write("key %s" %str(key))
        self._key = key

    def addViewpoint( self, key ):
        _infoer.function = str(self.addViewpoint)
        _infoer.write("key %s " %str(key))
        if (not key  in self._key2ListBoxItemIdx):
            index = self.widget().listBox2.count()
            self._key2ListBoxItemIdx[key] = index
            self._ListBoxItemIdx2Key[index] = key
            self._nextIdx = self._nextIdx+1
            self._addedKey = key

    #called if the list view is clicked in gui
    def listViewClick(self, lbi):
        _infoer.function= str(self.listViewClick)
        _infoer.write("%s " %str(lbi))
        if lbi :
            itemName = str(lbi.text())
            key = self._getKey(self.widget().listBox2.row(lbi))
            self._currentKey=key
            self._updateParamEdits()
            params = self._key2Params[self._currentKey]
            params.isVisible = True
            ObjectMgr().setParams( self._currentKey, params )
            self.sendMgrParams()

    def setFlyMode(self, mode):
        _infoer.function = str(self.setFlyMode)
        _infoer.write("")
        # set the checkbox
        self.widget().checkFlyingMode.setChecked(mode)
        # send MgrParams to negotiator
        self.sendMgrParams()

    def setClipplaneMode(self, mode):
        _infoer.function = str(self.setClipplaneMode)
        _infoer.write("")
        # set the checkbox
        #self.widget().checkClipplaneMode.setChecked(mode)
        # send MgrParams to negotiator
        #self.sendMgrParams()

    def toggleFlyMode(self):
        _infoer.function = str(self.toggleFlyMode)
        _infoer.write("")
        # set flag in menu
        Application.vrpApp.mw.viewpointsFlying_ModeAction.setChecked(self.widget().checkFlyingMode.isChecked())
        # send params to negotiator
        self.sendMgrParams()

    def toggleClipplaneMode(self):
        _infoer.function = str(self.toggleClipplaneMode)
        _infoer.write("")
        # set flag in menu
        # send params to negotiator
        self.sendMgrParams()

    #updates the name of the current step
    def _updateParamEdits( self ):
        _infoer.function = str(self._updateParamEdits)
        _infoer.write("%s " %str(self._currentKey))
        key = self._currentKey
        self.widget().NameLE.setEnabled(False)
        if (key != None) and (key in self._key2Params):
            self.widget().NameLE.setText( self._key2Params[key].name )
            self.widget().NameLE.setEnabled(True)
        else:
            self.widget().NameLE.setText("")

    def selectItem( self, lbIndex ):
        _infoer.function = str(self.selectItem)
        _infoer.write("%s " %str(lbIndex))
        # REM self.widget().listBox2.selectItem( lbIndex )
        self._currentKey = self._getKey(lbIndex)
        self._updateParamEdits()

    #create new viewpoint point
    def new(self, fromPresentation=False):
        _infoer.function = str(self.new)
        _infoer.write("%s " %str(fromPresentation))
        #send msg to covise
        theGuiMsgHandler().sendKeyWord("saveViewPoint")
        self._fromPresentation = fromPresentation

    #change viewpoint point
    def change(self):
        _infoer.function = str(self.change)
        _infoer.write("")
        # add asker
        msgBox = QtWidgets.QMessageBox(Application.vrpApp.mw)
        msgBox.setWindowTitle(self.__tr("Save changed viewpoint"))
        msgBox.setText(self.__tr("Do you want to save this view for the viewpoint?"))
        msgBox.setStandardButtons(QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Cancel)
        msgBox.setDefaultButton(QtWidgets.QMessageBox.Save)
        acceptedOrRejected = msgBox.exec_()
        if acceptedOrRejected == QtWidgets.QMessageBox.Save:
            # take eighter current key or if selection in list is disabled use the latest
            key = -1
            if self._currentKey != -1:
                key = self._currentKey
            elif self.lastKey != -1:
                key = self.lastKey

            if key > -1:
                # get COVER id
                id = ObjectMgr().getParamsOfObject(key).id
                msg = coGRChangeViewpointMsg( id )
                covise.sendRendMsg(msg.c_str())

    #clear the list and rebuild it
    def _updateList( self ):
        _infoer.function = str(self._updateList)
        _infoer.write("%s " %str(self._ListBoxItemIdx2Key))
        #clear the listbox
        self.widget().listBox2.clear()

        #put all steps into the listbox
        for index in self._ListBoxItemIdx2Key:
            key = self._ListBoxItemIdx2Key[index]
            #if not mgr
            if not key==self._key:
                params = self._key2Params[key]
                self._key2Params[key] = params
                #insert into listview
                self.widget().listBox2.addItem( params.name )

        #select the added key
        if self._addedKey:
            # go to viewpoint
            params = ObjectMgr().getParamsOfObject(self._addedKey)
            params.isVisible = True
            if  params.id:
                self._currentKey = self._addedKey
                self._addedKey = None
                self.setParams(self._currentKey, params)
            if self._fromPresentation:
                Application.vrpApp.mw.presenterManager.requestPresentationStep()
                self._fromPresentation = False

        #select the current step
        self.widget().NameLE.setEnabled(False)
        if self._currentKey in self._key2ListBoxItemIdx:
            self.selectItem( self._key2ListBoxItemIdx[self._currentKey] )
            self.widget().NameLE.setEnabled(True)
        else:
            self.widget().NameLE.setText('')
        # if no point, disable all buttons
        if len(self._key2ListBoxItemIdx) == 0:
            self._enableViewpoints(False)
            Application.vrpApp.mw.enableViewpoint(False)
        else:
            self._enableViewpoints(True)
            Application.vrpApp.mw.enableViewpoint(True)

    #set the params of a step
    def setParams( self, key, params ):
        _infoer.function = str(self.setParams)
        _infoer.write("%s %s " %(str(key) ,str(params)))
        if params:
            if params.isVisible:
                self._currentKey = key
            self._key2Params[key] = CopyParams(params)
            if not params.view == 'default':
                self._updateList()
            else:
                self.__buttonIds[params.name] = key
                #add an icon to iconbar
                if not params.name in self.__buttons:
                    # create the new button
                    self.__buttons[params.name] = QtWidgets.QToolButton(self.iconGroup)
                    # position the new button
                    num = len(self.__buttons)-1
                    # self.iconGroup.insert(self.__buttons[params.name], num)
                    self.__buttons[params.name].setGeometry(QtCore.QRect(5+num*30,5,25,25))
                    self.__buttons[params.name].setMaximumSize(QtCore.QSize(25,25))
                    palette = QtGui.QPalette()
                    palette.setColor(self.__buttons[params.name].backgroundRole(), QtGui.QColor(238,234,238))
                    self.__buttons[params.name].setPalette(palette)
                    # self.__buttons[params.name].setCursor(QtGui.QCursor(QtCore.Qt.CursorShape(0)))
                    # set the icon-pixmap or the name
                    if params.name in self.__iconOfViewpoints:
                        self.__buttons[params.name].setIcon(QtGui.QIcon(QtGui.QPixmap(self.__iconOfViewpoints[params.name])))
                    else:
                        self.__buttons[params.name].setText(str(params.name))
                    # set the tooltip
                    self.__buttons[params.name].setToolTip(self.__tr("Viewpoint ")+str(params.name))
                    self.__buttons[params.name].setAutoRaise(1)
                    self.__buttons[params.name].setCheckable(True)
                    self.__buttons[params.name].show()
                    # connect the buttonsGroup of default viewpoints
                    self.__buttons[params.name].clicked.connect(self.__functions[params.name])

    # viewpoint params changed
    def changedParams(self, val=0):
        _infoer.function = str(self.changedParams)
        _infoer.write("%s" %str(self._currentKey))
        if self._currentKey in self._key2Params:
            param = self._key2Params[self._currentKey]
            param.name = str( self.widget().NameLE.text() )
            param.isVisible = True
            self.setParams( self._currentKey, param )
            self.sendMgrParams()
            ObjectMgr().setParams( self._currentKey, self._key2Params[self._currentKey] )

    def paramChangedFromNeg( self, key ):
        _infoer.function = str(self.paramChangedFromNeg)
        _infoer.write("")
        """ params of object key changed"""
        if self._key==key:
            self.update()
        if key in self._key2Params:
            params = ObjectMgr().getParamsOfObject(key)
            self.setParams(key, params)

    def viewAll(self):
        _infoer.function = str(self.viewAll)
        _infoer.write("")
        theGuiMsgHandler().sendKeyWord("viewAll")
               
    def orthographicProjection(self):
        _infoer.function = str(self.orthographicProjection)
        _infoer.write("")
        theGuiMsgHandler().sendKeyWord("orthographicProjection")
    
    def turntableAnimation(self):
        _infoer.function = str(self.turntableAnimation)
        _infoer.write("")
        time = covise.getCoConfigEntry("vr-prepare.TurntableAnimationTime")
        fTime = 10.0
        if (time != None):
            try:
                fTime = float(time)
            except exception.ValueError:
                pass 
        msg = coGRTurnTableAnimationMsg(fTime)
        covise.sendRendMsg(msg.c_str())
        
    def turntableRotate45(self):
        _infoer.function = str(self.turntableRotate45)
        _infoer.write("")
        theGuiMsgHandler().sendKeyWord("turntableRotate45")
        
    def snapAll(self):
        _infoer.function = str(self.snapAll)
        _infoer.write("")
        #theGuiMsgHandler().sendKeyWord("snapAll")
        msg = coGRSnapshotMsg( "", "snapAll" )
        covise.sendRendMsg(msg.c_str())

    def takeSnapshot(self):
        _infoer.function = str(self.takeSnapshot)
        _infoer.write("")
        filename = ""
        if covise.coConfigIsOn("vr-prepare.ShowSnapshotDialog", True):
            directory = covise.getCoConfigEntry("COVER.Plugin.PBufferSnapShot.Directory")
            if (directory == None):
                directory = "snapshot.png"
            else:
                directory = directory + "/" + "snapshot.png"  
            filenameQt = QtWidgets.QFileDialog.getSaveFileName(
                self,
                self.__tr('Snapshot'),
                directory,
                self.__tr('Image (*.png)'),
                None,
                QtWidgets.QFileDialog.DontConfirmOverwrite)
            if filenameQt == "":
                return
            #filenameQt is filename + extension touple
            filename = filenameQt[0]
            print(filename)
            if not filename.lower().endswith(".png"):
                filename += ".png"
            if  os.path.exists(filename):
                asker = ReallyWantToOverrideAsker(self, filename)
                decicion = asker.exec_()
                if decicion == QtWidgets.QDialog.Rejected:
                    self.statusBar().showMessage( self.__tr('Cancelled overwrite of "%s"') % filename )
                    return
        msg = coGRSnapshotMsg( filename, "snapOnce" )
        covise.sendRendMsg(msg.c_str())

    # is called if an icon of the default viewpoints is clicked
    def showViewpoint( self, clickedId ):
        _infoer.function = str(self.showViewpoint)
        _infoer.write("%s" %str(self._currentKey))
        self._currentKey=None
        self._updateParamEdits()
        key = self.__buttonIds[clickedId]
        params = self._key2Params[key]
        params.isVisible = True
        ObjectMgr().setParams( key, params )
        self.sendMgrParams()

    def showViewpointFront( self ):
        _infoer.function = str(self.showViewpoint)
        _infoer.write("%s" %str(self._currentKey))
        self._currentKey=None
        self._updateParamEdits()
        if not 'front' in self.__buttons:
            return
        key = self.__buttonIds['front']
        params = self._key2Params[key]
        params.isVisible = True
        ObjectMgr().setParams( key, params )
        self.sendMgrParams()

    def showViewpointBack( self ):
        _infoer.function = str(self.showViewpoint)
        _infoer.write("%s" %str(self._currentKey))
        self._currentKey=None
        self._updateParamEdits()
        if not 'back' in self.__buttons:
            return
        key = self.__buttonIds['back']
        params = self._key2Params[key]
        params.isVisible = True
        ObjectMgr().setParams( key, params )
        self.sendMgrParams()
        
    def showViewpointLeft( self ):
        _infoer.function = str(self.showViewpoint)
        _infoer.write("%s" %str(self._currentKey))
        self._currentKey=None
        self._updateParamEdits()
        if not 'left' in self.__buttons:
            return        
        key = self.__buttonIds['left']
        params = self._key2Params[key]
        params.isVisible = True
        ObjectMgr().setParams( key, params )
        self.sendMgrParams()
        
    def showViewpointRight( self ):
        _infoer.function = str(self.showViewpoint)
        _infoer.write("%s" %str(self._currentKey))
        self._currentKey=None
        self._updateParamEdits()
        if not 'right' in self.__buttons:
            return
        key = self.__buttonIds['right']
        params = self._key2Params[key]
        params.isVisible = True
        ObjectMgr().setParams( key, params )
        self.sendMgrParams()

    def showViewpointTop( self ):
        _infoer.function = str(self.showViewpoint)
        _infoer.write("%s" %str(self._currentKey))
        self._currentKey=None
        self._updateParamEdits()
        if not 'top' in self.__buttons:
            return        
        key = self.__buttonIds['top']
        params = self._key2Params[key]
        params.isVisible = True
        ObjectMgr().setParams( key, params )
        self.sendMgrParams()

    def showViewpointBottom( self ):
        _infoer.function = str(self.showViewpoint)
        _infoer.write("%s" %str(self._currentKey))
        self._currentKey=None
        self._updateParamEdits()
        if not 'bottom' in self.__buttons:
            return        
        key = self.__buttonIds['bottom']
        params = self._key2Params[key]
        params.isVisible = True
        ObjectMgr().setParams( key, params )
        self.sendMgrParams()


    def _enableViewpoints(self, b):
        _infoer.function = str(self._enableViewpoints)
        _infoer.write("")
        self.widget().DeleteButton.setEnabled(b)
        self.widget().ChangeButton.setEnabled(b)


    def setDefaultVPActionsChecked(self, activeId):
        actions = {}
        actions["front"] = Application.vrpApp.mw.actionFront
        actions["back"] = Application.vrpApp.mw.actionBack
        actions["left"] = Application.vrpApp.mw.actionLeft
        actions["right"] = Application.vrpApp.mw.actionRight
        actions["top"] = Application.vrpApp.mw.actionTop
        actions["bottom"] = Application.vrpApp.mw.actionBottom
        for action in actions.values():
            action.setChecked(False)
        for button in self.__buttons.values():
            button.setChecked(False)
        if (id == -1) or not hasattr(self, "_ViewpointManager__buttonIds"):
            return
        keyToDefaultVP = dict([(key, name) for (name, key) in iter(self.__buttonIds.items())])
        for (key, params) in iter(self._key2Params.items()):
            if (params.id == activeId):
                if (key in keyToDefaultVP):
                    name = keyToDefaultVP[key]
                    if (name in actions):
                        actions[name].setChecked(True)
                    if (name in self.__buttons):
                        self.__buttons[name].setChecked(True)
                        

    def __tr(self,s,c = None):
        return coTranslate(s)


    def __str__(self):
        ret = "Viewpoint Manager\n"
        ret = ret + str(self._key2Params[self._key].__dict__)
        ret = ret + '\n' + str(self._key2Params)
        return ret

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    widget = QtWidgets.QDockWidget()
    ui = ViewpointManager(widget)
    # ui.setupUi(widget)
    widget.show()
    sys.exit(app.exec_())

# eof
