
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH

from PyQt5 import QtCore, QtGui, QtWidgets

import MainWindow
import Application

from Gui2Neg import theGuiMsgHandler
from Utils import CopyParams
from ObjectMgr import ObjectMgr
import PresenterManager

from vtrans import coTranslate 

class ListManager(QtWidgets.QDockWidget):

    def __init__(self, parent):
        QtWidgets.QDockWidget.__init__(self, parent)
        # default is prenseter manager

        # RE M self.setSizePolicy(QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum,QtWidgets.QSizePolicy.Minimum,self.sizePolicy().hasHeightForWidth()))

        #properties of QtWidgets.QDockWidget
        # REM self.setCloseMode(QtWidgets.QDockWidget.Always)
        # REM self.setResizeEnabled(True)
        # REM self.setMovingEnabled(True)        #allow to be outside the window
        # REM self.setHorizontallyStretchable(True)
        # REM self.setVerticallyStretchable(True)
        # REM self.setOrientation(Qt.Vertical)

        #current object key
        self._key= -1

        self._key2Params = {} # Note: Default-Viewpoints are in key2Params but not in the following two dicts
        self._key2ListBoxItemIdx = {}
        self._ListBoxItemIdx2Key = {}

        self._currentKey = None
        self._nextIdx = 0

        # flag for relaod item
        self._reloadStep = False

        #self.resize(QtCore.QSize(300,350).expandedTo(self.minimumSizeHint()))
        # REM self.clearWState(Qt.WState_Polished)

    def show(self):
        QtWidgets.QWidget.show(self)

    def hide(self):
        QtWidgets.QWidget.hide(self)

    def paramChanged(self, key):
        """ params of object key changed"""
        if self._key==key:
            self.update()

    def update( self ):
        if self._key!=-1:
            self.updateForObject( self._key )

    def updateForObject( self, key ):
        """ called from MainWindow to update the content to the choosen object key """
        self._key = key
        params = ObjectMgr().getParamsOfObject(key)
        self._setParams( params )

    def _getParams(self):
        return None

    def _setParams(self, params):
        return

    def addStep( self, key ):
        index = self.widget().listBox2.count()
        if key not in self._key2ListBoxItemIdx:
            self._key2ListBoxItemIdx[key] = index
            self._ListBoxItemIdx2Key[index] = key
            self._nextIdx = self._nextIdx+1

    #called if the list view is clicked in gui
    def listViewClick(self, lbi):
        if lbi :
            itemName = str(lbi.text())
            row=self.widget().listBox2.row(lbi)
            key = self._getKey(row)
            # for reload
            if key==self._currentKey:
                self._reloadStep = not self._reloadStep
            self._currentKey=key
            self._updateParamEdits()
            self.sendMgrParams()

    def _getKey( self, lbIndex ):
        return self._ListBoxItemIdx2Key[lbIndex]

    #updates the name and timeout of the current step
    def _updateParamEdits( self ):
        key = self._currentKey
        self.widget().NameLE.setEnabled(True)
        self.widget().NameLE.setText( self._key2Params[key].name )

    def selectItem( self, lbIndex ):
        # self.widget().listBox2.setSelected( lbIndex, True )
        self.widget().listBox2.setCurrentRow( lbIndex )
        self._currentKey = self._getKey(lbIndex)
        self._updateParamEdits()
        #need to rebuild the state of the whole project
        self.sendMgrParams()

    def up(self):
        #change the indexes in the listbox
        if self._currentKey in self._key2ListBoxItemIdx:
            idx = self._key2ListBoxItemIdx[self._currentKey]
            if idx>0:
                secondKey = self._getKey(idx-1)
                swap = self.widget().listBox2.takeItem(idx-1)
                self.widget().listBox2.insertItem( idx, swap)
                self._key2ListBoxItemIdx[ self._currentKey ] = idx-1
                self._ListBoxItemIdx2Key[idx-1] = self._currentKey
                self._key2ListBoxItemIdx[ secondKey ] = idx
                self._ListBoxItemIdx2Key[idx] = secondKey
                params1 = self._key2Params[self._currentKey]
                params1.index = idx-1
                params2 = self._key2Params[secondKey]
                params2.index = idx
                ObjectMgr().setParams( self._currentKey, params1 )
                ObjectMgr().setParams( secondKey, params2 )

    def down(self):
        #change the indexes in listbox
        if self._currentKey in self._key2ListBoxItemIdx:
            idx = self._key2ListBoxItemIdx[self._currentKey]
            if idx<self.widget().listBox2.count()-1:
                secondKey = self._getKey(idx+1)
                swap = self.widget().listBox2.takeItem(idx+1)
                self.widget().listBox2.insertItem( idx, swap )
                self._key2ListBoxItemIdx[ self._currentKey ] = idx+1
                self._ListBoxItemIdx2Key[idx+1] = self._currentKey
                self._key2ListBoxItemIdx[ secondKey ] = idx
                self._ListBoxItemIdx2Key[idx] = secondKey
                params1 = self._key2Params[self._currentKey]
                params1.index = idx+1
                params2 = self._key2Params[secondKey]
                params2.index = idx
                ObjectMgr().setParams( self._currentKey, params1 )
                ObjectMgr().setParams( secondKey, params2 )


    #create new step
    def new(self):
        return

    #delete current entry (just send the delete to the negotiator)
    def delete(self):
        # add asker
        msgBox = QtWidgets.QMessageBox(Application.vrpApp.mw)
        msgBox.setWindowTitle(self.__tr("Delete"))
        if (isinstance(self, PresenterManager.PresenterManager)):
            msgBox.setText(self.__tr("Delete current presentation step?"))
        else:
            msgBox.setText(self.__tr("Delete current viewpoint?"))
        msgBox.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        msgBox.setDefaultButton(QtWidgets.QMessageBox.Yes)
        acceptedOrRejected = msgBox.exec_()
        if acceptedOrRejected == QtWidgets.QMessageBox.No:
            return
        if (self._currentKey in self._key2Params):
            ObjectMgr().deleteObject(self._currentKey)

    # obj was deleted in negotiator (update list)
    def objDeleted( self, key, parentKey):
        if (key in self._key2Params):
            params = self._key2Params[key]
            del self._key2Params[key]
            if (key in self._key2ListBoxItemIdx):
                index = self._key2ListBoxItemIdx[key]
                del self._key2ListBoxItemIdx[key]
                #change index of the following items
                while index < len(self._ListBoxItemIdx2Key)-1:
                    key = self._ListBoxItemIdx2Key[index+1]
                    params = self._key2Params[key]
                    self._ListBoxItemIdx2Key[index] = key
                    self._key2ListBoxItemIdx[key] = index
                    params.index = index
                    ObjectMgr().setParams( key, params )
                    index = index +1
                del self._ListBoxItemIdx2Key[index]
            self._updateList()

    def sendMgrParams( self ):
        if not self._key == -1:
            ObjectMgr().setParams(self._key,self._getParams())
            theGuiMsgHandler().setParams(self._key,self._getParams())

    #clear the list and rebuild it
    def _updateList( self ):
        return

    #set the params of a step
    def setParams( self, key, params ):
        return

    # step params changed
    def changedParams(self, val=0):
        return


    def __tr(self,s,c = None):
        return coTranslate(s)


    def __str__(self):
        ret = "ListManager\n"
        ret = ret + str(self._key2Params[self.__key].__dict__)
        ret = ret + '\n' + str(self._key2Params)
        return ret


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    widget = QtWidgets.QDockWidget()
    ui = ListManager(widget)
    # ui.setupUi(widget)
    widget.show()
    sys.exit(app.exec_())

# eof
