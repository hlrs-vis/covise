
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH

import os
import sys
#from itertools import ifilter

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject, pyqtSignal

from printing import InfoPrintCapable

from auxils import OneToOne
from qtauxils import SelectedItemsIterator, itemFromProxyIndex
# import StaticImages

_infoer = InfoPrintCapable()
_infoer.doPrint = False # True

# Decorator for printing function-calls.
_logger = InfoPrintCapable()
_logger.doPrint = False # True
_logger.startString = '(log)'
_logger.module = __name__
def _log(func):
    def logged_func(*args, **kwargs):
        _logger.function = repr(func)
        _logger.write('')
        return func(*args, **kwargs)
    return logged_func

from vtrans import coTranslate 

# Suitable list-view for tree-structure with items with keys


class KeyedTreeView(QtWidgets.QTreeView):
    """Viewer and controller for items with unique keys.

    None is the key for the root-item.

    Viewer:
    o Hirarchy of items.
    o Support of several different items (e.g. checkable or not).
    Controller:
    o Emission of selection change.
    o Emission of checked change.
    o Emission of contex-menu-request

    """

    # To the implementation:

    # Using single underscore to prefix private
    # entities for easier testing opposed to the usual
    # double-underscore-prefix for privates.
    sigItemChecked = pyqtSignal(int,bool)
    sigSelectionChanged = pyqtSignal()
    sigNoItemClicked = pyqtSignal()
    sigRemoveFilter = pyqtSignal()
    sigContextMenuRequest = pyqtSignal()
    sigItemClicked = pyqtSignal(int)
        
        
    def __init__(self, parent=None, proxy=None, heading=""):
        QtWidgets.QTreeView.__init__(self,parent)
        self.itemDataDict = {}
        self.item = {}
        self.__latestItem_sigItemClicked = None # None stands for there is no last.

        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.header().hide()

        self.fullModel = QtGui.QStandardItemModel(parent)
        if proxy:
            self.proxyModel = proxy
        else:
            # We always add a proxy to keep things simple.
            # So create a dummy proxy here if nescessary.
            self.proxyModel = QtCore.QSortFilterProxyModel(parent)
            self.proxyModel.setDynamicSortFilter(True)
        self.proxyModel.setSourceModel(self.fullModel)
        self.setModel(self.proxyModel)

        # self.__tree.setRootIsDecorated(1)
        # self.addColumn(self.__tr(heading))

        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        
        self.clicked.connect(self.emitSignalClicked)
        self.selectionModel().selectionChanged.connect(self.emitSignalClicked)
        self.customContextMenuRequested.connect(self.emitContextMenuRequest)

    def emitSignalClicked(self, lvi):
        """Emit signals about click onto lvi-"""
        if not lvi:
            return
        if type(lvi) == QtCore.QItemSelection:
            indexes = lvi.indexes()
            if len(indexes) == 0:
                return
            lvi = indexes[0]
        aKey = itemFromProxyIndex(self.proxyModel, lvi).data()
        self._emitSignalForClickOnItemWithKey( aKey )

    def emitSelectionChanged(self, selected, deselected):
        """Workaround for selection with mouse without click.

        Emits signal sigSelectionChanged."""
        selectedKeys = self.getSelectedKeys()
        # send signal if multiselection
        if len(selectedKeys) > 1:
            self.sigSelectionChanged.emit()
            # unset latest clicked item if multiselection
            self.__latestItem_sigItemClicked = None

    def emitContextMenuRequest(self, point):
        self.sigContextMenuRequest.emit(self.mapToGlobal(point))

    def emitSignalPressed(self, lvi):
        """Emit signals about press onto lvi-"""
        #print "Pressed" , lvi.data().toString()
        self._emitSignalForPressOnItemWithKey( itemFromProxyIndex(self.proxyModel, lvi).data() )

    def _emitSignalForPressOnItemWithKey(self, aKey):
        """Emit signals sigItemPressed and sigNoItemClicked.

        Thought for use as slot after user presses something.
        Is not fully private for easy testing.

        """

        # set last clicked Item None in case of multiselection
        # or click to no item
        #REM if len(self.getSelectedKeys()) > 1 or not aKey:
        #REM     self.__latestItem_sigItemClicked = None

        # no item was clicked
        if not aKey:
            self.sigNoItemClicked.emit()
            return
        lvi = self.item[aKey]

        if hasattr(self.itemDataDict[aKey], 'isChecked'):
            # test if checkbox was toggled
            #print "Press ", lvi.checkState()==QtCore.Qt.Checked, self.itemDataDict[aKey].isChecked
            if (not lvi.checkState()==QtCore.Qt.Checked) ==  self.itemDataDict[aKey].isChecked:
                # store the value
                self.itemDataDict[aKey].wasPressed = lvi.checkState()
                # switch it to the old state
                if lvi.checkState()==QtCore.Qt.Checked:
                    lvi.setCheckState(QtCore.Qt.Unchecked)
                else:
                    lvi.setCheckState(QtCore.Qt.Checked)

    def _emitSignalForClickOnItemWithKey(self, aKey):

        """Emit signals sigItemClicked and sigItemChecked.

        Thought for use as slot after user clicks something.
        Is not fully private for easy testing.

        """
        lvi = self.item[aKey]
        if hasattr(self.itemDataDict[aKey], 'isChecked'):

            if hasattr(self.itemDataDict[aKey], 'wasPressed'):

                lvi.setCheckState(self.itemDataDict[aKey].wasPressed)

            # Seperation of checkbox-toggle or not.

            # test if checkbox was toggled
            if (lvi.checkState()==QtCore.Qt.Checked) != self.itemDataDict[aKey].isChecked:
                self.itemDataDict[aKey].isChecked = (lvi.checkState()==QtCore.Qt.Checked)
                self.sigItemChecked.emit(aKey, lvi.checkState()==QtCore.Qt.Checked)
                # select latest clicked item again
                #REM if self.__latestItem_sigItemClicked:
                #REM     self.setSelected(self.__latestItem_sigItemClicked, True)
                # deselect the toggled item
                #REM if self.__latestItem_sigItemClicked and not self.__latestItem_sigItemClicked == self.item[aKey] :
                #REM     self.setSelected(self.item[aKey], False)
                #show panel if lastclicked item is empty
                #REM if not self.__latestItem_sigItemClicked:
                #REM     self.emit(QtCore.SIGNAL('sigItemClicked'), aKey)
                #REM     self.__latestItem_sigItemClicked = self.item[aKey]

            self.sigItemClicked.emit(aKey)
            self.__latestItem_sigItemClicked = self.item[aKey]

        else:
            self.sigItemClicked.emit( aKey)
            self.__latestItem_sigItemClicked = self.item[aKey]


    def has_key(self, key):
        return key in self.item

    def getKeys(self):
        return self.item.keys()

    def getItemData(self, key):
        return self.itemDataDict[key]

    def getSelectedKeys(self):
        return list(map(lambda item: item.data(), SelectedItemsIterator(self)))

    def unselectAll(self):
        self.selectionModel().reset()

    def itemIsOpen(self, key):
        return self.item[key].isOpen()

    def insert(self, parentKey, key, itemData):
        #if hasattr( itemData, "name" ) : print "Insert ", parentKey, key, itemData.name
        if None == parentKey: parentItem = self.fullModel.invisibleRootItem()
        else: parentItem = self.item[parentKey]
        item = self._createItem(parentItem, key, itemData)
        self.item[key] = item
        self.itemDataDict[key] = itemData
        self.sigRemoveFilter.emit() # the filter has to be removed while adding/deleting items
    def erase(self):
        """Delete the complete tree """
        keysToDel = []
        # search for parent nodes
        for key in self.itemDataDict:
            parent = self.parentKey(key)
            if parent==None:
                keysToDel.append(key)
        for key in keysToDel:
            self.delete(key)
        self.sigRemoveFilter.emit() # the filter has to be removed while adding/deleting items

    def delete(self, key):
        """Delete representation of key from the tree.

        Note: This implies deletion of all sub-items.

        """
        if not key in self.item.keys():
            return
        for child in self.childKeys(key):
            self.delete(child)
        keepForDeletion = self.item[key]
        parent = keepForDeletion.parent()
        if not parent: # top-level-item
            parent = keepForDeletion.model()
        parent.takeRow(keepForDeletion.row())
        del keepForDeletion
        del self.itemDataDict[key]
        del self.item[key]
        self.sigRemoveFilter.emit() # the filter has to be removed while adding/deleting items

    @_log
    def setItemData(self, key, itemData):
        if isinstance(itemData, str) and (key in self.itemDataDict.keys() ):
            oldData = self.itemDataDict[key]
            if not isinstance(oldData, str):
                oldData.name = itemData
                self.itemDataDict[key] = oldData
            else:
                self.itemDataDict[key] = itemData
        elif isinstance(itemData, str):
            self.itemDataDict[key] = itemData
        elif not isinstance(itemData, str):
            self.itemDataDict[key] = itemData
        self._updateItemFromData(key, itemData)

    def childKeys(self, key):
        if not key in self.item.keys():
            return []
        if None == key:
            parent = self
        else:
            parent = self.item[key]
        children = []
        i = 0
        while (i<parent.rowCount()):
            children.append( parent.child(i).data() )
            i = i + 1
        return children

    def recursiveChildKeys(self, key):
        if not key in self.item.keys():
            return []
        if None == key:
            parent = self
        else:
            parent = self.item[key]
        children = []
        i = 0
        while (i<parent.rowCount()):
            children.append( parent.child(i).data() )
            children.extend( self.recursiveChildKeys(parent.child(i).data()) )
            i = i + 1
        return children


    def parentKey(self, key):
        item = self.item[key]
        parentItem = item.parent()
        if None == parentItem: return None
        return parentItem.data()

    def setItemSelected(self, key, multiselect=False, select=True):
        if self.has_key(key):
            if multiselect and select:
                selectionFlag = QtCore.QItemSelectionModel.Select
            elif multiselect and not select:
                selectionFlag = QtCore.QItemSelectionModel.Deselect
            elif not multiselect and not select:
                selectionFlag = QCore.QItemSelectionModel.Deselect
            else:
                selectionFlag = QtCore.QItemSelectionModel.ClearAndSelect
            self.selectionModel().select(self.proxyModel.mapFromSource(self.item[key].index()), selectionFlag)
        else:
            self.unselectAll()

    def setFilter(self, text="", reduceSceneGraphItems=False):
        self.proxyModel.setFilter(text, reduceSceneGraphItems)
        # Changing the filter collapses some items but never expands any. So just expand all, to make sure the matching nodes are visible.
        self.expandAll()
        # Keep the selected item visible.
        if len(self.selectionModel().selectedIndexes()) > 0:
            self.scrollTo(self.selectionModel().selectedIndexes())

    def _createItem(self, parentItem, key, itemData):
        if isinstance(itemData, str):
            newLvi = QtGui.QStandardItem(itemData)
        elif (hasattr(itemData, 'name') and
              hasattr(itemData, 'isChecked')):
            name = itemData.name
            newLvi = QtGui.QStandardItem(name)
            newLvi.setCheckable(True)
            if itemData.isChecked:
                newLvi.setCheckState(QtCore.Qt.Checked)
            else:
                newLvi.setCheckState(QtCore.Qt.Unchecked)
        else:
            assert False, 'reached unreachable point'

        newLvi.setData(key)
        parentItem.appendRow(newLvi)
        if (self != parentItem):
            index = newLvi.index()
            index = self.proxyModel.mapFromSource(newLvi.index())
            self.setExpanded(index , True)
        return newLvi



    @_log
    def _updateItemFromData(self, key, itemData):
        lvi = self.item[key]
        if isinstance(itemData, str):
            lvi.setText(itemData)
            return
        if hasattr(itemData, 'name'):
            lvi.setText(itemData.name)
        if hasattr(itemData, 'isChecked'):
            if itemData.isChecked: lvi.setCheckState(QtCore.Qt.Checked)
            else: lvi.setCheckState(QtCore.Qt.Unchecked )
            # lvi.listView().update()
        if hasattr(itemData, 'pixmap'):
            lvi.setPixmap( QtCore.QPixmap.fromMimeSource(itemData.pixmap) )

    def __rmbClickedSlot(self, lvi, pt, col):
        """Reaction on right mouse button click."""
        if not lvi: return
        self.sigContextMenuRequest.emit( self, pt, itemFromProxyIndex(self.proxyModel, lvi).data())

    def __tr(self,s,c = None):
        return coTranslate(s)


# Free standing helpers

def keyPathToKey(aKeyedTree, key):
    parentKey = aKeyedTree.parentKey(key)
    if None == parentKey: return []
    thePath = keyPathToKey(aKeyedTree, parentKey)
    thePath.append(parentKey)
    return thePath



class A:
    name = "Test"
    isChecked = True

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = KeyedTreeView()
    ui.setupUi(Form)
    ui.insert( None, 0, "0")
    ui.insert( 0, 1, "1")
    e = A()
    ui.insert( None, 2, e)
    ui.insert( 2, 5, "3")
    f = A()
    ui.insert( 2, 3, f)
    g = A()
    ui.insert( 2, 4, g)
    Form.show()
    sys.exit(app.exec_())
# eof
