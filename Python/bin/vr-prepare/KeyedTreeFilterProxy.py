# Part of the vr-prepare program

# Copyright (c) 2011 Visenso GmbH

from PyQt5 import QtCore, QtGui

import Application
from KeydObject import TYPE_SCENEGRAPH_ITEM

class KeyedTreeFilterProxy(QtCore.QSortFilterProxyModel):

    def __init__(self, parent):
        QtCore.QSortFilterProxyModel.__init__(self, parent)
        self.filterText = ""
        self.reduceSceneGraphItems = False

    def filterAcceptsRow(self, source_row, source_parent):
        if (self.reduceSceneGraphItems and self.isSoleChildOfSceneGraphItem(source_row, source_parent)):
            return False
        return self.containsFilterText(source_row, source_parent)

    def containsFilterText(self, source_row, source_parent):
        if (self.filterText == ""):
            return True
        source_index = self.sourceModel().index(source_row, 0, source_parent)
        item = self.sourceModel().itemFromIndex(source_index)
        if item.text().contains(self.filterRegExp):
            return True
        for i in range(self.sourceModel().rowCount(source_index)):
            if self.containsFilterText(i, source_index):
                return True
        return False

    def isSoleChildOfSceneGraphItem(self, source_row, source_parent):
        # check if parent is a SceneGraphItem
        parentItem = self.sourceModel().itemFromIndex(source_parent)
        if not parentItem:
            return False
        parentKey = parentItem.data()
        if (Application.vrpApp.key2type[parentKey] != TYPE_SCENEGRAPH_ITEM):
            return False
        # check number of siblings
        if (self.sourceModel().rowCount(source_parent) != 1):
            return False
        # check if this is a leaf
        source_index = self.sourceModel().index(source_row, 0, source_parent)
        if (self.sourceModel().rowCount(source_index) == 0 ):
            return True
        # check recursively
        return self.isSoleChildOfSceneGraphItem(0, source_index)

    def setFilter(self, text, reduceSceneGraphItems):
        self.filterText = text
        self.reduceSceneGraphItems = reduceSceneGraphItems
        self.filterRegExp = QtCore.QRegExp(text, QtCore.Qt.CaseInsensitive, QtCore.QRegExp.Wildcard)
        self.invalidate()
