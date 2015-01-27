
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH


import os.path

import covise

from PyQt5 import QtCore, QtGui, QtWidgets

from auxils import cLikeCounter
from printing import InfoPrintCapable
from qtauxils import DeepLVIterator

from coviseCase import (
    DimensionSeperatedCase,
    GEOMETRY_2D,
    GEOMETRY_3D,
    PartWithFileReference,
    )
from KeyedTreeView import KeyedTreeView, keyPathToKey
from CasesFilterBase import Ui_CasesFilterBase

_infoer = InfoPrintCapable()
_infoer.doPrint =  False #True # 

_logger = InfoPrintCapable()
_logger.doPrint =  False #True # 
_logger.startString = '(log)'
_logger.module = __name__

def _log(func):
    def logged_func(*args, **kwargs):
        _logger.function = repr(func)
        _logger.write('')
        return func(*args, **kwargs)
    return logged_func


class TreeSelectionHandler(object):

    """Bring together two KeyedTreeView instances for building a choice."""

    def __init__(self, sourceListView, choiceListView):
        _infoer.function = str(self.__init__)
        _infoer.write("")
        self._sourceLV = sourceListView
        self._choiceLV = choiceListView
        
    def getSelectedSourceKeys(self):
        return self._sourceLV.getSelectedKeys()
        
    def getSelectedChoiceKeys(self):
        return self._choiceLV.getSelectedKeys()    
        
    def getChoiceKeys(self):
        return self._choiceLV.getKeys()

    def addToChoice(self):
        _infoer.function = str(self.addToChoice)
        _infoer.write("")
        #for every selected key 
        for key in self._sourceLV.getSelectedKeys():
            # check if key in listview
            if self._choiceLV.has_key(key): continue
            # get the path (all parents) to the selected key
            pathToKey = keyPathToKey(self._sourceLV, key)
            parentOfKeyOnPath = None
            # add every partent of key if not already in 
            for keyOnPath in pathToKey:
                if not self._choiceLV.has_key(keyOnPath):
                    self._choiceLV.insert(
                        parentOfKeyOnPath,
                        keyOnPath,
                        self._sourceLV.getItemData(keyOnPath))
                parentOfKeyOnPath = keyOnPath
            self._choiceLV.insert(
                parentOfKeyOnPath, key, self._sourceLV.getItemData(key))


    def removeFromChoice(self):
        _infoer.function = str(self.removeFromChoice)
        _infoer.write("")
        for key in self._choiceLV.getSelectedKeys():
            if self._choiceLV.has_key(key): 
                self._choiceLV.delete(key)


def setAllChildrenSelected(lv):
    _infoer.function = str(setAllChildrenSelected)
    _infoer.write("")
    for key in lv.getSelectedKeys():
        for subkey in lv.recursiveChildKeys(key):
            lv.setItemSelected(subkey, True)


class CasesFilter(QtWidgets.QDialog, Ui_CasesFilterBase):

    @_log
    def __init__(self, parent=None, choicePolicy=TreeSelectionHandler):
        _infoer.function = str(self.__init__)
        _infoer.write("")
        QtWidgets.QDialog.__init__(self, parent)
        Ui_CasesFilterBase.__init__(self)
        self.setupUi(self)

        self.checkBox5.hide() # not working

        # hide dialog specific buttons
        self.pushButtonOK.hide()
        self.pushButtonCancel.hide()
            
        self.completeTree = KeyedTreeView(
            self.completeWidgetStack, None, 'All Parts')
        #self.completeTree.setSelectionMode(QListView.ExtendedSelection)
        self.completeWidgetStack.addWidget(self.completeTree)
        self.completeWidgetStack.setCurrentWidget(self.completeTree)

        self.filteredTree = KeyedTreeView(
            self.filteredWidgetStack, None, 'Selected Parts')
        #self.filteredTree.setSelectionMode(QListView.Extended)
        self.filteredWidgetStack.addWidget(self.filteredTree)
        self.filteredWidgetStack.setCurrentWidget(self.filteredTree)

        self.treeSelectionHandler = choicePolicy(
            self.completeTree, self.filteredTree)

        self.__completeTreeKeyCounter = cLikeCounter()
        self.__key2CaseItem = {} # for class-internal lookup
        self.__usefulOnlyWithChild = []
        self.__cases = []

        self.selectAllButton.clicked.connect(self.__selectAllInCompleteTree)
        self.deselectAllButton.clicked.connect(self.__deselectAllInCompleteTree)
        
        self.completeTree.clicked.connect(self.setCompleteChildrenSelected)
        self.filteredTree.clicked.connect(self.setFilteredChildrenSelected)
        
        self.completeTree.doubleClicked.connect(self.completeTreeDoubleClicked)
        self.filteredTree.doubleClicked.connect(self.filteredTreeDoubleClicked)


    @_log
    def getAllCases(self):
        _infoer.function = str(self.getAllCases)
        _infoer.write("")
        return self.__cases

    @_log
    def addDimensionSeperatedCase(self, aCase):
        _infoer.function = str(self.addDimensionSeperatedCase)
        _infoer.write("")

        """Add a representation of the case to the widget."""
        def internRegister(key, thing, usefulOnlyWithChild):
            self.__key2CaseItem[key] = thing
            if usefulOnlyWithChild: self.__usefulOnlyWithChild.append(key)
        caseKey = next(self.__completeTreeKeyCounter)
        self.completeTree.insert(
            parentKey = None, key = caseKey, itemData = str(aCase.name))
        internRegister(caseKey, aCase, True)
        parts2d = aCase.parts2d
        parts2dKey = next(self.__completeTreeKeyCounter)
        self.completeTree.insert(caseKey, parts2dKey, parts2d.name)
        internRegister(parts2dKey, parts2d, True)
        def addParts(parentKey):
            for part in self.__key2CaseItem[parentKey]:
                partKey = next(self.__completeTreeKeyCounter)
                self.completeTree.insert(parentKey, partKey, part.name)
                internRegister(partKey, part, False)
        addParts(parts2dKey)
        parts3d = aCase.parts3d
        parts3dKey = next(self.__completeTreeKeyCounter)
        self.completeTree.insert(caseKey, parts3dKey, parts3d.name)
        internRegister(parts3dKey, parts3d, True)
        addParts(parts3dKey)
        self.__cases.append(aCase)

    def erase(self):
        _infoer.function = str(self.erase)
        _infoer.write("")
        """ delete the complete content """
        self.completeTree.erase()
        self.filteredTree.erase()
        self.__completeTreeKeyCounter = cLikeCounter() # reset
        self.__key2CaseItem.clear()
        self.__usefulOnlyWithChild = []
        self.__cases = []

    @_log
    def getChoice(self):
        _infoer.function = str(self.getChoice)
        _infoer.write("")
        """Return list of the choice of dimension-separated-cases."""
        self.__decorateChoice()
        fineDscs = self.__createFineChosenList()
        roughDscs = self.__createRoughChosenList()
        self.__undecorateChoice()
        return  fineDscs, roughDscs

    def addToChoice(self):
        _infoer.function = str(self.addToChoice)
        _infoer.write("")
        self.treeSelectionHandler.addToChoice()
        self.__clearAllSelections()

    def removeFromChoice(self):
        _infoer.function = str(self.removeFromChoice)
        _infoer.write("")
        # remove selection from listview
        keys = self.treeSelectionHandler.getSelectedChoiceKeys()
        self.treeSelectionHandler.removeFromChoice()
        for key in keys: self.filteredTree.delete(key)
        # now delete the keys 
        childLessUOWCKeys = self.__keysOfUOWCWithoutChild()
        while childLessUOWCKeys:
            for key in childLessUOWCKeys:
                self.filteredTree.delete(key)
            childLessUOWCKeys = self.__keysOfUOWCWithoutChild()
        self.__clearAllSelections()


    #def collapseEquivalentInFilteredTree(self, lvi):
        #_infoer.function = str(self.collapseEquivalentInFilteredTree)
        #_infoer.write("")
        #key = self.completeTree.keyAndItem.preimage(lvi)
        #if self.filteredTree.has_key(key):
            #self.filteredTree.setItemOpen(key, False)
        ## ignore if no equivalent

    #def expandEquivalentInFilteredTree(self, lvi):
        #_infoer.function = str(self.expandEquivalentInFilteredTree)
        #_infoer.write("")
        #key = self.completeTree.keyAndItem.preimage(lvi)
        #if self.filteredTree.has_key(key):
            #self.filteredTree.setItemOpen(key, True)
        ## ignore if no equivalent

    #def collapseEquivalentInCompleteTree(self, lvi):
        #_infoer.function = str(self.collapseEquivalentInCompleteTree)
        #_infoer.write("")
        #key = self.filteredTree.keyAndItem.preimage(lvi)
        #if self.completeTree.has_key(key):
            #self.completeTree.setItemOpen(key, False)
        ## ignore if no equivalent

    #def expandEquivalentInCompleteTree(self, lvi):
        #_infoer.function = str(self.expandEquivalentInCompleteTree)
        #_infoer.write("")
        #key = self.filteredTree.keyAndItem.preimage(lvi)
        #if self.completeTree.has_key(key):
            #self.completeTree.setItemOpen(key, True)
        ## ignore if no equivalent

    def setCompleteChildrenSelected(self, lvi):
        _infoer.function = str(self.setCompleteChildrenSelected)
        _infoer.write("")
        setAllChildrenSelected(self.completeTree)

    def setFilteredChildrenSelected(self, lvi):
        _infoer.function = str(self.setFilteredChildrenSelected)
        _infoer.write("")
        setAllChildrenSelected(self.filteredTree)

    def completeTreeDoubleClicked(self, lvi):
        self.addButton.clicked.emit() # treat doubleclick as click on the add button (StartPreparation.py acts on the signal as well)

    def filteredTreeDoubleClicked(self, lvi):
        self.removeButton.clicked.emit() # treat doubleclick as click on the remove button (StartPreparation.py acts on the signal as well)

    def __keysOfUOWCWithoutChild(self): # UOWC = UsefulOnlyWithChild (a dict)
        _infoer.function = str(self.__keysOfUOWCWithoutChild)
        _infoer.write("")
        keys = []
        for key in self.__usefulOnlyWithChild:
            if self.filteredTree.has_key(key) and (self.filteredTree.childKeys(key) == []):
                keys.append(key)
        return keys

    def __clearAllSelections(self):
        _infoer.function = str(self.__clearAllSelections)
        _infoer.write("")
        self.completeTree.clearSelection()
        self.filteredTree.clearSelection()

    def __selectAllInCompleteTree(self):
        _infoer.function = str(self.__selectAllInCompleteTree)
        _infoer.write("")
        self.completeTree.selectAll()
        self.addToChoice()

    def __deselectAllInCompleteTree(self):
        _infoer.function = str(self.__deselectAllInCompleteTree)
        _infoer.write("")
        self.filteredTree.selectAll()
        self.removeFromChoice()

    def __decorateChoice(self):
        _infoer.function = str(self.__decorateChoice)
        _infoer.write("")
        selectedKeys = self.treeSelectionHandler.getChoiceKeys()
        for key in self.completeTree.getKeys():
            self.__key2CaseItem[key].chosen = (key in selectedKeys)

    def __createFineChosenList(self):
        _infoer.function = str(self.__createFineChosenList)
        def createSubDsc(aDsc):
            """Return sub-dsc with items with chosen == True."""
            def copyPartAndAppend(part, aList):
                copyOfPart = PartWithFileReference(
                    part.filename, part.name)
                copyOfPart.variables = part.variables
                aList.append(copyOfPart)

            subDsc = DimensionSeperatedCase(aDsc.name)
            partLists = [
                (aDsc.parts2d, subDsc.parts2d),
                (aDsc.parts3d, subDsc.parts3d)]
            for partList, partListSub in partLists:
                if partList.chosen:
                    for part in partList:
                        if part.chosen:
                            copyPartAndAppend(part, partListSub)
            _infoer.write("subDsc %s" %str(subDsc))
            return subDsc
        dscs = []
        for dsc in self.__cases:
            if dsc.chosen: dscs.append(createSubDsc(dsc))
        _infoer.write("dscs %s" %str(dscs))
        return dscs

    def __createRoughChosenList(self):
        _infoer.function = str(self.__createRoughChosenList)
        _infoer.write("")
        """Return list of added DSCS that have any chose part."""
        dscs = []
        for dsc in self.__cases:
            if dsc.chosen: dscs.append(dsc)
        return dscs

    def __undecorateChoice(self):
        _infoer.function = str(self.__undecorateChoice)
        _infoer.write("")
        for key in self.completeTree.getKeys():
            del self.__key2CaseItem[key].chosen

# eof
