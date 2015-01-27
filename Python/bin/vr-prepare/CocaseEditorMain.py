try:
    import cPickle as pickle
except:
    import pickle

import os
import sys
from itertools import ifilter
import copy

import covise

from PyQt5 import QtCore, QtGui

from auxils import cLikeCounter
from qtauxils import SelectedItemsIterator

from coviseCase import CoviseCaseFile, CoviseCaseFileItem
from vrpconstants import (
    GEOMETRY_2D, GEOMETRY_3D, SCALARVARIABLE, VECTOR3DVARIABLE)
from CocaseEditorBase import CocaseEditorBase

from vtrans import coTranslate 

class PartItem(object):
    def __init__(self, type, name, filename):
        self.type = type
        self.name = name
        self.filename = filename

class PartNode(object):
    def __init__(self, parent, key):
        """parent==None means root-node."""
        self._key = key
        self._parent = parent
        if parent: # this if is only for the root-node
            parent.children().append(self)
        self._children = []

    def parent(self): return self._parent
    def children(self): return self._children
    def key(self): return self._key


def printNodeIndendedRecursively(aNode, indentLevel):
    print(indentLevel*' ' + str(aNode.key()))
    for node in aNode.children():
        printNodeIndendedRecursively(node, indentLevel + 1)


class CocaseFileCentral(object):
    def __init__(self):
        self._initMembers()


    def toCoviseCaseFile(self, cocaseDir):
        #cocaseDir is path to which the file will be saved
        theCase = CoviseCaseFile()
        for node in self._rootNode.children():
            part = self._idPartMap[node.key()]
            item = None
            #name of data
            itemName = part.filename.lstrip('/').split('/')
            itemName.reverse()
            itemName = itemName[0]
            relPath = self.toShortPath(cocaseDir, os.path.dirname(part.filename))
            relPath = relPath + itemName
            if '2D' == part.type:
                item = CoviseCaseFileItem(part.name, GEOMETRY_2D, relPath)
            elif '3D' == part.type:
                item = CoviseCaseFileItem(part.name, GEOMETRY_3D, relPath)
            else:
                assert False
            for cnode in node.children():
                cpart = self._idPartMap[cnode.key()]
                relPath = self.toShortPath(cocaseDir, os.path.dirname(cpart.filename))
                partName = cpart.filename.lstrip('/').split('/')
                partName.reverse()
                partName = partName[0]
                relPath = relPath+partName
                if 'S' == cpart.type:
                    item.addVariableAndFilename(
                        cpart.name, relPath, SCALARVARIABLE)
                if 'V' == cpart.type:
                    item.addVariableAndFilename(
                        cpart.name, relPath, VECTOR3DVARIABLE)
            theCase.add(item)
        return theCase
        
    # create relative path
    def toShortPath(self, path1, path2):
        p1 = path1.lstrip('/').split('/')
        p1.reverse()
        p2 = path2.lstrip('/').split('/')
        p2.reverse()
        returnPath = ""
        pathToDir = copy.copy(p2)
        if self.testPathEqual(p1,p2):
            return "./"
        # test if subpath of path2 is in path1
        for dir2 in p2:
            ret = self.testDirInPath(dir2, p1, pathToDir)
            if not ret=="":
                return ret+returnPath
            else:
                returnPath=dir2+"/"+returnPath
            pathToDir.remove(dir2)
        # no subpath in path1
        for dir1 in p1:
            returnPath = "../"+returnPath
        return returnPath

        
    # test if directory is in path and return path to it
    def testDirInPath(self, directory, path, pathToDir):
        returnPath = "./"
        if not directory in path:
            return ""
        p = copy.copy(path)
        for pDir in path:
            if pDir==directory and self.testPathEqual(pathToDir, p):            
                return returnPath
            else:
                returnPath += "../"
            p.remove(pDir)
        return ""
            
            
    def testPathEqual(self, path1, path2):
        if not len(path1) == len(path2):
            return False
        for counter in range(len(path1)):
            if not path1[counter]==path2[counter]:
                    return False
        return True
        


    def addItem(self, parentId, item):
        """Add item with parent parentId."""
        id = next(self._idCtr)
        if -1 == parentId: parentNode = self._rootNode
        else: parentNode = self._idNodeMap[parentId]
        node = PartNode(parentNode, id)
        self._idNodeMap[id] = node
        self._idPartMap[id] = item
        return id

    def deleteItem(self, id):
        """Remove item with the given id and all its children."""
        node = self._idNodeMap[id]
        node.parent().children().remove(node)
        for child in node.children(): # this loop is void for leaves
            self.deleteItem(child.key())
        del self._idNodeMap[id]
        del self._idPartMap[id]

    def initFromCoviseCaseFile(self, aCase, dataDir):
        """Adding data from covise case file to the nodeMap"""
        
        #dataDir should finish with '/'
        if dataDir.rfind('/')!=len(dataDir)-1:
           dataDir+='/'
        geoDimtype2str = {
            GEOMETRY_2D:'2D',
            GEOMETRY_3D:'3D'}
        varDimtype2str = {
            SCALARVARIABLE: 'S',
            VECTOR3DVARIABLE: 'V'
            }
        self._initMembers()
        for cfi in aCase.items_: # case file item
            id = self.addItem(
                -1, PartItem(geoDimtype2str[cfi.dimensionality_],
                             cfi.name_,
                             dataDir+cfi.geometryFileName_))
            for vi in cfi.variables_: # variable item
                vid = self.addItem(
                    id, PartItem(varDimtype2str[vi[2]],
                                 vi[0],
                                 dataDir+vi[1]))

    def _initMembers(self):
        self._idCtr = cLikeCounter()
        self._rootNode = PartNode(None, -1)
        self._idNodeMap = {}
        self._idPartMap = {}


class CocaseEditor(CocaseEditorBase):
    """Gui for the editor"""
    def __init__(self,parent = None,name = None,fl = 0):
        CocaseEditorBase.__init__(self)
        #self.listView1.setRootIsDecorated(1)
        self._lviIdMap = {}
        self._cocaseCentral = CocaseFileCentral()
        self._currentId = None
        self._currentFilePath = os.getcwd() #current working directory
        self._delItem.setEnabled(False)
        self.groupBox1.setEnabled(False)
        self.groupBox2.setEnabled(False)

        self.listViewModel = QtGui.QStandardItemModel()
        self.listView1.setModel(self.listViewModel)
        self.selectedLVI = None

        self._typeString2ComboIdx = {
            '2D': 0, '3D': 1,
            'S': 0, 'V': 1,
            }
        self._cbId2GEOTypeString = ['2D', '3D']
        self._cbId2VARTypeString = ['S', 'V']

        self._addGeo.clicked.connect(self.addGeoItemSlot)
        self._addVar.clicked.connect(self.addVarItemSlot)
        self._delItem.clicked.connect(self.deleteGeoItemSlot)

        
        self.listView1.selectionModel().selectionChanged.connect(self.selectionChangedSlot)
        self._geoDim.activated.connect(self._geoDimChangeSlot)
        self._geoName.textChanged.connect(self._nameChangeOfCurrentId)
        self._varDim.activated.connect(self._varDimChangeSlot)
        self._varName.textChanged.connect(self._nameChangeOfCurrentId)

    def closeEvent(self, event):
        covise.clean()
        covise.quit()

    def _geoDimChangeSlot(self, choice):
        assert self._currentId is not None
        id = self._currentId
        self._cocaseCentral._idPartMap[id].type = self._cbId2GEOTypeString[choice]
        #self._updateListView()
        self._setCurrentId(id)

    def _nameChangeOfCurrentId(self, nameQt):
        assert self._currentId is not None
        id = self._currentId
        self._cocaseCentral._idPartMap[id].name = str(nameQt)
        self.listView1.selectionModel().blockSignals(True)
        self._updateListView()
        self._setCurrentId(id)
        self.listView1.selectionModel().blockSignals(False)

    def _varDimChangeSlot(self, choice):
        assert self._currentId is not None
        id = self._currentId
        self._cocaseCentral._idPartMap[id].type = self._cbId2VARTypeString[choice]
        #self._updateListView()
        self._setCurrentId(id)

    def selectionChangedSlot(self, select, unselect):
        if select.count() > 0:
            self.selectedLVI = (self.listViewModel).itemFromIndex(select.indexes()[0])
        else:
            self.selectedLVI = None
        #lvi = firstSelectedLvi(self.listView1)
        lvi = self.selectedLVI
        if not lvi:
            self._delItem.setEnabled(False)
            self.groupBox1.setEnabled(False)
            self.groupBox2.setEnabled(False)
            self._currentId = None
            return
        id = self._lviIdMap[lvi]
        self._currentId = id
        node = self._cocaseCentral._idNodeMap[id]
        item = self._cocaseCentral._idPartMap[id]
        parentNode = node.parent().key()
        if -1 == parentNode: # toplevel
            self._delItem.setEnabled(True)
            self.groupBox1.setEnabled(True)
            self.groupBox2.setEnabled(False)

            self._geoName.blockSignals(True)
            self._geoName.setText(item.name)
            self._geoName.blockSignals(False)

            self._geoDim.blockSignals(True)
            self._geoDim.setCurrentIndex(self._typeString2ComboIdx[item.type])
            self._geoDim.blockSignals(False)
        else: # below toplevel
            self._delItem.setEnabled(True)
            self.groupBox1.setEnabled(False)
            self.groupBox2.setEnabled(True)

            self._varName.blockSignals(True)
            self._varName.setText(item.name)
            self._varName.blockSignals(False)
            
            self._varDim.blockSignals(True)
            self._varDim.setCurrentIndex(self._typeString2ComboIdx[item.type])
            self._varDim.blockSignals(False)

    def fileNew(self):
        self._cocaseCentral._initMembers()
        self._updateListView()

    def fileOpen(self):
        fd = QtWidgets.QFileDialog(self)
        fd.setMinimumWidth(1050)
        fd.setMinimumHeight(700)
        fd.setNameFilter(self.__tr('Covise-Case-Files (*.cocase)'))
        fd.setWindowTitle(self.__tr('open file dialog'))
        fd.setDirectory(self._currentFilePath)

        acceptedOrRejected = fd.exec_()
        if acceptedOrRejected != QtWidgets.QDialog.Accepted :
            return
        filenamesQt = fd.selectedFiles()
        if filenamesQt.isEmpty():
            return
        self._currentFilePath = os.path.dirname(str(filenamesQt[0]))
        aFilename = str(filenamesQt[0])
        inputFile = open(aFilename, 'rb')
        theCase = pickle.load(inputFile)
        inputFile.close()
        self._cocaseCentral.initFromCoviseCaseFile(theCase, self._currentFilePath)
        self._updateListView()

    def fileSave(self):
        print("CocaseEditor.fileSave(): Not implemented yet")

    def fileSaveAs(self):
        fd = QtWidgets.QFileDialog(self)
        fd.setMinimumWidth(1050)
        fd.setMinimumHeight(700)
        fd.setNameFilter(self.__tr('Covise-Case-Files (*.cocase)'))
        fd.setWindowTitle(self.__tr('save file dialog'))
        fd.setDirectory(self._currentFilePath)

        acceptedOrRejected = fd.exec_()
        if acceptedOrRejected != QtWidgets.QDialog.Accepted :
            return
        filenamesQt = fd.selectedFiles()
        if filenamesQt.isEmpty():
            return
        self._currentFilePath = os.path.dirname(str(filenamesQt[0]))
        aFilename = str(filenamesQt[0])    
        # make sure to end with '.cocase'
        if aFilename.rfind('.cocase') == -1 :
            aFilename += '.cocase'
        cocase = self._cocaseCentral.toCoviseCaseFile(self._currentFilePath)
        output = open(aFilename, 'wb')
        pickle.dump(cocase, output)
        output.close()

    def filePrint(self):
        print("CocaseEditor.filePrint(): Not implemented yet")

    def fileExit(self):
        self.close()
        # finish python interface
        sys.exit()

    def _getCoviseFilename(self, description):
        fd = QtWidgets.QFileDialog(self)
        fd.setMinimumWidth(1050)
        fd.setMinimumHeight(700)
        fd.setNameFilter(self.__tr(('Covise-%s-Files (*.covise)') % description ))
        fd.setWindowTitle(self.__tr(('Choose Covise %s File') % description ))
        fd.setDirectory(self._currentFilePath)

        acceptedOrRejected = fd.exec_()
        if acceptedOrRejected != QtWidgets.QDialog.Accepted :
            return
        filenamesQt = fd.selectedFiles()
        if filenamesQt.isEmpty():
            return
        self._currentFilePath = os.path.dirname(str(filenamesQt[0]))
        return str(filenamesQt[0])    

    def addGeoItemSlot(self):
        aFilename = self._getCoviseFilename('Geometry')
        if not aFilename: return
        name = os.path.basename(aFilename)
        name = name[0:name.rfind('.covise')]
        id = self._cocaseCentral.addItem(-1, PartItem('2D', name, aFilename))
        self._updateListView()
        self._setCurrentId(id)

    def _setCurrentId(self, id):
        assert id in self._lviIdMap.values()
        # get lvi to id:
        lvi = None
        for lvi2, id2 in iter(self._lviIdMap.items()):
            if id2 == id: lvi = lvi2
        assert lvi is not None
        self.selectedLVI = lvi
        #self.listView1.setSelected ( lvi, True )
        selectionFlag = QtGui.QItemSelectionModel.ClearAndSelect
        self.listView1.selectionModel().select(lvi.index(), selectionFlag)
        self._currentId = id

    def addVarItemSlot(self):
        pid = self._getParentIdForVarCreate()
        if pid is None:
            self.statusBar().message(
                self.__tr('Refusing variable add without parent.'))
            return
        aFilename = self._getCoviseFilename('Data')
        if not aFilename: return
        name = os.path.basename(aFilename)
        name = name[0:name.rfind('.covise')]
        id = self._cocaseCentral.addItem(pid, PartItem('S', name, aFilename))
        self._updateListView()
        self._setCurrentId(id)

    def deleteGeoItemSlot(self):
        for id in self._selectedIds():
            self._cocaseCentral.deleteItem(id)
        self._updateListView()
        self._delItem.setEnabled(False)
        self.groupBox1.setEnabled(False)
        self.groupBox2.setEnabled(False)
        self._currentId = None

    def _selectedIds(self):
        """Return list of all selected ids."""
        return list(map(lambda item: self._lviIdMap[item], SelectedItemsIterator(self.listView1)))

    def _updateListView(self):
        self.listView1.selectionModel().clear() 
        self.listViewModel.clear()
        self._lviIdMap.clear()
        for node in self._cocaseCentral._rootNode.children():
            id = node.key()
            item = self._cocaseCentral._idPartMap[id]
            lvi = QtGui.QStandardItem(item.name)
            self.listViewModel.appendRow(lvi)
            self._lviIdMap[lvi] = id
            for childNode in node.children():
                c_id = childNode.key()
                item = self._cocaseCentral._idPartMap[c_id]
                c_lvi = QtGui.QStandardItem(item.name)
                lvi.appendRow(c_lvi)
                self._lviIdMap[c_lvi] = c_id
            self.listView1.setExpanded(lvi.index(), True)

    def _getParentIdForVarCreate(self):
        """Return id of a selected toplevel item if possible, else None."""
        #for lvi in ifilter(lambda x: x.isSelected(), FlatLVIterator(self.listView1)):
        if self.selectedLVI:
            return self._lviIdMap[self.selectedLVI]
        return None


    def __tr(self,s,c = None):
        return coTranslate(s)

def main():
    a = QtWidgets.QApplication(sys.argv)
    a.lastWindowClosed().connect(quit)
    Form = QtWidgets.QMainWindow()
    w = CocaseEditor(Form)
    w.show()
    #make sure to end the python interface after closing
    sys.exit(a.exec_())

