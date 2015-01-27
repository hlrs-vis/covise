
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject, pyqtSignal

from DataSetsSelectionBase import Ui_DataSetsSelectionBase
import VRPCoviseNetAccess
import os
import covise
import Utils


from printing import InfoPrintCapable

_infoer = InfoPrintCapable()
_infoer.doPrint = False # True # 

from vtrans import coTranslate 

class DataSetsSelection(QtWidgets.QWidget,Ui_DataSetsSelectionBase):

    """For collecting case-files."""

    caseFilenamesChange = pyqtSignal()

    def __init__(self, parent=None):
        _infoer.function = str(self.__init__)
        _infoer.write("")
        QtWidgets.QWidget.__init__(self, parent)
        Ui_DataSetsSelectionBase.__init__(self)
        self.setupUi(self)

        self._caseFilenames = []
        self._datasetFilenames = []
        
        self.ServerhostcheckBox.setChecked(False)
        self.ServerhostcheckBox.setEnabled(False)
        self.ServerhostcheckBox.hide()
            
        self.__cocaseFilenameSuggestion = covise.getCoConfigEntry("vr-prepare.InitialDatasetSearchPath")
        if not self.__cocaseFilenameSuggestion:
            self.__cocaseFilenameSuggestion = os.getcwd()

    def caseFilenames(self):
        _infoer.function = str(self.caseFilenames)
        _infoer.write( "caseFilenames %s" %(str(self._caseFilenames)) )
        return self._caseFilenames

    def datasetFilenames(self):
        _infoer.function = str(self.datasetFilenames)
        _infoer.write( "datasetFilenames %s" %(str(self._datasetFilenames)) )
        return self._datasetFilenames

    def certainServerHostWanted(self):
        _infoer.function = str(self.certainServerHostWanted)
        _infoer.write("")
        return self.ServerhostcheckBox.isChecked()

    def addCasefilename(self, aName):
        _infoer.function = str(self.addCasefilename)
        _infoer.write(str(aName))
        self._caseFilenames.append(aName)
        self.listBox1.addItem(str(aName))
        self.caseFilenamesChange.emit()

        # TODO (move): to a better place.
        if self.certainServerHostWanted():
            Utils.addServerHostFromConfig()
                
    def addDatasetFilename(self, aName):
        _infoer.function = str(self.addDatasetFilename)
        _infoer.write(str(aName))
        self._datasetFilenames.append(aName)
        self.listBox1.addItem(str(aName))
        self.caseFilenamesChange.emit()
        

    def removeSelectedCasefilenames(self):
        _infoer.function = str(self.removeSelectedCasefilenames)
        _infoer.write("")
        selected = []
        for i in range(self.listBox1.count()):
            if self.listBox1.isItemSelected(self.listBox1.item(i)):
                #                selected.append(i)
                selected.insert(0, i) # reverse sort for removal
        for i in selected:
            itemName = str(self.listBox1.item(i).text())
            self.listBox1.takeItem(i)
            if itemName in self._caseFilenames:
                del self._caseFilenames[self._caseFilenames.index(itemName)]
                #print("found item in case")
            elif itemName in self._datasetFilenames:
                del self._datasetFilenames[self._datasetFilenames.index(itemName)]
                #print("found item in dataset")
            else:
                print("Error: couldn't find selected Case or dataset in internal structure!")
        self.caseFilenamesChange.emit()

    def letUserAddCasefilename(self):
        _infoer.function = str(self.letUserAddCasefilename)
        _infoer.write("")
        #filetypes = 'Case-files (*.cocase)\n'
        filetypes = Utils.getImportFileTypes()
        filenamesQt = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            self.__tr('Add Datasets'),
            self.__cocaseFilenameSuggestion,
            filetypes)

        if filenamesQt == "":
            self.statusMessage.emit(('Choosing filename cancelled',))
            return
            
        for filenameQt in filenamesQt:
            filename = str(filenameQt)
            load = True
            if not os.access(filename, os.R_OK):
                QtWidgets.QMessageBox.information(
                    self,
                    covise.getCoConfigEntry("vr-prepare.ProductName"),
                    self.__tr("The file \"")
                    + filenameQt
                    + self.__tr("\" is not accessable.\n")
                    + self.__tr("You may check the permissions."),
                    self.__tr("&Ok"),
                    "",
                    "",
                    0,
                    0)
                load = False
                #return
            if load:
                if str(filenameQt).endswith(".cocase") :
                    self.addCasefilename(str(filenameQt))
                elif os.path.splitext(str(filenameQt))[1].lower() in Utils.getImportFileTypesFlat():
                    self.addDatasetFilename(str(filenameQt))
                else:
                    print("Warning: Trying to add file of unknown type: ", str(filenameQt))
                self.__cocaseFilenameSuggestion = str(filenameQt)

    def __tr(self, s, c = None):
        _infoer.function = str(self.__tr)
        _infoer.write("")
        return coTranslate(s)

# eof
