
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH


import os.path
import os
import time

from PyQt5 import QtCore, QtGui, QtWidgets

import StaticImages_rc

from CasesFilter import CasesFilter
from CasesFilterHelpBase import Ui_CasesFilterHelpBase
class CasesFilterHelpBase(QtWidgets.QWidget, Ui_CasesFilterHelpBase):
    def __init__(self, parent=None, f=QtCore.Qt.WindowFlags()):
        QtWidgets.QWidget.__init__(self, parent, f)
        self.setupUi(self)
from DataSetsSelection import DataSetsSelection
from DataSetsSelectionHelpBase import Ui_DataSetsSelectionHelpBase
class DataSetsSelectionHelpBase(QtWidgets.QWidget, Ui_DataSetsSelectionHelpBase):
    def __init__(self, parent=None, f=QtCore.Qt.WindowFlags()):
        QtWidgets.QWidget.__init__(self, parent, f)
        self.setupUi(self)
from ProjectSetUp import ProjectSetUp
from ProjectSetUpHelpBase import Ui_ProjectSetUpHelpBase

class ProjectSetUpHelpBase(QtWidgets.QWidget, Ui_ProjectSetUpHelpBase):
    def __init__(self, parent=None, f=QtCore.Qt.WindowFlags()):
        QtWidgets.QWidget.__init__(self, parent, f)
        self.setupUi(self)
from StartPreparationBase import Ui_StartPreparationBase

from printing import InfoPrintCapable

import Application
import ObjectMgr
import coviseCase

_infoer = InfoPrintCapable()
_infoer.doPrint = False # True # 

from vtrans import coTranslate 

class PatienceEvent(QtCore.QEvent):
    eventType = QtCore.QEvent.Type(65438)
    def __init__(self):
        QtCore.QEvent.__init__(self, self.eventType )


class StartPreparation(QtWidgets.QDialog,Ui_StartPreparationBase):

    """Some 'phases' can be passed through.

    The phases follow an ordering.
    BTW the user can make some settings for each phase.

    """

    def __init__(self, parent=None, cocaseFile=None):
        _infoer.function = str(self.__init__)
        _infoer.write("parent %s cocaseFile %s" %(parent, cocaseFile))
        QtWidgets.QDialog.__init__(self, parent)
        Ui_StartPreparationBase.__init__(self)
        self.setupUi(self)

        # save cocaseFile
        self.__cocaseFilename=cocaseFile

        # handling patience Dialog
        self.pdManager=None
        self.__unspawn = True

        self.__setUpWidget = ProjectSetUp()
        self.__dataSetsSelectionWidget = DataSetsSelection()
        self.__filterDatasetsWidget = CasesFilter()
        
        self.__phaseWidgets = [
            self.__setUpWidget,
            self.__dataSetsSelectionWidget,
            self.__filterDatasetsWidget]
        for phaseWidget in (self.__phaseWidgets):
            self.widgetStack.addWidget(phaseWidget)

        self.__phaseHelpWidgets = [
            ProjectSetUpHelpBase(self.helpStack),
            DataSetsSelectionHelpBase(self.helpStack),
            CasesFilterHelpBase(self.helpStack)]
        for phaseHelpWidget in (self.__phaseHelpWidgets):
            phaseHelpWidget.setupUi(phaseHelpWidget)
            self.helpStack.addWidget(phaseHelpWidget)

        # is cocaseFile widget starts with phase 2
        if(self.__cocaseFilename==None):
            self.__phase = phase = 0
        else :
            self.__dataSetsSelectionWidget.addCasefilename(self.__cocaseFilename)
            self.__phase = phase = 1

        self.widgetStack.show()
        self.helpStack.show()

        self.__setWidgetsForPhase(phase)
        self.__setAppearanceForPhase(phase)
        self.__patienceDialog=None
        self.__proc=None

        self.__dataSetsSelectionWidget.caseFilenamesChange.connect(self.__enableForwardDependentOnCaseFilenames)
        self.__filterDatasetsWidget.addButton.clicked.connect(self.__enableForwardDependentOnFilter)
        self.__filterDatasetsWidget.selectAllButton.clicked.connect(self.__enableForwardDependentOnFilter)
        self.__filterDatasetsWidget.deselectAllButton.clicked.connect(self.__enableForwardDependentOnFilter)
        self.__filterDatasetsWidget.removeButton.clicked.connect(self.__enableForwardDependentOnFilter)


    def back(self):
        _infoer.function = str(self.back)
        _infoer.write("")
        assert not self.__isFirstPhase(self.phase())
        if 2 == self.phase(): self.__unsetCasesAndConnectionsForPhase2()
        self.__phase -= 1
        self.__setWidgetsForPhase(self.phase())
        self.__setAppearanceForPhase(self.phase())

    def forward(self):
        _infoer.function = str(self.forward)
        _infoer.write("")
        if self.__isLastPhase(self.phase()): # this is finish
            ev = PatienceEvent()
            QtWidgets.QApplication.postEvent(self, ev)
            return
        if 1 == self.phase():
            self.__setCasesAndConnectionsForPhase2() # 2 here == 3 in gui!
        self.__phase += 1
        self.__setWidgetsForPhase(self.phase())
        self.__setAppearanceForPhase(self.phase())

    def phase(self):
        _infoer.function = str(self.__init__)
        _infoer.write("")
        return self.__phase

    def numberOfPhases(self):
        _infoer.function = str(self.numberOfPhases)
        _infoer.write(" %s" %(len(self.__phaseWidgets)))
        return len(self.__phaseWidgets)

    def customEvent(self,e):
        _infoer.function = str(self.customEvent)
        _infoer.write("")
        if e.type() == PatienceEvent.eventType:
            self.__spawnPatienceDialog()
            
            ObjectMgr.ObjectMgr().initProject()
            
            # import case-files
            ObjectMgr.ObjectMgr().importCases(self.__filterDatasetsWidget.getChoice())
            
            # import non-case-files
            for datasetFilename in self.__dataSetsSelectionWidget.datasetFilenames():
                ObjectMgr.ObjectMgr().importFile(datasetFilename)
                
            self.__setUpWidget.updateProjectInformation()
            
            # dialog is unspawn in message handler
            self.__unSpawnPatienceDialog() 
            #Application.vrpApp.mw.globalAccessToTreeView().clearSelection()
            Application.vrpApp.mw.WidgetStackRight.setCurrentWidget(Application.vrpApp.mw.noSelectionPanel)
            Application.vrpApp.mw.advice.raiseInStack(Application.vrpApp.mw.mainWindowHelp)
            time.sleep(0.5)
            self.close()            

    def __setWidgetsForPhase(self, phase):
        _infoer.function = str(self.__setWidgetsForPhase)
        _infoer.write("")
        self.widgetStack.setCurrentWidget(self.__phaseWidgets[phase])
        self.helpStack.setCurrentWidget(self.__phaseHelpWidgets[phase])

    def __setAppearanceForPhase(self, phase):
        _infoer.function = str(self.__setAppearanceForPhase)
        _infoer.write("")
        self.__indicatePhase(phase)
        self.__accomodateBackForward(phase)

    def __indicatePhase(self, phase):
        _infoer.function = str(self.__indicatePhase)
        _infoer.write("")
        phasesIndicators = [
            self.ProjectSetUpLabel,
            self.DataSetsSelectionLabel,
            self.FilterDatasetsLabel]

        for indicator in phasesIndicators:
            indicator.setEnabled(indicator == phasesIndicators[phase])

        if phase == 0:
            self.pixmapLabel1.setPixmap(QtGui.QPixmap(":/1-active.png"))
            self.pixmapLabel2.setPixmap(QtGui.QPixmap(":/2-inactive.png"))
            self.pixmapLabel3.setPixmap(QtGui.QPixmap(":/3-inactive.png"))
        elif phase == 1:
            self.pixmapLabel1.setPixmap(QtGui.QPixmap(":/1-inactive.png"))
            self.pixmapLabel2.setPixmap(QtGui.QPixmap(":/2-active.png"))
            self.pixmapLabel3.setPixmap(QtGui.QPixmap(":/3-inactive.png"))
        elif phase == 2:
            self.pixmapLabel1.setPixmap(QtGui.QPixmap(":/1-inactive.png"))
            self.pixmapLabel2.setPixmap(QtGui.QPixmap(":/2-inactive.png"))
            self.pixmapLabel3.setPixmap(QtGui.QPixmap(":/3-active.png"))

    def __accomodateBackForward(self, phase):
        _infoer.function = str(self.__accomodateBackForward)
        _infoer.write("")
        self.backButton.setEnabled(not self.__isFirstPhase(phase))
        self.nextButton.setEnabled(True)
        if 1 == self.phase():
            self.__enableForwardDependentOnCaseFilenames()
        if 2 == self.phase():
            self.__enableForwardDependentOnFilter()
        if self.__isLastPhase(phase):
            nextButtonText = self.__tr('&Finish')
        else:
            nextButtonText = self.__tr('&Next >')
        self.nextButton.setText(nextButtonText)

    def __isFirstPhase(self, phase):
        _infoer.function = str(self.__isFirstPhase)
        _infoer.write("")
        return 0 == phase

    def __isLastPhase(self, phase):
        _infoer.function = str(self.__isLastPhase)
        _infoer.write("")
        return phase == self.numberOfPhases() - 1

    def __setCasesAndConnectionsForPhase2(self):

        """ATTENTION: This function is very pseudo.  It
        does something useful in special cases.  Seems
        this is at the very first call .
        """

        _infoer.function = str(self.__setCasesAndConnectionsForPhase2)
        _infoer.write("")

        for filename in self.__dataSetsSelectionWidget.caseFilenames():
            _infoer.write("filename: %s" %(filename) )
            namedCase = coviseCase.NameAndCoviseCase()
            namedCase.setFromFile(filename)
            dsc = coviseCase.coviseCase2DimensionSeperatedCase(
                namedCase.case, namedCase.name, os.path.dirname(filename))
            self.__filterDatasetsWidget.addDimensionSeperatedCase(dsc)

    def __unsetCasesAndConnectionsForPhase2(self):
        _infoer.function = str(self.__unsetCasesAndConnectionsForPhase2)
        _infoer.write("")
        self.__filterDatasetsWidget.erase()

    def __enableForwardDependentOnCaseFilenames(self):
        _infoer.function = str(self.__enableForwardDependentOnCaseFilenames)
        _infoer.write("")
        # nothing to be done, we allow projects without casefiles

    def __enableForwardDependentOnFilter(self):
        _infoer.function = str(self.__enableForwardDependentOnFilter)
        _infoer.write("")
        # if we have a casefile, something should have been added
        allCases = self.__filterDatasetsWidget.getAllCases()
        choiceFine,choiceRough = self.__filterDatasetsWidget.getChoice()
        self.nextButton.setEnabled((len(allCases) == 0) or (len(choiceFine) > 0))

    def __tr(self, s, c = None):
        _infoer.function = str(self.__tr)
        _infoer.write("")
        return coTranslate(s)

    def __spawnPatienceDialog(self):
        _infoer.function = str(self.__spawnPatienceDialog)
        _infoer.write("")
        Application.vrpApp.mw.spawnPatienceDialog()

    def __unSpawnPatienceDialog(self):
        _infoer.function = str(self.__unSpawnPatienceDialog)
        _infoer.write("")
        Application.vrpApp.mw.unSpawnPatienceDialog()

# eof
