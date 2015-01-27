
# Part of the vr-prepare program for dc

# Copyright (c) 2006-2007 Visenso GmbH


import sys

from PyQt5 import QtCore, QtGui, QtWidgets

import Application

from TransformManager import TransformManager
from DatasetInformationPanelBase import Ui_DatasetInformationPanelBase
from ObjectMgr import ObjectMgr
from coCaseMgr import coCaseMgrParams

from vtrans import coTranslate 

class DatasetInformationPanel(QtWidgets.QWidget, Ui_DatasetInformationPanelBase, TransformManager):
    """Suitable to let the user see data related to a dataset."""
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        Ui_DatasetInformationPanelBase.__init__(self)
        self.setupUi(self)
        TransformManager.__init__(self, self.emitDataChanged, True)

        self.__key = -1

        self.lineEditNameDataset.setText(coTranslate('<ds-filename not set yet>'))
        self.__dsNumberGeometryObjects = 0
        self.__dsNumberOfGrids = 0
        self.__dsConversionDate = coTranslate('<ds-date not set yet>')
        self.__inUseNumberGeometryObjects = 0
        self.__inUseNumberOfGrids = 0
        self.__updateFromIntern()
        # hide dummy information for "content"
        self.vrpFrameInformation.hide()

        #validators:
        # allow only double values for transform lineEdits
        doubleValidator = QtGui.QDoubleValidator(self)
        self.floatX.setValidator(doubleValidator)
        self.floatY.setValidator(doubleValidator)
        self.floatZ.setValidator(doubleValidator)
    
    def updateForObject( self, key ):
        """ called from MainWindow to update the content to the choosen object key """
        self.__key = key
        params = ObjectMgr().getParamsOfObject(key)
        self.__setParams( params )        
        
        
    def __getParams(self):
        data = coCaseMgrParams()
        data.name = str(self.lineEditNameDataset.text())
        # Transformation
        self.TransformManagerGetParams(data)
        return data

    def __setParams(self, params):
        self.TransformManagerBlockSignals(True)

        self.setDSFilename(params.name)
        # Transformation
        self.TransformManagerSetParams(params)
        if params.origDsc and params.filteredDsc:
            # update dataset-panel
            self.setDSNumberGeometryObjects(params.origDsc.getNum2dParts())
            self.setDSNumberOfGrids(params.origDsc.getNum3dParts())
            self.setInUseNumberGeometryObjects(params.filteredDsc.getNum2dParts())
            self.setInUseNumberOfGrids(params.filteredDsc.getNum3dParts())

        self.TransformManagerBlockSignals(False)

    # any data has changed
    def emitDataChanged(self):
        if not self.__key==-1:
            params = self.__getParams()
            Application.vrpApp.key2params[self.__key] = params
            ObjectMgr().setParams( self.__key, params )


    def setDSFilename(self, aName):
        self.lineEditNameDataset.setText(aName)
    def setDSNumberGeometryObjects(self, aNumber):
        self.__dsNumberGeometryObjects = aNumber
        self.__updateFromIntern()
    def setDSNumberOfGrids(self, aNumber):
        self.__dsNumberOfGrids = aNumber
        self.__updateFromIntern()
    def setInUseNumberGeometryObjects(self, aNumber):
        self.__inUseNumberGeometryObjects = aNumber
        self.__updateFromIntern()
    def setInUseNumberOfGrids(self, aNumber):
        self.__inUseNumberOfGrids = aNumber
        self.__updateFromIntern()
    def __updateFromIntern(self):
        casefileInfoFmtString = coTranslate('''Format: "COVISE casefile".
Converted from EnSight 6.0 to COVISE.
Conversion date: %ss.''')
        
        newText = casefileInfoFmtString % self.__dsConversionDate
        
        self.textEditInformation.setText(newText)
        caseUseInfoFmtString = coTranslate('''All available Objects:
%(numAllGeoObjects)6d Geometry Objects
%(numAllGridObjects)6d Grids

Selected Objects for COVISE Project:
%(numInUseGeoObjects)6d Geometry Objects
%(numInUseGridObjects)6d Grids''')
        self.textEditContent.setText(caseUseInfoFmtString % {
            'numAllGeoObjects': self.__dsNumberGeometryObjects,
            'numAllGridObjects': self.__dsNumberOfGrids,
            'numInUseGeoObjects': self.__inUseNumberGeometryObjects,
            'numInUseGridObjects': self.__inUseNumberOfGrids,
            })

# eof
