
from PyQt5 import QtCore, QtGui

#from qttable import *

import sys
from TimestepSelectorDialogBase import Ui_TimestepSelectorDialogBase

class TimestepSelectorDialog(Ui_TimestepSelectorDialogBase):

    def __init__(self, parent=None):
        Ui_TimestepSelectorDialogBase.__init__(self,parent)
        self.setupUi(self)

        self.modelTimesteps = QtGui.QStandardItemModel(self)

        self.listViewTimesteps.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)
        self.listViewTimesteps.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection) # MultiSelection ???
        self.listViewTimesteps.setModel( self.modelTimesteps )

        self.pushButtonOK.clicked.connect(self.accept)
        self.pushButtonCancel.clicked.connect(self.reject)

    def fillListView(self, numTimesteps):
        for i in range(numTimesteps):
            new = QtGui.QStandardItem( str(i) )
            self.modelTimesteps.appendRow( new )

    def getSelectionString(self):

        selectionList = list(map(lambda i: int(self.modelTimesteps.itemFromIndex(i).text()), self.listViewTimesteps.selectionModel().selectedIndexes()))
        selectionList.sort()

        selectionString = ""
        sequenceLength = 0
        for i,v in enumerate(selectionList):
            if sequenceLength == 0:
                selectionString = selectionString + str(v)
                sequenceLength = 1
            elif sequenceLength == 1:
                if v == selectionList[i-1]+1:
                    sequenceLength = 2
                else:
                    selectionString = selectionString + " " + str(v)
                    sequenceLength = 1
            elif sequenceLength == 2:
                if v == selectionList[i-1]+1:
                    sequenceLength += 1
                else:
                    selectionString = selectionString + " " + str(selectionList[i-1]) + " " + str(v)
                    sequenceLength = 1
            else:
                if v == selectionList[i-1]+1:
                    sequenceLength += 1
                else:
                    selectionString = selectionString + "-" + str(selectionList[i-1]) + " " + str(v)
                    sequenceLength = 1

        if sequenceLength == 0:
            pass
        elif sequenceLength == 1:
            pass
        elif sequenceLength == 2:
            selectionString = selectionString + " " + str(selectionList[len(selectionList)-1])
        else:
            selectionString = selectionString + "-" + str(selectionList[len(selectionList)-1])

        return selectionString


    def setFromSelectionString(self, selectionString):
        """ selects entries based on a given selectionString """
        #selectionList = selectionString.split()
        import re
        selectionList = []
        shortSelectionList = re.findall('\d+\s*\-\s*\d+|\d+', selectionString)
        for val in shortSelectionList:
            try:
                i = int(val)
                selectionList.append(i)
            except ValueError:
                t = val.split('-')
                selectionList.extend(range(int(t[0]), int(t[1])+1))

        self.listViewTimesteps.selectionModel().reset() # clear first
        for timestep in selectionList:
            itemlist = self.modelTimesteps.findItems(str(timestep))
            if (len(itemlist) == 1):
                self.listViewTimesteps.selectionModel().select(itemlist[0].index(), QtGui.QItemSelectionModel.Select)

