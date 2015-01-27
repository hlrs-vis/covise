
# Part of the vr-prepare program for dc

# Copyright (c) 2006 Visenso GmbH

from PyQt5 import QtCore, QtGui

import os
import covise

from printing import InfoPrintCapable

from SceneObjectListWindowBase import Ui_SceneObjectListWindowBase

import Application
import ObjectMgr

from KeydObject import VIS_SCENE_OBJECT

from Utils import ReallyWantToOverrideAsker

from vtrans import coTranslate

class SceneObjectListWindow(Ui_SceneObjectListWindowBase):


    def __init__(self, parent=None):
        Ui_SceneObjectListWindowBase.__init__(self, parent)
        self.saveButton.clicked.connect(self._save)
        self.setupUi(self)

        objects = []
        tempObjects = ObjectMgr.ObjectMgr().getAllElementsOfType(VIS_SCENE_OBJECT)
        for so in tempObjects:
            if (so.params.product_line == u"Lights") or (so.params.product_line == u"Ceiling Pendants"): 
                objects.append(so)

        text = unicode(coTranslate("List of objects\n"))
        text = text + u"==============================================\n"
        text = text + u"\n"

        for so in objects:
            text = text + so.params.name + u"\n"
            
        text = text + u"\n"
        text = text + u"\n"
        text = text + unicode(coTranslate("Details\n"))
        text = text + u"==============================================\n"

        for so in objects:
            text = text + u"\n"
            text = text + so.params.name + u"\n"
            text = text + so.params.description + u"\n"
            text = text + u"----------------------------------------------\n"

        self.textBrowser.setPlainText(text)


    def _save(self):

        filenameQt = QtWidgets.QFileDialog.getSaveFileName(
            self,
            coTranslate('Create List'),
            coTranslate("ObjectList.txt"),
            coTranslate('TextFiles (*.txt)'),
            None,
            QtWidgets.QFileDialog.DontConfirmOverwrite)
        if filenameQt == "":
            return
        filename = str(filenameQt)
        if not filename.lower().endswith(".txt"):
            filename += ".txt"
        if  os.path.exists(filename):
            asker = ReallyWantToOverrideAsker(self, filename)
            decicion = asker.exec_()
            if decicion == QtWidgets.QDialog.Rejected:
                self.statusBar().showMessage(
                    coTranslate('Cancelled overwrite of "%s"') % filename)
                return

        doc = open(filename, mode="w")
        doc.write(unicode(self.textBrowser.toPlainText()).encode('utf8'))
        doc.close()


