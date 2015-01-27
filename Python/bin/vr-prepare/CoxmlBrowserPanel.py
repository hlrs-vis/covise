
from PyQt5 import QtCore, QtGui

import sip

from CoxmlBrowserPanelBase import Ui_CoxmlBrowserPanelBase
from CoxmlBrowserPanelConnector import CoxmlBrowserPanelConnector

import os
import re
import xml.dom.minidom

import covise

from printing import InfoPrintCapable

_infoer = InfoPrintCapable()
_infoer.doPrint = False #True #

MODEL_CAPTION = u"Model"

class CoxmlPart():

    def __init__(self):
        self.name = ""
        self.productLine = ""
        self.model = []
        self.variants = {} # variantName -> [variant1, variant2, ...]
        self.description = ""
        self.filename = ""
        self.imagefilename = ""


class CoxmlFilterProxy(QtCore.QSortFilterProxyModel):

    def __init__(self, parent, parts):
        QtCore.QSortFilterProxyModel.__init__(self, parent)
        self.productLine = ""
        self.variants = {}
        self.parts = parts

    def filterAcceptsRow(self, source_row, source_parent):
        source_index = self.sourceModel().index(source_row, 0, source_parent)
        item = self.sourceModel().itemFromIndex(source_index)
        if not item in self.parts:
            return False
        part = self.parts[item]
        if (part.productLine != self.productLine):
            return False
        for v_name, v_value in iter(self.variants.items()):
            if v_name == MODEL_CAPTION:
                if not v_value in part.model:
                    return False
            else:
                if not v_name in part.variants:
                    return False
                if not v_value in part.variants[v_name]:
                    return False
        return True

    def setProductLine(self, productLine):
        self.productLine = productLine
        self.variants = {}
        self.invalidate()

    def addVariant(self, v_name, v_value):
        self.variants[v_name] = v_value
        self.invalidate()

    def removeVariant(self, name):
        if name in self.variants:
            del self.variants[name]
            self.invalidate()


class CoxmlBrowserPanel(Ui_CoxmlBrowserPanelBase):
    def __init__(self, parent=None):
        _infoer.function = str(self.__init__)
        _infoer.write("")
        Ui_CoxmlBrowserPanelBase.__init__(self, parent)
        self.setupUi(self)

        self.__mainWidget = None

        self.__pattern = re.compile(".*\.coxml$")

        # add a box for the widgets in the filterArea (ScrollArea).
        self.filterBox = QtWidgets.QFrame(self.filterArea)
        self.filterArea.setWidget(self.filterBox)

        # product line mapping from buttons to names in the coxml
        self.plButtonMapping = {
            self.plButton1 : u"Room",
            self.plButton2 : u"Tables",
            self.plButton3 : u"Lights",
            self.plButton4 : u"Ceiling Pendants",
            self.plButton5 : u"Medical Equipment",
            self.plButton6 : u"Accessories"
            }

        self.plButton7.setVisible(False) # we only have 6 product lines
        # tooltips
        for b, s in iter(self.plButtonMapping.items()):
            b.setToolTip(s)

        self.resourceDir = covise.getCoConfigEntry("vr-prepare.Coxml.ResourceDirectory")
        if (self.resourceDir != None):
            self.coxmlDir = self.resourceDir + "/coxml/"

        self.__readDatabase()

        self.__setProductLine(u"Room")

        CoxmlBrowserPanelConnector(self)

    def __readDatabase(self):
        self.parts = {}
        # create model
        self.model = QtGui.QStandardItemModel(self)
        self.proxyModel = CoxmlFilterProxy(self, self.parts)
        self.proxyModel.setSourceModel(self.model)
        self.listView.setModel(self.proxyModel)
        # read coxml files
        if (self.resourceDir != None):
            self.__readDirectory("")
        # sort
        self.proxyModel.sort(0)

    def __readDirectory(self, subdir):
        fullDir = self.coxmlDir + subdir
        try:
            tmpList = os.listdir(fullDir)
        except:
            print("Error: Can't access resource directory", fullDir)
            return
        # files
        coxmlList = [s for s in tmpList if os.path.isfile(fullDir + s) and self.__pattern.match(s)]
        for filename in coxmlList:
            self.__readCoxml(subdir + filename)
        # directories
        subdirList = [s + "/" for s in tmpList if os.path.isdir(fullDir + s)]
        for s in subdirList:
            self.__readDirectory(subdir + s)

    def __readCoxml(self, filename):
        dom = xml.dom.minidom.parse(self.coxmlDir + filename)
        # search classification
        classificationElems = dom.getElementsByTagName("classification")
        if len(classificationElems) != 1:
            return
        classificationElem = classificationElems[0]
        # create part
        part = CoxmlPart()
        part.filename = filename
        # READ name
        elems = classificationElem.getElementsByTagName("name")
        if len(elems) != 1:
            del part
            return
        part.name = unicode(elems[0].getAttribute("value"))
        # READ product_line
        elems = classificationElem.getElementsByTagName("product_line")
        if len(elems) != 1:
            del part
            return
        part.productLine = unicode(elems[0].getAttribute("value"))
        # READ model
        elems = classificationElem.getElementsByTagName("model")
        if (len(elems) == 1):
            model = unicode(elems[0].getAttribute("value")).split(",")
            model = list(map(lambda x: x.strip(), model))
            part.model = model
        else:
            part.model = [""]
        # READ variant information
        for child in classificationElem.getElementsByTagName("variant"):
            variant = unicode(child.getAttribute("value")).split(",")
            variant = list(map(lambda x: x.strip(), variant))
            part.variants[unicode(child.getAttribute("name"))] = variant
        # READ description
        elems = classificationElem.getElementsByTagName("description")
        if (len(elems) == 1) and (elems[0].firstChild != None) and (elems[0].firstChild.nodeType == xml.dom.Node.TEXT_NODE):
            part.description = unicode(elems[0].firstChild.data)
        # READ image filename
        elems = classificationElem.getElementsByTagName("image")
        if len(elems) == 1:
            part.imagefilename = str(elems[0].getAttribute("value"))
        # create item
        item = QtGui.QStandardItem(part.name)
        item.setEditable(False)
        # add item
        self.model.appendRow(item)
        self.parts[item] = part

    def setDependingWidgets(self, mw):
        self.__mainWidget = mw

    def hide(self):
        QtWidgets.QWidget.hide(self)
        self.__mainWidget.update()

    # SLOT
    def listViewClicked(self, index):
        sourceIndex = self.proxyModel.mapToSource(index)
        item = self.model.itemFromIndex(sourceIndex)
        part = self.parts[item]

        # display name
        self.name_label.setText(part.name)

        # display description
        self.descriptionEdit.setPlainText(part.description)

        # display image
        imgPath = self.resourceDir + "/image/" + part.imagefilename
        try:
            icon = QtGui.QPixmap(imgPath)
            icon = icon.scaledToHeight(self.image_label.height(), QtCore.Qt.SmoothTransformation)
        except:
            icon = QtGui.QPixmap()
        self.image_label.setPixmap(icon)

    # SLOT
    def listViewDoubleClicked(self, index):
        sourceIndex = self.proxyModel.mapToSource(index)
        item = self.model.itemFromIndex(sourceIndex)
        part = self.parts[item]
        filename = part.filename
        self.importCoxml.emit(filename)

    # SLOT
    def productLineClicked(self):
        if self.sender() in self.plButtonMapping:
            productLine = self.plButtonMapping[self.sender()]
            self.__setProductLine(productLine);

    # SLOT
    def filterChanged(self, text):
        combo = self.sender()
        # get variant name
        names = [item[0] for item in self.filterElements.items() if item[1][1] == combo]
        if len(names) == 0:
            return
        name = names[0]
        # get selected value
        if (combo.currentIndex() == 0):
            self.proxyModel.removeVariant(name)
        else:
            self.proxyModel.addVariant(name, unicode(combo.currentText()))
        self.__updateFilterElements(False)

    def __setProductLine(self, productLine):
        self.productLine = productLine
        # change filter proxy
        self.proxyModel.setProductLine(productLine);
        # clear name
        self.name_label.setText("")
        # clear description
        self.descriptionEdit.setPlainText("")
        # clear pixmap
        self.image_label.setPixmap(QtGui.QPixmap())
        # prepare filter elements
        self.__updateFilterElements(True)

    def __updateFilterElements(self, reset):

        # collect all (filtered) variants
        allVariants = {}
        allVariants[MODEL_CAPTION] = []
        for i in range(0, self.proxyModel.rowCount()):
            proxyIndex = self.proxyModel.index(i, 0)
            sourceIndex = self.proxyModel.mapToSource(proxyIndex)
            item = self.model.itemFromIndex(sourceIndex)
            if item in self.parts:
                part = self.parts[item]
                for m in part.model:
                    if m != "" and not m in allVariants[MODEL_CAPTION]:
                        allVariants[MODEL_CAPTION].append(m)
                for v_name, v_value in iter(part.variants.items()):
                    if not v_name in allVariants:
                        allVariants[v_name] = []
                    for v in v_value:
                        if v_value != "" and not v in allVariants[v_name]:
                            allVariants[v_name].append(v)
        # sort values
        for v_name, v_values in iter(allVariants.items()):
            v_values.sort()

        # clear old elements and create new ones
        if reset:
            # make list of tuples with MODEL_CAPTION beeing the first one
            orderedVariants = [item for item in allVariants.items() if item[0] != MODEL_CAPTION]
            orderedVariants.insert(0, (MODEL_CAPTION, allVariants[MODEL_CAPTION]))
            # delete old combos and layout (if present)
            if hasattr(self, "filterElements"):
                for label, combo in self.filterElements.values():
                    self.filterBoxLayout.removeWidget(label)
                    label.setParent(None)
                    sip.delete(label)
                    self.filterBoxLayout.removeWidget(combo)
                    combo.setParent(None)
                    sip.delete(combo)
                sip.delete(self.filterBoxLayout) # sip properly removes the underlying C object so another layout can be added to the widget
            self.filterElements = {}
            # create new layout
            self.filterBoxLayout = QtWidgets.QGridLayout(self.filterBox)
            # create all combos
            cnt = 0
            for v_name, v_values in orderedVariants:
                label = QtWidgets.QLabel(self.filterBox)
                label.setText(v_name)
                self.filterBoxLayout.addWidget(label, cnt, 0)
                combo = QtWidgets.QComboBox(self.filterBox)
                self.filterBoxLayout.addWidget(combo, cnt, 1)
                combo.activated.connect(self.filterChanged)
                self.filterElements[v_name] = (label, combo)
                cnt = cnt + 1
            # add spacer
            spacer = QtWidgets.QSpacerItem(16, 16, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
            self.filterBoxLayout.addItem(spacer)
            # set size of filterArea to avoid scrollbar
            self.filterArea.setFixedHeight(12 + cnt*30) # this is bad, but I don't have a better way for now
            
        filterChanged = False

        # create / change items
        for v_name, (label, combo) in iter(self.filterElements.items()):
            if (v_name == MODEL_CAPTION) and not reset:
                continue # the MODEL_CAPTION combo is only changed on reset
            if (v_name in allVariants.keys()):
                if (combo.currentIndex() > 0):
                    continue # dont change the combo if something is selected
                label.setVisible(True)
                combo.setVisible(True)
                # clear items
                while (combo.count() > 0):
                    combo.removeItem(0)
                # add items
                combo.addItem("")
                for value in allVariants[v_name]:
                    combo.addItem(value)
            else:
                label.setVisible(False)
                combo.setVisible(False)
                if (combo.currentIndex() > 0):
                    combo.setCurrentIndex(0)
                    self.proxyModel.removeVariant(v_name)
                    filterChanged = True

        # if the filter was changed, update the filter elements again
        if filterChanged:
            self.__updateFilterElements(False)

