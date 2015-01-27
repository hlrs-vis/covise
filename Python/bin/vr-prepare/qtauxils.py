
# Copyright (c) 2006 Visenso GmbH

# Initial: 2006-12-20, mw@visenso.de

"""Collection of qt-dependent entities.

Hopefully they ease the developers life.

"""

from PyQt5 import QtCore, QtGui


# Connection testing

class ConnBox(QtCore.QObject):

    """Class for qt-connection testing.

    This class is of use to test classes that use the
    signal/slot mechanism.

    Based on http://www.opendocspublishing.com/pyqt.
    (Could not refind the source at 2006-07-28, mw@visenso.de.)

    """

    def __init__(self, *args):
        apply(QtCore.QObject.__init__,(self,)+args)
        self.signalArrived = 0
        self.args = []
        self.arg = None

    def slotSlot(self, *args):
        self.signalArrived = 1
        self.args = args

    def slotSetOneArgToStr(self, arg):
        self.signalArrived = 1
        self.arg = str(arg)


class ConnectionBox(ConnBox):

    """Provide some asserts for detailed signal-testing."""

    def __init__(self, *args):
        ConnBox.__init__(self, *args)

    def assertSignalArrived(self, signal=None):
        if  not self.signalArrived:
            raise AssertionError("signal %s did not arrive" % signal)

    def assertNumberOfArguments(self, number):
        if number != len(self.args):
            raise AssertionError\
                  ("Signal generated %i arguments, but %i were expected" %
                                    (len(self.args), number))

    def assertArgumentTypes(self, *args):
        if len(args) != len(self.args):
            raise AssertionError\
         ("Signal generated %i arguments, but %i were given to this function" %
                                 (len(self.args), len(args)))
        for i in range(len(args)):
            if type(self.args[i]) != args[i]:
                raise AssertionError\
                      ( "Arguments don't match: %s received, should be %s." %
                                      (type(self.args[i]), args[i]))

# TreeView

def itemFromProxyIndex(proxyModel, index):
    return proxyModel.sourceModel().itemFromIndex(proxyModel.mapToSource(index))

def SelectedItemsIterator(view):
    if hasattr(view.model(), "sourceModel"): # proxy
        for index in view.selectionModel().selectedIndexes():
            yield itemFromProxyIndex(view.model(), index)
    else:
        for index in view.selectionModel().selectedIndexes():
            yield view.model().itemFromIndex(index)

# As it is, it's not working with QT4!
def FlatLVIterator(parent):
    """Iterate over first level children of a QListView or a QListViewItem.

    Taken from
    http://www.diotavelli.net/PyQtWiki/ListBoxAndListViewIterators
    2007-01-11, mw@visenso.de.

    """
    child = parent.child(0)
    while child:
        yield child
        child = child.child(0)
    """
    child = parent.firstChild()
    while child:
        yield child
        child = child.nextSibling()
    """

# As it is, it's not working with QT4!
def DeepLVIterator(parent):
    """Iterate over all children of a QListView or a QListViewItem.

    Taken from
    http://www.diotavelli.net/PyQtWiki/ListBoxAndListViewIterators
    2007-01-11, mw@visenso.de.

    """

    child = parent.child(0)
    while child:
        yield child
        for c in DeepLVIterator(child):
            yield c
        child = child.child(0)
    """    
    child = parent.firstChild()
    while child:
        yield child
        for c in DeepLVIterator(child):
            yield c
        child = child.nextSibling()
    """

def hasSelectedSubItem(parent):
    for item in DeepLVIterator(parent):
        if item.isSelected(): return True
    return False

# eof
