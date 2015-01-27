"""Unittests

suite() -- return the tests of tests of this file
python <thisfile> -- run the tests of this file


Copyright (c) 2006-2007 Visenso GmbH

"""


import sys
import unittest

from qt import *

from qtauxils import ConnectionBox

from Neg2GuiMessages import (
    GUI_ERROR_SIGNAL,
    GUI_OK_SIGNAL,
    INIT_SIGNAL,
    NEG_ERROR_SIGNAL,
    NEG_OK_SIGNAL,
    PARAM_SIGNAL,
    REQUEST_OBJ_SIGNAL,
    REQUEST_PARAMS_SIGNAL,
    RUN_OBJ_SIGNAL,
    SET_PARAMS_SIGNAL,
    negGuiMsg,
    negGuiObjMsg,
    )


class neg2GuiMsgTestCase(unittest.TestCase):

    def test_getters(self):
        msg = negGuiMsg(GUI_ERROR_SIGNAL, requestNr = 42, parentKey = 43)
        self.assertEqual(PYSIGNAL(GUI_ERROR_SIGNAL), msg.getSignal())
        self.assertEqual(43, msg.getParentKey())

    def test_sendObj(self):
        cb = ConnectionBox()
        msg = negGuiMsg(GUI_ERROR_SIGNAL, requestNr = 42, parentKey = 43)
        QObject.connect(msg, msg.getSignal(), cb.slotSlot)
        anyObject = 42
        msg.sendObj(anyObject)
        self.assertEqual(1, cb.signalArrived)


class negGuiObjMsgTestCase(unittest.TestCase):

    def test_getter(self):
        aKey = 23
        msg = negGuiObjMsg(
            signal = GUI_ERROR_SIGNAL, requestNr = 42, parentKey = 43, key = aKey)
        self.assertEqual(23, msg.getKey())


def suite():
    allTestCases = [
        neg2GuiMsgTestCase,
        negGuiObjMsgTestCase,
        ]
    suites = map(
        lambda x: unittest.defaultTestLoader.loadTestsFromTestCase(x),
        allTestCases)
    collSuite = unittest.TestSuite()
    collSuite.addTests(tuple(suites))
    return collSuite


def _main():
    app = QApplication(sys.argv)
    runner = unittest.TextTestRunner()
    runner.run(suite())

if __name__ == "__main__":
    _main()

# eof
