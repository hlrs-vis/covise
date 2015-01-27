"""Unittests

suite() -- return the suite of tests of this file
python <thisfile> -- run the suite of this file


Copyright (c) 2006-2007 Visenso GmbH

"""

import unittest
import sys

from qt import *

# Disable info-printing.  Info-printing disturbs the
# output while test runs.
from printing import InfoPrintCapable
InfoPrintCapable.masterSwitch = False # True


import test_Neg2GuiMessages

def suite():
    return unittest.TestSuite(
        (
        test_Neg2GuiMessages.suite(),
        ))


def _main():
    import sys
    app = QApplication(sys.argv)
    runner = unittest.TextTestRunner()
    runner.run(suite())
    sys.exit()

if __name__ == "__main__":
    _main()

# eof
