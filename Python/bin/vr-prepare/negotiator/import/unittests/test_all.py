"""File test_all.py

For running all unittests related to the import.

This file must be kept up to date when sub-test-files
have been created or deleted.

Copyright (c) 2006 Visenso GmbH

"""

import unittest

import test_ImportManager

def suite():
    return unittest.TestSuite((
        test_ImportManager.suite(),))

def test_main():
    import sys
    runner = unittest.TextTestRunner()
    runner.run(suite())
    sys.exit()

if __name__=="__main__":
    test_main()
