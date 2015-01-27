"""Unittests

suite() -- return the suite of tests of this file
Start: covise --script <name of this file>

(Environment-variable PYTHONPATH can be used to let
python find the necessary modules including this
one.)

Copyright (c) 2008 Visenso GmbH

"""

import os.path
import sys
import unittest

from ImportManager import ImportModule, Import2DModule, Import3DModule
from ImportSampleManager import ImportSample3DModule
from ImportGroupManager import ImportSimpleGroupModule, ImportGroup3DModule, ImportGroup2DModule, ImportSimpleGroup2DModule, COMPOSED_VELOCITY
from ImportTransformManager import ImportTransform2DModule
from ImportGroupAssembleUsg import ImportGroupAssembleUsg
from ImportGroupReduceSet import ImportGroupReduceSet
from VRPCoviseNetAccess import theNet, connect, ConnectionPoint, saveExecute, globalRenderer
from coPyModules import BoundingBox, Colors, Collect, Renderer
from coviseCase import NameAndCoviseCase, coviseCase2DimensionSeperatedCase
import coviseStartup
from BoundingBox import BoundingBoxParser


class ImportModuleTestCase(unittest.TestCase):
    def _Case(self):
        nameAndCase = NameAndCoviseCase()
        nameAndCase.setFromFile('/work/common/Projekte/DC-CFDGui/datasets/w221/PERMAS_Temp_t0/W221_powerwall.cocase')
        return coviseCase2DimensionSeperatedCase(nameAndCase.case,nameAndCase.name, nameAndCase.pathToCaseFile)

    def _ComposedCase(self):
        nameAndCase = NameAndCoviseCase()
        nameAndCase.setFromFile('/data/Kunden/Kaercher/demo/pumpe.cocase')
        return coviseCase2DimensionSeperatedCase(nameAndCase.case,nameAndCase.name, nameAndCase.pathToCaseFile)

    def _TransientCase(self):
        nameAndCase = NameAndCoviseCase()
        nameAndCase.setFromFile('/work/common/svn/branches/Covise6.0/covise/src/application/ui/vr-prepare/negotiator/import/unittests/data/transientTiny.cocase')
        return coviseCase2DimensionSeperatedCase(nameAndCase.case,nameAndCase.name, nameAndCase.pathToCaseFile)
    
    def _ImportModule1(self):
        dsc = self._Case()
        print("Loading ",  dsc.parts2d[8].name)
        return Import2DModule( dsc.parts2d[8] )

    def _ImportModule2(self):
        dsc = self._Case()
        print("Loading ",  dsc.parts2d[8].name)
        return ImportTransform2DModule( dsc.parts2d[8] )

    def _TransientGroup(self):
        self.__currentImportGroupModules = []
        dsc = self._TransientCase()
        p1 = Import2DModule( dsc.parts2d[0] )
        self.__currentImportGroupModules.append(p1)
        p2 = Import2DModule( dsc.parts2d[1] )
        self.__currentImportGroupModules.append(p2)
        gip = ImportGroup2DModule(True)
        gip.addImport(p1)
        gip.addImport(p2)
        return gip
    
    def _SampleGroup(self):
        self.__currentImportGroupModules = []
        dsc = self._ComposedCase()
        p1 = ImportSample3DModule( dsc.parts3d[0] )
        self.__currentImportGroupModules.append(p1)
        p2 = ImportSample3DModule( dsc.parts3d[1] )
        self.__currentImportGroupModules.append(p2)
        gip = ImportGroup3DModule()
        gip.addImport(p1, 'Vels')
        gip.addImport(p2, 'vel')
        return gip

    def _LargeGroup(self):
        self.__currentImportGroupModules = []
        nameAndCase = NameAndCoviseCase()
        nameAndCase.setFromFile('/work/common/Projekte/DC-CFDGui/datasets/w221/PERMAS_Temp_t0/W221_powerwall.cocase')
        dsc = coviseCase2DimensionSeperatedCase(nameAndCase.case,nameAndCase.name, nameAndCase.pathToCaseFile)
        gip = ImportGroup2DModule()
        for part in dsc.parts2d:
            ip = Import2DModule(part)
            self.__currentImportGroupModules.append(ip)
            print(str(ip))
            gip.addImport(ip)
        return gip

    def _LargeSimpleGroup(self):
        self.__currentImportGroupModules = []
        nameAndCase = NameAndCoviseCase()
        nameAndCase.setFromFile('/work/common/Projekte/DC-CFDGui/datasets/w221/PERMAS_Temp_t0/W221_powerwall.cocase')
        dsc = coviseCase2DimensionSeperatedCase(nameAndCase.case, nameAndCase.name, nameAndCase.pathToCaseFile)
        gip = ImportSimpleGroupModule(2)
        for part in dsc.parts2d:
            ip = Import2DModule(part)
            self.__currentImportGroupModules.append(ip)
            print(str(ip))
            gip.addImport(ip)
        return gip

    def _RemoveImportGroupModules(self):
        for ip in self.__currentImportGroupModules:
            ip.delete()

    #
    # test functions
    #

    def test_Geo(self):
        moduleCount = theNet().moduleCount()
        ip = self._ImportModule1()
        box = ip.getBox()
        (x,y,z) = box.getCenter()
        self.assertEqual((x,y,z), (-0.041321500000000011, 0.0, 0.1766115))
        # delete
        ip.delete()
        self.assertEqual(theNet().moduleCount(), moduleCount)

    def test_Data(self):
        moduleCount = theNet().moduleCount()
        ip = self._ImportModule1()
        value = 'Temperature'
        (a,b) = ip.getDataMinMax(value)
        self.assertEqual((a,b), (78.848465, 446.557648))
        # delete
        ip.delete()
        self.assertEqual(theNet().moduleCount(), moduleCount)

    def testTransform(self):
        moduleCount = theNet().moduleCount()
        bias = 0.1

        ip = self._ImportModule2()
        box = ip.getBox()
        (preX,preY,preZ) = box.getCenter()

        ip.setRotation(20.0, 1.0, 0.0, 0.0)
        ip.setTranslation(0.0, bias, bias)

        box = ip.getBox(True) # execute
        (postX,postY,postZ) = box.getCenter()

        theNet().save("test_Transform.net")

        self.assertEqual(preX, postX)
        self.assert_(preY != postY)
        self.assert_(preZ != postZ)

        # delete
        ip.delete()
        self.assertEqual(theNet().moduleCount(), moduleCount)


    def testSample(self):
        moduleCount = theNet().moduleCount()
        #ip = testSampleImportModule1()
        ip = self._SampleGroup()
        ip.executeGeo()
        #ip.executeOct()
#        ip.executeSampleData(COMPOSED_VELOCITY) # commented out: does not work at the moment
        #(a,b) = ip.getDataMinMax('rotvel')#COMPOSED_VELOCITY)
        #print("min, max " , a, b)
        #ip.executeSampleData('Temperature')
        # delete
        ip.delete()
        self._RemoveImportGroupModules()
        self.assertEqual(theNet().moduleCount(), moduleCount)

    def testGroupGeo(self):
        moduleCount = theNet().moduleCount()
        gip = self._LargeGroup()
        bp = gip.getBox()
        (x,y,z) = bp.getCenter()
        self.assertEqual((x,y,z), (1.6700575000000002, 0.0, 0.40254699999999999))
        # delete
        gip.delete()
        self._RemoveImportGroupModules()
        self.assertEqual(theNet().moduleCount(), moduleCount)

    def test_TransientReduceSetGeo(self):
        moduleCount = theNet().moduleCount()
        gip_core = self._TransientGroup()
        gip = ImportGroupReduceSet(gip_core)
        gip.setReductionFactor(2)
        gip.executeGeo()
        bp = gip.getBox()
        (x,y,z) = bp.getCenter()
        self.assertEqual((x,y,z),(0.55000000000000004, -0.25, 0.050000000000000003))
        # delete
        gip.delete()
        gip_core.delete()
        self._RemoveImportGroupModules()
        self.assertEqual(theNet().moduleCount(), moduleCount)

    def testTransientReduceSetData(self):
        moduleCount = theNet().moduleCount()
        gip_core = self._TransientGroup()
        gip = ImportGroupReduceSet(gip_core)
        gip.setReductionFactor(2)
        value = 'Pressure'
        c = Colors()
        theNet().add(c)
        connect( gip.dataConnectionPoint(value), ConnectionPoint( c, 'Data' ) )
        coll = Collect()
        theNet().add(coll)
        connect( gip.geoConnectionPoint(), ConnectionPoint( coll, 'grid' ) )
        theNet().connect( c, 'texture', coll, 'textures' )
        theNet().connect( coll, 'geometry',  globalRenderer()._module, 'RenderData')
        gip.executeGeo()
        gip.executeData(value)
        (a,b) = gip.getDataMinMax(value)
        self.assertEqual((a,b),(-0.018360999999999999, 2.0))
        # delete
        theNet().remove(c)
        theNet().remove(coll)
        gip.delete()
        gip_core.delete()
        self._RemoveImportGroupModules()
        self.assertEqual(theNet().moduleCount(), moduleCount)

    def testTransientGroupGeo(self):
        moduleCount = theNet().moduleCount()
        gip = self._TransientGroup()
        bp = gip.getBox()
        (x,y,z) = bp.getCenter()
        self.assertEqual((x,y,z), (0.099999999999999978, 0.012500000000000011, 0.050000000000000003))
        # delete
        gip.delete()
        self._RemoveImportGroupModules()
        self.assertEqual(theNet().moduleCount(), moduleCount)

    def testGroupData(self):
        moduleCount = theNet().moduleCount()
        gip = self._LargeSimpleGroup()        
        value = 'Temperature'
        c = Colors()
        theNet().add(c)
        connect( gip.dataConnectionPoint(value), ConnectionPoint( c, 'Data' ) )
        coll = Collect()
        theNet().add(coll)
        connect( gip.geoConnectionPoint(), ConnectionPoint( coll, 'grid' ) )
        theNet().connect( c, 'texture', coll, 'textures' )
        theNet().connect( coll, 'geometry',  globalRenderer()._module, 'RenderData')
        gip.executeGeo()
        gip.executeData(value)
        # delete
        theNet().remove(c)
        theNet().remove(coll)
        gip.delete()
        self._RemoveImportGroupModules()
        self.assertEqual(theNet().moduleCount(), moduleCount)


def suite():
    def loadTests(testCaseClass):
        return unittest.defaultTestLoader.loadTestsFromTestCase(testCaseClass)

    # Create Renderer (Use VRPCoviseNetAccess._globalRenderer, so saveExecute works correctly)
    globalRenderer()
    # Use the same Renderer for all tests!
    # There seems to be a problem when creating and deleting a Renderer for each test individually.
    # If the Renderer is deleted immediatelly after the test, the finishedBarrier doesn't work anymore.
    # (Maybe that's because the Renderer is still receiving geometry while it's beeing removed.)

    testSuite = unittest.TestSuite()
    testSuite.addTests(loadTests(ImportModuleTestCase))

    return testSuite


def _main():
    runner = unittest.TextTestRunner(verbosity = 2)
    runner.run(suite())

if __name__ == "__main__":
    _main()

